"""Demonstrate the Trading Logistics Driver on the canonical SPY put example."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from agentlightning_adapter import to_lightning_transition
from data_loader import (
    DEFAULT_OPTION_DIR,
    DEFAULT_STOCK_DIR,
    find_optimal_exit_plan,
    get_stock_close_on_date,
    get_stock_close_series,
    load_market_snapshot,
)
from logistics_driver import evaluate_trajectory, format_markdown_report
from regime_classifier import RegimeFeatures, classify_regime, features_from_stock_history
from trade_trajectory import TradeTrajectory
from trajectory_reward import locked_spy_reward

DEFAULT_TICKER = "SPY"
MIN_EXIT_HOLD_DAYS = 7
MAX_EXIT_HOLD_DAYS = 14
DEFAULT_MAX_PREMIUM = 1.50
SCRIPT_PATH = Path(__file__).resolve()


def _as_utc(value) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.to_pydatetime().replace(tzinfo=timezone.utc)
    return ts.to_pydatetime().astimezone(timezone.utc)


def _project_underlying_exit(
    stock_df,
    underlying_entry: float,
    entry_time: datetime,
    exit_time: datetime,
) -> float:
    """Use future close if available, otherwise extrapolate from recent drift."""
    projected = get_stock_close_on_date(stock_df, exit_time)
    if projected is not None and projected != underlying_entry:
        return projected

    closes = get_stock_close_series(stock_df)
    if len(closes) >= 6:
        drift_5d = float(closes.iloc[-1] / closes.iloc[-6] - 1.0)
        hold_days = max((exit_time.date() - entry_time.date()).days, 1)
        return underlying_entry * (1.0 + drift_5d * (hold_days / 5.0))
    return underlying_entry


def _volume_ratio(stock_df) -> float:
    volume = pd.to_numeric(stock_df.get("Volume", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(volume) < 20:
        return 0.0
    recent = volume.iloc[-5:].mean()
    prior = volume.iloc[-20:-5].mean()
    if prior == 0:
        return 0.0
    return float((recent - prior) / prior)


def build_spy_put_trajectory(
    stock_dir: str = DEFAULT_STOCK_DIR,
    option_dir: str = DEFAULT_OPTION_DIR,
    max_premium: float = DEFAULT_MAX_PREMIUM,
    min_exit_hold_days: int = MIN_EXIT_HOLD_DAYS,
    max_exit_hold_days: int = MAX_EXIT_HOLD_DAYS,
) -> tuple[TradeTrajectory, dict]:
    """
    Build the SPY put trajectory from the latest CSV snapshot.

    Entry is anchored to the chain snapshot date (today). Exit is scheduled
    7-14 calendar days out using the best available chain bid on the same strike.
    """
    snapshot = load_market_snapshot(
        DEFAULT_TICKER,
        stock_dir=stock_dir,
        option_dir=option_dir,
        is_put=True,
        min_dte=0,
        max_dte=max_exit_hold_days,
        max_premium=max_premium,
    )
    spot = snapshot["spot"]
    contract = snapshot["contract"]
    stock_df = snapshot["stock_df"]
    option_df = snapshot["option_df"]
    reference_date = snapshot["reference_date"]
    ref_entry = _as_utc(reference_date)
    latest_entry = _as_utc(snapshot["latest_date"])
    entry_time = latest_entry if latest_entry.date() >= ref_entry.date() else ref_entry
    underlying_entry = get_stock_close_on_date(stock_df, entry_time) or spot

    entry_price = contract.get("entry_price") or (
        contract["ask"] if contract["ask"] > 0 else contract["mid"]
    )

    exit_plan = find_optimal_exit_plan(
        option_df,
        contract,
        reference_date=entry_time.replace(tzinfo=None),
        is_put=True,
        min_hold_days=min_exit_hold_days,
        max_hold_days=max_exit_hold_days,
    )
    exit_time = entry_time + timedelta(days=exit_plan["hold_days"])
    exit_price = round(exit_plan["exit_bid"], 2)
    underlying_exit = _project_underlying_exit(
        stock_df, underlying_entry, entry_time, exit_time,
    )

    iv_entry = contract["iv"]
    iv_exit = float(exit_plan.get("exit_iv", iv_entry))
    dte = int(contract.get("dte", 0))
    gamma = contract.get("gamma", 0.0) or 0.04
    theta = -(entry_price / max(dte, 1)) * 0.85
    vega = entry_price * 0.04

    strike = contract["strike"]
    notes = (
        f"Entry {entry_time.date()} @ spot ${underlying_entry:.2f}; "
        f"exit {exit_time.date()} (+{exit_plan['hold_days']}d, bid ${exit_price:.2f}); "
        f"chain {snapshot['chain_date']} put {contract['symbol']} "
        f"ask ${entry_price:.2f} (max ${max_premium:.2f}); "
        "bearish thesis vs cheap short-dated put trajectory."
    )

    trade = TradeTrajectory(
        ticker=DEFAULT_TICKER,
        contract_type="put",
        strike=strike,
        expiry=exit_plan.get("exit_expiry") or contract["expiry"],
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        underlying_entry=underlying_entry,
        underlying_exit=underlying_exit,
        dte=dte,
        implied_vol_entry=iv_entry,
        implied_vol_exit=iv_exit,
        delta=contract["delta"],
        gamma=gamma,
        theta=theta,
        vega=vega,
        notes=notes,
        cvd_entry=-0.12,
        cvd_exit=-0.18,
        volume_ratio=_volume_ratio(stock_df),
    )
    snapshot["exit_plan"] = exit_plan
    return trade, snapshot


def build_regime_features(trade: TradeTrajectory) -> RegimeFeatures:
    """Use recent stock history plus live IV change for regime classification."""
    history = features_from_stock_history(
        trade.ticker,
        lookback_days=20,
        iv_change=trade.iv_change,
    )
    return RegimeFeatures(
        underlying_return=trade.underlying_move_pct,
        realized_volatility=history.realized_volatility,
        iv_change=trade.iv_change,
        cvd=-0.06,
        volume_change=trade.volume_ratio,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the SPY logistics-driver demo using live CSV market data.",
    )
    parser.add_argument(
        "--stock-dir",
        default=DEFAULT_STOCK_DIR,
        help=f"Stock CSV directory (default: {DEFAULT_STOCK_DIR})",
    )
    parser.add_argument(
        "--option-dir",
        default=DEFAULT_OPTION_DIR,
        help=f"Option chain root directory (default: {DEFAULT_OPTION_DIR})",
    )
    parser.add_argument(
        "--max-premium",
        type=float,
        default=DEFAULT_MAX_PREMIUM,
        help=f"Maximum option ask to pay at entry (default: {DEFAULT_MAX_PREMIUM:.2f})",
    )
    parser.add_argument(
        "--min-exit-days",
        type=int,
        default=MIN_EXIT_HOLD_DAYS,
        help=f"Minimum hold days before exit (default: {MIN_EXIT_HOLD_DAYS})",
    )
    parser.add_argument(
        "--max-exit-days",
        type=int,
        default=MAX_EXIT_HOLD_DAYS,
        help=f"Maximum hold days before exit (default: {MAX_EXIT_HOLD_DAYS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Script: {SCRIPT_PATH}")
    print(f"Loading {DEFAULT_TICKER} from stock_dir={args.stock_dir} option_dir={args.option_dir}")
    print(f"Max entry premium (ask): ${args.max_premium:.2f}")
    print(f"Exit window: {args.min_exit_days}-{args.max_exit_days} days (best chain bid)")

    try:
        trade, snapshot = build_spy_put_trajectory(
            stock_dir=args.stock_dir,
            option_dir=args.option_dir,
            max_premium=args.max_premium,
            min_exit_hold_days=args.min_exit_days,
            max_exit_hold_days=args.max_exit_days,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Market data error: {exc}", file=sys.stderr)
        print(
            "This demo requires live CSV inputs. "
            "Ensure F:/inputs/stocks/SPY.csv and the latest SPY option chain exist.",
            file=sys.stderr,
        )
        sys.exit(1)

    contract = snapshot["contract"]
    exit_plan = snapshot["exit_plan"]
    print(
        f"Entry: {trade.entry_time.date()} spot=${trade.underlying_entry:.2f} | "
        f"{contract['symbol']} ask=${trade.entry_price:.2f} DTE={contract.get('dte', 0)}"
    )
    print(
        f"Exit:  {trade.exit_time.date()} (+{exit_plan['hold_days']}d) bid=${trade.exit_price:.2f} | "
        f"spot=${trade.underlying_exit:.2f} | chain={snapshot['chain_date']}"
    )

    reward = locked_spy_reward()
    features = build_regime_features(trade)

    assert reward.total_reward() == -2.75, f"expected -2.75, got {reward.total_reward()}"
    assert reward.classification() == "correct_direction_wrong_vehicle"
    assert trade.entry_price <= args.max_premium + 0.001, (
        f"expected entry ask <= {args.max_premium}, got {trade.entry_price}"
    )

    regime = classify_regime(features)
    report = evaluate_trajectory(trade, reward, features=features)
    markdown = format_markdown_report(report)
    print()
    print(markdown)

    if regime != "cruise_descent":
        print(
            f"\nNote: current market path classified as `{regime}` "
            f"(underlying move {trade.underlying_move_pct:+.2%}); "
            "reward schema and classification remain locked to the SPY lesson."
        )

    transition = to_lightning_transition(trade, reward, report.regime)
    print("\n--- AgentLightning stub transition (keys) ---")
    print(", ".join(sorted(transition.keys())))


if __name__ == "__main__":
    main()
