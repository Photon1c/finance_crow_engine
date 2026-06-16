import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_loader import (
    DEFAULT_OPTION_DIR,
    DEFAULT_STOCK_DIR,
    find_optimal_contract,
    get_most_recent_option_date,
    get_stock_close_series,
    load_market_snapshot,
    parse_price,
)
from packet_persistence import append_packet, save_report
from packet_similarity import (
    SIMILAR_TO_CHOICES,
    find_similar_for_ticker,
    format_query_context,
    format_similar_markdown,
    parse_filters,
)
from recursive_weight_engine import (
    format_theory_line,
    format_weights_markdown,
    load_weights,
    save_weights,
    score_packet,
    update_weights as apply_weight_update,
    weights_path,
)

# =====================================================================
# 1. DATA STRUCTURE DEFINITIONS (NumPy Structured Arrays)
# =====================================================================

trade_state_dtype = np.dtype([
    ('trade_id', 'i4'),
    ('instrument', 'U20'),
    ('status', 'U30'),
    ('theta_risk', 'f4'),
    ('cvd_confirmation', 'b1'),
    ('failure_type', 'U30'),
])

forecast_scores_dtype = np.dtype([
    ('trade_id', 'i4'),
    ('direction_score', 'f4'),
    ('timing_score', 'f4'),
    ('magnitude_score', 'f4'),
    ('volatility_score', 'f4'),
    ('catalyst_score', 'f4'),
    ('exit_score', 'f4'),
])

update_weights_dtype = np.dtype([
    ('trade_id', 'i4'),
    ('absorption_capacity_mod', 'f4'),
    ('selling_pressure_mod', 'f4'),
    ('model_update_note', 'U60'),
])

DEFAULT_TICKER = "SPY"

DEMO_TRADES = [
    {
        "trade_id": 1,
        "instrument": "SPY Put",
        "status": "Near Breakeven",
        "theta_risk": 0.9,
        "cvd_confirmation": True,
        "forecast": (1, 0.5, -0.5, 0.0, 0.0, 0.2, 0.8),
        "update": (1, 0.3, -0.1, "Increase timing decay factor; down-weight premium acceleration"),
    },
    {
        "trade_id": 2,
        "instrument": "UEC Weekly Call",
        "status": "-100% Premium",
        "theta_risk": 0.4,
        "cvd_confirmation": False,
        "forecast": (2, -1.0, -0.8, -1.0, -0.9, -1.0, 0.0),
        "update": (2, -0.5, 0.6, "Catalyst inversion; trigger structural IV crush penalty"),
    },
]

SCRIPT_DIR = Path(__file__).resolve().parent


# =====================================================================
# 2. HELPER FUNCTIONS FOR CLASSIFICATION & REPORTING
# =====================================================================

def compute_failure_score(forecast_row) -> float:
    components = [
        forecast_row['direction_score'],
        forecast_row['timing_score'],
        forecast_row['magnitude_score'],
        forecast_row['volatility_score'],
        forecast_row['catalyst_score'],
        forecast_row['exit_score'],
    ]
    penalties = [0.5 * (1.0 - c) for c in components]
    return float(np.mean(penalties))


def classify_failure(forecast_row, state_row) -> str:
    metrics = {
        'Direction': forecast_row['direction_score'],
        'Timing': forecast_row['timing_score'],
        'Magnitude': forecast_row['magnitude_score'],
        'Volatility': forecast_row['volatility_score'],
        'Catalyst': forecast_row['catalyst_score'],
        'Exit Discipline': forecast_row['exit_score'],
    }
    worst_component = min(metrics, key=metrics.get)
    if metrics[worst_component] >= 0.0:
        return "Controlled Dissipation"
    return worst_component


def extract_ticker(instrument: str) -> str:
    return instrument.split()[0].upper()


def is_put_instrument(instrument: str) -> bool:
    return "put" in instrument.lower()


def _clip_score(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def format_packet_fields(contract: dict, ticker: str) -> str:
    strike = contract["strike"]
    return (
        f"{ticker} ${strike:.0f} | IV {contract['iv']:.2f} | "
        f"d {contract['delta']:+.2f} | mid ${contract['mid']:.2f}"
    )


def format_contract_label(contract: dict) -> str:
    strike = contract["strike"]
    sym = contract.get("symbol", "")
    opt_type = "P" if "P" in sym else "C"
    return f"{opt_type}${strike:.0f}\nIV {contract['iv']:.2f}  d {contract['delta']:+.2f}"


def build_visual_context(
    ticker: str,
    stock_dir: str,
    option_dir: str,
    instrument: Optional[str] = None,
    failure_type: Optional[str] = None,
    status: Optional[str] = None,
) -> dict:
    """Load SPY (or ticker) CSV data and build labels for the packet animation."""
    instrument = instrument or f"{ticker} Put"
    is_put = is_put_instrument(instrument)

    try:
        snapshot = load_market_snapshot(
            ticker,
            stock_dir=stock_dir,
            option_dir=option_dir,
            is_put=is_put,
            min_dte=1,
            max_dte=21,
        )
        spot = snapshot["spot"]
        chain_date = snapshot["chain_date"]
        contract = snapshot["contract"]
    except FileNotFoundError as exc:
        print(f"Visual context fallback for {ticker}: {exc}", file=sys.stderr)
        contract = {
            "symbol": f"{ticker}P00000000" if is_put else f"{ticker}C00000000",
            "strike": 0.0,
            "iv": 0.0,
            "delta": -0.45 if is_put else 0.45,
            "bid": 0.0,
            "ask": 0.0,
            "mid": 0.0,
            "volume": 0.0,
            "open_interest": 0.0,
            "expiry": "unknown",
        }
        spot = 0.0
        chain_date = "unknown"

    return {
        "ticker": ticker,
        "instrument": instrument,
        "spot": spot,
        "chain_date": chain_date,
        "failure_type": failure_type or "unknown",
        "status": status or "Open",
        "optimal_contract": contract,
        "packet_fields": format_packet_fields(contract, ticker),
        "contract_label": format_contract_label(contract),
        "cloud_label": f"{ticker} event-space @ ${spot:.2f}" if spot else f"{ticker} event-space",
        "raw_event_label": f"{ticker}\n${spot:.0f}" if spot else ticker,
        "outcome_label": contract["symbol"] if contract.get("symbol") else ticker,
        "pressure_label": "theta\nrisk",
    }


def derive_scores_from_csv(
    instrument: str,
    stock_df: pd.DataFrame,
    option_df: pd.DataFrame,
    *,
    spot: Optional[float] = None,
    chain_date: Optional[str] = None,
    reference_date=None,
    contract: Optional[dict] = None,
    min_dte: int = 1,
    max_dte: Optional[int] = 21,
) -> dict:
    """Map historical stock + option chain CSV data into forecast component scores."""
    closes = get_stock_close_series(stock_df)
    spot = float(spot if spot is not None else closes.iloc[-1])
    ret_5 = float(closes.iloc[-1] / closes.iloc[-6] - 1.0) if len(closes) >= 6 else 0.0
    ret_20 = float(closes.iloc[-1] / closes.iloc[-21] - 1.0) if len(closes) >= 21 else ret_5
    realized_vol = float(closes.pct_change().dropna().tail(20).std() * np.sqrt(252))
    stock_volume = pd.to_numeric(stock_df.get("Volume", pd.Series(dtype=float)), errors="coerce")
    vol_ratio = float(stock_volume.iloc[-1] / stock_volume.tail(20).mean()) if len(stock_volume) >= 20 else 1.0

    is_put = is_put_instrument(instrument)
    if contract is None:
        row = find_optimal_contract(
            option_df,
            spot,
            is_put=is_put,
            reference_date=reference_date,
            min_dte=min_dte,
            max_dte=max_dte,
        )
    else:
        row = contract
    iv = row["iv"]
    delta = row["delta"]
    opt_volume = row["volume"]
    net = row["mid"] - row["bid"] if row["bid"] else 0.0

    iv = 0.2 if np.isnan(iv) else iv
    delta = -0.5 if np.isnan(delta) else delta
    realized_vol = 0.15 if np.isnan(realized_vol) or realized_vol == 0 else realized_vol

    signed_move = -ret_5 if is_put else ret_5
    direction_score = _clip_score(signed_move * 12.0)
    magnitude_score = _clip_score(abs(ret_20) * 8.0 - 0.25)
    timing_score = _clip_score(-abs(delta) - iv * 0.35)
    volatility_score = _clip_score((iv - realized_vol) * -2.5 if is_put else (iv - realized_vol) * 2.0)
    catalyst_score = _clip_score((vol_ratio - 1.0) * 0.8 + (opt_volume / 100000.0))
    exit_score = _clip_score(0.6 + net * 0.15) if not np.isnan(net) else 0.0

    theta_risk = float(np.clip(iv * 0.85 + abs(delta) * 0.35, 0.0, 1.0))
    failure_score = float(np.mean([0.5 * (1.0 - s) for s in [
        direction_score, timing_score, magnitude_score,
        volatility_score, catalyst_score, exit_score,
    ]]))

    if failure_score > 0.65:
        status = "-100% Premium"
    elif failure_score > 0.35:
        status = "Near Breakeven"
    else:
        status = "Open"

    latest_date = stock_df["Date"].iloc[-1] if "Date" in stock_df.columns else "unknown"
    if chain_date is None:
        chain_date = get_most_recent_option_date(extract_ticker(instrument), verbose=False)

    return {
        "forecast": (
            direction_score, timing_score, magnitude_score,
            volatility_score, catalyst_score, exit_score,
        ),
        "status": status,
        "theta_risk": theta_risk,
        "cvd_confirmation": direction_score >= 0,
        "market_context": (
            f"Spot ${spot:.2f} ({latest_date}); chain {chain_date}; "
            f"optimal {row['symbol']} ${row['strike']:.0f}; "
            f"IV {iv:.2f}; delta {delta:+.2f}; DTE {row.get('dte', 0)}; "
            f"5d ret {ret_5:+.2%}; realized vol {realized_vol:.2%}"
        ),
        "update_note": (
            f"CSV-derived optimal {row['symbol']}: IV {iv:.2f}, delta {delta:.2f}, "
            f"vol {opt_volume:.0f}; adjust timing decay and absorption from chain skew"
        ),
        "optimal_contract": row,
        "absorption_capacity_mod": _clip_score((iv - realized_vol) * 0.8),
        "selling_pressure_mod": _clip_score(-ret_5 * 2.0 if is_put else ret_5 * 2.0),
    }


def load_trade_from_csv(spec: dict, stock_dir: str, option_dir: str) -> dict:
    ticker = extract_ticker(spec["instrument"])
    try:
        snapshot = load_market_snapshot(
            ticker,
            stock_dir=stock_dir,
            option_dir=option_dir,
            is_put=is_put_instrument(spec["instrument"]),
            min_dte=1,
            max_dte=21,
        )
        derived = derive_scores_from_csv(
            spec["instrument"],
            snapshot["stock_df"],
            snapshot["option_df"],
            spot=snapshot["spot"],
            chain_date=snapshot["chain_date"],
            reference_date=snapshot["reference_date"],
            contract=snapshot["contract"],
        )
        trade = dict(spec)
        trade["status"] = derived["status"]
        trade["theta_risk"] = derived["theta_risk"]
        trade["cvd_confirmation"] = derived["cvd_confirmation"]
        trade["forecast"] = (spec["trade_id"], *derived["forecast"])
        trade["update"] = (
            spec["trade_id"],
            derived["absorption_capacity_mod"],
            derived["selling_pressure_mod"],
            derived["update_note"],
        )
        trade["market_context"] = derived["market_context"]
        trade["optimal_contract"] = derived.get("optimal_contract")
        return trade
    except FileNotFoundError as exc:
        print(f"CSV fallback for {spec['instrument']}: {exc}", file=sys.stderr)
        return dict(spec)


def build_trade_tables(trade_specs):
    trade_states = np.empty(len(trade_specs), dtype=trade_state_dtype)
    forecast_scores = np.empty(len(trade_specs), dtype=forecast_scores_dtype)
    update_weights = np.empty(len(trade_specs), dtype=update_weights_dtype)
    contexts = {}

    for i, spec in enumerate(trade_specs):
        trade_states[i] = (
            spec["trade_id"],
            spec["instrument"],
            spec["status"],
            spec["theta_risk"],
            spec["cvd_confirmation"],
            "",
        )
        forecast_scores[i] = spec["forecast"]
        update_weights[i] = spec["update"]
        contexts[spec["trade_id"]] = spec.get("market_context")

    for i in range(len(trade_states)):
        t_id = trade_states[i]['trade_id']
        f_row = forecast_scores[forecast_scores['trade_id'] == t_id][0]
        trade_states[i]['failure_type'] = classify_failure(f_row, trade_states[i])

    return trade_states, forecast_scores, update_weights, contexts


def build_packet_record(state, forecast, update, failure_score, weighted_score, timestamp):
    ticker = extract_ticker(str(state["instrument"]))
    return {
        "timestamp": timestamp,
        "trade_id": int(state["trade_id"]),
        "ticker": ticker,
        "instrument": str(state["instrument"]),
        "status": str(state["status"]),
        "failure_type": str(state["failure_type"]),
        "failure_score": float(failure_score),
        "weighted_score": float(weighted_score),
        "theta_risk": float(state["theta_risk"]),
        "cvd_confirmation": bool(state["cvd_confirmation"]),
        "direction_score": float(forecast["direction_score"]),
        "timing_score": float(forecast["timing_score"]),
        "magnitude_score": float(forecast["magnitude_score"]),
        "volatility_score": float(forecast["volatility_score"]),
        "catalyst_score": float(forecast["catalyst_score"]),
        "exit_score": float(forecast["exit_score"]),
        "absorption_capacity_mod": float(update["absorption_capacity_mod"]),
        "selling_pressure_mod": float(update["selling_pressure_mod"]),
        "model_update_note": str(update["model_update_note"]),
    }


def run_recursive_regression(
    trade_states,
    forecast_scores,
    update_weights,
    base_dir,
    similar_to=None,
    filters=None,
):
    """Score packets, update weights, append ledger rows, and find similar history."""
    weights_by_ticker = {}
    packet_meta = []
    query_context = format_query_context(filters or [], similar_to)

    for state, forecast, update in zip(trade_states, forecast_scores, update_weights):
        ticker = extract_ticker(str(state["instrument"]))
        failure_score = compute_failure_score(forecast)
        packet_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        wpath = weights_path(base_dir, ticker)
        if ticker not in weights_by_ticker:
            weights_by_ticker[ticker] = load_weights(wpath)

        weights = weights_by_ticker[ticker]
        weights_before = dict(weights)
        packet = build_packet_record(
            state, forecast, update, failure_score, weighted_score=0.0, timestamp=packet_timestamp,
        )
        packet["weighted_score"] = score_packet(packet, weights)
        weights = apply_weight_update(packet, weights)
        save_weights(weights, wpath)
        weights_by_ticker[ticker] = weights

        append_packet(base_dir, ticker, packet)
        similar = find_similar_for_ticker(
            base_dir,
            ticker,
            packet,
            top_k=5,
            filters=filters,
            weights=weights_before,
            similar_to=similar_to,
        )
        theory = format_theory_line(packet, similar, weights_before, weights)
        packet_meta.append(
            {
                "trade_id": int(state["trade_id"]),
                "instrument": str(state["instrument"]),
                "similar": similar,
                "theory": theory,
                "query_context": query_context,
            }
        )

    return weights_by_ticker, packet_meta


def build_report(
    trade_states,
    forecast_scores,
    update_weights,
    contexts,
    weights_by_ticker,
    packet_meta,
):
    lines = [
        "# RECURSIVE TRADE FAILURE AGGREGATOR - REPORT",
        "",
        "> *AI-Intermediary Log File Format - Structured Memory Vector Initialization*",
        "",
    ]
    meta_by_id = {item["trade_id"]: item for item in packet_meta}

    for state in trade_states:
        trade_id = int(state["trade_id"])
        lines.append(summarize_trade(
            trade_id,
            trade_states,
            forecast_scores,
            update_weights,
            market_context=contexts.get(trade_id),
        ).rstrip())
        if trade_id in meta_by_id:
            meta = meta_by_id[trade_id]
            lines.append(
                format_similar_markdown(
                    meta["similar"],
                    meta["instrument"],
                    query_context=meta.get("query_context", ""),
                ).rstrip()
            )
            lines.append(f"**Updated Theory:** {meta['theory']}")
            lines.append("")

    lines.append("## Recursive Model State")
    lines.append("")
    for ticker in sorted(weights_by_ticker):
        lines.append(format_weights_markdown(weights_by_ticker[ticker], ticker).rstrip())

    return "\n".join(lines) + "\n"


def run_report(trade_states, forecast_scores, update_weights, contexts, weights_by_ticker, packet_meta):
    report = build_report(
        trade_states, forecast_scores, update_weights, contexts,
        weights_by_ticker, packet_meta,
    )
    print(report, end="")
    return report


def summarize_trade(trade_id, states, forecasts, updates, market_context=None) -> str:
    state = states[states['trade_id'] == trade_id][0]
    forecast = forecasts[forecasts['trade_id'] == trade_id][0]
    update = updates[updates['trade_id'] == trade_id][0]
    fail_score = compute_failure_score(forecast)

    report = f"### Trade Packet {trade_id}: {state['instrument']}\n"
    report += f"- **Current Status:** {state['status']}\n"
    report += f"- **Primary Classified Failure:** `{state['failure_type']}`\n"
    report += f"- **Aggregate Failure Score:** `{fail_score:.2f}` (0=Pass, 1=Total Failure)\n"
    if market_context:
        report += f"- **Market Context (CSV):** {market_context}\n"
    report += "\n"

    report += "| Component | Score (-1 to +1) |\n"
    report += "| :--- | :---: |\n"
    report += f"| Directional Thesis | {forecast['direction_score']:+.1f} |\n"
    report += f"| Timing Thesis      | {forecast['timing_score']:+.1f} |\n"
    report += f"| Magnitude Thesis   | {forecast['magnitude_score']:+.1f} |\n"
    report += f"| Volatility Dynamics| {forecast['volatility_score']:+.1f} |\n"
    report += f"| Catalyst Quality   | {forecast['catalyst_score']:+.1f} |\n"
    report += f"| Exit Discipline    | {forecast['exit_score']:+.1f} |\n\n"

    report += "**Model Update Actions:**\n"
    report += (
        f"- *Suggested Parameter Adjustments:* "
        f"Absorption d({update['absorption_capacity_mod']:+.2f}), "
        f"Pressure d({update['selling_pressure_mod']:+.2f})\n"
    )
    report += f"- *AI Log:* `{update['model_update_note']}`\n"
    report += "\n---\n"
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate recursive trade failure packets from demo or CSV market data.",
    )
    parser.add_argument(
        "--csv",
        nargs="?",
        const="default",
        default="default",
        metavar="DIR",
        help=(
            "Load historical stock prices and option chains from CSV inputs "
            f"(stock dir: {DEFAULT_STOCK_DIR}, options: {DEFAULT_OPTION_DIR}). "
            "Pass a project data directory to override both paths' parent."
        ),
    )
    parser.add_argument(
        "--no-csv",
        dest="csv",
        action="store_const",
        const=None,
        help="Use built-in demo trades instead of CSV market data.",
    )
    parser.add_argument(
        "--stock-dir",
        default=DEFAULT_STOCK_DIR,
        help=f"Directory containing TICKER.csv stock history (default: {DEFAULT_STOCK_DIR}).",
    )
    parser.add_argument(
        "--option-dir",
        default=DEFAULT_OPTION_DIR,
        help=f"Root directory for option chain logs (default: {DEFAULT_OPTION_DIR}).",
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help=f"Primary ticker for CSV loading and --visual labels (default: {DEFAULT_TICKER}).",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help=f"Render packet pipeline animation for --ticker (default: {DEFAULT_TICKER}).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR),
        help="Directory for --visual outputs.",
    )
    parser.add_argument(
        "--visual-name",
        default=None,
        help="Base filename for --visual outputs (default: recursive_trade_failure_SPY).",
    )
    parser.add_argument(
        "--similar-to",
        choices=SIMILAR_TO_CHOICES,
        default=None,
        help="Weight similarity search toward one component (e.g. timing, direction).",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Investigator filter on ledger rows (e.g. failure_type=Timing, status=Open). Repeatable.",
    )
    return parser.parse_args()


def resolve_csv_dirs(args):
    if args.csv is None:
        return None, None
    if args.csv == "default":
        return args.stock_dir, args.option_dir
    base = Path(args.csv)
    return str(base / "stocks"), str(base / "options" / "log")


def main():
    args = parse_args()
    stock_dir, option_dir = resolve_csv_dirs(args)

    if stock_dir:
        print(f"Loading market data from CSV (stocks: {stock_dir}, options: {option_dir})\n")
        trade_specs = [
            load_trade_from_csv(spec, stock_dir, option_dir)
            for spec in DEMO_TRADES
        ]
    else:
        print("Using built-in demo trades (--no-csv)\n")
        trade_specs = DEMO_TRADES

    trade_states, forecast_scores, update_weights, contexts = build_trade_tables(trade_specs)
    try:
        filters = parse_filters(args.filter)
    except ValueError as exc:
        print(f"Filter error: {exc}", file=sys.stderr)
        sys.exit(2)

    weights_by_ticker, packet_meta = run_recursive_regression(
        trade_states,
        forecast_scores,
        update_weights,
        SCRIPT_DIR,
        similar_to=args.similar_to,
        filters=filters,
    )
    report = run_report(
        trade_states, forecast_scores, update_weights, contexts,
        weights_by_ticker, packet_meta,
    )
    report_file = save_report(SCRIPT_DIR, args.ticker.upper(), report)
    print(f"\n> Saved report: {report_file}", file=sys.stderr)

    if args.visual:
        print(f"\n## PACKET PIPELINE VISUALIZATION ({args.ticker})\n")
        from misc_packet_visualizer import render_packet_animation

        primary = next(
            (s for s in trade_specs if extract_ticker(s["instrument"]) == args.ticker.upper()),
            trade_specs[0],
        )
        failure_type = "unknown"
        for state in trade_states:
            if args.ticker.upper() in str(state["instrument"]):
                failure_type = str(state["failure_type"])
                break

        visual_stock_dir = stock_dir or args.stock_dir
        visual_option_dir = option_dir or args.option_dir
        visual_ctx = build_visual_context(
            ticker=args.ticker.upper(),
            stock_dir=visual_stock_dir,
            option_dir=visual_option_dir,
            instrument=primary.get("instrument", f"{args.ticker} Put"),
            failure_type=failure_type,
            status=primary.get("status"),
        )

        basename = args.visual_name or f"recursive_trade_failure_{args.ticker.upper()}"
        outputs = render_packet_animation(
            output_dir=args.output_dir,
            basename=basename,
            context=visual_ctx,
        )
        print("\nVisual outputs:")
        for path in outputs:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
