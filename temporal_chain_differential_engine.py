"""Temporal option chain differential — compare chain state across time.

CSV replay only. Does not modify sacred ontology.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from chain_compatibility_engine import assess_snapshot_compatibility, validate_snapshot_compatibility
from chain_integrity_engine import assess_chain_integrity, clean_chain_expirations
from data_loader import DEFAULT_OPTION_DIR, DEFAULT_STOCK_DIR, load_option_chain_data, load_stock_data
from laser_falcon_data_adapter import (
    LaserFalconSnapshot,
    load_laser_falcon_snapshot,
    normalize_option_chain,
    normalize_stock_df,
    reference_date_from_chain,
)
from options_pressure_mapper import compute_options_pressure_metrics
from volatility_skew_engine import compute_skew_metrics, filter_strikes_near_spot

ChainInput = Union[str, Path, pd.DataFrame, LaserFalconSnapshot, dict[str, Any]]

PRESSURE_DIRECTIONS = ("PRESSURE_INCREASING", "PRESSURE_DISSIPATING", "STABLE", "INSUFFICIENT")


def list_option_chain_dates(ticker: str, *, option_dir: str = DEFAULT_OPTION_DIR) -> list[str]:
    """List available option chain snapshot date folders for a ticker."""
    ticker_dir = Path(option_dir) / ticker.lower()
    if not ticker_dir.exists():
        return []
    return sorted(
        [d.name for d in ticker_dir.iterdir() if d.is_dir()],
        key=lambda name: pd.to_datetime(name.replace("_", "-"), errors="coerce"),
    )


def resolve_chain_date_pair(
    ticker: str,
    *,
    prior_date: Optional[str] = None,
    current_date: Optional[str] = None,
    option_dir: str = DEFAULT_OPTION_DIR,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve prior/current chain dates; default to two most recent snapshots."""
    dates = list_option_chain_dates(ticker, option_dir=option_dir)
    if not dates:
        return None, None
    current = current_date if current_date in dates else dates[-1]
    if prior_date and prior_date in dates:
        prior = prior_date
    elif len(dates) >= 2:
        prior = dates[-2] if dates[-2] != current else dates[-3] if len(dates) >= 3 else None
    else:
        prior = None
    return prior, current


def _load_chain_input(
    source: ChainInput,
    *,
    ticker: str,
    stock_dir: str = DEFAULT_STOCK_DIR,
    option_dir: str = DEFAULT_OPTION_DIR,
) -> dict[str, Any]:
    """Normalize diverse inputs into option_df, stock_df, spot, chain_date."""
    if isinstance(source, LaserFalconSnapshot):
        return {
            "option_df": source.option_df,
            "stock_df": source.stock_df,
            "spot": source.spot,
            "chain_date": source.chain_date,
        }

    if isinstance(source, dict) and "option_df" in source:
        return {
            "option_df": source["option_df"],
            "stock_df": source.get("stock_df", pd.DataFrame()),
            "spot": float(source["spot"]),
            "chain_date": source.get("chain_date", ""),
        }

    ticker_upper = ticker.upper()
    stock_raw = load_stock_data(ticker_upper, base_dir=stock_dir)
    stock_df = normalize_stock_df(stock_raw)
    spot = float(stock_df["Close"].iloc[-1])

    if isinstance(source, pd.DataFrame):
        option_raw = source
        chain_date = ""
        reference = pd.Timestamp.now().to_pydatetime()
    else:
        path = Path(str(source))
        if path.suffix.lower() == ".csv" and path.exists():
            option_raw = pd.read_csv(path, skiprows=3 if "quotedata" in path.name.lower() else 0)
            chain_date = path.parent.name
            reference = reference_date_from_chain(chain_date)
        else:
            chain_date = str(source)
            option_raw = load_option_chain_data(ticker_upper, date=chain_date, base_dir=option_dir)
            reference = reference_date_from_chain(chain_date)

    option_df = normalize_option_chain(option_raw, spot=spot, reference_date=reference)
    return {
        "option_df": option_df,
        "stock_df": stock_df,
        "spot": spot,
        "chain_date": chain_date,
    }


def extract_chain_pressure_state(
    option_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    *,
    spot: float,
) -> dict[str, Any]:
    """Snapshot of options pressure metrics at a single chain date."""
    skew = compute_skew_metrics(option_df, spot=spot)
    pressure = compute_options_pressure_metrics(
        option_df=option_df,
        stock_df=stock_df,
        spot=spot,
        skew_metrics=skew,
    )
    band = filter_strikes_near_spot(option_df, spot)
    total_oi = float(pd.to_numeric(band.get("Open Interest", 0), errors="coerce").fillna(0).sum()) if not band.empty else 0.0

    return {
        "atm_iv": skew.get("atm_iv"),
        "call_wing_iv": skew.get("call_wing_iv"),
        "put_wing_iv": skew.get("put_wing_iv"),
        "skew_ratio": skew.get("skew_ratio"),
        "skew_asymmetry_pressure": pressure.get("skew_asymmetry_pressure"),
        "gamma_compression_score": pressure.get("gamma_compression_score"),
        "dealer_hedging_stress_score": pressure.get("dealer_hedging_stress_score"),
        "surface_curvature": skew.get("surface_curvature"),
        "total_open_interest": round(total_oi, 2),
        "expiration": skew.get("expiration"),
        "n_strikes_in_band": skew.get("n_strikes_in_band"),
    }


def _delta(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    if current is None or prior is None:
        return None
    try:
        return round(float(current) - float(prior), 4)
    except (TypeError, ValueError):
        return None


def interpret_pressure_direction(deltas: dict[str, Any]) -> str:
    """Classify whether options pressure is building or dissipating."""
    signals: list[int] = []
    for key in (
        "delta_atm_iv",
        "delta_call_wing_iv",
        "delta_put_wing_iv",
        "delta_dealer_stress",
        "delta_gamma_concentration",
        "delta_skew",
    ):
        value = deltas.get(key)
        if value is None:
            continue
        if value > 0:
            signals.append(1)
        elif value < 0:
            signals.append(-1)

    if not signals:
        return "INSUFFICIENT"
    score = sum(signals)
    if score >= 2:
        return "PRESSURE_INCREASING"
    if score <= -2:
        return "PRESSURE_DISSIPATING"
    return "STABLE"


def compare_option_chain_snapshots(
    prior: ChainInput,
    current: ChainInput,
    *,
    ticker: str,
    stock_dir: str = DEFAULT_STOCK_DIR,
    option_dir: str = DEFAULT_OPTION_DIR,
) -> dict[str, Any]:
    """
    Compare option chain state across two snapshots.

    Accepts chain date strings, CSV paths, raw wide DataFrames, LaserFalconSnapshot,
    or dict bundles with option_df/stock_df/spot.
    """
    prior_bundle = _load_chain_input(prior, ticker=ticker, stock_dir=stock_dir, option_dir=option_dir)
    current_bundle = _load_chain_input(current, ticker=ticker, stock_dir=stock_dir, option_dir=option_dir)

    compatibility = assess_snapshot_compatibility(
        prior_bundle["option_df"],
        current_bundle["option_df"],
        ticker=ticker,
        spot_prior=prior_bundle["spot"],
        spot_current=current_bundle["spot"],
    )

    base_result = {
        "ticker": ticker.upper(),
        "prior_chain_date": prior_bundle.get("chain_date", ""),
        "current_chain_date": current_bundle.get("chain_date", ""),
        "compatibility": compatibility,
    }

    if compatibility["status"] == "INVALID":
        return {
            **base_result,
            "status": "INVALID",
            "reason": "Contract universe drift detected",
            "warnings": compatibility.get("warnings", []),
            "errors": compatibility.get("errors", []),
        }

    prior_state = extract_chain_pressure_state(
        prior_bundle["option_df"],
        prior_bundle["stock_df"],
        spot=prior_bundle["spot"],
    )
    current_state = extract_chain_pressure_state(
        current_bundle["option_df"],
        current_bundle["stock_df"],
        spot=current_bundle["spot"],
    )

    deltas = {
        "delta_atm_iv": _delta(current_state.get("atm_iv"), prior_state.get("atm_iv")),
        "delta_call_wing_iv": _delta(current_state.get("call_wing_iv"), prior_state.get("call_wing_iv")),
        "delta_put_wing_iv": _delta(current_state.get("put_wing_iv"), prior_state.get("put_wing_iv")),
        "delta_skew": _delta(current_state.get("skew_asymmetry_pressure"), prior_state.get("skew_asymmetry_pressure")),
        "delta_skew_ratio": _delta(current_state.get("skew_ratio"), prior_state.get("skew_ratio")),
        "delta_gamma_concentration": _delta(
            current_state.get("gamma_compression_score"),
            prior_state.get("gamma_compression_score"),
        ),
        "delta_open_interest": _delta(current_state.get("total_open_interest"), prior_state.get("total_open_interest")),
        "delta_dealer_stress": _delta(
            current_state.get("dealer_hedging_stress_score"),
            prior_state.get("dealer_hedging_stress_score"),
        ),
        "delta_vol_surface_curvature": _delta(
            current_state.get("surface_curvature"),
            prior_state.get("surface_curvature"),
        ),
    }
    direction = interpret_pressure_direction(deltas)

    result = {
        **base_result,
        "prior_state": prior_state,
        "current_state": current_state,
        "deltas": deltas,
        "pressure_direction": direction,
        "interpretation": _interpretation_text(deltas, direction),
    }
    if compatibility["status"] == "DEGRADED":
        result["status"] = "DEGRADED"
        result["warnings"] = compatibility.get("warnings", []) + [
            "Temporal interpretation confidence reduced due to contract universe drift"
        ]
        result["reason"] = "Comparable with reduced confidence"
    else:
        result["status"] = "OK"
    return result


def compare_ticker_chain_dates(
    ticker: str,
    *,
    prior_date: Optional[str] = None,
    current_date: Optional[str] = None,
    stock_dir: str = DEFAULT_STOCK_DIR,
    option_dir: str = DEFAULT_OPTION_DIR,
) -> dict[str, Any]:
    """Compare yesterday vs today (or two most recent) chain folders for a ticker."""
    prior, current = resolve_chain_date_pair(
        ticker,
        prior_date=prior_date,
        current_date=current_date,
        option_dir=option_dir,
    )
    if not prior or not current:
        return {
            "ticker": ticker.upper(),
            "status": "INSUFFICIENT",
            "reason": "Need at least two option chain snapshot dates",
            "available_dates": list_option_chain_dates(ticker, option_dir=option_dir),
        }
    result = compare_option_chain_snapshots(
        prior,
        current,
        ticker=ticker,
        stock_dir=stock_dir,
        option_dir=option_dir,
    )
    if "status" not in result:
        result["status"] = "OK"
    return result


def _interpretation_text(deltas: dict[str, Any], direction: str) -> str:
    atm = deltas.get("delta_atm_iv")
    dealer = deltas.get("delta_dealer_stress")
    skew = deltas.get("delta_skew")
    parts = [f"Options pressure {direction.lower().replace('_', ' ')}"]
    if atm is not None:
        parts.append(f"ATM IV change {atm:+.2f}%")
    if dealer is not None:
        parts.append(f"dealer stress change {dealer:+.4f}")
    if skew is not None:
        parts.append(f"skew asymmetry change {skew:+.4f}")
    return "; ".join(parts) + "."
