"""Volatility arbitrage / relative-value dislocation detector (CSV replay)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from volatility_skew_engine import compute_skew_metrics


def _nearest_dte_slice(option_df: pd.DataFrame, target_dte: int) -> pd.DataFrame:
    if option_df.empty or "dte" not in option_df.columns:
        return option_df
    dtes = option_df["dte"].dropna().unique()
    if len(dtes) == 0:
        return option_df
    nearest = int(min(dtes, key=lambda d: abs(int(d) - target_dte)))
    expiries = option_df.loc[option_df["dte"] == nearest, "Expiration Date"].unique()
    if len(expiries) == 0:
        return option_df
    return option_df[option_df["Expiration Date"] == expiries[0]]


def detect_vol_arbitrage(
    target_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    target_spot: float,
    benchmark_spot: float,
    target_ticker: str,
    benchmark_ticker: str,
    target_dte: int = 45,
    historical_spread_pct: Optional[float] = None,
) -> dict[str, Any]:
    """
    Detect IV spread dislocation between two tickers at similar DTE.

    Example: SPY 45DTE IV=21%, QQQ 45DTE IV=32%, historical spread 7%, current 11%.
    """
    t_slice = _nearest_dte_slice(target_df, target_dte)
    b_slice = _nearest_dte_slice(benchmark_df, target_dte)

    t_skew = compute_skew_metrics(t_slice if not t_slice.empty else target_df, spot=target_spot)
    b_skew = compute_skew_metrics(b_slice if not b_slice.empty else benchmark_df, spot=benchmark_spot)

    t_atm = t_skew.get("atm_iv")
    b_atm = b_skew.get("atm_iv")
    if t_atm is None or b_atm is None:
        return {
            "status": "INSUFFICIENT",
            "target_ticker": target_ticker,
            "benchmark_ticker": benchmark_ticker,
            "target_dte_requested": target_dte,
        }

    current_spread = round(float(t_atm) - float(b_atm), 4)
    hist_spread = historical_spread_pct if historical_spread_pct is not None else round(float(b_atm) * 0.35, 4)
    dislocation = round(current_spread - hist_spread, 4)

    mispriced_put_wing = False
    mispriced_call_wing = False
    broken_skew = bool(t_skew.get("skew_inversion_flag"))
    t_put = t_skew.get("put_wing_iv")
    t_call = t_skew.get("call_wing_iv")
    b_put = b_skew.get("put_wing_iv")
    b_call = b_skew.get("call_wing_iv")
    if t_put is not None and b_put is not None and t_atm:
        mispriced_put_wing = t_put > b_put * 1.25
    if t_call is not None and b_call is not None and t_atm:
        mispriced_call_wing = t_call > b_call * 1.25

    potential_dislocation = abs(dislocation) > max(3.0, abs(hist_spread) * 0.5)

    return {
        "status": "OK",
        "target_ticker": target_ticker.upper(),
        "benchmark_ticker": benchmark_ticker.upper(),
        "target_dte_requested": target_dte,
        "target_dte_used": int(t_slice["dte"].iloc[0]) if not t_slice.empty and "dte" in t_slice.columns else None,
        "target_atm_iv": t_atm,
        "benchmark_atm_iv": b_atm,
        "current_spread_pct": current_spread,
        "historical_spread_pct": hist_spread,
        "dislocation_pct": dislocation,
        "potential_dislocation": potential_dislocation,
        "broken_skew": broken_skew,
        "mispriced_put_wing": mispriced_put_wing,
        "mispriced_call_wing": mispriced_call_wing,
        "interpretation": (
            f"{target_ticker.upper()} ATM IV {t_atm}% vs {benchmark_ticker.upper()} {b_atm}% "
            f"(spread {current_spread:+.2f}%, dislocation {dislocation:+.2f}% vs hist {hist_spread}%)"
        ),
    }
