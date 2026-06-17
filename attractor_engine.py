"""Attractor engine — equilibrium field, gamma wall pinning, restorative force."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def compute_attractor_field(
    df: pd.DataFrame,
    *,
    gamma: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Compute attractor strength, wall pinning, deviation, restorative force."""
    result = df.copy()
    gamma = gamma or {}
    close = pd.to_numeric(result.get("Close", pd.Series(np.nan, index=result.index)), errors="coerce")

    vwap = pd.to_numeric(result.get("vwap", pd.Series(np.nan, index=result.index)), errors="coerce")
    vwap_distance = (close - vwap).abs() / close.replace(0, np.nan)
    vwap_pull = (1.0 - vwap_distance * 20.0).clip(0.0, 1.0).fillna(0.5)

    ma20 = close.rolling(20, min_periods=8).mean()
    ma_distance = (close - ma20).abs() / close.replace(0, np.nan)
    ma_pull = (1.0 - ma_distance * 15.0).clip(0.0, 1.0).fillna(0.5)

    f_r = pd.to_numeric(result.get("F_r", pd.Series(0.5, index=result.index)), errors="coerce").fillna(0.5)

    flip_strike = gamma.get("gamma_flip_strike")
    gamma_dist_pct = pd.to_numeric(
        result.get("gamma_flip_distance_pct", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )

    if flip_strike is not None and not pd.isna(flip_strike):
        strike = float(flip_strike)
        result["equilibrium_pin_price"] = strike
        dist_abs = gamma_dist_pct.abs().fillna(999.0)
        gamma_attractor = (1.0 - dist_abs / 5.0).clip(0.0, 1.0)
        result["gamma_attractor_strength"] = gamma_attractor

        call_wall = strike * 1.02
        put_wall = strike * 0.98
        call_dist = ((close - call_wall) / close.replace(0, np.nan) * 100.0).abs()
        put_dist = ((close - put_wall) / close.replace(0, np.nan) * 100.0).abs()
        result["call_wall_distance_pct"] = call_dist
        result["put_wall_distance_pct"] = put_dist

        above_call = close > call_wall
        near_call = call_dist < 1.5
        near_put = put_dist < 1.5
        pinning = pd.Series(0.0, index=result.index)
        pinning = pinning.where(~(above_call & near_call), 0.85)
        pinning = pinning.where(~near_put, np.maximum(pinning, 0.70))
        pinning = pinning.where(~(dist_abs < 1.0), np.maximum(pinning, 0.75))
        result["wall_pinning_strength"] = pinning.fillna(0.0).clip(0.0, 1.0)
    else:
        result["equilibrium_pin_price"] = np.nan
        result["gamma_attractor_strength"] = 0.0
        result["call_wall_distance_pct"] = np.nan
        result["put_wall_distance_pct"] = np.nan
        result["wall_pinning_strength"] = 0.0

    gamma_str = pd.to_numeric(result.get("gamma_attractor_strength", 0.0), errors="coerce").fillna(0.0)
    wall_pin = pd.to_numeric(result.get("wall_pinning_strength", 0.0), errors="coerce").fillna(0.0)

    result["A_field"] = (
        0.30 * vwap_pull + 0.25 * ma_pull + 0.20 * f_r + 0.15 * gamma_str + 0.10 * wall_pin
    ).clip(0.0, 1.0)
    result["attractor_field_strength"] = result["A_field"]
    result["equilibrium_field_strength"] = result["A_field"]

    composite_deviation = (0.40 * vwap_distance.fillna(0.0) + 0.35 * ma_distance.fillna(0.0) + 0.25 * (gamma_dist_pct.abs().fillna(0.0) / 100.0)) * 100.0
    ceiling = composite_deviation.rolling(20, min_periods=5).quantile(0.95).replace(0, np.nan)
    result["deviation_from_equilibrium"] = (composite_deviation / ceiling).clip(0.0, 1.0).fillna(0.0)
    result["attractor_deviation"] = result["deviation_from_equilibrium"]

    result["restorative_force_estimate"] = (
        -1.0 * result["A_field"] * result["deviation_from_equilibrium"]
    ).clip(-1.0, 0.0)

    return result
