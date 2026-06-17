"""Restoration field engine — restoring field strength F_r and dissipation capacity D_c.

Metrics align with TRPR/ontology/packet_ontology.yaml keys F_r and D_c (map inward).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _normalize_distance(series: pd.Series, *, scale: float = 5.0) -> pd.Series:
    """Map absolute percent distance into [0, 1] restoring pull (1 = at equilibrium)."""
    distance = series.abs().fillna(0.0)
    return (1.0 - distance / scale).clip(0.0, 1.0)


def compute_restoration_field(df: pd.DataFrame) -> pd.DataFrame:
    """Compute F_r (restoring field strength) and D_c (dissipation capacity)."""
    result = df.copy()

    vwap_pull = _normalize_distance(
        pd.to_numeric(result.get("vwap_distance_pct", pd.Series(0.0, index=result.index)), errors="coerce"),
        scale=4.0,
    )

    b_s = pd.to_numeric(result.get("B_s", pd.Series(0.5, index=result.index)), errors="coerce").fillna(0.5)
    boundary_restoring = (1.0 - b_s).clip(0.0, 1.0)

    rsi = pd.to_numeric(result.get("rsi", pd.Series(50.0, index=result.index)), errors="coerce").fillna(50.0)
    rsi_restoring = (1.0 - (rsi - 50.0).abs() / 50.0).clip(0.0, 1.0)

    macd_hist = pd.to_numeric(result.get("macd_histogram", pd.Series(0.0, index=result.index)), errors="coerce")
    macd_restoring = (1.0 - macd_hist.abs() / macd_hist.abs().rolling(20, min_periods=5).quantile(0.95).replace(0, np.nan)).clip(0.0, 1.0)
    macd_restoring = macd_restoring.fillna(0.5)

    rupture = pd.to_numeric(
        result.get("rupture_pressure_score", pd.Series(0.0, index=result.index)),
        errors="coerce",
    ).fillna(0.0)
    pressure_relief = (1.0 - rupture / rupture.rolling(20, min_periods=5).quantile(0.95).replace(0, np.nan)).clip(0.0, 1.0)
    pressure_relief = pressure_relief.fillna(0.5)

    result["F_r"] = (
        0.30 * vwap_pull
        + 0.25 * boundary_restoring
        + 0.20 * rsi_restoring
        + 0.15 * macd_restoring
        + 0.10 * pressure_relief
    ).clip(0.0, 1.0)

    volume_inj = pd.to_numeric(
        result.get("volume_injection", pd.Series(1.0, index=result.index)),
        errors="coerce",
    ).fillna(1.0)
    volume_absorption = (1.0 / (1.0 + (volume_inj - 1.0).clip(lower=0.0))).clip(0.0, 1.0)

    e_i = pd.to_numeric(result.get("E_i", pd.Series(0.0, index=result.index)), errors="coerce").fillna(0.0)
    e_i_ceiling = e_i.rolling(20, min_periods=5).quantile(0.95).replace(0, np.nan)
    injection_dissipation = (1.0 - e_i / e_i_ceiling).clip(0.0, 1.0).fillna(0.5)

    rupture_decay = (-rupture.diff()).clip(lower=0.0)
    rupture_decay_norm = rupture_decay / rupture_decay.rolling(10, min_periods=3).quantile(0.95).replace(0, np.nan)
    rupture_decay_norm = rupture_decay_norm.fillna(0.0).clip(0.0, 1.0)

    close = pd.to_numeric(result.get("Close", pd.Series(np.nan, index=result.index)), errors="coerce")
    returns = close.pct_change()
    realized_vol = returns.rolling(20, min_periods=8).std()
    vol_dissipation = (1.0 / (1.0 + realized_vol * np.sqrt(252.0) * 8.0)).clip(0.0, 1.0).fillna(0.5)

    result["D_c"] = (
        0.30 * volume_absorption
        + 0.25 * injection_dissipation
        + 0.25 * vol_dissipation
        + 0.20 * rupture_decay_norm
    ).clip(0.0, 1.0)

    pressure_load = pd.to_numeric(
        result.get("LRP", result.get("rupture_pressure_score", pd.Series(0.05, index=result.index))),
        errors="coerce",
    ).fillna(0.05)
    result["restoration_ratio"] = (result["F_r"] / pressure_load.clip(lower=0.05)).clip(0.0, 3.0)
    result["dissipation_score"] = result["D_c"]

    return result
