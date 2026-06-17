"""Inward Drift Index (IDI) — pressure moving inward toward coupling center.

Maps to TRPR/ontology derived metric IDI. Conservative first-pass finance-crow proxy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IDI_REGIMES = (
    "OUTWARD_DISSIPATION",
    "NEUTRAL_DRIFT",
    "INWARD_COUPLING",
    "INWARD_COMPRESSION",
)


def compute_inward_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Compute IDI, inward_drift_regime, coupling_center, drift_velocity."""
    result = df.copy()
    close = pd.to_numeric(result.get("Close", pd.Series(np.nan, index=result.index)), errors="coerce")
    returns = close.pct_change()

    vwap_dist = pd.to_numeric(result.get("vwap_distance_pct", pd.Series(0.0, index=result.index)), errors="coerce").fillna(0.0)
    gamma_dist = pd.to_numeric(result.get("gamma_flip_distance_pct", pd.Series(np.nan, index=result.index)), errors="coerce")

    inward_vwap = (-vwap_dist.diff()).fillna(0.0)
    inward_vwap_norm = (inward_vwap / inward_vwap.abs().rolling(10, min_periods=3).quantile(0.95).replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)

    if gamma_dist.notna().any():
        inward_gamma = (-gamma_dist.diff()).fillna(0.0)
        inward_gamma_norm = (inward_gamma / inward_gamma.abs().rolling(10, min_periods=3).quantile(0.95).replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)
        coupling_center = gamma_dist.abs().fillna(vwap_dist.abs())
        center_label = pd.Series("gamma_boundary", index=result.index)
        center_label = center_label.where(gamma_dist.notna(), "vwap")
    else:
        inward_gamma_norm = pd.Series(0.0, index=result.index)
        coupling_center = vwap_dist.abs()
        center_label = pd.Series("vwap", index=result.index)

    if {"High", "Low"}.issubset(result.columns):
        daily_range = (pd.to_numeric(result["High"], errors="coerce") - pd.to_numeric(result["Low"], errors="coerce")) / close.replace(0, np.nan)
        range_narrowing = (-daily_range.diff()).clip(lower=0.0)
        range_norm = (range_narrowing / range_narrowing.rolling(10, min_periods=3).quantile(0.95).replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)
    else:
        range_norm = returns.rolling(5, min_periods=3).std().diff().mul(-1).clip(lower=0.0).fillna(0.0)
        range_norm = (range_norm / range_norm.rolling(10, min_periods=3).quantile(0.95).replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)

    cvd_chop = pd.to_numeric(result.get("cvd_imbalance", pd.Series(0.0, index=result.index)), errors="coerce").abs()
    cvd_chop_rising = cvd_chop.diff().clip(lower=0.0)
    cvd_norm = (cvd_chop_rising / cvd_chop_rising.rolling(10, min_periods=3).quantile(0.95).replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)

    volume_rising = pd.to_numeric(result.get("volume_injection", pd.Series(1.0, index=result.index)), errors="coerce").fillna(1.0)
    vol_pressure = ((volume_rising - 1.0).clip(lower=0.0) / 1.5).clip(0.0, 1.0)

    result["drift_velocity"] = (
        0.30 * inward_vwap_norm + 0.25 * inward_gamma_norm + 0.20 * range_norm + 0.15 * cvd_norm + 0.10 * vol_pressure
    ).clip(0.0, 1.0)

    result["IDI"] = result["drift_velocity"].clip(0.0, 1.0)
    result["coupling_center"] = center_label
    result["inward_drift_regime"] = result["IDI"].map(_classify_idi_regime)

    return result


def _classify_idi_regime(score: float) -> str:
    if pd.isna(score):
        return ""
    value = float(score)
    if value < 0.25:
        return "OUTWARD_DISSIPATION"
    if value < 0.45:
        return "NEUTRAL_DRIFT"
    if value < 0.65:
        return "INWARD_COUPLING"
    return "INWARD_COMPRESSION"
