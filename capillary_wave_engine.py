"""Capillary wave engine — micro oscillation amplitude, wave persistence, capillary wave score.

Local oscillatory instability near boundary (distinct from capillary_engine.py macro absorption).
Maps A_micro inward to sacred oscillation_amplitude key A_f via config/pressure_ontology.yaml.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAPILLARY_WAVE_REGIMES = (
    "FLAT_SURFACE",
    "CAPILLARY_ACTIVITY",
    "SURFACE_TENSION_STRESS",
    "PRE_RUPTURE",
)


def _rolling_autocorrelation(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    min_periods = max(lag + 2, window // 2)

    def autocorr(values: np.ndarray) -> float:
        if len(values) <= lag:
            return np.nan
        left = values[:-lag]
        right = values[lag:]
        mask = ~(np.isnan(left) | np.isnan(right))
        left = left[mask]
        right = right[mask]
        if len(left) < 2:
            return np.nan
        if left.std() == 0 or right.std() == 0:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    return series.rolling(window, min_periods=min_periods).apply(autocorr, raw=True)


def classify_capillary_wave_regime(score: float) -> str:
    if pd.isna(score):
        return ""
    value = float(score)
    if value < 0.25:
        return "FLAT_SURFACE"
    if value < 0.45:
        return "CAPILLARY_ACTIVITY"
    if value < 0.65:
        return "SURFACE_TENSION_STRESS"
    return "PRE_RUPTURE"


def compute_capillary_wave(df: pd.DataFrame, *, short_window: int = 5, long_window: int = 20) -> pd.DataFrame:
    """Compute A_micro, wave_persistence, C_w, capillary_wave_regime."""
    result = df.copy()
    close = pd.to_numeric(result.get("Close", pd.Series(np.nan, index=result.index)), errors="coerce")
    returns = close.pct_change()

    short_std = returns.rolling(short_window, min_periods=3).std()
    long_std = returns.rolling(long_window, min_periods=8).std()
    amplitude_raw = short_std / long_std.replace(0, np.nan)
    result["A_micro"] = (amplitude_raw / 2.0).clip(0.0, 1.0).fillna(0.0)

    abs_returns = returns.abs()
    wave_raw = _rolling_autocorrelation(abs_returns, window=10, lag=1)
    result["wave_persistence"] = ((wave_raw + 1.0) / 2.0).clip(0.0, 1.0).fillna(0.5)

    d_c = pd.to_numeric(result.get("D_c", pd.Series(0.5, index=result.index)), errors="coerce").fillna(0.5)
    d_r = (1.0 - d_c).clip(0.05, 1.0)
    result["D_r"] = d_r

    capillary_raw = (result["A_micro"] * result["wave_persistence"]) / d_r
    ceiling = capillary_raw.rolling(20, min_periods=5).quantile(0.95).replace(0, np.nan)
    result["C_w"] = (capillary_raw / ceiling).clip(0.0, 1.0).fillna(0.0)
    result["capillary_wave_score"] = result["C_w"]
    result["capillary_wave_regime"] = result["C_w"].map(classify_capillary_wave_regime)

    d_c_val = pd.to_numeric(result.get("D_c", pd.Series(0.5, index=result.index)), errors="coerce").fillna(0.5)
    result["capillary_absorption_state"] = np.where(
        d_c_val >= 0.55,
        "ABSORBING",
        np.where(d_c_val >= 0.35, "PARTIAL_ABSORPTION", "ABSORPTION_FAILURE"),
    )

    return result
