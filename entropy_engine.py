"""Entropy engine — long-term degradation scoring.

Maps entropy_score inward to sacred E_sys via config/pressure_ontology.yaml.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_shannon_entropy(series: pd.Series, window: int, *, bins: int = 5) -> pd.Series:
    min_periods = max(5, window // 2)

    def entropy(values: np.ndarray) -> float:
        clean = values[~np.isnan(values)]
        if len(clean) < 3:
            return np.nan
        counts, _ = np.histogram(clean, bins=bins)
        total = counts.sum()
        if total == 0:
            return np.nan
        probs = counts[counts > 0] / total
        return float(-(probs * np.log2(probs)).sum() / np.log2(bins))

    return series.rolling(window, min_periods=min_periods).apply(entropy, raw=True)


def compute_entropy(df: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    """Compute E_sys / entropy_score for long-term degradation."""
    result = df.copy()
    close = pd.to_numeric(result.get("Close", pd.Series(np.nan, index=result.index)), errors="coerce")
    returns = close.pct_change()

    return_entropy = _rolling_shannon_entropy(returns, window=window).fillna(0.5)

    if "regime_label" in result.columns:
        codes = pd.Categorical(result["regime_label"]).codes.astype(float)
        codes[codes < 0] = np.nan
        regime_entropy = _rolling_shannon_entropy(pd.Series(codes, index=result.index), window=window).fillna(0.5)
    else:
        regime_entropy = pd.Series(0.5, index=result.index)

    hysteresis = pd.to_numeric(
        result.get("historical_stress_memory", result.get("H_s", pd.Series(0.0, index=result.index))),
        errors="coerce",
    ).fillna(0.0)

    result["entropy_score"] = (
        0.45 * return_entropy.clip(0.0, 1.0)
        + 0.35 * regime_entropy.clip(0.0, 1.0)
        + 0.20 * hysteresis
    ).clip(0.0, 1.0)
    result["E_sys"] = result["entropy_score"]

    return result
