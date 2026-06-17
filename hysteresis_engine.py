"""Hysteresis engine — historical stress memory and recursive pressure carryover.

Maps historical_stress_memory inward to sacred H_s via config/pressure_ontology.yaml.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_hysteresis(df: pd.DataFrame, *, memory_span: int = 20, carry_decay: float = 0.92) -> pd.DataFrame:
    """Compute stress memory, carryover, and causal sensitivity multipliers."""
    result = df.copy()

    rupture = pd.to_numeric(
        result.get("rupture_pressure_score", pd.Series(0.0, index=result.index)),
        errors="coerce",
    ).fillna(0.0)
    close = pd.to_numeric(result.get("Close", pd.Series(np.nan, index=result.index)), errors="coerce")
    returns = close.pct_change()

    h_s = rupture.ewm(span=memory_span, adjust=False, min_periods=3).mean().clip(0.0, 1.0)
    result["historical_stress_memory"] = h_s
    result["H_s"] = h_s

    d_pressure = rupture.diff().fillna(0.0).clip(lower=0.0)
    carryover = pd.Series(0.0, index=result.index)
    prev = 0.0
    for idx in result.index:
        injected = float(d_pressure.loc[idx])
        prev = carry_decay * prev + injected
        carryover.loc[idx] = prev

    ceiling = carryover.rolling(20, min_periods=5).quantile(0.95).replace(0, np.nan)
    result["recursive_pressure_carryover"] = (carryover / ceiling).clip(0.0, 1.0).fillna(0.0)

    result["stress_retention_multiplier"] = (1.0 + 0.25 * h_s).clip(1.0, 1.35)
    result["post_stress_sensitivity"] = (
        1.0 + 0.20 * result["recursive_pressure_carryover"] + 0.10 * h_s
    ).clip(1.0, 1.40)

    price_recovered = (returns > 0) & (h_s > 0.35)
    result["recovery_incomplete_flag"] = price_recovered.astype(int)

    return result
