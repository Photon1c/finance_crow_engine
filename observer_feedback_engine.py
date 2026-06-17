"""Observer feedback engine — observer resolution modulates perceived system state.

Maps observer_feedback_score inward to sacred O_i via config/pressure_ontology.yaml.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_observer_feedback(df: pd.DataFrame, *, feedback_gain: float = 0.35) -> pd.DataFrame:
    """Compute O_i, effective_pressure, observer_state_coupling."""
    result = df.copy()

    r_o = pd.to_numeric(result.get("R_o", pd.Series(0.5, index=result.index)), errors="coerce").fillna(0.5)
    rupture = pd.to_numeric(
        result.get("rupture_pressure_score", pd.Series(0.0, index=result.index)),
        errors="coerce",
    ).fillna(0.0)
    lrp = pd.to_numeric(result.get("LRP", pd.Series(np.nan, index=result.index)), errors="coerce")

    blindspot = (1.0 - r_o).clip(0.0, 1.0)
    result["O_i"] = (blindspot * feedback_gain).clip(0.0, 1.0)
    result["observer_feedback_score"] = result["O_i"]

    base_pressure = lrp.fillna(rupture)
    result["effective_pressure"] = (base_pressure * (1.0 + result["O_i"])).clip(0.0, 1.0)

    d_r_o = r_o.diff().fillna(0.0)
    result["observer_state_coupling"] = (d_r_o.abs() * blindspot).clip(0.0, 1.0)

    return result
