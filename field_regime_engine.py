"""Field regime engine — classify sacred named regimes from computed physics metrics."""

from __future__ import annotations

import pandas as pd

NAMED_FIELD_REGIMES = (
    "RESTORED_EQUILIBRIUM",
    "SYNTHETIC_STABILITY",
    "ACTIVE_COMPENSATION",
    "WEAKENING_RESTORATION",
    "CAPILLARY_PRE_RUPTURE",
    "ENTROPIC_DEGRADATION",
    "COHERENT_MAINTENANCE_UNDER_STRESS",
)


def classify_field_regime_row(row: pd.Series) -> str:
    """Assign one primary named field regime (priority-ordered)."""
    restoration_ratio = float(row.get("restoration_ratio", 0.0) or 0.0)
    f_r = float(row.get("F_r", 0.0) or 0.0)
    d_c = float(row.get("D_c", 0.0) or 0.0)
    lrp = float(row.get("LRP", row.get("rupture_pressure_score", 0.0)) or 0.0)
    c_w = float(row.get("C_w", row.get("capillary_wave_score", 0.0)) or 0.0)
    a_micro = float(row.get("A_micro", row.get("A_f", 0.0)) or 0.0)
    entropy = float(row.get("entropy_score", row.get("E_sys", 0.0)) or 0.0)
    f_r_delta = float(row.get("d_F_r", 0.0) or 0.0)
    wall_pin = float(row.get("wall_pinning_strength", 0.0) or 0.0)
    recovery_incomplete = int(row.get("recovery_incomplete_flag", 0) or 0)
    h_s = float(row.get("H_s", row.get("historical_stress_memory", 0.0)) or 0.0)

    hidden_reservoir = float(row.get("hidden_reservoir_pressure", 0.0) or 0.0)
    false_stability = int(row.get("false_stability_flag", 0) or 0)

    if entropy >= 0.70:
        return "ENTROPIC_DEGRADATION"
    if c_w >= 0.65 and a_micro >= 0.45:
        return "CAPILLARY_PRE_RUPTURE"
    if recovery_incomplete and h_s >= 0.35:
        return "COHERENT_MAINTENANCE_UNDER_STRESS"
    if wall_pin >= 0.70 and f_r >= 0.45 and lrp < 0.55:
        return "ACTIVE_COMPENSATION"
    if (
        lrp < 0.35
        and f_r >= 0.60
        and 0.25 <= d_c <= 0.70
        and restoration_ratio >= 0.55
        and (
            a_micro >= 0.35
            or hidden_reservoir >= 0.25
            or false_stability == 1
            or (wall_pin >= 0.45 and c_w >= 0.20)
        )
    ):
        return "SYNTHETIC_STABILITY"
    if restoration_ratio >= 0.70 and lrp < 0.30:
        return "RESTORED_EQUILIBRIUM"
    if f_r >= 0.50 and d_c >= 0.50 and 0.30 <= lrp < 0.60:
        return "ACTIVE_COMPENSATION"
    if f_r < 0.40 or restoration_ratio < 0.40 or f_r_delta < -0.05:
        return "WEAKENING_RESTORATION"
    return ""


def compute_field_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Add field_regime column and d_F_r helper for regime transitions."""
    result = df.copy()
    if "F_r" in result.columns:
        result["d_F_r"] = pd.to_numeric(result["F_r"], errors="coerce").diff().fillna(0.0)
    else:
        result["d_F_r"] = 0.0
    result["field_regime"] = result.apply(classify_field_regime_row, axis=1)
    return result
