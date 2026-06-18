"""Orchestrate pressure-field physics engines (Phases 1–3).

All metrics map inward to TRPR/ontology/packet_ontology.yaml — no local primitive redefinition.

Capillary layers:
- capillary_engine.py — macro/mid micro-noise absorption from CanopyEnto CSV
- capillary_wave_engine.py — local oscillatory instability near boundary
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from attractor_engine import compute_attractor_field
from capillary_wave_engine import compute_capillary_wave
from entropy_engine import compute_entropy
from field_regime_engine import compute_field_regimes
from hysteresis_engine import compute_hysteresis
from elastic_rebound_engine import ELASTIC_REBOUND_EXPORT_COLUMNS, compute_elastic_rebound
from rupture_propagation_engine import RUPTURE_PROPAGATION_EXPORT_COLUMNS, compute_rupture_propagation
from inward_drift_engine import compute_inward_drift
from observer_feedback_engine import compute_observer_feedback
from restoration_field_engine import compute_restoration_field

PHYSICS_EXPORT_COLUMNS = (
    "F_r",
    "D_c",
    "restoration_ratio",
    "dissipation_score",
    "A_micro",
    "wave_persistence",
    "D_r",
    "C_w",
    "capillary_wave_score",
    "capillary_wave_regime",
    "capillary_absorption_state",
    "A_field",
    "attractor_field_strength",
    "gamma_attractor_strength",
    "wall_pinning_strength",
    "equilibrium_pin_price",
    "attractor_deviation",
    "restorative_force_estimate",
    "equilibrium_field_strength",
    "deviation_from_equilibrium",
    "H_s",
    "historical_stress_memory",
    "recursive_pressure_carryover",
    "stress_retention_multiplier",
    "post_stress_sensitivity",
    "recovery_incomplete_flag",
    "E_sys",
    "entropy_score",
    "IDI",
    "inward_drift_regime",
    "coupling_center",
    "drift_velocity",
    "O_i",
    "observer_feedback_score",
    "effective_pressure",
    "observer_state_coupling",
    "field_regime",
    "d_F_r",
    *ELASTIC_REBOUND_EXPORT_COLUMNS,
    *RUPTURE_PROPAGATION_EXPORT_COLUMNS,
)


def enrich_pressure_physics(
    df: pd.DataFrame,
    *,
    gamma: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Run restoration → capillary wave → attractor → hysteresis → entropy → IDI → observer → regimes."""
    frame = compute_restoration_field(df)
    frame = compute_capillary_wave(frame)
    frame = compute_attractor_field(frame, gamma=gamma)
    frame = compute_hysteresis(frame)
    frame = compute_entropy(frame)
    frame = compute_inward_drift(frame)
    if "R_o" in frame.columns:
        frame = compute_observer_feedback(frame)
    else:
        frame["O_i"] = 0.0
        frame["observer_feedback_score"] = 0.0
        frame["effective_pressure"] = pd.to_numeric(
            frame.get("rupture_pressure_score", 0.0), errors="coerce"
        ).fillna(0.0)
        frame["observer_state_coupling"] = 0.0
    frame = compute_field_regimes(frame)
    frame = compute_elastic_rebound(frame, gamma=gamma)
    frame = compute_rupture_propagation(frame)
    return frame
