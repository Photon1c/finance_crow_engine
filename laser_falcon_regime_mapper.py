"""Map Laser Falcon vol metrics into pressure-field vocabulary (finance-local only).

Does NOT edit sacred ontology — map inward via config/pressure_ontology.yaml.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _clip01(value: Optional[float]) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(np.clip(float(value), 0.0, 1.0))


def map_laser_falcon_regime_metrics(
    *,
    skew_metrics: dict[str, Any],
    surface_report: dict[str, Any],
    ou_result: dict[str, Any],
    data_health: dict[str, Any],
) -> dict[str, float]:
    """Produce local pressure-vocabulary scores from Laser Falcon outputs."""
    atm_iv = skew_metrics.get("atm_iv") or 0.0
    put_wing = skew_metrics.get("put_wing_iv") or atm_iv
    call_wing = skew_metrics.get("call_wing_iv") or atm_iv
    skew_slope = abs(skew_metrics.get("skew_slope") or 0.0)

    iv_pressure_score = _clip01(atm_iv / 100.0 if atm_iv and atm_iv > 1 else atm_iv)
    wing_spread = max(put_wing - atm_iv, 0.0) + max(call_wing - atm_iv, 0.0)
    skew_instability_score = _clip01(
        wing_spread / max(atm_iv, 1.0) * 0.5
        + (0.2 if skew_metrics.get("skew_inversion_flag") else 0.0)
        + min(skew_slope * 0.01, 0.3)
    )

    surface_status = surface_report.get("status", "SKIPPED")
    density = surface_report.get("density", {})
    surface_dislocation_score = 0.0
    if surface_status == "OK":
        surface_dislocation_score = _clip01(min(density.get("n_points", 0) / 100.0, 0.4) + wing_spread / max(atm_iv, 1.0) * 0.3)
    elif data_health.get("status") == "SPARSE":
        surface_dislocation_score = 0.55

    iv0 = ou_result.get("iv0", iv_pressure_score)
    terminal = ou_result.get("terminal_mean", iv0)
    vol_reversion_pressure = _clip01(abs(terminal - iv0) / max(iv0, 0.05))

    quote_unstable_pct = data_health.get("quote_unstable_pct", 0.0) / 100.0
    iv_coverage = data_health.get("iv_coverage_pct", 100.0) / 100.0
    option_liquidity_risk = _clip01(quote_unstable_pct * 0.6 + (1.0 - iv_coverage) * 0.4)

    return {
        "iv_pressure_score": round(iv_pressure_score, 4),
        "skew_instability_score": round(skew_instability_score, 4),
        "surface_dislocation_score": round(surface_dislocation_score, 4),
        "vol_reversion_pressure": round(vol_reversion_pressure, 4),
        "option_liquidity_risk": round(option_liquidity_risk, 4),
    }
