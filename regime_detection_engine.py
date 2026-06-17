"""Market volatility regime classification from options pressure metrics."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

REGIME_LABELS = (
    "LOW_VOL_REGIME",
    "HIGH_VOL_REGIME",
    "PANIC_REGIME",
    "POST_EVENT_CRUSH",
    "MELT_UP",
    "MEAN_REVERSION_LIKELY",
    "NORMAL",
)


def classify_vol_regime(
    *,
    skew_metrics: dict[str, Any],
    pressure_metrics: dict[str, Any],
    ou_result: Optional[dict[str, Any]] = None,
    anomaly: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Categorize market state from options-derived signals."""
    atm_iv = skew_metrics.get("atm_iv") or 0.0
    vol_exp = pressure_metrics.get("volatility_expansion_score") or 1.0
    skew_asym = pressure_metrics.get("skew_asymmetry_pressure") or 0.0
    iv0 = (ou_result or {}).get("iv0", atm_iv / 100.0 if atm_iv > 1 else atm_iv)
    terminal = (ou_result or {}).get("terminal_mean", iv0)
    reversion_delta = abs(terminal - iv0) if iv0 else 0.0

    primary = "NORMAL"
    if atm_iv < 15:
        primary = "LOW_VOL_REGIME"
    elif atm_iv > 50 and vol_exp > 2.0:
        primary = "PANIC_REGIME"
    elif atm_iv > 35:
        primary = "HIGH_VOL_REGIME"
    elif vol_exp > 2.5 and reversion_delta > 0.15:
        primary = "POST_EVENT_CRUSH"
    elif skew_asym is not None and skew_asym < -0.3 and skew_metrics.get("calls_overpriced_flag"):
        primary = "MELT_UP"
    elif reversion_delta > 0.1 and vol_exp > 1.8:
        primary = "MEAN_REVERSION_LIKELY"

    if anomaly and anomaly.get("primary_label") not in (None, "NORMAL"):
        if "IPO_INSTABILITY" in anomaly.get("labels", []):
            primary = "POST_EVENT_CRUSH" if primary == "NORMAL" else primary
        if "CALL_FOMO_DETECTED" in anomaly.get("labels", []):
            primary = "MELT_UP"

    confidence = min(
        1.0,
        0.3
        + (0.2 if atm_iv > 0 else 0)
        + (0.2 if vol_exp > 1.5 else 0)
        + (0.2 if anomaly and anomaly.get("primary_label") != "NORMAL" else 0),
    )

    return {
        "regime": primary,
        "confidence": round(float(confidence), 4),
        "atm_iv_pct": atm_iv,
        "volatility_expansion_score": vol_exp,
        "mean_reversion_delta": round(float(reversion_delta), 4),
    }
