"""Stable snapshot schema for pressure field and CanopyEnto latest JSON."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Core keys — preserve order for downstream scripts written before loop closure.
STABLE_CORE_SNAPSHOT_KEYS = (
    "ticker",
    "timestamp",
    "close",
    "macd_regime",
    "rsi_regime",
    "cvd_regime",
    "volume_ratio",
    "vwap_distance_pct",
    "gamma_flip",
    "gamma_flip_distance_pct",
    "canopy_regime",
    "T_a",
    "T_a_norm",
    "T_a_regime",
    "R_o",
    "T_v",
    "observer_profile",
    "LRP",
    "LRP_regime",
    "d_canopy_pressure",
    "dd_canopy_pressure",
    "d_R_o",
    "d_T_v",
    "d_gamma_flip_distance",
    "d_vwap_distance",
)

# Extended keys appended after core — experimental physics / loop closure.
STABLE_EXTENDED_SNAPSHOT_KEYS = (
    "LRP_adjusted",
    "LRP_adjusted_regime",
    "F_r",
    "D_c",
    "restoration_ratio",
    "dissipation_score",
    "A_micro",
    "C_w",
    "capillary_wave_score",
    "field_regime",
    "entropy_score",
)

STABLE_SNAPSHOT_KEYS = STABLE_CORE_SNAPSHOT_KEYS + STABLE_EXTENDED_SNAPSHOT_KEYS

LRP_DOCTRINE = (
    "Baseline LRP = pressure signal; "
    "LRP_adjusted (experimental) = pressure after restoration/capillary/hysteresis/observer modifiers."
)


def safe_float(value: Any) -> Optional[float]:
    """Convert to JSON-safe float; map NaN/inf to None."""
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def format_timestamp(date_val: Any) -> str:
    """Normalize a date/datetime into ISO-8601 UTC timestamp string."""
    if date_val is None or pd.isna(date_val):
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if hasattr(date_val, "strftime"):
        if hasattr(date_val, "hour"):
            return date_val.strftime("%Y-%m-%dT%H:%M:%SZ")
        return f"{date_val.strftime('%Y-%m-%d')}T00:00:00Z"
    return str(date_val)


def build_stable_snapshot(
    *,
    ticker: str,
    latest: pd.Series,
    spot: Optional[float] = None,
    gamma: Optional[dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> dict[str, Any]:
    """Build flat latest snapshot with stable keys for downstream ingestion."""
    gamma = gamma or {}
    close = safe_float(spot if spot is not None else latest.get("Close"))

    flip_strike = safe_float(gamma.get("gamma_flip_strike"))
    flip_distance = safe_float(gamma.get("distance_to_flip_pct"))

    values = {
        "ticker": ticker.upper(),
        "timestamp": timestamp or format_timestamp(latest.get("Date")),
        "close": close,
        "macd_regime": str(latest.get("macd_regime", "") or ""),
        "rsi_regime": str(latest.get("rsi_saturation", latest.get("rsi_regime", "")) or ""),
        "cvd_regime": str(latest.get("cvd_regime", "") or ""),
        "volume_ratio": safe_float(latest.get("volume_injection", latest.get("E_i"))),
        "vwap_distance_pct": safe_float(latest.get("vwap_distance_pct")),
        "gamma_flip": flip_strike,
        "gamma_flip_distance_pct": flip_distance,
        "canopy_regime": str(latest.get("regime_label", "") or ""),
        "T_a": safe_float(latest.get("T_a")),
        "T_a_norm": safe_float(latest.get("T_a_norm")),
        "T_a_regime": str(latest.get("T_a_regime", "") or ""),
        "R_o": safe_float(latest.get("R_o")),
        "T_v": safe_float(latest.get("T_v")),
        "observer_profile": str(latest.get("observer_profile", "") or ""),
        "LRP": safe_float(latest.get("LRP")),
        "LRP_regime": str(latest.get("LRP_regime", "") or ""),
        "d_canopy_pressure": safe_float(latest.get("d_canopy_pressure")),
        "dd_canopy_pressure": safe_float(latest.get("dd_canopy_pressure")),
        "d_R_o": safe_float(latest.get("d_R_o", latest.get("d_observability_R_o"))),
        "d_T_v": safe_float(latest.get("d_T_v", latest.get("d_visibility_horizon_T_v"))),
        "d_gamma_flip_distance": safe_float(latest.get("d_gamma_flip_distance")),
        "d_vwap_distance": safe_float(latest.get("d_vwap_distance", latest.get("d_vwap_attractor_distance"))),
        "LRP_adjusted": safe_float(latest.get("LRP_adjusted")),
        "LRP_adjusted_regime": str(latest.get("LRP_adjusted_regime", "") or ""),
        "F_r": safe_float(latest.get("F_r")),
        "D_c": safe_float(latest.get("D_c")),
        "restoration_ratio": safe_float(latest.get("restoration_ratio")),
        "dissipation_score": safe_float(latest.get("dissipation_score")),
        "A_micro": safe_float(latest.get("A_micro")),
        "C_w": safe_float(latest.get("C_w")),
        "capillary_wave_score": safe_float(latest.get("capillary_wave_score")),
        "field_regime": str(latest.get("field_regime", "") or ""),
        "entropy_score": safe_float(latest.get("entropy_score")),
    }
    return {key: values[key] for key in STABLE_SNAPSHOT_KEYS}


def write_stable_snapshot_json(
    snapshot: dict[str, Any],
    json_path: Path,
    *,
    extras: Optional[dict[str, Any]] = None,
) -> None:
    """Write stable snapshot JSON with core keys first, then extension blocks."""
    payload: dict[str, Any] = {key: snapshot[key] for key in STABLE_SNAPSHOT_KEYS if key in snapshot}
    if extras:
        payload.update(extras)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
