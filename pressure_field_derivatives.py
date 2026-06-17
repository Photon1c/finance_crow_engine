"""Derived pressure-field metrics: rate-of-change, LRP, and alerts.

LRP / R_o / T_v / T_a keys align with TRPR/ontology/packet_ontology.yaml (read-only).
Engines are not required to import the ontology loader yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

LRP_REGIMES = (
    "STABLE",
    "PRESSURE_BUILDING",
    "PRE_RUPTURE",
    "RUPTURE_IMMINENT",
)

LRP_CONFIDENCE_LEVELS = (
    "HIGH_CONFIDENCE",
    "MEDIUM_CONFIDENCE",
    "LOW_CONFIDENCE",
)

LRP_WEIGHTS = {
    "T_a_norm": 0.20,
    "observer_blindspot": 0.20,
    "gamma_flip_pressure": 0.20,
    "vwap_dislocation_pressure": 0.15,
    "cvd_imbalance_pressure": 0.10,
    "macd_pressure_acceleration": 0.15,
}

ABSORPTION_DAMPING = 0.12
SIGMOID_STEEPNESS = 4.0
SIGMOID_MIDPOINT = 0.5

DERIVATIVE_COLUMNS = (
    "d_macd_pressure",
    "d_rsi_energy",
    "d_cvd_force",
    "d_volume_energy",
    "d_vwap_attractor_distance",
    "d_gamma_flip_distance",
    "d_canopy_pressure",
    "d_observability_R_o",
    "d_visibility_horizon_T_v",
    "dd_canopy_pressure",
    "dd_observability_R_o",
    "dd_coherence_proxy",
    "d_R_o",
    "d_T_v",
    "d_vwap_distance",
)

LRP_COLUMNS = (
    "gamma_flip_distance_pct",
    "gamma_flip_pressure",
    "vwap_dislocation_pressure",
    "cvd_imbalance_pressure",
    "volume_absorption_capacity",
    "coherence_proxy",
    "macd_pressure_acceleration",
    "LRP_raw",
    "LRP",
    "LRP_regime",
    "LRP_confidence",
    "lrp_contrib_T_a",
    "lrp_contrib_observer",
    "lrp_contrib_gamma",
    "lrp_contrib_vwap",
    "lrp_contrib_cvd",
    "lrp_contrib_macd",
)

LRP_CONTRIB_COLUMNS = (
    "lrp_contrib_T_a",
    "lrp_contrib_observer",
    "lrp_contrib_gamma",
    "lrp_contrib_vwap",
    "lrp_contrib_cvd",
    "lrp_contrib_macd",
)


def clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def safe_delta(series: pd.Series, *, fill: float = 0.0) -> pd.Series:
    """First difference with safe fallback on missing/invalid values."""
    delta = series.astype(float).diff()
    delta = delta.replace([np.inf, -np.inf], np.nan).fillna(fill)
    return delta


def safe_second_delta(series: pd.Series, *, fill: float = 0.0) -> pd.Series:
    """Second difference with safe fallback."""
    return safe_delta(safe_delta(series, fill=fill), fill=fill)


def sigmoid_lrp(lrp_raw: pd.Series) -> pd.Series:
    """Compress raw LRP through sigmoid to reduce saturation at 1.0."""
    x = lrp_raw.astype(float).replace([np.inf, -np.inf], np.nan).fillna(SIGMOID_MIDPOINT)
    return 1.0 / (1.0 + np.exp(-SIGMOID_STEEPNESS * (x - SIGMOID_MIDPOINT)))


def classify_lrp_regime(lrp: float) -> str:
    if lrp is None or pd.isna(lrp):
        return ""
    value = clip01(float(lrp))
    if value < 0.30:
        return "STABLE"
    if value < 0.60:
        return "PRESSURE_BUILDING"
    if value < 0.85:
        return "PRE_RUPTURE"
    return "RUPTURE_IMMINENT"


def _series_clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def _sanitize_series(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan)


def _as_float_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _normalize_t_a_term(t_a_norm: pd.Series) -> pd.Series:
    positive = t_a_norm.astype(float).clip(lower=0.0)
    return _series_clip01(positive / 1.25)


def _gamma_flip_pressure(distance_pct: pd.Series) -> pd.Series:
    distance = distance_pct.astype(float).abs()
    return _series_clip01(1.0 - (distance / 3.0))


def _vwap_dislocation_pressure(vwap_distance_pct: pd.Series) -> pd.Series:
    return _series_clip01(vwap_distance_pct.astype(float).abs() / 3.0)


def _cvd_imbalance_pressure(cvd_imbalance: pd.Series) -> pd.Series:
    return _series_clip01(cvd_imbalance.astype(float).abs() / 0.75)


def _volume_absorption_capacity(volume_injection: pd.Series) -> pd.Series:
    injection = volume_injection.astype(float)
    return _series_clip01(1.5 - injection.clip(upper=1.5))


def _macd_pressure_acceleration(df: pd.DataFrame) -> pd.Series:
    if "macd_pressure_accel" in df.columns:
        accel = _as_float_series(df, "macd_pressure_accel").abs()
        return _series_clip01(accel / 2.0)
    hist = _as_float_series(df, "macd_histogram").abs()
    return _series_clip01(hist / 3.0)


def _coherence_proxy(df: pd.DataFrame) -> pd.Series:
    if "packet_completion_confidence" in df.columns and "continuation_probability" in df.columns:
        completion = pd.to_numeric(df["packet_completion_confidence"], errors="coerce")
        continuation = pd.to_numeric(df["continuation_probability"], errors="coerce")
        proxy = (completion + continuation) / 2.0
        if proxy.notna().any():
            return proxy
    if "hidden_process_uncertainty" in df.columns:
        return _series_clip01(1.0 - df["hidden_process_uncertainty"].astype(float))
    return pd.Series(np.nan, index=df.index, dtype=float)


def _build_lrp_components(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Normalized component series used by calibrated and legacy LRP models."""
    gamma_distance = df.get("gamma_flip_distance_pct", pd.Series(np.nan, index=df.index))
    volume_source = df.get("volume_injection", df.get("E_i", pd.Series(1.0, index=df.index)))

    return {
        "T_a_norm": _normalize_t_a_term(_as_float_series(df, "T_a_norm", default=0.0)),
        "observer_blindspot": _series_clip01(1.0 - _as_float_series(df, "R_o", default=0.5)),
        "gamma_flip_pressure": _gamma_flip_pressure(gamma_distance.fillna(999.0)).where(
            gamma_distance.notna(), 0.0
        ),
        "vwap_dislocation_pressure": _vwap_dislocation_pressure(
            _as_float_series(df, "vwap_distance_pct", default=0.0)
        ),
        "cvd_imbalance_pressure": _cvd_imbalance_pressure(
            _as_float_series(df, "cvd_imbalance", default=0.0)
        ),
        "volume_absorption_capacity": _volume_absorption_capacity(
            pd.to_numeric(volume_source, errors="coerce").fillna(1.0)
        ),
        "B_s": _as_float_series(df, "B_s", default=0.0),
        "macd_pressure_acceleration": _macd_pressure_acceleration(df),
    }


def _weighted_lrp_raw(components: dict[str, pd.Series]) -> pd.Series:
    """Weighted additive model with mild absorption dampening."""
    weighted = (
        LRP_WEIGHTS["T_a_norm"] * components["T_a_norm"]
        + LRP_WEIGHTS["observer_blindspot"] * components["observer_blindspot"]
        + LRP_WEIGHTS["gamma_flip_pressure"] * components["gamma_flip_pressure"]
        + LRP_WEIGHTS["vwap_dislocation_pressure"] * components["vwap_dislocation_pressure"]
        + LRP_WEIGHTS["cvd_imbalance_pressure"] * components["cvd_imbalance_pressure"]
        + LRP_WEIGHTS["macd_pressure_acceleration"] * components["macd_pressure_acceleration"]
    )
    absorption = _series_clip01(components["B_s"] + components["volume_absorption_capacity"])
    return (weighted * (1.0 - ABSORPTION_DAMPING * absorption)).clip(lower=0.0, upper=1.25)


def _row_component_snapshot(row: pd.Series, components: dict[str, pd.Series], idx: Any) -> dict[str, float]:
    snapshot = {
        "T_a_norm": round(float(components["T_a_norm"].loc[idx]), 6),
        "one_minus_R_o": round(float(components["observer_blindspot"].loc[idx]), 6),
        "gamma_flip_pressure": round(float(components["gamma_flip_pressure"].loc[idx]), 6),
        "vwap_dislocation_pressure": round(float(components["vwap_dislocation_pressure"].loc[idx]), 6),
        "cvd_imbalance_pressure": round(float(components["cvd_imbalance_pressure"].loc[idx]), 6),
        "B_s": round(float(components["B_s"].loc[idx]), 6),
        "volume_absorption_capacity": round(float(components["volume_absorption_capacity"].loc[idx]), 6),
        "macd_pressure_acceleration": round(float(components["macd_pressure_acceleration"].loc[idx]), 6),
    }
    return snapshot


def compute_lrp_confidence(
    row: pd.Series,
    *,
    gamma_available: bool,
    history_depth: int,
    min_history: int = 20,
) -> str:
    """Estimate confidence in LRP from data completeness and lookback depth."""
    checks = 0
    score = 0.0

    def _check(field: str, weight: float = 1.0) -> None:
        nonlocal checks, score
        checks += 1
        value = row.get(field)
        if value is not None and not pd.isna(value):
            score += weight

    _check("T_a_norm")
    _check("R_o")
    _check("rupture_pressure_score")
    _check("vwap_distance_pct")
    _check("cvd_imbalance")
    _check("macd_histogram", 0.5)
    if gamma_available:
        score += 1.0
        checks += 1

    completeness = score / max(checks, 1)
    if completeness >= 0.85 and gamma_available and history_depth >= min_history:
        return "HIGH_CONFIDENCE"
    if completeness >= 0.60:
        return "MEDIUM_CONFIDENCE"
    return "LOW_CONFIDENCE"


def apply_gamma_distance(
    df: pd.DataFrame,
    *,
    gamma_flip_strike: Optional[float],
) -> pd.DataFrame:
    """Attach per-row gamma flip distance using a fixed flip strike when available."""
    result = df.copy()
    if gamma_flip_strike is None or pd.isna(gamma_flip_strike):
        result["gamma_flip_distance_pct"] = np.nan
        return result

    strike = float(gamma_flip_strike)
    close = result["Close"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = _sanitize_series((close - strike) / close * 100.0)
    result["gamma_flip_distance_pct"] = distance
    return result


def compute_lrp(
    df: pd.DataFrame,
    *,
    gamma_available: bool = False,
) -> pd.DataFrame:
    """Compute calibrated LRP with weighted raw score and sigmoid compression."""
    result = df.copy()
    components = _build_lrp_components(result)

    result["gamma_flip_pressure"] = components["gamma_flip_pressure"]
    result["vwap_dislocation_pressure"] = components["vwap_dislocation_pressure"]
    result["cvd_imbalance_pressure"] = components["cvd_imbalance_pressure"]
    result["volume_absorption_capacity"] = components["volume_absorption_capacity"]
    result["macd_pressure_acceleration"] = components["macd_pressure_acceleration"]
    result["coherence_proxy"] = _coherence_proxy(result)

    result["lrp_contrib_T_a"] = LRP_WEIGHTS["T_a_norm"] * components["T_a_norm"]
    result["lrp_contrib_observer"] = LRP_WEIGHTS["observer_blindspot"] * components["observer_blindspot"]
    result["lrp_contrib_gamma"] = LRP_WEIGHTS["gamma_flip_pressure"] * components["gamma_flip_pressure"]
    result["lrp_contrib_vwap"] = LRP_WEIGHTS["vwap_dislocation_pressure"] * components["vwap_dislocation_pressure"]
    result["lrp_contrib_cvd"] = LRP_WEIGHTS["cvd_imbalance_pressure"] * components["cvd_imbalance_pressure"]
    result["lrp_contrib_macd"] = (
        LRP_WEIGHTS["macd_pressure_acceleration"] * components["macd_pressure_acceleration"]
    )

    result["LRP_raw"] = _weighted_lrp_raw(components)
    result["LRP"] = sigmoid_lrp(result["LRP_raw"])
    result["LRP_regime"] = result["LRP"].apply(classify_lrp_regime)

    confidence_labels = []
    for position, (_, row) in enumerate(result.iterrows(), start=1):
        confidence_labels.append(
            compute_lrp_confidence(
                row,
                gamma_available=gamma_available,
                history_depth=position,
            )
        )
    result["LRP_confidence"] = confidence_labels
    return result


def compute_pressure_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """Compute first/second derivatives for pressure-field variables."""
    result = df.copy()

    macd_source = result["macd_histogram"] if "macd_histogram" in result.columns else pd.Series(0.0, index=result.index)
    result["d_macd_pressure"] = safe_delta(macd_source)
    result["d_rsi_energy"] = safe_delta(_as_float_series(result, "rsi_energy"))
    result["d_cvd_force"] = safe_delta(_as_float_series(result, "cvd_imbalance"))
    volume_source = (
        result["volume_injection"]
        if "volume_injection" in result.columns
        else _as_float_series(result, "E_i")
    )
    result["d_volume_energy"] = safe_delta(volume_source)
    result["d_vwap_attractor_distance"] = safe_delta(_as_float_series(result, "vwap_distance_pct"))
    result["d_vwap_distance"] = result["d_vwap_attractor_distance"]
    result["d_gamma_flip_distance"] = safe_delta(_as_float_series(result, "gamma_flip_distance_pct"))

    canopy_pressure = _as_float_series(result, "rupture_pressure_score")
    result["d_canopy_pressure"] = safe_delta(canopy_pressure)
    result["dd_canopy_pressure"] = safe_second_delta(canopy_pressure)

    r_o = pd.to_numeric(result.get("R_o", pd.Series(np.nan, index=result.index)), errors="coerce")
    result["d_observability_R_o"] = safe_delta(r_o)
    result["d_R_o"] = result["d_observability_R_o"]
    result["dd_observability_R_o"] = safe_second_delta(r_o)

    t_v = pd.to_numeric(result.get("T_v", pd.Series(np.nan, index=result.index)), errors="coerce")
    result["d_visibility_horizon_T_v"] = safe_delta(t_v)
    result["d_T_v"] = result["d_visibility_horizon_T_v"]

    if "coherence_proxy" not in result.columns:
        result["coherence_proxy"] = _coherence_proxy(result)
    result["dd_coherence_proxy"] = safe_second_delta(result["coherence_proxy"])

    return result


def enrich_pressure_derivatives(
    df: pd.DataFrame,
    *,
    gamma: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Apply gamma distance, LRP, and derivative columns to a pressure frame."""
    gamma = gamma or {}
    gamma_available = gamma.get("gamma_flip_strike") is not None and not pd.isna(gamma.get("gamma_flip_strike"))
    frame = apply_gamma_distance(df, gamma_flip_strike=gamma.get("gamma_flip_strike"))
    frame = compute_lrp(frame, gamma_available=gamma_available)
    frame = compute_pressure_derivatives(frame)
    return frame


def select_calibration_sessions(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Pick low/normal/high pressure archetype sessions from history."""
    valid = df.dropna(subset=["LRP", "rupture_pressure_score"]).copy()
    if valid.empty:
        return {}

    valid = valid.sort_index()
    low_pool = valid[
        (valid["rupture_pressure_score"] <= valid["rupture_pressure_score"].quantile(0.20))
        & (valid.get("B_s", 0).fillna(0) <= 0.25)
    ]
    high_pool = valid[
        (valid["rupture_pressure_score"] >= valid["rupture_pressure_score"].quantile(0.80))
        | (valid.get("regime_label", "").astype(str) == "RUPTURE_CANDIDATE")
    ]
    if low_pool.empty:
        low_pool = valid.nsmallest(max(1, len(valid) // 10), "rupture_pressure_score")
    if high_pool.empty:
        high_pool = valid.nlargest(max(1, len(valid) // 10), "rupture_pressure_score")

    median_lrp = float(valid["LRP"].median())
    normal_idx = (valid["LRP"] - median_lrp).abs().idxmin()
    low_row = low_pool.iloc[-1]
    high_row = high_pool.iloc[-1]
    normal_row = valid.loc[normal_idx]

    def _session(label: str, row: pd.Series) -> dict[str, Any]:
        date_val = row.get("Date")
        date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)
        return {
            "label": label,
            "date": date_str,
            "close": round(float(row.get("Close", float("nan"))), 4),
            "LRP": round(float(row.get("LRP", float("nan"))), 4),
            "LRP_raw": round(float(row.get("LRP_raw", float("nan"))), 4),
            "LRP_regime": str(row.get("LRP_regime", "")),
            "LRP_confidence": str(row.get("LRP_confidence", "")),
            "rupture_pressure_score": round(float(row.get("rupture_pressure_score", 0.0)), 4),
        }

    return {
        "LOW_VOLATILITY_DAY": _session("LOW_VOLATILITY_DAY", low_row),
        "NORMAL_CRUISE_DAY": _session("NORMAL_CRUISE_DAY", normal_row),
        "HIGH_PRESSURE_BREAKOUT_DAY": _session("HIGH_PRESSURE_BREAKOUT_DAY", high_row),
    }


def build_lrp_debug_payload(
    df: pd.DataFrame,
    *,
    ticker: str,
    gamma: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Audit legacy saturation and export calibrated component breakdown."""
    components = _build_lrp_components(df)
    latest_idx = df.index[-1]
    latest = df.iloc[-1]

    legacy_components = _row_component_snapshot(latest, components, latest_idx)
    legacy_numerator = (
        legacy_components["T_a_norm"]
        + legacy_components["one_minus_R_o"]
        + legacy_components["gamma_flip_pressure"]
        + legacy_components["vwap_dislocation_pressure"]
        + legacy_components["cvd_imbalance_pressure"]
    )
    legacy_denominator = (
        legacy_components["B_s"] + legacy_components["volume_absorption_capacity"] + 1e-9
    )
    legacy_ratio = legacy_numerator / legacy_denominator
    legacy_clamped = clip01(legacy_ratio)

    calibrated_contributions = {
        "T_a contribution": round(float(latest.get("lrp_contrib_T_a", 0.0)), 4),
        "Observer decay": round(float(latest.get("lrp_contrib_observer", 0.0)), 4),
        "Gamma boundary pressure": round(float(latest.get("lrp_contrib_gamma", 0.0)), 4),
        "VWAP dislocation": round(float(latest.get("lrp_contrib_vwap", 0.0)), 4),
        "CVD imbalance": round(float(latest.get("lrp_contrib_cvd", 0.0)), 4),
        "MACD acceleration": round(float(latest.get("lrp_contrib_macd", 0.0)), 4),
    }

    return {
        "ticker": ticker.upper(),
        "diagnosis": (
            "Legacy model sums five 0..1 pressure terms then divides by a small "
            "absorption denominator (~0.3-0.7), causing ratio > 1 and hard clamp saturation."
        ),
        "legacy_formula_audit": {
            "components": legacy_components,
            "numerator_sum": round(legacy_numerator, 6),
            "denominator": round(legacy_denominator, 6),
            "ratio_before_clamp": round(legacy_ratio, 6),
            "LRP_legacy_clamped": round(legacy_clamped, 6),
            "saturation_driver": (
                "denominator_too_weak"
                if legacy_denominator < 0.8 and legacy_numerator > 1.0
                else "numerator_stack"
            ),
        },
        "calibrated_model": {
            "weights": LRP_WEIGHTS,
            "absorption_damping": ABSORPTION_DAMPING,
            "sigmoid": f"1 / (1 + exp(-{SIGMOID_STEEPNESS} * (LRP_raw - {SIGMOID_MIDPOINT})))",
            "latest_LRP_raw": round(float(latest.get("LRP_raw", float("nan"))), 6),
            "latest_LRP": round(float(latest.get("LRP", float("nan"))), 6),
            "latest_LRP_regime": str(latest.get("LRP_regime", "")),
            "latest_LRP_confidence": str(latest.get("LRP_confidence", "")),
            "component_contributions": calibrated_contributions,
            "absorption_terms": {
                "B_s": legacy_components["B_s"],
                "volume_absorption_capacity": legacy_components["volume_absorption_capacity"],
            },
        },
        "historical_calibration": select_calibration_sessions(df),
        "distribution": {
            "LRP_mean": round(float(df["LRP"].mean()), 4),
            "LRP_median": round(float(df["LRP"].median()), 4),
            "LRP_p95": round(float(df["LRP"].quantile(0.95)), 4),
            "LRP_max": round(float(df["LRP"].max()), 4),
            "share_above_0_95": round(float((df["LRP"] >= 0.95).mean()), 4),
        },
    }


def write_lrp_debug_json(
    payload: dict[str, Any],
    json_path: Path,
) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_rate_of_change_alerts(
    latest: pd.Series,
    *,
    previous: Optional[pd.Series] = None,
) -> list[str]:
    """Derive human-readable rate-of-change alerts from latest derivative state."""
    alerts: list[str] = []

    d_r_o = _float_or_zero(latest.get("d_R_o"))
    dd_r_o = _float_or_zero(latest.get("dd_observability_R_o"))
    if d_r_o < -0.02 and dd_r_o < -0.01:
        alerts.append("OBSERVABILITY_DECAY_ACCELERATING")

    dd_coherence = _float_or_zero(latest.get("dd_coherence_proxy"))
    d_coherence = _float_or_zero(latest.get("coherence_proxy")) - _float_or_zero(
        previous.get("coherence_proxy") if previous is not None else latest.get("coherence_proxy")
    )
    if dd_coherence < -0.03 or (previous is not None and d_coherence < -0.04):
        alerts.append("COHERENCE_LOSS_ACCELERATING")

    gamma_distance = _float_or_zero(latest.get("gamma_flip_distance_pct"))
    d_gamma = _float_or_zero(latest.get("d_gamma_flip_distance"))
    if gamma_distance != 0.0 and abs(gamma_distance) < 2.0 and d_gamma * gamma_distance < 0:
        alerts.append("GAMMA_BOUNDARY_APPROACHING")
    elif abs(d_gamma) > 0.15 and abs(gamma_distance) < 3.0:
        alerts.append("GAMMA_BOUNDARY_APPROACHING")

    vwap_distance = _float_or_zero(latest.get("vwap_distance_pct"))
    d_vwap = _float_or_zero(latest.get("d_vwap_distance"))
    if abs(vwap_distance) > 0.5 and abs(d_vwap) > 0.05 and np.sign(d_vwap) == np.sign(vwap_distance):
        alerts.append("VWAP_DISLOCATION_EXPANDING")

    d_canopy = _float_or_zero(latest.get("d_canopy_pressure"))
    dd_canopy = _float_or_zero(latest.get("dd_canopy_pressure"))
    t_a_norm = _float_or_zero(latest.get("T_a_norm"))
    if d_canopy > 0.01 and (dd_canopy > 0.0 or t_a_norm > 0.25):
        alerts.append("PRESSURE_ACCELERATION_POSITIVE")

    lrp = _float_or_zero(latest.get("LRP"))
    confidence = str(latest.get("LRP_confidence", ""))
    if lrp >= 0.90 and confidence != "LOW_CONFIDENCE":
        alerts.append("RUPTURE_IMMINENT_LRP")

    return alerts


def _float_or_zero(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0
