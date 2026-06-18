"""Rupture propagation metrics — Aerotrader pilot upgrade (CSV replay).

Models collective execution phases, persistence decay, restoration coefficient,
and interpretive latency from pressure-field sensors. Finance-local labels only;
map inward via config/pressure_ontology.yaml.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

EXECUTION_REGIMES = (
    "TYPE_I_DISTRIBUTED_EXECUTION",
    "TYPE_II_COLLECTIVE_EXECUTION",
    "TYPE_IIb_INTERRUPTED_EXECUTION",
    "TYPE_III_REFLEXIVE_COLLECTIVE_EXECUTION",
    "TYPE_IV_DISSIPATION_TRANSITION",
)

RUPTURE_PROPAGATION_EXPORT_COLUMNS = (
    "price_velocity",
    "price_acceleration",
    "synchronization_coefficient",
    "persistence_decay_rate",
    "cascade_energy",
    "persistence_half_life",
    "restoration_coefficient",
    "restoration_reentry_probability",
    "interpretive_latency",
    "execution_regime",
    "dissipation_onset_flag",
    "hold_position_score",
    "reduce_exposure_score",
)


def _as_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _rolling_norm(series: pd.Series, *, window: int = 20, min_periods: int = 5) -> pd.Series:
    ceiling = series.abs().rolling(window, min_periods=min_periods).quantile(0.95).replace(0, np.nan)
    return (series.abs() / ceiling).clip(0.0, 1.0).fillna(0.0)


def _compute_price_kinematics(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    velocity = close.pct_change().fillna(0.0)
    acceleration = velocity.diff().fillna(0.0)
    return velocity, acceleration


def _compute_synchronization_coefficient(df: pd.DataFrame, *, window: int = 10) -> pd.Series:
    close = _as_series(df, "Close")
    velocity = close.pct_change().fillna(0.0)
    signed = np.sign(velocity).replace(0, np.nan).fillna(0.0)
    cvd = _as_series(df, "cvd_imbalance")
    if cvd.eq(0).all():
        cvd = _as_series(df, "cvd_slope")
    cvd_sign = np.sign(cvd).replace(0, np.nan).fillna(0.0)
    alignment = (signed * cvd_sign > 0).astype(float)
    vol_inj = _as_series(df, "volume_injection", default=1.0)
    vol_spike = (vol_inj / vol_inj.rolling(window, min_periods=3).median().replace(0, np.nan)).clip(0.0, 3.0).fillna(1.0)
    t_a = _as_series(df, "T_a_norm")
    if t_a.eq(0).all():
        t_a = _as_series(df, "T_a").abs()
    sync = (
        0.40 * alignment.rolling(window, min_periods=3).mean().fillna(0.0)
        + 0.30 * _rolling_norm(vol_inj, window=window)
        + 0.30 * t_a.abs().clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    return (sync * (0.65 + 0.35 * (vol_spike / 3.0))).clip(0.0, 1.0)


def _compute_restoration_coefficient(close: pd.Series, *, window: int = 5) -> pd.Series:
    delta = close.diff().fillna(0.0)
    impulse = (-delta).clip(lower=0.0).rolling(window, min_periods=2).sum()
    recovery = delta.clip(lower=0.0).rolling(window, min_periods=2).sum()
    denom = impulse.replace(0, np.nan)
    r_c = (recovery / denom).clip(0.0, 2.0).fillna(0.5)
    return r_c.clip(0.0, 1.0)


def _compute_persistence_decay_rate(velocity: pd.Series, acceleration: pd.Series, *, window: int = 5) -> pd.Series:
    directional = velocity.abs() > velocity.rolling(10, min_periods=3).std().fillna(0.0)
    accel_mag = acceleration.abs()
    accel_declining = accel_mag < accel_mag.shift(1)
    same_direction = (np.sign(velocity) == np.sign(velocity.shift(1))).astype(float)
    decay_proxy = (directional & accel_declining & (same_direction > 0)).astype(float)
    rolling_decay = decay_proxy.rolling(window, min_periods=2).mean().fillna(0.0)
    near_zero_accel = (accel_mag < accel_mag.rolling(10, min_periods=3).quantile(0.35).fillna(0.0)).astype(float)
    return (0.60 * rolling_decay + 0.40 * near_zero_accel * directional.astype(float)).clip(0.0, 1.0)


def _compute_cascade_energy(df: pd.DataFrame, velocity: pd.Series) -> pd.Series:
    lrp = _as_series(df, "LRP", default=_as_series(df, "rupture_pressure_score"))
    d_c = _as_series(df, "D_c", default=0.5)
    cvd = _as_series(df, "cvd_imbalance").abs()
    if cvd.eq(0).all():
        cvd = _as_series(df, "cvd_slope").abs()
    momentum = _rolling_norm(velocity, window=15)
    carry = _as_series(df, "recursive_pressure_carryover")
    return (0.35 * lrp + 0.25 * momentum + 0.20 * cvd.clip(0, 1) + 0.10 * (1.0 - d_c) + 0.10 * carry).clip(0.0, 1.0)


def _compute_interpretive_latency(df: pd.DataFrame, *, window: int = 8) -> pd.Series:
    close = _as_series(df, "Close")
    returns = close.pct_change().fillna(0.0)
    choppiness = returns.rolling(window, min_periods=3).std() / returns.abs().rolling(window, min_periods=3).mean().replace(0, np.nan)
    choppiness = choppiness.fillna(0.0).clip(0.0, 5.0) / 5.0
    vol_inj = _as_series(df, "volume_injection", default=1.0)
    high_vol_low_move = (
        (vol_inj > 1.2)
        & (returns.abs() < returns.abs().rolling(20, min_periods=5).median().fillna(0.0))
    ).astype(float)
    c_w = _as_series(df, "C_w", default=_as_series(df, "capillary_wave_score"))
    r_o = _as_series(df, "R_o", default=0.5)
    blindspot = (1.0 - r_o).clip(0.0, 1.0)
    return (0.35 * choppiness + 0.30 * high_vol_low_move.rolling(3, min_periods=1).mean() + 0.20 * c_w + 0.15 * blindspot).clip(0.0, 1.0)


def _compute_persistence_half_life(persistence_decay_rate: pd.Series) -> pd.Series:
    daily_decay = persistence_decay_rate.clip(0.01, 1.0) / 20.0
    half_life = np.log(2.0) / daily_decay
    return half_life.clip(1.0, 500.0)


def classify_execution_regime_row(row: pd.Series) -> str:
    """Assign Aerotrader execution process state from local sensors."""
    p_d = float(row.get("persistence_decay_rate", 0.0) or 0.0)
    cascade = float(row.get("cascade_energy", 0.0) or 0.0)
    sync = float(row.get("synchronization_coefficient", 0.0) or 0.0)
    i_l = float(row.get("interpretive_latency", 0.0) or 0.0)
    lrp = float(row.get("LRP", row.get("rupture_pressure_score", 0.0)) or 0.0)
    r_c = float(row.get("restoration_coefficient", 0.0) or 0.0)
    f_r = float(row.get("F_r", 0.5) or 0.5)
    b_s = float(row.get("B_s", 0.5) or 0.5)
    t_a = float(row.get("T_a_norm", row.get("T_a", 0.0)) or 0.0)
    velocity = float(row.get("price_velocity", 0.0) or 0.0)

    if p_d >= 0.55 and cascade < 0.40:
        return "TYPE_IV_DISSIPATION_TRANSITION"
    if cascade >= 0.50 and sync >= 0.40 and r_c < 0.45 and velocity < 0:
        return "TYPE_III_REFLEXIVE_COLLECTIVE_EXECUTION"
    if i_l >= 0.48 and sync >= 0.28 and abs(velocity) < 0.004:
        return "TYPE_IIb_INTERRUPTED_EXECUTION"
    if sync >= 0.52 and abs(t_a) >= 0.30 and abs(velocity) >= 0.003:
        return "TYPE_II_COLLECTIVE_EXECUTION"
    if lrp < 0.38 and b_s < 0.50 and f_r >= 0.42:
        return "TYPE_I_DISTRIBUTED_EXECUTION"
    return ""


def _decision_scores(row: pd.Series) -> tuple[float, float]:
    """Hold vs reduce exposure scores from pilot decision framework."""
    sync = float(row.get("synchronization_coefficient", 0.0) or 0.0)
    p_d = float(row.get("persistence_decay_rate", 0.0) or 0.0)
    cascade = float(row.get("cascade_energy", 0.0) or 0.0)
    r_c = float(row.get("restoration_coefficient", 0.0) or 0.0)
    cvd = abs(float(row.get("cvd_imbalance", 0.0) or 0.0))
    accel = float(row.get("price_acceleration", 0.0) or 0.0)
    velocity = float(row.get("price_velocity", 0.0) or 0.0)

    hold = (
        0.25 * sync
        + 0.20 * cvd
        + 0.20 * cascade
        + 0.15 * (1.0 - p_d)
        + 0.10 * (1.0 - r_c)
        + 0.10 * (1.0 if velocity < 0 and accel < 0 else 0.0)
    )
    reduce = (
        0.30 * p_d
        + 0.25 * (1.0 - abs(accel))
        + 0.20 * r_c
        + 0.15 * (1.0 - cascade)
        + 0.10 * (1.0 if abs(velocity) < 0.002 else 0.0)
    )
    return float(np.clip(hold, 0.0, 1.0)), float(np.clip(reduce, 0.0, 1.0))


def compute_rupture_propagation(df: pd.DataFrame) -> pd.DataFrame:
    """Add rupture propagation metrics and execution regime labels."""
    result = df.copy()
    close = _as_series(result, "Close", default=np.nan)
    if close.isna().all():
        close = _as_series(result, "close", default=1.0)

    velocity, acceleration = _compute_price_kinematics(close)
    result["price_velocity"] = velocity
    result["price_acceleration"] = acceleration
    result["synchronization_coefficient"] = _compute_synchronization_coefficient(result)
    result["restoration_coefficient"] = _compute_restoration_coefficient(close)
    result["persistence_decay_rate"] = _compute_persistence_decay_rate(velocity, acceleration)
    result["cascade_energy"] = _compute_cascade_energy(result, velocity)
    result["persistence_half_life"] = _compute_persistence_half_life(result["persistence_decay_rate"])
    result["interpretive_latency"] = _compute_interpretive_latency(result)

    f_r = _as_series(result, "F_r", default=0.5)
    d_c = _as_series(result, "D_c", default=0.5)
    result["restoration_reentry_probability"] = (
        (1.0 - result["restoration_coefficient"]) * f_r * d_c
    ).clip(0.0, 1.0)

    result["execution_regime"] = result.apply(classify_execution_regime_row, axis=1)
    result["dissipation_onset_flag"] = (
        (result["persistence_decay_rate"] > 0.50)
        & (result["cascade_energy"].diff().fillna(0.0) < 0)
    ).astype(int)

    hold_scores: list[float] = []
    reduce_scores: list[float] = []
    for _, row in result.iterrows():
        hold, reduce = _decision_scores(row)
        hold_scores.append(hold)
        reduce_scores.append(reduce)
    result["hold_position_score"] = hold_scores
    result["reduce_exposure_score"] = reduce_scores

    return result


def build_propagation_snapshot(latest: pd.Series) -> dict[str, Any]:
    """Latest-row propagation metrics for JSON export."""
    return {
        "execution_regime": str(latest.get("execution_regime", "") or ""),
        "synchronization_coefficient": float(latest.get("synchronization_coefficient", 0.0) or 0.0),
        "persistence_decay_rate": float(latest.get("persistence_decay_rate", 0.0) or 0.0),
        "cascade_energy": float(latest.get("cascade_energy", 0.0) or 0.0),
        "persistence_half_life": float(latest.get("persistence_half_life", 0.0) or 0.0),
        "restoration_coefficient": float(latest.get("restoration_coefficient", 0.0) or 0.0),
        "restoration_reentry_probability": float(latest.get("restoration_reentry_probability", 0.0) or 0.0),
        "interpretive_latency": float(latest.get("interpretive_latency", 0.0) or 0.0),
        "dissipation_onset_flag": int(latest.get("dissipation_onset_flag", 0) or 0),
        "hold_position_score": float(latest.get("hold_position_score", 0.0) or 0.0),
        "reduce_exposure_score": float(latest.get("reduce_exposure_score", 0.0) or 0.0),
        "price_velocity": float(latest.get("price_velocity", 0.0) or 0.0),
        "price_acceleration": float(latest.get("price_acceleration", 0.0) or 0.0),
    }


def detect_regime_phases(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Scan time series for execution regime phase segments."""
    if frame.empty or "execution_regime" not in frame.columns:
        return []
    phases: list[dict[str, Any]] = []
    current_regime = ""
    start_pos = 0
    for pos, (_, row) in enumerate(frame.iterrows()):
        regime = str(row.get("execution_regime", "") or "")
        if regime != current_regime:
            if current_regime:
                segment = frame.iloc[start_pos:pos]
                phases.append(_phase_segment(current_regime, segment))
            current_regime = regime
            start_pos = pos
    if current_regime:
        phases.append(_phase_segment(current_regime, frame.iloc[start_pos:]))
    return [p for p in phases if p["regime"]]


def _phase_segment(regime: str, segment: pd.DataFrame) -> dict[str, Any]:
    return {
        "regime": regime,
        "start_date": str(segment.iloc[0].get("Date", "")),
        "end_date": str(segment.iloc[-1].get("Date", "")),
        "bars": int(len(segment)),
        "mean_cascade_energy": float(segment["cascade_energy"].mean()) if "cascade_energy" in segment else None,
        "mean_persistence_decay_rate": float(segment["persistence_decay_rate"].mean())
        if "persistence_decay_rate" in segment
        else None,
    }
