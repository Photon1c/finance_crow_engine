"""Elastic rebound and hidden reservoir metrics — gamma locked-fault / strain layer.

Finance-local labels only; map inward via config/pressure_ontology.yaml.
Not sacred ontology — additive market-domain pass.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

GAMMA_REBOUND_REGIMES = (
    "LOCKED_FAULT",
    "COMPRESSING_STRAIN",
    "PRE_RUPTURE",
    "REBOUND_RELEASE",
    "AFTERSHOCK_REPRICING",
    "NEW_EQUILIBRIUM",
)

ELASTIC_REBOUND_EXPORT_COLUMNS = (
    "elastic_strain_score",
    "gamma_rebound_regime",
    "hidden_reservoir_pressure",
    "pressure_relocation_ratio",
    "false_dissipation_risk",
    "false_stability_flag",
    "observability_gap_score",
)

RESERVOIR_DECAY = 0.92
DISSIPATION_LOW_THRESHOLD = 0.45


def _as_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _gamma_boundary_proximity(df: pd.DataFrame) -> pd.Series:
    if "gamma_attractor_strength" in df.columns:
        return _as_series(df, "gamma_attractor_strength").clip(0.0, 1.0)
    dist = _as_series(df, "gamma_flip_distance_pct", default=999.0)
    return (1.0 - dist.abs() / 5.0).clip(0.0, 1.0)


def _volatility_compression(df: pd.DataFrame) -> pd.Series:
    if "A_micro" in df.columns:
        return (1.0 - _as_series(df, "A_micro")).clip(0.0, 1.0)
    close = _as_series(df, "Close", default=np.nan)
    if close.isna().all():
        return pd.Series(0.5, index=df.index)
    returns = close.pct_change()
    short = returns.rolling(5, min_periods=3).std()
    long = returns.rolling(20, min_periods=8).std().replace(0, np.nan)
    ratio = (short / long).fillna(1.0)
    return (1.0 - ratio).clip(0.0, 1.0)


def _hidden_flow_divergence(df: pd.DataFrame) -> pd.Series:
    cvd = _as_series(df, "cvd_imbalance_pressure", default=_as_series(df, "cvd_imbalance").abs() / 0.75)
    vwap = _as_series(df, "vwap_dislocation_pressure", default=_as_series(df, "vwap_distance_pct").abs() / 3.0)
    blindspot = 1.0 - _as_series(df, "R_o", default=0.5)
    macd = _as_series(df, "macd_pressure_acceleration", default=_as_series(df, "macd_histogram").abs() / 3.0)
    return (0.30 * cvd.clip(0, 1) + 0.25 * vwap.clip(0, 1) + 0.25 * blindspot + 0.20 * macd.clip(0, 1)).clip(0.0, 1.0)


def _compression_duration_proxy(df: pd.DataFrame, *, window: int = 10) -> pd.Series:
    vol_comp = _volatility_compression(df)
    low_vol = (vol_comp > 0.55).astype(float)
    return low_vol.rolling(window, min_periods=3).mean().fillna(0.0).clip(0.0, 1.0)


def _compute_hidden_reservoir(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    lrp = _as_series(df, "LRP", default=_as_series(df, "rupture_pressure_score"))
    d_c = _as_series(df, "D_c", default=0.5)
    reservoir_values: list[float] = []
    relocation_values: list[float] = []
    false_dissipation_values: list[float] = []
    prev_lrp = float(lrp.iloc[0]) if len(lrp) else 0.0
    prev_reservoir = 0.0

    for idx in df.index:
        current_lrp = float(lrp.loc[idx])
        capacity = float(d_c.loc[idx])
        visible_drop = max(prev_lrp - current_lrp, 0.0)
        if capacity < DISSIPATION_LOW_THRESHOLD:
            injected = visible_drop * (1.0 - capacity)
            prev_reservoir = prev_reservoir + injected
            false_dissipation_values.append(min(1.0, injected * (1.0 - capacity)))
        else:
            prev_reservoir = prev_reservoir * RESERVOIR_DECAY
            false_dissipation_values.append(0.0)
        denominator = max(prev_lrp + prev_reservoir, 1e-6)
        relocation_values.append(min(1.0, prev_reservoir / denominator))
        reservoir_values.append(prev_reservoir)
        prev_lrp = current_lrp

    reservoir = pd.Series(reservoir_values, index=df.index, dtype=float)
    ceiling = reservoir.rolling(20, min_periods=1).max().replace(0, np.nan)
    reservoir_norm = (reservoir / ceiling).clip(0.0, 1.0)
    reservoir_norm = reservoir_norm.where(ceiling.notna(), reservoir.clip(0.0, 1.0)).fillna(0.0)
    relocation = pd.Series(relocation_values, index=df.index, dtype=float).clip(0.0, 1.0)
    false_dissipation = pd.Series(false_dissipation_values, index=df.index, dtype=float).clip(0.0, 1.0)
    return reservoir_norm, relocation, false_dissipation


def classify_gamma_rebound_row(row: pd.Series, *, gamma_regime: str = "") -> str:
    strain = float(row.get("elastic_strain_score", 0.0) or 0.0)
    strain_delta = float(row.get("d_elastic_strain", 0.0) or 0.0)
    wall_pin = float(row.get("wall_pinning_strength", 0.0) or 0.0)
    reservoir = float(row.get("hidden_reservoir_pressure", 0.0) or 0.0)
    c_w = float(row.get("C_w", 0.0) or 0.0)
    a_micro = float(row.get("A_micro", 0.0) or 0.0)
    d_lrp = float(row.get("d_lrp_session", 0.0) or 0.0)
    d_gamma = float(row.get("d_gamma_flip_distance", 0.0) or 0.0)
    gamma_dist = float(row.get("gamma_flip_distance_pct", 999.0) or 999.0)
    positive_gamma = "POSITIVE_GAMMA" in str(gamma_regime).upper()

    if strain >= 0.75 and (abs(gamma_dist) < 1.5 or abs(d_gamma) > 0.2):
        return "PRE_RUPTURE"
    if d_lrp > 0.08 and (abs(d_gamma) > 0.15 or strain >= 0.6):
        return "REBOUND_RELEASE"
    if c_w >= 0.65 and a_micro >= 0.5 and strain_delta < 0:
        return "AFTERSHOCK_REPRICING"
    if strain < 0.35 and reservoir < 0.25 and strain_delta <= 0:
        return "NEW_EQUILIBRIUM"
    if (wall_pin >= 0.55 or positive_gamma) and strain >= 0.45 and strain_delta > 0.02:
        return "COMPRESSING_STRAIN"
    if (wall_pin >= 0.55 or positive_gamma) and strain < 0.45:
        return "LOCKED_FAULT"
    if strain >= 0.55:
        return "PRE_RUPTURE"
    return ""


def compute_elastic_rebound(
    df: pd.DataFrame,
    *,
    gamma: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Compute elastic strain, hidden reservoir, false stability, gamma rebound regime."""
    result = df.copy()
    gamma = gamma or {}
    gamma_regime = str(gamma.get("gamma_regime", "") or "")

    gamma_prox = _gamma_boundary_proximity(result)
    vol_comp = _volatility_compression(result)
    flow_div = _hidden_flow_divergence(result)
    compression_dur = _compression_duration_proxy(result)

    raw_strain = compression_dur * flow_div * gamma_prox * (0.55 + 0.45 * vol_comp)
    result["elastic_strain_score"] = raw_strain.clip(0.0, 1.0)
    result["d_elastic_strain"] = result["elastic_strain_score"].diff().fillna(0.0)

    reservoir, relocation, false_dissipation = _compute_hidden_reservoir(result)
    result["hidden_reservoir_pressure"] = reservoir
    result["pressure_relocation_ratio"] = relocation
    result["false_dissipation_risk"] = false_dissipation

    lrp = _as_series(result, "LRP")
    result["d_lrp_session"] = lrp.diff().fillna(0.0)
    r_o = _as_series(result, "R_o", default=0.5)
    d_r_o = r_o.diff().fillna(0.0)

    observability_gap = (
        0.35 * (1.0 - r_o)
        + 0.30 * result["hidden_reservoir_pressure"]
        + 0.20 * result["elastic_strain_score"]
        + 0.15 * (-d_r_o).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    result["observability_gap_score"] = observability_gap

    lrp_falling = result["d_lrp_session"] < -0.01
    strain_rising = result["d_elastic_strain"] > 0.01
    reservoir_rising = result["hidden_reservoir_pressure"].diff().fillna(0.0) > 0.01
    r_o_falling = d_r_o < -0.01
    result["false_stability_flag"] = (
        lrp_falling & (strain_rising | reservoir_rising | r_o_falling)
    ).astype(int)

    result["gamma_rebound_regime"] = result.apply(
        lambda row: classify_gamma_rebound_row(row, gamma_regime=gamma_regime),
        axis=1,
    )
    return result
