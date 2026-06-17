"""Convert options market data into pressure metrics — map inward, not sacred ontology."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from chain_integrity_engine import data_quality_confidence
from volatility_skew_engine import SPOT_RANGE_PCT, compute_skew_metrics, filter_strikes_near_spot

REALIZED_VOL_WINDOW = 30


def _clip01(value: Optional[float]) -> float:
    if value is None or (isinstance(value, float) and (np.isnan(value) or not np.isfinite(value))):
        return 0.0
    return float(np.clip(float(value), 0.0, 1.0))


def compute_realized_volatility(stock_df: pd.DataFrame, *, window: int = REALIZED_VOL_WINDOW) -> float:
    """Annualized realized vol from close returns (percent)."""
    if stock_df is None or stock_df.empty or "Close" not in stock_df.columns:
        return float("nan")
    closes = pd.to_numeric(stock_df["Close"], errors="coerce").dropna()
    if len(closes) < max(window, 5):
        return float("nan")
    returns = closes.pct_change().dropna()
    recent = returns.tail(window)
    if recent.empty:
        return float("nan")
    daily_std = float(recent.std())
    return daily_std * np.sqrt(252.0) * 100.0


def compute_gamma_compression_score(
    option_df: pd.DataFrame,
    *,
    spot: float,
    expiration: Optional[str] = None,
    spot_range_pct: float = SPOT_RANGE_PCT,
) -> float:
    """
    G_c = sum(|Gamma_i|) weighted by inverse distance from spot.
    High concentration near spot => price locking / compressed pressure field.
    """
    if option_df.empty:
        return 0.0
    df = option_df.copy()
    if expiration:
        df = df[df["Expiration Date"] == expiration]
    df = filter_strikes_near_spot(df, spot, range_pct=spot_range_pct)
    if df.empty or "Gamma" not in df.columns:
        return 0.0

    gamma = pd.to_numeric(df["Gamma"], errors="coerce").fillna(0.0).abs()
    dist = (pd.to_numeric(df["Strike"], errors="coerce") - spot).abs() / max(spot, 1e-6)
    weight = 1.0 / (1.0 + dist * 20.0)
    raw = float((gamma * weight).sum())
    return _clip01(np.log1p(raw) / 8.0)


def compute_volatility_expansion_score(
    atm_iv_pct: Optional[float],
    realized_vol_pct: float,
) -> Optional[float]:
    """V_e = ATM IV / realized vol. >2 suggests market anticipating rupture."""
    if atm_iv_pct is None or np.isnan(realized_vol_pct) or realized_vol_pct <= 0:
        return None
    return round(float(atm_iv_pct) / float(realized_vol_pct), 4)


def compute_skew_asymmetry_pressure(
    *,
    atm_iv: Optional[float],
    put_wing_iv: Optional[float],
    call_wing_iv: Optional[float],
) -> Optional[float]:
    """S_p = (PutWing - CallWing) / ATM IV. Positive = fear; negative = upside instability."""
    if atm_iv is None or atm_iv <= 0:
        return None
    put_w = put_wing_iv if put_wing_iv is not None else atm_iv
    call_w = call_wing_iv if call_wing_iv is not None else atm_iv
    return round(float((put_w - call_w) / atm_iv), 4)


def compute_dealer_hedging_stress(
    option_df: pd.DataFrame,
    *,
    spot: float,
    expiration: Optional[str] = None,
    spot_range_pct: float = SPOT_RANGE_PCT,
) -> float:
    """D_h ~ sum(|Gamma_i| * OI_i) — dealers forced to dynamically hedge."""
    if option_df.empty:
        return 0.0
    df = option_df.copy()
    if expiration:
        df = df[df["Expiration Date"] == expiration]
    df = filter_strikes_near_spot(df, spot, range_pct=spot_range_pct)
    if df.empty:
        return 0.0
    gamma = pd.to_numeric(df.get("Gamma", 0), errors="coerce").fillna(0.0).abs()
    oi = pd.to_numeric(df.get("Open Interest", 0), errors="coerce").fillna(0.0)
    raw = float((gamma * oi).sum())
    if raw <= 0:
        return 0.0
    return _clip01(np.log1p(raw) / 12.0)


def compute_options_pressure_metrics(
    *,
    option_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    spot: float,
    skew_metrics: Optional[dict[str, Any]] = None,
    expiration: Optional[str] = None,
    chain_integrity: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Full options → pressure metric bundle."""
    skew = skew_metrics or compute_skew_metrics(option_df, spot=spot, expiration=expiration)
    confidence = data_quality_confidence(chain_integrity) if chain_integrity else 1.0
    atm_iv = skew.get("atm_iv")
    realized_vol = compute_realized_volatility(stock_df)

    gamma_compression = compute_gamma_compression_score(
        option_df, spot=spot, expiration=skew.get("expiration"), spot_range_pct=SPOT_RANGE_PCT
    )
    vol_expansion = compute_volatility_expansion_score(atm_iv, realized_vol)
    skew_pressure = compute_skew_asymmetry_pressure(
        atm_iv=atm_iv,
        put_wing_iv=skew.get("put_wing_iv"),
        call_wing_iv=skew.get("call_wing_iv"),
    )
    dealer_stress = compute_dealer_hedging_stress(
        option_df, spot=spot, expiration=skew.get("expiration"), spot_range_pct=SPOT_RANGE_PCT
    )

    return {
        "gamma_compression_score": round(gamma_compression * confidence, 4),
        "volatility_expansion_score": vol_expansion,
        "skew_asymmetry_pressure": skew_pressure,
        "dealer_hedging_stress_score": round(dealer_stress * confidence, 4),
        "realized_vol_30d_pct": round(realized_vol, 4) if not np.isnan(realized_vol) else None,
        "atm_iv_pct": atm_iv,
        "data_quality_confidence": round(confidence, 4),
    }


def map_to_pressure_vocabulary(pressure_metrics: dict[str, Any], *, skew_metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Map Laser Falcon options metrics inward to existing pressure-field vocabulary.
    Does NOT modify sacred ontology — local bridge only.
    """
    vol_exp = pressure_metrics.get("volatility_expansion_score") or 1.0
    gamma_c = pressure_metrics.get("gamma_compression_score") or 0.0
    dealer = pressure_metrics.get("dealer_hedging_stress_score") or 0.0
    skew_p = pressure_metrics.get("skew_asymmetry_pressure") or 0.0
    atm = pressure_metrics.get("atm_iv_pct") or 0.0

    energy_injection_proxy = _clip01(min(vol_exp / 4.0, 1.0))
    boundary_stress_proxy = _clip01(gamma_c)
    rupture_contributor = _clip01(dealer * 0.5 + (abs(skew_p) if skew_p else 0.0) * 0.3)
    lrp_contributor = _clip01(
        (atm / 100.0 if atm and atm > 1 else atm or 0.0) * 0.4
        + (abs(skew_p) if skew_p else 0.0) * 0.3
        + dealer * 0.3
    )
    observer_blindspot_proxy = _clip01(
        (vol_exp - 1.0) / 3.0 if vol_exp and vol_exp > 1 else 0.0
    )

    return {
        "energy_injection_proxy": round(energy_injection_proxy, 4),
        "boundary_stress_proxy": round(boundary_stress_proxy, 4),
        "rupture_pressure_contributor": round(rupture_contributor, 4),
        "lrp_contributor": round(lrp_contributor, 4),
        "observer_blindspot_proxy": round(observer_blindspot_proxy, 4),
        "maps_to": {
            "volatility_expansion_score": "E_i (energy_injection_rate proxy)",
            "gamma_compression_score": "B_s (boundary_stress_frequency proxy)",
            "dealer_hedging_stress_score": "rupture_pressure_score contributor",
            "skew_asymmetry_pressure": "LRP contributor",
            "surface_dislocation": "observer blindspot contributor",
        },
    }
