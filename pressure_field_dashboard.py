"""Pressure Field Dashboard — multi-sensor market state HTML dashboard.

Loads historical stock and option chain data via data_loader, computes
pressure-field metrics (MACD, RSI, CVD, Volume, VWAP, gamma flip), merges
CanopyEnto boundary/observer variables, and renders a self-contained HTML
dashboard rich in color and semantic interpretation.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from canopyento_boundary_engine import (
    compute_boundary_metrics,
    compute_observer_differential_metrics,
    compute_weekly_stance_metrics,
    load_stock_data_fallback,
)
from data_loader import (
    DEFAULT_OPTION_DIR,
    DEFAULT_STOCK_DIR,
    get_most_recent_option_date,
    load_option_chain_data,
    parse_price,
)
from pressure_field_derivatives import (
    DERIVATIVE_COLUMNS,
    LRP_ADJUSTED_COLUMNS,
    LRP_CONTRIB_COLUMNS,
    apply_lrp_loop_closure,
    build_lrp_debug_payload,
    compute_rate_of_change_alerts,
    enrich_pressure_derivatives,
    write_lrp_debug_json,
)
from pressure_field_physics import PHYSICS_EXPORT_COLUMNS, enrich_pressure_physics
from pressure_field_schema import LRP_DOCTRINE, build_stable_snapshot, write_stable_snapshot_json

LRP_ADJUSTED_EXPERIMENTAL_LABEL = "LRP Adjusted (experimental)"

ELASTIC_REBOUND_NOTES = (
    "Positive gamma may behave like a locked fault: surface volatility compresses "
    "while hidden positioning strain accumulates.",
    "Reduced visible pressure is not always true dissipation; some pressure may be "
    "relocating into a hidden reservoir.",
    "False stability is flagged when observable pressure falls while hidden strain "
    "or reservoir pressure rises.",
)

DEFAULT_TICKER = "SPY"
DEFAULT_LOOKBACK_DAYS = 120
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_RSI_PERIOD = 14
DEFAULT_VWAP_WINDOW = 20
DEFAULT_OUTPUT_HTML = "outputs/pressure_field_dashboard_{ticker}.html"
DEFAULT_OUTPUT_JSON = "outputs/pressure_field_latest_{ticker}.json"
DEFAULT_OUTPUT_CSV = "outputs/pressure_field_{ticker}.csv"
DEFAULT_OUTPUT_MD = "outputs/pressure_field_{ticker}.md"
DEFAULT_LRP_DEBUG_JSON = "outputs/lrp_debug_{ticker}.json"

PRESSURE_FIELD_EXPORT_COLUMNS = [
    "Date",
    "Close",
    "macd_regime",
    "rsi_saturation",
    "cvd_regime",
    "volume_injection",
    "vwap_distance_pct",
    "gamma_flip_distance_pct",
    "regime_label",
    "rupture_pressure_score",
    "T_a",
    "T_a_norm",
    "T_a_regime",
    "R_o",
    "T_v",
    "observer_profile",
    "LRP",
    "LRP_raw",
    "LRP_regime",
    "LRP_confidence",
    *LRP_CONTRIB_COLUMNS,
    *LRP_ADJUSTED_COLUMNS,
    *DERIVATIVE_COLUMNS,
    *PHYSICS_EXPORT_COLUMNS,
]

OBSERVER_QUOTE = (
    "Complex systems do not fail equally for all observers. Rupture becomes visible "
    "at different times depending on an observer's proximity to the system's internal "
    "pressure variables. The system changes before most observers are capable of "
    "perceiving the change."
)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _sanitize_series(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan)


def _finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or pd.isna(value):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def prepare_ohlcv(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV columns from a stock CSV."""
    df = stock_df.copy()
    close_col = "Close/Last" if "Close/Last" in df.columns else "Close"
    if close_col != "Close":
        df["Close"] = df[close_col]
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col].map(parse_price)
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)


def compute_macd(
    df: pd.DataFrame,
    *,
    fast: int = DEFAULT_MACD_FAST,
    slow: int = DEFAULT_MACD_SLOW,
    signal: int = DEFAULT_MACD_SIGNAL,
) -> pd.DataFrame:
    """MACD as pressure acceleration detector."""
    result = df.copy()
    close = result["Close"].astype(float)
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    result["macd_line"] = ema_fast - ema_slow
    result["macd_signal"] = _ema(result["macd_line"], signal)
    result["macd_histogram"] = result["macd_line"] - result["macd_signal"]
    hist_std = result["macd_histogram"].rolling(20, min_periods=5).std()
    result["macd_accel"] = result["macd_histogram"].diff()
    result["macd_pressure_accel"] = _sanitize_series(
        result["macd_accel"] / hist_std.where(hist_std > 1e-12)
    )
    result["macd_regime"] = result["macd_histogram"].apply(_macd_regime_label)
    return result


def _macd_regime_label(hist: float) -> str:
    if pd.isna(hist):
        return ""
    if hist > 0.5:
        return "ACCEL_UP"
    if hist > 0:
        return "PRESSURE_BUILDING"
    if hist > -0.5:
        return "PRESSURE_FADING"
    return "DECEL_DOWN"


def compute_rsi(df: pd.DataFrame, *, period: int = DEFAULT_RSI_PERIOD) -> pd.DataFrame:
    """RSI as local energy saturation detector."""
    result = df.copy()
    delta = result["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    result["rsi_saturation"] = result["rsi"].apply(_rsi_saturation_label)
    result["rsi_energy"] = (result["rsi"] - 50.0).abs() / 50.0
    return result


def _rsi_saturation_label(rsi: float) -> str:
    if pd.isna(rsi):
        return ""
    if rsi >= 70:
        return "OVERBOUGHT_SATURATION"
    if rsi <= 30:
        return "OVERSOLD_DEPLETION"
    if rsi >= 60:
        return "ENERGY_ACCUMULATING"
    if rsi <= 40:
        return "ENERGY_RELEASING"
    return "NEUTRAL_BAND"


def compute_cvd(df: pd.DataFrame) -> pd.DataFrame:
    """CVD proxy as internal force imbalance detector (no tick data)."""
    result = df.copy()
    direction = np.sign(result["Close"].diff()).fillna(0.0)
    result["signed_volume"] = direction * result["Volume"]
    result["cvd"] = result["signed_volume"].cumsum()
    result["cvd_slope"] = result["cvd"].diff(5)
    vol_scale = result["Volume"].rolling(20, min_periods=5).mean() * 5.0
    result["cvd_imbalance"] = _sanitize_series(
        result["cvd_slope"] / vol_scale.where(vol_scale > 1e-12)
    )
    result["cvd_regime"] = result["cvd_imbalance"].apply(_cvd_regime_label)
    return result


def _cvd_regime_label(imbalance: float) -> str:
    if pd.isna(imbalance):
        return ""
    if imbalance > 0.35:
        return "BUY_FORCE_DOMINANT"
    if imbalance < -0.35:
        return "SELL_FORCE_DOMINANT"
    if abs(imbalance) < 0.10:
        return "FORCE_BALANCED"
    return "IMBALANCE_BUILDING"


def compute_volume_field(df: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    """Volume as external energy injection field."""
    result = df.copy()
    avg_vol = result["Volume"].rolling(window, min_periods=max(1, window // 2)).mean()
    result["volume_injection"] = _sanitize_series(
        result["Volume"] / avg_vol.where(avg_vol > 1e-12)
    )
    result["volume_percentile"] = result["Volume"].rolling(60, min_periods=20).apply(
        lambda values: float((values[:-1] <= values[-1]).mean()) if len(values) > 1 else 0.5,
        raw=True,
    )
    result["volume_regime"] = result["volume_injection"].apply(_volume_regime_label)
    return result


def _volume_regime_label(ratio: float) -> str:
    if pd.isna(ratio):
        return ""
    if ratio >= 1.5:
        return "HIGH_INJECTION"
    if ratio >= 1.1:
        return "ELEVATED_FLOW"
    if ratio <= 0.7:
        return "LOW_INJECTION"
    return "NORMAL_FLOW"


def compute_vwap_field(df: pd.DataFrame, *, window: int = DEFAULT_VWAP_WINDOW) -> pd.DataFrame:
    """VWAP as equilibrium attractor field."""
    result = df.copy()
    typical = (result["High"] + result["Low"] + result["Close"]) / 3.0
    pv = typical * result["Volume"]
    cum_pv = pv.rolling(window, min_periods=max(1, window // 2)).sum()
    cum_vol = result["Volume"].rolling(window, min_periods=max(1, window // 2)).sum()
    result["vwap"] = _sanitize_series(cum_pv / cum_vol.where(cum_vol > 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        result["vwap_distance_pct"] = _sanitize_series(
            (result["Close"] - result["vwap"]) / result["vwap"] * 100.0
        )
    result["vwap_pull"] = result["vwap_distance_pct"].abs()
    result["vwap_regime"] = result["vwap_distance_pct"].apply(_vwap_regime_label)
    return result


def _vwap_regime_label(distance_pct: float) -> str:
    if pd.isna(distance_pct):
        return ""
    if distance_pct > 1.0:
        return "ABOVE_ATTRACTOR"
    if distance_pct < -1.0:
        return "BELOW_ATTRACTOR"
    if abs(distance_pct) < 0.25:
        return "AT_EQUILIBRIUM"
    return "MEAN_REVERT_PULL"


def empty_gamma_snapshot() -> dict[str, Any]:
    """Safe null gamma snapshot when option chain is unavailable."""
    return {
        "gamma_flip_strike": None,
        "distance_to_flip_pct": None,
        "net_gamma_at_spot": None,
        "gamma_regime": "NO_CHAIN",
        "call_gamma_oi": 0.0,
        "put_gamma_oi": 0.0,
    }


def compute_gamma_flip(
    option_df: Optional[pd.DataFrame],
    spot: float,
) -> dict[str, Any]:
    """Gamma flip level from option chain — phase boundary transition."""
    if option_df is None or len(option_df) == 0:
        return empty_gamma_snapshot()
    if spot <= 0 or not np.isfinite(spot):
        snapshot = empty_gamma_snapshot()
        snapshot["gamma_regime"] = "INVALID_SPOT"
        return snapshot

    df = option_df.copy()
    if "Strike" not in df.columns:
        return empty_gamma_snapshot()

    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
    df = df[df["Strike"].notna()].copy()
    if df.empty:
        return empty_gamma_snapshot()

    for col in ("Gamma", "Gamma.1", "Open Interest", "Open Interest.1"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].map(parse_price), errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    df["call_gamma_oi"] = df["Gamma"] * df["Open Interest"]
    df["put_gamma_oi"] = df["Gamma.1"] * df["Open Interest.1"]
    grouped = (
        df.groupby("Strike", as_index=False)
        .agg(
            call_gamma_oi=("call_gamma_oi", "sum"),
            put_gamma_oi=("put_gamma_oi", "sum"),
        )
        .sort_values("Strike")
        .reset_index(drop=True)
    )
    if grouped.empty:
        return empty_gamma_snapshot()

    total_oi = float(grouped["call_gamma_oi"].sum() + grouped["put_gamma_oi"].sum())
    if total_oi <= 0:
        snapshot = empty_gamma_snapshot()
        snapshot["gamma_regime"] = "ZERO_OI"
        return snapshot

    grouped["net_gamma"] = grouped["call_gamma_oi"] - grouped["put_gamma_oi"]

    flip_strike: Optional[float] = None
    strikes = grouped["Strike"].values
    net = grouped["net_gamma"].values
    for idx in range(1, len(strikes)):
        prev_net = float(net[idx - 1])
        curr_net = float(net[idx])
        if prev_net == 0.0:
            flip_strike = float(strikes[idx - 1])
            break
        if prev_net * curr_net < 0:
            weight = abs(prev_net) / (abs(prev_net) + abs(curr_net) + 1e-9)
            flip_strike = float(strikes[idx - 1] * (1 - weight) + strikes[idx] * weight)
            break

    spot_idx = (grouped["Strike"] - spot).abs().idxmin()
    net_at_spot = float(grouped.loc[spot_idx, "net_gamma"])
    distance_pct: Optional[float] = None
    if flip_strike is not None:
        distance_pct = (spot - flip_strike) / spot * 100.0

    if flip_strike is None:
        regime = "FLIP_UNDEFINED"
    elif distance_pct is not None and abs(distance_pct) < 0.5:
        regime = "AT_PHASE_BOUNDARY"
    elif distance_pct is not None and distance_pct > 0:
        regime = "ABOVE_FLIP_POSITIVE_GAMMA"
    elif distance_pct is not None:
        regime = "BELOW_FLIP_NEGATIVE_GAMMA"
    else:
        regime = "FLIP_UNDEFINED"

    return {
        "gamma_flip_strike": flip_strike,
        "distance_to_flip_pct": distance_pct,
        "net_gamma_at_spot": net_at_spot,
        "gamma_regime": regime,
        "call_gamma_oi": float(grouped["call_gamma_oi"].sum()),
        "put_gamma_oi": float(grouped["put_gamma_oi"].sum()),
    }


def build_pressure_frame(
    stock_df: pd.DataFrame,
    *,
    lookback: int = 20,
    tolerance: float = 0.003,
    volume_window: int = 20,
    weekly_window: int = 5,
) -> pd.DataFrame:
    """Full pressure field + CanopyEnto merge on OHLCV history."""
    ohlcv = prepare_ohlcv(stock_df)
    frame = compute_macd(ohlcv)
    frame = compute_rsi(frame)
    frame = compute_cvd(frame)
    frame = compute_volume_field(frame, window=volume_window)
    frame = compute_vwap_field(frame)

    canopy = compute_boundary_metrics(
        frame,
        lookback=lookback,
        tolerance=tolerance,
        volume_window=volume_window,
    )
    canopy = compute_weekly_stance_metrics(canopy, weekly_window=weekly_window)
    canopy = compute_observer_differential_metrics(canopy)
    return canopy


def _series_tail(series: pd.Series, n: int = 60) -> list[Optional[float]]:
    tail = series.tail(n)
    return [None if pd.isna(v) else round(float(v), 6) for v in tail]


def _date_labels(dates: pd.Series, n: int = 60) -> list[str]:
    tail = dates.tail(n)
    return [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in tail]


def _status_class(regime: str) -> str:
    positive = {"TAKEOFF_BEGINNING", "ACCEL_UP", "PRESSURE_BUILDING", "BUY_FORCE_DOMINANT",
                "HIGH_INJECTION", "ABOVE_FLIP_POSITIVE_GAMMA", "ACTIONABLE DIRECTIONAL STANCE"}
    warning = {"THRUST_LOSS", "DECEL_DOWN", "IMBALANCE_BUILDING", "AT_PHASE_BOUNDARY",
               "RUPTURE_CANDIDATE", "CONTAINMENT_STRESS", "LOW-CONFIDENCE CRUISE MODE"}
    danger = {"DISSIPATION_CASCADE", "SELL_FORCE_DOMINANT", "BELOW_FLIP_NEGATIVE_GAMMA",
              "WAIT / PACKET BUFFERING", "RUPTURE_IMMINENT", "PRE_RUPTURE",
              "ENTROPIC_DEGRADATION", "CAPILLARY_PRE_RUPTURE"}
    if regime in danger:
        return "status-danger"
    if regime in warning:
        return "status-warn"
    if regime in positive:
        return "status-good"
    if regime in {"RESTORED_EQUILIBRIUM", "ACTIVE_COMPENSATION"}:
        return "status-good"
    if regime == "WEAKENING_RESTORATION":
        return "status-warn"
    return "status-neutral"


def _lrp_status_class(regime: str) -> str:
    if regime in {"RUPTURE_IMMINENT", "PRE_RUPTURE"}:
        return "status-danger" if regime == "RUPTURE_IMMINENT" else "status-warn"
    if regime == "PRESSURE_BUILDING":
        return "status-warn"
    return "status-good"


def _format_lrp_contributions(latest: pd.Series) -> str:
    labels = (
        ("lrp_contrib_T_a", "T_a"),
        ("lrp_contrib_observer", "Observer decay"),
        ("lrp_contrib_gamma", "Gamma boundary"),
        ("lrp_contrib_vwap", "VWAP dislocation"),
        ("lrp_contrib_cvd", "CVD imbalance"),
        ("lrp_contrib_macd", "MACD acceleration"),
    )
    parts = []
    for key, label in labels:
        value = latest.get(key)
        if value is not None and pd.notna(value):
            parts.append(f"{label}: {float(value):.2f}")
    confidence = latest.get("LRP_confidence", "")
    raw = latest.get("LRP_raw")
    raw_text = f" · raw {float(raw):.3f}" if raw is not None and pd.notna(raw) else ""
    conf_text = f" · {confidence}" if confidence else ""
    if parts:
        return " · ".join(parts) + raw_text + conf_text
    return f"LRP confidence: {confidence or 'n/a'}{raw_text}"


def render_html_dashboard(
    df: pd.DataFrame,
    *,
    ticker: str,
    spot: float,
    chain_date: str,
    gamma: dict[str, Any],
    alerts: list[str],
    output_path: Path,
    chart_days: int = DEFAULT_LOOKBACK_DAYS,
) -> None:
    """Render self-contained HTML dashboard with embedded charts."""
    latest = df.iloc[-1]
    dates = _date_labels(df["Date"], chart_days)

    chart_payload = {
        "dates": dates,
        "close": _series_tail(df["Close"], chart_days),
        "macd_hist": _series_tail(df["macd_histogram"], chart_days),
        "rsi": _series_tail(df["rsi"], chart_days),
        "cvd": _series_tail(df["cvd"], chart_days),
        "volume": _series_tail(df["Volume"], chart_days),
        "vwap": _series_tail(df["vwap"], chart_days),
        "rupture": _series_tail(df["rupture_pressure_score"], chart_days),
        "T_a_norm": _series_tail(df["T_a_norm"], chart_days),
        "LRP": _series_tail(df["LRP"], chart_days),
        "LRP_adjusted": _series_tail(df.get("LRP_adjusted", pd.Series(dtype=float)), chart_days),
        "F_r": _series_tail(df["F_r"], chart_days),
        "D_c": _series_tail(df["D_c"], chart_days),
        "C_w": _series_tail(df["C_w"], chart_days),
    }

    latest_date = latest["Date"]
    latest_date_str = latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date)

    metrics = [
        {
            "id": "macd",
            "title": "MACD",
            "role": "Pressure Acceleration Detector",
            "value": f"{latest['macd_histogram']:.4f}",
            "sub": latest.get("macd_regime", ""),
            "detail": f"Line {latest['macd_line']:.4f} · Signal {latest['macd_signal']:.4f}",
            "color": "#00d4ff",
            "status_class": _status_class(str(latest.get("macd_regime", ""))),
        },
        {
            "id": "rsi",
            "title": "RSI",
            "role": "Local Energy Saturation Detector",
            "value": f"{latest['rsi']:.1f}",
            "sub": latest.get("rsi_saturation", ""),
            "detail": f"Energy load {latest['rsi_energy']:.2f}",
            "color": "#ff3cac",
            "status_class": _status_class(str(latest.get("rsi_saturation", ""))),
        },
        {
            "id": "cvd",
            "title": "CVD",
            "role": "Internal Force Imbalance Detector",
            "value": f"{latest['cvd_imbalance']:.3f}",
            "sub": latest.get("cvd_regime", ""),
            "detail": f"Cumulative delta slope (5d)",
            "color": "#ffb347",
            "status_class": _status_class(str(latest.get("cvd_regime", ""))),
        },
        {
            "id": "volume",
            "title": "Volume",
            "role": "External Energy Injection",
            "value": f"{latest['volume_injection']:.2f}x",
            "sub": latest.get("volume_regime", ""),
            "detail": f"E_i {latest['E_i']:.3f} · pct {latest['volume_percentile']:.0%}",
            "color": "#39ff14",
            "status_class": _status_class(str(latest.get("volume_regime", ""))),
        },
        {
            "id": "vwap",
            "title": "VWAP",
            "role": "Equilibrium Attractor Field",
            "value": f"${latest['vwap']:.2f}",
            "sub": latest.get("vwap_regime", ""),
            "detail": f"Distance {latest['vwap_distance_pct']:+.2f}% · pull {latest['vwap_pull']:.2f}",
            "color": "#9d4edd",
            "status_class": _status_class(str(latest.get("vwap_regime", ""))),
        },
        {
            "id": "gamma",
            "title": "Gamma Flip",
            "role": "Phase Boundary Transition",
            "value": (
                f"${gamma['gamma_flip_strike']:.2f}"
                if gamma.get("gamma_flip_strike") is not None
                else "N/A"
            ),
            "sub": gamma["gamma_regime"],
            "detail": (
                f"Spot dist {gamma['distance_to_flip_pct']:+.2f}% · "
                f"net gamma {gamma['net_gamma_at_spot']:.0f}"
                if gamma.get("distance_to_flip_pct") is not None
                and gamma.get("net_gamma_at_spot") is not None
                else gamma["gamma_regime"]
            ),
            "color": "#ffd700",
            "status_class": _status_class(str(gamma.get("gamma_regime", ""))),
        },
        {
            "id": "lrp",
            "title": "Latent Rupture Potential",
            "role": "Pressure-Driven Rupture Risk (baseline)",
            "value": f"{latest.get('LRP', float('nan')):.3f}",
            "sub": f"{latest.get('LRP_regime', '')} · {latest.get('LRP_confidence', '')}",
            "detail": _format_lrp_contributions(latest),
            "color": "#ff4d6d",
            "status_class": _lrp_status_class(str(latest.get("LRP_regime", ""))),
        },
        {
            "id": "lrp_adj",
            "title": LRP_ADJUSTED_EXPERIMENTAL_LABEL,
            "role": "Restoration-adjusted rupture risk — not canonical",
            "value": f"{latest.get('LRP_adjusted', float('nan')):.3f}",
            "sub": f"{latest.get('LRP_adjusted_regime', '') or '—'} · experimental",
            "detail": (
                f"delta vs baseline {float(latest.get('LRP_adjusted', 0)) - float(latest.get('LRP', 0)):+.3f} · "
                f"damper {latest.get('restoration_damper', float('nan')):.2f}"
            ),
            "color": "#c9184a",
            "status_class": _lrp_status_class(str(latest.get("LRP_adjusted_regime", ""))),
        },
        {
            "id": "restoration",
            "title": "Restoration Field",
            "role": "F_r · Restoring Equilibrium Pull",
            "value": f"{latest.get('F_r', float('nan')):.3f}",
            "sub": str(latest.get("field_regime", "") or "—"),
            "detail": (
                f"Ratio {latest.get('restoration_ratio', float('nan')):.2f} · "
                f"D_c {latest.get('D_c', float('nan')):.3f}"
            ),
            "color": "#4cc9f0",
            "status_class": _status_class(str(latest.get("field_regime", ""))),
        },
        {
            "id": "capillary",
            "title": "Capillary Wave",
            "role": "C_w · Micro-Oscillation Score",
            "value": f"{latest.get('C_w', float('nan')):.3f}",
            "sub": f"A_micro {latest.get('A_micro', float('nan')):.3f} · persist {latest.get('wave_persistence', float('nan')):.3f}",
            "detail": f"Dissipation {latest.get('dissipation_score', float('nan')):.3f}",
            "color": "#80ed99",
            "status_class": _status_class(str(latest.get("field_regime", ""))),
        },
    ]

    canopy_cards = [
        ("B_s", "Boundary Stability", f"{latest['B_s']:.4f}", "Containment stress from boundary tests"),
        ("E_i", "Injection Energy", f"{latest['E_i']:.4f}", "Volume vs rolling average"),
        ("T_a", "Transitional Acceleration", f"{latest.get('T_a_norm', float('nan')):.4f}", str(latest.get("T_a_regime", ""))),
        ("R_o", "Observational Resolution", f"{latest.get('R_o', float('nan')):.3f}", str(latest.get("observer_profile", ""))),
        ("T_v", "Visibility Horizon", f"{latest.get('T_v', float('nan'))} sessions", "T_v = f(R_o)"),
        ("rupture", "Rupture Pressure", f"{latest['rupture_pressure_score']:.4f}", str(latest.get("regime_label", ""))),
        ("LRP", "Latent Rupture Potential (baseline)", f"{latest.get('LRP', float('nan')):.3f}", str(latest.get("LRP_regime", ""))),
        ("LRP_adj", LRP_ADJUSTED_EXPERIMENTAL_LABEL, f"{latest.get('LRP_adjusted', float('nan')):.3f}", str(latest.get("LRP_adjusted_regime", ""))),
        ("F_r", "Restoring Field", f"{latest.get('F_r', float('nan')):.3f}", str(latest.get("field_regime", ""))),
        ("D_c", "Dissipation Capacity", f"{latest.get('D_c', float('nan')):.3f}", f"score {latest.get('dissipation_score', float('nan')):.3f}"),
        ("C_w", "Capillary Wave", f"{latest.get('C_w', float('nan')):.3f}", f"ratio {latest.get('restoration_ratio', float('nan')):.2f}"),
        ("strain", "Elastic Strain", f"{latest.get('elastic_strain_score', float('nan')):.3f}", str(latest.get("gamma_rebound_regime", ""))),
        ("reservoir", "Hidden Reservoir", f"{latest.get('hidden_reservoir_pressure', float('nan')):.3f}", f"reloc {latest.get('pressure_relocation_ratio', float('nan')):.2f}"),
    ]

    alerts_html = ""
    if alerts:
        chips = "".join(f'<span class="alert-chip">{alert}</span>' for alert in alerts)
        alerts_html = f"""
        <section class="chart-panel">
          <h2>Rate-of-Change Alerts</h2>
          <div class="alert-row">{chips}</div>
        </section>
        """

    metric_cards_html = "\n".join(
        f"""
        <article class="metric-card {m['status_class']}" style="--accent:{m['color']}">
          <header>
            <span class="metric-tag">{m['role']}</span>
            <h3>{m['title']}</h3>
          </header>
          <div class="metric-value">{m['value']}</div>
          <div class="metric-regime">{m['sub']}</div>
          <p class="metric-detail">{m['detail']}</p>
          <canvas id="chart-{m['id']}" height="80"></canvas>
        </article>
        """
        for m in metrics
    )

    canopy_html = "\n".join(
        f"""
        <div class="canopy-chip">
          <span class="chip-label">{label}</span>
          <span class="chip-value">{value}</span>
          <span class="chip-sub">{sub}</span>
        </div>
        """
        for _, label, value, sub in canopy_cards
    )

    stance_block = ""
    if pd.notna(latest.get("stance_quadrant")):
        stance_block = f"""
        <section class="stance-panel">
          <h2>CanopyEnto Weekly Stance</h2>
          <div class="stance-grid">
            <div><span>Quadrant</span><strong>{latest['stance_quadrant']}</strong></div>
            <div><span>Gate</span><strong>{latest['gate_stance']}</strong></div>
            <div><span>Confidence</span><strong>{latest['stance_confidence']:+.2f}</strong></div>
            <div><span>Direction</span><strong>{latest['direction_bias']}</strong></div>
          </div>
          <p class="stance-action">{latest['recommended_action']}</p>
        </section>
        """

    lrp_contrib_rows = [
        ("T_a contribution", "lrp_contrib_T_a"),
        ("Observer decay", "lrp_contrib_observer"),
        ("Gamma boundary pressure", "lrp_contrib_gamma"),
        ("VWAP dislocation", "lrp_contrib_vwap"),
        ("CVD imbalance", "lrp_contrib_cvd"),
        ("MACD acceleration", "lrp_contrib_macd"),
    ]
    contrib_lines = []
    for label, key in lrp_contrib_rows:
        value = latest.get(key)
        if value is not None and pd.notna(value):
            contrib_lines.append(
                f"<tr><td>{label}</td><td>{float(value):.3f}</td></tr>"
            )
    lrp_contrib_html = ""
    elastic_notes_html = "".join(f"<li>{note}</li>" for note in ELASTIC_REBOUND_NOTES)
    elastic_panel = f"""
  <section class="chart-panel">
    <h2>Elastic Rebound &amp; Hidden Reservoir</h2>
    <p class="lrp-meta">
      strain {latest.get('elastic_strain_score', float('nan')):.3f} ·
      regime {latest.get('gamma_rebound_regime', '—')} ·
      reservoir {latest.get('hidden_reservoir_pressure', float('nan')):.3f} ·
      false stability {int(latest.get('false_stability_flag', 0) or 0)}
    </p>
    <ul class="lrp-meta" style="margin-top:0.75rem;padding-left:1.2rem;">{elastic_notes_html}</ul>
  </section>
"""
    if contrib_lines:
        lrp_contrib_html = f"""
        <table class="contrib-table">
          <thead><tr><th>Component</th><th>Contribution</th></tr></thead>
          <tbody>{''.join(contrib_lines)}</tbody>
        </table>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Pressure Field Dashboard — {ticker}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #070b14;
      --panel: #10182b;
      --panel-2: #162038;
      --text: #e8eefc;
      --muted: #8fa3c8;
      --border: rgba(120, 160, 255, 0.18);
      --glow: rgba(0, 212, 255, 0.15);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      background: radial-gradient(ellipse 120% 80% at 10% -10%, #1a1040 0%, transparent 50%),
                  radial-gradient(ellipse 80% 60% at 90% 0%, #0a2840 0%, transparent 45%),
                  var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 2rem clamp(1rem, 4vw, 3rem) 3rem;
    }}
    .hero {{
      display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: flex-end;
      justify-content: space-between; margin-bottom: 2rem;
      padding-bottom: 1.5rem; border-bottom: 1px solid var(--border);
    }}
    .hero h1 {{
      font-size: clamp(1.8rem, 4vw, 2.6rem);
      background: linear-gradient(135deg, #00d4ff, #9d4edd, #ff3cac);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    .hero .meta {{ color: var(--muted); font-size: 0.95rem; line-height: 1.6; }}
    .spot-pill {{
      background: linear-gradient(135deg, #162038, #1e2d50);
      border: 1px solid var(--border); border-radius: 1rem;
      padding: 1rem 1.5rem; text-align: right;
      box-shadow: 0 0 40px var(--glow);
    }}
    .spot-pill .price {{ font-size: 2rem; font-weight: 700; color: #00d4ff; }}
    .quote-panel {{
      background: linear-gradient(135deg, rgba(157,78,221,0.12), rgba(0,212,255,0.08));
      border: 1px solid var(--border); border-radius: 1rem;
      padding: 1.25rem 1.5rem; margin-bottom: 2rem;
      font-style: italic; color: #c5d4f0; line-height: 1.7;
    }}
    .quote-panel strong {{ color: #ffd700; font-style: normal; }}
    .canopy-row {{
      display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 2rem;
    }}
    .canopy-chip {{
      flex: 1 1 140px; background: var(--panel);
      border: 1px solid var(--border); border-radius: 0.75rem;
      padding: 0.85rem 1rem;
    }}
    .chip-label {{ display: block; font-size: 0.72rem; text-transform: uppercase;
                   letter-spacing: 0.08em; color: var(--muted); }}
    .chip-value {{ display: block; font-size: 1.25rem; font-weight: 700; margin: 0.2rem 0; }}
    .chip-sub {{ font-size: 0.78rem; color: #7ec8e3; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 1.25rem; margin-bottom: 2rem;
    }}
    .metric-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 1rem; padding: 1.25rem;
      border-top: 3px solid var(--accent);
      box-shadow: 0 8px 32px rgba(0,0,0,0.35);
      transition: transform 0.2s, box-shadow 0.2s;
    }}
    .metric-card:hover {{
      transform: translateY(-3px);
      box-shadow: 0 12px 40px rgba(0,0,0,0.45), 0 0 24px color-mix(in srgb, var(--accent) 25%, transparent);
    }}
    .metric-tag {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em;
                   color: var(--accent); }}
    .metric-card h3 {{ font-size: 1.15rem; margin: 0.3rem 0 0.6rem; }}
    .metric-value {{ font-size: 1.75rem; font-weight: 700; color: var(--accent); }}
    .metric-regime {{ font-size: 0.82rem; color: #a8c4e8; margin: 0.25rem 0 0.5rem;
                      font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
    .metric-detail {{ font-size: 0.8rem; color: var(--muted); margin-bottom: 0.75rem; }}
    .status-good {{ --accent: #39ff14; }}
    .status-warn {{ --accent: #ffb347; }}
    .status-danger {{ --accent: #ff4d6d; }}
    .status-neutral {{ --accent: #00d4ff; }}
    .chart-panel {{
      background: var(--panel-2); border: 1px solid var(--border);
      border-radius: 1rem; padding: 1.5rem; margin-bottom: 2rem;
    }}
    .chart-panel h2 {{ margin-bottom: 1rem; font-size: 1.1rem; color: #b8cff5; }}
    .stance-panel {{
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 1rem; padding: 1.5rem;
    }}
    .stance-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 1rem; margin: 1rem 0;
    }}
    .stance-grid span {{ display: block; font-size: 0.75rem; color: var(--muted);
                         text-transform: uppercase; letter-spacing: 0.06em; }}
    .stance-grid strong {{ font-size: 1rem; color: #e8eefc; }}
    .stance-action {{ color: #c5d4f0; line-height: 1.6; font-size: 0.92rem; }}
    .observer-table {{
      width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.88rem;
    }}
    .observer-table th, .observer-table td {{
      padding: 0.6rem 0.8rem; border-bottom: 1px solid var(--border); text-align: left;
    }}
    .observer-table th {{ color: var(--muted); font-weight: 500; }}
    .alert-row {{ display: flex; flex-wrap: wrap; gap: 0.6rem; }}
    .alert-chip {{
      background: rgba(255, 77, 109, 0.12);
      border: 1px solid rgba(255, 77, 109, 0.35);
      color: #ffb4c2;
      border-radius: 999px;
      padding: 0.35rem 0.75rem;
      font-size: 0.75rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .contrib-table {{
      width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.86rem;
    }}
    .contrib-table th, .contrib-table td {{
      padding: 0.55rem 0.75rem; border-bottom: 1px solid var(--border); text-align: left;
    }}
    .lrp-meta {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 0.75rem; }}
    footer {{ margin-top: 2rem; color: var(--muted); font-size: 0.78rem; text-align: center; }}
  </style>
</head>
<body>
  <header class="hero">
    <div>
      <h1>Pressure Field Dashboard</h1>
      <p class="meta">{ticker} · as of {latest_date_str} · chain {chain_date}<br/>
      Multi-sensor market state · pressure accumulation · transition · rupture · dissipation</p>
    </div>
    <div class="spot-pill">
      <div class="meta">Spot</div>
      <div class="price">${spot:.2f}</div>
    </div>
  </header>

  <blockquote class="quote-panel">
    <strong>Observer Differential Theory</strong> — {OBSERVER_QUOTE}
  </blockquote>

  <section class="canopy-row">
    {canopy_html}
  </section>

  {stance_block}

  <section class="grid">
    {metric_cards_html}
  </section>

  {alerts_html}

  {elastic_panel}

  <section class="chart-panel">
    <h2>Latent Rupture Potential (LRP)</h2>
    <p class="lrp-meta">Raw {latest.get('LRP_raw', float('nan')):.3f} · Confidence {latest.get('LRP_confidence', 'n/a')}</p>
    <canvas id="chart-lrp" height="90"></canvas>
    <h3 style="margin-top:1rem;font-size:0.95rem;color:#b8cff5;">LRP Component Contributions</h3>
    {lrp_contrib_html}
  </section>

  <section class="chart-panel">
    <h2>Pressure &amp; Acceleration Timeline</h2>
    <canvas id="chart-composite" height="120"></canvas>
  </section>

  <section class="chart-panel">
    <h2>Observer Resolution Ladder</h2>
    <table class="observer-table">
      <thead><tr><th>Observer</th><th>R_o</th><th>Perceives</th><th>T_v (sessions)</th></tr></thead>
      <tbody>
        <tr><td>Passenger</td><td>0.25</td><td>Final rupture only</td><td>{round(min(10.0, max(0.0, 10 * (1 - 0.25) ** 1.4)), 1)}</td></tr>
        <tr><td>Pilot</td><td>0.55</td><td>Regime transitions, saturation extremes</td><td>{round(min(10.0, max(0.0, 10 * (1 - 0.55) ** 1.4)), 1)}</td></tr>
        <tr><td>Mechanic</td><td>0.85</td><td>Latent T_a shifts, B_s buildup, hidden uncertainty</td><td>{round(min(10.0, max(0.0, 10 * (1 - 0.85) ** 1.4)), 1)}</td></tr>
        <tr style="color:#00d4ff"><td><strong>Current ({latest.get('observer_profile', 'n/a')})</strong></td>
            <td><strong>{latest.get('R_o', float('nan')):.3f}</strong></td>
            <td>Computed from live latent signals</td>
            <td><strong>{latest.get('T_v', float('nan'))}</strong></td></tr>
      </tbody>
    </table>
  </section>

  <footer>Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} · finance_crow_engine · not live trading</footer>

  <script>
    const DATA = {json.dumps(chart_payload)};
    const chartDefaults = {{
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8fa3c8', maxTicksLimit: 8 }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        y: {{ ticks: {{ color: '#8fa3c8' }}, grid: {{ color: 'rgba(255,255,255,0.06)' }} }}
      }}
    }};

    function sparkline(canvasId, series, color, label) {{
      const el = document.getElementById(canvasId);
      if (!el || !series) return;
      new Chart(el, {{
        type: 'line',
        data: {{
          labels: DATA.dates,
          datasets: [{{ data: series, borderColor: color, backgroundColor: color + '33',
                       borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0.35 }}]
        }},
        options: {{ ...chartDefaults, plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{
          label: ctx => label + ': ' + (ctx.parsed.y?.toFixed(4) ?? '—')
        }} }} }} }}
      }});
    }}

    sparkline('chart-macd', DATA.macd_hist, '#00d4ff', 'MACD Hist');
    sparkline('chart-rsi', DATA.rsi, '#ff3cac', 'RSI');
    sparkline('chart-cvd', DATA.cvd, '#ffb347', 'CVD');
    sparkline('chart-volume', DATA.volume, '#39ff14', 'Volume');
    sparkline('chart-vwap', DATA.vwap, '#9d4edd', 'VWAP');
    sparkline('chart-lrp', DATA.LRP, '#ff4d6d', 'LRP');
    sparkline('chart-lrp_adj', DATA.LRP_adjusted, '#c9184a', 'LRP adj (exp)');
    sparkline('chart-restoration', DATA.F_r, '#4cc9f0', 'F_r');
    sparkline('chart-capillary', DATA.C_w, '#80ed99', 'C_w');

    const composite = document.getElementById('chart-composite');
    if (composite) {{
      new Chart(composite, {{
        type: 'line',
        data: {{
          labels: DATA.dates,
          datasets: [
            {{ label: 'Close', data: DATA.close, borderColor: '#e8eefc', yAxisID: 'y', tension: 0.2, pointRadius: 0 }},
            {{ label: 'Rupture P', data: DATA.rupture, borderColor: '#ff4d6d', yAxisID: 'y1', tension: 0.3, pointRadius: 0 }},
            {{ label: 'T_a norm', data: DATA.T_a_norm, borderColor: '#ffd700', yAxisID: 'y1', tension: 0.3, pointRadius: 0 }}
          ]
        }},
        options: {{
          responsive: true,
          interaction: {{ mode: 'index', intersect: false }},
          plugins: {{ legend: {{ labels: {{ color: '#c5d4f0' }} }} }},
          scales: {{
            x: {{ ticks: {{ color: '#8fa3c8', maxTicksLimit: 10 }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
            y: {{ position: 'left', ticks: {{ color: '#e8eefc' }}, grid: {{ color: 'rgba(255,255,255,0.06)' }} }},
            y1: {{ position: 'right', ticks: {{ color: '#ffb347' }}, grid: {{ drawOnChartArea: false }} }}
          }}
        }}
      }});
    }}
  </script>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def write_pressure_field_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """Write pressure field time series including LRP and derivative columns."""
    export_cols = [col for col in PRESSURE_FIELD_EXPORT_COLUMNS if col in df.columns]
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df[export_cols].to_csv(csv_path, index=False)


def write_pressure_field_report(
    df: pd.DataFrame,
    report_path: Path,
    *,
    ticker: str,
    chain_date: str,
    gamma: dict[str, Any],
    alerts: list[str],
) -> None:
    """Write markdown summary including LRP and derivative snapshot."""
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else None
    latest_date = latest["Date"]
    latest_date_str = latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date)

    lines = [
        f"# Pressure Field Report — {ticker.upper()}",
        "",
        f"- **As of:** {latest_date_str}",
        f"- **Chain date:** {chain_date}",
        f"- **Close:** {latest['Close']:.2f}",
        "",
        "## Latent Rupture Potential",
        "",
        f"- **LRP:** {_finite_float(latest.get('LRP'), default=float('nan')):.4f}"
        if pd.notna(latest.get("LRP"))
        else "- **LRP:** —",
        f"- **LRP_regime:** {latest.get('LRP_regime', '')}",
        "",
        "## Restoration-Adjusted Rupture Risk — experimental (not canonical)",
        "",
        f"- **{LRP_ADJUSTED_EXPERIMENTAL_LABEL}:** {_finite_float(latest.get('LRP_adjusted'), default=float('nan')):.4f}"
        if pd.notna(latest.get("LRP_adjusted"))
        else f"- **{LRP_ADJUSTED_EXPERIMENTAL_LABEL}:** —",
        f"- **LRP_adjusted_regime (experimental):** {latest.get('LRP_adjusted_regime', '')}",
        f"- **restoration_damper:** {latest.get('restoration_damper', float('nan')):.4f}",
        f"- **capillary_boost:** {latest.get('capillary_boost', float('nan')):.4f}",
        f"- **hysteresis_boost:** {latest.get('hysteresis_boost', float('nan')):.4f}",
        f"- **observer_boost:** {latest.get('observer_boost', float('nan')):.4f}",
        "",
        "_Baseline LRP = pressure-driven risk (canonical). "
        f"{LRP_ADJUSTED_EXPERIMENTAL_LABEL} = compensatory-capacity-adjusted risk — do not treat as canonical._",
        "",
        f"_{LRP_DOCTRINE}_",
        "",
        "## Rate-of-Change Snapshot",
        "",
        f"- **d_canopy_pressure:** {latest.get('d_canopy_pressure', 0.0):+.6f}",
        f"- **dd_canopy_pressure:** {latest.get('dd_canopy_pressure', 0.0):+.6f}",
        f"- **d_R_o:** {latest.get('d_R_o', 0.0):+.6f}",
        f"- **d_T_v:** {latest.get('d_T_v', 0.0):+.6f}",
        f"- **d_gamma_flip_distance:** {latest.get('d_gamma_flip_distance', 0.0):+.6f}",
        f"- **d_vwap_distance:** {latest.get('d_vwap_distance', 0.0):+.6f}",
        "",
        "## Derived Derivatives",
        "",
        "| Metric | Latest |",
        "| :--- | ---: |",
    ]
    for col in DERIVATIVE_COLUMNS:
        if col in latest.index:
            value = latest[col]
            lines.append(f"| {col} | {value:+.6f} |" if pd.notna(value) else f"| {col} | — |")

    lines.extend([
        "",
        "## Rate-of-Change Alerts",
        "",
    ])
    if alerts:
        for alert in alerts:
            lines.append(f"- {alert}")
    else:
        lines.append("- None active")

    lines.extend([
        "",
        "## Restoration & Capillary Physics",
        "",
        f"- **F_r:** {latest.get('F_r', float('nan')):.4f}",
        f"- **D_c / dissipation_score:** {latest.get('D_c', float('nan')):.4f}",
        f"- **restoration_ratio:** {latest.get('restoration_ratio', float('nan')):.4f}",
        f"- **A_micro:** {latest.get('A_micro', float('nan')):.4f}",
        f"- **C_w / capillary_wave_score:** {latest.get('C_w', float('nan')):.4f}",
        f"- **field_regime:** {latest.get('field_regime', '')}",
        f"- **entropy_score:** {latest.get('entropy_score', float('nan')):.4f}",
        "",
        "## Elastic Rebound & Hidden Reservoir",
        "",
        f"- **elastic_strain_score:** {latest.get('elastic_strain_score', float('nan')):.4f}",
        f"- **gamma_rebound_regime:** {latest.get('gamma_rebound_regime', '')}",
        f"- **hidden_reservoir_pressure:** {latest.get('hidden_reservoir_pressure', float('nan')):.4f}",
        f"- **pressure_relocation_ratio:** {latest.get('pressure_relocation_ratio', float('nan')):.4f}",
        f"- **false_stability_flag:** {int(latest.get('false_stability_flag', 0) or 0)}",
        f"- **observability_gap_score:** {latest.get('observability_gap_score', float('nan')):.4f}",
        "",
    ])
    for note in ELASTIC_REBOUND_NOTES:
        lines.append(f"- _{note}_")
    lines.extend([
        "",
        "## Gamma Flip",
        "",
        f"- **Flip strike:** {gamma.get('gamma_flip_strike')}",
        f"- **Distance %:** {gamma.get('distance_to_flip_pct')}",
        f"- **Regime:** {gamma.get('gamma_regime')}",
        "",
        "## Observer Differential",
        "",
        f"- **T_a_regime:** {latest.get('T_a_regime', '')}",
        f"- **R_o:** {latest.get('R_o', float('nan')):.4f}",
        f"- **T_v:** {latest.get('T_v', float('nan'))}",
        f"- **observer_profile:** {latest.get('observer_profile', '')}",
        "",
    ])
    if previous is not None:
        lines.append(f"_Previous session close: {previous['Close']:.2f}_")
        lines.append("")

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latest_json(
    latest: pd.Series,
    gamma: dict[str, Any],
    json_path: Path,
    *,
    ticker: str,
    spot: float,
    chain_date: str,
    alerts: list[str],
) -> None:
    """Write compact JSON snapshot with stable schema keys."""
    snapshot = build_stable_snapshot(
        ticker=ticker,
        latest=latest,
        spot=spot,
        gamma=gamma,
    )
    extras = {
        "chain_date": chain_date,
        "cvd_note": "Signed-volume proxy (close direction x volume), not tick-true CVD.",
        "rate_of_change_alerts": alerts,
        "LRP_confidence": str(latest.get("LRP_confidence", "") or ""),
        "LRP_raw": _finite_float(latest.get("LRP_raw"), default=None),
        "LRP_adjusted": _finite_float(latest.get("LRP_adjusted"), default=None),
        "LRP_adjusted_regime": str(latest.get("LRP_adjusted_regime", "") or ""),
        "LRP_adjusted_experimental": True,
        "lrp_doctrine": LRP_DOCTRINE,
        "loop_closure": {
            "restoration_damper": _finite_float(latest.get("restoration_damper"), default=None),
            "capillary_boost": _finite_float(latest.get("capillary_boost"), default=None),
            "hysteresis_boost": _finite_float(latest.get("hysteresis_boost"), default=None),
            "observer_boost": _finite_float(latest.get("observer_boost"), default=None),
            "LRP_adjusted_raw": _finite_float(latest.get("LRP_adjusted_raw"), default=None),
            "delta_vs_baseline_LRP": (
                None
                if latest.get("LRP_adjusted") is None or latest.get("LRP") is None
                or pd.isna(latest.get("LRP_adjusted"))
                or pd.isna(latest.get("LRP"))
                else _finite_float(float(latest["LRP_adjusted"]) - float(latest["LRP"]))
            ),
        },
        "lrp_contributions": {
            "T_a contribution": _finite_float(latest.get("lrp_contrib_T_a"), default=None),
            "Observer decay": _finite_float(latest.get("lrp_contrib_observer"), default=None),
            "Gamma boundary pressure": _finite_float(latest.get("lrp_contrib_gamma"), default=None),
            "VWAP dislocation": _finite_float(latest.get("lrp_contrib_vwap"), default=None),
            "CVD imbalance": _finite_float(latest.get("lrp_contrib_cvd"), default=None),
            "MACD acceleration": _finite_float(latest.get("lrp_contrib_macd"), default=None),
        },
        "derivatives": {
            col: _finite_float(latest.get(col), default=None)
            for col in DERIVATIVE_COLUMNS
            if col in latest.index
        },
        "pressure_sensors": {
            "macd_histogram": _finite_float(latest.get("macd_histogram"), default=None),
            "rsi": _finite_float(latest.get("rsi"), default=None),
            "volume_regime": str(latest.get("volume_regime", "") or ""),
            "vwap_regime": str(latest.get("vwap_regime", "") or ""),
            "gamma_regime": str(gamma.get("gamma_regime", "") or ""),
        },
        "canopyento": {
            "B_s": _finite_float(latest.get("B_s"), default=None),
            "E_i": _finite_float(latest.get("E_i"), default=None),
            "rupture_pressure_score": _finite_float(latest.get("rupture_pressure_score"), default=None),
        },
        "physics": {
            "F_r": _finite_float(latest.get("F_r"), default=None),
            "D_c": _finite_float(latest.get("D_c"), default=None),
            "restoration_ratio": _finite_float(latest.get("restoration_ratio"), default=None),
            "dissipation_score": _finite_float(latest.get("dissipation_score"), default=None),
            "A_micro": _finite_float(latest.get("A_micro"), default=None),
            "C_w": _finite_float(latest.get("C_w"), default=None),
            "capillary_wave_score": _finite_float(latest.get("capillary_wave_score"), default=None),
            "field_regime": str(latest.get("field_regime", "") or ""),
            "entropy_score": _finite_float(latest.get("entropy_score"), default=None),
            "equilibrium_field_strength": _finite_float(latest.get("equilibrium_field_strength"), default=None),
            "deviation_from_equilibrium": _finite_float(latest.get("deviation_from_equilibrium"), default=None),
            "historical_stress_memory": _finite_float(latest.get("historical_stress_memory"), default=None),
            "recursive_pressure_carryover": _finite_float(latest.get("recursive_pressure_carryover"), default=None),
            "observer_feedback_score": _finite_float(latest.get("observer_feedback_score"), default=None),
            "effective_pressure": _finite_float(latest.get("effective_pressure"), default=None),
        },
        "elastic_rebound": {
            "elastic_strain_score": _finite_float(latest.get("elastic_strain_score"), default=None),
            "gamma_rebound_regime": str(latest.get("gamma_rebound_regime", "") or ""),
            "hidden_reservoir_pressure": _finite_float(latest.get("hidden_reservoir_pressure"), default=None),
            "pressure_relocation_ratio": _finite_float(latest.get("pressure_relocation_ratio"), default=None),
            "false_dissipation_risk": _finite_float(latest.get("false_dissipation_risk"), default=None),
            "false_stability_flag": int(latest.get("false_stability_flag", 0) or 0),
            "observability_gap_score": _finite_float(latest.get("observability_gap_score"), default=None),
            "notes": list(ELASTIC_REBOUND_NOTES),
        },
    }
    if pd.notna(latest.get("stance_quadrant")):
        extras["weekly_stance"] = {
            "stance_quadrant": str(latest["stance_quadrant"]),
            "gate_stance": str(latest["gate_stance"]),
            "stance_confidence": _finite_float(latest.get("stance_confidence"), default=None),
            "recommended_action": str(latest["recommended_action"]),
        }
    write_stable_snapshot_json(snapshot, json_path, extras=extras)


def open_dashboard(html_path: Path) -> None:
    """Open generated dashboard HTML in the default browser."""
    uri = html_path.resolve().as_uri()
    if not webbrowser.open(uri):
        print(f"Could not open browser automatically. Open manually: {html_path.resolve()}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pressure Field Dashboard — HTML market-state dashboard with multi-sensor metrics."
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help=f"Ticker (default: {DEFAULT_TICKER})")
    parser.add_argument("--stock-dir", default=DEFAULT_STOCK_DIR, help="Stock CSV directory")
    parser.add_argument("--option-dir", default=DEFAULT_OPTION_DIR, help="Option chain directory")
    parser.add_argument("--lookback", type=int, default=20, help="CanopyEnto boundary lookback")
    parser.add_argument("--tolerance", type=float, default=0.003, help="Boundary tolerance fraction")
    parser.add_argument("--volume-window", type=int, default=20, dest="volume_window")
    parser.add_argument("--weekly-window", type=int, default=5, dest="weekly_window")
    parser.add_argument("--chart-days", type=int, default=DEFAULT_LOOKBACK_DAYS, dest="chart_days")
    parser.add_argument("--output", default=None, help="HTML output path")
    parser.add_argument("--csv", default=None, dest="csv_output", help="CSV time series output path")
    parser.add_argument("--report", default=None, help="Markdown report output path")
    parser.add_argument("--json", default=None, dest="json_output", help="JSON snapshot path")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open generated HTML dashboard in the default browser",
    )
    args = parser.parse_args(argv)

    ticker = args.ticker.upper()
    html_path = Path(args.output or DEFAULT_OUTPUT_HTML.format(ticker=ticker))
    json_path = Path(args.json_output or DEFAULT_OUTPUT_JSON.format(ticker=ticker))
    csv_path = Path(args.csv_output or DEFAULT_OUTPUT_CSV.format(ticker=ticker))
    report_path = Path(args.report or DEFAULT_OUTPUT_MD.format(ticker=ticker))

    try:
        stock_df = load_stock_data_fallback(ticker, base_dir=args.stock_dir)
        spot = float(prepare_ohlcv(stock_df)["Close"].iloc[-1])

        chain_date = "unavailable"
        option_df: Optional[pd.DataFrame] = None
        try:
            chain_date = get_most_recent_option_date(ticker, base_dir=args.option_dir, verbose=False)
            option_df = load_option_chain_data(ticker, date=chain_date, base_dir=args.option_dir)
        except FileNotFoundError as exc:
            print(f"Warning: option chain unavailable ({exc}). Gamma flip will be null.", file=sys.stderr)

        frame = build_pressure_frame(
            stock_df,
            lookback=args.lookback,
            tolerance=args.tolerance,
            volume_window=args.volume_window,
            weekly_window=args.weekly_window,
        )
        gamma = compute_gamma_flip(option_df, spot)
        frame = enrich_pressure_derivatives(frame, gamma=gamma)
        frame = enrich_pressure_physics(frame, gamma=gamma)
        frame = apply_lrp_loop_closure(frame)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    latest = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) > 1 else None
    alerts = compute_rate_of_change_alerts(latest, previous=previous)
    lrp_debug = build_lrp_debug_payload(frame, ticker=ticker, gamma=gamma)
    write_lrp_debug_json(lrp_debug, Path(f"outputs/lrp_debug_{ticker}.json"))

    write_pressure_field_csv(frame, csv_path)
    write_pressure_field_report(
        frame,
        report_path,
        ticker=ticker,
        chain_date=chain_date,
        gamma=gamma,
        alerts=alerts,
    )
    render_html_dashboard(
        frame,
        ticker=ticker,
        spot=spot,
        chain_date=chain_date,
        gamma=gamma,
        alerts=alerts,
        output_path=html_path,
        chart_days=args.chart_days,
    )
    write_latest_json(
        latest, gamma, json_path, ticker=ticker, spot=spot, chain_date=chain_date, alerts=alerts
    )

    print(f"Pressure Field Dashboard complete for {ticker}")
    print(f"  HTML:   {html_path}")
    print(f"  JSON:   {json_path}")
    print(f"  CSV:    {csv_path}")
    print(f"  Report: {report_path}")
    print(f"  Debug:  outputs/lrp_debug_{ticker}.json")
    print(
        f"  Latest: close=${spot:.2f} | MACD hist={latest['macd_histogram']:.4f} | "
        f"RSI={latest['rsi']:.1f} | CVD imb={latest['cvd_imbalance']:.3f} | "
        f"gamma flip={gamma.get('gamma_flip_strike')}"
    )
    if pd.notna(latest.get("R_o")):
        print(
            f"  Observer: R_o={latest['R_o']:.3f} | T_v={latest['T_v']} | "
            f"T_a={latest.get('T_a_regime', '')} | profile={latest.get('observer_profile', '')}"
        )
    if pd.notna(latest.get("LRP")):
        print(
            f"  LRP: {latest['LRP']:.3f} (raw {latest.get('LRP_raw', float('nan')):.3f}) "
            f"({latest.get('LRP_regime', '')}, {latest.get('LRP_confidence', '')})"
        )
    if pd.notna(latest.get("LRP_adjusted")):
        delta = float(latest["LRP_adjusted"]) - float(latest.get("LRP", 0.0))
        print(
            f"  {LRP_ADJUSTED_EXPERIMENTAL_LABEL}: {latest['LRP_adjusted']:.3f} "
            f"({latest.get('LRP_adjusted_regime', '')}, delta {delta:+.3f} vs baseline)"
        )
    if pd.notna(latest.get("F_r")):
        print(
            f"  Physics: F_r={latest['F_r']:.3f} D_c={latest.get('D_c', float('nan')):.3f} "
            f"C_w={latest.get('C_w', float('nan')):.3f} "
            f"field={latest.get('field_regime', '')}"
        )
    audit = lrp_debug.get("legacy_formula_audit", {})
    if audit:
        print(
            f"  Legacy audit: ratio={audit.get('ratio_before_clamp')} "
            f"numerator={audit.get('numerator_sum')} denominator={audit.get('denominator')} "
            f"driver={audit.get('saturation_driver')}"
        )
    if alerts:
        print(f"  Alerts: {', '.join(alerts)}")
    if args.open:
        open_dashboard(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
