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

DEFAULT_TICKER = "SPY"
DEFAULT_LOOKBACK_DAYS = 120
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_RSI_PERIOD = 14
DEFAULT_VWAP_WINDOW = 20
DEFAULT_OUTPUT_HTML = "outputs/pressure_field_dashboard_{ticker}.html"
DEFAULT_OUTPUT_JSON = "outputs/pressure_field_latest_{ticker}.json"

OBSERVER_QUOTE = (
    "Complex systems do not fail equally for all observers. Rupture becomes visible "
    "at different times depending on an observer's proximity to the system's internal "
    "pressure variables. The system changes before most observers are capable of "
    "perceiving the change."
)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


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
    result["macd_pressure_accel"] = result["macd_accel"] / hist_std.where(hist_std > 0)
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
    result["cvd_imbalance"] = result["cvd_slope"] / vol_scale.where(vol_scale > 0)
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
    avg_vol = result["Volume"].rolling(window, min_periods=window).mean()
    result["volume_injection"] = result["Volume"] / avg_vol.where(avg_vol > 0)
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
    cum_pv = pv.rolling(window, min_periods=window).sum()
    cum_vol = result["Volume"].rolling(window, min_periods=window).sum()
    result["vwap"] = cum_pv / cum_vol.where(cum_vol > 0)
    result["vwap_distance_pct"] = (result["Close"] - result["vwap"]) / result["vwap"] * 100.0
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


def compute_gamma_flip(
    option_df: pd.DataFrame,
    spot: float,
) -> dict[str, Any]:
    """Gamma flip level from option chain — phase boundary transition."""
    df = option_df.copy()
    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
    df = df[df["Strike"].notna()].copy()
    if df.empty:
        return {
            "gamma_flip_strike": float("nan"),
            "distance_to_flip_pct": float("nan"),
            "net_gamma_at_spot": float("nan"),
            "gamma_regime": "NO_CHAIN",
            "call_gamma_oi": 0.0,
            "put_gamma_oi": 0.0,
        }

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
    )
    grouped["net_gamma"] = grouped["call_gamma_oi"] - grouped["put_gamma_oi"]

    flip_strike = float("nan")
    strikes = grouped["Strike"].values
    net = grouped["net_gamma"].values
    for idx in range(1, len(strikes)):
        if net[idx - 1] == 0:
            flip_strike = float(strikes[idx - 1])
            break
        if net[idx - 1] * net[idx] < 0:
            weight = abs(net[idx - 1]) / (abs(net[idx - 1]) + abs(net[idx]) + 1e-9)
            flip_strike = float(strikes[idx - 1] * (1 - weight) + strikes[idx] * weight)
            break

    spot_idx = (grouped["Strike"] - spot).abs().idxmin()
    net_at_spot = float(grouped.loc[spot_idx, "net_gamma"])
    distance_pct = float("nan")
    if not np.isnan(flip_strike) and spot > 0:
        distance_pct = (spot - flip_strike) / spot * 100.0

    if np.isnan(flip_strike):
        regime = "FLIP_UNDEFINED"
    elif abs(distance_pct) < 0.5:
        regime = "AT_PHASE_BOUNDARY"
    elif distance_pct > 0:
        regime = "ABOVE_FLIP_POSITIVE_GAMMA"
    else:
        regime = "BELOW_FLIP_NEGATIVE_GAMMA"

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
              "WAIT / PACKET BUFFERING"}
    if regime in danger:
        return "status-danger"
    if regime in warning:
        return "status-warn"
    if regime in positive:
        return "status-good"
    return "status-neutral"


def render_html_dashboard(
    df: pd.DataFrame,
    *,
    ticker: str,
    spot: float,
    chain_date: str,
    gamma: dict[str, Any],
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
            "value": f"${gamma['gamma_flip_strike']:.2f}" if not np.isnan(gamma["gamma_flip_strike"]) else "N/A",
            "sub": gamma["gamma_regime"],
            "detail": (
                f"Spot dist {gamma['distance_to_flip_pct']:+.2f}% · "
                f"net Γ {gamma['net_gamma_at_spot']:.0f}"
                if not np.isnan(gamma.get("distance_to_flip_pct", float("nan")))
                else "Flip level undefined"
            ),
            "color": "#ffd700",
            "status_class": _status_class(str(gamma.get("gamma_regime", ""))),
        },
    ]

    canopy_cards = [
        ("B_s", "Boundary Stability", f"{latest['B_s']:.4f}", "Containment stress from boundary tests"),
        ("E_i", "Injection Energy", f"{latest['E_i']:.4f}", "Volume vs rolling average"),
        ("T_a", "Transitional Acceleration", f"{latest.get('T_a_norm', float('nan')):.4f}", str(latest.get("T_a_regime", ""))),
        ("R_o", "Observational Resolution", f"{latest.get('R_o', float('nan')):.3f}", str(latest.get("observer_profile", ""))),
        ("T_v", "Visibility Horizon", f"{latest.get('T_v', float('nan'))} sessions", "T_v = f(R_o)"),
        ("rupture", "Rupture Pressure", f"{latest['rupture_pressure_score']:.4f}", str(latest.get("regime_label", ""))),
    ]

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

  <section class="chart-panel">
    <h2>Pressure &amp; Acceleration Timeline</h2>
    <canvas id="chart-composite" height="120"></canvas>
  </section>

  <section class="chart-panel">
    <h2>Observer Resolution Ladder</h2>
    <table class="observer-table">
      <thead><tr><th>Observer</th><th>R_o</th><th>Perceives</th><th>T_v (sessions)</th></tr></thead>
      <tbody>
        <tr><td>Passenger</td><td>0.25</td><td>Final rupture only</td><td>{round(10 * (1 - 0.25) ** 1.4, 1)}</td></tr>
        <tr><td>Pilot</td><td>0.55</td><td>Regime transitions, saturation extremes</td><td>{round(10 * (1 - 0.55) ** 1.4, 1)}</td></tr>
        <tr><td>Mechanic</td><td>0.85</td><td>Latent T_a shifts, B_s buildup, hidden uncertainty</td><td>{round(10 * (1 - 0.85) ** 1.4, 1)}</td></tr>
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


def write_latest_json(
    latest: pd.Series,
    gamma: dict[str, Any],
    json_path: Path,
    *,
    ticker: str,
    spot: float,
    chain_date: str,
) -> None:
    """Write compact JSON snapshot for downstream ingestion."""
    date_val = latest["Date"]
    as_of = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)
    payload = {
        "ticker": ticker.upper(),
        "as_of_date": as_of,
        "spot": spot,
        "chain_date": chain_date,
        "observer_differential": {
            "quote": OBSERVER_QUOTE,
            "T_a": float(latest.get("T_a", float("nan"))) if pd.notna(latest.get("T_a")) else None,
            "T_a_norm": float(latest.get("T_a_norm", float("nan"))) if pd.notna(latest.get("T_a_norm")) else None,
            "T_a_regime": str(latest.get("T_a_regime", "")),
            "R_o": float(latest.get("R_o", float("nan"))) if pd.notna(latest.get("R_o")) else None,
            "T_v": float(latest.get("T_v", float("nan"))) if pd.notna(latest.get("T_v")) else None,
            "observer_profile": str(latest.get("observer_profile", "")),
        },
        "pressure_sensors": {
            "macd_histogram": float(latest["macd_histogram"]),
            "macd_regime": str(latest.get("macd_regime", "")),
            "rsi": float(latest["rsi"]),
            "rsi_saturation": str(latest.get("rsi_saturation", "")),
            "cvd_imbalance": float(latest["cvd_imbalance"]),
            "cvd_regime": str(latest.get("cvd_regime", "")),
            "volume_injection": float(latest["volume_injection"]),
            "volume_regime": str(latest.get("volume_regime", "")),
            "vwap": float(latest["vwap"]),
            "vwap_regime": str(latest.get("vwap_regime", "")),
            "gamma_flip": gamma,
        },
        "canopyento": {
            "B_s": float(latest["B_s"]),
            "E_i": float(latest["E_i"]),
            "rupture_pressure_score": float(latest["rupture_pressure_score"]),
            "regime_label": str(latest.get("regime_label", "")),
        },
    }
    if pd.notna(latest.get("stance_quadrant")):
        payload["weekly_stance"] = {
            "stance_quadrant": str(latest["stance_quadrant"]),
            "gate_stance": str(latest["gate_stance"]),
            "stance_confidence": float(latest["stance_confidence"]),
            "recommended_action": str(latest["recommended_action"]),
        }

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    parser.add_argument("--json", default=None, dest="json_output", help="JSON snapshot path")
    args = parser.parse_args(argv)

    ticker = args.ticker.upper()
    html_path = Path(args.output or DEFAULT_OUTPUT_HTML.format(ticker=ticker))
    json_path = Path(args.json_output or DEFAULT_OUTPUT_JSON.format(ticker=ticker))

    try:
        stock_df = load_stock_data_fallback(ticker, base_dir=args.stock_dir)
        chain_date = get_most_recent_option_date(ticker, base_dir=args.option_dir, verbose=False)
        option_df = load_option_chain_data(ticker, date=chain_date, base_dir=args.option_dir)
        spot = float(prepare_ohlcv(stock_df)["Close"].iloc[-1])

        frame = build_pressure_frame(
            stock_df,
            lookback=args.lookback,
            tolerance=args.tolerance,
            volume_window=args.volume_window,
            weekly_window=args.weekly_window,
        )
        gamma = compute_gamma_flip(option_df, spot)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    latest = frame.iloc[-1]
    render_html_dashboard(
        frame,
        ticker=ticker,
        spot=spot,
        chain_date=chain_date,
        gamma=gamma,
        output_path=html_path,
        chart_days=args.chart_days,
    )
    write_latest_json(latest, gamma, json_path, ticker=ticker, spot=spot, chain_date=chain_date)

    print(f"Pressure Field Dashboard complete for {ticker}")
    print(f"  HTML:  {html_path}")
    print(f"  JSON:  {json_path}")
    print(
        f"  Latest: close=${spot:.2f} | MACD hist={latest['macd_histogram']:.4f} | "
        f"RSI={latest['rsi']:.1f} | CVD imb={latest['cvd_imbalance']:.3f} | "
        f"gamma flip={gamma['gamma_flip_strike']}"
    )
    if pd.notna(latest.get("R_o")):
        print(
            f"  Observer: R_o={latest['R_o']:.3f} | T_v={latest['T_v']} | "
            f"T_a={latest.get('T_a_regime', '')} | profile={latest.get('observer_profile', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
