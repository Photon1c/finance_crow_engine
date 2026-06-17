"""IV skew engine — strike vs IV by expiration with benchmark comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chain_integrity_engine import clean_chain_expirations
from implied_vol_solver import fill_missing_iv

DEFAULT_OUTPUT_DIR = Path("outputs/laser_falcon")
SPOT_RANGE_PCT = 0.15
WING_OTM_PCT = 0.03
OVERPRICED_WING_THRESHOLD = 1.12


def _prepare_iv_frame(option_df: pd.DataFrame, *, spot: float) -> tuple[pd.DataFrame, list[str]]:
    cleaned, clean_diag = clean_chain_expirations(option_df)
    warnings = list(clean_diag.get("warnings", []))
    df = fill_missing_iv(cleaned, spot=spot)
    df = df[df["IV"].notna() & (df["IV"] > 0)].copy()
    if df.empty:
        return df, warnings
    df["iv_pct"] = df["IV"] * 100.0 if df["IV"].max() <= 3.0 else df["IV"]
    return df, warnings


def filter_strikes_near_spot(df: pd.DataFrame, spot: float, *, range_pct: float = SPOT_RANGE_PCT) -> pd.DataFrame:
    """Keep strikes within ±range_pct of spot for skew shape."""
    if df.empty:
        return df
    lower = spot * (1.0 - range_pct)
    upper = spot * (1.0 + range_pct)
    return df[(df["Strike"] >= lower) & (df["Strike"] <= upper)].copy()


def _atm_iv_for_side(side_df: pd.DataFrame, spot: float) -> float:
    if side_df.empty:
        return float("nan")
    idx = (side_df["Strike"] - spot).abs().idxmin()
    return float(side_df.loc[idx, "iv_pct"])


def _wing_iv(side_df: pd.DataFrame, spot: float, *, wing: str) -> float:
    """Select wing IV with OTM preference and graceful fallback for sparse chains."""
    if side_df.empty:
        return float("nan")

    if wing == "put":
        otm = side_df[side_df["Strike"] < spot * (1.0 - WING_OTM_PCT)]
        if not otm.empty:
            return float(otm["iv_pct"].max())
        below_spot = side_df[side_df["Strike"] <= spot]
        if not below_spot.empty:
            return float(below_spot.loc[(below_spot["Strike"] - spot).abs().idxmin(), "iv_pct"])
    else:
        otm = side_df[side_df["Strike"] > spot * (1.0 + WING_OTM_PCT)]
        if not otm.empty:
            return float(otm["iv_pct"].max())
        above_spot = side_df[side_df["Strike"] >= spot]
        if not above_spot.empty:
            return float(above_spot.loc[(above_spot["Strike"] - spot).abs().idxmin(), "iv_pct"])

    return float(side_df.loc[side_df["iv_pct"].idxmax(), "iv_pct"])


def smooth_iv_curve(strikes: np.ndarray, ivs: np.ndarray, *, eval_points: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Smooth discrete strike→IV into a continuous smile."""
    if len(strikes) < 3:
        return strikes, ivs
    order = min(2, len(strikes) - 1)
    try:
        coeffs = np.polyfit(strikes, ivs, order)
        x_smooth = np.linspace(float(strikes.min()), float(strikes.max()), eval_points)
        y_smooth = np.polyval(coeffs, x_smooth)
        return x_smooth, y_smooth
    except Exception:
        return strikes, ivs


def surface_curvature(strikes: np.ndarray, ivs: np.ndarray) -> float:
    """Second-derivative proxy of IV curve (smile curvature)."""
    if len(strikes) < 3:
        return 0.0
    try:
        order = min(2, len(strikes) - 1)
        coeffs = np.polyfit(strikes, ivs, order)
        if order < 2:
            return 0.0
        deriv2 = np.polyder(np.polyder(coeffs))
        mid = float(np.median(strikes))
        return float(abs(np.polyval(deriv2, mid)))
    except Exception:
        return 0.0


def compute_skew_metrics(
    option_df: pd.DataFrame,
    *,
    spot: float,
    expiration: Optional[str] = None,
    spot_range_pct: float = SPOT_RANGE_PCT,
) -> dict[str, Any]:
    """Compute skew metrics for one expiration (or nearest available)."""
    df, chain_warnings = _prepare_iv_frame(option_df, spot=spot)
    if df.empty:
        return _empty_skew_metrics(expiration, chain_warnings=chain_warnings)

    expirations = sorted(df["Expiration Date"].unique().tolist())
    chosen = expiration if expiration in expirations else expirations[0]
    slice_df = filter_strikes_near_spot(df[df["Expiration Date"] == chosen], spot, range_pct=spot_range_pct)
    if slice_df.empty:
        slice_df = df[df["Expiration Date"] == chosen]

    calls = slice_df[slice_df["option_type"] == "call"]
    puts = slice_df[slice_df["option_type"] == "put"]

    atm_call = _atm_iv_for_side(calls, spot)
    atm_put = _atm_iv_for_side(puts, spot)
    atm_iv = float(np.nanmean([atm_call, atm_put]))
    put_wing_iv = _wing_iv(puts, spot, wing="put")
    call_wing_iv = _wing_iv(calls, spot, wing="call")

    skew_slope = float("nan")
    if puts.shape[0] >= 2:
        reg = puts[["Strike", "iv_pct"]].dropna()
        if len(reg) >= 2:
            coeffs = np.polyfit(reg["Strike"], reg["iv_pct"], 1)
            skew_slope = float(coeffs[0])

    skew_ratio = None
    if not np.isnan(put_wing_iv) and not np.isnan(call_wing_iv) and call_wing_iv > 0:
        skew_ratio = round(float(put_wing_iv / call_wing_iv), 4)

    skew_asymmetry = None
    if not np.isnan(atm_iv) and atm_iv > 0 and not np.isnan(put_wing_iv) and not np.isnan(call_wing_iv):
        skew_asymmetry = round(float((put_wing_iv - call_wing_iv) / atm_iv), 4)

    skew_inversion_flag = bool(
        skew_ratio is not None and skew_ratio < 1.0
    )
    calls_overpriced_flag = bool(
        not np.isnan(call_wing_iv) and not np.isnan(atm_iv) and call_wing_iv > atm_iv * OVERPRICED_WING_THRESHOLD
    )
    puts_overpriced_flag = bool(
        not np.isnan(put_wing_iv) and not np.isnan(atm_iv) and put_wing_iv > atm_iv * OVERPRICED_WING_THRESHOLD
    )
    call_fomo_flag = calls_overpriced_flag
    put_fear_flag = puts_overpriced_flag

    put_strikes = puts["Strike"].to_numpy(dtype=float) if not puts.empty else np.array([])
    put_ivs = puts["iv_pct"].to_numpy(dtype=float) if not puts.empty else np.array([])
    smile_curvature = surface_curvature(put_strikes, put_ivs) if len(put_strikes) >= 3 else 0.0

    return {
        "expiration": chosen,
        "spot_range_pct": spot_range_pct,
        "n_strikes_in_band": int(slice_df["Strike"].nunique()),
        "atm_iv": round(atm_iv, 4) if not np.isnan(atm_iv) else None,
        "put_wing_iv": round(put_wing_iv, 4) if not np.isnan(put_wing_iv) else None,
        "call_wing_iv": round(call_wing_iv, 4) if not np.isnan(call_wing_iv) else None,
        "skew_slope": round(skew_slope, 6) if not np.isnan(skew_slope) else None,
        "skew_ratio": skew_ratio,
        "skew_asymmetry_pressure": skew_asymmetry,
        "skew_inversion_flag": skew_inversion_flag,
        "calls_overpriced_flag": calls_overpriced_flag,
        "puts_overpriced_flag": puts_overpriced_flag,
        "call_fomo_flag": call_fomo_flag,
        "put_fear_flag": put_fear_flag,
        "surface_curvature": round(smile_curvature, 6),
        "chain_warnings": chain_warnings,
        "status": "OK",
    }


def _empty_skew_metrics(expiration: Optional[str], *, chain_warnings: Optional[list[str]] = None) -> dict[str, Any]:
    return {
        "expiration": expiration or "",
        "spot_range_pct": SPOT_RANGE_PCT,
        "n_strikes_in_band": 0,
        "atm_iv": None,
        "put_wing_iv": None,
        "call_wing_iv": None,
        "skew_slope": None,
        "skew_ratio": None,
        "skew_asymmetry_pressure": None,
        "skew_inversion_flag": False,
        "calls_overpriced_flag": False,
        "puts_overpriced_flag": False,
        "call_fomo_flag": False,
        "put_fear_flag": False,
        "surface_curvature": 0.0,
        "chain_warnings": chain_warnings or [],
        "status": "INSUFFICIENT",
    }


def plot_iv_skew(
    option_df: pd.DataFrame,
    *,
    ticker: str,
    spot: float,
    expiration: Optional[str] = None,
    output_path: Optional[Path] = None,
    benchmark_metrics: Optional[dict[str, Any]] = None,
    spot_range_pct: float = SPOT_RANGE_PCT,
) -> Path:
    """Plot smoothed Strike vs IV for calls and puts."""
    output_path = Path(output_path or DEFAULT_OUTPUT_DIR / f"{ticker.upper()}_iv_skew.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df, _ = _prepare_iv_frame(option_df, spot=spot)
    fig, ax = plt.subplots(figsize=(10, 6))
    if df.empty:
        ax.text(0.5, 0.5, "Insufficient IV data for skew plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{ticker.upper()} IV Skew — no data")
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return output_path

    expirations = sorted(df["Expiration Date"].unique().tolist())
    chosen = expiration if expiration in expirations else expirations[0]
    slice_df = filter_strikes_near_spot(df[df["Expiration Date"] == chosen], spot, range_pct=spot_range_pct)
    if slice_df.empty:
        slice_df = df[df["Expiration Date"] == chosen]

    calls = slice_df[slice_df["option_type"] == "call"].sort_values("Strike")
    puts = slice_df[slice_df["option_type"] == "put"].sort_values("Strike")

    if not calls.empty:
        cx, cy = smooth_iv_curve(calls["Strike"].to_numpy(), calls["iv_pct"].to_numpy())
        ax.plot(cx, cy, "-", color="#2ecc71", label="Calls (smoothed)", linewidth=2)
        ax.scatter(calls["Strike"], calls["iv_pct"], color="#27ae60", s=30, alpha=0.7)
    if not puts.empty:
        px, py = smooth_iv_curve(puts["Strike"].to_numpy(), puts["iv_pct"].to_numpy())
        ax.plot(px, py, "-", color="#e74c3c", label="Puts (smoothed)", linewidth=2)
        ax.scatter(puts["Strike"], puts["iv_pct"], color="#c0392b", s=30, alpha=0.7)

    ax.axvline(spot, color="#3498db", linestyle="--", alpha=0.7, label=f"Spot ${spot:.2f}")
    ax.axvline(spot * (1 - spot_range_pct), color="#95a5a6", linestyle=":", alpha=0.5)
    ax.axvline(spot * (1 + spot_range_pct), color="#95a5a6", linestyle=":", alpha=0.5)

    metrics = compute_skew_metrics(option_df, spot=spot, expiration=chosen, spot_range_pct=spot_range_pct)
    subtitle = (
        f"ATM IV {metrics.get('atm_iv', 'n/a')}% | "
        f"Put wing {metrics.get('put_wing_iv', 'n/a')}% | "
        f"Call wing {metrics.get('call_wing_iv', 'n/a')}% | "
        f"skew ratio {metrics.get('skew_ratio', 'n/a')}"
    )
    if benchmark_metrics:
        subtitle += f" | vs benchmark ATM {benchmark_metrics.get('atm_iv', 'n/a')}%"
    ax.set_title(f"{ticker.upper()} IV Skew — {chosen}\n{subtitle}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compare_skew_to_benchmark(
    target_metrics: dict[str, Any],
    benchmark_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compare target ticker skew metrics against benchmark (e.g. SPCX vs SPY)."""
    def _delta(key: str) -> Optional[float]:
        a = target_metrics.get(key)
        b = benchmark_metrics.get(key)
        if a is None or b is None:
            return None
        return round(float(a) - float(b), 4)

    def _ratio(key: str) -> Optional[float]:
        a = target_metrics.get(key)
        b = benchmark_metrics.get(key)
        if a is None or b is None or float(b) == 0:
            return None
        return round(float(a) / float(b), 4)

    return {
        "target": target_metrics,
        "benchmark": benchmark_metrics,
        "atm_iv_delta": _delta("atm_iv"),
        "put_wing_iv_delta": _delta("put_wing_iv"),
        "call_wing_iv_delta": _delta("call_wing_iv"),
        "skew_slope_delta": _delta("skew_slope"),
        "volatility_ratio": _ratio("atm_iv"),
        "skew_ratio_delta": _delta("skew_ratio"),
        "relative_put_fear": bool(target_metrics.get("put_fear_flag")) and not benchmark_metrics.get("put_fear_flag"),
        "relative_call_fomo": bool(target_metrics.get("call_fomo_flag")) and not benchmark_metrics.get("call_fomo_flag"),
    }
