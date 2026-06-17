"""IV skew engine — strike vs IV by expiration with benchmark comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from implied_vol_solver import fill_missing_iv

DEFAULT_OUTPUT_DIR = Path("outputs/laser_falcon")


def _prepare_iv_frame(option_df: pd.DataFrame, *, spot: float) -> pd.DataFrame:
    df = fill_missing_iv(option_df, spot=spot)
    df = df[df["IV"].notna() & (df["IV"] > 0)].copy()
    if df.empty:
        return df
    df["iv_pct"] = df["IV"] * 100.0 if df["IV"].max() <= 3.0 else df["IV"]
    return df


def _atm_iv_for_side(side_df: pd.DataFrame, spot: float) -> float:
    if side_df.empty:
        return float("nan")
    idx = (side_df["Strike"] - spot).abs().idxmin()
    return float(side_df.loc[idx, "iv_pct"])


def _wing_iv(side_df: pd.DataFrame, spot: float, *, wing: str) -> float:
    if side_df.empty:
        return float("nan")
    if wing == "put":
        wing_df = side_df[side_df["Strike"] < spot * 0.95]
    else:
        wing_df = side_df[side_df["Strike"] > spot * 1.05]
    if wing_df.empty:
        return float("nan")
    return float(wing_df["iv_pct"].max())


def compute_skew_metrics(
    option_df: pd.DataFrame,
    *,
    spot: float,
    expiration: Optional[str] = None,
) -> dict[str, Any]:
    """Compute skew metrics for one expiration (or nearest available)."""
    df = _prepare_iv_frame(option_df, spot=spot)
    if df.empty:
        return {
            "expiration": expiration or "",
            "atm_iv": None,
            "put_wing_iv": None,
            "call_wing_iv": None,
            "skew_slope": None,
            "skew_inversion_flag": False,
            "call_fomo_flag": False,
            "put_fear_flag": False,
            "status": "INSUFFICIENT",
        }

    expirations = sorted(df["Expiration Date"].unique().tolist())
    chosen = expiration if expiration in expirations else expirations[0]
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

    skew_inversion_flag = bool(
        not np.isnan(put_wing_iv)
        and not np.isnan(call_wing_iv)
        and put_wing_iv < call_wing_iv
    )
    call_fomo_flag = bool(not np.isnan(call_wing_iv) and not np.isnan(atm_iv) and call_wing_iv > atm_iv * 1.15)
    put_fear_flag = bool(not np.isnan(put_wing_iv) and not np.isnan(atm_iv) and put_wing_iv > atm_iv * 1.15)

    return {
        "expiration": chosen,
        "atm_iv": round(atm_iv, 4) if not np.isnan(atm_iv) else None,
        "put_wing_iv": round(put_wing_iv, 4) if not np.isnan(put_wing_iv) else None,
        "call_wing_iv": round(call_wing_iv, 4) if not np.isnan(call_wing_iv) else None,
        "skew_slope": round(skew_slope, 6) if not np.isnan(skew_slope) else None,
        "skew_inversion_flag": skew_inversion_flag,
        "call_fomo_flag": call_fomo_flag,
        "put_fear_flag": put_fear_flag,
        "status": "OK",
    }


def plot_iv_skew(
    option_df: pd.DataFrame,
    *,
    ticker: str,
    spot: float,
    expiration: Optional[str] = None,
    output_path: Optional[Path] = None,
    benchmark_metrics: Optional[dict[str, Any]] = None,
) -> Path:
    """Plot Strike vs IV for calls and puts."""
    output_path = Path(output_path or DEFAULT_OUTPUT_DIR / f"{ticker.upper()}_iv_skew.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = _prepare_iv_frame(option_df, spot=spot)
    fig, ax = plt.subplots(figsize=(10, 6))
    if df.empty:
        ax.text(0.5, 0.5, "Insufficient IV data for skew plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{ticker.upper()} IV Skew — no data")
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return output_path

    expirations = sorted(df["Expiration Date"].unique().tolist())
    chosen = expiration if expiration in expirations else expirations[0]
    slice_df = df[df["Expiration Date"] == chosen]
    calls = slice_df[slice_df["option_type"] == "call"].sort_values("Strike")
    puts = slice_df[slice_df["option_type"] == "put"].sort_values("Strike")

    if not calls.empty:
        ax.plot(calls["Strike"], calls["iv_pct"], "o-", color="#2ecc71", label="Calls", linewidth=2)
    if not puts.empty:
        ax.plot(puts["Strike"], puts["iv_pct"], "o-", color="#e74c3c", label="Puts", linewidth=2)
    ax.axvline(spot, color="#3498db", linestyle="--", alpha=0.7, label=f"Spot ${spot:.2f}")

    metrics = compute_skew_metrics(option_df, spot=spot, expiration=chosen)
    subtitle = (
        f"ATM IV {metrics.get('atm_iv', 'n/a')}% | "
        f"Put wing {metrics.get('put_wing_iv', 'n/a')}% | "
        f"Call wing {metrics.get('call_wing_iv', 'n/a')}%"
    )
    if benchmark_metrics:
        subtitle += (
            f" | vs benchmark ATM {benchmark_metrics.get('atm_iv', 'n/a')}%"
        )
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

    return {
        "target": target_metrics,
        "benchmark": benchmark_metrics,
        "atm_iv_delta": _delta("atm_iv"),
        "put_wing_iv_delta": _delta("put_wing_iv"),
        "call_wing_iv_delta": _delta("call_wing_iv"),
        "skew_slope_delta": _delta("skew_slope"),
        "relative_put_fear": bool(target_metrics.get("put_fear_flag")) and not benchmark_metrics.get("put_fear_flag"),
        "relative_call_fomo": bool(target_metrics.get("call_fomo_flag")) and not benchmark_metrics.get("call_fomo_flag"),
    }
