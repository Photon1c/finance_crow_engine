"""Ornstein-Uhlenbeck IV mean-reversion projection engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_OUTPUT_DIR = Path("outputs/laser_falcon")


def ou_half_life(theta: float) -> float:
    """Mean reversion half-life in days (theta per year scaled to daily dt)."""
    if theta <= 0:
        return float("inf")
    daily_theta = theta / 365.0
    return float(np.log(2.0) / daily_theta)


def simulate_ou_iv_paths(
    *,
    iv0: float,
    long_run_mean_iv: float,
    theta: float,
    vol_of_vol: float,
    projection_days: int = 30,
    n_paths: int = 500,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """
    Simulate OU IV paths:
        dIV = theta * (mu - IV) * dt + sigma * dW
    with dt = 1/365 (daily steps).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 365.0
    steps = max(int(projection_days), 1)
    paths = np.zeros((n_paths, steps + 1), dtype=float)
    paths[:, 0] = iv0

    sqrt_dt = np.sqrt(dt)
    for t in range(1, steps + 1):
        dw = rng.standard_normal(n_paths) * sqrt_dt
        prev = paths[:, t - 1]
        paths[:, t] = prev + theta * (long_run_mean_iv - prev) * dt + vol_of_vol * dw
        paths[:, t] = np.clip(paths[:, t], 0.01, 5.0)

    mean_path = paths.mean(axis=0)
    p05 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    return {
        "iv0": iv0,
        "long_run_mean_iv": long_run_mean_iv,
        "theta": theta,
        "vol_of_vol": vol_of_vol,
        "projection_days": projection_days,
        "n_paths": n_paths,
        "half_life_days": ou_half_life(theta),
        "paths": paths,
        "mean_path": mean_path,
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "terminal_mean": float(mean_path[-1]),
        "terminal_p05": float(p05[-1]),
        "terminal_p95": float(p95[-1]),
    }


def plot_ou_iv_projection(
    result: dict[str, Any],
    *,
    ticker: str,
    output_path: Optional[Path] = None,
) -> Path:
    output_path = Path(output_path or DEFAULT_OUTPUT_DIR / f"{ticker.upper()}_ou_iv_projection.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    days = np.arange(result["mean_path"].shape[0])
    fig, ax = plt.subplots(figsize=(10, 6))
    sample = result["paths"][: min(40, result["n_paths"])]
    for path in sample:
        ax.plot(days, path * 100.0, color="#bdc3c7", alpha=0.25, linewidth=0.8)
    ax.fill_between(days, result["p05"] * 100.0, result["p95"] * 100.0, color="#3498db", alpha=0.2, label="5–95% band")
    ax.plot(days, result["mean_path"] * 100.0, color="#2c3e50", linewidth=2.5, label="Mean path")
    ax.axhline(result["long_run_mean_iv"] * 100.0, color="#e67e22", linestyle="--", label="Long-run mean IV")
    ax.set_title(
        f"{ticker.upper()} OU IV Mean Reversion | half-life {result['half_life_days']:.1f}d"
    )
    ax.set_xlabel("Projection day")
    ax.set_ylabel("Implied Volatility (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path
