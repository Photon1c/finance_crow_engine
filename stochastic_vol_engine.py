"""Stochastic volatility projection — Heston-like variance process."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_OUTPUT_DIR = Path("outputs/laser_falcon")


def simulate_stochastic_vol_paths(
    *,
    spot0: float,
    variance0: float,
    mu: float = 0.0,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.5,
    rho: float = -0.7,
    projection_days: int = 30,
    n_paths: int = 500,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """
    Euler-Maruyama Heston-like simulation:
        dS = mu*S*dt + sqrt(v)*S*dW1
        dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW2
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    steps = max(int(projection_days), 1)
    prices = np.zeros((n_paths, steps + 1), dtype=float)
    variances = np.zeros((n_paths, steps + 1), dtype=float)
    prices[:, 0] = spot0
    variances[:, 0] = max(variance0, 1e-6)

    sqrt_dt = np.sqrt(dt)
    for t in range(1, steps + 1):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        dw1 = z1 * sqrt_dt
        dw2 = (rho * z1 + np.sqrt(max(1.0 - rho ** 2, 0.0)) * z2) * sqrt_dt

        v_prev = np.clip(variances[:, t - 1], 1e-8, None)
        s_prev = prices[:, t - 1]
        v_next = v_prev + kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev) * dw2
        v_next = np.clip(v_next, 1e-8, None)
        s_next = s_prev + mu * s_prev * dt + np.sqrt(v_prev) * s_prev * dw1
        s_next = np.clip(s_next, 0.01, None)

        prices[:, t] = s_next
        variances[:, t] = v_next

    vol_paths = np.sqrt(variances)
    price_p05 = np.percentile(prices, 5, axis=0)
    price_p50 = np.percentile(prices, 50, axis=0)
    price_p95 = np.percentile(prices, 95, axis=0)
    vol_p05 = np.percentile(vol_paths, 5, axis=0)
    vol_p50 = np.percentile(vol_paths, 50, axis=0)
    vol_p95 = np.percentile(vol_paths, 95, axis=0)

    return {
        "spot0": spot0,
        "variance0": variance0,
        "mu": mu,
        "kappa": kappa,
        "theta": theta,
        "xi": xi,
        "rho": rho,
        "projection_days": projection_days,
        "n_paths": n_paths,
        "prices": prices,
        "variances": variances,
        "vol_paths": vol_paths,
        "price_p05": price_p05,
        "price_p50": price_p50,
        "price_p95": price_p95,
        "vol_p05": vol_p05,
        "vol_p50": vol_p50,
        "vol_p95": vol_p95,
        "terminal_price_p50": float(price_p50[-1]),
        "terminal_vol_p50": float(vol_p50[-1]),
    }


def plot_stochastic_vol_projection(
    result: dict[str, Any],
    *,
    ticker: str,
    output_path: Optional[Path] = None,
) -> Path:
    output_path = Path(output_path or DEFAULT_OUTPUT_DIR / f"{ticker.upper()}_stochastic_vol_projection.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    days = np.arange(result["price_p50"].shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    axes[0].fill_between(days, result["price_p05"], result["price_p95"], color="#3498db", alpha=0.2, label="5–95% price band")
    axes[0].plot(days, result["price_p50"], color="#2c3e50", linewidth=2, label="Median price")
    axes[0].set_ylabel("Price")
    axes[0].set_title(f"{ticker.upper()} Stochastic Volatility — Price Cone")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(days, result["vol_p05"] * 100.0, result["vol_p95"] * 100.0, color="#e67e22", alpha=0.2, label="5–95% vol band")
    axes[1].plot(days, result["vol_p50"] * 100.0, color="#c0392b", linewidth=2, label="Median vol")
    axes[1].set_xlabel("Projection day")
    axes[1].set_ylabel("Volatility (%)")
    axes[1].set_title("Volatility Cone")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path
