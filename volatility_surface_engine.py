"""3D implied volatility surface builder with sparse-data tolerance."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from chain_integrity_engine import validate_chain_for_analysis
from implied_vol_solver import fill_missing_iv

DEFAULT_OUTPUT_DIR = Path("outputs/laser_falcon")
MIN_SURFACE_POINTS = 12
MIN_STRIKES = 5
MIN_EXPIRATIONS = 2


def _prepare_surface_points(option_df: pd.DataFrame, *, spot: float) -> pd.DataFrame:
    df = fill_missing_iv(option_df, spot=spot)
    df = df[df["IV"].notna() & (df["IV"] > 0)].copy()
    if df.empty:
        return df
    df["iv_pct"] = df["IV"] * 100.0 if df["IV"].max() <= 3.0 else df["IV"]
    df["moneyness"] = df["Strike"] / max(spot, 1e-6)
    return df


def assess_surface_density(option_df: pd.DataFrame) -> dict[str, Any]:
    if option_df.empty:
        return {"sufficient": False, "reason": "empty chain", "n_points": 0}
    n_exp = option_df["Expiration Date"].nunique()
    n_strikes = option_df["Strike"].nunique()
    n_points = len(option_df)
    if n_points < MIN_SURFACE_POINTS or n_strikes < MIN_STRIKES:
        return {
            "sufficient": False,
            "reason": "insufficient strike density",
            "n_points": n_points,
            "n_expirations": int(n_exp),
            "n_strikes": int(n_strikes),
        }
    if n_exp < MIN_EXPIRATIONS:
        return {
            "sufficient": False,
            "reason": "single expiration only — 2D skew preferred",
            "n_points": n_points,
            "n_expirations": int(n_exp),
            "n_strikes": int(n_strikes),
        }
    return {
        "sufficient": True,
        "reason": "ok",
        "n_points": n_points,
        "n_expirations": int(n_exp),
        "n_strikes": int(n_strikes),
    }


def build_iv_surface_grid(
    option_df: pd.DataFrame,
    *,
    spot: float,
    use_moneyness: bool = True,
    ticker: Optional[str] = None,
) -> dict[str, Any]:
    """Interpolate IV onto a regular grid; tolerates thin chains."""
    validation = validate_chain_for_analysis(option_df, "surface", ticker=ticker, spot_price=spot)
    if not validation["valid"]:
        return {
            "status": "SKIPPED",
            "reason": validation["reason"],
            "chain_integrity": validation["integrity"],
            "density": {"sufficient": False, "reason": validation["reason"]},
            "grid": None,
        }

    df = _prepare_surface_points(option_df, spot=spot)
    density = assess_surface_density(df)
    if not density["sufficient"] or df.empty:
        return {"status": "SKIPPED", "density": density, "grid": None}

    x_col = "moneyness" if use_moneyness else "Strike"
    x = df[x_col].to_numpy(dtype=float)
    y = df["dte"].to_numpy(dtype=float)
    z = df["iv_pct"].to_numpy(dtype=float)

    x_grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 25)
    y_grid = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), 20)
    X, Y = np.meshgrid(x_grid, y_grid)

    try:
        from scipy.interpolate import griddata

        Z = griddata((x, y), z, (X, Y), method="linear")
        nan_mask = np.isnan(Z)
        if nan_mask.any():
            Z_nearest = griddata((x, y), z, (X, Y), method="nearest")
            Z[nan_mask] = Z_nearest[nan_mask]
    except Exception:
        # Fallback: simple averaging onto grid cells
        Z = np.full(X.shape, np.nanmean(z))
        for i, yv in enumerate(y_grid):
            for j, xv in enumerate(x_grid):
                mask = (np.abs(y - yv) <= max(np.ptp(y) / len(y_grid), 1)) & (
                    np.abs(x - xv) <= max(np.ptp(x) / len(x_grid), 0.01)
                )
                if mask.any():
                    Z[i, j] = float(np.nanmean(z[mask]))

    return {
        "status": "OK",
        "density": density,
        "grid": {
            "x_label": "Moneyness" if use_moneyness else "Strike",
            "x": x_grid.tolist(),
            "y_label": "DTE",
            "y": y_grid.tolist(),
            "z": Z.tolist(),
        },
    }


def plot_iv_surface(
    option_df: pd.DataFrame,
    *,
    ticker: str,
    spot: float,
    output_path: Optional[Path] = None,
    use_moneyness: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Save 3D IV surface plot or insufficient-density notice."""
    output_path = Path(output_path or DEFAULT_OUTPUT_DIR / f"{ticker.upper()}_iv_surface.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    surface = build_iv_surface_grid(option_df, spot=spot, use_moneyness=use_moneyness, ticker=ticker)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    if surface["status"] != "OK" or surface["grid"] is None:
        ax.text2D(
            0.5,
            0.5,
            f"Surface skipped: {surface['density'].get('reason', 'unknown')}",
            transform=ax.transAxes,
            ha="center",
        )
        ax.set_title(f"{ticker.upper()} IV Surface — insufficient density")
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return output_path, surface

    grid = surface["grid"]
    X, Y = np.meshgrid(grid["x"], grid["y"])  # numpy meshgrid (no torch dependency)
    Z = np.array(grid["z"])
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9, edgecolor="none")
    ax.set_xlabel(grid["x_label"])
    ax.set_ylabel(grid["y_label"])
    ax.set_zlabel("IV (%)")
    ax.set_title(f"{ticker.upper()} IV Surface")
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path, surface
