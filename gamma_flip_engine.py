"""Gamma flip computation with expiry-aware chain selection and explicit diagnostics."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from chain_integrity_engine import assess_chain_integrity, clean_chain_expirations
from data_loader import _reference_date_from_chain, _select_expiry, parse_price

EXPIRATION_COL = "Expiration Date"
DEFAULT_MAX_DTE = 30


def empty_gamma_snapshot() -> dict[str, Any]:
    """Safe null gamma snapshot when option chain is unavailable."""
    return {
        "gamma_flip_strike": None,
        "distance_to_flip_pct": None,
        "net_gamma_at_spot": None,
        "gamma_regime": "NO_CHAIN",
        "call_gamma_oi": 0.0,
        "put_gamma_oi": 0.0,
        "gamma_flip_method": "",
        "gamma_flip_reason": "no_option_chain",
        "gamma_expiry_used": "",
        "gamma_chain_strikes": 0,
        "gamma_chain_integrity": "",
    }


def _aggregate_gamma_by_strike(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["Strike"] = pd.to_numeric(work["Strike"], errors="coerce")
    work = work[work["Strike"].notna()].copy()
    if work.empty:
        return work

    for col in ("Gamma", "Gamma.1", "Open Interest", "Open Interest.1"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col].map(parse_price), errors="coerce").fillna(0.0)
        else:
            work[col] = 0.0

    work["call_gamma_oi"] = work["Gamma"] * work["Open Interest"]
    work["put_gamma_oi"] = work["Gamma.1"] * work["Open Interest.1"]
    grouped = (
        work.groupby("Strike", as_index=False)
        .agg(
            call_gamma_oi=("call_gamma_oi", "sum"),
            put_gamma_oi=("put_gamma_oi", "sum"),
        )
        .sort_values("Strike")
        .reset_index(drop=True)
    )
    if grouped.empty:
        return grouped
    grouped["net_gamma"] = grouped["call_gamma_oi"] - grouped["put_gamma_oi"]
    return grouped


def _flip_from_grouped(grouped: pd.DataFrame, spot: float) -> tuple[Optional[float], str]:
    if grouped.empty:
        return None, "empty_strike_ladder"

    strikes = grouped["Strike"].values
    net = grouped["net_gamma"].values

    for idx in range(1, len(strikes)):
        prev_net = float(net[idx - 1])
        curr_net = float(net[idx])
        if prev_net == 0.0:
            return float(strikes[idx - 1]), "zero_crossing"
        if prev_net * curr_net < 0:
            weight = abs(prev_net) / (abs(prev_net) + abs(curr_net) + 1e-9)
            flip = float(strikes[idx - 1] * (1 - weight) + strikes[idx] * weight)
            return flip, "interpolated_crossing"

    if len(strikes) >= 2 and net[0] * net[-1] > 0:
        slope = (float(net[-1]) - float(net[-2])) / (float(strikes[-1]) - float(strikes[-2]))
        if slope != 0.0:
            extrapolated = float(strikes[-1]) - float(net[-1]) / slope
            span = float(strikes[-1] - strikes[0])
            if strikes[0] - span <= extrapolated <= strikes[-1] + span:
                return extrapolated, "extrapolated_beyond_ladder"

    if len(strikes) >= 2 and net[0] * net[-1] > 0:
        return None, "uniform_net_gamma_sign"

    spot_idx = (grouped["Strike"] - spot).abs().idxmin()
    _ = float(grouped.loc[spot_idx, "net_gamma"])
    return None, "flip_not_bracketed"


def _gamma_regime_from_flip(
    flip_strike: Optional[float],
    spot: float,
    *,
    method: str,
) -> str:
    if flip_strike is None:
        return "FLIP_UNDEFINED"
    distance_pct = (spot - flip_strike) / spot * 100.0
    if method == "extrapolated_beyond_ladder":
        if abs(distance_pct) < 0.5:
            return "AT_PHASE_BOUNDARY_EXTRAPOLATED"
        if distance_pct > 0:
            return "ABOVE_FLIP_POSITIVE_GAMMA_EXTRAPOLATED"
        return "BELOW_FLIP_NEGATIVE_GAMMA_EXTRAPOLATED"
    if abs(distance_pct) < 0.5:
        return "AT_PHASE_BOUNDARY"
    if distance_pct > 0:
        return "ABOVE_FLIP_POSITIVE_GAMMA"
    return "BELOW_FLIP_NEGATIVE_GAMMA"


def _iter_expiry_slices(option_df: pd.DataFrame, spot: float, reference_date) -> list[tuple[str, pd.DataFrame]]:
    if EXPIRATION_COL not in option_df.columns:
        return [("", option_df.copy())]

    slices: list[tuple[str, pd.DataFrame, int]] = []
    for expiry in sorted(option_df[EXPIRATION_COL].dropna().astype(str).unique()):
        subset = option_df[option_df[EXPIRATION_COL].astype(str) == expiry].copy()
        if subset.empty:
            continue
        dte = 0
        parsed = pd.to_datetime(expiry, errors="coerce")
        if pd.notna(parsed):
            dte = max((parsed.date() - reference_date.date()).days, 0)
        slices.append((expiry, subset, dte))

    if not slices:
        return [("", option_df.copy())]

    slices.sort(key=lambda item: (abs(item[2] - 7), item[2]))
    return [(expiry, subset) for expiry, subset, _ in slices]


def _select_chain_for_gamma(
    option_df: pd.DataFrame,
    spot: float,
    *,
    chain_date: Optional[str],
    max_dte: int,
) -> tuple[pd.DataFrame, str, str]:
    reference_date = _reference_date_from_chain(chain_date)
    if EXPIRATION_COL not in option_df.columns:
        return option_df.copy(), "", "no_expiration_column"

    selected = _select_expiry(option_df, spot, reference_date, max_dte=max_dte)
    if not selected.empty and EXPIRATION_COL in selected.columns:
        expiry = str(selected[EXPIRATION_COL].iloc[0])
        grouped = _aggregate_gamma_by_strike(selected)
        flip, _ = _flip_from_grouped(grouped, spot)
        if flip is not None:
            return selected, expiry, "selected_expiry"

    for expiry, subset in _iter_expiry_slices(option_df, spot, reference_date):
        grouped = _aggregate_gamma_by_strike(subset)
        flip, _ = _flip_from_grouped(grouped, spot)
        if flip is not None:
            return subset, expiry, "scanned_expiry"

    if not selected.empty:
        expiry = str(selected[EXPIRATION_COL].iloc[0]) if EXPIRATION_COL in selected.columns else ""
        return selected, expiry, "selected_expiry_no_crossing"
    return option_df.copy(), "", "full_chain_fallback"


def compute_gamma_flip(
    option_df: Optional[pd.DataFrame],
    spot: float,
    *,
    chain_date: Optional[str] = None,
    max_dte: int = DEFAULT_MAX_DTE,
    ticker: str = "",
) -> dict[str, Any]:
    """Gamma flip from a single expiry slice; never silently drop failure reason."""
    if option_df is None or len(option_df) == 0:
        return empty_gamma_snapshot()
    if spot <= 0 or not np.isfinite(spot):
        snapshot = empty_gamma_snapshot()
        snapshot["gamma_regime"] = "INVALID_SPOT"
        snapshot["gamma_flip_reason"] = "invalid_spot"
        return snapshot

    cleaned, clean_diag = clean_chain_expirations(option_df)
    if EXPIRATION_COL in cleaned.columns:
        integrity = assess_chain_integrity(
            cleaned,
            ticker=ticker or "UNKNOWN",
            spot_price=spot,
        )
    else:
        integrity = {
            "status": "DEGRADED",
            "warnings": ["Missing Expiration Date column"],
        }
    if cleaned.empty:
        snapshot = empty_gamma_snapshot()
        snapshot["gamma_regime"] = "CHAIN_EMPTY_AFTER_CLEAN"
        snapshot["gamma_flip_reason"] = "all_rows_removed_by_expiration_cleaning"
        snapshot["gamma_chain_integrity"] = integrity.get("status", "")
        snapshot["gamma_cleaning"] = clean_diag
        snapshot["gamma_chain_integrity_detail"] = integrity
        return snapshot

    if "Strike" not in cleaned.columns:
        snapshot = empty_gamma_snapshot()
        snapshot["gamma_flip_reason"] = "missing_strike_column"
        snapshot["gamma_chain_integrity"] = integrity.get("status", "")
        return snapshot

    chain_slice, expiry_used, selection_method = _select_chain_for_gamma(
        cleaned,
        spot,
        chain_date=chain_date,
        max_dte=max_dte,
    )
    grouped = _aggregate_gamma_by_strike(chain_slice)
    if grouped.empty:
        snapshot = empty_gamma_snapshot()
        snapshot["gamma_flip_reason"] = "no_valid_strikes"
        snapshot["gamma_chain_integrity"] = integrity.get("status", "")
        snapshot["gamma_cleaning"] = clean_diag
        return snapshot

    total_oi = float(grouped["call_gamma_oi"].sum() + grouped["put_gamma_oi"].sum())
    if total_oi <= 0:
        snapshot = empty_gamma_snapshot()
        snapshot["gamma_regime"] = "ZERO_OI"
        snapshot["gamma_flip_reason"] = "zero_open_interest"
        snapshot["gamma_expiry_used"] = expiry_used
        snapshot["gamma_chain_strikes"] = int(len(grouped))
        snapshot["gamma_chain_integrity"] = integrity.get("status", "")
        return snapshot

    flip_strike, flip_method = _flip_from_grouped(grouped, spot)
    spot_idx = (grouped["Strike"] - spot).abs().idxmin()
    net_at_spot = float(grouped.loc[spot_idx, "net_gamma"])
    distance_pct: Optional[float] = None
    if flip_strike is not None:
        distance_pct = (spot - flip_strike) / spot * 100.0

    regime = _gamma_regime_from_flip(flip_strike, spot, method=flip_method)
    reason = flip_method if flip_strike is not None else flip_method
    if flip_strike is None and integrity.get("status") == "INVALID":
        reason = f"{reason};chain_integrity_invalid"

    return {
        "gamma_flip_strike": flip_strike,
        "distance_to_flip_pct": distance_pct,
        "net_gamma_at_spot": net_at_spot,
        "gamma_regime": regime,
        "call_gamma_oi": float(grouped["call_gamma_oi"].sum()),
        "put_gamma_oi": float(grouped["put_gamma_oi"].sum()),
        "gamma_flip_method": flip_method if flip_strike else selection_method,
        "gamma_flip_reason": reason,
        "gamma_expiry_used": expiry_used,
        "gamma_chain_strikes": int(len(grouped)),
        "gamma_chain_integrity": integrity.get("status", ""),
        "gamma_cleaning": clean_diag,
        "gamma_chain_integrity_detail": integrity,
    }
