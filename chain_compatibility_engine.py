"""Temporal snapshot compatibility — detect Contract Universe Drift (CUD)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from chain_integrity_engine import assess_chain_integrity, clean_chain_expirations

CompatibilityStatus = ("COMPATIBLE", "DEGRADED", "INVALID")
EXPIRATION_COL = "Expiration Date"
STRIKE_COL = "Strike"


def _overlap_score(shared: int, a: int, b: int) -> float:
    if shared <= 0:
        return 0.0
    if a <= 0 or b <= 0:
        return 0.0
    return float(shared / max(min(a, b), 1))


def _contract_set(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    if "option_type" in df.columns:
        keys = (
            df[EXPIRATION_COL].astype(str)
            + "|"
            + df[STRIKE_COL].astype(str)
            + "|"
            + df["option_type"].astype(str)
        )
    else:
        keys = df[EXPIRATION_COL].astype(str) + "|" + df[STRIKE_COL].astype(str)
    return set(keys.tolist())


def _moneyness_band_strikes(df: pd.DataFrame, spot: Optional[float], band: tuple[float, float]) -> set[float]:
    if df.empty or spot is None or spot <= 0:
        return set()
    low, high = band
    strikes = pd.to_numeric(df[STRIKE_COL], errors="coerce")
    mask = (strikes >= spot * low) & (strikes <= spot * high)
    return set(strikes[mask].dropna().astype(float).tolist())


def assess_snapshot_compatibility(
    prior_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    ticker: Optional[str] = None,
    spot_prior: Optional[float] = None,
    spot_current: Optional[float] = None,
    min_expiration_overlap: float = 0.50,
    min_strike_overlap: float = 0.50,
    min_contract_overlap: float = 0.40,
    moneyness_band: tuple[float, float] = (0.85, 1.15),
) -> dict[str, Any]:
    """Compare two snapshots for temporal comparability."""
    prior_clean, prior_clean_diag = clean_chain_expirations(prior_df)
    current_clean, current_clean_diag = clean_chain_expirations(current_df)

    prior_integrity = assess_chain_integrity(prior_clean, ticker=ticker, spot_price=spot_prior)
    current_integrity = assess_chain_integrity(current_clean, ticker=ticker, spot_price=spot_current)

    prior_exp = set(prior_clean[EXPIRATION_COL].astype(str).unique()) if not prior_clean.empty else set()
    current_exp = set(current_clean[EXPIRATION_COL].astype(str).unique()) if not current_clean.empty else set()
    shared_exp = prior_exp & current_exp

    prior_strikes = set(pd.to_numeric(prior_clean[STRIKE_COL], errors="coerce").dropna().astype(float).tolist()) if not prior_clean.empty else set()
    current_strikes = set(pd.to_numeric(current_clean[STRIKE_COL], errors="coerce").dropna().astype(float).tolist()) if not current_clean.empty else set()
    shared_strikes = prior_strikes & current_strikes

    prior_contracts = _contract_set(prior_clean)
    current_contracts = _contract_set(current_clean)
    shared_contracts = prior_contracts & current_contracts

    prior_money = _moneyness_band_strikes(prior_clean, spot_prior, moneyness_band)
    current_money = _moneyness_band_strikes(current_clean, spot_current, moneyness_band)
    shared_money = prior_money & current_money

    expiration_overlap = _overlap_score(len(shared_exp), len(prior_exp), len(current_exp))
    strike_overlap = _overlap_score(len(shared_strikes), len(prior_strikes), len(current_strikes))
    contract_overlap = _overlap_score(len(shared_contracts), len(prior_contracts), len(current_contracts))
    moneyness_overlap = _overlap_score(len(shared_money), len(prior_money), len(current_money))

    blank_detected = (
        prior_clean_diag.get("blank_expiration_rows", 0) > 0
        or current_clean_diag.get("blank_expiration_rows", 0) > 0
        or prior_clean_diag.get("malformed_expiration_rows", 0) > 0
        or current_clean_diag.get("malformed_expiration_rows", 0) > 0
    )
    broker_omission = blank_detected or (
        current_integrity.get("blank_expiration_ratio", 0) > 0.05
        or prior_integrity.get("blank_expiration_ratio", 0) > 0.05
    )
    expiration_rolloff = len(prior_exp) >= 2 and len(current_exp) < len(prior_exp) and expiration_overlap < 0.5
    contract_universe_drift = (
        expiration_overlap < 0.30
        or strike_overlap < 0.25
        or contract_overlap < 0.20
        or (len(prior_exp) >= 3 and len(current_exp) == 1)
        or expiration_rolloff
    )

    health_component = min(prior_integrity.get("chain_health_score", 0), current_integrity.get("chain_health_score", 0))
    compatibility_score = float(np.clip(
        expiration_overlap * 0.35
        + strike_overlap * 0.25
        + contract_overlap * 0.20
        + moneyness_overlap * 0.10
        + health_component * 0.10,
        0.0,
        1.0,
    ))

    warnings: list[str] = []
    errors: list[str] = []

    if not shared_exp:
        errors.append("No shared expirations between snapshots")
    if expiration_rolloff:
        warnings.append("Expiration rolloff detected between snapshots")
    if broker_omission:
        warnings.append("Blank or malformed expiration rows detected (broker omission)")
    if contract_universe_drift:
        errors.append("Contract universe drift detected")
    if expiration_overlap < min_expiration_overlap:
        warnings.append(f"Expiration overlap {expiration_overlap:.2%} below preferred {min_expiration_overlap:.0%}")
    if strike_overlap < min_strike_overlap:
        warnings.append(f"Strike overlap {strike_overlap:.2%} below preferred {min_strike_overlap:.0%}")

    status = "COMPATIBLE"
    if (
        not shared_exp
        or expiration_overlap < 0.30
        or strike_overlap < 0.25
        or contract_overlap < 0.20
        or (broker_omission and expiration_overlap < 0.40)
        or (len(prior_exp) >= 3 and len(current_exp) == 1)
    ):
        status = "INVALID"
    elif (
        expiration_overlap < min_expiration_overlap
        or strike_overlap < min_strike_overlap
        or contract_overlap < min_contract_overlap
        or broker_omission
        or prior_integrity.get("status") == "INVALID"
        or current_integrity.get("status") == "INVALID"
        or prior_integrity.get("chain_health_score", 0) < 0.60
        or current_integrity.get("chain_health_score", 0) < 0.60
    ):
        status = "DEGRADED"

    return {
        "status": status,
        "compatibility_score": round(compatibility_score, 4),
        "ticker": (ticker or "").upper(),
        "prior_expiration_count": len(prior_exp),
        "current_expiration_count": len(current_exp),
        "shared_expiration_count": len(shared_exp),
        "shared_expirations": sorted(shared_exp),
        "expiration_overlap_score": round(expiration_overlap, 4),
        "prior_strike_count": len(prior_strikes),
        "current_strike_count": len(current_strikes),
        "shared_strike_count": len(shared_strikes),
        "strike_overlap_score": round(strike_overlap, 4),
        "contract_overlap_score": round(contract_overlap, 4),
        "moneyness_overlap_score": round(moneyness_overlap, 4),
        "contract_universe_drift_flag": bool(contract_universe_drift),
        "expiration_rolloff_flag": bool(expiration_rolloff),
        "broker_omission_flag": bool(broker_omission),
        "blank_expiration_detected": bool(blank_detected),
        "prior_chain_health_score": prior_integrity.get("chain_health_score"),
        "current_chain_health_score": current_integrity.get("chain_health_score"),
        "warnings": warnings,
        "errors": errors,
        "prior_integrity": prior_integrity,
        "current_integrity": current_integrity,
    }


def validate_snapshot_compatibility(
    prior_df: pd.DataFrame,
    current_df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper with valid bool and reason."""
    diagnostics = assess_snapshot_compatibility(prior_df, current_df, **kwargs)
    status = diagnostics["status"]
    valid = status == "COMPATIBLE"
    reason = "Snapshots are comparable"
    if status == "DEGRADED":
        valid = False
        reason = "Comparable with reduced confidence: " + "; ".join(diagnostics.get("warnings") or ["degraded overlap"])
    if status == "INVALID":
        valid = False
        reason = diagnostics.get("errors", ["Contract universe drift detected"])[0]
    return {
        "valid": valid,
        "status": status,
        "reason": reason,
        "diagnostics": diagnostics,
    }
