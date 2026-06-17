"""Option chain structural integrity diagnostics for Laser Falcon.

CSV replay only. Does not modify sacred ontology.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

AnalysisType = Literal["skew", "surface", "temporal", "pressure"]
ChainStatus = Literal["HEALTHY", "DEGRADED", "INVALID"]

EXPIRATION_COL = "Expiration Date"
STRIKE_COL = "Strike"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _is_normalized(chain_df: pd.DataFrame) -> bool:
    return "option_type" in chain_df.columns


def _blank_expiration_mask(series: pd.Series) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(dtype=bool)
    as_str = series.astype(str).str.strip()
    return series.isna() | as_str.eq("") | as_str.str.lower().isin({"nan", "none", "nat", "null"})


def _malformed_expiration_mask(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    blank = _blank_expiration_mask(series)
    return (~blank) & parsed.isna()


def clean_chain_expirations(chain_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Remove blank/malformed expiration rows and report diagnostics.
    Does not silently coerce bad values.
    """
    df = chain_df.copy()
    diagnostics: dict[str, Any] = {
        "rows_removed": 0,
        "blank_expiration_rows": 0,
        "malformed_expiration_rows": 0,
        "warnings": [],
    }
    if df.empty or EXPIRATION_COL not in df.columns:
        diagnostics["warnings"].append("Missing Expiration Date column")
        return df, diagnostics

    blank = _blank_expiration_mask(df[EXPIRATION_COL])
    malformed = _malformed_expiration_mask(df[EXPIRATION_COL])
    bad = blank | malformed
    diagnostics["blank_expiration_rows"] = int(blank.sum())
    diagnostics["malformed_expiration_rows"] = int(malformed.sum())
    diagnostics["rows_removed"] = int(bad.sum())

    if diagnostics["blank_expiration_rows"] > 0:
        diagnostics["warnings"].append(
            f"{diagnostics['blank_expiration_rows']} rows with blank expiration removed"
        )
    if diagnostics["malformed_expiration_rows"] > 0:
        diagnostics["warnings"].append(
            f"{diagnostics['malformed_expiration_rows']} rows with malformed expiration removed"
        )

    cleaned = df.loc[~bad].copy()
    if not cleaned.empty:
        cleaned[EXPIRATION_COL] = cleaned[EXPIRATION_COL].astype(str)
    return cleaned, diagnostics


def _contract_keys(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str)
    if _is_normalized(df):
        return (
            df[EXPIRATION_COL].astype(str)
            + "|"
            + df[STRIKE_COL].astype(str)
            + "|"
            + df["option_type"].astype(str)
        )
    return df[EXPIRATION_COL].astype(str) + "|" + df[STRIKE_COL].astype(str)


def _missing_ratio(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 1.0
    values = pd.to_numeric(series, errors="coerce") if series.dtype != object else series
    if values.dtype == object:
        missing = values.isna() | values.astype(str).str.strip().eq("")
    else:
        missing = values.isna() | (values <= 0)
    return _safe_ratio(missing.sum(), len(series))


def _wide_spread_ratio(df: pd.DataFrame, *, threshold_pct: float) -> float:
    if df.empty:
        return 0.0
    if "quote_unstable_flag" in df.columns:
        return _safe_ratio(df["quote_unstable_flag"].astype(bool).sum(), len(df))
    bid_col = "Bid" if "Bid" in df.columns else None
    ask_col = "Ask" if "Ask" in df.columns else None
    if not bid_col or not ask_col:
        return 0.0
    bid = pd.to_numeric(df[bid_col], errors="coerce")
    ask = pd.to_numeric(df[ask_col], errors="coerce")
    mid = (bid + ask) / 2.0
    spread = (ask - bid) / mid.replace(0, np.nan)
    wide = spread > threshold_pct
    return _safe_ratio(wide.fillna(False).sum(), len(df))


def _strikes_by_expiration(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or EXPIRATION_COL not in df.columns:
        return {}
    grouped = df.groupby(EXPIRATION_COL)[STRIKE_COL].nunique()
    return {str(k): int(v) for k, v in grouped.items()}


def _compute_health_score(components: dict[str, float], penalties: dict[str, float]) -> float:
    score = (
        components.get("expiration_coverage", 0.0) * 0.20
        + components.get("strike_density", 0.0) * 0.20
        + components.get("iv_completeness", 0.0) * 0.20
        + components.get("quote_sanity", 0.0) * 0.15
        + components.get("oi_volume_availability", 0.0) * 0.10
        + components.get("gamma_availability", 0.0) * 0.05
    )
    score -= penalties.get("duplicate_penalty", 0.0) * 0.10
    score -= penalties.get("malformed_penalty", 0.0) * 0.10
    return float(np.clip(score, 0.0, 1.0))


def _status_from_score(score: float) -> ChainStatus:
    if score >= 0.85:
        return "HEALTHY"
    if score >= 0.60:
        return "DEGRADED"
    return "INVALID"


def assess_chain_integrity(
    chain_df: pd.DataFrame,
    *,
    ticker: Optional[str] = None,
    spot_price: Optional[float] = None,
    min_expirations: int = 2,
    min_strikes_per_expiration: int = 5,
    max_missing_iv_ratio: float = 0.35,
    max_blank_expiration_ratio: float = 0.05,
    max_wide_spread_ratio: float = 0.40,
    wide_spread_threshold_pct: float = 0.25,
) -> dict[str, Any]:
    """Validate a single option chain snapshot and return structured diagnostics."""
    raw_count = len(chain_df)
    cleaned, clean_diag = clean_chain_expirations(chain_df)
    warnings: list[str] = list(clean_diag.get("warnings", []))
    errors: list[str] = []

    if cleaned.empty:
        return {
            "status": "INVALID",
            "chain_health_score": 0.0,
            "ticker": (ticker or "").upper(),
            "row_count": raw_count,
            "valid_row_count": 0,
            "expiration_count": 0,
            "blank_expiration_count": clean_diag.get("blank_expiration_rows", 0),
            "blank_expiration_ratio": 1.0 if raw_count else 0.0,
            "malformed_expiration_count": clean_diag.get("malformed_expiration_rows", 0),
            "strikes_total": 0,
            "strikes_by_expiration": {},
            "min_strikes_per_expiration": 0,
            "missing_iv_ratio": 1.0,
            "missing_bid_ask_ratio": 1.0,
            "missing_gamma_ratio": 1.0,
            "missing_open_interest_ratio": 1.0,
            "duplicate_contract_count": 0,
            "wide_spread_ratio": 0.0,
            "zero_volume_ratio": 1.0,
            "zero_open_interest_ratio": 1.0,
            "warnings": warnings,
            "errors": ["No valid expiration rows after cleaning"],
            "cleaning": clean_diag,
        }

    iv_col = "IV" if "IV" in cleaned.columns else None
    gamma_col = "Gamma" if "Gamma" in cleaned.columns else None
    oi_col = "Open Interest" if "Open Interest" in cleaned.columns else None
    vol_col = "Volume" if "Volume" in cleaned.columns else None

    expiration_count = int(cleaned[EXPIRATION_COL].nunique())
    strikes_by_exp = _strikes_by_expiration(cleaned)
    min_strikes = int(min(strikes_by_exp.values())) if strikes_by_exp else 0
    strikes_total = int(cleaned[STRIKE_COL].nunique())

    blank_ratio = _safe_ratio(clean_diag.get("blank_expiration_rows", 0) + clean_diag.get("malformed_expiration_rows", 0), max(raw_count, 1))
    missing_iv = _missing_ratio(cleaned[iv_col]) if iv_col else 1.0
    missing_bid_ask = 0.0
    if "Bid" in cleaned.columns and "Ask" in cleaned.columns:
        missing_bid_ask = max(_missing_ratio(cleaned["Bid"]), _missing_ratio(cleaned["Ask"]))
    missing_gamma = _missing_ratio(cleaned[gamma_col]) if gamma_col else 1.0
    missing_oi = _missing_ratio(cleaned[oi_col]) if oi_col else 1.0
    wide_spread = _wide_spread_ratio(cleaned, threshold_pct=wide_spread_threshold_pct)

    keys = _contract_keys(cleaned)
    duplicate_count = int(len(keys) - keys.nunique())

    zero_volume_ratio = 0.0
    if vol_col:
        vol = pd.to_numeric(cleaned[vol_col], errors="coerce").fillna(0)
        zero_volume_ratio = _safe_ratio((vol <= 0).sum(), len(cleaned))

    zero_oi_ratio = 0.0
    if oi_col:
        oi = pd.to_numeric(cleaned[oi_col], errors="coerce").fillna(0)
        zero_oi_ratio = _safe_ratio((oi <= 0).sum(), len(cleaned))

    exp_coverage = min(1.0, expiration_count / max(min_expirations, 1))
    strike_density = min(1.0, min_strikes / max(min_strikes_per_expiration, 1))
    iv_completeness = max(0.0, 1.0 - missing_iv)
    quote_sanity = max(0.0, 1.0 - max(missing_bid_ask, wide_spread))
    oi_volume_availability = max(0.0, 1.0 - (zero_oi_ratio * 0.6 + zero_volume_ratio * 0.4))
    gamma_availability = max(0.0, 1.0 - missing_gamma)

    duplicate_penalty = min(1.0, duplicate_count / max(len(cleaned), 1) * 5.0)
    malformed_penalty = min(1.0, blank_ratio / max(max_blank_expiration_ratio, 0.01))

    health_score = _compute_health_score(
        {
            "expiration_coverage": exp_coverage,
            "strike_density": strike_density,
            "iv_completeness": iv_completeness,
            "quote_sanity": quote_sanity,
            "oi_volume_availability": oi_volume_availability,
            "gamma_availability": gamma_availability,
        },
        {"duplicate_penalty": duplicate_penalty, "malformed_penalty": malformed_penalty},
    )
    status = _status_from_score(health_score)

    if expiration_count < min_expirations:
        warnings.append(f"Only {expiration_count} valid expiration(s); surface/temporal may be limited")
    if min_strikes < min_strikes_per_expiration:
        warnings.append(f"Minimum strikes per expiration is {min_strikes}")
    if missing_iv > max_missing_iv_ratio:
        warnings.append(f"Missing IV ratio {missing_iv:.2%} exceeds threshold")
    if blank_ratio > max_blank_expiration_ratio:
        warnings.append(f"Blank/malformed expiration ratio {blank_ratio:.2%}")
    if wide_spread > max_wide_spread_ratio:
        warnings.append(f"Wide bid/ask spread ratio {wide_spread:.2%}")
    if duplicate_count > 0:
        warnings.append(f"{duplicate_count} duplicate contract keys detected")
    if status == "INVALID":
        errors.append("Chain health score below minimum for reliable analysis")

    return {
        "status": status,
        "chain_health_score": round(health_score, 4),
        "ticker": (ticker or "").upper(),
        "spot_price": spot_price,
        "row_count": raw_count,
        "valid_row_count": int(len(cleaned)),
        "expiration_count": expiration_count,
        "blank_expiration_count": int(clean_diag.get("blank_expiration_rows", 0)),
        "blank_expiration_ratio": round(blank_ratio, 4),
        "malformed_expiration_count": int(clean_diag.get("malformed_expiration_rows", 0)),
        "strikes_total": strikes_total,
        "strikes_by_expiration": strikes_by_exp,
        "min_strikes_per_expiration": min_strikes,
        "missing_iv_ratio": round(missing_iv, 4),
        "missing_bid_ask_ratio": round(missing_bid_ask, 4),
        "missing_gamma_ratio": round(missing_gamma, 4),
        "missing_open_interest_ratio": round(missing_oi, 4),
        "duplicate_contract_count": duplicate_count,
        "wide_spread_ratio": round(wide_spread, 4),
        "zero_volume_ratio": round(zero_volume_ratio, 4),
        "zero_open_interest_ratio": round(zero_oi_ratio, 4),
        "warnings": warnings,
        "errors": errors,
        "cleaning": clean_diag,
    }


def validate_chain_for_analysis(
    chain_df: pd.DataFrame,
    analysis_type: AnalysisType = "skew",
    *,
    ticker: Optional[str] = None,
    spot_price: Optional[float] = None,
) -> dict[str, Any]:
    """Return whether chain is suitable for a specific analysis type."""
    integrity = assess_chain_integrity(chain_df, ticker=ticker, spot_price=spot_price)
    suitable = True
    reason = "ok"

    if analysis_type == "skew":
        suitable = integrity["valid_row_count"] > 0 and integrity["expiration_count"] >= 1
        if not suitable:
            reason = "No valid expiration rows for skew"
    elif analysis_type == "surface":
        suitable = (
            integrity["status"] != "INVALID"
            and integrity["expiration_count"] >= 2
            and integrity["min_strikes_per_expiration"] >= 5
        )
        if integrity["status"] == "INVALID":
            reason = "Chain integrity INVALID for surface"
        elif integrity["expiration_count"] < 2:
            reason = "Surface requires at least two expirations"
        elif integrity["min_strikes_per_expiration"] < 5:
            reason = "Insufficient strike density for surface"
    elif analysis_type == "temporal":
        suitable = integrity["status"] != "INVALID" and integrity["expiration_count"] >= 1
        if not suitable:
            reason = "Chain integrity INVALID for temporal comparison"
    elif analysis_type == "pressure":
        suitable = integrity["valid_row_count"] > 0
        if integrity["status"] == "INVALID":
            reason = "Chain integrity INVALID; pressure confidence reduced"

    return {
        "valid": suitable,
        "analysis_type": analysis_type,
        "reason": reason,
        "integrity": integrity,
    }


def data_quality_confidence(integrity: dict[str, Any]) -> float:
    """Map chain health to analysis confidence multiplier."""
    score = integrity.get("chain_health_score", 0.0)
    status = integrity.get("status", "INVALID")
    if status == "HEALTHY":
        return float(np.clip(score, 0.85, 1.0))
    if status == "DEGRADED":
        return float(np.clip(score, 0.45, 0.84))
    return float(np.clip(score * 0.5, 0.0, 0.44))


def json_safe_integrity(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure diagnostics serialize without NaN/inf."""
    def _clean(obj: Any) -> Any:
        if isinstance(obj, float):
            return None if not math.isfinite(obj) else obj
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    return _clean(payload)
