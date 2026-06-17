"""Laser Falcon data adapter — normalize stock and option chain CSVs for research replay.

Wraps data_loader.py; no live trading or broker APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from data_loader import (
    DEFAULT_OPTION_DIR,
    DEFAULT_STOCK_DIR,
    get_latest_stock_row,
    get_most_recent_option_date,
    load_option_chain_data,
    load_stock_data,
    parse_price,
)

CALL_COLUMNS = {
    "symbol": "Calls",
    "bid": "Bid",
    "ask": "Ask",
    "last": "Last Sale",
    "iv": "IV",
    "delta": "Delta",
    "gamma": "Gamma",
    "volume": "Volume",
    "open_interest": "Open Interest",
}

PUT_COLUMNS = {
    "symbol": "Puts",
    "bid": "Bid.1",
    "ask": "Ask.1",
    "last": "Last Sale.1",
    "iv": "IV.1",
    "delta": "Delta.1",
    "gamma": "Gamma.1",
    "volume": "Volume.1",
    "open_interest": "Open Interest.1",
}

NORMALIZED_OPTION_COLUMNS = (
    "option_type",
    "Expiration Date",
    "Strike",
    "symbol",
    "Bid",
    "Ask",
    "Last Sale",
    "IV",
    "Delta",
    "Gamma",
    "Volume",
    "Open Interest",
    "mid_price",
    "moneyness",
    "dte",
    "quote_spread_pct",
    "quote_unstable_flag",
)


def _parse_expiration(value: Any) -> Optional[pd.Timestamp]:
    if value in ("", None) or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def reference_date_from_chain(chain_date: Optional[str]) -> datetime:
    if not chain_date:
        return datetime.now()
    parsed = pd.to_datetime(chain_date.replace("_", "-"), errors="coerce")
    if pd.isna(parsed):
        return datetime.now()
    return parsed.to_pydatetime()


def normalize_stock_df(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize stock CSV to Date + Close columns."""
    df = stock_df.copy()
    if "Date" not in df.columns:
        raise ValueError("Stock CSV missing Date column")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    close_col = "Close/Last" if "Close/Last" in df.columns else "Close"
    if close_col not in df.columns:
        raise ValueError("Stock CSV missing Close/Last or Close column")
    df["Close"] = df[close_col].map(parse_price)
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
    return df[["Date", "Close"] + [c for c in df.columns if c not in ("Date", "Close", close_col)]].copy()


def _side_rows(
    raw_df: pd.DataFrame,
    *,
    option_type: str,
    side_cols: dict[str, str],
    spot: float,
    reference_date: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in raw_df.iterrows():
        strike = parse_price(row.get("Strike"))
        if np.isnan(strike) or strike <= 0:
            continue
        expiry = _parse_expiration(row.get("Expiration Date"))
        if expiry is None:
            continue
        bid = parse_price(row.get(side_cols["bid"]))
        ask = parse_price(row.get(side_cols["ask"]))
        last = parse_price(row.get(side_cols["last"])) if side_cols["last"] in row.index else float("nan")
        iv = parse_price(row.get(side_cols["iv"]))
        delta = parse_price(row.get(side_cols["delta"]))
        gamma = parse_price(row.get(side_cols["gamma"])) if side_cols["gamma"] in row.index else float("nan")
        volume = parse_price(row.get(side_cols["volume"]))
        oi = parse_price(row.get(side_cols["open_interest"]))
        if np.isnan(bid) and np.isnan(ask):
            continue
        mid = (bid + ask) / 2.0 if not np.isnan(bid) and not np.isnan(ask) else (ask if not np.isnan(ask) else bid)
        spread_pct = float("nan")
        if not np.isnan(bid) and not np.isnan(ask) and mid > 0:
            spread_pct = max((ask - bid) / mid, 0.0)
        dte = max((expiry.date() - reference_date.date()).days, 0)
        rows.append(
            {
                "option_type": option_type,
                "Expiration Date": str(row.get("Expiration Date", "")),
                "Strike": float(strike),
                "symbol": str(row.get(side_cols["symbol"], "") or ""),
                "Bid": float(bid) if not np.isnan(bid) else np.nan,
                "Ask": float(ask) if not np.isnan(ask) else np.nan,
                "Last Sale": float(last) if not np.isnan(last) else np.nan,
                "IV": float(iv) if not np.isnan(iv) else np.nan,
                "Delta": float(delta) if not np.isnan(delta) else np.nan,
                "Gamma": float(gamma) if not np.isnan(gamma) else np.nan,
                "Volume": float(volume) if not np.isnan(volume) else 0.0,
                "Open Interest": float(oi) if not np.isnan(oi) else 0.0,
                "mid_price": float(mid) if not np.isnan(mid) else np.nan,
                "moneyness": float(strike / max(spot, 1e-6)),
                "dte": int(dte),
                "quote_spread_pct": spread_pct,
                "quote_unstable_flag": bool(
                    (not np.isnan(spread_pct) and spread_pct > 0.25)
                    or (not np.isnan(bid) and bid <= 0 and ask > 0)
                ),
            }
        )
    return rows


def normalize_option_chain(
    option_df: pd.DataFrame,
    *,
    spot: float,
    reference_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Expand duplicate call/put columns into long normalized option chain."""
    ref = reference_date or datetime.now()
    raw = option_df.copy()
    if "Strike" not in raw.columns or "Expiration Date" not in raw.columns:
        raise ValueError("Option chain missing Strike or Expiration Date columns")

    call_rows = _side_rows(raw, option_type="call", side_cols=CALL_COLUMNS, spot=spot, reference_date=ref)
    put_rows = _side_rows(raw, option_type="put", side_cols=PUT_COLUMNS, spot=spot, reference_date=ref)
    normalized = pd.DataFrame(call_rows + put_rows)
    if normalized.empty:
        return pd.DataFrame(columns=list(NORMALIZED_OPTION_COLUMNS))
    return normalized[list(NORMALIZED_OPTION_COLUMNS)].sort_values(
        ["Expiration Date", "option_type", "Strike"]
    ).reset_index(drop=True)


@dataclass
class LaserFalconSnapshot:
    ticker: str
    spot: float
    chain_date: str
    reference_date: datetime
    stock_df: pd.DataFrame
    option_df_raw: pd.DataFrame
    option_df: pd.DataFrame
    data_health: dict[str, Any] = field(default_factory=dict)


def assess_data_health(option_df: pd.DataFrame, *, ticker: str) -> dict[str, Any]:
    """Summarize chain density and quality flags for graceful degradation."""
    if option_df.empty:
        return {
            "ticker": ticker.upper(),
            "status": "INSUFFICIENT",
            "n_contracts": 0,
            "n_expirations": 0,
            "n_strikes": 0,
            "iv_coverage_pct": 0.0,
            "quote_unstable_pct": 0.0,
            "warnings": ["No normalized option contracts available"],
        }

    expirations = option_df["Expiration Date"].nunique()
    strikes = option_df["Strike"].nunique()
    iv_ok = option_df["IV"].notna() & (option_df["IV"] > 0)
    unstable = option_df["quote_unstable_flag"].astype(bool)
    warnings: list[str] = []
    if expirations < 2:
        warnings.append("Sparse expirations — single-expiration skew mode")
    if strikes < 5:
        warnings.append("Low strike density — surface interpolation may be skipped")
    if iv_ok.mean() < 0.5:
        warnings.append("IV sparse — Black-Scholes inversion will be attempted")
    if unstable.mean() > 0.3:
        warnings.append("Wide bid/ask spreads — quote instability elevated")

    status = "OK"
    if expirations < 1 or strikes < 3:
        status = "INSUFFICIENT"
    elif expirations < 2 or strikes < 8:
        status = "SPARSE"

    return {
        "ticker": ticker.upper(),
        "status": status,
        "n_contracts": int(len(option_df)),
        "n_expirations": int(expirations),
        "n_strikes": int(strikes),
        "iv_coverage_pct": round(float(iv_ok.mean() * 100.0), 2),
        "quote_unstable_pct": round(float(unstable.mean() * 100.0), 2),
        "warnings": warnings,
    }


def load_laser_falcon_snapshot(
    ticker: str,
    *,
    chain_date: Optional[str] = None,
    stock_dir: str = DEFAULT_STOCK_DIR,
    option_dir: str = DEFAULT_OPTION_DIR,
) -> LaserFalconSnapshot:
    """Load synchronized stock + option chain snapshot for Laser Falcon analysis."""
    ticker_upper = ticker.upper()
    resolved_date = chain_date or get_most_recent_option_date(ticker_upper, base_dir=option_dir, verbose=False)
    stock_raw = load_stock_data(ticker_upper, base_dir=stock_dir)
    option_raw = load_option_chain_data(ticker_upper, date=resolved_date, base_dir=option_dir)
    spot, _ = get_latest_stock_row(ticker_upper, base_dir=stock_dir)
    reference = reference_date_from_chain(resolved_date)
    stock_df = normalize_stock_df(stock_raw)
    option_df = normalize_option_chain(option_raw, spot=spot, reference_date=reference)
    health = assess_data_health(option_df, ticker=ticker_upper)
    return LaserFalconSnapshot(
        ticker=ticker_upper,
        spot=float(spot),
        chain_date=resolved_date,
        reference_date=reference,
        stock_df=stock_df,
        option_df_raw=option_raw,
        option_df=option_df,
        data_health=health,
    )


def list_expirations(option_df: pd.DataFrame) -> list[str]:
    if option_df.empty:
        return []
    return sorted(option_df["Expiration Date"].dropna().unique().tolist())
