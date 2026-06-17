"""Synthetic option chain for Laser Falcon unit tests."""

from __future__ import annotations

import pandas as pd


def make_synthetic_option_chain(*, spot: float = 100.0, sparse: bool = False) -> pd.DataFrame:
    """Build raw wide-format chain matching data_loader expectations."""
    if sparse:
        expirations = ["2026-07-18"]
        strikes = [spot * 0.95, spot, spot * 1.05]
    else:
        expirations = ["2026-07-18", "2026-08-15", "2026-09-19"]
        strikes = [spot * x for x in [0.90, 0.95, 1.00, 1.05, 1.10]]

    rows = []
    for exp in expirations:
        for strike in strikes:
            iv_call = 0.22 + abs(strike / spot - 1.0) * 0.5
            iv_put = 0.24 + max(0, 1.0 - strike / spot) * 0.8
            rows.append(
                {
                    "Expiration Date": exp,
                    "Strike": strike,
                    "Calls": f"C{int(strike)}",
                    "Bid": max(spot - strike, 0.5) * 0.08,
                    "Ask": max(spot - strike, 0.5) * 0.09 + 0.05,
                    "Last Sale": 0.5,
                    "IV": iv_call,
                    "Delta": 0.5,
                    "Gamma": 0.02,
                    "Volume": 100,
                    "Open Interest": 500,
                    "Puts": f"P{int(strike)}",
                    "Bid.1": max(strike - spot, 0.5) * 0.08,
                    "Ask.1": max(strike - spot, 0.5) * 0.09 + 0.05,
                    "Last Sale.1": 0.5,
                    "IV.1": iv_put,
                    "Delta.1": -0.5,
                    "Gamma.1": 0.02,
                    "Volume.1": 80,
                    "Open Interest.1": 400,
                }
            )
    return pd.DataFrame(rows)


def make_synthetic_stock_df(rows: int = 30, spot: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=rows, freq="B")
    return pd.DataFrame({"Date": dates, "Close/Last": [spot] * rows})
