"""Black-Scholes pricing and implied volatility inversion for Laser Falcon."""

from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np

OptionKind = Literal["call", "put"]
DEFAULT_RISK_FREE_RATE = 0.045


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(
    *,
    spot: float,
    strike: float,
    time_years: float,
    volatility: float,
    option_type: OptionKind = "call",
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    if spot <= 0 or strike <= 0 or time_years <= 0 or volatility <= 0:
        return 0.0
    sqrt_t = math.sqrt(time_years)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_years) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discount = math.exp(-risk_free_rate * time_years)
    if option_type == "call":
        return spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
    return strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def implied_volatility(
    *,
    price: float,
    spot: float,
    strike: float,
    time_years: float,
    option_type: OptionKind = "call",
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    max_iterations: int = 60,
) -> Optional[float]:
    """Newton-Raphson IV solver with Brent-style bracket fallback."""
    if price <= 0 or spot <= 0 or strike <= 0 or time_years <= 0:
        return None

    intrinsic = max(0.0, spot - strike) if option_type == "call" else max(0.0, strike - spot)
    if price < intrinsic * 0.99:
        return None

    sigma = 0.30
    for _ in range(max_iterations):
        model = black_scholes_price(
            spot=spot,
            strike=strike,
            time_years=time_years,
            volatility=sigma,
            option_type=option_type,
            risk_free_rate=risk_free_rate,
        )
        diff = model - price
        if abs(diff) < 1e-6:
            return float(np.clip(sigma, 0.01, 5.0))

        sqrt_t = math.sqrt(time_years)
        d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * sigma ** 2) * time_years) / (sigma * sqrt_t)
        vega = spot * math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi) * sqrt_t
        if vega < 1e-8:
            break
        sigma -= diff / vega
        sigma = float(np.clip(sigma, 0.001, 5.0))

    # Brent bracket search fallback
    lo, hi = 0.001, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        model = black_scholes_price(
            spot=spot,
            strike=strike,
            time_years=time_years,
            volatility=mid,
            option_type=option_type,
            risk_free_rate=risk_free_rate,
        )
        if model > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-5:
            return float(mid)
    return None


def fill_missing_iv(
    option_df,
    *,
    spot: float,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
):
    """Return copy with IV filled from mid_price where vendor IV missing."""
    import pandas as pd

    df = option_df.copy()
    if df.empty:
        return df
    if "iv_source" not in df.columns:
        df["iv_source"] = np.where(df["IV"].notna() & (df["IV"] > 0), "vendor", "missing")

    for idx, row in df.iterrows():
        iv = row.get("IV")
        if not (pd.isna(iv) or float(iv) <= 0):
            continue
        mid = row.get("mid_price")
        dte = int(row.get("dte", 0))
        if pd.isna(mid) or float(mid) <= 0 or dte <= 0:
            continue
        solved = implied_volatility(
            price=float(mid),
            spot=spot,
            strike=float(row["Strike"]),
            time_years=max(dte / 365.0, 1 / 365.0),
            option_type=str(row.get("option_type", "call")),
            risk_free_rate=risk_free_rate,
        )
        if solved is not None:
            df.at[idx, "IV"] = solved
            df.at[idx, "iv_source"] = "solved"
    return df
