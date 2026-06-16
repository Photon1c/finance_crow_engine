"""Road-condition regime classifier for underlying price paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from data_loader import load_stock_data
from trade_trajectory import TradeTrajectory

REGIMES = (
    "cruise_descent",
    "rupture",
    "chop",
    "compression",
    "reversal",
    "unknown",
)


@dataclass
class RegimeFeatures:
    """Inputs for regime classification."""

    underlying_return: float
    realized_volatility: float
    iv_change: float
    cvd: Optional[float] = None
    volume_change: Optional[float] = None


def _realized_volatility(closes: pd.Series) -> float:
    if len(closes) < 2:
        return 0.0
    returns = closes.pct_change().dropna()
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(252))


def _volume_change(stock_df: pd.DataFrame) -> Optional[float]:
    if "Volume" not in stock_df.columns or len(stock_df) < 2:
        return None
    vol = pd.to_numeric(stock_df["Volume"], errors="coerce").dropna()
    if len(vol) < 2:
        return None
    recent = vol.iloc[-5:].mean() if len(vol) >= 5 else vol.iloc[-1]
    prior = vol.iloc[-10:-5].mean() if len(vol) >= 10 else vol.iloc[0]
    if prior == 0:
        return None
    return float((recent - prior) / prior)


def features_from_trajectory(trade: TradeTrajectory) -> RegimeFeatures:
    """Derive regime features from a completed trade."""
    return RegimeFeatures(
        underlying_return=trade.underlying_move_pct,
        realized_volatility=abs(trade.underlying_move_pct) * np.sqrt(252 / max(trade.dte, 1)),
        iv_change=trade.iv_change,
        cvd=(trade.cvd_exit - trade.cvd_entry) if trade.cvd_entry is not None and trade.cvd_exit is not None else None,
        volume_change=trade.volume_ratio,
    )


def features_from_stock_history(
    ticker: str,
    lookback_days: int = 20,
    iv_change: float = 0.0,
    base_dir: Optional[str] = None,
) -> RegimeFeatures:
    """Build regime features from historical stock CSV via data_loader."""
    kwargs = {"base_dir": base_dir} if base_dir else {}
    df = load_stock_data(ticker, **kwargs)
    col = "Close/Last" if "Close/Last" in df.columns else "Close"
    closes = df[col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    closes = pd.to_numeric(closes, errors="coerce").dropna()
    window = closes.tail(lookback_days)
    if len(window) < 2:
        return RegimeFeatures(
            underlying_return=0.0,
            realized_volatility=0.0,
            iv_change=iv_change,
        )

    underlying_return = float((window.iloc[-1] - window.iloc[0]) / window.iloc[0])
    return RegimeFeatures(
        underlying_return=underlying_return,
        realized_volatility=_realized_volatility(window),
        iv_change=iv_change,
        volume_change=_volume_change(df.tail(lookback_days)),
    )


def classify_regime(features: RegimeFeatures) -> str:
    """
    Classify market path into a road-condition regime.

    cruise_descent: orderly drift lower, vol contained, IV not expanding.
    """
    ret = features.underlying_return
    rv = features.realized_volatility
    iv_delta = features.iv_change

    if abs(ret) < 0.002 and rv < 0.12 and abs(iv_delta) < 0.02:
        return "compression"

    if ret <= -0.003 and rv < 0.22 and iv_delta <= 0.03:
        if features.volume_change is None or features.volume_change < 0.35:
            return "cruise_descent"

    if abs(ret) >= 0.04 or rv >= 0.35 or iv_delta >= 0.08:
        return "rupture"

    if features.cvd is not None and np.sign(features.cvd) != np.sign(ret) and abs(ret) >= 0.01:
        return "reversal"

    if abs(ret) < 0.015 and rv >= 0.15:
        return "chop"

    if ret <= -0.003 and rv < 0.22:
        return "cruise_descent"

    if abs(ret) < 0.01:
        return "chop"

    return "unknown"


def describe_regime(regime: str) -> str:
    descriptions = {
        "cruise_descent": "Orderly drift lower with contained realized vol and stable IV.",
        "rupture": "Sharp move or volatility expansion; path broke out of prior range.",
        "chop": "Sideways, noisy path without clean directional follow-through.",
        "compression": "Tight range, low realized vol, IV largely unchanged.",
        "reversal": "Flow or CVD diverges from price; potential trend handoff.",
        "unknown": "Insufficient or mixed signals for a confident regime label.",
    }
    return descriptions.get(regime, descriptions["unknown"])
