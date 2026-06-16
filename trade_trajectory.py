"""Completed option trade represented as a trajectory, not a single P/L scalar."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TradeTrajectory:
    """Snapshot of a completed option trade path."""

    ticker: str
    contract_type: str  # "call" or "put"
    strike: float
    expiry: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    underlying_entry: float
    underlying_exit: float
    dte: int
    implied_vol_entry: float
    implied_vol_exit: float
    delta: float
    gamma: float
    theta: float
    vega: float
    notes: str = ""
    cvd_entry: Optional[float] = field(default=None)
    cvd_exit: Optional[float] = field(default=None)
    volume_ratio: Optional[float] = field(default=None)

    @property
    def pnl_pct(self) -> float:
        """Return on premium paid (negative if option expired worthless)."""
        if self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price

    @property
    def underlying_move_pct(self) -> float:
        if self.underlying_entry == 0:
            return 0.0
        return (self.underlying_exit - self.underlying_entry) / self.underlying_entry

    @property
    def moneyness(self) -> float:
        """Spot relative to strike (>1 ITM call / OTM put context)."""
        if self.strike == 0:
            return 0.0
        return self.underlying_entry / self.strike

    @property
    def expired_worthless(self) -> bool:
        return self.exit_price <= 0.01 and self.dte <= 1

    @property
    def iv_change(self) -> float:
        return self.implied_vol_exit - self.implied_vol_entry

    def was_direction_correct(self) -> bool:
        """
        Thesis aligned with realized underlying move.

        Puts benefit from lower spot; calls benefit from higher spot.
        """
        move = self.underlying_move_pct
        ctype = self.contract_type.lower()
        if ctype == "put":
            return move < 0.0
        if ctype == "call":
            return move > 0.0
        raise ValueError(f"contract_type must be 'call' or 'put', got {self.contract_type!r}")

    def summary_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "contract_type": self.contract_type,
            "strike": self.strike,
            "expiry": self.expiry,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "underlying_entry": self.underlying_entry,
            "underlying_exit": self.underlying_exit,
            "dte": self.dte,
            "pnl_pct": self.pnl_pct,
            "underlying_move_pct": self.underlying_move_pct,
            "moneyness": self.moneyness,
            "expired_worthless": self.expired_worthless,
            "direction_correct": self.was_direction_correct(),
            "notes": self.notes,
        }
