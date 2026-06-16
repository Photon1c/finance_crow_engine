"""Stub adapter for future AgentLightning RL integration."""

from __future__ import annotations

from typing import Any, Optional

from trade_trajectory import TradeTrajectory
from trajectory_reward import RewardSchema


def _trade_state_vector(trade: TradeTrajectory, regime: str) -> dict[str, Any]:
    """Compact state representation for replay / training pipelines."""
    return {
        "ticker": trade.ticker,
        "contract_type": trade.contract_type,
        "strike": trade.strike,
        "dte": trade.dte,
        "moneyness": trade.moneyness,
        "underlying_move_pct": trade.underlying_move_pct,
        "iv_entry": trade.implied_vol_entry,
        "iv_exit": trade.implied_vol_exit,
        "delta": trade.delta,
        "gamma": trade.gamma,
        "theta": trade.theta,
        "vega": trade.vega,
        "regime": regime,
        "direction_correct": trade.was_direction_correct(),
    }


def _trade_action_vector(trade: TradeTrajectory) -> dict[str, Any]:
    """Action taken: structure selection at entry."""
    return {
        "vehicle": f"{trade.contract_type}_long",
        "strike": trade.strike,
        "expiry": trade.expiry,
        "entry_price": trade.entry_price,
        "dte": trade.dte,
    }


def to_lightning_transition(
    trade: TradeTrajectory,
    reward_schema: RewardSchema,
    regime: str,
    *,
    next_trade: Optional[TradeTrajectory] = None,
    done: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build a plain transition dict for future AgentLightning wiring.

    Does not import or require AgentLightning at runtime.
    """
    transition: dict[str, Any] = {
        "state": _trade_state_vector(trade, regime),
        "action": _trade_action_vector(trade),
        "reward": reward_schema.total_reward(),
        "reward_decomposition": reward_schema.as_dict(),
        "next_state": (
            _trade_state_vector(next_trade, regime) if next_trade else None
        ),
        "done": done,
        "metadata": {
            "classification": reward_schema.classification(),
            "regime": regime,
            "pnl_pct": trade.pnl_pct,
            "lesson_scope": "trajectory_not_pl",
            **(metadata or {}),
        },
    }
    return transition
