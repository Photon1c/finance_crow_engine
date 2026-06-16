"""Reward decomposition for option trade trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_WEIGHTS: dict[str, float] = {
    "direction": 1.00,
    "timing": 0.75,
    "magnitude": 1.00,
    "volatility": 1.25,
    "exit": 1.50,
}

CLASSIFICATIONS = (
    "correct_direction_wrong_vehicle",
    "wrong_direction",
    "good_trade_path",
    "unstable_trade_path",
    "exit_failure",
)


@dataclass
class RewardSchema:
    """Component scores in [-1, +1] with configurable weights."""

    direction_score: float
    timing_score: float
    magnitude_score: float
    volatility_score: float
    exit_score: float
    direction_weight: float = DEFAULT_WEIGHTS["direction"]
    timing_weight: float = DEFAULT_WEIGHTS["timing"]
    magnitude_weight: float = DEFAULT_WEIGHTS["magnitude"]
    volatility_weight: float = DEFAULT_WEIGHTS["volatility"]
    exit_weight: float = DEFAULT_WEIGHTS["exit"]

    def total_reward(self) -> float:
        """Weighted sum of component scores."""
        return (
            self.direction_score * self.direction_weight
            + self.timing_score * self.timing_weight
            + self.magnitude_score * self.magnitude_weight
            + self.volatility_score * self.volatility_weight
            + self.exit_score * self.exit_weight
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize scores, weights, total, and classification."""
        return {
            "direction_score": self.direction_score,
            "timing_score": self.timing_score,
            "magnitude_score": self.magnitude_score,
            "volatility_score": self.volatility_score,
            "exit_score": self.exit_score,
            "weights": {
                "direction": self.direction_weight,
                "timing": self.timing_weight,
                "magnitude": self.magnitude_weight,
                "volatility": self.volatility_weight,
                "exit": self.exit_weight,
            },
            "total_reward": self.total_reward(),
            "classification": self.classification(),
        }

    def classification(self) -> str:
        """
        Map component scores to a trajectory label.

        Priority order resolves overlapping signals (e.g. correct thesis but bad vehicle).
        """
        d = self.direction_score
        m = self.magnitude_score
        v = self.volatility_score
        e = self.exit_score

        if d < 0.0:
            return "wrong_direction"

        if d >= 0.0 and m >= 0.0 and v >= 0.0 and e >= 0.0:
            return "good_trade_path"

        if d >= 0.5 and m <= -0.5 and v <= -0.5:
            return "correct_direction_wrong_vehicle"

        if e <= -0.5 and d >= 0.0 and m > -0.5 and v > -0.5:
            return "exit_failure"

        if v <= -0.5:
            return "unstable_trade_path"

        if d >= 0.5 and (m <= -0.5 or e <= -0.5):
            return "correct_direction_wrong_vehicle"

        return "unstable_trade_path"


def locked_spy_reward() -> RewardSchema:
    """Canonical SPY put example from the logistics driver spec."""
    return RewardSchema(
        direction_score=1.0,
        timing_score=0.0,
        magnitude_score=-1.0,
        volatility_score=-1.0,
        exit_score=-1.0,
    )
