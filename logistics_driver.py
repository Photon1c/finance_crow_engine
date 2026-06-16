"""Trading Logistics Driver — trajectory evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from regime_classifier import (
    RegimeFeatures,
    classify_regime,
    describe_regime,
    features_from_trajectory,
)
from trade_trajectory import TradeTrajectory
from trajectory_reward import RewardSchema


@dataclass
class LogisticsReport:
    """Structured output from a trajectory evaluation pass."""

    trade: TradeTrajectory
    reward: RewardSchema
    regime: str
    regime_description: str
    lesson: str
    suggested_vehicle: str

    def to_dict(self) -> dict:
        return {
            "trade": self.trade.summary_dict(),
            "regime": self.regime,
            "regime_description": self.regime_description,
            "reward": self.reward.as_dict(),
            "total_reward": self.reward.total_reward(),
            "classification": self.reward.classification(),
            "lesson": self.lesson,
            "suggested_vehicle": self.suggested_vehicle,
        }


def _suggest_vehicle(trade: TradeTrajectory, regime: str, classification: str) -> str:
    ctype = trade.contract_type.lower()
    bearish = ctype == "put"

    if classification == "correct_direction_wrong_vehicle" and regime == "cruise_descent":
        if bearish:
            return (
                "Longer-dated put spread or bear put vertical; avoid front-month "
                "long put when decay dominates a slow drift lower."
            )
        return (
            "Longer-dated call spread or diagonal; avoid expensive short-dated "
            "long call in a grind-higher regime."
        )

    if classification == "wrong_direction":
        return "Re-evaluate directional thesis before selecting any option structure."

    if classification == "exit_failure":
        return "Predefined exit rules or trailing stop on premium; reduce open-ended hold time."

    if classification == "unstable_trade_path":
        return "Smaller size or defined-risk spread; avoid naked long premium in unstable vol."

    if classification == "good_trade_path":
        return "Repeat structure when regime and thesis align; size per risk budget."

    if regime == "rupture":
        return "Short-dated long gamma only with tight risk; prefer spreads over naked premium."

    if regime == "compression":
        return "Sell premium or use calendars; long premium needs a catalyst break."

    return "Match tenor and structure to regime: defined risk in chop, spreads in descent."


def _build_lesson(trade: TradeTrajectory, reward: RewardSchema, regime: str) -> str:
    classification = reward.classification()
    direction_ok = trade.was_direction_correct()
    thesis = "Bearish" if trade.contract_type.lower() == "put" else "Bullish"

    if classification == "correct_direction_wrong_vehicle" and regime == "cruise_descent":
        vehicle = "short-dated expensive put" if trade.contract_type.lower() == "put" else "short-dated expensive call"
        return (
            f"{thesis} thesis was correct, but {vehicle} was the wrong vehicle "
            f"for a controlled cruise descent."
        )

    if classification == "wrong_direction":
        return f"{thesis} thesis did not match realized path; direction score penalized."

    if classification == "good_trade_path":
        return "Trajectory stayed on route; component scores support repeating this playbook."

    if classification == "exit_failure":
        return "Directional read was acceptable, but exit discipline failed to protect premium."

    if classification == "unstable_trade_path":
        return "Volatility path was unstable; long premium carried excess path risk."

    if direction_ok and reward.total_reward() < 0:
        return (
            "Directional thesis had merit, but magnitude, vol, or exit components "
            "dominated the trajectory reward."
        )

    return "Review component scores and regime before reusing this structure."


def evaluate_trajectory(
    trade: TradeTrajectory,
    reward: RewardSchema,
    features: Optional[RegimeFeatures] = None,
) -> LogisticsReport:
    """
    Evaluate a completed trade: classify regime, apply reward schema, emit lesson.

    Parameters
    ----------
    trade:
        Completed option trajectory.
    reward:
        Decomposed component scores for the trade path.
    features:
        Optional precomputed regime features; defaults to trajectory-derived values.
    """
    feat = features or features_from_trajectory(trade)
    regime = classify_regime(feat)
    classification = reward.classification()

    return LogisticsReport(
        trade=trade,
        reward=reward,
        regime=regime,
        regime_description=describe_regime(regime),
        lesson=_build_lesson(trade, reward, regime),
        suggested_vehicle=_suggest_vehicle(trade, regime, classification),
    )


def format_markdown_report(report: LogisticsReport) -> str:
    """Render a human-readable Markdown evaluation report."""
    trade = report.trade
    reward = report.reward
    lines = [
        f"# Trading Logistics Driver Report: {trade.ticker} {trade.contract_type.title()}",
        "",
        "## Trade Summary",
        "",
        f"- **Contract:** {trade.ticker} {trade.contract_type.upper()} ${trade.strike:.0f} exp {trade.expiry}",
        f"- **Entry / Exit:** {trade.entry_time.date()} -> {trade.exit_time.date()} ({trade.dte} DTE at entry)",
        f"- **Premium:** ${trade.entry_price:.2f} -> ${trade.exit_price:.2f} ({trade.pnl_pct:+.1%})",
        f"- **Underlying:** ${trade.underlying_entry:.2f} -> ${trade.underlying_exit:.2f} "
        f"({trade.underlying_move_pct:+.2%})",
        f"- **Direction correct:** {trade.was_direction_correct()}",
        f"- **Notes:** {trade.notes or '—'}",
        "",
        "## Road Condition (Regime)",
        "",
        f"- **Regime:** `{report.regime}`",
        f"- **Description:** {report.regime_description}",
        "",
        "## Reward Decomposition",
        "",
        "| Component | Score | Weight | Weighted |",
        "| :--- | ---: | ---: | ---: |",
        f"| Direction | {reward.direction_score:+.1f} | {reward.direction_weight:.2f} | "
        f"{reward.direction_score * reward.direction_weight:+.2f} |",
        f"| Timing | {reward.timing_score:+.1f} | {reward.timing_weight:.2f} | "
        f"{reward.timing_score * reward.timing_weight:+.2f} |",
        f"| Magnitude | {reward.magnitude_score:+.1f} | {reward.magnitude_weight:.2f} | "
        f"{reward.magnitude_score * reward.magnitude_weight:+.2f} |",
        f"| Volatility | {reward.volatility_score:+.1f} | {reward.volatility_weight:.2f} | "
        f"{reward.volatility_score * reward.volatility_weight:+.2f} |",
        f"| Exit | {reward.exit_score:+.1f} | {reward.exit_weight:.2f} | "
        f"{reward.exit_score * reward.exit_weight:+.2f} |",
        "",
        f"**Total reward:** {reward.total_reward():+.2f}",
        "",
        f"**Classification:** `{reward.classification()}`",
        "",
        "## Model Lesson",
        "",
        report.lesson,
        "",
        "## Suggested Better Vehicle",
        "",
        report.suggested_vehicle,
        "",
    ]
    return "\n".join(lines)
