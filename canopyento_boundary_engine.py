"""CanopyEnto boundary rupture engine — containment stress from boundary tests and volume."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    from data_loader import DEFAULT_STOCK_DIR, load_stock_data, parse_price
except ImportError:
    DEFAULT_STOCK_DIR = "F:/inputs/stocks"

    def parse_price(value) -> float:
        if pd.isna(value):
            return float("nan")
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value).replace("$", "").replace(",", "").strip())

    def load_stock_data(ticker, base_dir=DEFAULT_STOCK_DIR):
        ticker_upper = ticker.upper()
        filepath = Path(base_dir) / f"{ticker_upper}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Stock file not found: {filepath}")
        df = pd.read_csv(filepath, parse_dates=["Date"])
        if "Date" in df.columns:
            df = df.sort_values("Date", ascending=True)
        return df


DEFAULT_TICKER = "SPY"
DEFAULT_WEEKLY_WINDOW = 5
REQUIRED_COLUMNS = ("Date", "Volume", "Open", "High", "Low")
CLOSE_ALIASES = ("Close/Last", "Close")

BOUNDARY_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "rolling_high",
    "rolling_low",
    "upper_boundary_test",
    "lower_boundary_test",
    "boundary_test",
    "boundary_tests_count",
    "B_s",
    "rolling_avg_volume",
    "E_i",
    "rupture_pressure_score",
    "regime_label",
    "regime_persistence",
]

WEEKLY_STANCE_SCORE_KEYS = (
    "direction_score",
    "timing_score",
    "magnitude_score",
    "volatility_score",
    "packet_completion_confidence",
    "absorption_confidence",
    "hidden_process_uncertainty",
    "continuation_probability",
    "regime_persistence",
    "rupture_probability",
)

WEEKLY_STANCE_META_KEYS = (
    "stance_confidence",
    "gate_stance",
    "direction_bias",
    "trade_permission",
    "stance_quadrant",
    "recommended_action",
)

OUTPUT_COLUMNS = (
    list(BOUNDARY_COLUMNS)
    + [key for key in WEEKLY_STANCE_SCORE_KEYS if key not in BOUNDARY_COLUMNS]
    + list(WEEKLY_STANCE_META_KEYS)
)

GATE_STANCES = (
    "WAIT / PACKET BUFFERING",
    "BEARISH BUT ABSORBED",
    "ACTIONABLE DIRECTIONAL STANCE",
    "LOW-CONFIDENCE CRUISE MODE",
)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _clip11(value: float) -> float:
    return float(max(-1.0, min(1.0, value)))


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    return int(value)


def load_stock_data_fallback(
    ticker: str,
    base_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Load OHLCV data via data_loader, with a direct CSV fallback."""
    stock_dir = base_dir or DEFAULT_STOCK_DIR
    ticker_upper = ticker.upper()

    try:
        df = load_stock_data(ticker_upper, base_dir=stock_dir)
    except FileNotFoundError:
        filepath = Path(stock_dir) / f"{ticker_upper}.csv"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Stock file not found for {ticker_upper}. "
                f"Expected: {filepath}"
            ) from None
        df = pd.read_csv(filepath, parse_dates=["Date"])

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for {ticker_upper}: {', '.join(missing)}"
        )

    close_col = next((col for col in CLOSE_ALIASES if col in df.columns), None)
    if close_col is None:
        raise ValueError(
            f"Missing close column for {ticker_upper}. "
            f"Expected one of: {', '.join(CLOSE_ALIASES)}"
        )

    df = df.copy()
    if close_col != "Close":
        df["Close"] = df[close_col]
    df["Close"] = df["Close"].map(parse_price)

    for col in ("Open", "High", "Low"):
        df[col] = df[col].map(parse_price)

    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date", ascending=True)
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No usable rows found for {ticker_upper}")

    return df


def classify_regime(b_s: float, e_i: float, rupture_pressure_score: float) -> str:
    """Classify containment regime from boundary stress and energy injection."""
    if rupture_pressure_score >= 0.50:
        return "RUPTURE_CANDIDATE"
    if b_s >= 0.20 and e_i >= 1.20:
        return "CONTAINMENT_STRESS"
    if b_s >= 0.20 and e_i < 1.20:
        return "PRESSURE_BUILDING"
    if b_s < 0.20 and e_i >= 1.20:
        return "ENERGY_INJECTION"
    return "IDLE"


def compute_regime_persistence(regime_labels: pd.Series) -> pd.Series:
    """Count consecutive sessions in the current regime label."""
    labels = regime_labels.astype(str)
    run_ids = (labels != labels.shift()).cumsum()
    return labels.groupby(run_ids, sort=False).cumcount() + 1


def derive_weekly_stance_scores(
    week: pd.DataFrame,
    latest: pd.Series,
) -> dict[str, float]:
    """
    Derive weekly stance vector from a trailing window of boundary metrics.

    Scores are normalized to project conventions:
      direction_score in [-1, +1]  (negative = bearish, positive = bullish)
      all other scores in [0, 1]
    """
    if len(week) < 2:
        return {key: float("nan") for key in WEEKLY_STANCE_SCORE_KEYS}

    close = float(latest["Close"])
    rolling_high = float(latest["rolling_high"])
    rolling_low = float(latest["rolling_low"])
    b_s = float(latest["B_s"]) if pd.notna(latest["B_s"]) else 0.0
    e_i = float(latest["E_i"]) if pd.notna(latest["E_i"]) else 1.0
    rupture = float(latest["rupture_pressure_score"]) if pd.notna(latest["rupture_pressure_score"]) else 0.0
    regime = str(latest.get("regime_label", "IDLE"))

    weekly_return = float(week["Close"].iloc[-1] / week["Close"].iloc[0] - 1.0)
    range_width = max(rolling_high - rolling_low, close * 1e-6)
    range_position = _clip01((close - rolling_low) / range_width)

    direction_score = _clip11(weekly_return * 6.0 + (range_position - 0.5) * 1.5)
    if week["lower_boundary_test"].any() and not week["upper_boundary_test"].any():
        direction_score = _clip11(direction_score - 0.25)
    elif week["upper_boundary_test"].any() and not week["lower_boundary_test"].any():
        direction_score = _clip11(direction_score + 0.25)

    setup_regimes = {"PRESSURE_BUILDING", "CONTAINMENT_STRESS", "RUPTURE_CANDIDATE"}
    timing_score = _clip01(
        b_s * 1.2
        + (0.25 if regime in setup_regimes else 0.0)
        + rupture * 0.35
    )

    magnitude_score = _clip01(range_width / close * 12.0)

    vol_expansion = max(e_i - 1.0, 0.0)
    volatility_score = _clip01(vol_expansion * 0.45 + b_s * 0.35 + rupture * 0.25)

    regime_counts = week["regime_label"].value_counts(normalize=True)
    dominant_regime_share = float(regime_counts.iloc[0]) if not regime_counts.empty else 0.0
    both_boundaries = bool(week["upper_boundary_test"].any() and week["lower_boundary_test"].any())
    packet_completion_confidence = _clip01(
        dominant_regime_share * 0.65
        + (0.0 if both_boundaries else 0.25)
        + (0.10 if regime in {"IDLE", "RUPTURE_CANDIDATE"} else 0.0)
    )

    hold_distance = min(close - rolling_low, rolling_high - close) / range_width
    absorption_confidence = _clip01(
        b_s * hold_distance * 1.8
        + (0.35 if regime == "PRESSURE_BUILDING" else 0.0)
        + (0.20 if regime == "CONTAINMENT_STRESS" and hold_distance > 0.25 else 0.0)
        + (0.15 if e_i < 1.20 and b_s >= 0.20 else 0.0)
    )

    regime_changes = float((week["regime_label"] != week["regime_label"].shift()).sum()) / len(week)
    daily_returns = week["Close"].pct_change().dropna()
    sign_flip_ratio = 0.0
    if len(daily_returns) >= 2:
        signs = daily_returns.apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
        sign_flip_ratio = float((signs != signs.shift()).sum()) / max(len(signs) - 1, 1)

    hidden_process_uncertainty = _clip01(
        (0.35 if both_boundaries else 0.0)
        + regime_changes * 0.30
        + sign_flip_ratio * 0.25
        + (1.0 - packet_completion_confidence) * 0.35
    )

    if len(daily_returns) > 0:
        if weekly_return >= 0:
            aligned_days = float((daily_returns > 0).mean())
        else:
            aligned_days = float((daily_returns < 0).mean())
    else:
        aligned_days = 0.0

    continuation_probability = _clip01(
        aligned_days * 0.55
        + min(abs(weekly_return) * 15.0, 0.35)
        + (0.10 if not both_boundaries else 0.0)
    )

    regime_persistence = _safe_int(latest.get("regime_persistence"), default=1)
    rupture_probability = _clip01(rupture + (0.15 if regime_persistence > 4 else 0.0))

    # Long-lived regimes are deceptive: stability may be rising, or latent rupture energy may be.
    if regime_persistence > 4:
        if regime in setup_regimes:
            packet_completion_confidence = _clip01(packet_completion_confidence - 0.12)
            hidden_process_uncertainty = _clip01(hidden_process_uncertainty + 0.08)
        elif regime == "IDLE":
            packet_completion_confidence = _clip01(packet_completion_confidence + 0.10)

    return {
        "direction_score": direction_score,
        "timing_score": timing_score,
        "magnitude_score": magnitude_score,
        "volatility_score": volatility_score,
        "packet_completion_confidence": packet_completion_confidence,
        "absorption_confidence": absorption_confidence,
        "hidden_process_uncertainty": hidden_process_uncertainty,
        "continuation_probability": continuation_probability,
        "regime_persistence": float(regime_persistence),
        "rupture_probability": rupture_probability,
    }


def compute_stance_confidence(weekly_stance: dict[str, float]) -> float:
    """
    Trade-permission confidence: direction is intentionally excluded.

    Positive values favor stance permission; negative values favor waiting.
    """
    positive = (
        weekly_stance["timing_score"]
        + weekly_stance["magnitude_score"]
        + weekly_stance["volatility_score"]
        + weekly_stance["packet_completion_confidence"]
        + weekly_stance["continuation_probability"]
    )
    negative = (
        weekly_stance["absorption_confidence"]
        + weekly_stance["hidden_process_uncertainty"]
    )
    return float((positive - negative) / 7.0)


def evaluate_stance_gate(weekly_stance: dict[str, float]) -> str:
    """Final permission gate separating directional bias from trade permission."""
    direction_score = weekly_stance["direction_score"]
    if weekly_stance["hidden_process_uncertainty"] > 0.60:
        return "WAIT / PACKET BUFFERING"
    if weekly_stance["absorption_confidence"] > 0.65 and direction_score < 0:
        return "BEARISH BUT ABSORBED"
    if (
        weekly_stance["continuation_probability"] > 0.60
        and weekly_stance["packet_completion_confidence"] > 0.55
    ):
        return "ACTIONABLE DIRECTIONAL STANCE"
    return "LOW-CONFIDENCE CRUISE MODE"


def resolve_stance_quadrant(
    direction_score: float,
    gate_stance: str,
) -> tuple[str, str, str]:
    """Map direction bias and gate into the four stance states."""
    direction_bias = "bearish" if direction_score < 0 else "bullish"
    trade_permission = "actionable" if gate_stance == "ACTIONABLE DIRECTIONAL STANCE" else "weak"
    stance_quadrant = f"{direction_bias} / {'actionable' if trade_permission == 'actionable' else 'unresolved'}"
    return direction_bias, trade_permission, stance_quadrant


def recommend_model_action(
    weekly_stance: dict[str, float],
    *,
    direction_bias: str,
    trade_permission: str,
    gate_stance: str,
    stance_confidence: float,
) -> str:
    """Human-readable action guidance for weekly stance output."""
    if gate_stance == "WAIT / PACKET BUFFERING":
        return (
            "Wait for packet resolution. Hidden process uncertainty is elevated; "
            "the system is still buffering and a directional stance is premature."
        )
    if gate_stance == "BEARISH BUT ABSORBED":
        return (
            "Bearish bias detected, but absorption is high and downside continuation "
            "is not confirmed. Reduce size or wait for release before aggressive put exposure."
        )
    if gate_stance == "ACTIONABLE DIRECTIONAL STANCE":
        side = "bullish" if direction_bias == "bullish" else "bearish"
        return (
            f"Actionable {side} stance. Packet completion and continuation probability "
            "support a directional position with normal sizing."
        )
    if trade_permission == "weak" and direction_bias == "bearish":
        persistence = _safe_int(weekly_stance.get("regime_persistence"))
        maturity_note = (
            " The observed packet has not resolved enough to trust the directional conclusion."
        )
        persistence_note = ""
        if persistence > 4:
            persistence_note = (
                f" Regime `{persistence}` sessions old — long persistence may mask latent rupture energy."
            )
        return (
            "Bearish bias detected, but packet completion is low. "
            "Current state: bearish / unresolved. "
            "Wait for confirmation or reduce position size."
            + maturity_note
            + persistence_note
        )
    if trade_permission == "weak" and direction_bias == "bullish":
        persistence = _safe_int(weekly_stance.get("regime_persistence"))
        maturity_note = (
            " The observed packet has not resolved enough to trust the directional conclusion."
        )
        persistence_note = ""
        if persistence > 4:
            persistence_note = (
                f" Regime `{persistence}` sessions old — long persistence may mask latent rupture energy."
            )
        return (
            "Bullish bias detected, but packet completion is low. "
            "Current state: bullish / unresolved. "
            "Wait for confirmation or reduce position size."
            + maturity_note
            + persistence_note
        )
    return (
        "Low-confidence cruise mode. Directional edge is insufficient relative to "
        f"absorption and unresolved internals (stance confidence {stance_confidence:+.2f})."
    )


def build_weekly_stance(
    week: pd.DataFrame,
    latest: pd.Series,
) -> dict[str, Any]:
    """Assemble the full weekly stance packet for one evaluation window."""
    scores = derive_weekly_stance_scores(week, latest)
    stance_confidence = compute_stance_confidence(scores)
    gate_stance = evaluate_stance_gate(scores)
    direction_bias, trade_permission, stance_quadrant = resolve_stance_quadrant(
        scores["direction_score"],
        gate_stance,
    )
    recommended_action = recommend_model_action(
        scores,
        direction_bias=direction_bias,
        trade_permission=trade_permission,
        gate_stance=gate_stance,
        stance_confidence=stance_confidence,
    )
    return {
        **scores,
        "stance_confidence": stance_confidence,
        "gate_stance": gate_stance,
        "direction_bias": direction_bias,
        "trade_permission": trade_permission,
        "stance_quadrant": stance_quadrant,
        "recommended_action": recommended_action,
    }


def compute_weekly_stance_metrics(
    df: pd.DataFrame,
    weekly_window: int = DEFAULT_WEEKLY_WINDOW,
) -> pd.DataFrame:
    """Attach rolling weekly stance scores to each row with enough history."""
    if weekly_window < 2:
        raise ValueError("weekly_window must be >= 2")

    result = df.copy()
    for key in WEEKLY_STANCE_SCORE_KEYS:
        result[key] = float("nan")
    for key in WEEKLY_STANCE_META_KEYS:
        result[key] = pd.NA

    required = ("Close", "rolling_high", "rolling_low", "B_s", "E_i", "rupture_pressure_score", "regime_label")
    if any(col not in result.columns for col in required):
        return result

    stance_rows: dict[Any, dict[str, Any]] = {}
    min_idx = max(weekly_window - 1, 0)
    for idx in range(min_idx, len(result)):
        row = result.iloc[idx]
        if any(pd.isna(row[col]) for col in ("Close", "rolling_high", "rolling_low", "B_s")):
            continue

        week = result.iloc[idx - weekly_window + 1 : idx + 1]
        stance_rows[result.index[idx]] = build_weekly_stance(week, row)

    if stance_rows:
        stance_df = pd.DataFrame.from_dict(stance_rows, orient="index")
        result.update(stance_df)

    return result


def compute_boundary_metrics(
    df: pd.DataFrame,
    lookback: int = 20,
    tolerance: float = 0.003,
    volume_window: int = 20,
) -> pd.DataFrame:
    """Compute boundary stress, energy injection, and rupture pressure scores."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if volume_window < 1:
        raise ValueError("volume_window must be >= 1")

    result = df.copy()

    result["rolling_high"] = result["High"].rolling(window=lookback, min_periods=lookback).max()
    result["rolling_low"] = result["Low"].rolling(window=lookback, min_periods=lookback).min()

    result["upper_boundary_test"] = result["Close"] >= result["rolling_high"] * (1.0 - tolerance)
    result["lower_boundary_test"] = result["Close"] <= result["rolling_low"] * (1.0 + tolerance)
    result["boundary_test"] = result["upper_boundary_test"] | result["lower_boundary_test"]

    result["boundary_tests_count"] = (
        result["boundary_test"]
        .rolling(window=lookback, min_periods=lookback)
        .sum()
    )
    result["B_s"] = result["boundary_tests_count"] / lookback

    result["rolling_avg_volume"] = (
        result["Volume"]
        .rolling(window=volume_window, min_periods=volume_window)
        .mean()
    )
    result["E_i"] = result["Volume"] / result["rolling_avg_volume"]
    result["E_i"] = result["E_i"].where(result["rolling_avg_volume"] > 0)

    result["rupture_pressure_score"] = result["B_s"] * result["E_i"]
    result["regime_label"] = result.apply(
        lambda row: classify_regime(row["B_s"], row["E_i"], row["rupture_pressure_score"]),
        axis=1,
    )
    result["regime_persistence"] = compute_regime_persistence(result["regime_label"])

    return result


def write_report(
    df: pd.DataFrame,
    report_path: Path,
    *,
    ticker: str,
    lookback: int,
    tolerance: float,
    volume_window: int,
    weekly_window: int,
) -> None:
    """Write a markdown summary of the latest boundary rupture and weekly stance state."""
    latest = df.iloc[-1]
    latest_date = latest["Date"]
    if hasattr(latest_date, "strftime"):
        latest_date_str = latest_date.strftime("%Y-%m-%d")
    else:
        latest_date_str = str(latest_date)

    top10 = (
        df.dropna(subset=["rupture_pressure_score"])
        .sort_values("rupture_pressure_score", ascending=False)
        .head(10)
    )

    lines = [
        "# CanopyEnto Boundary Rupture Report",
        "",
        f"- **Ticker:** {ticker.upper()}",
        f"- **Lookback:** {lookback} sessions",
        f"- **Boundary tolerance:** {tolerance:.4f}",
        f"- **Volume window:** {volume_window} sessions",
        f"- **Weekly stance window:** {weekly_window} sessions",
        "",
        "## Latest Snapshot",
        "",
        f"- **Latest date:** {latest_date_str}",
        f"- **Latest close:** {latest['Close']:.2f}",
        f"- **Latest B_s:** {latest['B_s']:.4f}",
        f"- **Latest E_i:** {latest['E_i']:.4f}",
        f"- **Latest rupture_pressure_score:** {latest['rupture_pressure_score']:.4f}",
        f"- **Latest regime_label:** {latest['regime_label']}",
        f"- **Latest regime_persistence:** {_safe_int(latest.get('regime_persistence'))} sessions",
        "",
    ]

    if pd.notna(latest.get("stance_quadrant")):
        lines.extend([
            "## Weekly Stance Filter",
            "",
            "A forecast should not only predict direction. It should estimate whether the "
            "observed system has finished becoming the thing being predicted. "
            "This engine forecasts **state maturity**, not price.",
            "",
            f"- **Direction bias:** {latest['direction_bias']} (`direction_score={latest['direction_score']:.2f}`)",
            f"- **Trade permission:** {latest['trade_permission']} (`stance_confidence={latest['stance_confidence']:+.2f}`)",
            f"- **Stance quadrant:** {latest['stance_quadrant']}",
            f"- **Gate stance:** {latest['gate_stance']}",
            f"- **Regime persistence:** {_safe_int(latest['regime_persistence'])} consecutive sessions in `{latest['regime_label']}`",
            "",
            "### Stance Vector",
            "",
            "| Component | Score |",
            "| :--- | ---: |",
            f"| direction_score | {latest['direction_score']:.2f} |",
            f"| timing_score | {latest['timing_score']:.2f} |",
            f"| magnitude_score | {latest['magnitude_score']:.2f} |",
            f"| volatility_score | {latest['volatility_score']:.2f} |",
            f"| packet_completion_confidence | {latest['packet_completion_confidence']:.2f} |",
            f"| absorption_confidence | {latest['absorption_confidence']:.2f} |",
            f"| hidden_process_uncertainty | {latest['hidden_process_uncertainty']:.2f} |",
            f"| continuation_probability | {latest['continuation_probability']:.2f} |",
            f"| regime_persistence | {_safe_int(latest['regime_persistence'])} |",
            f"| rupture_probability | {latest['rupture_probability']:.2f} |",
            "",
            "### Recommended Model Action",
            "",
            str(latest["recommended_action"]),
            "",
        ])

    lines.extend([
        "## Boundary Model",
        "",
        "Rupture probability increases when a constrained system is repeatedly stressed "
        "while continuing to absorb energy without release.",
        "",
        "## Top 10 Rupture Pressure Scores",
        "",
        "| Date | Close | B_s | E_i | Rupture Score | Regime | Stance |",
        "| :--- | ---: | ---: | ---: | ---: | :--- | :--- |",
    ])

    for _, row in top10.iterrows():
        row_date = row["Date"]
        if hasattr(row_date, "strftime"):
            row_date = row_date.strftime("%Y-%m-%d")
        stance = row.get("stance_quadrant", "")
        if pd.isna(stance):
            stance = ""
        lines.append(
            f"| {row_date} | {row['Close']:.2f} | {row['B_s']:.4f} | "
            f"{row['E_i']:.4f} | {row['rupture_pressure_score']:.4f} | "
            f"{row['regime_label']} | {stance} |"
        )

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_weekly_stance_json(
    stance: dict[str, Any],
    json_path: Path,
    *,
    ticker: str,
    as_of_date: str,
) -> None:
    """Write the latest weekly stance packet as JSON for dashboard ingestion."""
    payload = {
        "ticker": ticker.upper(),
        "as_of_date": as_of_date,
        "philosophy": (
            "Forecast state maturity, not price. Direction may exist while trade permission "
            "remains weak until the observed packet resolves."
        ),
        "weekly_stance": {key: stance[key] for key in WEEKLY_STANCE_SCORE_KEYS},
        "stance_confidence": stance["stance_confidence"],
        "gate_stance": stance["gate_stance"],
        "direction_bias": stance["direction_bias"],
        "trade_permission": stance["trade_permission"],
        "stance_quadrant": stance["stance_quadrant"],
        "recommended_action": stance["recommended_action"],
    }
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CanopyEnto boundary rupture engine — detect containment stress."
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help=f"Stock ticker symbol (default: {DEFAULT_TICKER})",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=20,
        help="Rolling lookback window for boundary detection (default: 20)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.003,
        help="Boundary proximity tolerance as a fraction (default: 0.003)",
    )
    parser.add_argument(
        "--volume-window",
        type=int,
        default=20,
        dest="volume_window",
        help="Rolling window for average volume (default: 20)",
    )
    parser.add_argument(
        "--weekly-window",
        type=int,
        default=DEFAULT_WEEKLY_WINDOW,
        dest="weekly_window",
        help="Trailing sessions for weekly stance evaluation (default: 5)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path (default: outputs/canopyento_boundary_{TICKER}.csv)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Markdown report path (default: outputs/canopyento_boundary_{TICKER}.md)",
    )
    parser.add_argument(
        "--stance-json",
        default=None,
        dest="stance_json",
        help="Weekly stance JSON path (default: outputs/canopyento_weekly_stance_{TICKER}.json)",
    )
    parser.add_argument(
        "--stock-dir",
        default=DEFAULT_STOCK_DIR,
        help=f"Directory containing stock CSV files (default: {DEFAULT_STOCK_DIR})",
    )

    args = parser.parse_args(argv)
    ticker = args.ticker.upper()
    output_path = Path(args.output or f"outputs/canopyento_boundary_{ticker}.csv")
    report_path = Path(args.report or f"outputs/canopyento_boundary_{ticker}.md")
    stance_json_path = Path(args.stance_json or f"outputs/canopyento_weekly_stance_{ticker}.json")

    try:
        raw_df = load_stock_data_fallback(ticker, base_dir=args.stock_dir)
        metrics_df = compute_boundary_metrics(
            raw_df,
            lookback=args.lookback,
            tolerance=args.tolerance,
            volume_window=args.volume_window,
        )
        metrics_df = compute_weekly_stance_metrics(
            metrics_df,
            weekly_window=args.weekly_window,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    export_df = metrics_df[OUTPUT_COLUMNS].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False)

    write_report(
        metrics_df,
        report_path,
        ticker=ticker,
        lookback=args.lookback,
        tolerance=args.tolerance,
        volume_window=args.volume_window,
        weekly_window=args.weekly_window,
    )

    latest = metrics_df.iloc[-1]
    latest_date = latest["Date"]
    if hasattr(latest_date, "strftime"):
        latest_date_str = latest_date.strftime("%Y-%m-%d")
    else:
        latest_date_str = str(latest_date)

    if pd.notna(latest.get("stance_quadrant")):
        latest_stance = {key: latest[key] for key in WEEKLY_STANCE_SCORE_KEYS + WEEKLY_STANCE_META_KEYS}
        write_weekly_stance_json(
            latest_stance,
            stance_json_path,
            ticker=ticker,
            as_of_date=latest_date_str,
        )

    print(f"CanopyEnto boundary analysis complete for {ticker}")
    print(f"  CSV:         {output_path}")
    print(f"  Report:      {report_path}")
    if pd.notna(latest.get("stance_quadrant")):
        print(f"  Stance JSON: {stance_json_path}")
    print(
        f"  Latest: {latest['Date']} | close={latest['Close']:.2f} | "
        f"B_s={latest['B_s']:.4f} | E_i={latest['E_i']:.4f} | "
        f"score={latest['rupture_pressure_score']:.4f} | "
        f"regime={latest['regime_label']}"
    )
    if pd.notna(latest.get("stance_quadrant")):
        print(
            f"  Stance: {latest['stance_quadrant']} | gate={latest['gate_stance']} | "
            f"confidence={latest['stance_confidence']:+.2f} | "
            f"persistence={_safe_int(latest['regime_persistence'])} | "
            f"rupture_prob={latest['rupture_probability']:.2f}"
        )
        print(f"  Action: {latest['recommended_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
