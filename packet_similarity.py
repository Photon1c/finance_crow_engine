import csv
from pathlib import Path
from typing import Optional

import numpy as np

from packet_persistence import LEDGER_FIELDS, ledger_path
from recursive_weight_engine import COMPONENT_MAP, DEFAULT_WEIGHTS

FEATURE_FIELDS = [
    "direction_score",
    "timing_score",
    "magnitude_score",
    "volatility_score",
    "catalyst_score",
    "exit_score",
    "theta_risk",
    "cvd_confirmation",
]

FEATURE_WEIGHT_KEYS = [
    ("direction", "direction_score"),
    ("timing", "timing_score"),
    ("magnitude", "magnitude_score"),
    ("volatility", "volatility_score"),
    ("catalyst", "catalyst_score"),
    ("exit", "exit_score"),
    ("theta_risk", "theta_risk"),
    ("cvd_confirmation", "cvd_confirmation"),
]

SIMILAR_TO_CHOICES = [key for key, _ in FEATURE_WEIGHT_KEYS]

FILTERABLE_FIELDS = {
    "failure_type",
    "status",
    "instrument",
    "ticker",
    "trade_id",
}

SIMILAR_BOOST = 3.0


def _parse_float(value, default=0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_ledger(path) -> list:
    path = Path(path)
    if not path.is_file():
        return []

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def packet_to_feature_vector(packet: dict) -> np.ndarray:
    values = []
    for field in FEATURE_FIELDS:
        if field == "cvd_confirmation":
            raw = packet.get(field, 0)
            if isinstance(raw, str):
                values.append(1.0 if raw.strip().lower() in {"1", "true", "yes"} else 0.0)
            else:
                values.append(1.0 if bool(raw) else 0.0)
        else:
            values.append(_parse_float(packet.get(field)))
    return np.asarray(values, dtype=float)


def build_feature_matrix(rows: list) -> np.ndarray:
    if not rows:
        return np.empty((0, len(FEATURE_FIELDS)), dtype=float)
    return np.vstack([packet_to_feature_vector(row) for row in rows])


def parse_filter_expression(expression: str) -> tuple:
    if "=" not in expression:
        raise ValueError(f"Filter must be KEY=VALUE, got: {expression}")
    key, value = expression.split("=", 1)
    key = key.strip().lower()
    value = value.strip()
    if key not in FILTERABLE_FIELDS:
        allowed = ", ".join(sorted(FILTERABLE_FIELDS))
        raise ValueError(f"Unsupported filter '{key}'. Allowed: {allowed}")
    return key, value


def parse_filters(filter_args: list) -> list:
    return [parse_filter_expression(item) for item in filter_args]


def apply_ledger_filters(rows: list, filters: list) -> list:
    if not filters:
        return rows

    filtered = []
    for row in rows:
        keep = True
        for key, expected in filters:
            actual = str(row.get(key, "")).strip()
            if key == "trade_id":
                keep = actual == str(expected)
            else:
                keep = actual.lower() == expected.lower()
            if not keep:
                break
        if keep:
            filtered.append(row)
    return filtered


def build_feature_weights(weights: Optional[dict] = None, similar_to: Optional[str] = None) -> np.ndarray:
    weights = weights or DEFAULT_WEIGHTS
    if similar_to and similar_to not in SIMILAR_TO_CHOICES:
        allowed = ", ".join(SIMILAR_TO_CHOICES)
        raise ValueError(f"Unsupported --similar-to '{similar_to}'. Allowed: {allowed}")

    feature_weights = []
    for key, _ in FEATURE_WEIGHT_KEYS:
        value = float(weights.get(key, DEFAULT_WEIGHTS.get(key, 1.0)))
        if similar_to and key == similar_to:
            value *= SIMILAR_BOOST
        feature_weights.append(value)
    return np.asarray(feature_weights, dtype=float)


def weighted_distance(query_vec: np.ndarray, row_vec: np.ndarray, feature_weights: np.ndarray) -> float:
    diff = (query_vec - row_vec) * feature_weights
    return float(np.sqrt(np.sum(diff * diff)))


def find_similar_packets(
    query_packet: dict,
    ledger_file,
    top_k: int = 5,
    exclude_timestamp: Optional[str] = None,
    exclude_trade_id: Optional[int] = None,
    filters: Optional[list] = None,
    weights: Optional[dict] = None,
    similar_to: Optional[str] = None,
) -> list:
    rows = load_ledger(ledger_file)
    if not rows:
        return []

    rows = apply_ledger_filters(rows, filters or [])
    if not rows:
        return []

    vectors = build_feature_matrix(rows)
    query_vec = packet_to_feature_vector(query_packet)
    feature_weights = build_feature_weights(weights, similar_to=similar_to)
    distances = np.array([
        weighted_distance(query_vec, vectors[i], feature_weights)
        for i in range(len(rows))
    ])

    ranked = np.argsort(distances)
    results = []
    for idx in ranked:
        row = rows[int(idx)]
        same_row = (
            exclude_timestamp
            and row.get("timestamp") == exclude_timestamp
            and str(row.get("trade_id")) == str(exclude_trade_id)
        )
        if same_row:
            continue
        results.append(
            {
                "distance": float(distances[int(idx)]),
                "timestamp": row.get("timestamp", ""),
                "trade_id": row.get("trade_id", ""),
                "instrument": row.get("instrument", ""),
                "failure_type": row.get("failure_type", ""),
                "failure_score": _parse_float(row.get("failure_score")),
                "weighted_score": _parse_float(row.get("weighted_score")),
                "status": row.get("status", ""),
                "direction_score": _parse_float(row.get("direction_score")),
                "timing_score": _parse_float(row.get("timing_score")),
                "magnitude_score": _parse_float(row.get("magnitude_score")),
                "volatility_score": _parse_float(row.get("volatility_score")),
                "catalyst_score": _parse_float(row.get("catalyst_score")),
                "exit_score": _parse_float(row.get("exit_score")),
            }
        )
        if len(results) >= top_k:
            break
    return results


def find_similar_for_ticker(
    base_dir,
    ticker: str,
    query_packet: dict,
    top_k: int = 5,
    filters: Optional[list] = None,
    weights: Optional[dict] = None,
    similar_to: Optional[str] = None,
) -> list:
    path = ledger_path(base_dir, ticker)
    return find_similar_packets(
        query_packet,
        path,
        top_k=top_k,
        exclude_timestamp=query_packet.get("timestamp"),
        exclude_trade_id=query_packet.get("trade_id"),
        filters=filters,
        weights=weights,
        similar_to=similar_to,
    )


def format_query_context(filters: list, similar_to: Optional[str]) -> str:
    parts = []
    if similar_to:
        parts.append(f"component focus `{similar_to}`")
    if filters:
        parts.append("filters " + ", ".join(f"{k}={v}" for k, v in filters))
    if not parts:
        return ""
    return "Investigator query: " + "; ".join(parts) + "."


def format_similar_markdown(
    matches: list,
    instrument: str,
    query_context: str = "",
) -> str:
    lines = [
        f"### Most Similar Historical Packets: {instrument}",
        "",
    ]
    if query_context:
        lines.append(f"_{query_context}_")
        lines.append("")
    if not matches:
        lines.append("_No matching prior packets in ledger yet._")
        lines.append("")
        return "\n".join(lines)

    lines.extend([
        "| Distance | Timestamp | Trade ID | Instrument | Failure | Failure Score | Weighted Score | Status |",
        "| :---: | :--- | :---: | :--- | :--- | :---: | :---: | :--- |",
    ])
    for match in matches:
        lines.append(
            f"| {match['distance']:.3f} | {match['timestamp']} | {match['trade_id']} | "
            f"{match['instrument']} | {match['failure_type']} | {match['failure_score']:.2f} | "
            f"{match['weighted_score']:.2f} | {match['status']} |"
        )
    lines.append("")
    return "\n".join(lines)
