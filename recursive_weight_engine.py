import json
from pathlib import Path

import numpy as np

WEIGHT_MIN = 0.1
WEIGHT_MAX = 2.5
DEFAULT_LR = 0.05

COMPONENT_MAP = [
    ("direction", "direction_score"),
    ("timing", "timing_score"),
    ("magnitude", "magnitude_score"),
    ("volatility", "volatility_score"),
    ("catalyst", "catalyst_score"),
    ("exit", "exit_score"),
]

DEFAULT_WEIGHTS = {
    "direction": 1.0,
    "timing": 1.0,
    "magnitude": 1.0,
    "volatility": 1.0,
    "catalyst": 1.0,
    "exit": 1.0,
    "theta_risk": 1.0,
    "cvd_confirmation": 1.0,
}


def clamp_weight(value: float) -> float:
    return float(np.clip(value, WEIGHT_MIN, WEIGHT_MAX))


def initialize_default_weights() -> dict:
    return dict(DEFAULT_WEIGHTS)


def load_weights(path) -> dict:
    path = Path(path)
    if not path.is_file():
        weights = initialize_default_weights()
        save_weights(weights, path)
        return weights

    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    weights = initialize_default_weights()
    weights.update({key: float(loaded[key]) for key in weights if key in loaded})
    return weights


def save_weights(weights: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: clamp_weight(float(weights.get(key, DEFAULT_WEIGHTS[key]))) for key in DEFAULT_WEIGHTS}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _cvd_value(packet: dict) -> float:
    value = packet.get("cvd_confirmation", 0)
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in {"1", "true", "yes"} else 0.0
    return 1.0 if bool(value) else 0.0


def score_packet(packet: dict, weights: dict) -> float:
    """Weighted failure score in [0, 1]; higher means worse packet quality."""
    weighted_penalties = []
    weight_sum = 0.0

    for key, score_field in COMPONENT_MAP:
        score = float(packet[score_field])
        penalty = 0.5 * (1.0 - score)
        weight = float(weights.get(key, 1.0))
        weighted_penalties.append(weight * penalty)
        weight_sum += weight

    theta_weight = float(weights.get("theta_risk", 1.0))
    cvd_weight = float(weights.get("cvd_confirmation", 1.0))
    weighted_penalties.append(theta_weight * float(packet.get("theta_risk", 0.0)))
    weighted_penalties.append(cvd_weight * (1.0 - _cvd_value(packet)))
    weight_sum += theta_weight + cvd_weight

    if weight_sum <= 0.0:
        return 0.0
    return float(sum(weighted_penalties) / weight_sum)


def update_weights(packet: dict, weights: dict, lr: float = DEFAULT_LR) -> dict:
    """Apply a localized recursive update from one observed packet."""
    updated = dict(weights)

    for key, score_field in COMPONENT_MAP:
        score = float(packet[score_field])
        penalty = 0.5 * (1.0 - score)
        updated[key] = clamp_weight(float(updated.get(key, 1.0)) + lr * penalty)

    updated["theta_risk"] = clamp_weight(
        float(updated.get("theta_risk", 1.0)) + lr * float(packet.get("theta_risk", 0.0))
    )
    updated["cvd_confirmation"] = clamp_weight(
        float(updated.get("cvd_confirmation", 1.0)) + lr * (1.0 - _cvd_value(packet))
    )
    return updated


def weights_path(base_dir, ticker: str) -> Path:
    from packet_persistence import outputs_dir

    return outputs_dir(base_dir) / f"recursive_weights_{ticker.upper()}.json"


def format_weights_markdown(weights: dict, ticker: str) -> str:
    lines = [
        f"## Model Weights After Update ({ticker})",
        "",
        "| Component | Weight |",
        "| :--- | :---: |",
    ]
    for key in DEFAULT_WEIGHTS:
        lines.append(f"| {key.replace('_', ' ').title()} | {weights[key]:.3f} |")
    lines.append("")
    return "\n".join(lines)


FAILURE_TYPE_TO_COMPONENT = {
    "Direction": "direction",
    "Timing": "timing",
    "Magnitude": "magnitude",
    "Volatility": "volatility",
    "Catalyst": "catalyst",
    "Exit Discipline": "exit",
}


def _component_penalty(score: float) -> float:
    return 0.5 * (1.0 - float(score))


def format_theory_line(
    packet: dict,
    similar_matches: list,
    weights_before: dict,
    weights_after: dict,
) -> str:
    failure_type = str(packet.get("failure_type", "unknown"))
    component = FAILURE_TYPE_TO_COMPONENT.get(failure_type)

    if not component:
        return (
            "Updated theory: hold component weights steady; current packet resolved as "
            f"`{failure_type}` without a single dominant penalty channel."
        )

    if not similar_matches:
        delta = weights_after[component] - weights_before[component]
        direction = "increase" if delta > 0.001 else "decrease" if delta < -0.001 else "hold"
        return (
            f"Updated theory: {direction} `{component}` weight from the current packet only; "
            "no similar historical cases matched the investigator query."
        )

    same_failure = sum(
        1 for match in similar_matches
        if str(match.get("failure_type", "")).lower() == failure_type.lower()
    )
    score_field = next(field for key, field in COMPONENT_MAP if key == component)
    recurring_penalty = sum(
        1 for match in similar_matches
        if _component_penalty(float(match.get(score_field, 0.0))) >= 0.35
    )

    delta = weights_after[component] - weights_before[component]
    direction = "increase" if delta > 0.001 else "decrease" if delta < -0.001 else "hold"
    total = len(similar_matches)

    return (
        f"Updated theory: {direction} `{component}` weight because `{failure_type}` recurred in "
        f"{same_failure}/{total} similar cases and `{component}` penalty appeared in "
        f"{recurring_penalty}/{total} of them."
    )
