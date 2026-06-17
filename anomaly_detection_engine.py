"""Anomaly detection for sparse / IPO option chains vs benchmark."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

ANOMALY_LABELS = (
    "NORMAL",
    "POST_EVENT_EXPANSION",
    "IPO_INSTABILITY",
    "GAMMA_COMPRESSION_EVENT",
    "CALL_FOMO_DETECTED",
    "DEALER_STRESS_EVENT",
    "PUT_FEAR_DETECTED",
    "SKEW_INVERSION",
)


def build_anomaly_features(
    *,
    skew_metrics: dict[str, Any],
    pressure_metrics: dict[str, Any],
    benchmark_skew: Optional[dict[str, Any]] = None,
    data_health: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    atm_iv = skew_metrics.get("atm_iv") or 0.0
    put_wing = skew_metrics.get("put_wing_iv")
    call_wing = skew_metrics.get("call_wing_iv")

    bench_atm = (benchmark_skew or {}).get("atm_iv") or atm_iv
    volatility_ratio = round(float(atm_iv) / max(float(bench_atm), 0.01), 4) if bench_atm else None

    call_skew_ratio = None
    put_skew_ratio = None
    if atm_iv and atm_iv > 0:
        if call_wing is not None:
            call_skew_ratio = round(float(call_wing) / float(atm_iv), 4)
        if put_wing is not None:
            put_skew_ratio = round(float(put_wing) / float(atm_iv), 4)

    return {
        "volatility_ratio": volatility_ratio,
        "call_skew_ratio": call_skew_ratio,
        "put_skew_ratio": put_skew_ratio,
        "surface_curvature": skew_metrics.get("surface_curvature", 0.0),
        "volatility_expansion_score": pressure_metrics.get("volatility_expansion_score"),
        "gamma_compression_score": pressure_metrics.get("gamma_compression_score"),
        "dealer_hedging_stress_score": pressure_metrics.get("dealer_hedging_stress_score"),
        "data_status": (data_health or {}).get("status", "UNKNOWN"),
    }


def classify_anomalies(features: dict[str, Any], *, skew_metrics: dict[str, Any]) -> dict[str, Any]:
    """Classify chain anomalies from feature bundle."""
    labels: list[str] = []
    vol_ratio = features.get("volatility_ratio")
    vol_exp = features.get("volatility_expansion_score")
    gamma_c = features.get("gamma_compression_score") or 0.0
    dealer = features.get("dealer_hedging_stress_score") or 0.0
    curvature = features.get("surface_curvature") or 0.0

    if vol_ratio is not None and vol_ratio > 3.0:
        labels.append("IPO_INSTABILITY")
    if vol_exp is not None and vol_exp > 2.5:
        labels.append("POST_EVENT_EXPANSION")
    if gamma_c > 0.55:
        labels.append("GAMMA_COMPRESSION_EVENT")
    if skew_metrics.get("call_fomo_flag") or skew_metrics.get("calls_overpriced_flag"):
        labels.append("CALL_FOMO_DETECTED")
    if skew_metrics.get("put_fear_flag") or skew_metrics.get("puts_overpriced_flag"):
        labels.append("PUT_FEAR_DETECTED")
    if dealer > 0.6:
        labels.append("DEALER_STRESS_EVENT")
    if skew_metrics.get("skew_inversion_flag"):
        labels.append("SKEW_INVERSION")
    if curvature > 0.05 and vol_ratio is not None and vol_ratio > 2.0:
        labels.append("POST_EVENT_EXPANSION")

    if not labels:
        labels.append("NORMAL")

    severity = min(1.0, len([l for l in labels if l != "NORMAL"]) * 0.2 + (vol_ratio or 1.0) / 10.0)

    return {
        "labels": labels,
        "primary_label": labels[0],
        "severity_score": round(float(severity), 4),
        "features": features,
    }


def detect_anomalies(
    *,
    skew_metrics: dict[str, Any],
    pressure_metrics: dict[str, Any],
    benchmark_skew: Optional[dict[str, Any]] = None,
    data_health: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    features = build_anomaly_features(
        skew_metrics=skew_metrics,
        pressure_metrics=pressure_metrics,
        benchmark_skew=benchmark_skew,
        data_health=data_health,
    )
    return classify_anomalies(features, skew_metrics=skew_metrics)
