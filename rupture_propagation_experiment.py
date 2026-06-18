"""Rupture propagation experiment — phased narrative from CSV replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from canopyento_boundary_engine import load_stock_data_fallback
from pressure_field_dashboard import build_pressure_frame, prepare_ohlcv
from pressure_field_derivatives import apply_lrp_loop_closure, enrich_pressure_derivatives
from pressure_field_physics import enrich_pressure_physics
from rupture_propagation_engine import build_propagation_snapshot, detect_regime_phases

DEFAULT_OUTPUT_JSON = "outputs/rupture_propagation_{ticker}.json"
DEFAULT_OUTPUT_MD = "outputs/rupture_propagation_{ticker}.md"


def build_pressure_frame_with_propagation(
    stock_df: pd.DataFrame,
    *,
    lookback: int = 20,
    tolerance: float = 0.003,
    volume_window: int = 20,
    weekly_window: int = 5,
    gamma: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    frame = build_pressure_frame(
        stock_df,
        lookback=lookback,
        tolerance=tolerance,
        volume_window=volume_window,
        weekly_window=weekly_window,
    )
    frame = enrich_pressure_derivatives(frame, gamma=gamma)
    frame = enrich_pressure_physics(frame, gamma=gamma)
    frame = apply_lrp_loop_closure(frame)
    return frame


def build_experiment_payload(
    frame: pd.DataFrame,
    *,
    ticker: str,
) -> dict[str, Any]:
    latest = frame.iloc[-1]
    snapshot = build_propagation_snapshot(latest)
    phases = detect_regime_phases(frame)
    return {
        "ticker": ticker.upper(),
        "experiment": "rup_prop_exp",
        "description": "Rupture propagation phase model — pressure accumulation through dissipation",
        "latest": snapshot,
        "phases": phases,
        "phase_sequence": [p["regime"] for p in phases],
        "decision": {
            "hold_position_score": snapshot["hold_position_score"],
            "reduce_exposure_score": snapshot["reduce_exposure_score"],
            "recommendation": _recommendation(snapshot),
        },
    }


def _recommendation(snapshot: dict[str, Any]) -> str:
    hold = float(snapshot.get("hold_position_score", 0.0))
    reduce = float(snapshot.get("reduce_exposure_score", 0.0))
    regime = str(snapshot.get("execution_regime", ""))
    if regime == "TYPE_IV_DISSIPATION_TRANSITION" or reduce > hold + 0.12:
        return "reduce_exposure"
    if regime in ("TYPE_II_COLLECTIVE_EXECUTION", "TYPE_III_REFLEXIVE_COLLECTIVE_EXECUTION") and hold > reduce:
        return "hold_through_cascade"
    if regime == "TYPE_IIb_INTERRUPTED_EXECUTION":
        return "wait_for_resolution"
    return "neutral"


def render_experiment_markdown(payload: dict[str, Any]) -> str:
    latest = payload["latest"]
    lines = [
        f"# Rupture Propagation Experiment — {payload['ticker']}",
        "",
        "Aerotrader rupture propagation phase model (CSV replay). See `pilot_upgrade.md` and `Glossary.md`.",
        "",
        "## Latest State",
        "",
        f"- **Execution regime:** {latest.get('execution_regime') or 'unclassified'}",
        f"- **Synchronization coefficient:** {latest.get('synchronization_coefficient', 0):.3f}",
        f"- **Persistence decay rate (P_d):** {latest.get('persistence_decay_rate', 0):.3f}",
        f"- **Cascade energy:** {latest.get('cascade_energy', 0):.3f}",
        f"- **Persistence half-life (P_h):** {latest.get('persistence_half_life', 0):.1f} bars",
        f"- **Restoration coefficient (R_c):** {latest.get('restoration_coefficient', 0):.3f}",
        f"- **Restoration reentry probability:** {latest.get('restoration_reentry_probability', 0):.3f}",
        f"- **Interpretive latency (I_l):** {latest.get('interpretive_latency', 0):.3f}",
        f"- **Dissipation onset flag:** {latest.get('dissipation_onset_flag', 0)}",
        "",
        "## Decision Framework",
        "",
        f"- **Hold position score:** {latest.get('hold_position_score', 0):.3f}",
        f"- **Reduce exposure score:** {latest.get('reduce_exposure_score', 0):.3f}",
        f"- **Recommendation:** {payload['decision']['recommendation']}",
        "",
        "## Phase Timeline",
        "",
    ]
    phases = payload.get("phases", [])
    if not phases:
        lines.append("_No classified execution phases in lookback window._")
    else:
        for phase in phases:
            lines.append(
                f"- **{phase['regime']}** — {phase['start_date']} → {phase['end_date']} "
                f"({phase['bars']} bars, cascade={phase.get('mean_cascade_energy', 0):.3f}, "
                f"P_d={phase.get('mean_persistence_decay_rate', 0):.3f})"
            )
    lines.extend(
        [
            "",
            "## Phase Sequence",
            "",
            " → ".join(payload.get("phase_sequence", [])) or "_none_",
            "",
        ]
    )
    return "\n".join(lines)


def write_rupture_propagation_experiment(
    frame: pd.DataFrame,
    *,
    ticker: str,
    json_path: Path,
    md_path: Path,
) -> dict[str, Any]:
    payload = build_experiment_payload(frame, ticker=ticker)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_experiment_markdown(payload), encoding="utf-8")
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rupture propagation experiment (rup_prop_exp).")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol")
    parser.add_argument("--stock-dir", default="data/stock", help="Stock CSV directory")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=0.003)
    parser.add_argument("--volume-window", type=int, default=20, dest="volume_window")
    parser.add_argument("--weekly-window", type=int, default=5, dest="weekly_window")
    parser.add_argument("--json", default=None, dest="json_output")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    ticker = args.ticker.upper()
    json_path = Path(args.json_output or DEFAULT_OUTPUT_JSON.format(ticker=ticker))
    md_path = Path(args.report or DEFAULT_OUTPUT_MD.format(ticker=ticker))

    try:
        stock_df = load_stock_data_fallback(ticker, base_dir=args.stock_dir)
        prepare_ohlcv(stock_df)
        frame = build_pressure_frame_with_propagation(
            stock_df,
            lookback=args.lookback,
            tolerance=args.tolerance,
            volume_window=args.volume_window,
            weekly_window=args.weekly_window,
        )
        payload = write_rupture_propagation_experiment(frame, ticker=ticker, json_path=json_path, md_path=md_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Latest regime: {payload['latest'].get('execution_regime') or 'unclassified'}")
    print(f"Recommendation: {payload['decision']['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
