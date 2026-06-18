"""Tests for rupture propagation engine (pilot_upgrade integration)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_field_dashboard import build_pressure_frame
from pressure_field_derivatives import apply_lrp_loop_closure, enrich_pressure_derivatives
from pressure_field_physics import PHYSICS_EXPORT_COLUMNS, enrich_pressure_physics
from pressure_field_schema import STABLE_PROPAGATION_SNAPSHOT_KEYS, build_stable_snapshot
from rupture_propagation_engine import (
    RUPTURE_PROPAGATION_EXPORT_COLUMNS,
    classify_execution_regime_row,
    compute_rupture_propagation,
    detect_regime_phases,
)
from rupture_propagation_experiment import build_experiment_payload, write_rupture_propagation_experiment


def _pipeline_frame(rows: int = 30, *, trend: float = 0.0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = 100.0 + np.cumsum(np.full(rows, trend)) + np.sin(np.linspace(0, 4, rows)) * 0.3
    base = pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 0.2,
            "High": close + 0.3,
            "Low": close - 0.3,
            "Close": close,
            "Volume": np.full(rows, 1_500_000.0),
        }
    )
    frame = build_pressure_frame(base)
    frame = enrich_pressure_derivatives(frame, gamma={})
    frame = enrich_pressure_physics(frame, gamma={})
    return apply_lrp_loop_closure(frame)


class TestRupturePropagationMetrics(unittest.TestCase):
    def test_export_columns_in_physics_pipeline(self):
        frame = _pipeline_frame()
        for col in RUPTURE_PROPAGATION_EXPORT_COLUMNS:
            self.assertIn(col, frame.columns, msg=col)
            self.assertIn(col, PHYSICS_EXPORT_COLUMNS, msg=col)

    def test_bounded_scores(self):
        out = compute_rupture_propagation(_pipeline_frame())
        for col in (
            "synchronization_coefficient",
            "persistence_decay_rate",
            "cascade_energy",
            "restoration_coefficient",
            "restoration_reentry_probability",
            "interpretive_latency",
            "hold_position_score",
            "reduce_exposure_score",
        ):
            series = out[col].dropna()
            self.assertTrue((series >= 0).all(), msg=col)
            self.assertTrue((series <= 1).all(), msg=col)

    def test_persistence_half_life_positive(self):
        out = compute_rupture_propagation(_pipeline_frame())
        self.assertTrue((out["persistence_half_life"].dropna() > 0).all())


class TestExecutionRegimeClassification(unittest.TestCase):
    def test_type_iv_dissipation(self):
        row = pd.Series(
            {
                "persistence_decay_rate": 0.62,
                "cascade_energy": 0.25,
                "synchronization_coefficient": 0.2,
                "interpretive_latency": 0.2,
                "LRP": 0.4,
                "restoration_coefficient": 0.5,
                "F_r": 0.5,
                "B_s": 0.5,
                "T_a_norm": 0.1,
                "price_velocity": -0.001,
            }
        )
        self.assertEqual(classify_execution_regime_row(row), "TYPE_IV_DISSIPATION_TRANSITION")

    def test_type_i_distributed(self):
        row = pd.Series(
            {
                "persistence_decay_rate": 0.1,
                "cascade_energy": 0.2,
                "synchronization_coefficient": 0.15,
                "interpretive_latency": 0.1,
                "LRP": 0.2,
                "restoration_coefficient": 0.6,
                "F_r": 0.55,
                "B_s": 0.35,
                "T_a_norm": 0.1,
                "price_velocity": 0.001,
            }
        )
        self.assertEqual(classify_execution_regime_row(row), "TYPE_I_DISTRIBUTED_EXECUTION")


class TestPhaseDetection(unittest.TestCase):
    def test_detect_regime_phases_returns_segments(self):
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=4, freq="B"),
                "execution_regime": [
                    "TYPE_I_DISTRIBUTED_EXECUTION",
                    "TYPE_I_DISTRIBUTED_EXECUTION",
                    "TYPE_IV_DISSIPATION_TRANSITION",
                    "TYPE_IV_DISSIPATION_TRANSITION",
                ],
                "cascade_energy": [0.2, 0.2, 0.1, 0.1],
                "persistence_decay_rate": [0.1, 0.1, 0.6, 0.6],
            }
        )
        phases = detect_regime_phases(frame)
        self.assertEqual(len(phases), 2)
        self.assertEqual(phases[0]["regime"], "TYPE_I_DISTRIBUTED_EXECUTION")
        self.assertEqual(phases[1]["regime"], "TYPE_IV_DISSIPATION_TRANSITION")


class TestSchemaAndExperiment(unittest.TestCase):
    def test_stable_snapshot_includes_propagation_keys(self):
        frame = _pipeline_frame()
        snap = build_stable_snapshot(ticker="SPY", latest=frame.iloc[-1])
        for key in STABLE_PROPAGATION_SNAPSHOT_KEYS:
            self.assertIn(key, snap, msg=key)

    def test_experiment_writes_outputs(self):
        frame = _pipeline_frame()
        payload = build_experiment_payload(frame, ticker="SPY")
        self.assertEqual(payload["experiment"], "rup_prop_exp")
        self.assertIn("latest", payload)
        self.assertIn("decision", payload)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_rupture_propagation_experiment(
                frame,
                ticker="SPY",
                json_path=root / "rup.json",
                md_path=root / "rup.md",
            )
            self.assertTrue((root / "rup.json").exists())
            self.assertTrue((root / "rup.md").exists())
            self.assertIn("Rupture Propagation Experiment", (root / "rup.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
