"""Tests for pressure-field physics engines (Phases 1–3)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from capillary_wave_engine import CAPILLARY_WAVE_REGIMES, compute_capillary_wave
from field_regime_engine import NAMED_FIELD_REGIMES, classify_field_regime_row, compute_field_regimes
from pressure_field_dashboard import build_pressure_frame, prepare_ohlcv
from pressure_field_derivatives import apply_lrp_loop_closure, enrich_pressure_derivatives
from pressure_field_physics import PHYSICS_EXPORT_COLUMNS, enrich_pressure_physics
from restoration_field_engine import compute_restoration_field


def _synthetic_ohlcv(rows: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = 100.0 + np.linspace(0, 4, rows) + np.sin(np.linspace(0, 6, rows)) * 0.8
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 0.2,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(rows, 1_000_000.0),
        }
    )


class TestRestorationField(unittest.TestCase):
    def test_f_r_and_d_c_bounded(self):
        frame = build_pressure_frame(_synthetic_ohlcv())
        out = compute_restoration_field(frame)
        self.assertTrue((out["F_r"].dropna() >= 0).all())
        self.assertTrue((out["F_r"].dropna() <= 1).all())
        self.assertTrue((out["D_c"].dropna() >= 0).all())
        self.assertTrue((out["D_c"].dropna() <= 1).all())

    def test_restoration_ratio_present(self):
        frame = compute_restoration_field(build_pressure_frame(_synthetic_ohlcv()))
        self.assertIn("restoration_ratio", frame.columns)
        self.assertIn("dissipation_score", frame.columns)


class TestCapillaryWave(unittest.TestCase):
    def test_capillary_wave_metrics_bounded(self):
        base = compute_restoration_field(build_pressure_frame(_synthetic_ohlcv()))
        out = compute_capillary_wave(base)
        for col in ("A_micro", "wave_persistence", "C_w", "capillary_wave_score", "capillary_wave_regime"):
            self.assertIn(col, out.columns)
            if col != "capillary_wave_regime":
                series = out[col].dropna()
                self.assertTrue((series >= 0).all())
                self.assertTrue((series <= 1).all())

    def test_capillary_wave_regime_in_ladder(self):
        out = compute_capillary_wave(compute_restoration_field(build_pressure_frame(_synthetic_ohlcv())))
        regime = str(out.iloc[-1]["capillary_wave_regime"])
        self.assertIn(regime, {"", *CAPILLARY_WAVE_REGIMES})


class TestPressurePhysicsPipeline(unittest.TestCase):
    def test_enrich_pressure_physics_exports(self):
        frame = build_pressure_frame(_synthetic_ohlcv())
        frame = enrich_pressure_derivatives(frame, gamma={})
        out = enrich_pressure_physics(frame, gamma={})
        for col in PHYSICS_EXPORT_COLUMNS:
            self.assertIn(col, out.columns)

    def test_full_loop_closure_pipeline(self):
        frame = apply_lrp_loop_closure(
            enrich_pressure_physics(
                enrich_pressure_derivatives(build_pressure_frame(_synthetic_ohlcv()), gamma={}),
                gamma={},
            )
        )
        self.assertIn("LRP_adjusted", frame.columns)
        self.assertIn("LRP", frame.columns)

    def test_field_regime_is_sacred_named_or_empty(self):
        frame = apply_lrp_loop_closure(
            enrich_pressure_physics(
                enrich_pressure_derivatives(build_pressure_frame(_synthetic_ohlcv()), gamma={}),
                gamma={},
            )
        )
        regime = str(frame.iloc[-1]["field_regime"])
        self.assertTrue(
            regime in {"", *NAMED_FIELD_REGIMES}
            or regime == "COHERENT_MAINTENANCE_UNDER_STRESS"
        )


class TestFieldRegimeClassification(unittest.TestCase):
    def test_restored_equilibrium_regime(self):
        row = pd.Series(
            {
                "restoration_ratio": 0.85,
                "F_r": 0.7,
                "D_c": 0.6,
                "LRP": 0.2,
                "C_w": 0.2,
                "A_micro": 0.2,
                "entropy_score": 0.3,
                "d_F_r": 0.0,
                "wall_pinning_strength": 0.0,
                "recovery_incomplete_flag": 0,
                "H_s": 0.1,
            }
        )
        self.assertEqual(classify_field_regime_row(row), "RESTORED_EQUILIBRIUM")

    def test_synthetic_stability_regime(self):
        row = pd.Series(
            {
                "restoration_ratio": 0.80,
                "F_r": 0.72,
                "D_c": 0.45,
                "LRP": 0.22,
                "C_w": 0.25,
                "A_micro": 0.42,
                "entropy_score": 0.35,
                "d_F_r": 0.0,
                "wall_pinning_strength": 0.55,
                "recovery_incomplete_flag": 0,
                "H_s": 0.15,
                "hidden_reservoir_pressure": 0.30,
                "false_stability_flag": 0,
            }
        )
        self.assertEqual(classify_field_regime_row(row), "SYNTHETIC_STABILITY")

    def test_capillary_pre_rupture_regime(self):
        row = pd.Series(
            {
                "restoration_ratio": 0.5,
                "F_r": 0.5,
                "D_c": 0.4,
                "LRP": 0.5,
                "C_w": 0.75,
                "A_micro": 0.55,
                "entropy_score": 0.4,
                "d_F_r": 0.0,
                "wall_pinning_strength": 0.0,
                "recovery_incomplete_flag": 0,
                "H_s": 0.2,
            }
        )
        self.assertEqual(classify_field_regime_row(row), "CAPILLARY_PRE_RUPTURE")


if __name__ == "__main__":
    unittest.main()
