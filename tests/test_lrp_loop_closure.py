"""Loop-closure tests — LRP_adjusted vs baseline LRP."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from attractor_engine import compute_attractor_field
from capillary_wave_engine import CAPILLARY_WAVE_REGIMES, classify_capillary_wave_regime, compute_capillary_wave
from field_regime_engine import classify_field_regime_row
from inward_drift_engine import compute_inward_drift
from pressure_field_derivatives import apply_lrp_loop_closure, enrich_pressure_derivatives
from pressure_field_schema import build_stable_snapshot, safe_float
from restoration_field_engine import compute_restoration_field


def _base_row(**overrides) -> pd.DataFrame:
    row = {
        "LRP_raw": 0.55,
        "LRP": 0.52,
        "LRP_regime": "PRESSURE_BUILDING",
        "restoration_ratio": 1.0,
        "F_r": 0.5,
        "D_c": 0.5,
        "C_w": 0.2,
        "recursive_pressure_carryover": 0.1,
        "post_stress_sensitivity": 1.0,
        "observer_feedback_score": 0.1,
        "O_i": 0.1,
        "rupture_pressure_score": 0.4,
        "Close": 100.0,
        "vwap_distance_pct": 0.5,
        "vwap": 99.5,
        "R_o": 0.6,
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestLrpLoopClosure(unittest.TestCase):
    def test_strong_restoration_lowers_lrp_adjusted(self):
        weak = apply_lrp_loop_closure(_base_row(restoration_ratio=0.5, F_r=0.4))
        strong = apply_lrp_loop_closure(_base_row(restoration_ratio=2.5, F_r=0.85))
        self.assertLess(float(strong.iloc[0]["LRP_adjusted"]), float(weak.iloc[0]["LRP_adjusted"]))
        self.assertEqual(float(strong.iloc[0]["LRP"]), float(weak.iloc[0]["LRP"]))

    def test_weak_restoration_high_cw_raises_lrp_adjusted(self):
        calm = apply_lrp_loop_closure(_base_row(restoration_ratio=1.2, C_w=0.1))
        stressed = apply_lrp_loop_closure(_base_row(restoration_ratio=0.4, C_w=0.8))
        self.assertGreater(float(stressed.iloc[0]["LRP_adjusted"]), float(calm.iloc[0]["LRP_adjusted"]))

    def test_hysteresis_increases_sensitivity(self):
        low = apply_lrp_loop_closure(_base_row(recursive_pressure_carryover=0.05, post_stress_sensitivity=1.0))
        high = apply_lrp_loop_closure(_base_row(recursive_pressure_carryover=0.75, post_stress_sensitivity=1.25))
        self.assertGreater(float(high.iloc[0]["hysteresis_boost"]), float(low.iloc[0]["hysteresis_boost"]))
        self.assertGreater(float(high.iloc[0]["LRP_adjusted"]), float(low.iloc[0]["LRP_adjusted"]))

    def test_baseline_lrp_unchanged_by_loop_closure(self):
        frame = _base_row()
        before = float(frame.iloc[0]["LRP"])
        after = apply_lrp_loop_closure(frame)
        self.assertEqual(float(after.iloc[0]["LRP"]), before)
        self.assertIn("LRP_adjusted", after.columns)

    def test_adjustment_terms_bounded(self):
        out = apply_lrp_loop_closure(_base_row())
        row = out.iloc[0]
        self.assertGreaterEqual(float(row["restoration_damper"]), 0.55)
        self.assertLessEqual(float(row["restoration_damper"]), 1.15)
        self.assertLessEqual(float(row["capillary_boost"]), 1.40)
        self.assertLessEqual(float(row["hysteresis_boost"]), 1.35)
        self.assertLessEqual(float(row["observer_boost"]), 1.25)
        self.assertGreaterEqual(float(row["LRP_adjusted"]), 0.0)
        self.assertLessEqual(float(row["LRP_adjusted"]), 1.0)


class TestGammaAttractor(unittest.TestCase):
    def test_gamma_pinning_active_compensation(self):
        frame = pd.DataFrame(
            [
                {
                    "Close": 102.5,
                    "vwap": 100.0,
                    "vwap_distance_pct": 2.5,
                    "F_r": 0.55,
                    "D_c": 0.5,
                    "gamma_flip_distance_pct": 1.2,
                    "LRP": 0.42,
                    "C_w": 0.2,
                    "A_micro": 0.2,
                    "entropy_score": 0.3,
                    "recovery_incomplete_flag": 0,
                    "H_s": 0.2,
                    "restoration_ratio": 1.1,
                }
            ]
        )
        gamma = {"gamma_flip_strike": 100.0}
        out = compute_attractor_field(frame, gamma=gamma)
        row = out.iloc[0]
        self.assertGreater(float(row["wall_pinning_strength"]), 0.5)
        regime_row = row.copy()
        regime_row["field_regime"] = classify_field_regime_row(regime_row)
        self.assertEqual(classify_field_regime_row(regime_row), "ACTIVE_COMPENSATION")

    def test_missing_gamma_does_not_break_attractor(self):
        frame = pd.DataFrame([{"Close": 100.0, "vwap": 99.0, "F_r": 0.5}])
        out = compute_attractor_field(frame, gamma={})
        self.assertIn("A_field", out.columns)
        self.assertEqual(float(out.iloc[0]["gamma_attractor_strength"]), 0.0)


class TestCapillaryWaveRegimes(unittest.TestCase):
    def test_regime_ladder(self):
        self.assertEqual(classify_capillary_wave_regime(0.1), "FLAT_SURFACE")
        self.assertEqual(classify_capillary_wave_regime(0.35), "CAPILLARY_ACTIVITY")
        self.assertEqual(classify_capillary_wave_regime(0.55), "SURFACE_TENSION_STRESS")
        self.assertEqual(classify_capillary_wave_regime(0.8), "PRE_RUPTURE")
        self.assertIn("FLAT_SURFACE", CAPILLARY_WAVE_REGIMES)


class TestIdiAndNaming(unittest.TestCase):
    def test_idi_bounded(self):
        dates = pd.date_range("2024-01-01", periods=15, freq="B")
        close = 100 + np.sin(np.linspace(0, 4, 15))
        frame = pd.DataFrame(
            {
                "Date": dates,
                "Close": close,
                "High": close + 0.5,
                "Low": close - 0.5,
                "vwap_distance_pct": np.linspace(1.0, 0.2, 15),
                "cvd_imbalance": np.linspace(0.1, 0.4, 15),
                "volume_injection": np.linspace(1.0, 1.3, 15),
            }
        )
        out = compute_inward_drift(frame)
        self.assertTrue((out["IDI"].dropna() >= 0).all())
        self.assertTrue((out["IDI"].dropna() <= 1).all())

    def test_a_micro_not_ambiguous_with_attractor(self):
        base = compute_restoration_field(
            pd.DataFrame(
                {
                    "Close": [100.0, 101.0],
                    "vwap_distance_pct": [0.5, 0.4],
                    "B_s": [0.3, 0.3],
                    "rsi": [50, 50],
                    "macd_histogram": [0.1, 0.1],
                    "rupture_pressure_score": [0.3, 0.3],
                    "volume_injection": [1.0, 1.0],
                    "E_i": [0.2, 0.2],
                }
            )
        )
        wave = compute_capillary_wave(base)
        attractor = compute_attractor_field(wave, gamma={"gamma_flip_strike": 100.0})
        self.assertIn("A_micro", wave.columns)
        self.assertIn("A_field", attractor.columns)
        self.assertNotIn("A_f", wave.columns)


class TestJsonSafety(unittest.TestCase):
    def test_snapshot_no_nan_inf(self):
        latest = apply_lrp_loop_closure(_base_row()).iloc[0]
        snap = build_stable_snapshot(ticker="SPY", latest=latest, spot=100.0, gamma={})
        payload = {k: snap[k] for k in snap}
        text = json.dumps(payload)
        self.assertIsInstance(text, str)
        for key in ("LRP", "LRP_adjusted", "F_r", "D_c"):
            if key in payload and payload[key] is not None:
                self.assertTrue(safe_float(payload[key]) is not None or payload[key] is None)

    def test_core_snapshot_keys_precede_extended(self):
        from pressure_field_schema import STABLE_CORE_SNAPSHOT_KEYS, STABLE_EXTENDED_SNAPSHOT_KEYS

        latest = apply_lrp_loop_closure(_base_row()).iloc[0]
        snap = build_stable_snapshot(ticker="SPY", latest=latest, spot=100.0, gamma={})
        keys = list(snap.keys())
        core_end = keys.index(STABLE_CORE_SNAPSHOT_KEYS[-1])
        ext_start = keys.index(STABLE_EXTENDED_SNAPSHOT_KEYS[0])
        self.assertLess(core_end, ext_start)

    def test_no_option_chain_pipeline_succeeds(self):
        from pressure_field_dashboard import build_pressure_frame, empty_gamma_snapshot

        from pressure_field_physics import enrich_pressure_physics

        frame = build_pressure_frame(_synthetic_ohlcv(25))
        gamma = empty_gamma_snapshot()
        frame = enrich_pressure_derivatives(frame, gamma=gamma)
        frame = enrich_pressure_physics(frame, gamma=gamma)
        out = apply_lrp_loop_closure(frame)
        latest = out.iloc[-1]
        self.assertIn("LRP_adjusted", out.columns)
        self.assertEqual(float(latest["gamma_attractor_strength"]), 0.0)
        self.assertEqual(float(latest["wall_pinning_strength"]), 0.0)
        self.assertTrue(pd.isna(latest.get("gamma_flip_distance_pct")))
        snap = build_stable_snapshot(ticker="SPY", latest=latest, spot=100.0, gamma=gamma)
        self.assertIsNone(snap["gamma_flip"])


def _synthetic_ohlcv(rows: int = 25) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = 100.0 + np.linspace(0, 2, rows)
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


if __name__ == "__main__":
    unittest.main()
