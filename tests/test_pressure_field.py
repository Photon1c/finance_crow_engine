"""Lightweight tests for pressure field hardening and schema consistency."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from canopyento_boundary_engine import (
    compute_boundary_metrics,
    compute_observational_resolution,
    compute_observer_differential_metrics,
    compute_visibility_horizon,
)
from pressure_field_dashboard import (
    build_pressure_frame,
    compute_gamma_flip,
    compute_macd,
    compute_rsi,
    compute_vwap_field,
    empty_gamma_snapshot,
    prepare_ohlcv,
)
from pressure_field_derivatives import (
    build_lrp_debug_payload,
    compute_rate_of_change_alerts,
    enrich_pressure_derivatives,
)
from pressure_field_schema import STABLE_SNAPSHOT_KEYS, build_stable_snapshot, safe_float


def _synthetic_ohlcv(rows: int = 30, *, flat: bool = False, zero_volume: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    if flat:
        close = np.full(rows, 100.0)
    else:
        close = 100.0 + np.linspace(0, 5, rows) + np.sin(np.linspace(0, 3, rows))
    volume = np.zeros(rows) if zero_volume else np.full(rows, 1_000_000.0)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 0.2,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": volume,
        }
    )


class TestIndicatorRobustness(unittest.TestCase):
    def test_macd_rsi_small_csv_does_not_crash(self):
        df = prepare_ohlcv(_synthetic_ohlcv(5))
        out = compute_rsi(compute_macd(df))
        self.assertEqual(len(out), 5)
        self.assertIn("macd_histogram", out.columns)
        self.assertIn("rsi", out.columns)

    def test_vwap_zero_volume_does_not_produce_inf(self):
        df = prepare_ohlcv(_synthetic_ohlcv(10, zero_volume=True))
        out = compute_vwap_field(df, window=5)
        self.assertFalse(np.isinf(out["vwap_distance_pct"].fillna(0)).any())
        self.assertTrue(out["vwap"].isna().all() or out["vwap"].notna().any())


class TestGammaFlipSafety(unittest.TestCase):
    def test_missing_option_chain_returns_null(self):
        gamma = compute_gamma_flip(None, 100.0)
        self.assertIsNone(gamma["gamma_flip_strike"])
        self.assertEqual(gamma["gamma_regime"], "NO_CHAIN")

    def test_empty_option_chain_returns_null(self):
        gamma = compute_gamma_flip(pd.DataFrame(), 100.0)
        self.assertIs(gamma["gamma_flip_strike"], None)

    def test_zero_oi_returns_zero_oi_regime(self):
        option_df = pd.DataFrame(
            {
                "Strike": [95, 100, 105],
                "Gamma": [0.01, 0.02, 0.01],
                "Gamma.1": [0.01, 0.02, 0.01],
                "Open Interest": [0, 0, 0],
                "Open Interest.1": [0, 0, 0],
            }
        )
        gamma = compute_gamma_flip(option_df, 100.0)
        self.assertEqual(gamma["gamma_regime"], "ZERO_OI")
        self.assertIsNone(gamma["gamma_flip_strike"])

    def test_empty_gamma_snapshot_is_json_safe(self):
        snap = empty_gamma_snapshot()
        json.dumps(snap)


class TestObserverDifferentialBounds(unittest.TestCase):
    def test_r_o_clamped_to_unit_interval(self):
        row = pd.Series(
            {
                "B_s": 1.0,
                "rupture_pressure_score": 2.0,
                "hidden_process_uncertainty": 1.0,
                "regime_label": "RUPTURE_CANDIDATE",
                "regime_persistence": 10,
            }
        )
        r_o = compute_observational_resolution(row, t_a_norm=5.0)
        self.assertGreaterEqual(r_o, 0.0)
        self.assertLessEqual(r_o, 1.0)

    def test_t_v_clamped_0_to_10(self):
        self.assertEqual(compute_visibility_horizon(0.0), 10.0)
        self.assertEqual(compute_visibility_horizon(1.0), 0.0)
        tv = compute_visibility_horizon(0.5)
        self.assertGreaterEqual(tv, 0.0)
        self.assertLessEqual(tv, 10.0)

    def test_flat_pressure_series_no_inf_t_a_norm(self):
        df = compute_boundary_metrics(prepare_ohlcv(_synthetic_ohlcv(25, flat=True)))
        out = compute_observer_differential_metrics(df)
        norms = out["T_a_norm"].replace([np.inf, -np.inf], np.nan).dropna()
        self.assertTrue((norms.abs() < 1e6).all() or norms.empty)

    def test_observer_metrics_deterministic(self):
        df = compute_boundary_metrics(prepare_ohlcv(_synthetic_ohlcv(30)))
        first = compute_observer_differential_metrics(df.copy())
        second = compute_observer_differential_metrics(df.copy())
        pd.testing.assert_series_equal(first["R_o"], second["R_o"], check_names=False)


class TestDerivedMetrics(unittest.TestCase):
    def test_first_row_derivatives_are_zero(self):
        df = build_pressure_frame(_synthetic_ohlcv(12), lookback=3, volume_window=3, weekly_window=2)
        enriched = enrich_pressure_derivatives(df, gamma=empty_gamma_snapshot())
        self.assertEqual(float(enriched.iloc[0]["d_macd_pressure"]), 0.0)
        self.assertEqual(float(enriched.iloc[0]["d_canopy_pressure"]), 0.0)

    def test_lrp_bounded_0_to_1(self):
        df = build_pressure_frame(_synthetic_ohlcv(40))
        enriched = enrich_pressure_derivatives(df, gamma={"gamma_flip_strike": 100.0})
        lrp = enriched["LRP"].dropna()
        self.assertTrue((lrp >= 0.0).all())
        self.assertTrue((lrp <= 1.0).all())
        self.assertLess(float(lrp.max()), 0.99)

    def test_lrp_sigmoid_avoids_mass_saturation(self):
        df = build_pressure_frame(_synthetic_ohlcv(60))
        enriched = enrich_pressure_derivatives(df, gamma={"gamma_flip_strike": 100.0})
        share_extreme = float((enriched["LRP"] >= 0.95).mean())
        self.assertLess(share_extreme, 0.10)

    def test_lrp_confidence_labels(self):
        from pressure_field_derivatives import compute_lrp_confidence

        row = pd.Series({"T_a_norm": 0.5, "R_o": 0.5, "rupture_pressure_score": 0.2})
        label = compute_lrp_confidence(row, gamma_available=True, history_depth=25)
        self.assertIn(label, {"HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE"})

    def test_lrp_debug_payload_structure(self):
        df = enrich_pressure_derivatives(build_pressure_frame(_synthetic_ohlcv(40)), gamma={"gamma_flip_strike": 100.0})
        payload = build_lrp_debug_payload(df, ticker="TEST")
        self.assertIn("legacy_formula_audit", payload)
        self.assertIn("calibrated_model", payload)
        self.assertIn("component_contributions", payload["calibrated_model"])
        self.assertIn("historical_calibration", payload)

    def test_lrp_regime_labels(self):
        from pressure_field_derivatives import classify_lrp_regime

        self.assertEqual(classify_lrp_regime(0.1), "STABLE")
        self.assertEqual(classify_lrp_regime(0.45), "PRESSURE_BUILDING")
        self.assertEqual(classify_lrp_regime(0.7), "PRE_RUPTURE")
        self.assertEqual(classify_lrp_regime(0.9), "RUPTURE_IMMINENT")

    def test_missing_gamma_produces_null_distance_and_safe_lrp(self):
        df = build_pressure_frame(_synthetic_ohlcv(20), lookback=3, volume_window=3, weekly_window=2)
        enriched = enrich_pressure_derivatives(df, gamma=empty_gamma_snapshot())
        self.assertTrue(enriched["gamma_flip_distance_pct"].isna().all())
        self.assertTrue(enriched["LRP"].notna().any())

    def test_rate_of_change_alerts_returns_list(self):
        df = enrich_pressure_derivatives(build_pressure_frame(_synthetic_ohlcv(30)), gamma={"gamma_flip_strike": 102.0})
        alerts = compute_rate_of_change_alerts(df.iloc[-1], previous=df.iloc[-2])
        self.assertIsInstance(alerts, list)


class TestStableSnapshotSchema(unittest.TestCase):
    def test_build_stable_snapshot_keys(self):
        df = enrich_pressure_derivatives(
            build_pressure_frame(_synthetic_ohlcv(40)),
            gamma={"gamma_flip_strike": 100.0},
        )
        latest = df.iloc[-1]
        snap = build_stable_snapshot(ticker="TEST", latest=latest, spot=100.0, gamma=empty_gamma_snapshot())
        for key in STABLE_SNAPSHOT_KEYS:
            self.assertIn(key, snap)

    def test_safe_float_rejects_non_finite(self):
        self.assertIsNone(safe_float(float("nan")))
        self.assertIsNone(safe_float(float("inf")))
        self.assertEqual(safe_float(1.25), 1.25)


class TestEndToEndSmallFrame(unittest.TestCase):
    def test_build_pressure_frame_short_history(self):
        frame = build_pressure_frame(_synthetic_ohlcv(8), lookback=3, volume_window=3, weekly_window=2)
        self.assertGreater(len(frame), 0)
        self.assertIn("macd_regime", frame.columns)
        self.assertIn("T_a", frame.columns)


if __name__ == "__main__":
    unittest.main()
