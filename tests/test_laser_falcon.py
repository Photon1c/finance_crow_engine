"""Tests for Laser Falcon options research engine."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from implied_vol_solver import black_scholes_price, implied_volatility
from laser_falcon_data_adapter import assess_data_health, normalize_option_chain, normalize_stock_df
from laser_falcon_regime_mapper import map_laser_falcon_regime_metrics
from ou_iv_engine import ou_half_life, simulate_ou_iv_paths
from stochastic_vol_engine import simulate_stochastic_vol_paths
from tests.laser_falcon_fixtures import make_synthetic_option_chain, make_synthetic_stock_df
from volatility_skew_engine import compute_skew_metrics
from volatility_surface_engine import assess_surface_density, build_iv_surface_grid


class TestDataAdapter(unittest.TestCase):
    def test_normalize_option_chain_long_format(self):
        raw = make_synthetic_option_chain(spot=100.0, sparse=False)
        normalized = normalize_option_chain(raw, spot=100.0, reference_date=datetime(2026, 6, 15))
        self.assertFalse(normalized.empty)
        self.assertIn("mid_price", normalized.columns)
        self.assertIn("moneyness", normalized.columns)
        self.assertIn("dte", normalized.columns)
        self.assertEqual(set(normalized["option_type"]), {"call", "put"})

    def test_sparse_chain_health(self):
        raw = make_synthetic_option_chain(spot=50.0, sparse=True)
        normalized = normalize_option_chain(raw, spot=50.0, reference_date=datetime(2026, 6, 15))
        health = assess_data_health(normalized, ticker="SPCX")
        self.assertIn(health["status"], {"SPARSE", "OK", "INSUFFICIENT"})
        self.assertGreaterEqual(health["n_contracts"], 1)


class TestImpliedVolSolver(unittest.TestCase):
    def test_iv_roundtrip(self):
        price = black_scholes_price(spot=100, strike=100, time_years=30 / 365, volatility=0.25, option_type="call")
        solved = implied_volatility(
            price=price,
            spot=100,
            strike=100,
            time_years=30 / 365,
            option_type="call",
        )
        self.assertIsNotNone(solved)
        self.assertAlmostEqual(solved, 0.25, places=2)


class TestSkewEngine(unittest.TestCase):
    def test_skew_metrics_bounded_flags(self):
        raw = make_synthetic_option_chain(spot=100.0)
        normalized = normalize_option_chain(raw, spot=100.0, reference_date=datetime(2026, 6, 15))
        metrics = compute_skew_metrics(normalized, spot=100.0)
        self.assertEqual(metrics["status"], "OK")
        self.assertIsNotNone(metrics["atm_iv"])
        self.assertIsInstance(metrics["put_fear_flag"], bool)


class TestSurfaceEngine(unittest.TestCase):
    def test_sparse_surface_skipped(self):
        raw = make_synthetic_option_chain(spot=100.0, sparse=True)
        normalized = normalize_option_chain(raw, spot=100.0, reference_date=datetime(2026, 6, 15))
        density = assess_surface_density(normalized)
        self.assertFalse(density["sufficient"])

    def test_dense_surface_ok(self):
        raw = make_synthetic_option_chain(spot=100.0, sparse=False)
        normalized = normalize_option_chain(raw, spot=100.0, reference_date=datetime(2026, 6, 15))
        surface = build_iv_surface_grid(normalized, spot=100.0)
        self.assertEqual(surface["status"], "OK")
        self.assertIsNotNone(surface["grid"])


class TestSimulationEngines(unittest.TestCase):
    def test_ou_paths_finite(self):
        result = simulate_ou_iv_paths(iv0=0.3, long_run_mean_iv=0.22, theta=4.0, vol_of_vol=0.1, projection_days=10, n_paths=50, seed=1)
        self.assertTrue(np.all(np.isfinite(result["mean_path"])))
        self.assertGreater(ou_half_life(4.0), 0)

    def test_stochastic_vol_paths_finite(self):
        result = simulate_stochastic_vol_paths(spot0=100, variance0=0.04, projection_days=10, n_paths=50, seed=1)
        self.assertTrue(np.all(np.isfinite(result["price_p50"])))
        self.assertTrue(np.all(result["price_p50"] > 0))


class TestPrimaryEngine(unittest.TestCase):
    @patch("laser_falcon_primary_engine.load_laser_falcon_snapshot")
    def test_run_analysis_writes_artifacts(self, mock_load):
        from laser_falcon_data_adapter import LaserFalconSnapshot
        from laser_falcon_primary_engine import run_laser_falcon_analysis

        raw = make_synthetic_option_chain(spot=100.0, sparse=False)
        normalized = normalize_option_chain(raw, spot=100.0, reference_date=datetime(2026, 6, 15))
        stock = normalize_stock_df(make_synthetic_stock_df(spot=100.0))
        snapshot = LaserFalconSnapshot(
            ticker="TEST",
            spot=100.0,
            chain_date="2026_06_15",
            reference_date=datetime(2026, 6, 15),
            stock_df=stock,
            option_df_raw=raw,
            option_df=normalized,
            data_health=assess_data_health(normalized, ticker="TEST"),
        )
        mock_load.return_value = snapshot

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = run_laser_falcon_analysis("TEST", benchmark="TEST", output_dir=out, projection_days=10, n_paths=30)
            self.assertTrue(Path(result["json_path"]).exists())
            self.assertTrue(Path(result["summary_md"]).exists())
            payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
            self.assertIn("regime_metrics", payload)
            self.assertIn("iv_pressure_score", payload["regime_metrics"])


class TestRegimeMapper(unittest.TestCase):
    def test_regime_scores_clamped(self):
        metrics = map_laser_falcon_regime_metrics(
            skew_metrics={"atm_iv": 30, "put_wing_iv": 45, "call_wing_iv": 28, "skew_slope": -0.5},
            surface_report={"status": "OK", "density": {"n_points": 50}},
            ou_result={"iv0": 0.3, "terminal_mean": 0.22},
            data_health={"quote_unstable_pct": 10, "iv_coverage_pct": 90, "status": "OK"},
        )
        for key, val in metrics.items():
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)


if __name__ == "__main__":
    unittest.main()
