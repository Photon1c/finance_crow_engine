"""Tests for elastic rebound and hidden reservoir metrics."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from elastic_rebound_engine import compute_elastic_rebound
from pressure_field_dashboard import build_pressure_frame, empty_gamma_snapshot
from pressure_field_derivatives import apply_lrp_loop_closure, enrich_pressure_derivatives
from pressure_field_physics import enrich_pressure_physics
from pressure_field_schema import (
    STABLE_CORE_SNAPSHOT_KEYS,
    STABLE_ELASTIC_SNAPSHOT_KEYS,
    STABLE_EXTENDED_SNAPSHOT_KEYS,
    build_stable_snapshot,
)


def _pipeline_frame(rows: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = 100.0 + np.sin(np.linspace(0, 3, rows)) * 0.5
    base = pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 0.2,
            "High": close + 0.3,
            "Low": close - 0.3,
            "Close": close,
            "Volume": np.full(rows, 1_000_000.0),
        }
    )
    frame = build_pressure_frame(base)
    frame = enrich_pressure_derivatives(frame, gamma={})
    frame = enrich_pressure_physics(frame, gamma={})
    return apply_lrp_loop_closure(frame)


class TestElasticStrain(unittest.TestCase):
    def test_elastic_strain_score_clamped(self):
        out = compute_elastic_rebound(_pipeline_frame())
        series = out["elastic_strain_score"].dropna()
        self.assertTrue((series >= 0).all())
        self.assertTrue((series <= 1).all())


class TestHiddenReservoir(unittest.TestCase):
    def _two_row_frame(self, *, prev_lrp: float, curr_lrp: float, d_c: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "LRP": prev_lrp,
                    "D_c": d_c,
                    "R_o": 0.5,
                    "gamma_flip_distance_pct": 1.0,
                    "cvd_imbalance": 0.2,
                    "vwap_distance_pct": 0.5,
                    "wall_pinning_strength": 0.6,
                    "C_w": 0.3,
                    "A_micro": 0.2,
                },
                {
                    "LRP": curr_lrp,
                    "D_c": d_c,
                    "R_o": 0.5,
                    "gamma_flip_distance_pct": 1.0,
                    "cvd_imbalance": 0.2,
                    "vwap_distance_pct": 0.5,
                    "wall_pinning_strength": 0.6,
                    "C_w": 0.3,
                    "A_micro": 0.2,
                },
            ]
        )

    def test_reservoir_rises_when_lrp_drops_and_dissipation_low(self):
        out = compute_elastic_rebound(self._two_row_frame(prev_lrp=0.6, curr_lrp=0.4, d_c=0.2))
        self.assertGreater(float(out.iloc[1]["hidden_reservoir_pressure"]), float(out.iloc[0]["hidden_reservoir_pressure"]))

    def test_reservoir_decays_when_dissipation_high(self):
        rows = [
            {"LRP": 0.6, "D_c": 0.2, "R_o": 0.5, "gamma_flip_distance_pct": 1.0, "cvd_imbalance": 0.2,
             "vwap_distance_pct": 0.5, "wall_pinning_strength": 0.6, "C_w": 0.3, "A_micro": 0.2},
            {"LRP": 0.4, "D_c": 0.2, "R_o": 0.5, "gamma_flip_distance_pct": 1.0, "cvd_imbalance": 0.2,
             "vwap_distance_pct": 0.5, "wall_pinning_strength": 0.6, "C_w": 0.3, "A_micro": 0.2},
            {"LRP": 0.4, "D_c": 0.85, "R_o": 0.5, "gamma_flip_distance_pct": 1.0, "cvd_imbalance": 0.2,
             "vwap_distance_pct": 0.5, "wall_pinning_strength": 0.6, "C_w": 0.3, "A_micro": 0.2},
        ]
        out = compute_elastic_rebound(pd.DataFrame(rows))
        self.assertGreater(float(out.iloc[1]["hidden_reservoir_pressure"]), 0.0)
        self.assertLess(float(out.iloc[2]["hidden_reservoir_pressure"]), float(out.iloc[1]["hidden_reservoir_pressure"]))


class TestFalseStability(unittest.TestCase):
    def test_false_stability_flag_triggers(self):
        frame = pd.DataFrame(
            [
                {"LRP": 0.55, "D_c": 0.25, "R_o": 0.6, "gamma_flip_distance_pct": 0.8, "cvd_imbalance": 0.3,
                 "vwap_distance_pct": 0.4, "wall_pinning_strength": 0.7, "C_w": 0.2, "A_micro": 0.15},
                {"LRP": 0.48, "D_c": 0.25, "R_o": 0.55, "gamma_flip_distance_pct": 0.7, "cvd_imbalance": 0.35,
                 "vwap_distance_pct": 0.45, "wall_pinning_strength": 0.75, "C_w": 0.25, "A_micro": 0.12},
            ]
        )
        out = compute_elastic_rebound(frame)
        self.assertEqual(int(out.iloc[1]["false_stability_flag"]), 1)


class TestSchemaOrdering(unittest.TestCase):
    def test_core_keys_unchanged_and_precede_extended_and_elastic(self):
        latest = compute_elastic_rebound(_pipeline_frame()).iloc[-1]
        snap = build_stable_snapshot(ticker="SPY", latest=latest, spot=100.0, gamma={})
        keys = list(snap.keys())
        self.assertEqual(keys[: len(STABLE_CORE_SNAPSHOT_KEYS)], list(STABLE_CORE_SNAPSHOT_KEYS))
        ext_start = keys.index(STABLE_EXTENDED_SNAPSHOT_KEYS[0])
        elastic_start = keys.index(STABLE_ELASTIC_SNAPSHOT_KEYS[0])
        self.assertEqual(ext_start, len(STABLE_CORE_SNAPSHOT_KEYS))
        self.assertEqual(elastic_start, len(STABLE_CORE_SNAPSHOT_KEYS) + len(STABLE_EXTENDED_SNAPSHOT_KEYS))


class TestNoOptionChain(unittest.TestCase):
    def test_missing_gamma_still_computes_elastic_metrics(self):
        frame = build_pressure_frame(
            pd.DataFrame(
                {
                    "Date": pd.date_range("2024-01-01", periods=10, freq="B"),
                    "Open": np.full(10, 100.0),
                    "High": np.full(10, 100.5),
                    "Low": np.full(10, 99.5),
                    "Close": np.full(10, 100.0),
                    "Volume": np.full(10, 1_000_000.0),
                }
            )
        )
        gamma = empty_gamma_snapshot()
        frame = enrich_pressure_derivatives(frame, gamma=gamma)
        frame = enrich_pressure_physics(frame, gamma=gamma)
        out = compute_elastic_rebound(apply_lrp_loop_closure(frame), gamma=gamma)
        self.assertIn("elastic_strain_score", out.columns)
        self.assertTrue(out["elastic_strain_score"].notna().any())


if __name__ == "__main__":
    unittest.main()
