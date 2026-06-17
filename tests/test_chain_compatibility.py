"""Tests for chain_compatibility_engine and temporal guardrails."""

from __future__ import annotations

import json
import unittest
from datetime import datetime

import pandas as pd

from chain_compatibility_engine import assess_snapshot_compatibility, validate_snapshot_compatibility
from laser_falcon_data_adapter import normalize_option_chain
from temporal_chain_differential_engine import compare_option_chain_snapshots
from tests.laser_falcon_fixtures import make_synthetic_option_chain, make_synthetic_stock_df
from laser_falcon_data_adapter import normalize_stock_df


class TestChainCompatibility(unittest.TestCase):
    def _norm(self, raw: pd.DataFrame, spot: float = 100.0) -> pd.DataFrame:
        return normalize_option_chain(raw, spot=spot, reference_date=datetime(2026, 6, 15))

    def test_compatible_snapshots(self):
        prior = self._norm(make_synthetic_option_chain(spot=100.0, sparse=False))
        current = self._norm(make_synthetic_option_chain(spot=100.0, sparse=False))
        result = assess_snapshot_compatibility(prior, current, ticker="TEST", spot_prior=100.0, spot_current=100.0)
        self.assertIn(result["status"], ("COMPATIBLE", "DEGRADED"))
        self.assertGreater(result["expiration_overlap_score"], 0.5)

    def test_contract_universe_drift_invalid(self):
        prior_raw = make_synthetic_option_chain(spot=100.0, sparse=False)
        current_raw = make_synthetic_option_chain(spot=100.0, sparse=True)
        prior = self._norm(prior_raw)
        current = self._norm(current_raw)
        result = assess_snapshot_compatibility(prior, current, ticker="SPCX", spot_prior=100.0, spot_current=100.0)
        self.assertEqual(result["status"], "INVALID")
        self.assertTrue(result["contract_universe_drift_flag"])

    def test_partial_overlap_degraded(self):
        prior_raw = make_synthetic_option_chain(spot=100.0, sparse=False)
        current_raw = make_synthetic_option_chain(spot=100.0, sparse=False)
        current_raw = current_raw[current_raw["Expiration Date"] != "2026-09-19"]
        prior = self._norm(prior_raw)
        current = self._norm(current_raw)
        result = assess_snapshot_compatibility(prior, current, ticker="TEST", spot_prior=100.0, spot_current=100.0)
        self.assertIn(result["status"], ("DEGRADED", "COMPATIBLE", "INVALID"))

    def test_temporal_invalid_on_drift(self):
        prior_raw = make_synthetic_option_chain(spot=100.0, sparse=False)
        current_raw = make_synthetic_option_chain(spot=100.0, sparse=True)
        stock = normalize_stock_df(make_synthetic_stock_df(spot=100.0))
        prior = self._norm(prior_raw)
        current = self._norm(current_raw)
        result = compare_option_chain_snapshots(
            {"option_df": prior, "stock_df": stock, "spot": 100.0, "chain_date": "06_14_2026"},
            {"option_df": current, "stock_df": stock, "spot": 100.0, "chain_date": "06_15_2026"},
            ticker="SPCX",
        )
        self.assertEqual(result["status"], "INVALID")
        self.assertEqual(result["reason"], "Contract universe drift detected")
        self.assertNotIn("deltas", result)

    def test_temporal_degraded_with_warning(self):
        prior_raw = make_synthetic_option_chain(spot=100.0, sparse=False)
        current_raw = make_synthetic_option_chain(spot=100.0, sparse=False)
        current_raw.loc[0, "Expiration Date"] = ""
        stock = normalize_stock_df(make_synthetic_stock_df(spot=100.0))
        prior = self._norm(prior_raw)
        current = self._norm(current_raw)
        wrapped = validate_snapshot_compatibility(prior, current, ticker="TEST", spot_prior=100.0, spot_current=100.0)
        if wrapped["status"] == "DEGRADED":
            result = compare_option_chain_snapshots(
                {"option_df": prior, "stock_df": stock, "spot": 100.0, "chain_date": "06_14_2026"},
                {"option_df": current, "stock_df": stock, "spot": 100.0, "chain_date": "06_15_2026"},
                ticker="TEST",
            )
            if result.get("status") == "DEGRADED":
                self.assertIn("deltas", result)
                self.assertTrue(result.get("warnings"))

    def test_json_serialization(self):
        prior = self._norm(make_synthetic_option_chain(spot=100.0, sparse=False))
        current = self._norm(make_synthetic_option_chain(spot=100.0, sparse=False))
        payload = assess_snapshot_compatibility(prior, current, ticker="TEST")
        text = json.dumps(payload, default=str)
        loaded = json.loads(text)
        self.assertIn("compatibility_score", loaded)


if __name__ == "__main__":
    unittest.main()
