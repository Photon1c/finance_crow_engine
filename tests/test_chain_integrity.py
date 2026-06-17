"""Tests for chain_integrity_engine."""

from __future__ import annotations

import json
import math
import unittest
from datetime import datetime

import pandas as pd

from chain_integrity_engine import (
    assess_chain_integrity,
    clean_chain_expirations,
    data_quality_confidence,
    json_safe_integrity,
    validate_chain_for_analysis,
)
from laser_falcon_data_adapter import normalize_option_chain
from tests.laser_falcon_fixtures import make_synthetic_option_chain


class TestChainIntegrity(unittest.TestCase):
    def _normalized(self, raw: pd.DataFrame, spot: float = 100.0) -> pd.DataFrame:
        return normalize_option_chain(raw, spot=spot, reference_date=datetime(2026, 6, 15))

    def test_healthy_chain(self):
        raw = make_synthetic_option_chain(spot=100.0, sparse=False)
        norm = self._normalized(raw)
        result = assess_chain_integrity(norm, ticker="TEST")
        self.assertIn(result["status"], ("HEALTHY", "DEGRADED"))
        self.assertGreaterEqual(result["chain_health_score"], 0.6)
        self.assertGreaterEqual(result["expiration_count"], 2)

    def test_blank_expiration_rows(self):
        raw = make_synthetic_option_chain(spot=100.0, sparse=False)
        norm = self._normalized(raw)
        blank_row = norm.iloc[[0]].copy()
        blank_row["Expiration Date"] = ""
        norm = pd.concat([norm, blank_row], ignore_index=True)
        cleaned, diag = clean_chain_expirations(norm)
        integrity = assess_chain_integrity(norm, ticker="TEST")
        self.assertGreater(diag["blank_expiration_rows"], 0)
        self.assertIn(integrity["status"], ("DEGRADED", "INVALID", "HEALTHY"))

    def test_sparse_spcx_like_chain(self):
        raw = make_synthetic_option_chain(spot=50.0, sparse=True)
        norm = self._normalized(raw, spot=50.0)
        skew_ok = validate_chain_for_analysis(norm, "skew", ticker="SPCX", spot_price=50.0)
        surface_ok = validate_chain_for_analysis(norm, "surface", ticker="SPCX", spot_price=50.0)
        self.assertTrue(skew_ok["valid"])
        self.assertFalse(surface_ok["valid"])

    def test_wide_spread_reduces_score(self):
        raw = make_synthetic_option_chain(spot=100.0, sparse=False)
        norm = self._normalized(raw)
        norm["quote_unstable_flag"] = True
        degraded = assess_chain_integrity(norm, ticker="TEST")
        baseline = assess_chain_integrity(self._normalized(make_synthetic_option_chain()), ticker="TEST")
        self.assertLessEqual(degraded["chain_health_score"], baseline["chain_health_score"])

    def test_duplicate_contracts_reduce_score(self):
        raw = make_synthetic_option_chain(spot=100.0, sparse=False)
        norm = self._normalized(raw)
        dup = pd.concat([norm, norm.iloc[[0]]], ignore_index=True)
        dup_integrity = assess_chain_integrity(dup, ticker="TEST")
        self.assertGreater(dup_integrity["duplicate_contract_count"], 0)

    def test_json_serialization(self):
        raw = make_synthetic_option_chain(spot=100.0)
        norm = self._normalized(raw)
        payload = json_safe_integrity(assess_chain_integrity(norm, ticker="TEST"))
        text = json.dumps(payload)
        loaded = json.loads(text)
        self.assertIn("chain_health_score", loaded)
        self.assertTrue(math.isfinite(loaded["chain_health_score"]))

    def test_confidence_modifier(self):
        raw = make_synthetic_option_chain(spot=100.0, sparse=True)
        norm = self._normalized(raw, spot=100.0)
        integrity = assess_chain_integrity(norm, ticker="SPCX")
        confidence = data_quality_confidence(integrity)
        self.assertLessEqual(confidence, 1.0)
        self.assertGreaterEqual(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
