"""Tests for expiry-aware gamma flip engine."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from data_loader import load_option_chain_data, get_most_recent_option_date
from gamma_flip_engine import compute_gamma_flip, empty_gamma_snapshot
from canopyento_boundary_engine import load_stock_data_fallback


class TestGammaFlipEngine(unittest.TestCase):
    def test_spy_live_chain_resolves_flip_with_expiry_selection(self):
        date = get_most_recent_option_date("SPY", verbose=False)
        option_df = load_option_chain_data("SPY", date=date)
        spot = float(load_stock_data_fallback("SPY")["Close"].iloc[-1])

        gamma = compute_gamma_flip(option_df, spot, chain_date=date, ticker="SPY")
        self.assertIsNotNone(gamma["gamma_flip_strike"])
        self.assertNotEqual(gamma["gamma_regime"], "FLIP_UNDEFINED")
        self.assertGreater(gamma["gamma_chain_strikes"], 0)
        self.assertTrue(gamma["gamma_expiry_used"])
        self.assertTrue(gamma["gamma_flip_reason"])
        json.dumps(gamma)

    def test_mixed_expiry_aggregation_no_longer_returns_silent_null(self):
        """Regression: aggregating all expirations used to yield FLIP_UNDEFINED for SPY."""
        date = get_most_recent_option_date("SPY", verbose=False)
        option_df = load_option_chain_data("SPY", date=date)
        spot = float(load_stock_data_fallback("SPY")["Close"].iloc[-1])
        gamma = compute_gamma_flip(option_df, spot, chain_date=date, ticker="SPY")
        self.assertIsNotNone(gamma["gamma_flip_strike"])
        self.assertIn("gamma_flip_reason", gamma)
        self.assertIn("gamma_chain_integrity", gamma)

    def test_empty_gamma_snapshot_has_diagnostics(self):
        snap = empty_gamma_snapshot()
        self.assertEqual(snap["gamma_flip_reason"], "no_option_chain")
        json.dumps(snap)


if __name__ == "__main__":
    unittest.main()
