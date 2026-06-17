"""Tests for sacred TRPR packet ontology (read-only)."""

from __future__ import annotations

import unittest

from TRPR.ontology.packet_ontology_loader import (
    classify_regime,
    get_derived_metric,
    get_named_regime,
    get_primitive,
    list_named_regimes,
    load_packet_ontology,
    ontology_reference,
    schema_version,
)


class TestPacketOntology(unittest.TestCase):
    def test_loads_sacred_ontology(self):
        ontology = load_packet_ontology()
        self.assertEqual(ontology["metadata"]["project"], "Temporal Relational Packet Reconstructor")
        self.assertEqual(schema_version(), "0.1.1")

    def test_core_primitives_present(self):
        ontology = load_packet_ontology()
        for name in ("magnitude", "coherence", "observability", "compression"):
            self.assertIn(name, ontology["core_primitives"])

    def test_derived_metric_lrp(self):
        lrp = get_derived_metric("LRP")
        self.assertEqual(lrp["key"], "LRP")
        self.assertEqual(lrp["range"], [0.0, 1.0])

    def test_classify_regime_from_sacred_ranges(self):
        self.assertEqual(classify_regime(0.20), "STABLE")
        self.assertEqual(classify_regime(0.45), "PRESSURE_BUILDING")
        self.assertEqual(classify_regime(0.70), "PRE_RUPTURE")
        self.assertEqual(classify_regime(0.90), "RUPTURE_IMMINENT")

    def test_ontology_reference_path(self):
        self.assertEqual(ontology_reference(), "TRPR/ontology/packet_ontology.yaml")

    def test_derived_metric_f_r(self):
        f_r = get_derived_metric("F_r")
        self.assertEqual(f_r["key"], "F_r")
        self.assertEqual(f_r["range"], [0.0, 1.0])

    def test_named_regimes_present(self):
        names = list_named_regimes()
        self.assertIn("RESTORED_EQUILIBRIUM", names)
        self.assertIn("CAPILLARY_PRE_RUPTURE", names)
        regime = get_named_regime("ENTROPIC_DEGRADATION")
        self.assertEqual(regime["key"], "ENTROPIC_DEGRADATION")


if __name__ == "__main__":
    unittest.main()
