"""Ablation integrity and analytic insertion checks before exploratory v2 run."""

import math
import unittest
from unittest.mock import patch

import numpy as np

import insertion_correction
import vendor_geometry
from endpoint_geometry import BoundaryPolynomial
from test_decoder import horizontal_maps


class InsertionCorrectionTests(unittest.TestCase):
    def test_deep_tangent_evaluated_at_curved_insertion(self):
        superficial = BoundaryPolynomial((0,), (-100, 100))
        deep = BoundaryPolynomial((20, .1, .002), (-100, 100))
        corrected = insertion_correction.corrected_segment(20, 10, 40, 2, superficial, deep, 200)
        self.assertTrue(corrected.valid)
        expected_x = min(np.roots([.002, .1 - math.tan(math.radians(40)), 10 + 20 * math.tan(math.radians(40))]))
        expected_pa = 40 - math.degrees(math.atan(.004 * expected_x + .1))
        self.assertAlmostEqual(corrected.anchor_x, expected_x)
        self.assertAlmostEqual(corrected.pa_deg, expected_pa)

    def test_intersections_on_same_side_are_rejected(self):
        superficial = BoundaryPolynomial((-2, 3, -1), (-10, 10))
        deep = BoundaryPolynomial((12, -7, 1), (-10, 10))
        corrected = insertion_correction.corrected_segment(0, 0, 0, .2, superficial, deep, 20)
        self.assertFalse(corrected.valid)
        self.assertEqual(corrected.reason, "no_supported_consistent_root_pair")

    def test_legacy_scalar_estimators_unchanged_on_straight_boundaries(self):
        apo, fasc = horizontal_maps()
        old = vendor_geometry.measure(apo, fasc, .5, .5, mt_mode="centre_minus_half", fl_mode="extend_centre", fl_factor=.9, pa_at=.9)
        corrected = insertion_correction.measure(apo, fasc, .5, .5)
        self.assertTrue(old["ok"] and corrected["ok"])
        for target in ("pa_deg", "fl_mm", "mt_mm"):
            self.assertAlmostEqual(old[target], corrected[target], places=8)
        np.testing.assert_allclose(old["segs"], corrected["segs"], atol=1e-9)

    def test_full_fusion_retains_exact_public_formula(self):
        apo, fasc = horizontal_maps()
        raw = insertion_correction.measure(apo, fasc, .5, .5)
        with patch.object(insertion_correction.vendor_texture, "texture_angle", return_value=(30, 30)):
            fused = insertion_correction.full_prediction(apo, fasc, .5, .5, np.zeros((110, 400)), 2)
        angle = (raw["pa_deg"] + 30) / 2
        length = .75 * raw["fl_mm"] + .25 * raw["mt_mm"] / math.sin(math.radians(np.clip(angle, 5, 45)))
        self.assertAlmostEqual(fused["pa_deg"], np.clip(angle, 5, 45))
        self.assertAlmostEqual(fused["fl_mm"], np.clip(length, 30, 200))
        self.assertAlmostEqual(fused["mt_mm"], np.clip(raw["mt_mm"], 10, 50))

    def test_failure_stays_distinguishable_from_legacy_defaults(self):
        empty = np.zeros((100, 100))
        fused = insertion_correction.full_prediction(empty, empty, .5, .5, empty, 2)
        self.assertFalse(fused["valid"])
        self.assertTrue(fused["diagnostics"]["legacy_defaults_present"])
        self.assertEqual(fused["status"], "aponeuroses_not_identified")


if __name__ == "__main__":
    unittest.main()
