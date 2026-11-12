"""End-to-end analytic mask checks, independent of benchmark scalar labels."""

import math
import unittest

import numpy as np

from decoder import extract_fascicle_tracks, measure


def horizontal_maps(fascicles=3):
    aponeuroses = np.zeros((110, 400), dtype=float)
    aponeuroses[19:22, :] = .9
    aponeuroses[79:82, :] = .9
    fascicle = np.zeros_like(aponeuroses)
    for offset in (20, 140, 260)[:fascicles]:
        for delta in range(41):
            fascicle[30 + delta, offset + delta] = .9
    return aponeuroses, fascicle


class DecoderTests(unittest.TestCase):
    def test_masks_to_three_fascicle_physical_measurements(self):
        papo, pfasc = horizontal_maps()
        result = measure(papo, pfasc, .5, .5)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["diagnostics"]["selected_fascicle_count"], 3)
        self.assertAlmostEqual(result["mt_mm"], 28.5, places=8)
        self.assertAlmostEqual(result["pa_deg"], 45, places=8)
        self.assertAlmostEqual(result["fl_mm"], 28.5 * math.sqrt(2), places=8)
        self.assertFalse(result["fallback_used"])

    def test_weighted_centroid_preserves_asymmetric_observed_interval(self):
        probability = np.zeros((50, 50))
        probability[10:31, 20] = np.linspace(.51, .99, 21)
        tracks, _ = extract_fascicle_tracks(probability, .5, .5)
        self.assertEqual(len(tracks), 1)
        line = tracks[0].line
        self.assertNotAlmostEqual(line.observed_t[0], -line.observed_t[1])
        self.assertGreater(line.center_mm[1], 10)
        reconstructed = [np.asarray(line.center_mm) + t * np.asarray(line.direction) for t in line.observed_t]
        np.testing.assert_allclose(reconstructed, [(10, 5), (10, 15)], atol=1e-10)

    def test_boundary_ablation_agrees_on_exact_straight_masks(self):
        papo, pfasc = horizontal_maps()
        robust = measure(papo, pfasc, .5, .5, robust_boundary=True)
        ordinary = measure(papo, pfasc, .5, .5, robust_boundary=False)
        self.assertTrue(robust["valid"] and ordinary["valid"])
        for target in ("pa_deg", "fl_mm", "mt_mm"):
            self.assertAlmostEqual(robust[target], ordinary[target], places=8)

    def test_fewer_than_three_is_explicit_and_still_measured(self):
        papo, pfasc = horizontal_maps(fascicles=1)
        result = measure(papo, pfasc, .5, .5)
        self.assertTrue(result["valid"])
        self.assertEqual(result["diagnostics"]["selected_fascicle_count"], 1)
        self.assertEqual(result["status"], "measured_with_fewer_than_three_fascicles")

    def test_missing_fascicles_preserve_mt_without_fabricated_pa_or_fl(self):
        papo, pfasc = horizontal_maps(fascicles=0)
        result = measure(papo, pfasc, .5, .5)
        self.assertFalse(result["valid"])
        self.assertAlmostEqual(result["mt_mm"], 28.5)
        self.assertIsNone(result["pa_deg"])
        self.assertIsNone(result["fl_mm"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["status"], "no_valid_fascicles")

    def test_no_boundary_produces_explicit_failure(self):
        empty = np.zeros((100, 100))
        result = measure(empty, empty, .5, .5)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "aponeuroses_not_identified")
        self.assertIsNone(result["mt_mm"])
        self.assertFalse(result["fallback_used"])

    def test_anisotropic_maps_use_physical_pca_angle(self):
        papo, pfasc = horizontal_maps()
        result = measure(papo, pfasc, .25, .5)
        self.assertTrue(result["valid"], result)
        expected_angle = math.degrees(math.atan(2))
        self.assertAlmostEqual(result["pa_deg"], expected_angle, places=8)
        self.assertAlmostEqual(result["fl_mm"], 28.5 / math.sin(math.radians(expected_angle)), places=8)

    def test_nonfinite_and_out_of_range_probabilities_are_rejected(self):
        papo, pfasc = horizontal_maps()
        pfasc[0, 0] = np.nan
        result = measure(papo, pfasc, .5, .5)
        self.assertEqual(result["status"], "nonfinite_probability_map")
        pfasc[0, 0] = 2
        result = measure(papo, pfasc, .5, .5)
        self.assertEqual(result["status"], "probabilities_outside_unit_interval")

    def test_elongation_and_length_filters_are_recorded(self):
        probability = np.zeros((100, 100))
        probability[10:20, 10:20] = .9
        probability[50:56, 50] = .9
        tracks, diagnostics = extract_fascicle_tracks(probability, .5, .5)
        self.assertEqual(tracks, [])
        self.assertEqual(diagnostics["extraction_rejections"], {"component_not_elongated": 1, "visible_segment_too_short": 1})


if __name__ == "__main__":
    unittest.main()
