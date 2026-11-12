"""Analytic invariants; no competition labels or tuned benchmark scenarios."""

import math
import unittest

import numpy as np
from numpy.polynomial import Polynomial

from endpoint_geometry import (
    BoundaryPolynomial,
    FascicleLine,
    boundary_from_pixels,
    fit_boundary,
    line_from_pixel_endpoints,
    measure_fascicle,
    vertical_thickness,
)


class EndpointGeometryTests(unittest.TestCase):
    def test_horizontal_boundaries_closed_form(self):
        theta = math.radians(30)
        superficial = BoundaryPolynomial((0,), (-100, 100))
        deep = BoundaryPolynomial((20,), (-100, 100))
        line = FascicleLine((0, 10), (math.cos(theta), math.sin(theta)), (-2, 2))
        result = measure_fascicle(line, superficial, deep)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.fl_mm, 20 / math.sin(theta), places=10)
        self.assertAlmostEqual(result.pa_deg, 30, places=10)
        self.assertAlmostEqual(result.extrapolated_fraction, .9, places=10)
        thickness = vertical_thickness(superficial, deep, (-20, 0, 20))
        self.assertTrue(thickness.valid)
        self.assertAlmostEqual(thickness.mt_mm, 20)

    def test_tilted_parallel_boundaries_use_perpendicular_thickness_for_length(self):
        beta, theta = np.radians([20, 25])
        superficial = BoundaryPolynomial((0, math.tan(beta)), (-100, 100))
        deep = BoundaryPolynomial((20, math.tan(beta)), (-100, 100))
        line = FascicleLine((0, 10), (math.cos(beta + theta), math.sin(beta + theta)), (-1, 1))
        result = measure_fascicle(line, superficial, deep)
        self.assertTrue(result.valid, result.reason)
        expected = 20 * math.cos(beta) / math.sin(theta)
        self.assertAlmostEqual(result.fl_mm, expected, places=10)
        self.assertAlmostEqual(result.pa_deg, 25, places=10)
        self.assertNotAlmostEqual(result.fl_mm, 20 / math.sin(theta), places=2)
        self.assertAlmostEqual(vertical_thickness(superficial, deep, (-20, 0, 20)).mt_mm, 20)

    def test_curved_boundary_pa_at_actual_insertion(self):
        superficial = BoundaryPolynomial((0,), (-100, 100))
        deep = BoundaryPolynomial((20, .1, .002), (-100, 100))
        angle = math.radians(40)
        line = FascicleLine((20, 10), (math.cos(angle), math.sin(angle)), (-1, 1))
        result = measure_fascicle(line, superficial, deep)
        self.assertTrue(result.valid, result.reason)
        # Independently solve .002*x² + (.1-tan40)*x +20-10+20*tan40 =0.
        roots = np.roots([.002, .1 - math.tan(angle), 10 + 20 * math.tan(angle)])
        expected_x = min(roots)
        expected_pa = 40 - math.degrees(math.atan(.004 * expected_x + .1))
        self.assertAlmostEqual(result.deep_endpoint_mm[0], expected_x, places=9)
        self.assertAlmostEqual(result.pa_deg, expected_pa, places=9)
        midpoint_pa = 40 - math.degrees(math.atan(.004 * 20 + .1))
        self.assertGreater(abs(result.pa_deg - midpoint_pa), 4)

    def test_vertical_fascicle_does_not_require_finite_line_slope(self):
        superficial = BoundaryPolynomial((3, .2), (0, 100))
        deep = BoundaryPolynomial((23, .2), (0, 100))
        line = FascicleLine((50, 23), (0, 10), (-3, 3))
        result = measure_fascicle(line, superficial, deep)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.fl_mm, 20)
        self.assertAlmostEqual(result.pa_deg, 90 - math.degrees(math.atan(.2)))

    def test_horizontal_reflection_preserves_measurements(self):
        superficial = BoundaryPolynomial((0,), (-100, 100))
        deep = BoundaryPolynomial((20, .1, .002), (-100, 100))
        line = FascicleLine((20, 10), (math.cos(.7), math.sin(.7)), (-1, 1))
        original = measure_fascicle(line, superficial, deep)
        reflected_deep = BoundaryPolynomial((20, -.1, .002), (-100, 100))
        reflected_line = FascicleLine((-20, 10), (-math.cos(.7), math.sin(.7)), (-1, 1))
        reflected = measure_fascicle(reflected_line, superficial, reflected_deep)
        self.assertTrue(original.valid and reflected.valid)
        self.assertAlmostEqual(original.fl_mm, reflected.fl_mm)
        self.assertAlmostEqual(original.pa_deg, reflected.pa_deg)
        self.assertAlmostEqual(original.anchor_x, -reflected.anchor_x)

    def test_reversing_line_parameter_has_no_effect(self):
        superficial = BoundaryPolynomial((0,), (-100, 100))
        deep = BoundaryPolynomial((20,), (-100, 100))
        original = measure_fascicle(FascicleLine((0, 10), (1, 1), (-1, 3)), superficial, deep)
        reversed_line = measure_fascicle(FascicleLine((0, 10), (-1, -1), (-3, 1)), superficial, deep)
        self.assertTrue(original.valid and reversed_line.valid)
        self.assertAlmostEqual(original.fl_mm, reversed_line.fl_mm)
        self.assertAlmostEqual(original.pa_deg, reversed_line.pa_deg)
        np.testing.assert_allclose(original.deep_endpoint_mm, reversed_line.deep_endpoint_mm)

    def test_anisotropic_pixel_calibration_is_physical(self):
        # 200 pixels horizontally and 100 vertically both represent 20 mm.
        spacing = (.1, .2)
        superficial = boundary_from_pixels((0,), (0, 1000), spacing)
        deep = boundary_from_pixels((100,), (0, 1000), spacing)
        line = line_from_pixel_endpoints((410, 5), (590, 95), spacing)
        result = measure_fascicle(line, superficial, deep)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.fl_mm, math.hypot(20, 20))
        self.assertAlmostEqual(result.pa_deg, 45)
        self.assertAlmostEqual(result.support, .9)

    def test_anisotropic_calibration_transforms_boundary_tangent(self):
        spacing = (.1, .2)
        # Pixel slope .5 corresponds to physical slope1, a 45-degree boundary.
        superficial = boundary_from_pixels((0, .5), (0, 1000), spacing)
        deep = boundary_from_pixels((100, .5), (0, 1000), spacing)
        line = line_from_pixel_endpoints((500, 255), (500, 345), spacing)
        result = measure_fascicle(line, superficial, deep)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.fl_mm, 20)
        self.assertAlmostEqual(result.pa_deg, 45)

    def test_uniform_pixel_resampling_preserves_physical_measurement(self):
        p1, p2 = (410, 5), (590, 95)
        results = []
        for factor in (1, 2, .5):
            spacing = (.1 / factor, .2 / factor)
            superficial = boundary_from_pixels((0,), (0, 1000 * factor), spacing)
            deep = boundary_from_pixels((100 * factor,), (0, 1000 * factor), spacing)
            line = line_from_pixel_endpoints(tuple(factor * x for x in p1), tuple(factor * x for x in p2), spacing)
            results.append(measure_fascicle(line, superficial, deep))
        self.assertTrue(all(r.valid for r in results))
        np.testing.assert_allclose([r.fl_mm for r in results], results[0].fl_mm)
        np.testing.assert_allclose([r.pa_deg for r in results], results[0].pa_deg)

    def test_two_roots_do_not_select_wrong_side_or_exit_and_reentry(self):
        # Line y=x. Deep y=.1*x² -x+6 intersects at x=10±sqrt(40).
        # Center (2,2) is inside the muscle; its first deep exit is the low root.
        superficial = BoundaryPolynomial((0,), (-10, 30))
        deep = BoundaryPolynomial((6, -1, .1), (-10, 30))
        line = FascicleLine((2, 2), (1, 1), (-.25, .25))
        result = measure_fascicle(line, superficial, deep)
        self.assertTrue(result.valid, result.reason)
        expected = 10 - math.sqrt(40)
        self.assertAlmostEqual(result.deep_endpoint_mm[0], expected)
        self.assertAlmostEqual(result.fl_mm, math.sqrt(2) * expected)

    def test_intersections_only_on_same_side_are_rejected(self):
        # The center is inside these ordered curves, but all four intersections
        # with y=0 lie to its right; none encloses the visible segment.
        superficial = BoundaryPolynomial((-2, 3, -1), (-10, 10))
        deep = BoundaryPolynomial((12, -7, 1), (-10, 10))
        result = measure_fascicle(FascicleLine((0, 0), (1, 0), (-.1, .1)), superficial, deep)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "no_supported_consistent_root_pair")

    def test_crossed_boundaries_rejected_between_measurement_positions(self):
        superficial = BoundaryPolynomial((0,), (-5, 5))
        deep = BoundaryPolynomial((-1, 0, 1), (-5, 5))
        result = vertical_thickness(superficial, deep, (-2, 2))
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "boundaries_cross_or_touch")

    def test_no_intersections_and_out_of_support(self):
        superficial = BoundaryPolynomial((0,), (-10, 10))
        deep = BoundaryPolynomial((20,), (-10, 10))
        parallel = measure_fascicle(FascicleLine((0, 10), (1, 0), (-1, 1)), superficial, deep)
        self.assertFalse(parallel.valid)
        self.assertEqual(parallel.reason, "missing_boundary_intersection")
        shallow = measure_fascicle(FascicleLine((0, 10), (1, .01), (-1, 1)), superficial, deep)
        self.assertFalse(shallow.valid)
        self.assertEqual(shallow.reason, "no_supported_consistent_root_pair")

    def test_observed_segment_must_be_enclosed(self):
        superficial = BoundaryPolynomial((0,), (-100, 100))
        deep = BoundaryPolynomial((20,), (-100, 100))
        result = measure_fascicle(FascicleLine((0, 10), (0, 1), (-1, 15)), superficial, deep)
        self.assertFalse(result.valid)

    def test_finite_input_validation(self):
        for coefficients in ((float("nan"),), (float("inf"),)):
            with self.assertRaises(ValueError):
                BoundaryPolynomial(coefficients, (0, 10))
        with self.assertRaises(ValueError):
            FascicleLine((0, 1), (0, 0), (-1, 1))
        with self.assertRaises(ValueError):
            FascicleLine((0, float("nan")), (1, 1), (-1, 1))
        with self.assertRaises(ValueError):
            line_from_pixel_endpoints((0, 0), (1, 1), (-1, 1))

    def test_exact_polynomial_fit(self):
        x = np.linspace(10, 100, 31)
        y = 3 + .1 * x + .002 * x ** 2
        fitted = fit_boundary(np.column_stack((x, y)), degree=2)
        np.testing.assert_allclose(fitted.coefficients, (3, .1, .002), rtol=1e-9, atol=1e-9)
        self.assertEqual(fitted.x_support, (10, 100))

    def test_robust_fit_resists_isolated_outlier(self):
        x = np.linspace(0, 100, 51)
        clean = 3 + .2 * x
        contaminated = clean.copy()
        contaminated[25] += 50
        fitted = fit_boundary(np.column_stack((x, contaminated)), degree=1)
        np.testing.assert_allclose(fitted.value(x), clean, atol=1e-5)

    def test_zero_weight_points_do_not_inflate_support(self):
        fitted = fit_boundary([(0, 1), (1, 3), (2, 5), (999, -999)], [1, 1, 1, 0], degree=1)
        self.assertEqual(fitted.x_support, (0, 2))
        np.testing.assert_allclose(fitted.coefficients, (1, 2), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
