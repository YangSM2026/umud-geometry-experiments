"""Probability-map adapter for independently implemented endpoint geometry.

Attribution boundary: ``vendor_geometry.aponeuroses`` performs the public
component detection/grouping and supplies column profiles. Its source and
license are preserved in vendor_geometry.py, NOTICE_UPSTREAM, LICENSE_UPSTREAM.
This adapter's weighted PCA, robust inner-border fits and endpoint measurement
are separate implementations of established methods, not a novelty claim.

The constants below were fixed before evaluation on benchmark scalar labels.
They are design assumptions, not validated optimal choices. Missing geometry
stays missing: this module never substitutes an average or a submission row.
"""

from collections import Counter
from dataclasses import asdict, dataclass
import warnings

import cv2
import numpy as np

try:  # Both package imports and Kaggle's flat attached-directory imports work.
    from . import vendor_geometry
    from .endpoint_geometry import BoundaryPolynomial, FascicleLine, fit_boundary, measure_fascicle, vertical_thickness
    from .reliability import select_fascicles
except ImportError:
    import vendor_geometry
    from endpoint_geometry import BoundaryPolynomial, FascicleLine, fit_boundary, measure_fascicle, vertical_thickness
    from reliability import select_fascicles


FROZEN_PARAMETERS = {
    "aponeurosis_probability_threshold": 0.5,
    "aponeurosis_merge_mm": 4.0,
    "aponeurosis_min_gap_mm": 6.0,
    "aponeurosis_min_relative_extent": 0.35,
    "quadratic_boundary_width_threshold_mm": 15.0,
    "huber_delta": 1.345,
    "huber_max_iterations": 30,
    "fascicle_probability_threshold": 0.5,
    "fascicle_min_pixels": 6,
    "fascicle_min_visible_length_mm": 4.0,
    "fascicle_min_elongation": 3.0,
    "pca_singular_value_epsilon": 1e-6,
    "mt_relative_positions": (0.2, 0.5, 0.8),
    "maximum_boundary_extrapolation_shared_spans": 1.0,
    "minimum_insertion_separation_shared_span_fraction": 0.05,
    "maximum_fascicles": 3,
}


@dataclass(frozen=True)
class TrackEvidence:
    line: FascicleLine
    pixel_count: int
    mean_probability: float
    orthogonal_rmse_mm: float
    elongation: float


def extract_fascicle_tracks(pfasc, mx, my):
    """Probability-weighted PCA retaining actual projected support and centroid.

    Components are inherited candidates, not guaranteed biological fascicles.
    Spatially distinct insertion selection later limits repeated fragments;
    curved or merged components remain a documented segmentation limitation.
    """
    threshold = FROZEN_PARAMETERS["fascicle_probability_threshold"]
    count, labels, stats, _ = cv2.connectedComponentsWithStats((pfasc > threshold).astype(np.uint8), connectivity=8)
    tracks, reasons = [], Counter()
    for component in range(1, count):
        pixel_count = int(stats[component, cv2.CC_STAT_AREA])
        if pixel_count < FROZEN_PARAMETERS["fascicle_min_pixels"]:
            reasons["too_few_pixels"] += 1
            continue
        ys, xs = np.nonzero(labels == component)
        points = np.column_stack((xs * mx, ys * my))
        weights = pfasc[ys, xs].astype(float)
        center = np.average(points, axis=0, weights=weights)
        centered = points - center
        _, singular_values, axes = np.linalg.svd(centered * np.sqrt(weights[:, None]), full_matrices=False)
        direction = axes[0].copy()
        if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
            direction *= -1
        projected = centered @ direction
        observed_interval = (float(projected.min()), float(projected.max()))
        visible_length = observed_interval[1] - observed_interval[0]
        if visible_length < FROZEN_PARAMETERS["fascicle_min_visible_length_mm"]:
            reasons["visible_segment_too_short"] += 1
            continue
        epsilon = FROZEN_PARAMETERS["pca_singular_value_epsilon"]
        elongation = float(singular_values[0] / (singular_values[1] + epsilon))
        if elongation < FROZEN_PARAMETERS["fascicle_min_elongation"]:
            reasons["component_not_elongated"] += 1
            continue
        perpendicular = centered @ np.array([-direction[1], direction[0]])
        tracks.append(TrackEvidence(
            FascicleLine(tuple(center), tuple(direction), observed_interval, f"component_{component}"),
            pixel_count, float(weights.mean()),
            float(np.sqrt(np.average(perpendicular ** 2, weights=weights))), elongation,
        ))
    return tracks, {"connected_components": count - 1, "extraction_rejections": dict(reasons)}


def _profile_weights(band, probability, mx, my):
    """Mean above-threshold confidence for each supplied band-column profile."""
    height, width = probability.shape
    weights = []
    for x, top, bottom in zip(band["X"], band["top"], band["bot"]):
        col = int(round(float(x) / mx))
        lower = max(0, int(round(float(top) / my + .5)))
        upper = min(height - 1, int(round(float(bottom) / my - .5)))
        if not 0 <= col < width or lower > upper:
            raise ValueError("Public band profile is outside the probability map")
        values = probability[lower:upper + 1, col]
        values = values[values > FROZEN_PARAMETERS["aponeurosis_probability_threshold"]]
        if not len(values):
            raise ValueError("Public band profile has no foreground confidence")
        weights.append(float(values.mean()))
    return np.asarray(weights)


def _inner_boundary(band, edge, probability, mx, my, robust):
    if not robust:
        # NumPy polyfit/vendor representation is DESCENDING; our module is ASCENDING.
        return BoundaryPolynomial(tuple(band["fit"][edge][::-1]), (float(band["lo"]), float(band["hi"])))
    degree = 2 if band["hi"] - band["lo"] > FROZEN_PARAMETERS["quadratic_boundary_width_threshold_mm"] else 1
    return fit_boundary(
        np.column_stack((band["X"], band[edge])),
        _profile_weights(band, probability, mx, my),
        degree=degree,
        huber_delta=FROZEN_PARAMETERS["huber_delta"],
        max_iterations=FROZEN_PARAMETERS["huber_max_iterations"],
    )


def measure(papo, pfasc, mx, my, robust_boundary=True):
    """Decode one calibrated pair of probability maps with explicit validity.

    mx/my are millimetres per map pixel, supplied from verified calibration.
    PA/FL average up to three selected fascicles; MT averages three prespecified
    vertical measurements. Fewer than three fascicles is explicitly reported.
    A valid MT may be returned alongside missing PA/FL with ``valid=False``.
    """
    result = {
        "valid": False, "ok": False, "status": "not_measured",
        "pa_deg": None, "fl_mm": None, "mt_mm": None,
        "fallback_used": False, "fallback_status": "not_applied",
        "reasons": [], "diagnostics": {"robust_boundary": bool(robust_boundary)},
    }

    def fail(reason):
        result["status"] = reason
        result["reasons"].append(reason)
        return result

    try:
        papo, pfasc = np.asarray(papo, dtype=float), np.asarray(pfasc, dtype=float)
        mx, my = float(mx), float(my)
    except (TypeError, ValueError):
        return fail("invalid_input_types")
    if papo.ndim != 2 or pfasc.shape != papo.shape or min(papo.shape, default=0) < 2:
        return fail("invalid_probability_map_shape")
    if not np.isfinite([mx, my]).all() or mx <= 0 or my <= 0:
        return fail("invalid_physical_calibration")
    if not np.isfinite(papo).all() or not np.isfinite(pfasc).all():
        return fail("nonfinite_probability_map")
    if np.any((papo < 0) | (papo > 1)) or np.any((pfasc < 0) | (pfasc > 1)):
        return fail("probabilities_outside_unit_interval")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", np.exceptions.RankWarning)
            bands = vendor_geometry.aponeuroses(
                papo, mx, my,
                thr=FROZEN_PARAMETERS["aponeurosis_probability_threshold"],
                merge_mm=FROZEN_PARAMETERS["aponeurosis_merge_mm"],
                min_gap_mm=FROZEN_PARAMETERS["aponeurosis_min_gap_mm"],
                min_rel_extent=FROZEN_PARAMETERS["aponeurosis_min_relative_extent"],
            )
            if bands is None:
                return fail("aponeuroses_not_identified")
            superficial_band, deep_band = bands
            superficial = _inner_boundary(superficial_band, "bot", papo, mx, my, robust_boundary)
            deep = _inner_boundary(deep_band, "top", papo, mx, my, robust_boundary)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, np.exceptions.RankWarning) as exc:
        result["diagnostics"]["boundary_failure_type"] = type(exc).__name__
        return fail("boundary_detection_or_fitting_failed")

    lo = max(superficial.x_support[0], deep.x_support[0])
    hi = min(superficial.x_support[1], deep.x_support[1])
    span = hi - lo
    result["diagnostics"].update({
        "superficial_inner_boundary": asdict(superficial),
        "deep_inner_boundary": asdict(deep),
        "shared_x_support_mm": (lo, hi),
    })
    if span <= 0:
        return fail("no_shared_boundary_support")

    thickness_positions = lo + span * np.asarray(FROZEN_PARAMETERS["mt_relative_positions"])
    thickness = vertical_thickness(superficial, deep, thickness_positions)
    result["diagnostics"]["thickness"] = asdict(thickness)
    if not thickness.valid:
        return fail("invalid_vertical_thickness")
    result["mt_mm"] = thickness.mt_mm

    try:
        tracks, extraction_diagnostics = extract_fascicle_tracks(pfasc, mx, my)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        result["diagnostics"]["fascicle_failure_type"] = type(exc).__name__
        return fail("fascicle_component_extraction_failed")
    result["diagnostics"].update(extraction_diagnostics)
    allowance = span * FROZEN_PARAMETERS["maximum_boundary_extrapolation_shared_spans"]
    min_separation = span * FROZEN_PARAMETERS["minimum_insertion_separation_shared_span_fraction"]
    candidates = [measure_fascicle(track.line, superficial, deep, max_boundary_extrapolation_mm=allowance) for track in tracks]
    selection = select_fascicles(candidates, min_separation=min_separation, max_count=FROZEN_PARAMETERS["maximum_fascicles"])
    result["diagnostics"].update({
        "boundary_extrapolation_allowance_mm": allowance,
        "minimum_insertion_separation_mm": min_separation,
        "extracted_track_count": len(tracks),
        "valid_candidate_count": sum(candidate.valid for candidate in candidates),
        "selected_fascicle_count": len(selection.selected),
        "selection_status": selection.status,
        "selected_track_ids": [candidate.track_id for candidate in selection.selected],
        "candidates": [dict(asdict(candidate), evidence=asdict(track)) for candidate, track in zip(candidates, tracks)],
        "candidate_rejection_reasons": dict(Counter(candidate.reason for candidate in candidates if not candidate.valid)),
    })
    if not selection.selected:
        return fail("no_valid_fascicles")
    result["pa_deg"] = float(np.mean([candidate.pa_deg for candidate in selection.selected]))
    result["fl_mm"] = float(np.mean([candidate.fl_mm for candidate in selection.selected]))
    result["valid"] = result["ok"] = True
    result["status"] = "measured" if len(selection.selected) == FROZEN_PARAMETERS["maximum_fascicles"] else "measured_with_fewer_than_three_fascicles"
    return result
