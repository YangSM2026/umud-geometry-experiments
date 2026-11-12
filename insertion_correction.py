"""Exploratory v2: isolate insertion geometry while retaining public calibration.

This ablation was specified after the complete v1 decoder underperformed. It is
therefore exploratory development, not a newly independent confirmation. Public
band/segment extraction, PA depth regression, FL weighting/factor, texture blend,
thickness/sine mixture and output clips remain explicitly attributed to the
vendored baseline. Only segment intersection/PA measurement changes.

The public component extractor loses the actual projected support interval.
For this isolated ablation, its centroid/angle/length are retained and observed
support is approximated as [-L/2, L/2]. This approximation is disclosed; using the
new weighted-PCA extractor would confound the geometry-only comparison.
"""

from collections import Counter
from dataclasses import asdict

import numpy as np

try:
    from . import vendor_geometry as vendor
    from . import vendor_texture
    from .endpoint_geometry import BoundaryPolynomial, FascicleLine, measure_fascicle
except ImportError:
    import vendor_geometry as vendor
    import vendor_texture
    from endpoint_geometry import BoundaryPolynomial, FascicleLine, measure_fascicle


DEFAULT = dict(vendor.DEFAULT, mt_mode="centre_minus_half", fl_mode="extend_centre", fl_factor=.9, pa_at=.9)
RANGES = {"pa_deg": (5, 45), "fl_mm": (30, 200), "mt_mm": (10, 50)}
EXTRAPOLATION_SHARED_SPANS = 1.0


def corrected_segment(cx, cy, angle_deg, visible_length_mm, superficial, deep, allowance_mm, track_id=""):
    """Insertion measurement with unchanged legacy centroid/orientation/length."""
    angle = np.radians(angle_deg)
    line = FascicleLine(
        (cx, cy), (np.cos(angle), np.sin(angle)),
        (-visible_length_mm / 2, visible_length_mm / 2), track_id,
    )
    return measure_fascicle(line, superficial, deep, max_boundary_extrapolation_mm=allowance_mm)


def measure(papo, pfasc, mx, my, **kwargs):
    """Vendor-compatible measurement dictionary with corrected per-segment rows."""
    config = dict(DEFAULT)
    unknown = set(kwargs) - set(config)
    if unknown:
        raise ValueError(f"Unknown legacy parameters: {sorted(unknown)}")
    config.update(kwargs)
    diagnostics = {
        "variant": "insertion_geometry_only_with_legacy_calibration",
        "observed_span_assumption": "symmetric +/-L/2 around public component centroid; actual support unavailable",
        "boundary_extrapolation_shared_spans": EXTRAPOLATION_SHARED_SPANS,
        "legacy_parameters": config,
        "segments": [], "rejection_reasons": {},
    }
    result = dict(vendor.FALLBACK, ok=False, valid=False, segs=np.zeros((0, 4)), correction_diagnostics=diagnostics)
    bands = vendor.aponeuroses(papo, mx, my, thr=config["apo_thr"])
    if bands is None:
        diagnostics["status"] = "aponeuroses_not_identified"
        return result
    superficial_band, deep_band = bands
    result["apo"] = bands
    result["mt_mm"] = vendor.muscle_thickness(superficial_band, deep_band, config["mt_mode"], config["mt_xs"])
    inner = config["fl_mode"] != "extend_centre"
    superficial = BoundaryPolynomial(tuple(superficial_band["fit"]["bot" if inner else "c"][::-1]), (superficial_band["lo"], superficial_band["hi"]))
    deep = BoundaryPolynomial(tuple(deep_band["fit"]["top" if inner else "c"][::-1]), (deep_band["lo"], deep_band["hi"]))
    shared_span = min(superficial.x_support[1], deep.x_support[1]) - max(superficial.x_support[0], deep.x_support[0])
    if shared_span <= 0:
        diagnostics["status"] = "no_shared_boundary_support"
        return result
    allowance = shared_span * EXTRAPOLATION_SHARED_SPANS
    diagnostics["boundary_extrapolation_allowance_mm"] = allowance
    segments = vendor.fascicle_segments(pfasc, mx, my)
    diagnostics["extracted_segment_count"] = len(segments)
    rows, rejections = [], Counter()
    for index, (cx, cy, angle, length) in enumerate(segments):
        ys = np.polyval(superficial_band["fit"]["c"], cx)
        yd = np.polyval(deep_band["fit"]["c"], cx)
        if not ys - 2 < cy < yd + 2:
            rejections["legacy_centroid_outside_band"] += 1
            continue
        geometry = corrected_segment(cx, cy, angle, length, superficial, deep, allowance, f"segment_{index}")
        diagnostics["segments"].append(asdict(geometry))
        if not geometry.valid:
            rejections[geometry.reason] += 1
            continue
        if not 2 < geometry.pa_deg < 60:
            rejections["legacy_pa_acceptance_filter"] += 1
            continue
        relative_depth = (cy - ys) / max(yd - ys, 1e-6)
        rows.append((relative_depth, geometry.pa_deg, length, geometry.fl_mm))
    diagnostics["rejection_reasons"] = dict(rejections)
    diagnostics["retained_segment_count"] = len(rows)
    if not rows:
        diagnostics["status"] = "no_consistent_segments"
        return result
    data = np.asarray(rows)
    result["segs"] = data
    result["ok"] = result["valid"] = True
    diagnostics["status"] = "measured"

    # These are deliberately the unchanged public scalar estimators/settings.
    angle = None
    if config["pa_mode"] == "depth_fit":
        angle = vendor.pa_depth_fit(data, config["pa_at"])
    elif config["pa_mode"] == "deep_half":
        deep_rows = data[data[:, 0] > .5]
        if len(deep_rows) >= 2:
            angle = vendor._wmedian(deep_rows[:, 1], deep_rows[:, 2])
    result["pa_deg"] = angle if angle is not None else vendor._wmedian(data[:, 1], data[:, 2])
    sine_length = result["mt_mm"] / np.sin(np.radians(max(result["pa_deg"], 3.0)))
    usable = np.isfinite(data[:, 3]) & (data[:, 3] > 10) & (data[:, 3] < 300)
    if not usable.any():
        extended_length = sine_length
    elif config["fl_select"] == "top3":
        eligible = data[usable]
        extended_length = float(np.mean(eligible[np.argsort(-eligible[:, 2])[:3], 3]))
    elif config["fl_select"] == "visible":
        extended_length = vendor._wmedian(data[usable, 3], data[usable, 2] ** 2 / data[usable, 3])
    else:
        extended_length = vendor._wmedian(data[usable, 3], data[usable, 2])
    if config["fl_mode"] == "mt_over_sin":
        length = sine_length
    elif config["fl_mode"] == "avg":
        length = .5 * (extended_length + sine_length)
    else:
        length = extended_length
    result["fl_mm"] = float(length * config["fl_factor"])
    return result


def full_prediction(papo, pfasc, mx, my, gray, scale):
    """Same argument order and public texture/final corrections as the control.

    Callers must use ``valid`` to distinguish measurement from inherited legacy
    defaults and apply/report any whole-triplet baseline fallback themselves.
    """
    measured = measure(papo, pfasc, mx, my)
    texture = None
    if "apo" in measured:
        texture = vendor_texture.texture_angle(gray, scale, scale, measured["apo"])[1]
        if np.isfinite(texture):
            measured["pa_deg"] = (measured["pa_deg"] + texture) / 2
    measured["fl_mm"] = .75 * measured["fl_mm"] + .25 * measured["mt_mm"] / np.sin(np.radians(np.clip(measured["pa_deg"], 5, 45)))
    return {
        **{target: float(np.clip(measured[target], *limits)) for target, limits in RANGES.items()},
        "valid": bool(measured["ok"]),
        "status": measured["correction_diagnostics"]["status"],
        "diagnostics": {
            **measured["correction_diagnostics"],
            "texture_pa_deg": None if texture is None or not np.isfinite(texture) else float(texture),
            "corrected_segment_rows": measured["segs"].tolist(),
            "legacy_defaults_present": not measured["ok"],
            "texture_weight": .5,
            "final_sine_mixture_weight": .25,
            "output_ranges": RANGES,
        },
    }
