"""Independent, physically calibrated geometry for visible muscle fascicles.

This module implements standard analytic geometry from a mathematical
specification. It does not contain copied notebook code, trained weights,
segmentation preprocessing, or empirically tuned submission corrections.
These equations are established methods, not a claim of methodological novelty.

Conventions
-----------
* x and y are physical millimetres; image y increases toward the deep boundary.
* Boundary coefficients are in *ascending* order, as in numpy.polynomial.
* A line is r(t) = center_mm + t * unit(direction), with t in millimetres.
* observed_t describes the actually visible portion, not extrapolated endpoints.
* PA is the acute angle between the fascicle and the deep tangent at insertion.
* FL is the distance between the two consistent boundary intersections.
* MT is vertical separation of the inner boundaries at prespecified positions.

Callers must supply inner boundary curves and verify pixel calibration. Stability
and visibility diagnostics returned here are not calibrated confidence intervals.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial import Polynomial


Point = Tuple[float, float]


@dataclass(frozen=True)
class BoundaryPolynomial:
    coefficients: Tuple[float, ...]
    x_support: Point

    def __post_init__(self):
        coefficients = tuple(float(v) for v in self.coefficients)
        support = tuple(float(v) for v in self.x_support)
        if not coefficients or not np.isfinite(coefficients).all():
            raise ValueError("Boundary coefficients must be finite and nonempty")
        if len(support) != 2 or not np.isfinite(support).all() or support[0] >= support[1]:
            raise ValueError("x_support must be a finite increasing interval")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "x_support", support)

    def value(self, x):
        return Polynomial(self.coefficients)(x)

    def slope(self, x):
        return Polynomial(self.coefficients).deriv()(x)


@dataclass(frozen=True)
class FascicleLine:
    center_mm: Point
    direction: Point
    observed_t: Point
    track_id: str = ""

    def __post_init__(self):
        center = tuple(float(v) for v in self.center_mm)
        direction = np.asarray(self.direction, dtype=float)
        interval = tuple(float(v) for v in self.observed_t)
        if len(center) != 2 or not np.isfinite(center).all():
            raise ValueError("center_mm must contain two finite coordinates")
        if direction.shape != (2,) or not np.isfinite(direction).all():
            raise ValueError("direction must contain two finite components")
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("direction must have nonzero finite length")
        if (len(interval) != 2 or not np.isfinite(interval).all()
                or interval[0] >= interval[1] or interval[0] > 0 or interval[1] < 0):
            raise ValueError("observed_t must be increasing, finite, and contain zero")
        object.__setattr__(self, "center_mm", center)
        object.__setattr__(self, "direction", tuple(direction / norm))
        object.__setattr__(self, "observed_t", interval)
        object.__setattr__(self, "track_id", str(self.track_id))


@dataclass(frozen=True)
class FascicleMeasurement:
    valid: bool
    reason: str
    track_id: str
    fl_mm: Optional[float] = None
    pa_deg: Optional[float] = None
    anchor_x: Optional[float] = None
    extrapolated_fraction: Optional[float] = None
    support: Optional[float] = None
    superficial_endpoint_mm: Optional[Point] = None
    deep_endpoint_mm: Optional[Point] = None
    visible_length_mm: Optional[float] = None


@dataclass(frozen=True)
class ThicknessMeasurement:
    valid: bool
    reason: str
    mt_mm: Optional[float]
    x_positions_mm: Tuple[float, ...]
    values_mm: Tuple[float, ...]


def _real_roots(polynomial: Polynomial) -> np.ndarray:
    """Finite real roots; coincident curves supply no identifiable endpoint."""
    coefficients = np.asarray(polynomial.coef, dtype=float)
    if not np.isfinite(coefficients).all():
        return np.empty(0)
    # Exact trailing zeros are safe to remove; relative trimming could erase a
    # small but meaningful highest-order term when coordinate units differ.
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients = coefficients[:-1]
    if len(coefficients) <= 1:
        return np.empty(0)
    try:
        roots = Polynomial(coefficients / np.max(np.abs(coefficients))).roots()
    except np.linalg.LinAlgError:
        return np.empty(0)
    roots = np.asarray(roots, dtype=complex)
    real = roots.real[np.isfinite(roots) & (np.abs(roots.imag) <= 1e-8 * (1 + np.abs(roots.real)))]
    return np.sort(real)


def _minimum(polynomial: Polynomial, lo: float, hi: float) -> float:
    stationary = _real_roots(polynomial.deriv())
    stationary = stationary[(stationary > lo) & (stationary < hi)]
    values = polynomial(np.r_[lo, hi, stationary])
    return float(np.min(values)) if np.isfinite(values).all() else float("-inf")


def _on_line(boundary: BoundaryPolynomial, line: FascicleLine, scale: float) -> Polynomial:
    """Boundary height minus line height, in the well-scaled variable u=t/scale."""
    cx, cy = line.center_mm
    vx, vy = line.direction
    return Polynomial(boundary.coefficients)(Polynomial([cx, vx * scale])) - Polynomial([cy, vy * scale])


def _within_support(xlo: float, xhi: float, boundary: BoundaryPolynomial, allowance: float, tol: float) -> bool:
    return xlo >= boundary.x_support[0] - allowance - tol and xhi <= boundary.x_support[1] + allowance + tol


def measure_fascicle(
    line: FascicleLine,
    superficial: BoundaryPolynomial,
    deep: BoundaryPolynomial,
    *,
    max_boundary_extrapolation_mm: float = 0.0,
    tolerance_mm: float = 1e-7,
) -> FascicleMeasurement:
    """Measure a line only when endpoints consistently enclose its visible span.

    Enumerating root pairs is necessary for curved boundaries. A candidate must
    contain the observed segment, stay inside ordered boundaries throughout its
    length, and remain in boundary support (unless explicit extrapolation is
    allowed). The shortest admissible enclosure is chosen. No PA or FL clipping,
    fitted multiplier, or thickness/sine mixture is applied.
    """
    if not np.isfinite(max_boundary_extrapolation_mm) or max_boundary_extrapolation_mm < 0:
        raise ValueError("Boundary extrapolation allowance must be finite and nonnegative")
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("tolerance_mm must be finite and positive")

    def invalid(reason):
        return FascicleMeasurement(False, reason, line.track_id)

    cx, cy = line.center_mm
    vx, vy = line.direction
    sup_at_center, deep_at_center = superficial.value(cx), deep.value(cx)
    if not np.isfinite([sup_at_center, deep_at_center]).all():
        return invalid("nonfinite_boundary_evaluation")
    if not sup_at_center + tolerance_mm < cy < deep_at_center - tolerance_mm:
        return invalid("center_not_strictly_inside_muscle")

    scale = max(1.0, line.observed_t[1] - line.observed_t[0],
                superficial.x_support[1] - superficial.x_support[0],
                deep.x_support[1] - deep.x_support[0])
    superficial_u = _on_line(superficial, line, scale)
    deep_u = _on_line(deep, line, scale)
    sup_roots = _real_roots(superficial_u)
    deep_roots = _real_roots(deep_u)
    if len(sup_roots) == 0 or len(deep_roots) == 0:
        return invalid("missing_boundary_intersection")

    candidates = []
    ordered_difference = Polynomial(deep.coefficients) - Polynomial(superficial.coefficients)
    for us in sup_roots:
        for ud in deep_roots:
            ulo, uhi = sorted((float(us), float(ud)))
            tlo, thi = ulo * scale, uhi * scale
            if tlo > line.observed_t[0] + tolerance_mm or thi < line.observed_t[1] - tolerance_mm:
                continue
            if not tlo < 0 < thi or thi - tlo <= tolerance_mm:
                continue
            xs, xd = cx + us * scale * vx, cx + ud * scale * vx
            xlo, xhi = sorted((xs, xd))
            if not all(_within_support(xlo, xhi, boundary, max_boundary_extrapolation_mm, tolerance_mm)
                       for boundary in (superficial, deep)):
                continue
            # Analytic extrema checks cover the whole interval, not a sampled grid.
            if _minimum(ordered_difference, xlo, xhi) <= tolerance_mm:
                continue
            if _minimum(-superficial_u, ulo, uhi) < -tolerance_mm:
                continue
            if _minimum(deep_u, ulo, uhi) < -tolerance_mm:
                continue
            endpoint_errors = [abs(superficial_u(us)), abs(deep_u(ud))]
            if not np.isfinite(endpoint_errors).all() or max(endpoint_errors) > tolerance_mm * 10:
                continue
            candidates.append((thi - tlo, float(us), float(ud)))

    if not candidates:
        return invalid("no_supported_consistent_root_pair")
    length, us, ud = min(candidates)
    p_sup = (float(cx + us * scale * vx), float(cy + us * scale * vy))
    p_deep = (float(cx + ud * scale * vx), float(cy + ud * scale * vy))
    slope = float(deep.slope(p_deep[0]))
    # atan2(cross, dot) avoids inverse-cosine instability for shallow angles.
    tangent = np.array([1.0, slope])
    tangent /= np.linalg.norm(tangent)
    cross = abs(vx * tangent[1] - vy * tangent[0])
    dot = abs(vx * tangent[0] + vy * tangent[1])
    angle = float(np.degrees(np.arctan2(cross, dot)))
    visible_length = line.observed_t[1] - line.observed_t[0]
    support = float(np.clip(visible_length / length, 0.0, 1.0))
    if not np.isfinite([length, angle, *p_sup, *p_deep, support]).all():
        return invalid("nonfinite_measurement")
    return FascicleMeasurement(
        True, "ok", line.track_id, float(length), angle, p_deep[0], 1.0 - support,
        support, p_sup, p_deep, float(visible_length),
    )


def vertical_thickness(
    superficial: BoundaryPolynomial,
    deep: BoundaryPolynomial,
    x_positions_mm: Sequence[float],
    *,
    tolerance_mm: float = 1e-7,
) -> ThicknessMeasurement:
    """Mean vertical inner-border separation at explicitly selected locations."""
    positions = tuple(float(x) for x in x_positions_mm)
    if not positions or not np.isfinite(positions).all():
        return ThicknessMeasurement(False, "invalid_measurement_positions", None, positions, ())
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("tolerance_mm must be finite and positive")
    lo, hi = min(positions), max(positions)
    if not all(_within_support(lo, hi, boundary, 0.0, tolerance_mm) for boundary in (superficial, deep)):
        return ThicknessMeasurement(False, "positions_outside_boundary_support", None, positions, ())
    difference = Polynomial(deep.coefficients) - Polynomial(superficial.coefficients)
    if _minimum(difference, lo, hi) <= tolerance_mm:
        return ThicknessMeasurement(False, "boundaries_cross_or_touch", None, positions, ())
    values = tuple(float(v) for v in difference(np.asarray(positions)))
    if not np.isfinite(values).all():
        return ThicknessMeasurement(False, "nonfinite_thickness", None, positions, ())
    return ThicknessMeasurement(True, "ok", float(np.mean(values)), positions, values)


def fit_boundary(
    points_mm: Sequence[Point],
    weights: Optional[Sequence[float]] = None,
    *,
    degree: int = 2,
    huber_delta: float = 1.345,
    max_iterations: int = 30,
) -> BoundaryPolynomial:
    """Weighted Huber IRLS polynomial fit in centred/scaled physical coordinates.

    The conventional Huber constant is a robust-statistics default, not a value
    selected on competition predictions. This fits one boundary; paired fits
    must still pass noncrossing checks during measurement. Zero-weight points
    are excluded from both fitting and the reported observation support.
    """
    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points_mm must be a finite N-by-2 array")
    if not isinstance(degree, int) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    if not np.isfinite(huber_delta) or huber_delta <= 0 or max_iterations < 1:
        raise ValueError("Huber constant and iteration count must be positive")
    base_weights = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=float)
    if base_weights.shape != (len(points),) or not np.isfinite(base_weights).all() or np.any(base_weights < 0):
        raise ValueError("weights must be finite, nonnegative, and match points")
    keep = base_weights > 0
    points, base_weights = points[keep], base_weights[keep]
    if len(points) < degree + 1 or len(np.unique(points[:, 0])) < max(degree + 1, 2):
        raise ValueError("Insufficient distinct positive-weight x coordinates")
    x, y = points.T
    center = float((x.max() + x.min()) / 2)
    scale = float((x.max() - x.min()) / 2)
    z = (x - center) / scale
    design = np.polynomial.polynomial.polyvander(z, degree)
    combined_weights = base_weights / base_weights.max()
    previous = None
    for _ in range(max_iterations):
        square_root = np.sqrt(combined_weights)
        coefficients, _, rank, _ = np.linalg.lstsq(design * square_root[:, None], y * square_root, rcond=None)
        if rank < degree + 1:
            raise ValueError("Polynomial fitting design is rank deficient")
        residual = y - design @ coefficients
        mad = float(np.median(np.abs(residual - np.median(residual))))
        robust_scale = max(1.4826 * mad, np.finfo(float).eps * max(1.0, float(np.max(np.abs(y)))) * 100)
        robust_weights = np.minimum(1.0, huber_delta * robust_scale / np.maximum(np.abs(residual), np.finfo(float).tiny))
        if previous is not None and np.allclose(coefficients, previous, rtol=1e-9, atol=1e-10):
            break
        previous = coefficients.copy()
        combined_weights = (base_weights / base_weights.max()) * robust_weights
    physical = Polynomial(coefficients)(Polynomial([-center / scale, 1.0 / scale]))
    return BoundaryPolynomial(tuple(physical.coef), (float(x.min()), float(x.max())))


def boundary_from_pixels(
    coefficients: Sequence[float],
    x_support: Point,
    spacing_mm: Point,
) -> BoundaryPolynomial:
    """Transform a pixel polynomial to physical coordinates, allowing anisotropy.

    Anisotropic spacing is supported when acquisition metadata warrants it; the
    helper must not be used to invent horizontal calibration from image shape.
    """
    sx, sy = _spacing(spacing_mm)
    physical = tuple(sy * float(c) / sx ** i for i, c in enumerate(coefficients))
    return BoundaryPolynomial(physical, (x_support[0] * sx, x_support[1] * sx))


def line_from_pixel_endpoints(
    first_px: Point,
    second_px: Point,
    spacing_mm: Point,
    *,
    track_id: str = "",
) -> FascicleLine:
    """Create a physically calibrated visible line segment from pixel endpoints."""
    spacing = np.asarray(_spacing(spacing_mm))
    endpoints = np.asarray([first_px, second_px], dtype=float)
    if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all():
        raise ValueError("Pixel endpoints must each contain two finite coordinates")
    first, second = endpoints * spacing
    delta = second - first
    length = float(np.linalg.norm(delta))
    return FascicleLine(tuple((first + second) / 2), tuple(delta), (-length / 2, length / 2), track_id)


def _spacing(spacing_mm: Point) -> Point:
    spacing = tuple(float(v) for v in spacing_mm)
    if len(spacing) != 2 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("Pixel spacing must contain two finite positive values")
    return spacing
