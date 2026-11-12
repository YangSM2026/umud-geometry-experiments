"""Robust, provenance-aware aggregation for ultrasound geometry candidates.

This module contains no trained parameters, leaderboard thresholds, image IDs,
or calibration claims. Head disagreement measures sensitivity to the available
models; correlated models and shared bias prevent interpreting it as a clinical
confidence interval. A single surviving source has *unknown* disagreement.

Stable ``source_id`` values should identify the actual checkpoint(s), e.g. a hash
of the two checkpoint hashes. Renaming/repeating a source must not increase its
weight. Without source_id, head_id is used and callers own its uniqueness.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence


TARGETS = ("mt_mm", "pa_deg", "fl_mm")


@dataclass(frozen=True)
class HeadEstimate:
    head_id: str
    family_id: str
    mt_mm: float | None
    pa_deg: float | None
    fl_mm: float | None
    valid: bool = True
    weight: float = 1.0
    source_id: str | None = None


@dataclass(frozen=True)
class AggregateResult:
    estimate: dict[str, float] | None
    status: str
    source_weights: dict[str, float]
    family_weights: dict[str, float]
    diagnostics: dict[str, Any]
    excluded_sources: dict[str, str]


@dataclass(frozen=True)
class FascicleSelection:
    selected: tuple[Any, ...]
    status: str
    eligible_count: int
    rejected_count: int
    requested_count: int
    min_separation: float


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _positive_weight(value: Any, label: str) -> float:
    if not _finite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _canonical_value(value: Any) -> Any:
    # Invalid records may carry NaN sentinels, which are not equal to themselves.
    if isinstance(value, Real) and math.isnan(value):
        return "NaN"
    return value


def _signature(head: HeadEstimate) -> tuple[Any, ...]:
    return (
        head.family_id, head.valid, float(head.weight),
        *(_canonical_value(getattr(head, target)) for target in TARGETS),
    )


def _weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    """Inverse weighted empirical CDF, with a midpoint convention at median ties."""
    pairs = sorted(zip(values, weights))
    total = math.fsum(weight for _, weight in pairs)
    grouped: list[tuple[float, float]] = []
    for value, weight in pairs:
        if grouped and value == grouped[-1][0]:
            grouped[-1] = value, grouped[-1][1] + weight
        else:
            grouped.append((value, weight))
    target = q * total
    cumulative = 0.0
    for index, (value, weight) in enumerate(grouped):
        cumulative = math.fsum((cumulative, weight))
        tied = math.isclose(cumulative, target, rel_tol=0.0, abs_tol=8 * math.ulp(total))
        if cumulative > target and not tied:
            return float(value)
        if tied:
            if q == 0.5 and index + 1 < len(grouped):
                return float((value + grouped[index + 1][0]) / 2)
            return float(value)
    return float(grouped[-1][0])


def aggregate_heads(
    heads: Iterable[HeadEstimate], *, family_weights: Mapping[str, float] | None = None,
) -> AggregateResult:
    """Aggregate valid endpoints with family-balanced coordinatewise medians.

    A family's configured mass is shared by its unique, valid, positively
    weighted sources in proportion to ``HeadEstimate.weight``. Available family
    masses are then renormalized. This separates model-family importance from
    the number of folds and prevents repeated checkpoint records gaining weight.

    If family_weights is supplied, every input family must be listed explicitly;
    zero disables a family. Conflicting records for a repeated source_id raise an
    error rather than resolving by input order. Target values remain associated
    within input heads, but the robust output uses per-target weighted medians;
    it is not asserted to be a single anatomically reconstructable geometry.
    """
    records = list(heads)
    unique: dict[str, HeadEstimate] = {}
    for head in records:
        if not isinstance(head.valid, bool):
            raise ValueError("Head validity must be an explicit bool")
        if not head.head_id or not head.family_id:
            raise ValueError("head_id and family_id must be nonempty")
        _positive_weight(head.weight, f"head {head.head_id} weight")
        source = head.source_id if head.source_id is not None else head.head_id
        if not source:
            raise ValueError("source_id must be nonempty when supplied")
        if source in unique and _signature(unique[source]) != _signature(head):
            raise ValueError(f"Conflicting repeated source: {source}")
        unique[source] = head
    configured = {}
    for family in sorted({head.family_id for head in unique.values()}):
        if family_weights is not None and family not in family_weights:
            raise ValueError(f"Missing explicit weight for family: {family}")
        configured[family] = _positive_weight(
            1.0 if family_weights is None else family_weights[family], f"family {family} weight"
        )

    excluded = {}
    usable: dict[str, HeadEstimate] = {}
    for source, head in sorted(unique.items()):
        values = [getattr(head, target) for target in TARGETS]
        if not head.valid:
            excluded[source] = "head_marked_invalid"
        elif not all(_finite(value) for value in values):
            excluded[source] = "nonfinite_or_missing_measurement"
        elif head.mt_mm <= 0 or head.fl_mm <= 0 or not 0 <= head.pa_deg <= 90:
            excluded[source] = "nonphysical_measurement"
        elif head.weight == 0 or configured[head.family_id] == 0:
            excluded[source] = "zero_configured_weight"
        else:
            usable[source] = head
    active_families = sorted({head.family_id for head in usable.values()})
    family_total = math.fsum(configured[family] for family in active_families)
    normalized_families = {
        family: configured[family] / family_total for family in active_families
    }
    source_weights = {}
    for family in active_families:
        sources = [source for source, head in usable.items() if head.family_id == family]
        denominator = math.fsum(usable[source].weight for source in sources)
        for source in sources:
            source_weights[source] = normalized_families[family] * usable[source].weight / denominator

    count = len(usable)
    status = "no_valid_sources" if not count else "single_source" if count == 1 else "multiple_sources"
    diagnostics: dict[str, Any] = {
        "input_record_count": len(records),
        "unique_source_count": len(unique),
        "duplicate_record_count": len(records) - len(unique),
        "valid_source_count": count,
        "active_family_count": len(active_families),
        "effective_weight_count": None if not count else 1.0 / math.fsum(w * w for w in source_weights.values()),
        "effective_weight_count_interpretation": "weight concentration only; not an independent sample size",
        "disagreement_status": "unavailable" if count < 2 else "uncalibrated_model_disagreement",
        "uncertainty_interpretation": "not a confidence interval; shared bias and model dependence are not measured",
        "targets": {},
    }
    if not count:
        return AggregateResult(None, status, {}, {}, diagnostics, excluded)

    estimate = {}
    weights = [source_weights[source] for source in usable]
    for target in TARGETS:
        values = [float(getattr(head, target)) for head in usable.values()]
        center = _weighted_quantile(values, weights, 0.5)
        estimate[target] = center
        mean = math.fsum(v * w for v, w in zip(values, weights)) / math.fsum(weights)
        diagnostics["targets"][target] = {
            "weighted_mean": mean,
            "weighted_mad": None if count < 2 else _weighted_quantile([abs(v - center) for v in values], weights, 0.5),
            "weighted_iqr": None if count < 2 else _weighted_quantile(values, weights, 0.75) - _weighted_quantile(values, weights, 0.25),
            "range": None if count < 2 else max(values) - min(values),
            "weighted_sd": None if count < 2 else math.sqrt(math.fsum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / math.fsum(weights)),
        }
    return AggregateResult(estimate, status, source_weights, normalized_families, diagnostics, excluded)


def _get(candidate: Any, name: str, default: Any = None) -> Any:
    return candidate.get(name, default) if isinstance(candidate, Mapping) else getattr(candidate, name, default)


def select_fascicles(
    candidates: Iterable[Any], *, min_separation: float, max_count: int = 3,
) -> FascicleSelection:
    """Select spatially separated fascicles with minimum total extrapolation.

    Candidates can be endpoint-geometry dataclasses or mappings. Required fields:
    ``track_id``, ``anchor_x`` (deep insertion position), ``extrapolated_fraction``,
    ``support`` and ``valid``. Return the original objects, ordered by anchor_x.

    The caller must prespecify minimum insertion separation in the SAME UNITS as
    anchor_x. No image-wide spacing, leaderboard-derived cutoff, or claim that
    least-extrapolated selection is most accurate is embedded here. Zero spacing
    still requires distinct insertion coordinates. Dynamic programming first
    maximizes count up to max_count, then minimizes summed extrapolation, then
    maximizes summed support. It does not greedily sacrifice feasible fascicles.
    """
    _positive_weight(min_separation, "min_separation")
    if not isinstance(max_count, int) or isinstance(max_count, bool) or max_count < 1:
        raise ValueError("max_count must be a positive integer")
    records = list(candidates)
    rows = []
    seen_ids: set[str] = set()
    for candidate in records:
        validity = _get(candidate, "valid", False)
        if not isinstance(validity, bool):
            raise ValueError("Fascicle validity must be an explicit bool")
        if not validity:
            continue
        track_id = _get(candidate, "track_id")
        x = _get(candidate, "anchor_x")
        extrapolated = _get(candidate, "extrapolated_fraction")
        support = _get(candidate, "support")
        if track_id is None or not all(_finite(value) for value in (x, extrapolated, support)):
            continue
        if not 0 <= extrapolated <= 1 or not 0 <= support <= 1:
            continue
        key = str(track_id)
        if key in seen_ids:
            raise ValueError(f"Repeated fascicle track_id: {key}")
        seen_ids.add(key)
        rows.append((float(x), float(extrapolated), float(support), key, candidate))
    rows.sort(key=lambda row: (row[0], row[3]))
    xs = [row[0] for row in rows]
    # State is (sum extrapolation, negative sum support, track-id tuple, row-index tuple).
    states: list[list[tuple | None]] = [[None] * (max_count + 1) for _ in range(len(rows) + 1)]
    states[0][0] = (0.0, 0.0, (), ())
    for index, row in enumerate(rows):
        states[index + 1] = states[index].copy()
        compatible = bisect_left(xs, row[0], 0, index) if min_separation == 0 else bisect_right(xs, row[0] - min_separation, 0, index)
        for count in range(1, max_count + 1):
            previous = states[compatible][count - 1]
            if previous is None:
                continue
            chosen_indices = previous[3] + (index,)
            choice = (math.fsum(rows[i][1] for i in chosen_indices),
                      -math.fsum(rows[i][2] for i in chosen_indices),
                      previous[2] + (row[3],), chosen_indices)
            incumbent = states[index + 1][count]
            def preferable(a, b):
                for pos in (0, 1):
                    tolerance = 8 * math.ulp(max(abs(a[pos]), abs(b[pos]), 1.0))
                    if abs(a[pos] - b[pos]) > tolerance:
                        return a[pos] < b[pos]
                return a[2] < b[2]
            if incumbent is None or preferable(choice, incumbent):
                states[index + 1][count] = choice
    chosen_count = next(count for count in range(max_count, -1, -1) if states[-1][count] is not None)
    indices = states[-1][chosen_count][3]
    selected = tuple(rows[index][4] for index in indices)
    status = "complete" if chosen_count == max_count else "insufficient_distinct_fascicles" if chosen_count else "no_valid_fascicles"
    return FascicleSelection(selected, status, len(rows), len(records) - len(rows), max_count, float(min_separation))
