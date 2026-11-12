"""Invariance, failure and optimization checks; these do not establish accuracy."""

import itertools
import math
import unittest

from reliability import HeadEstimate, aggregate_heads, select_fascicles


def head(name, value, family="folds", **kwargs):
    return HeadEstimate(name, family, value, value / 2, value * 3, **kwargs)


def fascicle(name, x, extra, support=1.0, valid=True):
    return dict(track_id=name, anchor_x=x, extrapolated_fraction=extra, support=support, valid=valid)


class ReliabilityTests(unittest.TestCase):
    def test_string_validity_is_rejected(self):
        for value in ["False", "True", 0, 1, None]:
            with self.assertRaisesRegex(ValueError, "explicit bool"):
                aggregate_heads([head("a", 20, valid=value)])
            with self.assertRaisesRegex(ValueError, "explicit bool"):
                select_fascicles([fascicle("a", 0, .1, valid=value)], min_separation=1)

    def test_float_sum_tie_uses_support(self):
        records=[fascicle(str(i),x,e,s) for i,(x,e,s) in enumerate([(3,.5,.8),(9,.3,.4),(18,.4,.5),(29,.5,.7)])]
        selected=select_fascicles(records,min_separation=3,max_count=3).selected
        self.assertEqual([r["track_id"] for r in selected],["0","1","2"])

    def test_median_rejects_minority_extreme(self):
        result = aggregate_heads([head("a", 20), head("b", 21), head("c", 100)])
        self.assertEqual(result.estimate["mt_mm"], 21)
        self.assertEqual(result.diagnostics["targets"]["mt_mm"]["weighted_mad"], 1)

    def test_two_equal_heads_use_midpoint(self):
        self.assertEqual(aggregate_heads([head("a", 20), head("b", 30)]).estimate["mt_mm"], 25)

    def test_record_and_checkpoint_alias_duplication_invariance(self):
        records = [head("a", 20, source_id="checkpoint-a"), head("b", 21, source_id="checkpoint-b"), head("c", 40, source_id="checkpoint-c")]
        original = aggregate_heads(records)
        duplicated = aggregate_heads(records + [records[0], head("renamed-a", 20, source_id="checkpoint-a")])
        self.assertEqual(original.estimate, duplicated.estimate)
        self.assertEqual(original.source_weights, duplicated.source_weights)
        self.assertEqual(original.diagnostics["targets"], duplicated.diagnostics["targets"])
        self.assertEqual(duplicated.diagnostics["duplicate_record_count"], 2)

    def test_conflicting_alias_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            aggregate_heads([head("a", 20, source_id="same"), head("alias", 30, source_id="same")])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            aggregate_heads([head("a", 20, source_id="same"), head("alias", 20, family="other", source_id="same")])

    def test_family_mass_does_not_depend_on_fold_count(self):
        result = aggregate_heads([head("base", 20, "base"), head("f1", 30), head("f2", 32)], family_weights={"base": 0.5, "folds": 0.5})
        self.assertEqual(result.source_weights, {"base": 0.5, "f1": 0.25, "f2": 0.25})
        failed = aggregate_heads([head("base", 20, "base"), head("f1", 30), head("f2", 32, valid=False)], family_weights={"base": 0.5, "folds": 0.5})
        self.assertEqual(failed.source_weights, {"base": 0.5, "f1": 0.5})

    def test_head_weights_apply_within_family(self):
        result = aggregate_heads([head("a", 20, weight=3), head("b", 50, weight=1)])
        self.assertEqual(result.estimate["mt_mm"], 20)
        self.assertEqual(result.source_weights, {"a": 0.75, "b": 0.25})

    def test_missing_family_weight_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing explicit"):
            aggregate_heads([head("a", 20)], family_weights={"typo": 1})

    def test_no_valid_sources_has_no_fabricated_prediction(self):
        result = aggregate_heads([head("a", 20, valid=False), head("b", math.nan)])
        self.assertEqual(result.status, "no_valid_sources")
        self.assertIsNone(result.estimate)
        self.assertEqual(result.source_weights, {})
        self.assertEqual(result.diagnostics["targets"], {})
        self.assertEqual(aggregate_heads([]).status, "no_valid_sources")

    def test_single_source_reports_unavailable_disagreement(self):
        result = aggregate_heads([head("a", 20), head("b", 25, valid=False)])
        self.assertEqual(result.status, "single_source")
        self.assertEqual(result.estimate["mt_mm"], 20)
        self.assertIsNone(result.diagnostics["targets"]["mt_mm"]["weighted_sd"])
        self.assertEqual(result.diagnostics["disagreement_status"], "unavailable")

    def test_matching_models_have_zero_observed_disagreement_only(self):
        result = aggregate_heads([head("a", 20), head("b", 20)])
        self.assertEqual(result.diagnostics["targets"]["mt_mm"]["weighted_sd"], 0)
        self.assertIn("uncalibrated", result.diagnostics["disagreement_status"])

    def test_permutation_and_unit_equivariance(self):
        records = [head("a", 20), head("b", 22), head("c", 70)]
        expected = aggregate_heads(records)
        for permutation in itertools.permutations(records):
            actual = aggregate_heads(permutation)
            self.assertEqual(actual.estimate, expected.estimate)
            self.assertEqual(actual.source_weights, expected.source_weights)
        scaled = [HeadEstimate(h.head_id, h.family_id, h.mt_mm * 10, h.pa_deg, h.fl_mm * 10) for h in records]
        result = aggregate_heads(scaled)
        self.assertEqual(result.estimate["mt_mm"], expected.estimate["mt_mm"] * 10)
        self.assertEqual(result.estimate["pa_deg"], expected.estimate["pa_deg"])

    def test_zero_and_nonphysical_values_excluded(self):
        result = aggregate_heads([head("a", -1), head("b", 20, weight=0)])
        self.assertEqual(result.status, "no_valid_sources")
        with self.assertRaises(ValueError):
            aggregate_heads([head("a", 20, weight=-1)])


class FascicleSelectionTests(unittest.TestCase):
    def test_minimizes_extrapolation_given_separation(self):
        candidates = [fascicle("a", 0, 0.1), fascicle("b", 1, 0.01), fascicle("c", 10, 0.2), fascicle("d", 20, 0.3)]
        result = select_fascicles(candidates, min_separation=5)
        self.assertEqual([x["track_id"] for x in result.selected], ["b", "c", "d"])

    def test_global_count_not_greedy_best_first(self):
        candidates = [fascicle("left", 0, 0.2), fascicle("middle", 5, 0), fascicle("right", 10, 0.2)]
        result = select_fascicles(candidates, min_separation=6, max_count=2)
        self.assertEqual([x["track_id"] for x in result.selected], ["left", "right"])

    def test_optimizer_matches_exhaustive_reference(self):
        candidates = [fascicle(str(i), x, e, s) for i, (x, e, s) in enumerate([(0, .2, .8), (2, .1, .4), (4, .3, .9), (6, .1, .5), (9, .2, .8), (12, .3, .7)])]
        for spacing in [0, 2, 4, 7, 20]:
            choices = [combo for k in range(4) for combo in itertools.combinations(candidates, k) if all(b["anchor_x"] - a["anchor_x"] >= spacing for a, b in zip(combo, combo[1:]))]
            optimum = min(choices, key=lambda c: (-len(c), sum(x["extrapolated_fraction"] for x in c), -sum(x["support"] for x in c), tuple(x["track_id"] for x in c)))
            actual = select_fascicles(candidates, min_separation=spacing)
            self.assertEqual(actual.selected, optimum)

    def test_same_insertion_not_selected_twice_at_zero_spacing(self):
        result = select_fascicles([fascicle("a", 1, .2), fascicle("b", 1, .1)], min_separation=0)
        self.assertEqual(len(result.selected), 1)
        self.assertEqual(result.selected[0]["track_id"], "b")

    def test_incomplete_empty_and_invalid(self):
        result = select_fascicles([fascicle("a", 1, .2), fascicle("b", 9, .1, valid=False)], min_separation=1)
        self.assertEqual(result.status, "insufficient_distinct_fascicles")
        self.assertEqual(result.rejected_count, 1)
        self.assertEqual(select_fascicles([], min_separation=1).status, "no_valid_fascicles")
        with self.assertRaises(ValueError):
            select_fascicles([], min_separation=-1)

    def test_order_invariance_and_support_tie_break(self):
        candidates = [fascicle("a", 1, .1, .5), fascicle("b", 2, .1, .9), fascicle("c", 10, .1)]
        selected = select_fascicles(candidates, min_separation=3, max_count=2)
        self.assertEqual([x["track_id"] for x in selected.selected], ["b", "c"])
        for permutation in itertools.permutations(candidates):
            self.assertEqual(select_fascicles(permutation, min_separation=3, max_count=2).selected, selected.selected)


if __name__ == "__main__":
    unittest.main()
