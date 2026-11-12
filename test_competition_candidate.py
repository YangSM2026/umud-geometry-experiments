import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import build_competition_candidate as candidate


class CompetitionCandidateTests(unittest.TestCase):
    def make_heads(self):
        prior = pd.DataFrame({"image_id": ["a", "b"], "pa_deg": [11., 12.], "fl_mm": [60., 70.], "mt_mm": [21., 22.]})
        heads = {tag: pd.DataFrame({"image_id": ["a", "b"], "pa_deg": [20. + h] * 2,
            "fl_mm": [80. + h] * 2, "mt_mm": [25. + h] * 2, "ok": [True, True]}) for h, tag in enumerate(candidate.TAGS)}
        return heads, prior

    def test_all_failed_preserves_exact_prior_whole_triplet(self):
        heads, prior = self.make_heads()
        for frame in heads.values(): frame["ok"] = False
        out, audit, fallback = candidate.combine_heads(heads, prior)
        pd.testing.assert_frame_equal(out, prior)
        self.assertTrue(fallback.all())
        self.assertTrue((audit.normalized_weight == 0).all())
        self.assertTrue((audit.fallback == "prior_failure_aware_triplet").all())

    def test_all_failed_without_failure_aware_source_aborts(self):
        heads, prior = self.make_heads()
        for frame in heads.values(): frame["ok"] = False
        with self.assertRaisesRegex(ValueError, "unavailable"):
            candidate.combine_heads(heads, prior, allow_prior_fallback=False)

    def test_one_valid_head_uses_same_triplet(self):
        heads, prior = self.make_heads()
        for tag in candidate.TAGS[1:]: heads[tag]["ok"] = False
        out, audit, fallback = candidate.combine_heads(heads, prior)
        np.testing.assert_array_equal(out[list(candidate.COLS)], heads["test"][list(candidate.COLS)])
        self.assertFalse(fallback.any())
        self.assertTrue((audit[audit["head"] == "test"].normalized_weight == 1).all())

    def test_mismatched_identifiers_abort(self):
        heads, prior = self.make_heads(); heads["ens_f0"].loc[0, "image_id"] = "bad"
        with self.assertRaisesRegex(ValueError, "ID/order"):
            candidate.combine_heads(heads, prior)

    def test_texture_is_group_median_then_original_sine_fusion(self):
        rows = [{"image_id": i, "pa_deg": 10., "fl_mm": 100., "mt_mm": 20., "ok": True} for i in ["a", "b"]]
        old = pd.DataFrame({"run": [77, 77], "pa_tex": [20., 40.]})
        segs = [np.array([[.5, 10., 8., 100.]]), np.array([[.6, 11., 8., 100.]])]
        with patch.object(candidate.vendor, "pa_depth_fit", return_value=14.) as fit:
            out = candidate.fuse_head(rows, segs, old)
        fit.assert_called_once()
        np.testing.assert_allclose(out.pa_deg, 22.)
        np.testing.assert_allclose(out.fl_mm, .75 * 100 + .25 * 20 / np.sin(np.radians(22.)))
        self.assertEqual(out.run.tolist(), [77, 77])


if __name__ == "__main__": unittest.main()
