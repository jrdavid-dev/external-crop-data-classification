import unittest
from unittest.mock import patch

import numpy as np

from crop_pipeline.training import external_report as reporting


class ExternalReportTests(unittest.TestCase):
    def test_metric_calculations(self):
        matrix = np.array([[3, 1, 0], [0, 1, 1], [1, 0, 2]])
        values = reporting.metric_values(matrix)
        expected = {
            ("precision", "carrot"): 3 / 4,
            ("recall", "cabbage"): 1 / 2,
            ("f1", "potato"): 2 / 3,
            ("overall_accuracy", ""): 6 / 9,
            ("kappa", ""): (54 - 29) / (81 - 29),
            ("macro_f1", ""): (3 / 4 + 1 / 2 + 2 / 3) / 3,
            ("balanced_accuracy", ""): (3 / 4 + 1 / 2 + 2 / 3) / 3,
        }
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(values[key], value)

    def test_incomplete_crop_coverage_and_undefined_kappa(self):
        parcels = {"p1": ("test", "carrot", "carrot")}
        _, points, samples = reporting.evaluate(parcels, "carrot", 10, np.random.default_rng(8))
        for metric in ("macro_f1", "balanced_accuracy", "kappa"):
            key = (metric, "")
            result = reporting.interval(points["model"][key], samples["model"][key])
            self.assertEqual(result, dict(estimate=None, ci_low=None, ci_high=None, valid_replicates=0))
        self.assertEqual(points["model"]["f1", "carrot"], 1)
        self.assertEqual(points["model"]["recall", "cabbage"], 0)
        self.assertEqual(points["model"]["precision", "cabbage"], 0)

    def test_model_and_majority_reuse_the_same_draw(self):
        parcels = {
            "p1": ("a", "carrot", "carrot"),
            "p2": ("a", "carrot", "potato"),
            "p3": ("a", "cabbage", "cabbage"),
            "p4": ("b", "potato", "potato"),
        }
        rng = np.random.default_rng(8)
        draws = np.array([[0, 0, 2, 3], [1, 1, 2, 3]])
        with patch.object(reporting, "parcel_draws", return_value=draws) as resample:
            _, _, samples = reporting.evaluate(parcels, "carrot", 2, rng)
        resample.assert_called_once_with(list(parcels.values()), 2, rng)
        key = ("macro_f1", "")
        np.testing.assert_allclose(samples["model"][key], [1, 0.5])
        np.testing.assert_allclose(samples["model"][key] - samples["majority"][key],
                                   [1 - 2 / 9, 0.5 - 2 / 9])

    def test_resampling_preserves_each_campaign_and_reference_crop_count(self):
        rows = [("a", "carrot", "carrot"), ("a", "carrot", "potato"),
                ("b", "carrot", "carrot"), ("b", "potato", "potato")]
        draws = reporting.parcel_draws(rows, 20, np.random.default_rng(8))
        self.assertTrue(np.isin(draws[:, :2], [0, 1]).all())
        np.testing.assert_array_equal(draws[:, 2:], np.tile([2, 3], (20, 1)))
        self.assertTrue((draws[:, 0] == draws[:, 1]).any())

    def test_revised_scope_filters_reference_crops_and_recomputes_f1(self):
        mixed = "non_jecam_carrot_potato_supplement_v1"
        parcels = {
            "a": ("jecam", "carrot", "carrot"),
            "b": ("jecam", "cabbage", "cabbage"),
            "c": (mixed, "potato", "potato"),
            "d": (mixed, "potato", "carrot"),
            "e": (mixed, "carrot", "potato"),
            "f": ("non_jecam_cabbage_supplement_v1", "cabbage", "potato"),
        }
        retained = {key: row for key, row in parcels.items() if reporting.retained_reference(*row[:2])}
        self.assertEqual(set(retained), {"a", "b", "c", "d"})
        _, points, _ = reporting.evaluate(retained, "carrot", 10, np.random.default_rng(8))
        self.assertAlmostEqual(points["model"]["f1", "potato"], 2 / 3)
        self.assertAlmostEqual(points["model"]["macro_f1", ""], 7 / 9)

    def test_matching_compares_identities_and_labels(self):
        fixed = {"a": ("jecam", "carrot", "carrot")}
        reporting.require_matched(fixed, {"a": ("jecam", "carrot", "potato")})
        for aligned in ({"b": ("jecam", "carrot", "carrot")},
                        {"a": ("jecam", "potato", "carrot")}):
            with self.subTest(aligned=aligned), self.assertRaisesRegex(ValueError, "identical parcel IDs"):
                reporting.require_matched(fixed, aligned)


if __name__ == "__main__":
    unittest.main()
