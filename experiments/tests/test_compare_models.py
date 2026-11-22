"""Shared-parcel scoring for the window, variant and annual comparison."""

import unittest

from experiments.annual import compare_models


class SharedParcelTests(unittest.TestCase):
    def test_models_are_scored_on_parcels_all_of_them_predicted(self):
        models = {
            "a": {"p1": ("jecam", "carrot", "carrot"), "p2": ("jecam", "potato", "carrot"),
                  "p3": ("california_dwr_carrot_2023_v1", "carrot", "carrot")},
            "b": {"p1": ("jecam", "carrot", "potato"), "p3": ("california_dwr_carrot_2023_v1", "carrot", "potato")},
        }
        parcels, predictions = compare_models.shared(models, "jecam")
        self.assertEqual(set(parcels), {"p1"})
        self.assertEqual(predictions["b"], {"p1": "potato"})
        parcels, _ = compare_models.shared(models, "combined")
        self.assertEqual(set(parcels), {"p1", "p3"})

    def test_reference_disagreement_is_refused(self):
        models = {"a": {"p1": ("jecam", "carrot", "carrot")}, "b": {"p1": ("jecam", "potato", "carrot")}}
        with self.assertRaises(ValueError):
            compare_models.shared(models, "jecam")


if __name__ == "__main__":
    unittest.main()
