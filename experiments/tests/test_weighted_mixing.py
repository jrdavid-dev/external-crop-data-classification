"""Weighted mixed training: place-block splits, block-wise folds and parcel weights."""

import unittest
from unittest.mock import patch

import numpy as np

from experiments.mixed_training import mixing, weighted


def external_rows():
    """Madagascar test parcels in six 0.1-degree blocks, two crops per block, plus development rows."""
    rows = []
    for block in range(6):
        for crop in ("carrot", "potato"):
            for parcel in range(2):
                rows.append({"partition": "test", "partition_group": f"mg-{block}-{crop}-{parcel}", "crop": crop,
                             "lat": -19.95 + .1 * block, "lon": 47.05, "label_status": "reported",
                             "test_cohort": "jecam"})
    for crop in ("carrot", "cabbage", "potato"):
        for parcel in range(3):
            rows.append({"partition": "train", "partition_group": f"eu-{crop}-{parcel}", "crop": crop,
                         "lat": 48.5, "lon": 2.5, "label_status": "reported", "test_cohort": ""})
    return rows


class SplitTests(unittest.TestCase):
    def test_whole_blocks_stay_on_one_side_across_crops(self):
        rows = external_rows()
        for seed in (1, 2, 3):
            assignment = weighted.split_by_place(rows, seed=seed)
            sides = {}
            for r in rows:
                if r["partition"] == "test":
                    sides.setdefault(mixing.block_of(r), set()).add(assignment[r["partition_group"]])
            self.assertTrue(all(len(s) == 1 for s in sides.values()))
            self.assertEqual(set(assignment.values()), {"added", "held_out"})
            added = sum(v == "added" for v in assignment.values())
            self.assertTrue(8 <= added <= 16)  # about half of 24 parcels, in whole blocks of 4

    def test_folds_keep_blocks_together(self):
        rows = external_rows()
        index = np.asarray([i for i, r in enumerate(rows) if r["partition"] == "test"])
        folds = weighted.cv_folds(rows, index)
        by_block = {}
        for i in index:
            by_block.setdefault(mixing.block_of(rows[i]), set()).add(folds[rows[i]["partition_group"]])
        self.assertTrue(all(len(f) == 1 for f in by_block.values()))
        self.assertEqual(set(folds.values()), {0, 1, 2})


class WeightTests(unittest.TestCase):
    def test_added_parcels_count_weight_times_a_development_parcel(self):
        rows = [{"partition_group": g, "crop": "potato", "label_status": "reported"} for g in ("a", "b")]
        captured = {}

        class Forest:
            def __init__(self, **kwargs):
                pass

            def fit(self, X, y, sample_weight):
                captured["w"] = sample_weight
                return self

        with patch("sklearn.ensemble.RandomForestClassifier", Forest):
            weighted.weighted_fit(np.zeros((2, 1)), rows, np.array([0, 1]), {"b"}, 5.0)
        w = captured["w"]
        self.assertAlmostEqual(w[1] / w[0], 5.0)
        self.assertAlmostEqual(float(np.mean(w)), 1.0)

    def test_weight_choice_prefers_smaller_on_ties(self):
        with patch.object(weighted, "macro_f1", side_effect=[0.5, 0.5, 0.4]), \
                patch.object(weighted, "weighted_fit"), patch.object(weighted, "predict", return_value={}):
            rows = external_rows()
            added = np.asarray([i for i, r in enumerate(rows) if r["partition"] == "test"])
            development = np.asarray([i for i, r in enumerate(rows) if r["partition"] == "train"])
            chosen, scores = weighted.choose_weight(np.zeros((len(rows), 1)), rows, development, added, 1)
        self.assertEqual(chosen, 1.0)
        self.assertEqual(scores, {1.0: 0.5, 2.0: 0.5, 5.0: 0.4})


if __name__ == "__main__":
    unittest.main()
