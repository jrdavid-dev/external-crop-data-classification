import unittest

from crop_pipeline.lean_preprocessing import select_cohorts


class LeanPreprocessingTests(unittest.TestCase):
    def test_matched_selection_honors_locked_partitions_and_validation_pixels(self):
        def key(group, x):
            return (group, f"fragment-{group}", "EPSG:32631", x, 0)

        def candidate(group, partition="development"):
            return (None, {"partition_group": group, "partition": partition,
                           "crop": "carrot", "source_dataset": "francecrops"})

        locked_key = key("locked-validation", 10)
        collision = key("train-collision", 10)
        eligible = key("new-train", 20)
        other_validation = key("collection-validation", 30)
        test = key("jecam", 40)
        rows = {
            collision: candidate("train-collision"),
            eligible: candidate("new-train"),
            other_validation: candidate("collection-validation"),
            test: candidate("jecam", "test"),
        }
        selected, _ = select_cohorts(
            {"fixed": dict(rows), "aligned": dict(rows)},
            {"locked-validation": "validation", "train-collision": "train",
             "new-train": "train", "collection-validation": "validation"},
            {cohort: {locked_key: {}} for cohort in
             ("fixed_full", "fixed_matched", "aligned_full", "aligned_matched")},
        )
        self.assertEqual(set(selected["fixed_matched"]), {eligible, test})
        self.assertEqual(selected["fixed_matched"], selected["aligned_matched"])
        self.assertEqual(selected["aligned_matched"][eligible][1]["partition"], "train")

    def test_reserved_test_cannot_become_a_development_field(self):
        key = ("jecam", "fragment", "EPSG:32631", 10, 0)
        details = {"partition": "test", "partition_group": "jecam", "crop": "carrot"}
        with self.assertRaisesRegex(ValueError, "reserved test field"):
            select_cohorts({"fixed": {key: (None, details)}, "aligned": {key: (None, details)}},
                           {"jecam": "train"}, {"fixed_full": {}})


if __name__ == "__main__":
    unittest.main()
