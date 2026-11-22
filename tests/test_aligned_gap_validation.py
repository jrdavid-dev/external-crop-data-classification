import unittest

from crop_pipeline.training.aligned_gap_validation import locked_training


class AlignedGapValidationTests(unittest.TestCase):
    def test_locked_validation_and_duplicate_locations_stay_out_of_training(self):
        def pixel(group, crop, x):
            return {"partition_group": group, "region_id": group, "crop": crop,
                    "source_crs": "EPSG:32631", "x": x, "y": 0,
                    "source_dataset": "francecrops"}

        held = pixel("held", "cabbage", 0)
        candidates = [held, pixel("overlap", "cabbage", 0), pixel("old", "cabbage", 10),
                      pixel("new_train", "carrot", 20), pixel("new_val", "carrot", 30),
                      pixel("new_potato", "potato", 40), pixel("unknown", "potato", 50)]
        rows = locked_training(
            candidates, {"held": "validation", "overlap": "train", "old": "train"},
            {"new_train", "new_val", "new_potato"},
            {"held": "train", "old": "validation", "new_train": "train",
             "new_val": "validation", "new_potato": "train", "unknown": "train"},
            [held],
        )
        self.assertEqual({row["partition_group"] for row in rows}, {"old", "new_train", "new_potato"})
        self.assertTrue(all(row["partition"] == "train" for row in rows))


if __name__ == "__main__":
    unittest.main()
