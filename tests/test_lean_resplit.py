import unittest

from crop_pipeline.lean_resplit import resplit
from crop_pipeline.windows.external import DEVELOPMENT


class LeanResplitTests(unittest.TestCase):
    def test_fresh_split_uses_all_eligible_matched_fields_and_preserves_test(self):
        rows = {}
        for crop_index, crop in enumerate(("carrot", "cabbage", "potato")):
            for index in range(8):
                group = f"{crop}-{index}"
                key = (group, group, "EPSG:32631", crop_index * 1000 + index * 10, 0)
                rows[key] = (None, {"partition_group": group, "partition": DEVELOPMENT,
                                    "crop": crop, "source_dataset": "francecrops"})
        jecam = ("jecam", "reserved", "EPSG:32631", 4000, 0)
        rows[jecam] = (None, {"partition_group": "jecam", "partition": "test",
                             "crop": "carrot", "source_dataset": "cropharvest"})
        split, selected, _ = resplit({"fixed": dict(rows), "aligned": dict(rows)})
        self.assertEqual(len(split["matched_parcel_groups"]), 24)
        self.assertEqual(list(split["assignments"].values()).count("validation"), 6)
        self.assertNotIn("jecam", split["assignments"])
        self.assertEqual(selected["fixed_matched"], selected["aligned_matched"])
        self.assertEqual(selected["fixed_matched"][jecam][1]["partition"], "test")
        self.assertEqual(len(selected["fixed_matched"]), 25)


if __name__ == "__main__":
    unittest.main()
