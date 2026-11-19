import unittest

from crop_pipeline.training.geographic_validation import spatial_folds


class GeographicValidationTests(unittest.TestCase):
    def test_cells_and_parcels_stay_together(self):
        rows = [
            {"crop": crop, "site_cell": f"45,{cell}", "partition_group": f"{cell}-{crop}"}
            for cell in range(10)
            for crop in ("carrot", "cabbage", "potato")
        ]
        groups = {row["partition_group"] for row in rows}
        seen = set()
        for train, held in spatial_folds(rows * 2):
            self.assertEqual(train | held, groups)
            self.assertFalse(train & held)
            self.assertFalse(seen & held)
            self.assertFalse({g.split("-")[0] for g in train} & {g.split("-")[0] for g in held})
            seen.update(held)
        self.assertEqual(seen, groups)

    def test_parcel_spanning_cells_merges_blocks(self):
        rows = [
            {"crop": "carrot", "partition_group": str(cell), "site_cell": f"45,{cell}"}
            for cell in range(10)
        ]
        rows.append({**rows[0], "site_cell": rows[1]["site_cell"]})
        for _, held in spatial_folds(rows):
            self.assertEqual("0" in held, "1" in held)
