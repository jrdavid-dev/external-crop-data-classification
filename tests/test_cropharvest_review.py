"""Regression coverage for the notebook's extracted CropHarvest label audit."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from shapely.geometry import Point, Polygon, box, mapping

from crop_pipeline.sources.cropharvest_review import review_labels


def feature(index, label="potato", dataset="jecam", geometry=None, **changes):
    if geometry is None:
        geometry = box(index, 10, index + 0.01, 10.01)
    properties = {
        "index": index,
        "dataset": dataset,
        "label": label,
        "is_crop": 1,
        "is_test": False,
        "collection_date": "2020-06-15",
        "planting_date": None,
        "harvest_date": None,
        "export_end_date": "2020-12-31",
        "classification_label": (
            "root_tuber" if label == "potato" else "vegetables_melons"
        ),
        "lon": geometry.centroid.x,
        "lat": geometry.centroid.y,
    }
    properties.update(changes)
    return {"type": "Feature", "geometry": mapping(geometry), "properties": properties}


class CropHarvestReviewTests(unittest.TestCase):
    def review(self, folder, features, **kwargs):
        root = Path(folder)
        path = root / "labels.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
        return review_labels(
            path, root / "missing_boundaries.zip", root / "review", **kwargs
        )

    def test_aliases_reserved_labels_and_existing_screening_rules(self):
        invalid = Polygon([(8, 10), (8.01, 10.01), (8, 10.01), (8.01, 10), (8, 10)])
        rows = [
            feature(1),
            feature(2, " CAROTTE "),
            feature(3, "chou", is_test=True),
            feature(4, dataset="croplands"),
            feature(5, geometry=Point(5, 10)),
            feature(6, "vegetables"),
            feature(7, "cauliflower and brocoli"),
            feature(8, geometry=invalid),
            # Unknown classification group remains a diagnostic, not a new exclusion.
            feature(9, classification_label="unexpected"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            result = self.review(folder, rows)
            table = result["reviewed_labels"].set_index("uid")
            self.assertEqual(
                set(table.index),
                {"jecam:1", "jecam:2", "jecam:3", "croplands:4", "jecam:8", "jecam:9"},
            )
            self.assertEqual(table.loc["jecam:2", "crop"], "carrot")
            self.assertTrue(table.loc["jecam:3", "is_test"])
            self.assertTrue(table.loc["jecam:3", "reliable"])
            self.assertEqual(
                table.loc["croplands:4", "fail_reason"], "Source not high grade"
            )
            self.assertEqual(table.loc["jecam:8", "fail_reason"], "Invalid polygon")
            self.assertTrue(table.loc["jecam:9", "group_mismatch"])
            self.assertTrue(table.loc["jecam:9", "reliable"])
            self.assertTrue(table.country.eq(table.dataset).all())
            exported = pd.read_csv(result["screened_labels_output"])
            self.assertIn("reliable", exported.columns)
            self.assertNotIn("geometry", exported.columns)
            self.assertNotIn("fail_reason", exported.columns)
            self.assertEqual(len(exported), len(table))
            self.assertEqual(result["summary"].loc["potato", "excluded"], 2)
            # Auditing does not replace or alter the source labels.
            self.assertEqual(
                json.loads((Path(folder) / "labels.geojson").read_text())["features"],
                json.loads(json.dumps(rows)),
            )

    def test_conflicting_crops_and_cross_source_duplicates_are_excluded(self):
        shared = box(1, 10, 1.01, 10.01)
        duplicate = box(2, 10, 2.01, 10.01)
        rows = [
            feature(1, geometry=shared),
            feature(2, "carrot", geometry=shared),
            feature(3, geometry=duplicate),
            feature(4, dataset="kenya", geometry=duplicate),
            feature(5),
        ]
        with tempfile.TemporaryDirectory() as folder:
            result = self.review(folder, rows)
            self.assertTrue(
                result["label_table"].set_index("uid").loc["jecam:1", "season_conflict"]
            )
            table = result["reviewed_labels"].set_index("uid")
            self.assertNotIn("jecam:1", table.index)
            self.assertNotIn("jecam:2", table.index)
            self.assertEqual(
                table.loc["jecam:3", "fail_reason"], "Duplicate across sources"
            )
            self.assertEqual(
                table.loc["kenya:4", "fail_reason"], "Duplicate across sources"
            )
            self.assertTrue(table.loc["jecam:5", "reliable"])

    def test_observation_year_decisions_and_overrides_are_saved(self):
        rows = [
            feature(1, collection_date="2019-01-01"),
            feature(2, collection_date="2018-01-01"),
            feature(3, collection_date=None),
            feature(4, collection_date="2016-06-15", export_end_date="2016-12-31"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            result = self.review(
                folder, rows, observation_year_overrides={"jecam:3": 2020}
            )
            audit = pd.read_csv(result["observation_year_audit"]).set_index("uid")
            self.assertEqual(len(audit), 4)
            self.assertEqual(audit.loc["jecam:2", "year_resolution_status"], "pending")
            self.assertEqual(
                audit.loc["jecam:3", "year_resolution_reason"], "explicit_year_override"
            )
            self.assertEqual(set(result["label_table"].uid), {"jecam:1", "jecam:3"})
            self.assertEqual(result["summary"].loc["carrot", "candidates"], 0)


if __name__ == "__main__":
    unittest.main()
