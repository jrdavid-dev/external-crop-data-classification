import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from shapely.geometry import box, mapping
from sklearn.ensemble import RandomForestClassifier

from crop_pipeline.features import experiments
from crop_pipeline.imagery import pixel_tables
from crop_pipeline.sources import test_cohort
from crop_pipeline.training import source_holdout
from crop_pipeline.windows import external


class SourceHoldoutTests(unittest.TestCase):
    def test_annual_handoff_names_resolve_to_source_roles(self):
        self.assertEqual(test_cohort.source_role("cropharvest_2019"), "cropharvest")
        self.assertEqual(test_cohort.source_role("sen4agrinet_2020"), "sen4agrinet")

    def test_overlapping_fragment_excludes_entire_test_parcel(self):
        def parcel(group, geometry):
            return {"properties": {"partition_group": group}, "geometry": mapping(geometry)}

        development = [parcel("dev", box(0, 0, 2, 2))]
        test = [parcel("overlap", box(1, 1, 3, 3)), parcel("overlap", box(4, 4, 5, 5)),
                parcel("touch", box(2, 0, 3, 1))]
        remaining, excluded = test_cohort.exclude_overlap(test, development)
        self.assertEqual(excluded, {"overlap"})
        self.assertEqual(remaining, [test[-1]])

    def test_matched_features_freeze_and_score_separate_sources(self):
        fixed, aligned = [], []
        columns = experiments.season_workflow.feature_columns()
        for partition, source in (("train", "sen4agrinet"), ("validation", "francecrops"),
                                  ("test", "cropharvest")):
            for index, crop in enumerate(source_holdout.CROPS):
                group = f"{partition}-{crop}"
                row = {
                    "field_id": group, "partition_group": group, "region_id": group,
                    "season_key": group, "crop": crop, "source_dataset": source,
                    "partition": partition, "label_status": "reported", "source_crs": "EPSG:32631",
                    "x": 100 * len(fixed) + 5, "y": 5,
                    **{column: .1 * (index + 1) for column in columns},
                }
                fixed.append({**row, "window_start": "2020-01-01", "window_end": "2020-05-01"})
                aligned.append({**row, "window_start": "2020-03-01", "window_end": "2020-07-01"})
        with TemporaryDirectory() as folder:
            path = Path(folder) / "windows.json"
            path.write_text(json.dumps({"fixed": fixed, "aligned": aligned, "quality": "fixture"}))
            external.export_features(path)
            development = path.parent / "features_development.json"
            test = path.parent / "features_test.json"
            with patch.object(source_holdout, "RandomForestClassifier",
                              side_effect=lambda **options: RandomForestClassifier(
                                  **{**options, "n_estimators": 3, "n_jobs": 1})):
                index = source_holdout.freeze(development)
            scores = source_holdout.score(test, index)
            self.assertEqual(set(scores), {"fixed_full", "aligned_full", "fixed_matched", "aligned_matched"})
            self.assertEqual(scores["fixed_matched"]["coverage"]["carrot"]["parcels"], 1)
            with self.assertRaisesRegex(ValueError, "Unexpected source"):
                source_holdout.score(development, index)

    def test_observation_losses_are_reported_without_changing_pixel_validity(self):
        plan = {
            "fragment_id": "tile", "fragment": {"properties": {
                "crop": "carrot", "partition_group": "parcel", "selected_pixels": 1,
            }},
            "settings": {"grid": {"crs": "EPSG:32631", "transform": [10, 0, 0, 0, -10, 10]}},
            "download_intervals": [{"name": "bin"}],
        }
        row = {"interval": "bin", "row": 0, "column": 0, "clear_count": 1,
               **dict.fromkeys(pixel_tables.quality.STUDY_BANDS, 1000)}
        legacy, _ = pixel_tables.frame_to_window_rows(pd.DataFrame([row]), plan, .8, "legacy")
        row.update(pre_mask_count=4, cloud_shadow_count=2, other_rejection_count=1)
        current, pixels = pixel_tables.frame_to_window_rows(pd.DataFrame([row]), plan, .8, "current")
        self.assertIsNone(legacy[0]["cloud_shadow_count"])
        self.assertEqual(current[0]["cloud_shadow_count"], 2)
        self.assertEqual(pixels[0]["valid"], 1)
