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
from crop_pipeline.training import notebook_workflow, source_holdout
from crop_pipeline.windows import external


class SourceHoldoutTests(unittest.TestCase):
    def test_supplement_excludes_pending_primary_footprint_and_has_separate_policy(self):
        from crop_pipeline.common import save_json
        from crop_pipeline.sources import adaptive

        def feature(group, dataset, geometry):
            return {"geometry": mapping(geometry), "properties": {
                "partition_group": group, "parcel_id": group, "country": "FR",
                "source_dataset": dataset, "crop": "carrot",
            }}

        primary = feature("primary", "jecam", box(0, 0, 2, 2))
        overlapping = feature("overlap", "ile-de-france", box(1, 1, 3, 3))
        independent = feature("independent", "ile-de-france", box(4, 4, 5, 5))
        with TemporaryDirectory() as tmp:
            report = Path(tmp) / "inventory"
            campaign = adaptive.root(report, test_cohort.TEST_POLICY)
            selection = campaign / "rounds/0001/primary/selection.json"
            save_json(selection, {"physical_keys": [adaptive.physical_key("cropharvest", primary["properties"])]})
            original = selection.read_bytes()
            with patch.object(test_cohort, "completed_sources", return_value=[]), \
                    patch.object(adaptive, "prepare_round_batches") as prepare:
                test_cohort.prepare_carrot_potato_supplement(
                    report, Path(tmp), [primary, overlapping, independent], {}, 50
                )
            self.assertEqual(prepare.call_args.args[1], [independent])
            self.assertEqual(prepare.call_args.kwargs["policy"], test_cohort.CARROT_POTATO_SUPPLEMENT_POLICY)
            self.assertNotEqual(adaptive.root(report, test_cohort.CARROT_POTATO_SUPPLEMENT_POLICY), campaign)
            self.assertEqual(original, selection.read_bytes())

    def test_fixed_window_uses_may_through_august_in_source_year(self):
        region = {"properties": {"source_year": 2022, "partition_group": "parcel", "fragment_id": "tile"}}
        window, = external.fixed_windows([region])
        self.assertEqual((window["window_start"], window["window_end"]), ("2022-05-01", "2022-09-01"))
        self.assertEqual(len(external.seasons.window_intervals(window)), 8)

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
                fixed.append({**row, "window_start": "2020-05-01", "window_end": "2020-09-01"})
                aligned.append({**row, "window_start": "2020-03-01", "window_end": "2020-07-01"})
        for crop, source, cohort in (
            ("cabbage", "denmark_marker", "denmark_marker_cabbage_2023_v1"),
            ("cabbage", "cropharvest", "non_jecam_cabbage_supplement_v1"),
            ("carrot", "california_dwr", "california_dwr_carrot_2023_v1"),
            ("potato", "cropharvest", "non_jecam_carrot_potato_supplement_v1"),
        ):
            for rows in (fixed, aligned):
                primary = next(r for r in rows if r["partition"] == "test" and r["crop"] == crop)
                rows.append({**primary, "field_id": cohort, "partition_group": cohort,
                             "region_id": cohort, "source_dataset": source, "test_cohort": cohort})
        with TemporaryDirectory() as folder:
            path = Path(folder) / "windows.json"
            path.write_text(json.dumps({"fixed": fixed, "aligned": aligned, "quality": "fixture", "fixed_start_month": 5}))
            external.export_features(path)
            development = path.parent / "features_development.json"
            test = path.parent / "features_test.json"
            with patch.object(source_holdout, "RandomForestClassifier",
                              side_effect=lambda **options: RandomForestClassifier(
                                  **{**options, "n_estimators": 3, "n_jobs": 1})):
                index = source_holdout.freeze(development)
            fixed_manifest = json.loads(index.read_text())["fixed_full"]["path"]
            recipe = json.loads(Path(fixed_manifest).with_name("recipe.json").read_text())
            self.assertEqual(recipe["window_recipe"], {"type": "fixed", "start_month": 5})
            scores = source_holdout.score(test, index)
            self.assertEqual(set(scores), {"fixed_full", "aligned_full", "fixed_matched", "aligned_matched"})
            self.assertEqual(scores["fixed_matched"]["coverage"]["carrot"]["parcels"], 1)
            with self.assertRaisesRegex(ValueError, "Unexpected source"):
                source_holdout.score(development, index)
            with patch.object(source_holdout, "RandomForestClassifier",
                              side_effect=lambda **options: RandomForestClassifier(
                                  **{**options, "n_estimators": 3, "n_jobs": 1})) as fit:
                trained = notebook_workflow.train(path.parent, development)
                calls = fit.call_count
                frozen = notebook_workflow.freeze(path.parent)
                self.assertEqual(fit.call_count, calls)
            scored = notebook_workflow.evaluate(path.parent, test)
            report = notebook_workflow.report(path.parent)
            reports = scored.parent / "reports"
            primary = json.loads((reports / "test_scores.json").read_text())
            combined = json.loads((reports / "combined/test_scores.json").read_text())
            denmark = json.loads((reports / "denmark_marker_cabbage_2023_v1/test_scores.json").read_text())
            self.assertEqual(primary, scores)
            self.assertEqual([combined["aligned_matched"]["coverage"][crop]["parcels"]
                              for crop in source_holdout.CROPS], [2, 3, 2])
            self.assertEqual(denmark["aligned_full"]["coverage"]["cabbage"]["parcels"], 1)
            self.assertNotIn("macro_f1", denmark["aligned_full"])
            self.assertIn("denmark_marker_cabbage_2023_v1", report.with_name("per_crop.csv").read_text())
            self.assertEqual(notebook_workflow.freeze(path.parent, trained), frozen)
            self.assertEqual(notebook_workflow.report(path.parent, scored), report)
            self.assertIn("Meets F1 criteria", report.with_name("results.md").read_text())
            self.assertTrue(report.with_name("per_crop.csv").is_file())

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
