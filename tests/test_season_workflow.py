"""Exercise evidence gates, calendar windows, missingness, and field leakage."""

import json
import importlib
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import season_workflow as workflow
import crop_workflow


def reviewed_fixture():
    configuration = {
        "frozen": True, "development_evidence": "Development field review report",
        "detector_parameters": {"relative_threshold": 0.2},
        "window_rule": "onset_month_plus_three", "missing_data_policy": "complete_only",
        "minimum_valid_fraction": 0.8, "random_seed": 42,
        "study_classes": ["carrot", "potato"],
        "confidence_weights": {"confirmed": 1.0, "reported": 0.5},
        "forest_parameters": {"n_estimators": 8, "n_jobs": 1},
    }
    candidate = {"region_id": "region", "status": "candidate",
                 "onset_interval_start": "2020-10-02", "onset_interval_end": "2020-10-08"}
    review = {"candidate_id": workflow.identify_candidates([candidate])[0]["candidate_id"],
              "decision": "accept", "season_id": "autumn", "crop": "carrot",
              "reviewer": "Reviewer", "review_notes": "One supported cycle",
              "label_status": "confirmed", "label_evidence": "Dated crop record",
              "timing_evidence": "Observed emergence", "cycle_evidence": "Same crop through window",
              "detector_fingerprint": workflow.record_fingerprint(configuration["detector_parameters"])}
    region = {"properties": {"fragment_id": "region", "crop": "carrot",
                             "partition_group": "group", "selected_pixels": 2}}
    return candidate, review, region, configuration


def feature_fixture():
    candidate, review, region, configuration = reviewed_fixture()
    seasons, _ = workflow.accept_seasons([candidate], [review], [region], configuration)
    season = seasons[0]
    quality_rows, pixel_rows = [], []
    for interval in workflow.window_intervals(season):
        quality_rows.append({**interval, "region_id": "region", "status": "quality_pass",
                             "valid_fraction": 1.0})
        pixel_rows.extend({"region_id": "region", "name": interval["name"],
                           "source_crs": "EPSG:32631", "x": coordinate, "y": 200,
                           "valid": 1, "clear_count": 2, "B3": 0.2, "B4": 0.1,
                           "B5": 0.3, "B8": 0.6, "B11": 0.4}
                          for coordinate in (100, 110))
    return season, quality_rows, pixel_rows


class ReviewTests(unittest.TestCase):
    def test_review_is_required_and_settings_must_be_frozen(self):
        candidate, review, region, configuration = reviewed_fixture()
        accepted, audit = workflow.accept_seasons([candidate], [], [region], configuration)
        self.assertFalse(accepted)
        self.assertEqual(audit[0]["exclusion_reason"], "review_pending")
        configuration["frozen"] = False
        with self.assertRaisesRegex(ValueError, "Freeze"):
            workflow.accept_seasons([candidate], [review], [region], configuration)

    def test_cross_year_window_and_independent_interval_validation(self):
        candidate, review, region, configuration = reviewed_fixture()
        review.update(reference_greenup_start="2020-10-10", reference_greenup_end="2020-10-12")
        seasons, _ = workflow.accept_seasons([candidate], [review], [region], configuration)
        self.assertEqual(seasons[0]["window_end"], "2021-02-01")
        self.assertEqual(seasons[0]["interval_distance_days"], 2)
        self.assertEqual(len(workflow.window_intervals(seasons[0])), 8)

    def test_month_boundary_requires_evidence(self):
        candidate, review, region, configuration = reviewed_fixture()
        candidate["onset_interval_start"] = "2020-09-29"
        review["candidate_id"] = workflow.identify_candidates([candidate])[0]["candidate_id"]
        accepted, _ = workflow.accept_seasons([candidate], [review], [region], configuration)
        self.assertFalse(accepted)
        review.update(resolved_onset_month="2020-10", month_resolution_evidence="Dated emergence visit")
        accepted, _ = workflow.accept_seasons([candidate], [review], [region], configuration)
        self.assertEqual(accepted[0]["window_start"], "2020-10-01")

    def test_stale_reviews_and_changed_settings_are_rejected(self):
        candidate, review, region, configuration = reviewed_fixture()
        configuration["detector_parameters"]["relative_threshold"] = 0.25
        accepted, audit = workflow.accept_seasons([candidate], [review], [region], configuration)
        self.assertFalse(accepted)
        self.assertIn("different_detector", audit[0]["exclusion_reason"])
        candidate["status"] = "ambiguous"
        with self.assertRaisesRegex(ValueError, "Stale"):
            workflow.accept_seasons([candidate], [review], [region], configuration)

    def test_rejected_detector_and_unknown_labels_cannot_be_accepted(self):
        candidate, review, region, configuration = reviewed_fixture()
        review["label_status"] = "unknown"
        self.assertFalse(workflow.accept_seasons([candidate], [review], [region], configuration)[0])
        candidate["status"] = "rejected"
        review["candidate_id"] = workflow.identify_candidates([candidate])[0]["candidate_id"]
        self.assertFalse(workflow.accept_seasons([candidate], [review], [region], configuration)[0])

    def test_templates_preserve_completed_review(self):
        candidate, _, region, configuration = reviewed_fixture()
        with tempfile.TemporaryDirectory() as folder:
            review_path = Path(folder) / "season_reviews.csv"
            review_path.write_text("human review", encoding="utf-8")
            workflow.prepare_review_files([candidate], [region], configuration["detector_parameters"], folder)
            self.assertEqual(review_path.read_text(), "human review")
            self.assertTrue((Path(folder) / "season_reviews.template.csv").is_file())


class FeatureTests(unittest.TestCase):
    def test_window_sensitivity_uses_consecutive_bins_across_year_boundary(self):
        season, _, _ = feature_fixture()
        season.update(window_start="2020-01-01", window_end="2020-05-01")
        earlier = workflow.shifted_window_intervals(season, -1)
        later = workflow.shifted_window_intervals(season, 1)
        self.assertEqual(earlier[0]["start"], "2019-12-16")
        self.assertEqual(earlier[-1]["end"], "2020-04-16")
        self.assertEqual(later[0]["start"], "2020-01-16")
        self.assertEqual(later[-1]["end"], "2020-05-16")
        self.assertTrue(all(first["end"] == second["start"]
                            for first, second in zip(earlier, earlier[1:])))

    def test_coverage_sensitivity_preserves_original_quality_flags(self):
        rows = [{"region_id": "region", "crop": "carrot", "valid_fraction": 0.8,
                 "NDVI": 0.5, "eligible": False}]
        decisions, summary = workflow.summarize_coverage_thresholds(rows, (0.7, 0.9))
        self.assertTrue(decisions[0]["coverage_experiment_accepted"])
        self.assertFalse(decisions[1]["coverage_experiment_accepted"])
        self.assertFalse(decisions[0]["eligible"])
        self.assertEqual(summary[1]["excluded"], 1)

    def test_features_use_scaled_bands_and_preserve_counts(self):
        season, quality_rows, pixel_rows = feature_fixture()
        accepted, exclusions, losses, bins = workflow.build_features([season], pixel_rows, quality_rows)
        self.assertEqual(len(accepted), 2)
        self.assertFalse(exclusions)
        self.assertEqual(losses[0]["excluded_pixels"], 0)
        self.assertEqual(len(bins), 8)
        self.assertAlmostEqual(accepted[0]["bin_01_NDVI"], 5 / 7)
        self.assertEqual(accepted[0]["bin_01_clear_count"], 2)
        self.assertEqual(len(workflow.feature_columns()), 64)
        self.assertEqual(len(workflow.feature_columns(False)), 40)

    def test_missing_pixels_bins_and_zero_denominators_are_excluded(self):
        season, quality_rows, pixel_rows = feature_fixture()
        pixel_rows[0]["B8"] = 0
        pixel_rows[0]["B4"] = 0
        accepted, exclusions, losses, bins = workflow.build_features([season], pixel_rows, quality_rows)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(exclusions), 1)
        self.assertEqual(bins[0]["usable_pixels"], 1)
        accepted, _, losses, bins = workflow.build_features([season], [], [])
        self.assertFalse(accepted)
        self.assertEqual(losses[0]["pixels_without_records"], 2)
        self.assertTrue(all(record["usable_pixels"] == 0 for record in bins))

    def test_low_coverage_and_absent_cross_year_bins_are_preserved(self):
        season, quality_rows, pixel_rows = feature_fixture()
        quality_rows[0]["valid_fraction"] = 0.7
        accepted, _, losses, _ = workflow.build_features([season], pixel_rows, quality_rows)
        self.assertFalse(accepted)
        self.assertEqual(losses[0]["excluded_pixels"], 2)
        accepted, exclusions, _, _ = workflow.build_features([season], pixel_rows, quality_rows[:-1])
        self.assertFalse(accepted)
        self.assertIn("2021_01_16_end", exclusions[0]["missing_bins"])

    def test_duplicate_pixel_records_fail(self):
        season, quality_rows, pixel_rows = feature_fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate pixel"):
            workflow.build_features([season], pixel_rows + [pixel_rows[0]], quality_rows)


def training_fixture():
    _, _, _, configuration = reviewed_fixture()
    rows = []
    for source, partition in (("external", "train"), ("local", "train"), ("local", "test")):
        for crop_index, crop in enumerate(configuration["study_classes"]):
            field_id = f"{source}_{partition}_{crop}"
            rows.append({"field_id": field_id, "partition_group": field_id,
                         "source": source, "partition": partition, "crop": crop,
                         "identity_evidence": "Reviewed independent geometry",
                         "season_id": "2020", "label_status": "confirmed",
                         "source_crs": "EPSG:32631", "x": len(rows) * 10, "y": 200,
                         **{column: crop_index + 0.1 for column in workflow.feature_columns()}})
    return rows, configuration


class PartitionAndModelTests(unittest.TestCase):
    def test_optional_checks_exclude_reserved_and_unassigned_fields(self):
        rows, _ = training_fixture()
        registry = [{column: row[column] for column in workflow.FIELD_COLUMNS} for row in rows]
        regions = [{"properties": {"fragment_id": row["field_id"],
                                    "partition_group": row["partition_group"]}} for row in rows]
        selected = workflow.development_region_ids(regions, registry)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all("test" not in identifier for identifier in selected))
        self.assertFalse(workflow.development_region_ids(regions, []))

    def test_field_fragments_cannot_cross_partitions(self):
        registry = [dict(partition_group="first", field_id="same", source="local",
                         partition="train", identity_evidence="Reviewed"),
                    dict(partition_group="second", field_id="same", source="local",
                         partition="test", identity_evidence="Reviewed")]
        with self.assertRaisesRegex(ValueError, "crosses"):
            workflow.validate_field_registry(registry)

    def test_duplicate_locations_cannot_claim_independent_fields(self):
        rows, _ = training_fixture()
        rows[-1]["x"] = rows[0]["x"]
        registry = [{column: row[column] for column in workflow.FIELD_COLUMNS} for row in rows]
        with self.assertRaisesRegex(ValueError, "duplicate location"):
            workflow.assign_partitions(rows, registry)

    def test_confidence_budget_does_not_grow_with_pixels_or_seasons(self):
        rows = [{"field_id": "same", "label_status": "reported"}] * 10
        self.assertAlmostEqual(sum(workflow.field_sample_weights(rows, {"reported": 0.5})), 0.5)

    def test_three_models_use_identical_test_rows_for_both_feature_sets(self):
        rows, configuration = training_fixture()
        with tempfile.TemporaryDirectory() as folder:
            for include_indices in (False, True):
                destination = Path(folder) / str(include_indices)
                metrics = workflow.train_comparisons(rows, configuration, destination, include_indices)
                self.assertEqual(set(metrics), {"external_only", "local_only", "combined"})
                self.assertTrue(all(result["test_fields"] == 2 for result in metrics.values()))
                predictions = [workflow.read_table(destination / f"{name}_predictions.csv") for name in metrics]
                self.assertTrue(all([row["field_id"] for row in table] ==
                                    [row["field_id"] for row in predictions[0]] for table in predictions))
                self.assertTrue((destination / "combined.joblib").is_file())

    def test_missing_class_and_nonconfirmed_test_labels_block_training(self):
        rows, configuration = training_fixture()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "configured crops"):
                workflow.train_comparisons(rows[1:], configuration, folder)
            rows[-1]["label_status"] = "reported"
            with self.assertRaisesRegex(ValueError, "confirmed"):
                workflow.train_comparisons(rows, configuration, folder)

    def test_packaging_hashes_files_and_reports_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            existing = Path(folder) / "features.csv"
            existing.write_text("features", encoding="utf-8")
            manifest = workflow.package_run(folder, {}, [existing], [Path(folder) / "absent"], {})
            self.assertEqual(manifest["inputs"][0]["sha256"], workflow.file_sha256(existing))
            self.assertEqual(len(manifest["missing_files"]), 1)
            json.loads((Path(folder) / "run_manifest.json").read_text())


class NotebookIntegrationTests(unittest.TestCase):
    def test_reviewed_seasons_reach_all_six_models_in_actual_notebook_cells(self):
        notebook_path = Path(__file__).resolve().parents[1] / "Sen4AgriNet_Workflow_Outline.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        executable_cells = [cell for cell in notebook["cells"][86:] if cell["cell_type"] == "code"]
        training_rows, configuration = training_fixture()
        candidates, reviews, regions, pixel_rows, quality_rows = [], [], [], [], []
        for training_row in training_rows:
            candidate, review, region, _ = reviewed_fixture()
            region_id = training_row["field_id"]
            candidate["region_id"] = region_id
            review.update(crop=training_row["crop"], candidate_id=workflow.identify_candidates([candidate])[0]["candidate_id"])
            region["properties"].update(fragment_id=region_id, crop=training_row["crop"],
                                        partition_group=region_id, selected_pixels=1)
            candidates.append(candidate)
            reviews.append(review)
            regions.append(region)
            _, example_quality, example_pixels = feature_fixture()
            quality_rows.extend({**row, "region_id": region_id} for row in example_quality)
            pixel_rows.extend({**row, "region_id": region_id, "x": training_row["x"]}
                              for row in example_pixels if row["x"] == 100)
        with tempfile.TemporaryDirectory() as folder:
            analysis_folder = Path(folder)
            review_folder = analysis_folder / "seasons/review"
            workflow.save_json(review_folder / "frozen_configuration.json", configuration)
            workflow.write_records(review_folder / "season_reviews.csv", reviews)
            workflow.write_records(review_folder / "field_registry.csv", training_rows)
            namespace = {"json": json, "Path": Path, "Counter": Counter, "importlib": importlib,
                         "crop_workflow": crop_workflow, "analysis_folder": analysis_folder,
                         "greenup_folder": analysis_folder / "greenup", "field_folder": analysis_folder / "fields",
                         "regions": regions, "greenup_results": candidates,
                         "detector_settings": configuration["detector_parameters"],
                         "pixel_rows": pixel_rows, "quality_rows": quality_rows, "detector_rows": []}
            for cell in executable_cells:
                exec("".join(cell["source"]), namespace)
            self.assertEqual(len(namespace["accepted_seasons"]), 6)
            self.assertEqual(len(namespace["feature_rows"]), 6)
            self.assertEqual(len(namespace["model_artifacts"]), 16)
            self.assertEqual(namespace["run_manifest"]["coverage"]["training_status"], "completed")


if __name__ == "__main__":
    unittest.main()
