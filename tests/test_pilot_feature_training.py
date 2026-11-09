"""Explicit 04 → 05 → 06 bundles preserve samples, features and model selection."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crop_pipeline import analysis_workflow
from crop_pipeline.common import save_json
from crop_pipeline.features import experiments
from crop_pipeline.imagery.retrieval import REQUEST_VERSION, RESAMPLING_POLICY
from crop_pipeline.training import exploratory, earth_engine
from crop_pipeline.windows import seasons
from test_exploratory_training import example_rows


def window_bundle(root):
    rows = example_rows()
    assignments = exploratory.parcel_partitions(rows)
    for row in rows:
        row.update(window_status="satellite_estimated", timing_review_status="automated_pass",
                   automatic_timing_status="passed", automatic_policy_fingerprint="saved-policy")
    path = root / "windows.csv"
    seasons.write_records(path, rows)
    save_json(root / "split.json", {"kind": "exploratory_existing_source_parcel_evaluation", "group_assignments": assignments})
    save_json(root / "coverage.json", {"imagery_request_version": REQUEST_VERSION, "resampling": RESAMPLING_POLICY,
              "targets_met": False, "minimum_valid_fraction": 0.8, "window_recipe": {"minimum_valid_fraction": 0.8, "cycle_selection_policy": "earliest_complete"}})
    return analysis_workflow.export_window_handoff([path], root / "window_exports", root / "split.json", root / "coverage.json")


class PilotFeatureTrainingTests(unittest.TestCase):
    def test_features_are_enhanced_once_and_training_consumes_the_same_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            windows = window_bundle(root)
            with patch.object(experiments, "enhance_rows", wraps=experiments.enhance_rows) as enhance:
                feature_path = experiments.export_pilot_feature_handoff(windows, root / "features")
                self.assertEqual(enhance.call_count, 1)
                again = experiments.export_pilot_feature_handoff(windows, root / "features")
                self.assertEqual(again, feature_path)
                self.assertEqual(enhance.call_count, 1)
            expected, _, _, manifest = experiments.load_pilot_feature_handoff(feature_path)
            with patch.object(experiments, "enhance_rows", side_effect=AssertionError("Training must not recalculate features")):
                rows, configuration, counts, training_path = exploratory.prepare_feature_inputs(feature_path, root / "training")
                repeated = exploratory.prepare_feature_inputs(feature_path, root / "training")
            self.assertEqual(rows, expected)
            self.assertEqual(repeated[0], expected)
            self.assertFalse(configuration["frozen"])
            self.assertFalse(configuration["feature_policy_frozen"])
            self.assertEqual(configuration["forest_parameters"]["n_estimators"], 1000)
            self.assertEqual(len(counts), 9)
            self.assertEqual(manifest["feature_sets"], experiments.feature_sets())
            self.assertTrue(training_path.is_file())
            prior = json.loads((root / "split.json").read_text())["group_assignments"]
            self.assertEqual({r["partition_group"]: r["partition"] for r in rows}, prior)

    def test_corrupted_feature_and_training_files_are_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature_path = experiments.export_pilot_feature_handoff(window_bundle(root), root / "features")
            _, _, _, training = exploratory.prepare_feature_inputs(feature_path, root / "training")
            split = training.parent / "parcel_split.csv"
            split.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Saved training inputs changed"):
                exploratory.prepare_feature_inputs(feature_path, root / "training")
            self.assertEqual(split.read_text(), "changed")
            csv = feature_path.parent / "experiment_features.csv"
            csv.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Feature input changed"):
                experiments.load_pilot_feature_handoff(feature_path)
            with self.assertRaisesRegex(ValueError, "Feature input changed"):
                experiments.export_pilot_feature_handoff(window_bundle(root), root / "features")
            self.assertEqual(csv.read_text(), "changed")

    def test_changed_feature_policy_uses_a_new_version_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            windows = window_bundle(root)
            original = experiments.export_pilot_feature_handoff(windows, root / "features")
            before = {p: p.read_bytes() for p in original.parent.iterdir()}
            policy = {**experiments.FEATURE_POLICY, "savi_soil_adjustment": 0.25}
            changed = experiments.export_pilot_feature_handoff(windows, root / "features", policy)
            self.assertNotEqual(changed.parent, original.parent)
            self.assertEqual({p: p.read_bytes() for p in original.parent.iterdir()}, before)

    def test_selected_package_uses_validation_and_copies_model_without_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = experiments.export_pilot_feature_handoff(window_bundle(root), root / "features")
            _, _, _, training = exploratory.prepare_feature_inputs(features, root / "training")
            folder = training.parent
            validation = {"interval_only": {"macro_f1": 0.2}, "shape_only": {"macro_f1": 0.3}, "interval_and_shape": {"macro_f1": 0.1}}
            selections = earth_engine.select_feature_designs({"external_parcel_holdout": validation})
            save_json(folder / "feature_selection.json", selections)
            model = folder / "shape_only/model.joblib"
            model.parent.mkdir()
            model.write_bytes(b"already-fitted-model-fixture")
            save_json(folder / "experiment_metrics.json", {
                "validation": validation, "test": {"interval_only": {"macro_f1": 0.99}},
                "model_artifacts": {"shape_only": {"sha256": seasons.file_sha256(model),
                    "feature_columns": experiments.feature_sets()["shape_only"]}},
            })
            with patch.object(exploratory.joblib, "load", side_effect=AssertionError("Do not load models to package")):
                package = exploratory.export_selected_model(folder, features)
                again = exploratory.export_selected_model(folder, features)
            self.assertEqual(package, again)
            self.assertEqual((package.parent / "model.joblib").read_bytes(), model.read_bytes())
            recipe = json.loads((package.parent / "recipe.json").read_text())
            self.assertEqual(recipe["selected_design"], "shape_only")
            self.assertEqual(recipe["feature_columns"], experiments.feature_sets()["shape_only"])
            self.assertEqual(recipe["window_recipe"]["cycle_selection_policy"], "earliest_complete")
            self.assertEqual(recipe["imagery"]["request_version"], REQUEST_VERSION)
            self.assertEqual(recipe["imagery"]["resampling"], RESAMPLING_POLICY)
            self.assertEqual(recipe["imagery"]["spectral_scale"], 0.0001)
            self.assertEqual(recipe["imagery"]["collection_bands"], ["B3", "B4", "B5", "B8", "B11", "B2"])
            self.assertEqual(recipe["imagery"]["cloud_probability_rule"], "probability < threshold")
            self.assertEqual(recipe["imagery"]["output_resolution_metres"], 10)
            self.assertIn("prototype", recipe["status"])
            original_model = model.read_bytes()
            model.write_bytes(b"another-model-swapped-into-the-same-filename")
            with self.assertRaisesRegex(ValueError, "model evaluated in the saved metrics"):
                exploratory.export_selected_model(folder, features)
            self.assertEqual((package.parent / "model.joblib").read_bytes(), original_model)
            model.write_bytes(original_model)
            selections["external_parcel_holdout"]["selected_design"] = "interval_only"
            save_json(folder / "feature_selection.json", selections)
            with self.assertRaisesRegex(ValueError, "validation-only"):
                exploratory.export_selected_model(folder, features)


if __name__ == "__main__":
    unittest.main()
