"""Verify feature mathematics, handoff isolation and comparable experiment splits."""

import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from crop_pipeline.imagery import quality as crop_workflow
from crop_pipeline.training import earth_engine as training
from crop_pipeline.features import experiments as features
from crop_pipeline.windows import seasons as season_workflow
from crop_pipeline.imagery.retrieval import REQUEST_VERSION, RESAMPLING_POLICY, SatelliteRetriever


def experiment_fixture():
    classes = ["carrot", "cabbage", "potato"]
    configuration = {
        "frozen": True,
        "development_evidence": "Synthetic development fixture",
        "window_rule": "onset_month_plus_three",
        "missing_data_policy": "complete_only",
        "window_policy": "reference_supported_only",
        "minimum_valid_fraction": 0.8,
        "study_classes": classes,
        "confidence_weights": {"confirmed": 1.0, "reported": 0.5},
        "random_seed": 42,
        "detector_parameters": {},
    }
    rows = []
    for source, partition in [
        ("external", "train"),
        ("local", "train"),
        ("local", "validation"),
        ("local", "test"),
    ]:
        rows.extend(partition_fixture(source, partition, classes, len(rows)))
    return rows, configuration


def partition_fixture(source, partition, classes, offset):
    rows = []
    for crop_number, crop in enumerate(classes):
        rows.extend(crop_fixture(source, partition, crop, crop_number, offset + len(rows)))
    return rows


def crop_fixture(source, partition, crop, crop_number, offset):
    rows = []
    for pixel_number in range(2):
        field_id = f"{source}_{partition}_{crop}"
        row = {
            "source": source,
            "partition": partition,
            "crop": crop,
            "partition_group": field_id,
            "field_id": field_id,
            "identity_evidence": "Synthetic separate field",
            "source_crs": "EPSG:32631",
            "x": (offset + pixel_number) * 10,
            "y": 1000,
            "season_id": "2020",
            "season_key": field_id + "|2020",
            "window_start": "2020-02-01",
            "window_end": "2020-06-01",
            "label_status": "confirmed",
            "window_status": "reference_supported",
            "timing_review_status": "reviewed",
        }
        row.update(synthetic_interval_features(crop_number, pixel_number))
        rows.append(row)
    return rows


def synthetic_interval_features(crop_number, pixel_number):
    values = {}
    for bin_number in range(1, 9):
        peak = 3 + crop_number * 2
        near_infrared = (
            0.2
            + (0.1 + crop_number * 0.03) * max(0, 1 - abs(bin_number - peak) / 4)
            + pixel_number * 0.001
        )
        bands = {
            "B3": 0.15,
            "B4": 0.1,
            "B5": 0.12 + crop_number * 0.03,
            "B8": near_infrared,
            "B11": 0.2,
        }
        indices = {
            "NDVI": crop_workflow.ratio(near_infrared, bands["B4"]),
            "NDRE": crop_workflow.ratio(near_infrared, bands["B5"]),
            "NDMI": crop_workflow.ratio(near_infrared, bands["B11"]),
        }
        values.update(
            {f"bin_{bin_number:02d}_{name}": value for name, value in {**bands, **indices}.items()}
        )
    return values


def export_fixture(folder, frozen=True):
    rows, configuration = experiment_fixture()
    coverage = {"imagery_request_version": REQUEST_VERSION, "resampling": RESAMPLING_POLICY}
    source = season_workflow.export_training_handoff(
        Path(folder) / "windows", rows, configuration, coverage
    )
    manifest = features.export_feature_handoff(
        source,
        Path(folder) / "features",
        frozen=frozen,
        development_evidence="Synthetic frozen feature recipe" if frozen else "",
    )
    return manifest


class FeatureMathematicsTests(unittest.TestCase):
    def test_three_design_counts_and_savi_formula(self):
        self.assertEqual(
            {name: len(columns) for name, columns in features.feature_sets().items()},
            {"interval_only": 72, "interval_and_shape": 120, "shape_only": 48},
        )
        self.assertAlmostEqual(features.savi_value(0.6, 0.1, 0.5), 0.625)
        with self.assertRaisesRegex(ValueError, "denominator"):
            features.savi_value(-0.3, -0.2, 0.5)

    def test_temporal_features_have_known_slopes_differences_and_integral(self):
        positions = np.arange(8, dtype=float) * 10
        values = [0, 1, 2, 3, 2, 1, 0, 0]
        derived = features.temporal_shape(values, positions, "NDVI")
        self.assertEqual(derived["shape_NDVI_peak_interval"], 4)
        self.assertEqual(derived["shape_NDVI_amplitude"], 3)
        self.assertAlmostEqual(derived["shape_NDVI_greenup_slope"], 0.1)
        self.assertAlmostEqual(derived["shape_NDVI_senescence_slope"], -0.08)
        self.assertAlmostEqual(derived["shape_NDVI_integral"], 90)
        self.assertEqual(derived["shape_NDVI_difference_04_05"], -1)

    def test_actual_calendar_spacing_covers_leap_year_and_cross_year(self):
        positions = features.calendar_positions(
            {"window_start": "2020-02-01", "window_end": "2020-06-01"}
        )
        self.assertEqual(positions[0], 7.5)
        self.assertEqual(positions[1], 22)
        self.assertEqual(positions[2], 36.5)
        self.assertEqual(
            len(
                features.calendar_positions(
                    {"window_start": "2020-11-01", "window_end": "2021-03-01"}
                )
            ),
            8,
        )

    def test_flat_and_boundary_peaks_are_explicit_without_nan(self):
        derived = features.temporal_shape([0.5] * 8, np.arange(8), "SAVI")
        self.assertEqual(derived["shape_SAVI_peak_interval"], 1)
        self.assertEqual(derived["shape_SAVI_greenup_slope"], 0)
        self.assertFalse(derived["shape_SAVI_greenup_observed"])
        self.assertTrue(derived["shape_SAVI_senescence_observed"])
        self.assertEqual(derived["shape_SAVI_amplitude"], 0)

    def test_all_existing_labels_and_partitions_are_preserved(self):
        rows, _ = experiment_fixture()
        enhanced = features.enhance_rows(rows, features.FEATURE_POLICY)
        for original, result in zip(rows, enhanced):
            self.assertTrue(all(result[name] == value for name, value in original.items()))
            self.assertTrue(
                all(
                    np.isfinite(result[column])
                    for column in features.feature_sets()["interval_and_shape"]
                )
            )

    def test_duplicate_or_incomplete_windows_fail_instead_of_changing_one_design(self):
        rows, _ = experiment_fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            features.enhance_rows([rows[0], rows[0]], features.FEATURE_POLICY)
        rows[0]["bin_03_B8"] = ""
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            features.enhance_rows(rows, features.FEATURE_POLICY)


class FeatureHandoffTests(unittest.TestCase):
    def test_round_trip_and_move_preserve_rows_schema_and_window_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = export_fixture(temporary)
            destination = Path(temporary) / "moved"
            shutil.copytree(path.parent, destination)
            rows, configuration, _, manifest = features.load_feature_handoff(
                destination / path.name
            )
            self.assertEqual(len(rows), 24)
            self.assertEqual(manifest["status"], "prepared")
            self.assertEqual(configuration["window_policy"], "reference_supported_only")
            self.assertEqual(set(manifest["feature_sets"]), set(features.FEATURE_DESIGNS))

    def test_unfrozen_recipe_stays_pending_and_training_refuses_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = export_fixture(temporary, frozen=False)
            rows, configuration, _, manifest = features.load_feature_handoff(path)
            self.assertEqual(manifest["status"], "pending_inputs")
            with self.assertRaisesRegex(ValueError, "Freeze the feature recipe"):
                training.validate_experiment(rows, configuration)

    def test_tampered_features_and_wrong_imagery_contract_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = export_fixture(temporary)
            with (path.parent / "experiment_features.csv").open("a") as stream:
                stream.write("changed\n")
            with self.assertRaisesRegex(ValueError, "Feature input changed"):
                features.load_feature_handoff(path)
            rows, configuration = experiment_fixture()
            source = season_workflow.export_training_handoff(
                Path(temporary) / "old", rows, configuration, {}
            )
            with self.assertRaisesRegex(ValueError, "bilinear"):
                features.export_feature_handoff(source, Path(temporary) / "rejected")

    def test_feature_and_training_notebooks_run_with_a_fresh_kernel_handoff(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            feature_namespace = {}
            notebook = json.loads((root / "notebooks/04_feature_engineering.ipynb").read_text())
            cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
            exec("".join(cells[0]["source"]), feature_namespace)
            season_folder = Path(temporary)
            rows, configuration = experiment_fixture()
            for name in ("training_input", "exploratory_training_input"):
                season_workflow.export_training_handoff(
                    season_folder / name,
                    rows,
                    configuration,
                    {"imagery_request_version": REQUEST_VERSION, "resampling": RESAMPLING_POLICY},
                )
            feature_namespace["season_folder"] = season_folder
            for cell in cells[1:]:
                exec("".join(cell["source"]), feature_namespace)
            handoff = feature_namespace["feature_handoffs"]["feature_input"]
            training_namespace = {}
            training_notebook = json.loads((root / "notebooks/05_model_training.ipynb").read_text())
            training_cells = [
                cell for cell in training_notebook["cells"] if cell["cell_type"] == "code"
            ]
            exec("".join(training_cells[0]["source"]), training_namespace)
            training_namespace["training_mode"] = "production"
            training_namespace["handoff_path"] = handoff
            for cell in training_cells[1:]:
                exec("".join(cell["source"]), training_namespace)
            self.assertEqual(
                training_namespace["training_coverage"]["training_status"], "pending_inputs"
            )
            self.assertFalse(training_namespace["comparison_metrics"])


class ExperimentTrainingTests(unittest.TestCase):
    def prepared_rows(self):
        rows, configuration = experiment_fixture()
        configuration.update(
            feature_policy=dict(features.FEATURE_POLICY),
            feature_policy_frozen=True,
            feature_development_evidence="Synthetic development freeze",
        )
        return features.enhance_rows(rows, features.FEATURE_POLICY), configuration

    def test_gee_settings_are_fixed_and_bootstrap_is_shared_and_reproducible(self):
        rows, configuration = self.prepared_rows()
        settings = training.forest_settings(configuration, 120)
        self.assertEqual(settings["numberOfTrees"], 1000)
        self.assertEqual(settings["seed"], 42)
        self.assertEqual(settings["variablesPerSplit"], 10)
        first = training.confidence_bootstrap(rows[:6], configuration)
        second = training.confidence_bootstrap(list(reversed(rows[:6])), configuration)
        self.assertEqual(first, second)
        self.assertEqual(
            Counter(row["crop"] for row in first), Counter(row["crop"] for row in rows[:6])
        )

    def test_feature_choice_uses_validation_only_and_no_validation_stays_pending(self):
        results = {
            "external_only": {
                "interval_only": {"macro_f1": 0.8},
                "interval_and_shape": {"macro_f1": 0.9},
                "shape_only": {"macro_f1": 0.9},
            }
        }
        self.assertEqual(
            training.select_feature_designs(results)["external_only"]["selected_design"],
            "shape_only",
        )
        pending = training.select_feature_designs(
            {"external_only": {name: None for name in features.FEATURE_DESIGNS}}
        )
        self.assertEqual(pending["external_only"]["status"], "pending_validation")

    def test_nine_models_keep_shared_test_rows_and_same_draw_per_source(self):
        rows, configuration = self.prepared_rows()
        fake_classifier = SimpleNamespace(
            explain=lambda: SimpleNamespace(getInfo=lambda: {"trees": ["fitted tree"] * 1000})
        )

        def predict(earth_engine, classifier, evaluation_rows, columns, classes):
            return [row["crop"] for row in evaluation_rows]

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(training, "fit_forest", return_value=fake_classifier), patch.object(
                training, "predict_rows", side_effect=predict
            ):
                metrics, selections = training.train_feature_experiments(
                    rows, configuration, temporary, None
                )
            models = list(Path(temporary).glob("*/*/model.json"))
            self.assertEqual(len(models), 9)
            fingerprints = {
                result["test_sample_fingerprint"]
                for source in metrics.values()
                for result in source.values()
            }
            self.assertEqual(len(fingerprints), 1)
            for source_name in metrics:
                records = [
                    json.loads(path.read_text())
                    for path in Path(temporary, source_name).glob("*/model.json")
                ]
                self.assertEqual(
                    len({record["bootstrap_sample_fingerprint"] for record in records}), 1
                )
            self.assertTrue(
                all(record["selected_design"] == "shape_only" for record in selections.values())
            )

    def test_prediction_order_is_restored_and_missing_rows_are_rejected(self):
        payload = {
            "features": [
                {"properties": {"sample_index": 1, "prediction": 0}},
                {"properties": {"sample_index": 0, "prediction": 1}},
            ]
        }
        self.assertEqual(
            training.decode_predictions(payload, 2, ["carrot", "cabbage"]), ["cabbage", "carrot"]
        )
        with self.assertRaisesRegex(ValueError, "lost or duplicated"):
            training.decode_predictions(payload, 3, ["carrot", "cabbage"])

    def test_validation_fields_cannot_overlap_training(self):
        rows, configuration = self.prepared_rows()
        rows[12]["field_id"] = rows[0]["field_id"]
        rows[13]["field_id"] = rows[0]["field_id"]
        with self.assertRaisesRegex(ValueError, "field crosses"):
            training.validate_experiment(rows, configuration)


class ResamplingContractTests(unittest.TestCase):
    def test_only_continuous_coarse_bands_request_bilinear_before_compositing(self):
        earth_engine = MagicMock()
        image, cloud = MagicMock(), MagicMock()
        earth_engine.Image.side_effect = lambda value: value
        retriever = SatelliteRetriever(earth_engine, 2022, "images", "reports")
        retriever.mask_acquisition(image, cloud)
        masked = image.updateMask.return_value
        masked.select.assert_any_call(["B5", "B11"])
        masked.select.return_value.resample.assert_called_once_with("bilinear")
        masked.select.return_value.resample.return_value.reproject.assert_called_once()

    def test_legacy_or_mislabeled_requests_cannot_pass_the_new_quality_contract(self):
        settings = {
            "version": 3,
            "start": "2022-01-01",
            "end": "2023-01-01",
            "cloud_probability_threshold": 40,
        }
        self.assertFalse(crop_workflow.compatible_request(settings, 2022, 40, REQUEST_VERSION))
        settings.update(version=REQUEST_VERSION, resampling="nearest")
        self.assertFalse(crop_workflow.compatible_request(settings, 2022, 40, REQUEST_VERSION))
        settings["resampling"] = RESAMPLING_POLICY
        self.assertTrue(crop_workflow.compatible_request(settings, 2022, 40, REQUEST_VERSION))


if __name__ == "__main__":
    unittest.main()
