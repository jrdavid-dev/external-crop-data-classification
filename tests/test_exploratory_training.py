"""Test parcel isolation, split provenance and held-out evaluation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from crop_pipeline.training import exploratory
from crop_pipeline.windows import seasons as season_workflow


def example_rows():
    rows = []
    for crop_number, crop in enumerate(exploratory.CLASSES):
        for group_number in range(5):
            group = f"{crop}|{group_number}"
            rows.extend(pixel_rows(crop, crop_number, group, group_number))
    return rows


def pixel_rows(crop, crop_number, group, group_number):
    return [
        {
            "crop": crop,
            "partition_group": group,
            "field_id": "",
            "label_status": "reported",
            "source_crs": "EPSG:32631",
            "x": crop_number * 10000 + group_number * 100 + index * 10,
            "y": 100,
            "season_id": group,
            "window_start": "2020-04-01",
            "window_end": "2020-08-01",
            **{
                column: 0.2 + crop_number * 0.1
                for column in season_workflow.feature_columns()
            },
        }
        for index in range(2)
    ]


class ExploratoryTrainingTests(unittest.TestCase):
    def test_deterministic_group_split_never_divides_parcels(self):
        rows = example_rows()
        assignments = exploratory.parcel_partitions(rows)
        self.assertEqual(
            assignments, exploratory.parcel_partitions(list(reversed(rows)))
        )
        for crop in exploratory.CLASSES:
            selected = {
                assignments[row["partition_group"]]
                for row in rows
                if row["crop"] == crop
            }
            self.assertEqual(selected, {"train", "validation", "test"})
        prepared = [
            exploratory.prepare_identity(row, assignments[row["partition_group"]])
            for row in rows
        ]
        partitions = {
            name: [row for row in prepared if row["partition"] == name]
            for name in ("train", "validation", "test")
        }
        exploratory.assert_partition_separation(partitions)
        partitions["test"].append(partitions["train"][0])
        with self.assertRaisesRegex(ValueError, "overlap"):
            exploratory.assert_partition_separation(partitions)

    def test_expanded_data_preserves_all_previous_parcel_assignments(self):
        original = example_rows()
        previous = exploratory.parcel_partitions(original)
        expanded = list(original)
        for crop_number, crop in enumerate(exploratory.CLASSES):
            for group_number in range(5, 9):
                expanded.extend(
                    pixel_rows(
                        crop, crop_number, f"{crop}|{group_number}", group_number
                    )
                )
        current = exploratory.parcel_partitions(expanded, previous)
        self.assertEqual({group: current[group] for group in previous}, previous)
        self.assertEqual(
            current, exploratory.parcel_partitions(list(reversed(expanded)), previous)
        )
        self.assertEqual(len(current), 27)

    def test_reserved_rows_and_shared_location_aliases_are_rejected(self):
        rows = example_rows()
        with self.assertRaisesRegex(ValueError, "reserved"):
            exploratory.parcel_partitions([{**rows[0], "partition": "test"}] + rows[1:])
        rows[2]["x"] = rows[0]["x"]
        with self.assertRaisesRegex(ValueError, "aliases"):
            exploratory.parcel_partitions(rows)

    def test_bootstrap_weight_scale_preserves_parcel_ratios_and_sample_count(self):
        rows = [{"field_id": "small", "label_status": "reported"}] + [
            {"field_id": "large", "label_status": "reported"} for _ in range(4)
        ]
        weights = exploratory.normalized_training_weights(rows, {"reported": 0.5})
        self.assertAlmostEqual(weights.sum(), len(rows))
        self.assertAlmostEqual(weights[0], weights[1:].sum())

    def test_saved_inputs_are_locked_and_three_models_use_shared_test_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "features.csv"
            season_workflow.write_records(inputs, example_rows())
            output = root / "evaluation"
            rows, configuration, summary, manifest = (
                exploratory.prepare_existing_inputs([inputs], output)
            )
            self.assertEqual(len(rows), 30)
            self.assertEqual(sum(record["pixels"] for record in summary), 30)
            exploratory.prepare_existing_inputs([inputs], output)

            # Smaller trees keep this integration test fast; production configuration is checked separately.
            def fast_forest(**parameters):
                self.assertEqual(parameters["n_estimators"], 1000)
                self.assertEqual(parameters["random_state"], 42)
                return RandomForestClassifier(
                    **{**parameters, "n_estimators": 8, "n_jobs": 1}
                )

            with patch(
                "crop_pipeline.training.exploratory.RandomForestClassifier", side_effect=fast_forest
            ):
                results, selections = exploratory.run_experiments(
                    rows, configuration, output
                )
            self.assertEqual(len(results["test"]), 3)
            self.assertEqual(
                selections["external_parcel_holdout"]["status"],
                "selected_on_validation",
            )
            test_samples = []
            for design, metrics in results["test"].items():
                self.assertGreaterEqual(metrics["accuracy"], 0)
                self.assertLessEqual(metrics["accuracy"], 1)
                self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 6)
                self.assertTrue((output / design / "model.joblib").is_file())
                with (output / design / "test_predictions.csv").open() as stream:
                    test_samples.append(
                        [row["sample_id"] for row in csv.DictReader(stream)]
                    )
            self.assertEqual(test_samples[0], test_samples[1])
            self.assertEqual(test_samples[0], test_samples[2])
            inputs.write_text(inputs.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                exploratory.prepare_existing_inputs([inputs], output)
