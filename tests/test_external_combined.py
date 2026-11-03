"""Test pooled CropHarvest/France harmonization, parcel-safe splits and scoring."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from crop_pipeline.features import experiments as feature_experiments
from crop_pipeline.training import external_combined


def source_tables(parcels_per_crop=6, pixels=4):
    generator = np.random.default_rng(0)
    tables = {}
    for table_number, table in enumerate(external_combined.ASSETS):
        records = []
        for crop_number, crop in enumerate(external_combined.CLASSES):
            for parcel in range(parcels_per_crop):
                for pixel in range(pixels):
                    record = {
                        "crop": crop,
                        "partition_group": f"{crop}{parcel}",
                        "asset_index": f"{crop}_{parcel}_{pixel}",
                        "lon": table_number * 10 + crop_number + parcel * 0.1 + pixel * 1e-3,
                        "lat": 45 + parcel * 0.1,
                    }
                    for column in external_combined.BASE_COLUMNS:
                        record[column] = 0.1 + 0.2 * crop_number + generator.normal(0, 0.02)
                    if table == "cropharvest":
                        record.update(country="France", year=2019)
                    else:
                        record.update(
                            source_dataset="sen4agrinet", source_year=2020, label_status="reported"
                        )
                    records.append(record)
        tables[table] = pd.DataFrame(records)
    return tables


def prepared(folder):
    data = external_combined.add_derived_features(external_combined.harmonize(source_tables()))
    data, _ = external_combined.link_split_units(data)
    data = external_combined.add_spatial_blocks(data)
    return external_combined.lock_test_split(data, Path(folder) / "split.csv")


class ExternalCombinedTest(unittest.TestCase):
    def test_harmonize_prefixes_groups_and_derives_all_designs(self):
        data = external_combined.add_derived_features(external_combined.harmonize(source_tables()))
        self.assertEqual(set(data["domain"]), {"cropharvest", "sen4agrinet"})
        self.assertTrue(data["partition_group"].str.contains(r"^(?:cropharvest|sen4agrinet)\|").all())
        self.assertEqual(data["partition_group"].nunique(), 36)
        columns = feature_experiments.feature_sets()["interval_and_shape"]
        self.assertTrue(np.isfinite(data[columns].to_numpy()).all())

    def test_shared_pixel_location_links_groups(self):
        data = external_combined.harmonize(source_tables())
        moved = data["partition_group"] == "sen4agrinet|carrot0"
        target = data[data["partition_group"] == "cropharvest|carrot0"].iloc[0]
        data.loc[moved, ["lon", "lat"]] = [target["lon"], target["lat"]]
        linked, audit = external_combined.link_split_units(data)
        units = linked.groupby("partition_group")["split_unit"].first()
        self.assertEqual(units["sen4agrinet|carrot0"], units["cropharvest|carrot0"])
        self.assertEqual(len(audit), 1)

    def test_conflicting_crop_within_group_is_rejected(self):
        data = external_combined.harmonize(source_tables())
        data.loc[0, "crop"] = "potato" if data.loc[0, "crop"] != "potato" else "carrot"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            external_combined.link_split_units(data)

    def test_test_split_keeps_units_whole_and_is_locked(self):
        with tempfile.TemporaryDirectory() as folder:
            data = prepared(folder)
            self.assertEqual(data.groupby("split_unit")["partition"].nunique().max(), 1)
            test = data[data["partition"] == "test"]
            self.assertEqual(test.groupby(["crop", "domain"])["split_unit"].nunique().min(), 1)
            again = external_combined.lock_test_split(
                data.drop(columns="partition"), Path(folder) / "split.csv"
            )
            self.assertTrue(again["partition"].equals(data["partition"]))
            changed = data.drop(columns="partition")
            changed = changed[changed["partition_group"] != "cropharvest|carrot0"]
            with self.assertRaisesRegex(ValueError, "locked test split"):
                external_combined.lock_test_split(changed, Path(folder) / "split.csv")

    def test_weights_balance_crops(self):
        data = external_combined.harmonize(source_tables())
        data = data[~((data["crop"] == "carrot") & (data["partition_group"].str.endswith("5")))]
        weights = external_combined.training_weights(data)
        totals = pd.Series(weights, index=data.index).groupby(data["crop"]).sum()
        self.assertTrue(np.allclose(totals, totals.mean()))
        self.assertAlmostEqual(weights.mean(), 1.0)

    def test_cross_validation_never_shares_units_and_scores_perfect_separation(self):
        with tempfile.TemporaryDirectory() as folder:
            development = prepared(folder)
            development = development[development["partition"] == "development"].reset_index(drop=True)
            small_forest = {**external_combined.FOREST_PARAMETERS, "n_estimators": 20, "n_jobs": 1}
            with patch.object(external_combined, "FOREST_PARAMETERS", small_forest):
                results = external_combined.cross_validate(
                    development, ["interval_only"], ["random_forest", "logistic_regression"], folds=3
                )
            self.assertEqual(len(results), 6)
            self.assertTrue((results["macro_f1"] > 0.9).all())
            summary = external_combined.summarize_cv(results)
            selection = external_combined.select_configuration(summary)
            self.assertEqual(selection["design"], "interval_only")

    def test_evaluate_scores_parcels_by_mean_probability(self):
        rows = pd.DataFrame(
            {"crop": ["carrot", "carrot", "cabbage"], "partition_group": ["a", "a", "b"]}
        )
        probabilities = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]])
        metrics = external_combined.evaluate(rows, probabilities)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["parcels"], 2)
        self.assertEqual(metrics["parcel_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
