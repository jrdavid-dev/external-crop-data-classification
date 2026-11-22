from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from crop_pipeline.training import source_holdout
from experiments.tuning import tuning


def development_rows(test_rows=False):
    random = np.random.default_rng(0)
    rows = []
    for partition in ("train", "validation"):
        for cell in range(10):
            for offset, crop in enumerate(source_holdout.CROPS):
                group = f"{partition}-{cell}-{crop}"
                for pixel in range(3):
                    row = {"partition_group": group, "crop": crop, "partition": partition,
                           "source_dataset": "sen4agrinet", "label_status": "confirmed",
                           "site_cell": f"45,{cell}"}
                    row.update({c: float(offset + random.normal(0, .3)) for c in tuning.columns()})
                    rows.append(row)
    if test_rows:
        rows.append({**rows[0], "partition": "test", "source_dataset": "cropharvest"})
    return {"rows": {cohort: rows for cohort in tuning.COHORTS}}


class TuningTests(unittest.TestCase):
    def test_grid_covers_space_and_rejects_unknown_parameters(self):
        grid = tuning.parameter_grid({"min_samples_leaf": [2, 4], "max_features": ["sqrt"]}, n_estimators=50)
        self.assertEqual([g["min_samples_leaf"] for g in grid], [2, 4])
        self.assertTrue(all(g["n_estimators"] == 50 and g["max_samples"] == .5 for g in grid))
        self.assertEqual(len(tuning.parameter_grid()), 128)
        with self.assertRaises(ValueError):
            tuning.parameter_grid({"criterion": ["gini"]})

    def test_test_partition_rows_are_rejected(self):
        with self.assertRaises(ValueError):
            tuning.cohort_split(development_rows(test_rows=True), "fixed_full")

    def test_validation_search_scores_each_setting_and_ranks(self):
        grid = tuning.parameter_grid({"min_samples_leaf": [1, 8]}, n_estimators=20)
        records = tuning.validation_search(development_rows(), grid, cohorts=["fixed_full"], progress=None)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(0 <= r["macro_f1"] <= 1 and "carrot_f1" in r for r in records))
        best = tuning.top_settings(records, "fixed_full", count=1)[0]
        self.assertEqual(best["min_samples_leaf"],
                         max(records, key=lambda r: r["macro_f1"])["min_samples_leaf"])

    def test_spatial_cross_validation_uses_training_parcels_only(self):
        data = development_rows()
        params = {**tuning.BASELINE, "n_estimators": 10}
        seen = []
        original = tuning.fit

        def recording_fit(rows, *args, **kwargs):
            seen.extend(row["partition"] for row in rows)
            return original(rows, *args, **kwargs)

        with patch.object(tuning, "fit", side_effect=recording_fit):
            pooled, folds = tuning.spatial_cross_validation(data, "fixed_full", params)
        self.assertEqual(set(seen), {"train"})
        self.assertEqual(len(folds), 5)
        self.assertEqual(sum(v["support"] for v in pooled["per_crop"].values()), 30)

    def test_runs_are_new_folders_under_tuning(self):
        with TemporaryDirectory() as tmp:
            features = Path(tmp) / "features_development.json"
            features.write_text("{}")
            first = tuning.new_run(tmp, "grid", features)
            second = tuning.new_run(tmp, "grid", features)
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, Path(tmp).resolve() / "pipeline_outputs/tuning")
            tuning.save_records(second, "results", [{"cohort": "fixed_full", "macro_f1": .5}])
            self.assertEqual(tuning.load_records(tuning.latest_run(tmp, "grid"), "results")[0]["macro_f1"], .5)
            self.assertFalse((Path(tmp) / "pipeline_outputs/training").exists())


if __name__ == "__main__":
    unittest.main()
