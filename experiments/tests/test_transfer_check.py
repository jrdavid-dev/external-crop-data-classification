import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from crop_pipeline.paoay import aligned
from experiments.transfer import transfer, transfer_check
from experiments.tuning import tuning


def parcels(predictions):
    crops = ["carrot", "cabbage", "potato"]
    return [{"partition_group": f"p{i}", "crop": crops[i % 3], "prediction": p} for i, p in enumerate(predictions)]


class TransferCheckTests(unittest.TestCase):
    def test_paired_bootstrap_reports_signed_difference_and_requires_same_parcels(self):
        truth = ["carrot", "cabbage", "potato"] * 10
        worse = parcels(["potato"] * 30)
        better = parcels(truth)
        result = transfer_check.paired_bootstrap(worse, better, repeats=200)
        self.assertGreater(result["difference"], .7)
        self.assertGreater(result["ci_low"], 0)
        self.assertEqual(result["share_of_resamples_improved"], 1.0)
        with self.assertRaises(ValueError):
            transfer_check.paired_bootstrap(worse, better[:-1])

    def test_majority_baseline_and_settings(self):
        rows = [{"partition_group": g, "crop": c} for g, c in (("a", "potato"), ("b", "potato"), ("c", "carrot"))]
        self.assertEqual(transfer_check.majority_class(rows), "potato")
        self.assertAlmostEqual(transfer_check.baseline_macro_f1(rows, "potato"), (0 + 0 + .8) / 3)
        choice = {c: {"params": {**tuning.BASELINE, "n_estimators": 300}, "series_variant": "indices_32",
                      "augmentation_settings": {"shifts": [-1, 1]}} for c in transfer_check.COHORTS}
        settings = transfer_check.combined_settings(choice)
        self.assertEqual(settings["aligned_full"]["params"]["n_estimators"], 1000)
        self.assertEqual(settings["fixed_full"]["variant"], "indices_32")

    def test_test_rows_keep_primary_cohort_and_reject_development_overlap(self):
        rows = [{"partition_group": "t1", "crop": "carrot", "partition": "test", "source_dataset": "cropharvest"},
                {"partition_group": "t2", "crop": "carrot", "partition": "test", "source_dataset": "cropharvest",
                 "test_cohort": "denmark"}]
        with TemporaryDirectory() as folder:
            manifest = Path(folder) / "model_manifest.json"
            manifest.write_text(json.dumps({"development_groups": ["d1"]}))
            kept = transfer_check.test_rows({"rows": {"fixed_full": rows}}, "fixed_full", [manifest])
            self.assertEqual([r["partition_group"] for r in kept], ["t1"])
            manifest.write_text(json.dumps({"development_groups": ["t1"]}))
            with self.assertRaises(ValueError):
                transfer_check.test_rows({"rows": {"fixed_full": rows}}, "fixed_full", [manifest])

    def test_paoay_status_and_agreement(self):
        mask = np.ones((2, 2), dtype=bool)
        codes = np.full((2, 2), aligned.CODES["carrot"], dtype="uint8")
        codes[0, 0] = aligned.CODES["Unknown crop type"]
        status = transfer_check.paoay_status(codes, mask)
        self.assertAlmostEqual(status["carrot"], .75)
        self.assertAlmostEqual(sum(status.values()), 1)
        first = {(0, 0): {"label": "carrot"}, (0, 1): {"label": "potato"}}
        second = {(0, 0): {"label": "carrot"}, (0, 1): {"label": "cabbage"}, (1, 1): {"label": "carrot"}}
        agreement = transfer_check.paoay_agreement(first, second)
        self.assertEqual(agreement["pixels"], 2)
        self.assertAlmostEqual(agreement["same_label_share"], .5)

    def test_plain_forest_scores_through_transfer_helpers(self):
        from experiments.tests.test_transfer import FAST, rows
        data = rows()
        train = [r for r in data if r["partition"] == "train"]
        model = transfer.forest(FAST).fit(transfer.matrix(train, "interval_72"), [r["crop"] for r in train])
        metrics, _ = transfer_check.score_rows(model, [r for r in data if r["partition"] == "validation"], "interval_72")
        self.assertGreater(metrics["macro_f1"], .8)


if __name__ == "__main__":
    unittest.main()
