import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from crop_pipeline.training import model_comparisons as comparison


class Batch2ComparisonTests(unittest.TestCase):
    def test_bin_selection_uses_fit_folds_and_earliest_tie(self):
        rows = [{"partition_group": f"{crop}-{site}", "crop": crop}
                for crop in comparison.CROPS for site in ("a", "b")]
        a = {row["partition_group"] for row in rows if row["partition_group"].endswith("-a")}
        b = {row["partition_group"] for row in rows if row["partition_group"].endswith("-b")}
        with patch.object(comparison, "fit", return_value=object()) as fit, patch.object(
            comparison.source_holdout, "parcel_metrics",
            side_effect=lambda model, held, columns: ({}, [{**row, "prediction": row["crop"]} for row in held]),
        ) as score:
            selected, results = comparison.select_bin(object(), rows, {7: ["late"], 2: ["early"]},
                                                      [(a, b), (b, a)])
        self.assertEqual(selected, 2)
        self.assertEqual(results, {2: 1.0, 7: 1.0})
        self.assertEqual(fit.call_count, 4)
        for fit_call, score_call in zip(fit.call_args_list, score.call_args_list):
            fitted = {row["partition_group"] for row in fit_call.args[1]}
            held = {row["partition_group"] for row in score_call.args[1]}
            self.assertFalse(fitted & held)
            self.assertEqual(fitted | held, a | b)

    def test_parcel_guard_uses_labels_and_campaigns_and_rejects_duplicates(self):
        rows = [{"partition_group": "p", "crop": "potato", "test_cohort": "jecam", "prediction": "carrot"}]
        comparison.require_shared_parcels(rows, [{**rows[0], "prediction": "potato"}])
        for observed in ([], rows * 2, [{**rows[0], "crop": "cabbage"}], [{**rows[0], "test_cohort": "other"}]):
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                comparison.require_shared_parcels(rows, observed)

    def test_external_scope_filters_by_reference_crop(self):
        mixed = "non_jecam_carrot_potato_supplement_v1"
        rows = [
            {"test_cohort": mixed, "crop": "carrot", "prediction": "potato"},
            {"test_cohort": mixed, "crop": "potato", "prediction": "carrot"},
            {"test_cohort": "jecam", "crop": "carrot", "prediction": "carrot"},
        ]
        self.assertEqual(comparison.scoped_predictions(rows, mixed), rows[1:2])
        self.assertEqual(comparison.scoped_predictions(rows, "combined"), rows[1:])

    def test_shape_fit_rejects_different_development_snapshot(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_stage = root / "models_stage.json"
            model_stage.write_text("original stage")
            (root / "features_development.json").write_text("different features")
            receipt = root / "comparators.json"
            receipt.write_text(json.dumps({
                "training_models_stage_sha256": comparison.file_sha256(model_stage),
                "development": {"feature_sha256": "expected hash"},
            }))
            with patch.object(comparison.notebook_workflow, "_verify", return_value=(model_stage, {})):
                with self.assertRaisesRegex(ValueError, "same frozen development inputs"):
                    comparison.freeze_shape_comparison(root, receipt)
            self.assertFalse((root / "shape_comparison").exists())
