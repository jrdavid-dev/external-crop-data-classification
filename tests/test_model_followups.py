import unittest
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from crop_pipeline.training import model_followups as followups


class MatchedModelFollowupTests(unittest.TestCase):
    def test_tuning_uses_frozen_columns_and_prefers_original_on_tie(self):
        model = RandomForestClassifier(n_estimators=3, random_state=42, max_samples=0.5)
        folds = [([], [])]
        columns = ["feature"]
        with patch.object(followups.comparison, "spatial_fold_rows", return_value=folds), patch.object(
            followups.comparison, "spatial_scores", return_value={"macro_f1": 0.5}
        ) as score:
            label, results = followups.select_settings(model, [], columns, folds)
        self.assertEqual(label, "original")
        self.assertEqual(set(results), set(followups.TUNING_CANDIDATES))
        for call, changes in zip(score.call_args_list, followups.TUNING_CANDIDATES.values(), strict=True):
            candidate, features, fit_folds = call.args
            self.assertEqual(candidate.get_params(), {**model.get_params(), **changes})
            self.assertIs(features, columns)
            self.assertIs(fit_folds, folds)

    def test_paired_difference_aligns_shuffled_ids_and_keeps_missing_macro_unavailable(self):
        original = [
            {"partition_group": str(i), "test_cohort": "jecam", "crop": crop, "prediction": "carrot"}
            for i, crop in enumerate(followups.comparison.CROPS)
        ]
        tuned = [{**row, "prediction": row["crop"]} for row in reversed(original)]
        result = followups.paired_macro_difference(original, tuned, 20, np.random.default_rng(42))
        for key in ("estimate", "ci_low", "ci_high"):
            self.assertAlmostEqual(result[key], 5 / 6)
        self.assertEqual(result["valid_replicates"], 20)
        missing = followups.paired_macro_difference(original[:1], tuned[-1:], 20, np.random.default_rng(42))
        self.assertIsNone(missing["estimate"])
        self.assertEqual(missing["valid_replicates"], 0)
