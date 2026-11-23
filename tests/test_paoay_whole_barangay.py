from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from unittest.mock import patch

from crop_pipeline.paoay import whole_barangay


class WholeBarangayTests(unittest.TestCase):
    def test_only_repeated_crop_majority_is_farmland(self):
        guidance = np.array([
            [3, 3, 4, 3, 3],
            [3, 3, 4, 3, 3],
            [3, 4, 4, 3, 3],
            [4, 4, 3, 2, 3],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
        ]).reshape(8, 1, 5)
        roi = np.array([[True, True, True, True, False]])
        np.testing.assert_array_equal(whole_barangay.farmland_mask(guidance, roi),
                                      [[1, 3, 2, 3, 0]])

    def test_regional_mask_is_the_only_gate_when_selected(self):
        roi = np.array([[True, True, True, False]])
        regional = np.array([[1, 0, 0, 1]], dtype=np.uint8)
        dynamic = np.array([[2, 1, 2, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(whole_barangay.regional_cover(regional, dynamic, roi),
                                      [[1, 3, 2, 0]])

    def test_audit_covers_only_completed_candidate_windows(self):
        self.assertEqual(whole_barangay.date_candidates(date(2026, 8, 16)), [
            "2024-01-01", "2024-05-01", "2025-01-01", "2025-05-01", "2026-01-01",
        ])

    def test_training_period_comes_from_the_verified_model(self):
        with TemporaryDirectory() as folder:
            model = Path(folder) / "model_manifest.json"
            model.write_text(json.dumps({
                "identity": {"recipe": {"window_recipe": {"type": "fixed", "start_month": 11}}}
            }))
            handoff = {"identity": {"model": whole_barangay.references.checksum(model)},
                       "model_manifest": str(model)}
            self.assertEqual(whole_barangay.fixed_training_period(handoff), "November–February")
            model.write_text("{}")
            self.assertEqual(whole_barangay.fixed_training_period(handoff), "")

    def test_shift_needs_opt_in_and_rechecks_other_imagery_settings(self):
        image = {"manifest": {"period": {"start": "2024-01-01"}}}
        recipe = {"window_recipe": {"type": "fixed", "start_month": 5}}
        with patch.object(whole_barangay.features, "validate_imagery", side_effect=[
            ValueError("calendar mismatch"), None,
        ]) as validate:
            self.assertEqual(whole_barangay._imagery_contract(recipe, image, True),
                             "exploratory_shifted_calendar")
            self.assertEqual(validate.call_args.args[0]["window_recipe"]["start_month"], 1)
        with patch.object(whole_barangay.features, "validate_imagery", side_effect=ValueError):
            with self.assertRaises(ValueError):
                whole_barangay._imagery_contract(recipe, image, False)


if __name__ == "__main__":
    unittest.main()
