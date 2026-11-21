import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from crop_pipeline.training import transfer_followups as followups


class TransferFollowupTests(unittest.TestCase):
    def test_ensemble_aligns_by_id_and_averages_probabilities(self):
        fixed = [
            {"partition_group": key, "crop": "carrot", "test_cohort": "jecam",
             "usable_pixels": 2, "prob_carrot": 0.6, "prob_cabbage": 0.0, "prob_potato": 0.4}
            for key in ("a", "b")
        ]
        aligned = [
            {**fixed[1], "prob_carrot": 0.1, "prob_potato": 0.9},
            {**fixed[0], "prob_carrot": 0.8, "prob_potato": 0.2},
        ]
        result = followups.average_windows(fixed, aligned)
        self.assertEqual([r["prediction"] for r in result], ["carrot", "potato"])
        self.assertAlmostEqual(result[1]["prob_potato"], 0.65)

    def test_medians_keep_parcel_split_and_take_featurewise_medians(self):
        rows = [
            {"partition_group": "a", "partition": "train", "crop": "potato", "band": value}
            for value in (1, 5, 11)
        ]
        self.assertEqual(followups.parcel_medians(rows, ["band"])[0]["band"], 5)
        with self.assertRaisesRegex(ValueError, "Conflicting labels or splits"):
            followups.parcel_medians([rows[0], dict(rows[1], partition="validation")], ["band"])

    def test_cycle_switch_requires_every_original_pixel_and_strict_gain(self):
        columns = [f"bin_{i:02d}_NDVI" for i in range(1, 9)]

        def make_row(group, pixel, new=False, amplitude=1):
            return {"partition_group": group, "source_crs": "EPSG:32610", "x": pixel, "y": 1,
                    "region_id": group, "crop": "carrot",
                    "window_start": "2019-09-01" if new else "2019-05-01",
                    "window_end": "2020-01-01" if new else "2019-09-01",
                    **{column: value for column, value in zip(columns, [0] * 7 + [amplitude])}}

        original = [make_row(group, pixel) for group in ("complete", "partial", "tie") for pixel in (1, 2)]
        candidates = [make_row("complete", p, True, 3) for p in (1, 2)] + [
            make_row("partial", 1, True, 5), *(make_row("tie", p, True, 1) for p in (1, 2))]
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "archive.csv"
            with archive.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
                writer.writeheader()
                writer.writerows(candidates)
            with patch.object(followups, "feature_columns", return_value=columns), patch.object(
                followups, "enhance_window", side_effect=lambda row, policy: row
            ):
                updated, changes, counts = followups.strongest_windows({"train": original}, archive, {})
        self.assertEqual([change["partition_group"] for change in changes], ["complete"])
        self.assertEqual(counts["train"]["parcels_with_complete_alternative"], 2)
        self.assertEqual([row["window_start"] for row in updated["train"]],
                         ["2019-09-01"] * 2 + ["2019-05-01"] * 4)
