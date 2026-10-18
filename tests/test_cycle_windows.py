"""Separate genuine cycles, retain numerical checks, and prefer earliest complete data."""

import unittest
from datetime import date, timedelta

from crop_pipeline.imagery import quality as crop_workflow
from crop_pipeline.windows import cycle_windows
from crop_pipeline.windows import seasons as season_workflow
from crop_pipeline.windows import timing_quality


def windows_fixture():
    return [
        dict(
            season_key=name,
            season_id=name,
            region_id="region",
            partition_group="parcel",
            crop="carrot",
            selected_pixels=2,
            window_start=start,
            window_end=end,
            timing_review_status="automated_pass",
            window_status="satellite_estimated",
        )
        for name, start, end in [
            ("early", "2020-04-01", "2020-08-01"),
            ("late", "2020-09-01", "2021-01-01"),
        ]
    ]


def feature_fixture(windows):
    quality, pixels = [], []
    for window in windows:
        for interval in season_workflow.window_intervals(window):
            quality.append(
                {
                    **interval,
                    "region_id": "region",
                    "status": "quality_pass",
                    "valid_fraction": 1.0,
                }
            )
            pixels.extend(
                {
                    "region_id": "region",
                    "name": interval["name"],
                    "source_crs": "EPSG:32631",
                    "x": coordinate,
                    "y": 200,
                    "valid": 1,
                    "clear_count": 2,
                    "B3": 0.2,
                    "B4": 0.1,
                    "B5": 0.3,
                    "B8": 0.6,
                    "B11": 0.4,
                }
                for coordinate in (100, 110)
            )
    return quality, pixels


class CycleSelectionTests(unittest.TestCase):
    def test_earliest_complete_wins_and_later_pixels_do_not_fill_gaps(self):
        windows = windows_fixture()
        quality, pixels = feature_fixture(windows)
        pixels[0]["valid"] = 0
        complete, excluded, losses, bins = season_workflow.build_features(
            windows, pixels, quality
        )
        selected, features, _, _, _, audit = cycle_windows.choose_complete_cycles(
            windows, complete, excluded, losses, bins
        )
        self.assertEqual([row["season_key"] for row in selected], ["early"])
        self.assertEqual(len(features), 1)
        self.assertEqual(audit[1]["cycle_selection_status"], "excluded_later")
        self.assertEqual(audit[1]["complete_usable_pixels"], 2)

    def test_bad_interval_uses_later_cycle_without_relaxing_coverage(self):
        windows = windows_fixture()
        quality, pixels = feature_fixture(windows)
        quality[0]["valid_fraction"] = 0.79
        complete, excluded, losses, bins = season_workflow.build_features(
            windows, pixels, quality
        )
        selected, features, _, _, _, audit = cycle_windows.choose_complete_cycles(
            windows, complete, excluded, losses, bins
        )
        self.assertEqual([row["season_key"] for row in selected], ["late"])
        self.assertEqual(len(features), 2)
        self.assertEqual(audit[0]["cycle_selection_status"], "excluded_incomplete")
        selected, _, _, _, _, audit = cycle_windows.choose_complete_cycles(
            windows, [], excluded, losses, bins
        )
        self.assertFalse(selected)
        self.assertTrue(
            all(row["cycle_selection_status"] == "excluded_incomplete" for row in audit)
        )

    def test_fragments_of_one_parcel_share_one_date_range(self):
        early, late = windows_fixture()
        fragment = dict(late, region_id="other_fragment", season_key="late_fragment")
        features = [dict(early, x=100), dict(fragment, x=200)]
        selected, features, _, _, _, _ = cycle_windows.choose_complete_cycles(
            [early, fragment], features, [], [], []
        )
        self.assertEqual([row["season_key"] for row in selected], ["early"])
        self.assertEqual(len(features), 1)

    def test_two_observed_cycles_pass_checks_separately(self):
        values = [0.1, 0.1, 0.1, 0.15, 0.3, 0.5, 0.7, 0.7, 0.6, 0.4, 0.2]
        observations = [
            {
                "region_id": "region",
                "date": (date(2020, month, 1) + timedelta(days=5 * index)).isoformat(),
                "source_kind": "observed_daily",
                "valid_fraction": 1.0,
                "NDVI": value,
                "NDRE": value * 0.5,
                "NDMI": value * 0.2,
            }
            for month in (4, 9)
            for index, value in enumerate(values)
        ]
        candidates = crop_workflow.detect_greenup(observations)
        self.assertEqual(
            [row["status"] for row in candidates], ["ambiguous", "ambiguous"]
        )
        windows = [
            dict(
                candidate,
                candidate_id=str(index),
                window_start=candidate["onset_interval_start"][:7] + "-01",
            )
            for index, candidate in enumerate(candidates)
        ]
        assessments = timing_quality.assess_timing(
            windows, observations, {}, cycle_selection_policy="earliest_complete"
        )
        self.assertTrue(
            all(row["automatic_timing_status"] == "passed" for row in assessments)
        )
        self.assertTrue(
            all(
                row["automatic_timing_status"] == "failed"
                for row in timing_quality.assess_timing(windows, observations, {})
            )
        )
        policy = timing_quality.DEFAULT_TIMING_POLICY
        # Another cycle cannot stand in for a missing match at a sensitivity threshold.
        problem = timing_quality.threshold_problem(
            windows[0], [candidates[1]], policy, "earliest_complete"
        )
        self.assertEqual(problem, "no_unique_supported_cycle")
        # Two possible matches near the same peak remain ambiguous.
        self.assertEqual(
            timing_quality.threshold_problem(
                windows[0], [candidates[0], candidates[0]], policy, "earliest_complete"
            ),
            "no_unique_supported_cycle",
        )
