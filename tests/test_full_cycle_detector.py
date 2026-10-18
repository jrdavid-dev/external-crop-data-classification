"""Exercise complete vegetation episodes, boundary uncertainty and fluctuation merging."""

import unittest
from datetime import date, timedelta

from crop_pipeline.imagery import quality as crop_workflow
from crop_pipeline.windows import cycle_windows
from crop_pipeline.windows import detector as full_cycle_detector

FULL_PULSE = [
    0.1,
    0.1,
    0.1,
    0.15,
    0.3,
    0.5,
    0.7,
    0.7,
    0.65,
    0.5,
    0.3,
    0.15,
    0.1,
    0.1,
    0.1,
    0.1,
]


def observations(values):
    return [
        {
            "region_id": "field",
            "partition_group": "parcel",
            "date": (date(2020, 4, 1) + timedelta(days=5 * index)).isoformat(),
            "source_kind": "observed_daily",
            "NDVI": value,
            "NDRE": value * 0.5,
            "NDMI": value * 0.2,
            "valid_fraction": 1.0,
        }
        for index, value in enumerate(values)
    ]


def detect(rows):
    return crop_workflow.detect_greenup(
        rows, {"cycle_detector": full_cycle_detector.METHOD}
    )


class FullCycleDetectorTests(unittest.TestCase):
    def test_complete_low_peak_low_episode_has_supported_boundaries(self):
        results = detect(observations(FULL_PULSE))
        complete = [
            row for row in results if row.get("cycle_status") == "complete_observed"
        ]
        self.assertEqual(len(complete), 1)
        self.assertLess(complete[0]["onset_interval_end"], complete[0]["peak_date"])
        self.assertLess(complete[0]["peak_date"], complete[0]["end_interval_start"])
        self.assertLess(
            complete[0]["cycle_length_min_days"], complete[0]["cycle_length_max_days"]
        )

    def test_minor_internal_peak_does_not_restart_the_growth_baseline(self):
        values = [
            0.1,
            0.1,
            0.1,
            0.15,
            0.3,
            0.5,
            0.45,
            0.5,
            0.7,
            0.8,
            0.6,
            0.4,
            0.15,
            0.1,
            0.1,
            0.1,
        ]
        complete = [
            row
            for row in detect(observations(values))
            if row.get("cycle_status") == "complete_observed"
        ]
        self.assertEqual(len(complete), 1)
        self.assertLessEqual(complete[0]["onset_interval_end"], "2020-04-26")

    def test_already_green_baseline_is_not_certified_as_a_new_crop_cycle(self):
        rows = observations(
            [0.5, 0.51, 0.55, 0.65, 0.75, 0.72, 0.6, 0.5, 0.49, 0.3, 0.1, 0.1, 0.1]
        )
        results = detect(rows)
        self.assertFalse(
            any(row.get("cycle_status") == "complete_observed" for row in results)
        )
        self.assertTrue(
            any(
                row["reason"] == "possible_partial_cycle_high_background"
                for row in results
            )
        )

    def test_missing_boundary_is_reported_not_fabricated(self):
        start_missing = detect(observations([0.8, 0.8, 0.7, 0.5, 0.3, 0.1, 0.1, 0.1]))
        self.assertTrue(
            any(row["status"] == "incomplete_start" for row in start_missing)
        )
        end_missing = detect(
            observations([0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 0.8, 0.8])
        )
        self.assertTrue(any(row["status"] == "incomplete_end" for row in end_missing))
        self.assertFalse(
            any(row.get("cycle_status") == "complete_observed" for row in end_missing)
        )

    def test_large_observation_gap_prevents_complete_cycle_certification(self):
        rows = observations(FULL_PULSE)
        for row in rows[7:]:
            row["date"] = (
                date.fromisoformat(row["date"]) + timedelta(days=30)
            ).isoformat()
        results = detect(rows)
        self.assertFalse(
            any(row.get("cycle_status") == "complete_observed" for row in results)
        )
        self.assertTrue(
            any(
                row["reason"] == "growth_cycle_obscured_by_observation_gap"
                for row in results
            )
        )

    def test_distinct_supported_cycles_are_retained_and_context_is_flagged(self):
        results = detect(observations(FULL_PULSE + FULL_PULSE))
        complete = [
            row for row in results if row.get("cycle_status") == "complete_observed"
        ]
        self.assertEqual(len(complete), 2)
        self.assertTrue(all(row["status"] == "ambiguous" for row in complete))
        window = {
            **complete[0],
            "window_start": "2020-04-01",
            "window_end": "2020-08-01",
        }
        flagged = cycle_windows.flag_four_month_context([window], results)[0]
        self.assertTrue(flagged["window_overlaps_next_cycle"])
        self.assertTrue(flagged["window_extends_past_observed_cycle"])

    def test_two_prominent_bumps_without_low_state_are_merged(self):
        dates = [date(2020, 1, 1) + timedelta(days=5 * index) for index in range(13)]
        import numpy as np

        values = np.array(
            [0.1, 0.1, 0.3, 0.6, 0.8, 0.5, 0.5, 0.6, 0.85, 0.6, 0.3, 0.1, 0.1]
        )
        episodes = full_cycle_detector.identify_episodes(
            dates, values, values, crop_workflow.DEFAULT_DETECTOR
        )
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["merged_peaks"], 2)

    def test_legacy_algorithm_remains_explicitly_replayable(self):
        rows = observations(FULL_PULSE)
        self.assertEqual(
            crop_workflow.detect_greenup(rows),
            crop_workflow.detect_greenup(
                rows, {"cycle_detector": "legacy_local_peaks"}
            ),
        )
        self.assertEqual(crop_workflow.DEFAULT_DETECTOR["minimum_amplitude"], 0.15)
        self.assertEqual(crop_workflow.DEFAULT_DETECTOR["relative_threshold"], 0.2)
