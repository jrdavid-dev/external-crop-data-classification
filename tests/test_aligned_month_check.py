from datetime import date, timedelta
import unittest
from unittest.mock import patch

from crop_pipeline.aligned_month_check import (
    alternate_windows, boundary_windows, complete_locations, counts,
    disagreement_month_pairs, longer_gap_windows, merge_locations, timing_rejection,
)
from crop_pipeline.imagery.quality import detect_greenup
from crop_pipeline.windows import seasons, timing_quality
from test_season_workflow import feature_fixture
from test_timing_quality import timing_fixture


class AlignedMonthCheckTests(unittest.TestCase):
    def test_supported_boundary_tries_both_starts_but_keeps_growth_checks(self):
        _, observations = timing_fixture()
        shifted = [
            {**row, "date": (date.fromisoformat(row["date"]) - timedelta(days=17)).isoformat()}
            for row in observations
        ]
        candidates = seasons.identify_candidates(detect_greenup(shifted))
        regions = [{"properties": {"fragment_id": "region", "crop": "carrot"}}]
        strict = {**candidates[0], "window_start": "2020-09-01"}
        assessment = timing_quality.assess_timing([strict], shifted, {})[0]
        self.assertIn("onset_month_uncertain", assessment["automatic_timing_reasons"])
        options = list(boundary_windows(candidates, regions, shifted, {}, None))
        self.assertEqual([row["window_start"] for row in options], ["2020-09-01", "2020-10-01"])
        self.assertEqual([row["window_end"] for row in options], ["2021-01-01", "2021-02-01"])
        self.assertNotEqual(options[0]["season_key"], options[1]["season_key"])
        flat = [{**row, "NDRE": 0.1} for row in shifted]
        self.assertEqual(list(boundary_windows(candidates, regions, flat, {}, None)), [])

    def test_threshold_disagreement_allows_only_two_adjacent_months_for_the_same_cycle(self):
        candidate, observations = timing_fixture()
        regions = [{"properties": {"fragment_id": "region", "crop": "carrot"}}]
        september = {**candidate, "onset_interval_start": "2020-09-28",
                     "onset_interval_end": "2020-10-02"}

        def selected_starts(other):
            sensitivity = {0.15: {"region": [other]}, 0.20: {"region": [candidate]},
                           0.25: {"region": [candidate]}}
            with patch.object(timing_quality, "run_threshold_checks", return_value=sensitivity):
                strict = timing_quality.assess_timing(
                    [candidate], observations, {}, None, "earliest_complete"
                )[0]
                if timing_rejection({"exclusion_reason": strict["automatic_timing_reasons"]}) != "disagreement":
                    return []
                return [row["window_start"] for row in alternate_windows(
                    [candidate], regions, observations, {}, None, disagreement_month_pairs
                )]

        self.assertEqual(selected_starts(september), ["2020-09-01", "2020-10-01"])
        self.assertEqual(selected_starts({**september, "onset_interval_start": "2020-08-28"}), [])
        self.assertEqual(selected_starts({**september, "peak_date": "2020-12-01"}), [])
        self.assertIsNone(timing_rejection({
            "exclusion_reason": "threshold_0.25_onset_month_changes;ndre_does_not_support_growth"
        }))

    def test_onset_gap_accepts_25_days_and_rejects_26(self):
        original, observations = timing_fixture()
        regions = [{"properties": {"fragment_id": "region", "crop": "carrot"}}]

        def options(gap, later_shift=0):
            shifted = [
                {**row, "date": (date.fromisoformat(row["date"]) + timedelta(
                    days=(gap - 5 if row["date"] >= original["onset_interval_end"] else 0)
                    + (later_shift if row["date"] > original["onset_interval_end"] else 0)
                )).isoformat()}
                for row in observations
            ]
            candidates = seasons.identify_candidates(detect_greenup(shifted))
            reasons = {row["candidate_id"]: "boundary" for row in candidates}
            return list(longer_gap_windows(
                candidates, reasons, regions, shifted, {}, timing_quality.DEFAULT_TIMING_POLICY
            ))

        self.assertEqual([row["window_start"] for row in options(25)],
                         ["2020-10-01", "2020-11-01"])
        self.assertEqual(options(26), [])
        self.assertEqual(options(25, later_shift=17), [])
        self.assertEqual(timing_rejection({
            "exclusion_reason": "onset_gap_too_large_or_invalid;threshold_0.25_onset_month_changes"
        }), "gap_disagreement")

    def test_window_completeness_still_rejects_bad_or_missing_bins(self):
        window, quality_rows, pixel_rows = feature_fixture()

        class Pixels:
            def for_regions(self, identifiers):
                return (row for row in pixel_rows if row["region_id"] in identifiers)

        self.assertEqual(len(list(complete_locations([window], Pixels(), quality_rows))), 2)
        self.assertEqual(list(complete_locations([window], Pixels(), quality_rows[:-1])), [])
        quality_rows[0]["valid_fraction"] = 0.79
        self.assertEqual(list(complete_locations([window], Pixels(), quality_rows)), [])

    def test_earliest_complete_window_can_change_the_matched_fields(self):
        old = ("a", "fragment", "EPSG:32631", 0, 0)
        new = ("a", "fragment", "EPSG:32631", 10, 0)
        fixed = {old: "carrot"}
        starts = {"a": "2020-05-01"}
        aligned = {"a": {old: "carrot"}}
        self.assertEqual(counts(fixed, aligned["a"])["matched"], {("carrot", "a")})
        merge_locations(starts, aligned, [
            (old, "carrot", "2020-05-01"), (new, "carrot", "2020-04-01")
        ])
        revised = counts(fixed, aligned["a"])
        self.assertEqual(revised["aligned_full"], {("carrot", "a")})
        self.assertEqual(revised["matched"], set())


if __name__ == "__main__":
    unittest.main()
