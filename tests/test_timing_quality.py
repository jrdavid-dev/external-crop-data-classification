"""Check automatic timing decisions without claiming independent date accuracy."""

import copy
import unittest
from datetime import date, timedelta

from crop_pipeline.imagery import quality as crop_workflow
from crop_pipeline.windows import seasons as season_workflow
from crop_pipeline.windows import timing_quality


def timing_fixture():
    values = [.1, .1, .1, .15, .3, .5, .7, .7, .6, .4, .2]
    observations = [{"region_id": "region", "date": (date(2020, 10, 1) + timedelta(days=5 * index)).isoformat(),
                     "source_kind": "observed_daily", "valid_fraction": 1.0,
                     "NDVI": value, "NDRE": value * .5, "NDMI": value * .2}
                    for index, value in enumerate(values)]
    candidate = crop_workflow.detect_greenup(observations)[0]
    window = {**candidate, "candidate_id": season_workflow.identify_candidates([candidate])[0]["candidate_id"],
              "window_start": "2020-10-01", "window_end": "2021-02-01",
              "window_status": "satellite_estimated", "timing_review_status": "pending",
              "label_relevance": "unverified", "label_status": "reported"}
    return window, observations


class AutomaticTimingTests(unittest.TestCase):
    def test_supported_growth_passes_without_a_human_review(self):
        window, observations = timing_fixture()
        assessment = timing_quality.assess_timing([window], observations, {})[0]
        self.assertEqual(assessment["automatic_timing_status"], "passed")
        self.assertGreaterEqual(assessment["later_confirmations"], 2)
        qualified, _, audit = season_workflow.qualify_estimated_windows(
            [window], [window], observations, {"detector_parameters": {}})
        self.assertEqual(qualified[0]["timing_review_status"], "automated_pass")
        self.assertEqual(qualified[0]["label_relevance"], "unverified")
        self.assertEqual(qualified[0]["timing_validation"], "algorithmic_consistency_only")
        self.assertNotIn("reviewer", qualified[0])
        self.assertEqual(audit[0]["window_status"], "satellite_estimated")

    def test_flat_ndre_rejects_an_ndvi_only_rise(self):
        window, observations = timing_fixture()
        for row in observations:
            row["NDRE"] = .1
        qualified, assessments, audit = season_workflow.qualify_estimated_windows(
            [window], [window], observations, {"detector_parameters": {}})
        self.assertFalse(qualified)
        self.assertIn("ndre_does_not_support_growth", assessments[0]["automatic_timing_reasons"])
        self.assertEqual(audit[0]["window_status"], "unresolved")

    def test_ndmi_is_diagnostic_and_does_not_have_to_increase(self):
        window, observations = timing_fixture()
        for index, row in enumerate(observations):
            row["NDMI"] = .5 - index * .01
        assessment = timing_quality.assess_timing([window], observations, {})[0]
        self.assertEqual(assessment["automatic_timing_status"], "passed")
        self.assertLess(assessment["ndmi_change"], 0)

    def test_duplicate_dates_cannot_manufacture_confirmations(self):
        window, observations = timing_fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate actual-date"):
            timing_quality.assess_timing([window], observations + [observations[0]], {})

    def test_missing_observations_and_long_onset_gaps_fail(self):
        window, _ = timing_fixture()
        assessment = timing_quality.assess_timing([window], [], {})[0]
        self.assertEqual(assessment["automatic_timing_status"], "failed")
        self.assertIn("onset_endpoints_not_supported", assessment["automatic_timing_reasons"])
        self.assertIn("insufficient_later_growth", assessment["automatic_timing_reasons"])
        window["onset_interval_start"] = "2020-09-01"
        assessment = timing_quality.assess_timing([window], [], {})[0]
        self.assertIn("onset_gap_too_large", assessment["automatic_timing_reasons"])

    def test_threshold_disagreement_is_not_automatically_resolved(self):
        window, _ = timing_fixture()
        changed_candidate = {**window, "onset_interval_start": "2020-09-29", "onset_interval_end": "2020-10-02"}
        reason = timing_quality.threshold_problem(window, [changed_candidate], timing_quality.DEFAULT_TIMING_POLICY)
        self.assertEqual(reason, "onset_month_changes")
        reason = timing_quality.threshold_problem(window, [window, window], timing_quality.DEFAULT_TIMING_POLICY)
        self.assertEqual(reason, "no_unique_supported_cycle")

    def test_changing_crop_names_does_not_change_timing_decisions(self):
        window, observations = timing_fixture()
        first = timing_quality.assess_timing([window], observations, {})
        for row in observations:
            row["crop"] = "unknown"
        second = timing_quality.assess_timing([window], observations, {})
        self.assertEqual(first, second)

    def test_changing_policy_is_recorded_and_bad_rules_are_rejected(self):
        window, observations = timing_fixture()
        first = timing_quality.assess_timing([window], observations, {})[0]
        second = timing_quality.assess_timing([window], observations, {}, {"minimum_ndre_increase": .9})[0]
        self.assertNotEqual(first["automatic_policy_fingerprint"], second["automatic_policy_fingerprint"])
        self.assertEqual(second["automatic_timing_status"], "failed")
        with self.assertRaisesRegex(ValueError, "three distinct"):
            timing_quality.assess_timing([window], observations, {}, {"relative_thresholds": [.2]})

    def test_failed_automatic_check_does_not_fabricate_or_overwrite_a_human_review(self):
        window, observations = timing_fixture()
        window.update(timing_review_status="reviewed", reviewer="Recorded reviewer")
        for row in observations:
            row["NDRE"] = .1
        qualified, assessments, _ = season_workflow.qualify_estimated_windows(
            [window], [window], observations, {"detector_parameters": {}})
        self.assertEqual(qualified[0]["timing_review_status"], "reviewed")
        self.assertEqual(qualified[0]["reviewer"], "Recorded reviewer")
        self.assertEqual(assessments[0]["automatic_timing_status"], "failed")

    def test_training_requires_the_matching_automatic_policy(self):
        window, observations = timing_fixture()
        qualified, _, _ = season_workflow.qualify_estimated_windows(
            [window], [window], observations, {"detector_parameters": {}})
        configuration = {"automatic_timing_policy": copy.deepcopy(timing_quality.DEFAULT_TIMING_POLICY),
                         "detector_parameters": {}}
        season_workflow.validate_automatic_policy(qualified[0], configuration)
        configuration["automatic_timing_policy"]["minimum_ndre_increase"] = .03
        with self.assertRaisesRegex(ValueError, "different policy"):
            season_workflow.validate_automatic_policy(qualified[0], configuration)


if __name__ == "__main__":
    unittest.main()
