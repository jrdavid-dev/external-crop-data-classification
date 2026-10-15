"""Reproducible quality assessment of estimated green-up timing.

These checks measure support and robustness in the same satellite observations.
They do not independently validate planting dates, harvest dates or crop identity.
"""

import hashlib
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from statistics import median

from crop_workflow import DEFAULT_DETECTOR, detect_greenup


DEFAULT_TIMING_POLICY = {
    "version": 1,
    "relative_thresholds": [0.15, 0.20, 0.25],
    "maximum_onset_gap_days": 20,
    "maximum_support_gap_days": 20,
    "baseline_days": 30,
    "confirmation_days": 30,
    "minimum_baseline_observations": 2,
    "minimum_confirmations": 2,
    "minimum_post_rise_fraction": 0.75,
    "minimum_ndre_increase": 0.02,
    "maximum_peak_shift_days": 15,
}


def fingerprint(value):
    serialized = json.dumps(value, sort_keys=True, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def finite_value(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) and number != -9999 else None


def validate_policy(policy):
    if set(policy) != set(DEFAULT_TIMING_POLICY) or policy["version"] != 1:
        raise ValueError("Unknown automatic timing policy keys or version.")
    thresholds = policy["relative_thresholds"]
    if len(set(thresholds)) < 3 or any(not 0 < threshold < 1 for threshold in thresholds):
        raise ValueError("Use at least three distinct relative thresholds in (0, 1).")
    bounded = policy["minimum_post_rise_fraction"]
    if not 0 < bounded <= 1 or not math.isfinite(policy["minimum_ndre_increase"]):
        raise ValueError("Invalid post-rise fraction or NDRE threshold.")
    integer_keys = [key for key in policy if key.endswith("_days") or key.endswith("_observations")]
    integer_keys.append("minimum_confirmations")
    if any(not isinstance(policy[key], int) or policy[key] < 1 for key in integer_keys):
        raise ValueError("Timing-policy day and observation counts must be positive integers.")
    if policy["minimum_ndre_increase"] <= 0:
        raise ValueError("NDRE must increase by a positive, predefined margin.")


def normalize_observations(rows):
    """Count actual dates once; repeated rows must not manufacture confirmations."""
    normalized, identifiers = [], set()
    for row in rows:
        if row.get("source_kind") != "observed_daily":
            continue
        observation_date = date.fromisoformat(row["date"]).isoformat()
        identifier = (row["region_id"], observation_date)
        if identifier in identifiers:
            raise ValueError(f"Duplicate actual-date summary: {identifier}")
        identifiers.add(identifier)
        normalized.append({"region_id": row["region_id"], "date": observation_date,
                           "source_kind": "observed_daily",
                           "valid_fraction": finite_value(row.get("valid_fraction")) or 0.0,
                           **{index: finite_value(row.get(index)) for index in ("NDVI", "NDRE", "NDMI")}})
    return sorted(normalized, key=lambda row: (row["region_id"], row["date"]))


def assess_timing(windows, observation_rows, detector_settings, policy=None):
    """Assess each proposal using fixed rules and threshold sensitivity reruns."""
    settings = {**DEFAULT_TIMING_POLICY, **(policy or {})}
    validate_policy(settings)
    if not windows:
        return []
    region_ids = {window["region_id"] for window in windows}
    observations = normalize_observations([row for row in observation_rows if row["region_id"] in region_ids])
    by_region = defaultdict(list)
    for row in observations:
        by_region[row["region_id"]].append(row)
    sensitivity = run_threshold_checks(observations, detector_settings, settings)
    policy_fingerprint = fingerprint({"quality_policy": settings, "detector_parameters": detector_settings})
    return [assess_window(window, by_region[window["region_id"]], sensitivity,
                          detector_settings, settings, policy_fingerprint) for window in windows]


def run_threshold_checks(observations, detector_settings, policy):
    results = {}
    for threshold in policy["relative_thresholds"]:
        by_region = defaultdict(list)
        outcomes = detect_greenup(observations, {**detector_settings, "relative_threshold": threshold})
        plausible = [outcome for outcome in outcomes if outcome["status"] in {"candidate", "provisional", "ambiguous"}]
        for outcome in plausible:
            by_region[outcome["region_id"]].append(outcome)
        results[threshold] = by_region
    return results


def assess_window(window, observations, sensitivity, detector_settings, policy, policy_fingerprint):
    detector = {**DEFAULT_DETECTOR, **detector_settings}
    eligible = [row for row in observations if row["NDVI"] is not None
                and detector["minimum_valid_fraction"] <= row["valid_fraction"] <= 1]
    failures, metrics = observation_support(window, eligible, detector, policy)
    stability_failures, details = threshold_stability(window, sensitivity, policy)
    failures.extend(stability_failures)
    return {"candidate_id": window["candidate_id"], "region_id": window["region_id"],
            "automatic_timing_status": "passed" if not failures else "failed",
            "automatic_timing_reasons": ";".join(failures) or "all_predefined_checks_passed",
            "automatic_policy_fingerprint": policy_fingerprint,
            "observation_fingerprint": fingerprint(observations),
            "eligible_observations": len(eligible), **metrics,
            "threshold_checks": json.dumps(details, sort_keys=True),
            "assessment_scope": "satellite_timing_consistency_only"}


def observation_support(window, observations, detector, policy):
    onset_start = date.fromisoformat(window["onset_interval_start"])
    onset_end = date.fromisoformat(window["onset_interval_end"])
    by_date = {row["date"]: row for row in observations}
    start_row = by_date.get(onset_start.isoformat())
    end_row = by_date.get(onset_end.isoformat())
    failures = []
    gap_days = (onset_end - onset_start).days
    if window.get("source_kind") != "observed_daily" or start_row is None or end_row is None:
        failures.append("onset_endpoints_not_supported_by_actual_observations")
    if not 0 < gap_days <= policy["maximum_onset_gap_days"]:
        failures.append("onset_gap_too_large_or_invalid")
    if onset_start.strftime("%Y-%m") != onset_end.strftime("%Y-%m"):
        failures.append("onset_month_uncertain")
    amplitude = finite_value(window.get("amplitude"))
    if amplitude is None or amplitude < detector["minimum_amplitude"]:
        failures.append("insufficient_supported_ndvi_amplitude")
    threshold = finite_value(window.get("ndvi_threshold"))
    if not supports_crossing(start_row, end_row, threshold):
        failures.append("raw_observations_do_not_bracket_threshold")
    before = [row for row in observations
              if onset_start - timedelta(days=policy["baseline_days"]) <= date.fromisoformat(row["date"]) <= onset_start]
    after = [row for row in observations
             if onset_end < date.fromisoformat(row["date"]) <= onset_end + timedelta(days=policy["confirmation_days"])]
    growth_failures, metrics = sustained_growth(before, after, onset_end, threshold, policy)
    failures.extend(growth_failures)
    return failures, {"onset_gap_days": gap_days, **metrics}


def supports_crossing(start_row, end_row, threshold):
    if start_row is None or end_row is None or threshold is None:
        return False
    return start_row["NDVI"] <= threshold < end_row["NDVI"]


def index_change(before, after, index):
    baseline = [row[index] for row in before if row[index] is not None]
    growth = [row[index] for row in after if row[index] is not None]
    if len(baseline) < 2 or len(growth) < 2:
        return None
    return median(growth) - median(baseline)


def sustained_growth(before, after, onset_end, threshold, policy):
    supporting = [row for row in after if threshold is not None and row["NDVI"] > threshold]
    fraction = len(supporting) / len(after) if after else 0.0
    support_dates = [onset_end] + [date.fromisoformat(row["date"]) for row in supporting]
    maximum_gap = max(((later - earlier).days for earlier, later in zip(support_dates, support_dates[1:])), default=0)
    ndre_change = index_change(before, after, "NDRE")
    failures = []
    if len(before) < policy["minimum_baseline_observations"]:
        failures.append("insufficient_nearby_baseline_observations")
    if len(supporting) < policy["minimum_confirmations"]:
        failures.append("insufficient_later_growth_confirmations")
    if fraction < policy["minimum_post_rise_fraction"]:
        failures.append("growth_not_sustained_in_post_onset_observations")
    if maximum_gap > policy["maximum_support_gap_days"]:
        failures.append("large_gap_between_growth_confirmations")
    if ndre_change is None or ndre_change < policy["minimum_ndre_increase"]:
        failures.append("ndre_does_not_support_growth")
    # NDMI is reported, not forced to rise: water conditions can change during growth.
    return failures, {"baseline_observations": len(before), "post_onset_observations": len(after),
                      "later_confirmations": len(supporting), "post_rise_fraction": fraction,
                      "maximum_confirmation_gap_days": maximum_gap, "ndre_change": ndre_change,
                      "ndmi_change": index_change(before, after, "NDMI")}


def threshold_stability(window, sensitivity, policy):
    failures, details = [], {}
    for threshold, by_region in sensitivity.items():
        candidates = by_region.get(window["region_id"], [])
        reason = threshold_problem(window, candidates, policy)
        details[str(threshold)] = reason or "same_cycle_and_onset_month"
        if reason:
            failures.append(f"threshold_{threshold:g}_{reason}")
    return failures, details


def threshold_problem(window, candidates, policy):
    if len(candidates) != 1:
        return "no_unique_supported_cycle"
    candidate = candidates[0]
    if candidate["status"] != "candidate" or candidate.get("source_kind") != "observed_daily":
        return "cycle_not_supported_by_actual_observations"
    month = window["window_start"][:7]
    if candidate["onset_interval_start"][:7] != month or candidate["onset_interval_end"][:7] != month:
        return "onset_month_changes"
    peak_shift = abs((date.fromisoformat(candidate["peak_date"]) - date.fromisoformat(window["peak_date"])).days)
    if peak_shift > policy["maximum_peak_shift_days"]:
        return "different_peak_cycle"
    return ""
