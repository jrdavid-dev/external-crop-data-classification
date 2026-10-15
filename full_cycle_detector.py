"""Find observed vegetation episodes; never equate their boundaries with crop dates."""

from datetime import date

import numpy as np
from scipy.signal import find_peaks

METHOD = "full_growth_cycles_v1"


def smooth_observations(observations, neighbour_days):
    """Reuse the existing three-observation median without bridging distant dates."""
    dates = [date.fromisoformat(row["date"]) for row in observations]
    values = np.asarray([float(row["NDVI"]) for row in observations])
    smooth = [
        np.median(
            [
                values[position]
                for position in range(max(0, index - 1), min(len(values), index + 2))
                if abs((dates[position] - dates[index]).days) <= neighbour_days
            ]
        )
        for index in range(len(values))
    ]
    return dates, values, np.asarray(smooth)


def identify_episodes(dates, values, smooth, settings):
    """Prominence suppresses minor peaks; supported low periods separate seasons."""
    peaks, properties = find_peaks(smooth, prominence=0)
    episodes = [
        {
            "peak": int(peak),
            "left": int(properties["left_bases"][index]),
            "right": int(properties["right_bases"][index]),
            "merged_peaks": 1,
        }
        for index, peak in enumerate(peaks)
        if max(
            smooth[peak] - smooth[properties["left_bases"][index]],
            smooth[peak] - smooth[properties["right_bases"][index]],
        )
        >= settings["minimum_amplitude"]
    ]
    # Endpoint peaks are retained as incomplete evidence, not invented full seasons.
    if (
        len(smooth) > 1
        and smooth[0] - smooth.min() >= settings["minimum_amplitude"]
        and smooth[0] >= smooth[1]
    ):
        episodes.insert(
            0,
            {"peak": 0, "left": 0, "right": int(np.argmin(smooth)), "merged_peaks": 1},
        )
    if (
        len(smooth) > 1
        and smooth[-1] - smooth.min() >= settings["minimum_amplitude"]
        and smooth[-1] >= smooth[-2]
    ):
        episodes.append(
            {
                "peak": len(smooth) - 1,
                "left": int(np.argmin(smooth)),
                "right": len(smooth) - 1,
                "merged_peaks": 1,
            }
        )
    merged = []
    for episode in episodes:
        if merged and not separate_low_state(
            merged[-1], episode, dates, values, smooth, settings
        ):
            merged[-1] = merge_episodes(merged[-1], episode, smooth)
            continue
        merged.append(episode)
    return merged


def separate_low_state(previous, current, dates, values, smooth, settings):
    """A second peak requires a real intervening low, not just a small canopy dip."""
    baseline = min(float(smooth[previous["left"]]), float(smooth[current["right"]]))
    peak_value = min(float(smooth[previous["peak"]]), float(smooth[current["peak"]]))
    threshold = baseline + settings["relative_threshold"] * (peak_value - baseline)
    lows = [
        index
        for index in range(previous["peak"] + 1, current["peak"])
        if smooth[index] <= threshold and values[index] <= threshold
    ]
    return supported_low_run(lows, dates, settings)


def supported_low_run(indices, dates, settings):
    """Use the existing confirmation count and time-gap checks for a low state."""
    from timing_quality import DEFAULT_TIMING_POLICY

    count = settings["minimum_confirmations"]
    for start in range(len(indices) - count + 1):
        group = indices[start : start + count]
        gaps = [
            (dates[later] - dates[earlier]).days
            for earlier, later in zip(group, group[1:])
        ]
        if (
            group[-1] - group[0] == count - 1
            and max(gaps, default=0)
            <= DEFAULT_TIMING_POLICY["maximum_support_gap_days"]
        ):
            return True
    return False


def merge_episodes(previous, current, smooth):
    strongest = (
        previous if smooth[previous["peak"]] >= smooth[current["peak"]] else current
    )
    return {
        "peak": strongest["peak"],
        "left": min(previous["left"], current["left"]),
        "right": max(previous["right"], current["right"]),
        "merged_peaks": previous["merged_peaks"] + current["merged_peaks"],
    }


def detect_region(region_id, source_rows, settings):
    """Keep incomplete episodes visible instead of manufacturing missing boundaries."""
    observations = sorted(
        [
            row
            for row in source_rows
            if row.get("NDVI") is not None
            and np.isfinite(float(row["NDVI"]))
            and float(row.get("valid_fraction", 0))
            >= settings["minimum_valid_fraction"]
        ],
        key=lambda row: row["date"],
    )
    if len(observations) < settings["minimum_points"]:
        return [
            {
                "region_id": region_id,
                "status": "rejected",
                "reason": "insufficient_supported_observations",
                "cycle_detector": METHOD,
            }
        ]
    dates, values, smooth = smooth_observations(
        observations, settings["smoothing_neighbour_days"]
    )
    episodes = identify_episodes(dates, values, smooth, settings)
    results = [
        describe_episode(episode, observations, dates, values, smooth, settings)
        for episode in episodes
    ]
    if not results:
        return [
            {
                "region_id": region_id,
                "status": "rejected",
                "reason": "no_prominent_growth_episode",
                "cycle_detector": METHOD,
            }
        ]
    complete = [result for result in results if result["status"] == "candidate"]
    if len(complete) > 1:
        for result in complete:
            result.update(
                status="ambiguous", reason="multiple_complete_observed_cycles"
            )
    return results


def onset_evidence(episode, dates, values, smooth, settings):
    """Search back from the broad peak; rejected local maxima never reset the baseline."""
    peak = episode["peak"]
    baseline_indices = [
        index
        for index in range(episode["left"], peak)
        if 0 < (dates[peak] - dates[index]).days <= settings["baseline_lookback_days"]
    ]
    if len(baseline_indices) < settings["minimum_confirmations"]:
        return None
    baseline = float(np.quantile(smooth[baseline_indices], 0.10))
    amplitude = float(smooth[peak] - baseline)
    threshold = baseline + settings["relative_threshold"] * amplitude
    lows = [
        index
        for index in baseline_indices
        if smooth[index] <= threshold and values[index] <= threshold
    ]
    if not lows or amplitude < settings["minimum_amplitude"]:
        return None
    low = lows[-1]
    crossings = [
        index
        for index in range(low + 1, peak + 1)
        if smooth[index] > threshold and values[index] > threshold
    ]
    if not crossings:
        return None
    return {
        "low": low,
        "crossing": crossings[0],
        "baseline": baseline,
        "amplitude": amplitude,
        "threshold": threshold,
    }


def decline_evidence(episode, dates, values, smooth, threshold, settings):
    """Require sustained observed decline with the existing confirmation thresholds."""
    from timing_quality import DEFAULT_TIMING_POLICY

    for index in range(episode["peak"] + 1, episode["right"] + 1):
        if values[index] > threshold or smooth[index] > threshold:
            continue
        after = [
            position
            for position in range(index, len(values))
            if (dates[position] - dates[index]).days <= settings["confirmation_days"]
        ]
        lows = [position for position in after if values[position] <= threshold]
        if len(lows) < settings["minimum_confirmations"]:
            continue
        if len(lows) / len(after) < DEFAULT_TIMING_POLICY["minimum_post_rise_fraction"]:
            continue
        preceding_highs = [
            position
            for position in range(episode["peak"], index)
            if values[position] > threshold and smooth[position] > threshold
        ]
        if not preceding_highs:
            continue
        return {
            "before": preceding_highs[-1],
            "after": index,
            "confirmations": lows[: settings["minimum_confirmations"]],
        }
    return None


def describe_episode(episode, observations, dates, values, smooth, settings):
    """Separate a complete vegetation cycle from a partial or gap-obscured episode."""
    peak = episode["peak"]
    result = {
        "region_id": observations[0]["region_id"],
        "partition_group": observations[0].get("partition_group"),
        "source_kind": observations[0].get("source_kind", "observed_daily"),
        "peak_date": dates[peak].isoformat(),
        "cycle_detector": METHOD,
        "merged_peaks": episode["merged_peaks"],
        "label_relevance": "not_verified",
        "validated": False,
    }
    onset = onset_evidence(episode, dates, values, smooth, settings)
    if onset is None:
        return {
            **result,
            "status": "incomplete_start",
            "cycle_status": "start_unobserved",
            "reason": "no_supported_pre_growth_low_state",
        }
    low, crossing = onset["low"], onset["crossing"]
    result.update(
        onset_interval_start=dates[low].isoformat(),
        onset_interval_end=dates[crossing].isoformat(),
        baseline=onset["baseline"],
        amplitude=onset["amplitude"],
        ndvi_threshold=onset["threshold"],
        gap_days=(dates[crossing] - dates[low]).days,
        month_boundary_uncertain=dates[low].month != dates[crossing].month,
    )
    decline = decline_evidence(
        episode, dates, values, smooth, onset["threshold"], settings
    )
    if decline is None:
        return {
            **result,
            "status": "incomplete_end",
            "cycle_status": "end_unobserved",
            "reason": "no_supported_post_peak_low_state",
        }
    result.update(
        end_interval_start=dates[decline["before"]].isoformat(),
        end_interval_end=dates[decline["after"]].isoformat(),
        cycle_length_min_days=(dates[decline["before"]] - dates[crossing]).days,
        cycle_length_max_days=(dates[decline["after"]] - dates[low]).days,
    )
    problem = cycle_problem(onset, decline, dates, values, smooth, settings)
    if problem:
        return {
            **result,
            "status": "unresolved",
            "cycle_status": "partial_or_obscured",
            "reason": problem,
        }
    confirmations = [
        index
        for index in range(crossing + 1, len(values))
        if 0 < (dates[index] - dates[crossing]).days <= settings["confirmation_days"]
        and values[index] > onset["threshold"]
    ]
    if len(confirmations) < settings["minimum_confirmations"]:
        return {
            **result,
            "status": "rejected",
            "cycle_status": "insufficient_growth",
            "reason": "insufficient_sustained_growth",
        }
    status = "candidate" if result["source_kind"] == "observed_daily" else "provisional"
    return {
        **result,
        "status": status,
        "cycle_status": (
            "complete_observed" if status == "candidate" else "composite_only"
        ),
        "reason": "supported_low_peak_low_cycle",
        "confirmations": len(confirmations),
    }


def cycle_problem(onset, decline, dates, values, smooth, settings):
    """Do not certify a cycle hidden by gaps or a rise inside an already-high canopy."""
    from timing_quality import DEFAULT_TIMING_POLICY

    low, crossing = onset["low"], onset["crossing"]
    annual_background = float(np.quantile(smooth, 0.10))
    low_state = annual_background + settings["relative_threshold"] * (
        float(smooth.max()) - annual_background
    )
    preceding = [
        index
        for index in range(crossing)
        if (dates[crossing] - dates[index]).days
        <= DEFAULT_TIMING_POLICY["baseline_days"]
    ]
    if (
        sum(
            values[index] <= low_state and smooth[index] <= low_state
            for index in preceding
        )
        < DEFAULT_TIMING_POLICY["minimum_baseline_observations"]
    ):
        return "possible_partial_cycle_high_background"
    last = decline["confirmations"][-1]
    gaps = [
        (dates[index] - dates[index - 1]).days for index in range(low + 1, last + 1)
    ]
    if max(gaps, default=0) > DEFAULT_TIMING_POLICY["maximum_support_gap_days"]:
        return "growth_cycle_obscured_by_observation_gap"
    return ""
