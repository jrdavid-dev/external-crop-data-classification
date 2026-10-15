"""Choose an observation window by timing evidence and completeness, never scores."""

from collections import Counter, defaultdict

POLICY = "earliest_complete"


def choose_complete_cycles(windows, features, exclusions, losses, bin_coverage):
    """Choose one four-month date range per source parcel, retaining all alternatives.

    A window is eligible when at least one pixel passes every unchanged quality
    check. Other pixels remain excluded. Fragments of the same parcel use the
    same selected date range; later windows cannot fill earlier-window gaps.
    """
    complete_counts = Counter(row["season_key"] for row in features)
    grouped = defaultdict(list)
    for window in windows:
        grouped[(window["crop"], window["partition_group"])].append(window)
    selected_keys, audit = set(), []
    for candidates in grouped.values():
        group_keys, group_audit = choose_group_cycles(candidates, complete_counts)
        selected_keys.update(group_keys)
        audit.extend(group_audit)
    selected_windows = [
        dict(
            window,
            cycle_selection_policy=POLICY,
            window_selection_reason="earliest_timing_qualified_window_with_complete_pixels",
        )
        for window in windows
        if window["season_key"] in selected_keys
    ]
    window_map = {window["season_key"]: window for window in selected_windows}
    selected_features = [
        {
            **row,
            "window_selection_reason": window_map[row["season_key"]][
                "window_selection_reason"
            ],
            "cycle_selection_policy": POLICY,
        }
        for row in features
        if row["season_key"] in selected_keys
    ]
    return (
        selected_windows,
        selected_features,
        [row for row in exclusions if row["season_key"] in selected_keys],
        [row for row in losses if row["season_key"] in selected_keys],
        [row for row in bin_coverage if row["season_key"] in selected_keys],
        audit,
    )


def choose_group_cycles(candidates, complete_counts):
    """One source parcel can have several fragments, all sharing one date range."""
    eligible = [
        window for window in candidates if complete_counts[window["season_key"]] > 0
    ]
    selected_dates = (
        min((window["window_start"], window["window_end"]) for window in eligible)
        if eligible
        else None
    )
    selected_keys, audit = set(), []
    for window in sorted(
        candidates, key=lambda row: (row["window_start"], row["season_key"])
    ):
        outcome = cycle_outcome(
            window, selected_dates, complete_counts[window["season_key"]]
        )
        audit.append(outcome)
        if outcome["cycle_selection_status"] == "selected":
            selected_keys.add(window["season_key"])
    return selected_keys, audit


def cycle_outcome(window, selected_dates, complete_count):
    if complete_count == 0:
        status, reason = "excluded_incomplete", "no_complete_quality_passing_pixels"
    elif (window["window_start"], window["window_end"]) == selected_dates:
        status, reason = "selected", "earliest_complete_window"
    else:
        status, reason = "excluded_later", "earlier_complete_window_available"
    return {
        **window,
        "complete_usable_pixels": complete_count,
        "cycle_selection_status": status,
        "cycle_selection_reason": reason,
        "cycle_selection_policy": POLICY,
    }


def flag_four_month_context(windows, candidates):
    """Keep the fixed input length distinct from the observed vegetation episode."""
    complete_by_group = defaultdict(list)
    for candidate in candidates:
        if candidate.get("cycle_status") == "complete_observed":
            complete_by_group[candidate["partition_group"]].append(candidate)
    return [
        context_flags(window, complete_by_group[window["partition_group"]])
        for window in windows
    ]


def context_flags(window, complete_cycles):
    next_onsets = sorted(
        candidate["onset_interval_start"]
        for candidate in complete_cycles
        if candidate["onset_interval_end"] > window["onset_interval_end"]
    )
    next_start = next_onsets[0] if next_onsets else ""
    return {
        **window,
        "next_complete_cycle_start": next_start,
        "window_overlaps_next_cycle": bool(
            next_start and next_start < window["window_end"]
        ),
        "window_extends_past_observed_cycle": bool(
            window.get("end_interval_end")
            and window["window_end"] > window["end_interval_end"]
        ),
    }
