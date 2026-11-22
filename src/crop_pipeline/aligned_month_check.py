"""Compare aligned timing rules on development and primary JECAM imagery."""

import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from crop_pipeline.analysis_workflow import resolve_stage_pointer
from crop_pipeline.cabbage_test_check import discover
from crop_pipeline.imagery import quality
from crop_pipeline.preprocessing_workflow import paths, pointer
from crop_pipeline.sources import adaptive, pilot_sources, test_cohort
from crop_pipeline.windows import annual_archive, external, json_rows, seasons, timing_quality


def onset_months(candidate):
    return tuple(date.fromisoformat(candidate[key]).replace(day=1)
                 for key in ("onset_interval_start", "onset_interval_end"))


def boundary_month_pairs(candidate):
    first, last = onset_months(candidate)
    if last == seasons.add_calendar_months(first, 1):
        yield first, last


def disagreement_month_pairs(candidate):
    first, last = onset_months(candidate)
    if first == last:
        yield seasons.add_calendar_months(first, -1), first
        yield first, seasons.add_calendar_months(first, 1)


def boundary_windows(candidates, regions, observations, detector, policy):
    """Try the two onset months only when the same cycle remains supported."""
    return alternate_windows(candidates, regions, observations, detector, policy, boundary_month_pairs)


def alternate_windows(candidates, regions, observations, detector, policy, month_pairs):
    properties = {region["properties"]["fragment_id"]: region["properties"] for region in regions}
    proposals = []
    for candidate in candidates:
        for months in month_pairs(candidate):
            proposals.append({
                **candidate, **properties[candidate["region_id"]],
                "window_start": months[0].isoformat(),
                "alternate_onset_months": tuple(month.isoformat()[:7] for month in months),
            })
    assessments = timing_quality.assess_timing(
        proposals, observations, detector, policy, "earliest_complete"
    )
    for proposal, assessment in zip(proposals, assessments, strict=True):
        if assessment["automatic_timing_status"] != "passed":
            continue
        for month in proposal["alternate_onset_months"]:
            start = date.fromisoformat(month + "-01")
            yield {
                **proposal,
                "season_key": proposal["region_id"] + "|" + proposal["candidate_id"] + "|" + month,
                "window_start": start.isoformat(),
                "window_end": seasons.add_calendar_months(start, 4).isoformat(),
                "window_status": "satellite_estimated",
                "timing_review_status": "automated_pass",
            }


def complete_locations(windows, pixels, quality_rows, record_feature=None):
    """Build features through the standard quality gate, keeping only identities."""
    by_region, bins = defaultdict(list), defaultdict(list)
    for window in windows:
        by_region[window["region_id"]].append(window)
    for row in quality_rows:
        if row["region_id"] in by_region:
            bins[row["region_id"]].append(row)
    for region, candidates in by_region.items():
        rows, _, _, _ = seasons.build_features(
            candidates, pixels.for_regions([region]), bins[region], minimum_fraction=0.8
        )
        for row in rows:
            key = external.pixel_key(row)
            if record_feature is not None:
                record_feature(key, row)
            yield key, row["crop"], row["window_start"]


def choose_locations(rows):
    """Keep locations from the earliest complete start across parcel fragments."""
    starts, locations = {}, {}
    for key, crop, start in rows:
        group = key[0]
        if group not in starts or start < starts[group]:
            starts[group], locations[group] = start, {}
        if start == starts[group]:
            locations[group][key] = crop
    return starts, locations


def population(row):
    if row["partition"] == external.DEVELOPMENT:
        return external.DEVELOPMENT
    if row["partition"] == "test" and row.get("test_cohort", "jecam") == "jecam":
        return "jecam"
    return None


def saved_locations(run, record_feature=None, *, include_jecam=True):
    fixed, references, estimated = defaultdict(dict), defaultdict(dict), defaultdict(list)
    print("Reading saved complete pixel identities (the saved export can be several GB).", flush=True)
    for recipe, row, span in json_rows.read_rows(run / "windows.json", {"fixed", "aligned"}):
        if (span is None or (cohort := population(row)) is None
                or (cohort == "jecam" and not include_jecam)):
            continue
        key = external.pixel_key(row)
        if recipe == "fixed":
            fixed[cohort][key] = row["crop"]
            continue
        if record_feature is not None and cohort == external.DEVELOPMENT:
            record_feature(key, row)
        if row["window_status"] == "reference_supported":
            references[cohort][key] = row["crop"]
        else:
            estimated[cohort].append((key, row["crop"], row["window_start"]))
    return {cohort: (fixed[cohort], *choose_locations(estimated[cohort]), references[cohort])
            for cohort in (external.DEVELOPMENT, "jecam")}


def new_sources(report, snapshot):
    campaign = adaptive.root(report)
    records = adaptive.registered_handoffs(campaign)
    additions = []
    for receipt in sorted(campaign.glob("rounds/*/*/completed.json")):
        if snapshot.get(str(receipt.resolve())) == seasons.file_sha256(receipt):
            continue
        additions.append(receipt.parent.name)
    known = {str(Path(record["path"]).resolve()) for record in records
             if snapshot.get(str(Path(record["path"]).resolve())) == record["sha256"]}
    # Campaign snapshots typically record receipts rather than the external handoff path.
    for receipt in campaign.glob("rounds/*/*/completed.json"):
        if snapshot.get(str(receipt.resolve())) == seasons.file_sha256(receipt):
            known.update(str(Path(record["path"]).resolve())
                         for record in json.loads(receipt.read_text())["handoffs"])
    manifests = sorted({str(Path(record["path"]).resolve()) for record in records} - known)
    sources = pilot_sources.load_sources(manifests) if manifests else []
    return additions, [source for source in sources
                       if test_cohort.source_role(source["source_name"]) in test_cohort.DEVELOPMENT]


def eligible_new_regions(sources, regions, report, saved_ids):
    reserved = [source for cohort in discover(report) for source in cohort["sources"]]
    _, excluded = test_cohort.exclude_overlap(
        test_cohort.source_footprints(sources), test_cohort.source_footprints(reserved)
    )
    assignments = adaptive.assignments(report)
    return [region for region in regions
            if region["properties"]["fragment_id"] not in saved_ids
            and region["properties"]["partition_group"] not in excluded
            and assignments.get(region["properties"]["partition_group"]) in {"train", "validation"}]


def eligible_locations(fixed, aligned, *, development=True):
    fixed, aligned = [
        {key: (None, {"partition": external.DEVELOPMENT, "crop": crop, "source_dataset": ""})
         for key, crop in rows.items()} for rows in (fixed, aligned)
    ]
    if development:
        for rows in (fixed, aligned):
            external.exclude_shared_development_locations(rows)
    return {
        cohort: {key: aligned[key][1]["crop"] for key in external.capped_keys(keys)}
        for cohort, keys in (("aligned_full", aligned), ("matched", fixed.keys() & aligned.keys()))
    }


def counts(fixed, aligned, *, development=True):
    return {cohort: {(crop, key[0]) for key, crop in rows.items()}
            for cohort, rows in eligible_locations(fixed, aligned, development=development).items()}


def timing_rejection(row):
    reason = row["exclusion_reason"]
    if reason == "month_boundary_requires_timing_review":
        return "boundary"
    failures = set(reason.split(";"))
    gap_failure = "onset_gap_too_large_or_invalid"
    has_gap = gap_failure in failures
    failures.discard(gap_failure)
    only_month_changes = all(part.startswith("threshold_") and part.endswith("_onset_month_changes")
                             for part in failures)
    if has_gap and only_month_changes:
        return "gap_disagreement" if failures else "gap"
    if failures and only_month_changes:
        return "disagreement"
    return None


def saved_rejected_candidates(run, region_ids):
    # Some review rows embed a large dataset revision record.
    previous_limit = csv.field_size_limit(8 * 1024 * 1024)
    try:
        with (run / "seasons/season_review_audit.csv").open(newline="", encoding="utf-8") as stream:
            return {row["candidate_id"]: timing_rejection(row) for row in csv.DictReader(stream)
                    if row["region_id"] in region_ids}
    finally:
        csv.field_size_limit(previous_limit)


def relaxed_locations(candidates, rejections, regions, observations, detector, policy,
                      pixels, quality_rows, record_feature=None):
    results = []
    for reason, month_pairs in (("boundary", boundary_month_pairs),
                                ("disagreement", disagreement_month_pairs)):
        flagged = [row for row in candidates if rejections.get(row["candidate_id"]) == reason]
        print(f"Checking {len(flagged)} {reason} candidates.", flush=True)
        options = alternate_windows(flagged, regions, observations, detector, policy, month_pairs)
        results.append(list(complete_locations(options, pixels, quality_rows, record_feature)))
    gap = list(complete_locations(
        longer_gap_windows(candidates, rejections, regions, observations, detector, policy),
        pixels, quality_rows, record_feature,
    ))
    return *results, gap


def longer_gap_windows(candidates, rejections, regions, observations, detector, policy):
    """Raise only the limit for the actual observations bracketing onset."""
    baseline_limit = policy["maximum_onset_gap_days"]

    def month_pairs(candidate):
        reason = rejections[candidate["candidate_id"]]
        if reason == "boundary":
            yield from boundary_month_pairs(candidate)
        elif reason == "gap_disagreement":
            yield from disagreement_month_pairs(candidate)
        else:
            start, _ = onset_months(candidate)
            yield (start,)

    eligible = [candidate for candidate in candidates
                if rejections.get(candidate["candidate_id"]) in {"boundary", "gap", "gap_disagreement"}
                and baseline_limit < candidate.get("gap_days", 0) <= 25]
    print(f"Checking {len(eligible)} longer onset-gap candidates.", flush=True)
    return alternate_windows(
        eligible, regions, observations, detector, {**policy, "maximum_onset_gap_days": 25},
        month_pairs,
    )


def merge_locations(starts, aligned, rows):
    added_starts, added = choose_locations(rows)
    for group, start in added_starts.items():
        if group not in starts or start < starts[group]:
            starts[group], aligned[group] = start, added[group]
        elif start == starts[group]:
            aligned[group].update(added[group])


def compare(repository, *, record_feature=None, record_matched=None, include_jecam=True):
    root, report, output = paths(repository)
    handoff = resolve_stage_pointer(pointer(root, "features"))
    baseline = json.loads(handoff.read_text()).get("baseline_features")
    run = Path(baseline).parent if baseline else handoff.parent
    saved = json.loads((run / "quality.json").read_text())
    snapshot = json.loads((run / "collection_snapshot.json").read_text())["campaign_metadata"]
    policy_record = json.loads((run / "seasons/automatic_timing_policy.json").read_text())
    detector, policy = policy_record["detector_parameters"], policy_record["quality_policy"]
    review_dir = run / "seasons/review"
    frozen = review_dir / "frozen_configuration.json"
    configuration = json.loads((frozen if frozen.exists() else
                                review_dir / "frozen_configuration.template.json").read_text())
    configuration.update(cycle_selection_policy="earliest_complete", automatic_timing_policy=policy)
    saved_regions = [region for region in saved["regions"]
                     if (region["properties"]["partition"] == external.DEVELOPMENT
                         or (include_jecam and population(region["properties"]) == "jecam"))
                     and region["properties"].get("source_split", "train") == "train"]
    region_population = {region["properties"]["fragment_id"]: population(region["properties"])
                         for region in saved_regions}
    rejections = saved_rejected_candidates(run, region_population)
    candidates = seasons.identify_candidates(json.loads((run / "greenup/candidate_greenup.json").read_text()))
    wanted = {row["region_id"] for row in candidates if rejections.get(row["candidate_id"])}
    observed = [row for row in json.loads((run / "timelines/observed_timelines.json").read_text())
                if row["region_id"] in wanted]
    saved_relaxations = relaxed_locations(
        candidates, rejections, saved_regions, observed, detector, policy,
        annual_archive.PixelPartitions(saved["pixel_paths"]), saved["quality_rows"], record_feature,
    )
    relaxations = {cohort: [[row for row in options if region_population[row[0][1]] == cohort]
                            for options in saved_relaxations]
                   for cohort in (external.DEVELOPMENT, "jecam")}
    boundary, disagreement, gap = relaxations[external.DEVELOPMENT]
    additions, sources = new_sources(report, snapshot)
    new_regions, new_strict = [], []
    if sources:
        print("Auditing newly completed batches: " + ", ".join(additions), flush=True)
        with TemporaryDirectory(prefix="aligned_month_", dir=output) as folder:
            scratch = Path(folder)
            regions, qa, observed, pixels = [], [], [], {}
            for source in sources:
                source_regions, source_qa, source_obs, source_pixels = annual_archive.audit_source(
                    source, scratch, 0.8
                )
                regions.extend(source_regions)
                qa.extend(source_qa)
                observed.extend(source_obs)
                pixels.update(source_pixels)
            new_regions = eligible_new_regions(sources, regions, report, saved["pixel_paths"])
            new_pixels = annual_archive.PixelPartitions(pixels)
            new_ids = {region["properties"]["fragment_id"] for region in new_regions}
            daily = [row for row in observed if row["region_id"] in new_ids]
            candidates = quality.detect_greenup(daily, detector)
            _, proposed, audit = seasons.select_season_windows(candidates, [], new_regions, configuration)
            strict, _, reviewed = seasons.qualify_estimated_windows(proposed, audit, daily, configuration)
            training = {region["properties"]["fragment_id"] for region in new_regions
                        if region["properties"].get("source_split", "train") == "train"}
            new_strict = list(complete_locations(
                (row for row in strict if row["region_id"] in training), new_pixels, qa, record_feature
            ))
            training_audit = [row for row in reviewed if row["region_id"] in training]
            new_boundary, new_disagreement, new_gap = relaxed_locations(
                training_audit, {row["candidate_id"]: timing_rejection(row) for row in training_audit},
                new_regions, daily, detector, policy, new_pixels, qa, record_feature,
            )
            boundary.extend(new_boundary)
            disagreement.extend(new_disagreement)
            gap.extend(new_gap)
            new_fixed = list(complete_locations(external.fixed_windows(new_regions), new_pixels, qa))
    else:
        new_fixed = []

    saved_cohorts = saved_locations(run, record_feature, include_jecam=include_jecam)
    development = saved_cohorts[external.DEVELOPMENT]
    fixed, starts, aligned, _ = development
    fixed.update((key, crop) for key, crop, _ in new_fixed)
    merge_locations(starts, aligned, new_strict)
    new_groups = {region["properties"]["partition_group"] for region in new_regions}
    development_reports = compare_rules(development, relaxations[external.DEVELOPMENT], new_groups,
                                        record_matched=record_matched)
    jecam_gap = (compare_rules(saved_cohorts["jecam"], relaxations["jecam"], set(),
                               development=False)[-1] if include_jecam else [])
    return run, additions, new_groups, *development_reports, jecam_gap


def compare_rules(saved, relaxations, new_groups, *, development=True, record_matched=None):
    fixed, starts, aligned, references = saved
    before = counts(fixed, flatten(aligned, references), development=development)
    labels = ("strict", "both_months", "threshold", "gap_25_days")
    reports = []
    for rows, before_label, after_label in zip(relaxations, labels[:-1], labels[1:], strict=True):
        merge_locations(starts, aligned, rows)
        locations = eligible_locations(fixed, flatten(aligned, references), development=development)
        if record_matched is not None and after_label in {"threshold", "gap_25_days"}:
            record_matched(after_label, locations["matched"], starts, references)
        after = {cohort: {(crop, key[0]) for key, crop in rows.items()}
                 for cohort, rows in locations.items()}
        reports.append(summarize(before, after, new_groups, before_label, after_label))
        before = after
    return reports


def flatten(aligned, references):
    return {key: crop for rows in (*aligned.values(), references) for key, crop in rows.items()}


def summarize(before, after, new_groups, before_label="strict", after_label="both_months"):
    rows = []
    for cohort, strict in before.items():
        relaxed = after[cohort]
        for crop in ("carrot", "cabbage", "potato"):
            old = {group for label, group in strict if label == crop}
            new = {group for label, group in relaxed if label == crop}
            rows.append({"cohort": cohort, "crop": crop, f"{before_label}_fields": len(old),
                         f"{after_label}_fields": len(new), "gained_fields": len(new - old),
                         "lost_fields": len(old - new), "net_change": len(new) - len(old),
                         f"new_download_fields_{before_label}": len(old & new_groups),
                         f"new_download_fields_{after_label}": len(new & new_groups)})
    return rows
