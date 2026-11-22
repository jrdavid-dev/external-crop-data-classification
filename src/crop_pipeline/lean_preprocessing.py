"""Publish the 25-day external recipe using verified saved imagery and new batches."""

from collections import defaultdict
from contextlib import ExitStack
import json
from math import floor
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

from shapely.geometry import shape

from crop_pipeline import aligned_month_check as timing
from crop_pipeline import preprocessing_workflow as workflow
from crop_pipeline.analysis_workflow import resolve_stage_pointer
from crop_pipeline.common import save_json
from crop_pipeline.features import experiments
from crop_pipeline.imagery import quality
from crop_pipeline.sources import adaptive, test_cohort
from crop_pipeline.windows import annual_archive, external, json_rows, seasons

RECIPES = ("fixed", "aligned")
COHORTS = ("fixed_full", "fixed_matched", "aligned_full", "aligned_matched")
POLICY = {"maximum_onset_gap_days": 25, "maximum_support_gap_days": 20,
          "adjacent_month_boundary": True, "adjacent_threshold_disagreement": True}


def load_json(path):
    return json.loads(path.read_text())


def baseline_source(root, explicit):
    path = ((root / Path(explicit).expanduser()).resolve() if explicit else
            resolve_stage_pointer(workflow.pointer(root, "features")))
    data = load_json(path)
    if data.get("baseline_features") and not explicit:
        return Path(data["baseline_features"]), path
    return path, None


def confirm_completed_inputs(baseline, current, report):
    """Allow new development receipts while holding existing inputs and test cohorts fixed."""
    dev_campaign = adaptive.root(report).resolve()
    for name in baseline.keys() | current.keys():
        path = Path(name)
        if path.name not in {"selection.json", "completed.json"}:
            continue
        if name in baseline and baseline[name] != current.get(name):
            raise ValueError(f"Saved collection input changed: {path.parent.name}/{path.name}")
        if name not in baseline and path.name == "completed.json" and not path.is_relative_to(dev_campaign):
            raise ValueError(f"New reserved test imagery requires a fresh overlap audit: {path.parent.name}")


def records_at(path, arrays):
    for name, row, span in json_rows.read_rows(path, arrays, container="rows"):
        if span is not None:
            yield name, row


def read_locked_validation(baseline, split):
    groups = {group for group in split["matched_parcel_groups"]
              if split["assignments"][group] == "validation"}
    rows = {name: {} for name in COHORTS}
    for name, row in records_at(baseline / "features_development.json", set(COHORTS)):
        if row["partition"] == "validation" and row["partition_group"] in groups:
            rows[name][external.pixel_key(row)] = row
    if (len(groups) != 69 or any({key[0] for key in rows[name]} != groups
                                for name in ("fixed_matched", "aligned_matched"))
            or rows["fixed_matched"].keys() != rows["aligned_matched"].keys()):
        raise ValueError("The baseline must contain the same 69 locked matched validation fields and pixels")
    return groups, rows


def compact_row(row, properties):
    fields = ("region_id", "partition_group", "crop", "source_crs", "x", "y", "window_start",
              "window_end", "window_status", "season_key", "label_status", "site_cell",
              "test_cohort", "timing_review_status")
    result = {name: row[name] for name in (*fields, *seasons.feature_columns()) if name in row}
    result.update(field_id=row.get("field_id") or row["partition_group"],
                  label_status=row.get("label_status", "reported"),
                  source_dataset=test_cohort.source_role(properties["source_dataset"]),
                  partition=properties["partition"])
    if properties.get("test_cohort"):
        result["test_cohort"] = properties["test_cohort"]
    return result


def new_windows(sources, saved, report, scratch, detector, policy, configuration):
    regions, qa, observations, paths = [], [], [], {}
    for source in sources:
        new_regions, new_qa, new_observations, new_paths = annual_archive.audit_source(source, scratch, .8)
        regions.extend(new_regions)
        qa.extend(new_qa)
        observations.extend(new_observations)
        paths.update(new_paths)
    eligible = timing.eligible_new_regions(sources, regions, report, saved["pixel_paths"])
    for region in eligible:
        p = region["properties"]
        p["partition"] = external.DEVELOPMENT
    training = [r for r in eligible if r["properties"].get("source_split", "train") == "train"]
    ids = {r["properties"]["fragment_id"] for r in training}
    daily = [row for row in observations if row["region_id"] in ids]
    candidates = quality.detect_greenup(daily, detector)
    _, proposals, audit = seasons.select_season_windows(candidates, [], training, configuration)
    strict, _, reviewed = seasons.qualify_estimated_windows(proposals, audit, daily, configuration)
    pixels = annual_archive.PixelPartitions(paths)
    return eligible, pixels, qa, daily, strict, reviewed


def finalize_indices(baseline_windows, regions, saved, report, scratch, detector, policy,
                     configuration, sources):
    props = {r["properties"]["fragment_id"]: r["properties"] for r in regions}
    metadata = {}
    fixed, aligned, references, starts = {}, {}, {}, {}

    def entry(row, span, path):
        key = external.pixel_key(row)
        p = props[row["region_id"]]
        details = metadata.setdefault(row["region_id"], {
            "partition_group": key[0], "partition": p["partition"], "crop": p["crop"],
            "source_dataset": test_cohort.source_role(p["source_dataset"]),
            "test_cohort": p.get("test_cohort", "jecam"),
        })
        return key, ((path, span), details)

    def add_aligned(key, value, start):
        group = key[0]
        if group not in starts or start < starts[group]:
            starts[group], aligned[group] = start, {}
        if start == starts[group]:
            aligned[group].setdefault(key, value)

    print("Reading archived window identities and checking additional timing candidates.", flush=True)
    for recipe, row, span in json_rows.read_rows(baseline_windows, RECIPES):
        if span is None:
            continue
        key, value = entry(row, span, baseline_windows)
        if recipe == "fixed":
            fixed.setdefault(key, value)
        elif row["window_status"] == "reference_supported":
            references.setdefault(key, value)
        else:
            add_aligned(key, value, row["window_start"])

    spool_path = scratch / "additional_windows.jsonl"
    with spool_path.open("wb") as spool:
        def record(recipe, key, row):
            p = props[key[1]]
            compact = compact_row(row, p)
            if "site_cell" not in compact:
                compact["site_cell"] = site_cells[key[1]]
            payload = json.dumps(compact, separators=(",", ":"), allow_nan=False).encode()
            offset = spool.tell()
            spool.write(payload + b"\n")
            _, value = entry(row, (offset, len(payload)), spool_path)
            if recipe == "fixed":
                fixed.setdefault(key, value)
            else:
                add_aligned(key, value, row["window_start"])

        site_cells = {}
        for region in regions:
            point = shape(region["geometry"]).representative_point()
            site_cells[region["properties"]["fragment_id"]] = f"{floor(point.y)},{floor(point.x)}"
        ids = {key for key, p in props.items() if p.get("source_split", "train") == "train"}
        rejections = timing.saved_rejected_candidates(baseline_windows.parent, ids)
        candidates = seasons.identify_candidates(load_json(baseline_windows.parent / "greenup/candidate_greenup.json"))
        wanted = {r["region_id"] for r in candidates if rejections.get(r["candidate_id"])}
        observations = [r for r in load_json(baseline_windows.parent / "timelines/observed_timelines.json")
                        if r["region_id"] in wanted]
        timing.relaxed_locations(candidates, rejections, regions, observations, detector, policy,
                                 annual_archive.PixelPartitions(saved["pixel_paths"]),
                                 saved["quality_rows"], lambda key, row: record("aligned", key, row))
        new_groups = set()
        if sources:
            print("Auditing newly completed development batches.", flush=True)
            added, pixels, qa, daily, strict, reviewed = new_windows(
                sources, saved, report, scratch, detector, policy, configuration
            )
            for region in added:
                p = region["properties"]
                props[p["fragment_id"]] = p
                new_groups.add(p["partition_group"])
                point = shape(region["geometry"]).representative_point()
                site_cells[p["fragment_id"]] = f"{floor(point.y)},{floor(point.x)}"
            list(timing.complete_locations(external.fixed_windows(added), pixels, qa,
                                           lambda key, row: record("fixed", key, row)))
            list(timing.complete_locations(strict, pixels, qa, lambda key, row: record("aligned", key, row)))
            timing.relaxed_locations(reviewed, {r["candidate_id"]: timing.timing_rejection(r) for r in reviewed},
                                     added, daily, detector, policy, pixels, qa,
                                     lambda key, row: record("aligned", key, row))

    indexed = {"fixed": fixed, "aligned": {key: row for parcel in aligned.values() for key, row in parcel.items()}}
    indexed["aligned"].update(references)
    return indexed, props, new_groups


def select_cohorts(indexed, assignments, locked):
    exclusions = {name: external.exclude_shared_development_locations(rows) for name, rows in indexed.items()}
    fixed, aligned = (indexed[name] for name in RECIPES)
    matched = list(external.capped_keys(key for key in fixed if key in aligned))
    keys = {"fixed_full": list(external.capped_keys(fixed)), "aligned_full": list(external.capped_keys(aligned)),
            "fixed_matched": matched, "aligned_matched": matched}
    held_locations = {key[2:] for rows in locked.values() for key in rows}
    selected = {}
    for name, cohort_keys in keys.items():
        source = indexed[name.split("_")[0]]
        cohort = {}
        for key in cohort_keys:
            pointer, details = source[key]
            if details["partition"] == "test":
                if key[0] in assignments:
                    raise ValueError(f"A reserved test field is assigned to development: {key[0]}")
                cohort[key] = (pointer, details)
            elif assignments.get(key[0]) == "train" and key[2:] not in held_locations:
                cohort[key] = (pointer, {**details, "partition": "train"})
        selected[name] = cohort
    if selected["fixed_matched"].keys() != selected["aligned_matched"].keys():
        raise ValueError("Matched export must use identical fixed and aligned pixel locations")
    return selected, exclusions


def write_packages(run, selected, locked, metadata, cohorts, props):
    summaries = {partition: {"rows": {name: [] for name in COHORTS}}
                 for partition in ("development", "test")}
    summaries["test"]["test_cohorts"] = cohorts
    with ExitStack() as stack:
        streams = {partition: stack.enter_context((run / f"features_{partition}.json").open("w"))
                   for partition in summaries}
        inputs = {}
        for partition, stream in streams.items():
            header = {**metadata, "test_cohorts": cohorts} if partition == "test" else metadata
            stream.write(json.dumps(header)[:-1] + ', "rows": {')
        for index, name in enumerate(COHORTS):
            for stream in streams.values():
                stream.write(("," if index else "") + json.dumps(name) + ":[")
            for key, (pointer, details) in selected[name].items():
                path, (offset, size) = pointer
                if path not in inputs:
                    inputs[path] = stack.enter_context(path.open("rb"))
                source = inputs[path]
                source.seek(offset)
                raw = compact_row(json.loads(source.read(size)), props[key[1]])
                raw["partition"] = details["partition"]
                row = experiments.enhance_window(raw, experiments.FEATURE_POLICY)
                partition = "test" if row["partition"] == "test" else "development"
                write_row(streams[partition], summaries[partition]["rows"][name], row)
            for row in locked[name].values():
                write_row(streams["development"], summaries["development"]["rows"][name], row)
            for stream in streams.values():
                stream.write("]")
        for stream in streams.values():
            stream.write("}}")
    return summaries["development"], summaries["test"]


def write_row(stream, summary, row):
    if summary:
        stream.write(",")
    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False))
    summary.append({key: row[key] for key in ("partition_group", "partition", "crop", "source_dataset", "test_cohort")
                    if key in row})


def window_coverage(development, test):
    coverage = {}
    for recipe in RECIPES:
        groups = defaultdict(set)
        for package in (development, test):
            for row in package["rows"][f"{recipe}_full"]:
                groups[f"{row['partition']}/{row['crop']}"].add(row["partition_group"])
        coverage[recipe] = {name: len(fields) for name, fields in sorted(groups.items())}
    return coverage


def run(repository, baseline_features=None):
    root, report, output = workflow.paths(repository)
    baseline_handoff, published = baseline_source(root, baseline_features)
    current = workflow.collection_signature(report)
    if published and load_json(published.parent / "collection_snapshot.json")["campaign_metadata"] == current:
        workflow.verify(published, "features")
        return published
    workflow.verify(baseline_handoff, "features")
    baseline = baseline_handoff.parent
    split = load_json(baseline / "development_split.json")
    validation, locked = read_locked_validation(baseline, split)
    saved = load_json(baseline / "quality.json")
    snapshot = load_json(baseline / "collection_snapshot.json")["campaign_metadata"]
    confirm_completed_inputs(snapshot, current, report)
    additions, sources = timing.new_sources(report, snapshot)
    detector_policy = load_json(baseline / "seasons/automatic_timing_policy.json")
    detector, policy = detector_policy["detector_parameters"], detector_policy["quality_policy"]
    if policy["maximum_onset_gap_days"] != 20 or policy["maximum_support_gap_days"] != 20:
        raise ValueError("Expected the saved strict 20-day timing baseline")
    review = baseline / "seasons/review"
    frozen = review / "frozen_configuration.json"
    configuration = load_json(frozen if frozen.exists() else review / "frozen_configuration.template.json")
    configuration.update(cycle_selection_policy="earliest_complete", automatic_timing_policy=policy)
    with TemporaryDirectory(prefix="lean_25d_", dir=output) as temporary:
        indexed, props, added_groups = finalize_indices(
            baseline / "windows.json", saved["regions"], saved, report, Path(temporary),
            detector, policy, configuration, sources
        )
        assignments = dict(split["assignments"])
        collected = adaptive.assignments(report)
        for group in added_groups:
            assignments.setdefault(group, collected[group])
        selected, exclusions = select_cohorts(indexed, assignments, locked)
        if workflow.collection_signature(report) != current:
            raise RuntimeError("Collection changed during preparation; rerun with the new snapshot")
        destination = Path(mkdtemp(prefix="external_25d_", dir=output))
        provenance = {"baseline_features": str(baseline_handoff), "timing_policy": POLICY,
                      "new_completed_development_batches": additions, "validation_fields": len(validation)}
        windows = destination / "windows.json"
        save_json(windows, {**provenance, "selected": {
            name: [list(key) for key in rows] + [list(key) for key in locked[name]]
            for name, rows in selected.items()}})
        metadata = {**provenance, "window_sha256": seasons.file_sha256(windows),
                    "quality": str(baseline / "quality.json"), "fixed_start_month": external.FIXED_START_MONTH}
        cohorts = load_json(baseline / "features_test.json")["test_cohorts"]
        development, test = write_packages(destination, selected, locked, metadata, cohorts, props)
        save_json(destination / external.SPLIT_FILENAME, {
            **split, "assignments": assignments, "locked_matched_validation": sorted(validation),
            "matched_parcel_groups": sorted({row["partition_group"] for row in development["rows"]["fixed_matched"]}),
            "window_sha256": metadata["window_sha256"],
        })
        save_json(destination / "collection_snapshot.json", {"campaign_metadata": current, "report_root": str(report)})
        save_json(destination / "development_overlap_exclusions.json", exclusions)
        save_json(destination / "window_coverage.json", window_coverage(development, test))
        handoff = destination / "features.json"
        save_json(handoff, metadata)
        workflow.write_feature_reports(handoff, development, test)
        if workflow.collection_signature(report) != current:
            raise RuntimeError("Collection changed while writing features; previous pointers were preserved")
        workflow.publish(root, "windows", windows, [windows, destination / "collection_snapshot.json",
                                                      Path(__file__), Path(timing.__file__)])
        return workflow.publish(root, "features", handoff, [*destination.iterdir(),
                                                           Path(__file__), Path(timing.__file__)])
