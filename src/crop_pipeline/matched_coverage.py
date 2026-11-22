"""Read matched coverage and batch contributions from a completed feature run."""

from collections import Counter
import json
from pathlib import Path

from crop_pipeline.analysis_workflow import resolve_stage_pointer
from crop_pipeline.preprocessing_workflow import paths, pointer
from crop_pipeline.sources import adaptive
from crop_pipeline.windows import json_rows, seasons

BATCHES = {
    "Carrot · EuroCrops": "eurocrops_matched_top_up_carrot_train_validation_0002",
    "Cabbage · FranceCrops": "francecrops_matched_top_up_cabbage_train_validation_0002",
}
COHORTS = {"fixed_full", "fixed_matched", "aligned_full", "aligned_matched"}


def _batch_selection(campaign, label):
    selections = list(campaign.glob(f"rounds/*/{label}/selection.json"))
    if len(selections) > 1:
        raise ValueError(f"Multiple collection selections found for {label}")
    return selections[0] if selections else None


def report(repository, feature_handoff=None):
    root, collection, _ = paths(repository)
    handoff = ((root / Path(feature_handoff).expanduser()).resolve() if feature_handoff
               else resolve_stage_pointer(pointer(root, "features")))
    run = handoff.parent
    coverage = [
        {key: row[key] for key in ("crop", "partition", "usable_parcel_groups",
                                  "usable_pixels", "parcel_shortfall", "pixel_shortfall")}
        for row in seasons.read_table(run / "usable_samples.csv")
        if row["cohort"] == "fixed_matched"
    ]
    split = json.loads((run / "development_split.json").read_text())
    snapshot = json.loads((run / "collection_snapshot.json").read_text())["campaign_metadata"]
    matched = set(split["matched_parcel_groups"])
    campaign = adaptive.root(collection)
    batches = []
    for crop, label in BATCHES.items():
        selection = _batch_selection(campaign, label)
        if selection is None:
            batches.append((crop, "No local batch selection", None, set()))
            continue
        groups = set(json.loads(selection.read_text())["groups"])
        completed = selection.with_name("completed.json")
        included = all(path.is_file() and snapshot.get(str(path.resolve())) == seasons.file_sha256(path)
                       for path in (selection, completed))
        if included:
            status = "Included in this saved run"
        else:
            status = ("Completed after this snapshot" if completed.is_file()
                      else "Collection pending for this snapshot")
        batches.append((crop, status, len(groups), groups & matched if included else set()))

    wanted = set().union(*(groups for _, _, _, groups in batches))
    pixels = Counter()
    if wanted:
        for cohort, row, span in json_rows.read_rows(
            run / "features_development.json", COHORTS, container="rows"
        ):
            if cohort == "fixed_matched" and span is not None and row["partition_group"] in wanted:
                pixels[row["partition_group"]] += 1
    contributions = []
    for batch, status, candidates, groups in batches:
        for partition in ("train", "validation"):
            selected = [group for group in groups if split["assignments"][group] == partition]
            contributions.append({
                "batch": batch, "snapshot_status": status, "candidate_parcels": candidates,
                "partition": partition, "matched_parcels": len(selected),
                "matched_pixels": sum(pixels[group] for group in selected),
            })
    return run, coverage, contributions


def _unshared_matches(indexed, baseline_windows):
    from crop_pipeline.sources.sampling import pixel_location
    from crop_pipeline.windows import external

    for rows in indexed.values():
        external.exclude_shared_development_locations(rows)
    locations = {
        name: {key[2:]: key[0] for key in rows}
        for name, rows in indexed.items()
    }
    collisions = {name: set() for name in indexed}
    for recipe, row, span in json_rows.read_rows(baseline_windows, indexed):
        if span is None or row["partition"] == "test":
            continue
        location = pixel_location(row)
        owner = locations[recipe].get(location)
        if owner is not None and owner != row["partition_group"]:
            collisions[recipe].add(location)
    for recipe, rows in indexed.items():
        for key in list(rows):
            if key[2:] in collisions[recipe]:
                del rows[key]
    return external.capped_keys(key for key in indexed["fixed"] if key in indexed["aligned"])


def check_top_ups(repository):
    """Audit completed new batches in isolation against the saved development pool."""
    from tempfile import TemporaryDirectory

    from crop_pipeline.cabbage_test_check import discover
    from crop_pipeline.common import save_json
    from crop_pipeline.sources import pilot_sources, test_cohort
    from crop_pipeline.windows import external
    from crop_pipeline.windows.workflow import WindowAnalysis

    root, collection, output = paths(repository)
    run = resolve_stage_pointer(pointer(root, "features")).parent
    baseline = run / "windows.json"
    if not baseline.is_file():
        raise FileNotFoundError(f"Saved baseline windows are required: {baseline}")
    snapshot = json.loads((run / "collection_snapshot.json").read_text())["campaign_metadata"]
    baseline_groups = set(json.loads((run / "development_split.json").read_text())["assignments"])
    campaign = adaptive.root(collection)
    summaries, owners, manifests = {}, {}, []
    for batch, label in BATCHES.items():
        selection = _batch_selection(campaign, label)
        summary = {"batch": batch, "status": "No local batch selection", "candidate_parcels": 0,
                   "matched_parcels": None, "matched_pixels": None}
        summaries[batch] = summary
        if selection is None:
            continue
        groups = set(json.loads(selection.read_text())["groups"])
        summary["candidate_parcels"] = len(groups)
        completed = selection.with_name("completed.json")
        state = adaptive.read_json(selection.with_name("state.json"), {}).get("status")
        if state != "completed" or not completed.is_file():
            summary["status"] = f"Awaiting local collection ({state or 'prepared'})"
            continue
        if snapshot.get(str(completed.resolve())) == seasons.file_sha256(completed):
            summary["status"] = "Already included in saved run"
            continue
        summary.update(status="Checked new imagery", matched_parcels=0, matched_pixels=0)
        for group in groups - baseline_groups:
            if group in owners:
                raise ValueError(f"Parcel appears in multiple top-ups: {group}")
            owners[group] = batch
        for item in json.loads(completed.read_text())["handoffs"]:
            manifest = Path(item["path"])
            if seasons.file_sha256(manifest) != item["sha256"]:
                raise ValueError(f"Completed batch handoff changed: {manifest}")
            manifests.append(manifest)

    parcel_rows = []
    if not manifests:
        return run, list(summaries.values()), parcel_rows
    sources = pilot_sources.load_sources(list(dict.fromkeys(manifests)))
    reserved = [source for cohort in discover(collection) for source in cohort["sources"]]
    _, excluded = test_cohort.exclude_overlap(test_cohort.source_footprints(sources),
                                             test_cohort.source_footprints(reserved))
    with TemporaryDirectory(prefix="matched_top_up_", dir=output.parent) as folder:
        scratch = Path(folder)
        analysis = WindowAnalysis(sources, [], scratch, collection, adaptive.COLLECTION_NAME)
        analysis.audit()
        eligible = owners.keys() - excluded
        regions = [
            {**region, "properties": {**region["properties"], "partition": external.DEVELOPMENT}}
            for region in analysis.regions
            if region["properties"]["partition_group"] in eligible
        ]
        ids = {region["properties"]["fragment_id"] for region in regions}
        if not ids:
            return run, list(summaries.values()), parcel_rows
        quality = scratch / "quality.json"
        save_json(quality, {
            "regions": regions, "sources": sources,
            "quality_rows": [row for row in analysis.quality_rows if row["region_id"] in ids],
            "composite_rows": [row for row in analysis.composite_rows if row["region_id"] in ids],
            "pixel_paths": {key: str(path) for key, path in analysis.pixel_rows.paths.items() if key in ids},
        })
        windows = external.select(quality, collection)
        indexed, _ = external.index_windows(windows)
        counts = Counter(key[0] for key in _unshared_matches(indexed, baseline))
        for group, count in sorted(counts.items()):
            batch = owners[group]
            summaries[batch]["matched_parcels"] += 1
            summaries[batch]["matched_pixels"] += count
            parcel_rows.append({"batch": batch, "parcel_group": group, "matched_pixels": count})
    return run, list(summaries.values()), parcel_rows
