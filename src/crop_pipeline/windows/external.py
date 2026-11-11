"""Local quality, window and feature handoffs for the external experiment."""

from collections import Counter
from datetime import date
import json
from pathlib import Path
from tempfile import mkdtemp
from math import floor

from shapely.geometry import shape

from crop_pipeline.common import save_json
from crop_pipeline.sources import adaptive, sampling, test_cohort
from crop_pipeline.windows import annual_archive, seasons
from crop_pipeline.windows.workflow import WindowAnalysis


def snapshot(report_folder):
    development = [s for s in test_cohort.completed_sources(report_folder)
                   if test_cohort.source_role(s["source_name"]) in test_cohort.DEVELOPMENT]
    if {test_cohort.source_role(s["source_name"]) for s in development} != test_cohort.DEVELOPMENT:
        raise ValueError("Complete Sen4AgriNet and FranceCrops imagery batches first")
    test = test_cohort.completed_sources(report_folder, test_cohort.TEST_POLICY)
    if {test_cohort.source_role(s["source_name"]) for s in test} != {test_cohort.TEST}:
        raise ValueError("Complete the dedicated CropHarvest JECAM collection first")
    _, overlaps = test_cohort.exclude_overlap(
        test_cohort.source_footprints(test), test_cohort.source_footprints(development)
    )
    return development + test, overlaps


def reference_profile(regions, quality_rows):
    """Count distinct parcels and summarize the bin evidence available per cohort."""
    profiles = {}
    by_region = {}
    for region in regions:
        p = region["properties"]
        key = p["source_dataset"], p["source_year"], p["crop"], p["partition"]
        profile = profiles.setdefault(key, {
            "groups": set(), "sampled_pixels": 0, "available_bins": 0,
            "quality_pass_bins": 0, "clear_pixel_dates": 0,
        })
        profile["groups"].add(p["partition_group"])
        profile["sampled_pixels"] += p["selected_pixels"]
        by_region[p["fragment_id"]] = profile
    for row in quality_rows:
        profile = by_region.get(row["region_id"])
        if profile is None:
            continue
        profile["available_bins"] += bool(row.get("image_path"))
        profile["quality_pass_bins"] += row["status"] == "quality_pass"
        profile["clear_pixel_dates"] += row.get("clear_pixel_dates", 0)
    return [
        {"source": source, "year": year, "crop": crop, "partition": partition,
         "parcels": len(profile["groups"]),
         **{k: v for k, v in profile.items() if k != "groups"}}
        for (source, year, crop, partition), profile in sorted(profiles.items())
    ]


def audit(report_folder, output_root):
    sources, overlaps = snapshot(report_folder)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix="external_", dir=root))
    analysis = WindowAnalysis(sources, [], run, report_folder, adaptive.COLLECTION_NAME)
    analysis.audit()
    assignment = adaptive.assignments(report_folder)
    regions = []
    for region in analysis.regions:
        p = region["properties"]
        group = p["partition_group"]
        source = test_cohort.source_role(p["source_dataset"])
        is_test = source == test_cohort.TEST
        partition = "test" if is_test else assignment[group]
        if group not in overlaps and (is_test or partition in {"train", "validation"}):
            regions.append({**region, "properties": {**p, "partition": partition, "source_dataset": source}})
    seasons.write_records(run / "reference_profile.csv", reference_profile(regions, analysis.quality_rows))
    handoff = run / "quality.json"
    save_json(handoff, {
        "regions": regions, "quality_rows": analysis.quality_rows,
        "composite_rows": analysis.composite_rows,
        "pixel_paths": {key: str(path) for key, path in analysis.pixel_rows.paths.items()},
        "sources": sources, "overlap_exclusions": sorted(overlaps),
        "mask_loss": "unavailable for legacy archives without pre-mask counts",
    })
    return handoff


def fixed_windows(regions):
    for region in regions:
        p = region["properties"]
        year = int(p["source_year"])
        start = date(year, 1, 1)
        yield {
            **p, "field_id": p["partition_group"], "region_id": p["fragment_id"], "season_key": p["fragment_id"] + "|fixed",
            "window_start": start.isoformat(),
            "window_end": seasons.add_calendar_months(start, 4).isoformat(),
            "window_status": "fixed_calendar", "label_status": "reported",
        }


def select(quality_handoff, report_folder):
    quality_handoff = Path(quality_handoff)
    saved = json.loads(quality_handoff.read_text())
    run = quality_handoff.parent
    analysis = WindowAnalysis(saved["sources"], [], run, report_folder, adaptive.COLLECTION_NAME)
    analysis.regions = saved["regions"]
    ids = {r["properties"]["fragment_id"] for r in analysis.regions}
    analysis.quality_rows = saved["quality_rows"]
    analysis.pixel_rows = annual_archive.PixelPartitions(saved["pixel_paths"])
    analysis.timeline_folder = run / "timelines"
    analysis.composite_rows = [r for r in saved["composite_rows"] if r["region_id"] in ids]
    analysis.example_regions = ids.intersection(
        next(row["region_id"] for row in analysis.composite_rows if row["crop"] == crop)
        for crop in {row["crop"] for row in analysis.composite_rows}
    )
    observed_path = analysis.timeline_folder / "observed_timelines.json"
    if observed_path.is_file():
        observed = json.loads(observed_path.read_text())
        save_json(observed_path, [row for row in observed if row["region_id"] in ids])
    analysis.select(candidates_only=True, excluded_sensitivity_groups={
        r["properties"]["partition_group"] for r in analysis.regions
        if r["properties"]["partition"] == "test"
    })
    fixed, _, losses, _ = annual_archive.build_partitioned_features(
        fixed_windows(analysis.regions), analysis.pixel_rows, analysis.quality_rows, .8, 0,
    )
    aligned = analysis.feature_rows + analysis.estimated_feature_rows
    partitions = {r["properties"]["partition_group"]: r["properties"]["partition"] for r in analysis.regions}
    sites = {}
    for region in analysis.regions:
        point = shape(region["geometry"]).representative_point()
        sites[region["properties"]["fragment_id"]] = f"{floor(point.y)},{floor(point.x)}"
    for row in fixed + aligned:
        row["site_cell"] = sites[row["region_id"]]
        row["partition"] = partitions[row["partition_group"]]
        if not row.get("field_id"):
            row["field_id"] = row["partition_group"]
    seasons.write_records(run / "fixed_losses.csv", losses)
    handoff = run / "windows.json"
    save_json(handoff, {"fixed": fixed, "aligned": aligned, "quality": str(quality_handoff)})
    return handoff


def pixel_key(row):
    return row["partition_group"], row["region_id"], *sampling.pixel_location(row)


def capped(rows):
    counts = Counter()
    result = []
    for key, row in sorted(rows.items(), key=lambda item: sampling.stable_rank(str(item[0]))):
        group = row["partition_group"]
        if counts[group] < adaptive.DEFAULT_POLICY["pixels_per_parcel"]:
            counts[group] += 1
            result.append(row)
    return result


def export_features(window_handoff):
    from crop_pipeline.features import experiments

    path = Path(window_handoff)
    data = json.loads(path.read_text())
    indexed = {}
    for name in ("fixed", "aligned"):
        indexed[name] = {}
        for row in data[name]:
            indexed[name].setdefault(pixel_key(row), row)
    common = indexed["fixed"].keys() & indexed["aligned"].keys()
    report = {"window_sha256": seasons.file_sha256(path), "quality": data["quality"]}
    for cohort in ("full", "matched"):
        for name, rows in indexed.items():
            selected = rows if cohort == "full" else {key: rows[key] for key in common}
            report[f"{name}_{cohort}"] = [
                experiments.enhance_window(row, experiments.FEATURE_POLICY)
                for row in capped(selected)
            ]
    handoff = path.parent / "features.json"
    coverage = {}
    for name in ("fixed", "aligned"):
        groups = {}
        for row in report[f"{name}_full"]:
            groups.setdefault((row["partition"], row["crop"]), set()).add(row["partition_group"])
        coverage[name] = {f"{partition}/{crop}": len(ids)
                          for (partition, crop), ids in sorted(groups.items())}
    save_json(path.parent / "window_coverage.json", coverage)
    metadata = {key: report[key] for key in ("quality", "window_sha256")}
    for split in ("development", "test"):
        save_json(path.parent / f"features_{split}.json", {
            **metadata,
            "rows": {
                name: [row for row in rows if (row["partition"] == "test") == (split == "test")]
                for name, rows in report.items() if name not in metadata
            },
        })
    save_json(handoff, metadata)
    return handoff
