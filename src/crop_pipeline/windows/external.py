"""Local quality, window and feature handoffs for the external experiment."""

from collections import Counter
from datetime import date
import json
from pathlib import Path
from tempfile import mkdtemp
from math import floor

from shapely.geometry import shape

from crop_pipeline.common import iter_json_object, save_json
from crop_pipeline.sources import adaptive, sampling, test_cohort
from crop_pipeline.windows import annual_archive, seasons
from crop_pipeline.windows.workflow import WindowAnalysis

FIXED_START_MONTH = 5


def snapshot(report_folder):
    development = [s for s in test_cohort.completed_sources(report_folder)
                   if test_cohort.source_role(s["source_name"]) in test_cohort.DEVELOPMENT]
    if not test_cohort.REQUIRED_DEVELOPMENT.issubset(
        test_cohort.source_role(s["source_name"]) for s in development
    ):
        raise ValueError("Complete Sen4AgriNet and FranceCrops imagery batches first")
    try:
        test = test_cohort.completed_sources(report_folder, test_cohort.TEST_POLICY)
    except FileNotFoundError as error:
        raise FileNotFoundError("Import a completed Notebook 04 JECAM bundle or finish Notebook 04 locally") from error
    if {test_cohort.source_role(s["source_name"]) for s in test} != {test_cohort.TEST}:
        raise ValueError("Complete the dedicated CropHarvest JECAM collection first")
    eurocrops = [s for s in development if test_cohort.source_role(s["source_name"]) == "eurocrops"]
    if eurocrops:
        other = [s for s in development if s not in eurocrops]
        _, collision = test_cohort.exclude_overlap(
            test_cohort.source_footprints(eurocrops), test_cohort.source_footprints(other)
        )
        if collision:
            raise ValueError(f"{len(collision)} EuroCrops parcels overlap existing development parcels")
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


def supplemental_inputs(report_folder, primary_sources):
    from crop_pipeline.cabbage_test_check import discover

    sources = []
    occupied = list(test_cohort.source_footprints(primary_sources))
    occupied_groups = {item["properties"]["partition_group"] for item in occupied}
    exclusions = {}
    supplemental_batches = {}
    for cohort in discover(report_folder):
        name = cohort["cohort"]
        if name == "jecam":
            continue
        supplemental_batches[name] = cohort["batches"]
        eligible, exclusions[name] = test_cohort.exclude_overlap(
            test_cohort.source_footprints(cohort["sources"]), occupied)
        exclusions[name].update(item["properties"]["partition_group"] for item in eligible
                                if item["properties"]["partition_group"] in occupied_groups)
        eligible = [item for item in eligible
                    if item["properties"]["partition_group"] not in exclusions[name]]
        occupied.extend(eligible)
        occupied_groups.update(item["properties"]["partition_group"] for item in eligible)
        print(f"{name}: {len(eligible)} test footprints, {len(exclusions[name])} overlap exclusions")
        sources.extend({**source, "test_cohort": name,
                        "excluded_test_groups": sorted(exclusions[name])}
                       for source in cohort["sources"])
    return sources, exclusions, supplemental_batches


def audit(report_folder, output_root):
    sources, overlaps = snapshot(report_folder)
    supplements, exclusions, supplemental_batches = supplemental_inputs(report_folder, sources)
    sources = sources + supplements
    exclusions["jecam"] = overlaps
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
        cohort = p.get("test_cohort") or ("jecam" if source == test_cohort.TEST else None)
        is_test = cohort is not None
        partition = "test" if is_test else assignment[group]
        if group not in exclusions.get(cohort, ()) and (is_test or partition in {"train", "validation"}):
            metadata = {"test_cohort": cohort} if is_test else {}
            regions.append({**region, "properties": {**p, **metadata,
                            "partition": partition, "source_dataset": source}})
    eligible_ids = {region["properties"]["fragment_id"] for region in regions}
    quality_rows = [row for row in analysis.quality_rows if row["region_id"] in eligible_ids]
    seasons.write_records(run / "reference_profile.csv", reference_profile(regions, quality_rows))
    overlap_report = run / "jecam_overlap.json"
    save_json(overlap_report, {"excluded_parcels": sorted(overlaps)})
    save_json(run / "test_overlap_exclusions.json",
              {name: sorted(groups) for name, groups in exclusions.items()})
    print(f"Excluded {len(overlaps)} overlapping JECAM parcels. Report: {overlap_report}")
    handoff = run / "quality.json"
    save_json(handoff, {
        "regions": regions, "quality_rows": quality_rows,
        "composite_rows": [row for row in analysis.composite_rows if row["region_id"] in eligible_ids],
        "pixel_paths": {key: str(path) for key, path in analysis.pixel_rows.paths.items() if key in eligible_ids},
        "sources": sources, "overlap_exclusions": sorted(overlaps),
        "supplemental_batches": supplemental_batches,
        "mask_loss": {
            "intervals_with_pre_mask_counts": sum(row.get("pre_mask_count") is not None for row in quality_rows),
            "intervals_without_pre_mask_counts": sum(row.get("pre_mask_count") is None for row in quality_rows),
            "note": "legacy archives without pre-mask counts report loss as unavailable, not zero",
        },
    })
    return handoff


def fixed_windows(regions):
    for region in regions:
        p = region["properties"]
        year = int(p["source_year"])
        start = date(year, FIXED_START_MONTH, 1)
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
    test_cohorts = {r["properties"]["fragment_id"]: r["properties"]
                    for r in analysis.regions if r["properties"]["partition"] == "test"}
    sites = {}
    for region in analysis.regions:
        point = shape(region["geometry"]).representative_point()
        sites[region["properties"]["fragment_id"]] = f"{floor(point.y)},{floor(point.x)}"
    for row in fixed + aligned:
        row["site_cell"] = sites[row["region_id"]]
        row["partition"] = partitions[row["partition_group"]]
        if row["partition"] == "test":
            reference = test_cohorts[row["region_id"]]
            row["test_cohort"] = reference.get("test_cohort", "jecam")
            row["source_dataset"] = reference["source_dataset"]
        if not row.get("field_id"):
            row["field_id"] = row["partition_group"]
    seasons.write_records(run / "fixed_losses.csv", losses)
    handoff = run / "windows.json"
    save_json(handoff, {"fixed": fixed, "aligned": aligned, "quality": str(quality_handoff),
                        "test_cohorts": ["jecam", *saved.get("supplemental_batches", {})],
                        "fixed_start_month": FIXED_START_MONTH})
    return handoff


# Fields read by pixel_key, exclude_shared_development_locations and capped.
_SELECTION_FIELDS = ("partition_group", "region_id", "partition", "crop", "source_dataset", "source_crs", "x", "y")


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


def exclude_shared_development_locations(rows):
    """Drop ambiguous ground coordinates without moving any parcel assignment."""
    groups_by_location = {}
    for row in rows.values():
        if row["partition"] in {"train", "validation"}:
            groups_by_location.setdefault(sampling.pixel_location(row), set()).add(row["partition_group"])
    shared = {location for location, groups in groups_by_location.items() if len(groups) > 1}
    retained, exclusions = {}, Counter()
    for key, row in rows.items():
        if row["partition"] in {"train", "validation"} and sampling.pixel_location(row) in shared:
            exclusions[row["partition"], row["source_dataset"], row["crop"], row["partition_group"]] += 1
        else:
            retained[key] = row
    return retained, {
        "shared_locations": len(shared),
        "excluded": [
            {"partition": partition, "source": source, "crop": crop, "parcel_group": group, "pixel_rows": count}
            for (partition, source, crop, group), count in sorted(exclusions.items())
        ],
    }


def export_features(window_handoff):
    from crop_pipeline.features import experiments

    path = Path(window_handoff)
    # Two streaming passes keep memory bounded by the capped selection, not the
    # full window handoff. Pass 1 holds only the fields used for de-duplication,
    # shared-location exclusion, matching and capping; pass 2 loads selected rows.
    names = ("fixed", "aligned")
    indexed = {name: {} for name in names}
    data = {}
    positions = Counter()
    strings = {}
    for name, row in iter_json_object(path, arrays=names):
        if name not in names:
            data[name] = row
            continue
        record = {field: strings.setdefault(row[field], row[field]) if isinstance(row[field], str) else row[field]
                  for field in _SELECTION_FIELDS if field in row}
        record["_position"] = positions[name]
        positions[name] += 1
        indexed[name].setdefault(pixel_key(record), record)
    exclusions = {}
    for name in names:
        indexed[name], exclusions[name] = exclude_shared_development_locations(indexed[name])
    common = indexed["fixed"].keys() & indexed["aligned"].keys()
    selections = {}
    for cohort in ("full", "matched"):
        for name, rows in indexed.items():
            selected = rows if cohort == "full" else {key: rows[key] for key in common}
            selections[name, cohort] = [record["_position"] for record in capped(selected)]
    del indexed, strings
    wanted = {name: {i for (window, _), chosen in selections.items() if window == name for i in chosen}
              for name in names}
    loaded = {name: {} for name in names}
    positions = Counter()
    for name, row in iter_json_object(path, arrays=names):
        if name in names:
            if positions[name] in wanted[name]:
                loaded[name][positions[name]] = row
            positions[name] += 1
    report = {"window_sha256": seasons.file_sha256(path), "quality": data["quality"],
              "fixed_start_month": data.get("fixed_start_month", 1)}
    for cohort in ("full", "matched"):
        for name in names:
            report[f"{name}_{cohort}"] = [
                experiments.enhance_window(loaded[name][position], experiments.FEATURE_POLICY)
                for position in selections[name, cohort]
            ]
    handoff = path.parent / "features.json"
    save_json(path.parent / "development_overlap_exclusions.json", exclusions)
    coverage = {}
    for name in ("fixed", "aligned"):
        groups = {}
        for row in report[f"{name}_full"]:
            groups.setdefault((row["partition"], row["crop"]), set()).add(row["partition_group"])
        coverage[name] = {f"{partition}/{crop}": len(ids)
                          for (partition, crop), ids in sorted(groups.items())}
    save_json(path.parent / "window_coverage.json", coverage)
    metadata = {key: report[key] for key in ("quality", "window_sha256", "fixed_start_month")}
    for split in ("development", "test"):
        save_json(path.parent / f"features_{split}.json", {
            **metadata,
            **({"test_cohorts": data.get("test_cohorts", ["jecam"])} if split == "test" else {}),
            "rows": {
                name: [row for row in rows if (row["partition"] == "test") == (split == "test")]
                for name, rows in report.items() if name not in metadata
            },
        })
    save_json(handoff, metadata)
    return handoff
