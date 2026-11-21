"""Safe notebook handoffs and coverage reporting for external preprocessing.

Only new analysis outputs and navigation pointers are written. Collection state,
source selection, imagery rules and external model feature schemas are unchanged.
"""

from collections import Counter
import json
from pathlib import Path

from crop_pipeline.analysis_workflow import resolve_stage_pointer, write_stage_pointer
from crop_pipeline.common import iter_json_object, save_json
from crop_pipeline.output_layout import refresh
from crop_pipeline.sources import adaptive
from crop_pipeline.windows import external, seasons

STAGES = ("quality", "windows", "features")
CROPS = ("carrot", "cabbage", "potato")
PARTITIONS = ("train", "validation", "test")


def paths(repository):
    root = Path(repository).resolve()
    settings_path = root / "settings.local.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    report = (root / Path(settings.get("report_root", "data/pipeline_work/inventory")).expanduser()).resolve()
    return root, report, report.parent / "analysis/external"


def collection_signature(report):
    """Fingerprint saved campaign metadata, including additions and pending states."""
    campaign_root = Path(report).parent / adaptive.COLLECTION_NAME
    return {str(p.resolve()): seasons.file_sha256(p)
            for p in sorted(campaign_root.rglob("*.json")) if p.is_file()}


def pointer(root, stage):
    return Path(root) / "pipeline_outputs/preprocessing" / f"current_{stage}.json"


def pending(root, stage):
    for name in STAGES[STAGES.index(stage):]:
        write_stage_pointer(pointer(root, name), status="pending",
                            reason=f"Preprocessing {stage} has not completed this attempt")


def record_path(handoff):
    return Path(handoff).parent / (Path(handoff).stem + "_stage_receipt.json")


def publish(root, stage, handoff, files):
    handoff = Path(handoff).resolve()
    receipt = {"kind": "external_preprocessing_stage", "stage": stage,
               "files": {str(Path(p).resolve()): seasons.file_sha256(p)
                         for p in [*files, __file__, external.__file__]}}
    save_json(record_path(handoff), receipt)
    refresh(root)
    write_stage_pointer(pointer(root, stage), handoff)
    return handoff


def verify(handoff, stage):
    handoff = Path(handoff).resolve()
    receipt = record_path(handoff)
    if not receipt.is_file():
        raise ValueError("Use a handoff from the new preprocessing notebooks; start a new imagery audit")
    data = json.loads(receipt.read_text())
    if data.get("kind") != "external_preprocessing_stage" or data.get("stage") != stage:
        raise ValueError("Preprocessing stage receipt does not match its handoff")
    if str(handoff) not in data["files"]:
        raise ValueError("Preprocessing receipt does not include its handoff")
    for path, digest in data["files"].items():
        if not Path(path).is_file() or seasons.file_sha256(path) != digest:
            raise ValueError(f"Saved preprocessing output changed: {path}")
    return handoff


def check_snapshot(handoff, report):
    saved = json.loads((Path(handoff).parent / "collection_snapshot.json").read_text())
    if saved["campaign_metadata"] != collection_signature(report):
        raise RuntimeError("Collection changed since the imagery audit. Finish collection and start a new audit; existing outputs remain saved.")


def quality_counts(data):
    rows = []
    for partition in PARTITIONS:
        for crop in CROPS:
            regions = [r["properties"] for r in data["regions"]
                       if r["properties"]["partition"] == partition and r["properties"]["crop"] == crop]
            ids = {r["fragment_id"] for r in regions}
            outcomes = Counter(r["status"] for r in data["quality_rows"] if r["region_id"] in ids)
            rows.append({"partition": partition, "crop": crop,
                         "input_parcel_groups": len({r["partition_group"] for r in regions}),
                         "selected_pixels_across_fragments": sum(r["selected_pixels"] for r in regions),
                         "audited_tile_intervals": sum(outcomes.values()),
                         "quality_pass_tile_intervals": outcomes["quality_pass"],
                         "other_tile_intervals": sum(v for k, v in outcomes.items() if k != "quality_pass")})
    return rows


LOSS_FIELDS = ("pre_mask_count", "cloud_shadow_count", "other_rejection_count", "clear_pixel_dates")


def _loss_row(rows):
    counted = [r for r in rows if r.get("pre_mask_count") is not None]
    totals = {field: sum(int(r.get(field) or 0) for r in counted) for field in LOSS_FIELDS}
    acquired = totals["pre_mask_count"]
    return {"audited_tile_intervals": len(rows), "intervals_without_pre_mask_counts": len(rows) - len(counted),
            "pixel_dates_with_acquisition": acquired, "cloud_or_shadow_pixel_dates": totals["cloud_shadow_count"],
            "other_rejected_pixel_dates": totals["other_rejection_count"], "clear_pixel_dates": totals["clear_pixel_dates"],
            "cloud_or_shadow_share": round(totals["cloud_shadow_count"] / acquired, 4) if acquired else None,
            "clear_share": round(totals["clear_pixel_dates"] / acquired, 4) if acquired else None,
            "quality_pass_tile_intervals": sum(r["status"] == "quality_pass" for r in rows)}


def observation_loss(data):
    """Pixel-date losses before compositing, by partition/crop and by calendar half-month.

    A pixel-date is one selected parcel pixel on one Sentinel-2 acquisition date. Rows from
    legacy archives without pre-mask counts are counted separately, never as zero loss.
    """
    partitions = {r["properties"]["fragment_id"]: r["properties"]["partition"] for r in data["regions"]}
    by_crop, by_half_month = {}, {}
    for row in data["quality_rows"]:
        by_crop.setdefault((partitions.get(row["region_id"], "unassigned"), row["crop"]), []).append(row)
        by_half_month.setdefault(row["start"][5:10], []).append(row)
    crop_rows = [{"partition": partition, "crop": crop, **_loss_row(by_crop[(partition, crop)])}
                 for partition in PARTITIONS for crop in CROPS if (partition, crop) in by_crop]
    month_rows = [{"half_month_start": start, **_loss_row(rows)} for start, rows in sorted(by_half_month.items())]
    return crop_rows, month_rows, _loss_row(data["quality_rows"])


def audit(repository):
    root, report, output = paths(repository)
    pending(root, "quality")
    before = collection_signature(report)
    # Existing snapshot/resolve_collection rejects pending batches; no import,
    # acquisition, or mutation of campaign files is performed here.
    handoff = external.audit(report, output)
    if before != collection_signature(report):
        raise RuntimeError("Collection changed during auditing; finish collection and start a new audit")
    snapshot = handoff.parent / "collection_snapshot.json"
    save_json(snapshot, {"campaign_metadata": before, "report_root": str(report)})
    data = json.loads(handoff.read_text())
    counts = handoff.parent / "quality_counts.csv"
    seasons.write_records(counts, quality_counts(data))
    overlap_report = handoff.parent / "test_overlap_exclusions.json"
    audit_reports = [overlap_report] if overlap_report.exists() else []
    loss_by_crop, loss_by_half_month, _ = observation_loss(data)
    loss = handoff.parent / "observation_loss.csv"
    loss_months = handoff.parent / "observation_loss_by_half_month.csv"
    seasons.write_records(loss, loss_by_crop)
    seasons.write_records(loss_months, loss_by_half_month)
    return publish(root, "quality", handoff, [handoff, snapshot, counts, *audit_reports, loss, loss_months,
                   *[Path(p) for p in data["pixel_paths"].values()]])


def choose_input(root, explicit, stage):
    path = Path(explicit).expanduser() if explicit else resolve_stage_pointer(pointer(root, stage))
    if not path.is_absolute():
        path = root / path
    return verify(path, stage)


def select(repository, quality_handoff=None):
    root, report, _ = paths(repository)
    pending(root, "windows")
    quality = choose_input(root, quality_handoff, "quality")
    check_snapshot(quality, report)
    existing = quality.parent / "windows.json"
    if record_path(existing).exists():
        handoff = verify(existing, "windows")
    else:
        if existing.exists() or (quality.parent / "greenup").exists():
            raise FileExistsError("Partial window outputs exist. Start a new imagery audit to preserve this attempt")
        handoff = external.select(quality, report)
        check_snapshot(quality, report)
        # Stream the handoff: it can be larger than available memory.
        pixels, groups = Counter(), {}
        for name, row in iter_json_object(handoff, arrays=("fixed", "aligned")):
            if name in ("fixed", "aligned"):
                key = name, row["partition"], row["crop"]
                pixels[key] += 1
                groups.setdefault(key, set()).add(row["partition_group"])
        counts = handoff.parent / "window_counts.csv"
        seasons.write_records(counts, [
            {"window": name, "partition": partition, "crop": crop,
             "complete_pixel_windows_before_cap": pixels[name, partition, crop],
             "parcel_groups": len(groups.get((name, partition, crop), ()))}
            for name in ("fixed", "aligned") for partition in PARTITIONS for crop in CROPS])
        return publish(root, "windows", handoff, [handoff, quality, quality.parent / "collection_snapshot.json", counts])
    refresh(root)
    write_stage_pointer(pointer(root, "windows"), handoff)
    return handoff


def coverage_rows(development, test):
    """Final capped field/pixel counts, including absent crops and unmet budgets."""
    result = []
    for cohort in ("fixed_full", "aligned_full", "fixed_matched", "aligned_matched"):
        rows = development["rows"][cohort] + test["rows"][cohort]
        for partition in PARTITIONS:
            for crop in CROPS:
                selected = [r for r in rows if r["partition"] == partition and r["crop"] == crop]
                fields = len({r["partition_group"] for r in selected})
                target = adaptive.DEFAULT_POLICY["targets"][partition]
                result.append({"cohort": cohort, "partition": partition, "crop": crop,
                               "usable_parcel_groups": fields, "usable_pixels": len(selected),
                               "target_parcels": target["parcels"], "target_pixels": target["pixels"],
                               "parcel_shortfall": max(0, target["parcels"] - fields),
                               "pixel_shortfall": max(0, target["pixels"] - len(selected)),
                               "budget_met": fields >= target["parcels"] and len(selected) >= target["pixels"]})
    return result


def test_population_coverage(test):
    populations = set(test.get("test_cohorts", ())) | {"jecam"}
    populations.update(row.get("test_cohort", "jecam")
                       for rows in test["rows"].values() for row in rows)
    empty_development = {"rows": {name: [] for name in test["rows"]}}
    result = []
    for population in [*sorted(populations), "combined"]:
        scoped = {"rows": {
            name: [r for r in rows if population == "combined"
                   or r.get("test_cohort", "jecam") == population]
            for name, rows in test["rows"].items()
        }}
        result.extend({"test_population": population, **row}
                      for row in coverage_rows(empty_development, scoped)
                      if row["partition"] == "test")
    return result


def export_features(repository, window_handoff=None):
    root, report, _ = paths(repository)
    pending(root, "features")
    windows = choose_input(root, window_handoff, "windows")
    check_snapshot(windows, report)
    existing = windows.parent / "features.json"
    if record_path(existing).exists():
        handoff = verify(existing, "features")
        refresh(root)
        write_stage_pointer(pointer(root, "features"), handoff)
        return handoff
    if any((windows.parent / name).exists() for name in
           ("features.json", "features_development.json", "features_test.json")):
        raise FileExistsError("Partial feature outputs exist. Start a new imagery audit to preserve this attempt")
    handoff = external.export_features(windows)
    check_snapshot(windows, report)
    development = handoff.parent / "features_development.json"
    test = handoff.parent / "features_test.json"
    test_data = json.loads(test.read_text())
    primary = {"rows": {name: [r for r in rows if r.get("test_cohort", "jecam") == "jecam"]
                        for name, rows in test_data["rows"].items()}}
    rows = coverage_rows(json.loads(development.read_text()), primary)
    counts = handoff.parent / "usable_samples.csv"
    seasons.write_records(counts, rows)
    population_counts = handoff.parent / "test_population_coverage.csv"
    seasons.write_records(population_counts, test_population_coverage(test_data))
    report_path = handoff.parent / "preprocessing_report.md"
    text = ["# Usable external samples", "",
            "Counts below are after window qualification, complete features and the existing per-parcel cap.",
            "Shared development pixel locations are excluded from both parcel records before capping. See development_overlap_exclusions.json.",
            "Budgets are collection targets, not accuracy guarantees. Zero means no usable samples for that cohort.",
            "Matched recipes use identical locations; full cohorts have separate coverage. Do not add the cohorts together.", "",
            "| Cohort | Partition | Crop | Parcel groups | Pixels | Parcel shortfall | Pixel shortfall |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    text += [f"| {r['cohort']} | {r['partition']} | {r['crop']} | {r['usable_parcel_groups']} | {r['usable_pixels']} | {r['parcel_shortfall']} | {r['pixel_shortfall']} |" for r in rows]
    text += ["", "The table shows development and primary JECAM coverage.",
             "See test_population_coverage.csv for JECAM, each supplement and the combined external pool.",
             "Original imagery masks, 0.80 coverage requirement, eight-bin completeness, timing rules and locked partitions are unchanged."]
    report_path.write_text("\n".join(text) + "\n")
    return publish(root, "features", handoff, [handoff, windows, development, test, counts, population_counts, report_path,
                   handoff.parent / "window_coverage.json", handoff.parent / "collection_snapshot.json",
                   handoff.parent / "development_overlap_exclusions.json"])
