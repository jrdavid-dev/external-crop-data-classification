"""Safe notebook handoffs and coverage reporting for external preprocessing.

Only new analysis outputs and navigation pointers are written. Collection state,
source selection, imagery rules and external model feature schemas are unchanged.
"""

from collections import Counter
import json
from pathlib import Path

from crop_pipeline.analysis_workflow import resolve_stage_pointer, write_stage_pointer
from crop_pipeline.common import save_json
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
    report = (root / Path(settings.get("report_root", "sen4agrinet_work/inventory")).expanduser()).resolve()
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
    return publish(root, "quality", handoff, [handoff, snapshot, counts,
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
        data = json.loads(handoff.read_text())
        counts = handoff.parent / "window_counts.csv"
        seasons.write_records(counts, [
            {"window": name, "partition": partition, "crop": crop,
             "complete_pixel_windows_before_cap": sum(r["partition"] == partition and r["crop"] == crop for r in data[name]),
             "parcel_groups": len({r["partition_group"] for r in data[name] if r["partition"] == partition and r["crop"] == crop})}
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
    rows = coverage_rows(json.loads(development.read_text()), json.loads(test.read_text()))
    counts = handoff.parent / "usable_samples.csv"
    seasons.write_records(counts, rows)
    report_path = handoff.parent / "preprocessing_report.md"
    text = ["# Usable external samples", "",
            "Counts below are after window qualification, complete features and the existing per-parcel cap.",
            "Shared development pixel locations are excluded from both parcel records before capping. See development_overlap_exclusions.json.",
            "Budgets are collection targets, not accuracy guarantees. Zero means no usable samples for that cohort.",
            "Matched recipes use identical locations; full cohorts have separate coverage. Do not add the cohorts together.", "",
            "| Cohort | Partition | Crop | Parcel groups | Pixels | Parcel shortfall | Pixel shortfall |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    text += [f"| {r['cohort']} | {r['partition']} | {r['crop']} | {r['usable_parcel_groups']} | {r['usable_pixels']} | {r['parcel_shortfall']} | {r['pixel_shortfall']} |" for r in rows]
    text += ["", "The test set is external CropHarvest JECAM, not local accuracy evidence.",
             "Original imagery masks, 0.80 coverage requirement, eight-bin completeness, timing rules and locked partitions are unchanged."]
    report_path.write_text("\n".join(text) + "\n")
    return publish(root, "features", handoff, [handoff, windows, development, test, counts, report_path,
                   handoff.parent / "window_coverage.json", handoff.parent / "collection_snapshot.json",
                   handoff.parent / "development_overlap_exclusions.json"])
