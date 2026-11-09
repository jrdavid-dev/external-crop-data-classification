"""Run/resume a bounded 100-tile full-year live selected-pixel benchmark."""
import json
import shutil
import argparse
import pandas as pd
import time
from datetime import datetime, timezone
from pathlib import Path

import ee
import numpy as np
import pyarrow.parquet as pq
import rasterio

from crop_pipeline.common import save_json
from crop_pipeline.imagery import batch_export as batch, pixel_tables, timelines, quality, batch_progress
from crop_pipeline.imagery.accelerated import AcceleratedRetriever
from crop_pipeline.imagery.retrieval import SatelliteRetriever
from crop_pipeline.sources import annual, annual_batch
from crop_pipeline.windows import annual_archive

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "sen4agrinet_work/pixel_table_benchmark_100"


def compare_first(ee, inputs, require_batch_daily=True):
    # Full-year numerical parity against the original raster transport for the first tile.
    direct = SatelliteRetriever(ee, 2022, DEST / "comparison/images", DEST / "comparison/reports")
    inventory = AcceleratedRetriever(ee, 2022, inputs["imagery_folder"], inputs["report_folder"])
    first = inventory.inventory_region(annual.read_regions(inputs)[0])
    first["download_intervals"] = annual.quality.interval_bins(2022)
    results = direct.download_region(first)
    if any(r["status"] == "failed" for r in results):
        raise ValueError("Direct comparison failed; resume the same benchmark")
    path, _ = pixel_tables.paths(Path(inputs["imagery_folder"]) / "pixel_tables", first)
    frame = pq.read_table(path).to_pandas()
    differences = []
    for interval in first["download_intervals"]:
        raster_path = direct.imagery_folder / first["fragment_id"] / batch.settings_fingerprint(first["settings"]) / f"{interval['name']}.tif"
        group = frame[frame["interval"] == interval["name"]]
        with rasterio.open(raster_path) as image:
            expected = image.read()[:, group["row"].to_numpy(), group["column"].to_numpy()].T
        actual = group[direct.output_bands].to_numpy(dtype="float32")
        np.testing.assert_array_equal(actual, expected)
        differences.append(float(np.max(np.abs(actual - expected))))
    direct_rows = timelines.retrieve_observed_timelines(
        [first], direct.mask_acquisition, ee, DEST / "comparison/timelines", .8)
    batch_daily_folder = Path(inputs["timeline_folder"]) / "tiles" / first["fragment_id"]
    daily_source = "batch_export"
    if batch.missing_daily_checkpoints([first], batch_daily_folder):
        if require_batch_daily:
            raise ValueError("Annual batch daily summaries have not finished")
        batch_daily_folder = DEST / "probe/batch_daily"
        daily_source = "interactive_probe"
    batch_rows = timelines.retrieve_observed_timelines(
        [first], inventory.mask_acquisition, ee,
        batch_daily_folder, .8)
    if direct_rows != batch_rows:
        raise ValueError("Batch daily summaries differ from direct summaries")
    table_quality, table_pixels = pixel_tables.audit_region(
        Path(inputs["imagery_folder"]) / "pixel_tables", first)
    raster_quality, raster_pixels = quality.quality_check(
        [first["fragment"]], direct.report_folder / "composite_progress", direct.imagery_folder,
        inputs["field_folder"], DEST / "comparison/quality", year=2022, required_request_version=4)
    raster_pixels = [{k: None if v == "" else v for k, v in row.items()} for row in raster_pixels]
    if table_pixels != raster_pixels:
        raise ValueError("Table window adapter differs from raster auditor")
    if [{k:v for k,v in r.items() if k != "image_path"} for r in table_quality] != [
            {k:v for k,v in r.items() if k != "image_path"} for r in raster_quality]:
        raise ValueError("Table coverage/quality decisions differ from raster auditor")
    return {"tiles": 1, "half_months": 24, "bands": 7,
            "exact_equal": True, "max_absolute_difference": max(differences),
            "daily_rows": len(direct_rows), "daily_exact_equal": True,
            "daily_source": daily_source, "window_adapter_exact_equal": True}


def show_status(ee, inputs, project):
    """Read-only cloud check; never submit, download results or delete assets."""
    plans = []
    for region in annual.read_regions(inputs):
        path = Path(inputs["report_folder"]) / "acquisition_inventory" / f"{region['properties']['fragment_id']}.json"
        if not path.exists():
            raise ValueError("Scene inventories are incomplete; run the benchmark without --status first")
        plan = json.loads(path.read_text())
        plan["fragment"] = region
        plan["download_intervals"] = annual.quality.interval_bins(2022)
        plans.append(plan)
    retriever = AcceleratedRetriever(None, 2022, inputs["imagery_folder"], inputs["report_folder"])
    missing_images = pixel_tables.pending(retriever, plans)
    missing_daily = [plan for plan in plans if batch.missing_daily_checkpoints(
        [plan], Path(inputs["timeline_folder"]) / "tiles" / plan["fragment_id"])]
    ledger = batch.load_ledger(DEST / "batch/pixel_tables/2022.json")
    # One project task-list request, not a remote call for every historical attempt.
    operations = {op["name"].rsplit("/", 1)[-1]: op.get("metadata", {})
                  for op in ee.data.listOperations()}
    for entry in ledger:
        if entry.get("collected"):
            continue
        metadata = operations.get(entry["task_id"])
        entry["state"] = metadata.get("state", "UNKNOWN") if metadata else "UNKNOWN"
    progress = batch_progress.snapshot(plans, missing_images, missing_daily, ledger)
    progress["remote_states_refreshed"] = True
    progress["benchmark_validation"] = ("passed" if (DEST / "benchmark.json").exists()
                                        and progress["tiles"]["complete"] == len(plans) else "pending")
    batch_progress.save_snapshot(DEST / "progress.json", progress)
    batch_progress.print_snapshot(progress)
    print(f"  Benchmark numerical verification: {progress['benchmark_validation']}", flush=True)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for flag, help_text in (
        ("--status", "Show read-only progress once; do not submit or collect jobs"),
        ("--watch", "Refresh read-only progress every 30 seconds; Ctrl+C to stop"),
        ("--probe", "Compare compact exports interactively on one tile"),
        ("--compare-first", "Compare the first downloaded annual table with direct imagery"),
        ("--download-probe", "Compare two retrieval transports on a completed asset"),
    ):
        modes.add_argument(flag, action="store_true", help=help_text)
    options = parser.parse_args()
    started = time.monotonic()
    project = json.loads((ROOT / "settings.local.json").read_text())["earth_engine_project"]
    ee.Initialize(project=project)
    DEST.mkdir(parents=True, exist_ok=True)
    plan_path = DEST / "source_plan.json"
    if not plan_path.exists() and (options.status or options.watch):
        raise ValueError("Start the benchmark first; no existing 100-tile run was found")
    if not plan_path.exists():
        sources = sorted((ROOT / "sen4agrinet_work/annual_archive/francecrops").glob(
            "runs/full_year_all_parcels/*/collection_plan.json"))
        if not sources:
            raise FileNotFoundError("Prepare FranceCrops references before this benchmark")
        original = json.loads(sources[-1].read_text())
        inputs = original["years"]["2022"]
        regions = annual.read_regions(inputs)[:100]
        if len(regions) != 100:
            raise ValueError("Benchmark requires 100 prepared tiles")
        fields = DEST / "fields/2022"
        fields.mkdir(parents=True, exist_ok=True)
        save_json(fields / "sampled_download_regions.geojson", {"type": "FeatureCollection", "features": regions})
        benchmark_inputs = {"field_folder": str(fields), "imagery_folder": str(DEST / "imagery/2022"),
                            "report_folder": str(DEST / "reports/2022"),
                            "timeline_folder": str(DEST / "timelines/2022")}
        # Reuse scene metadata already collected, without touching the original archive.
        original_reports = Path(inputs["report_folder"])
        for name in ("acquisition_inventory", "shared_scene_inventory"):
            source = original_reports / name
            if source.exists():
                shutil.copytree(source, Path(benchmark_inputs["report_folder"]) / name, dirs_exist_ok=True)
        plan = {"source": "francecrops", "revision": "pixel_table_benchmark_100_v1",
                "run_folder": str(DEST), "provenance": original["provenance"],
                "years": {"2022": benchmark_inputs}}
        save_json(plan_path, plan)
        selected = sum(r["properties"]["selected_pixels"] for r in regions)
        grid = sum(r["properties"]["sampling_grid"]["width"] * r["properties"]["sampling_grid"]["height"] for r in regions)
        save_json(DEST / "scope.json", {"tiles": 100, "year": 2022, "selected_pixels": selected,
                  "grid_pixels": grid, "annual_rows": selected * 24,
                  "created_at": datetime.now(timezone.utc).isoformat()})
    source = json.loads(plan_path.read_text())
    inputs = source["years"]["2022"]
    if options.watch or options.status:
        while True:
            progress = show_status(ee, inputs, project)
            if progress["tiles"]["complete"] == progress["tiles"]["total"]:
                if progress["benchmark_validation"] != "passed":
                    print("All data is local. Run without flags to finish benchmark verification.", flush=True)
                break
            if not options.watch:
                break
            if progress["jobs"]["ready_to_download"]:
                print("Results are ready. Run without --watch to download and verify them.", flush=True)
            print("Next read-only check in 30 seconds; Ctrl+C to stop.", flush=True)
            time.sleep(30)
        return
    if options.download_probe:
        entries = batch.load_ledger(DEST / "batch/pixel_tables/2022.json")
        for entry in entries:
            if entry["kind"] != "pixel_table" or entry.get("collected") or entry.get("state") in batch.FAILED_STATES:
                continue
            state, _ = batch.task_state(ee, project, entry["task_id"])
            if state not in batch.FINISHED_STATES:
                continue
            start = time.monotonic()
            pages = batch.read_table(ee, entry["asset_id"])
            page_seconds = time.monotonic() - start
            start = time.monotonic()
            csv = batch.read_pixel_table_csv(ee, entry["asset_id"])
            csv_seconds = time.monotonic() - start
            columns = ["region_id", "interval", "x", "y", *quality.STUDY_BANDS, "clear_count"]
            keys = ["region_id", "interval", "y", "x"]
            left = pages[columns].sort_values(keys).reset_index(drop=True)
            right = csv[columns].sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
            result = {"rows": len(csv), "feature_pages_seconds": page_seconds,
                      "csv_seconds": csv_seconds, "exact_equal": True,
                      "asset_id": entry["asset_id"]}
            save_json(DEST / "download_transport_comparison.json", result)
            print(json.dumps(result, indent=2), flush=True)
            return
        print("No completed uncollected pixel-table asset is available for the download probe.", flush=True)
        return
    if options.compare_first:
        result = compare_first(ee, inputs, require_batch_daily=False)
        save_json(DEST / "comparison_first.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return
    if options.probe:
        retriever = AcceleratedRetriever(ee, 2022, inputs["imagery_folder"], inputs["report_folder"])
        plan = retriever.inventory_region(annual.read_regions(inputs)[0])
        plan["download_intervals"] = annual.quality.interval_bins(2022)
        payload = batch.daily_collection(ee, retriever, [plan]).getInfo()
        daily = pd.DataFrame([f["properties"] for f in payload["features"]])
        batch.write_daily_checkpoints({plan["fragment_id"]: plan}, [plan["fragment_id"]],
                                      daily, DEST / "probe/batch_daily", .8)
        old = timelines.retrieve_observed_timelines([plan], retriever.mask_acquisition, ee,
                                                    DEST / "probe/direct_daily", .8)
        new = timelines.retrieve_observed_timelines([plan], retriever.mask_acquisition, ee,
                                                    DEST / "probe/batch_daily", .8)
        assert old == new, "Compact daily export differs from direct method"
        interval = plan["download_intervals"][0]
        collection = batch.composite_collection(ee, retriever, [(plan, [interval])], selected_only=True)
        exported = ee.data.computeFeatures({"expression": collection, "fileFormat": "PANDAS_DATAFRAME"})
        (DEST / "probe").mkdir(parents=True, exist_ok=True)
        path = DEST / "probe/direct.tif"
        direct = SatelliteRetriever(ee, 2022, DEST / "probe/images", DEST / "probe/reports")
        granules = [r for r in plan["acquisitions"] if interval["start"] <= r["date"] < interval["end"]]
        direct.fetch_composite(plan, interval, granules, path)
        columns = np.floor(exported["x"]).astype(int).to_numpy()
        rows = np.floor(exported["y"]).astype(int).to_numpy()
        ids = rows * plan["settings"]["grid"]["width"] + columns
        from crop_pipeline.imagery import sampling_mask
        assert sorted(ids.tolist()) == sampling_mask.load_selection(plan)["selected_ids"]
        with rasterio.open(path) as image:
            expected = image.read()[:, rows, columns].T
        np.testing.assert_array_equal(exported[direct.output_bands].to_numpy(dtype="float32"), expected)
        save_json(DEST / "probe/result.json", {"status": "passed", "daily_rows": len(old),
                  "daily_exact_equal": True, "selected_pixels": len(exported),
                  "composite_exact_equal": True, "interval": interval["name"]})
        print((DEST / "probe/result.json").read_text(), flush=True)
        return
    report = annual_batch.collect_batch_year(ee, source, 2022, inputs, project=project,
                                            archive_format="pixel_tables")
    elapsed = time.monotonic() - started
    record_path = DEST / "passes.json"
    passes = json.loads(record_path.read_text()) if record_path.exists() else []
    passes.append({"at": datetime.now(timezone.utc).isoformat(), "seconds": elapsed,
                   "completed_tiles": report["completed_tiles"], "pending_intervals": report["pending_images"]})
    save_json(record_path, passes)
    save_json(DEST / "last_report.json", report)
    batch_progress.save_snapshot(DEST / "progress.json", report["progress"])
    print(json.dumps({k:v for k,v in report.items()
                      if k not in {"retry_queue", "progress", "reused_images", "downloaded_images", "pending_images"}},
                     indent=2), flush=True)
    if report["retry_queue"]:
        ledger_rows = batch.load_ledger(DEST / "batch/pixel_tables/2022.json")
        covered = batch.in_flight(ledger_rows) | {
            item for entry in ledger_rows if entry.get("collected") for item in entry["items"]}
        for entry in ledger_rows:
            if entry.get("state") in batch.FAILED_STATES and set(entry["items"]) - covered:
                print("Unresolved failed task:", ee.data.getTaskStatus(entry["task_id"]), flush=True)
        print("PENDING: rerun this benchmark to collect existing jobs; do not start 5,000 tiles yet.", flush=True)
        return
    # Verify downstream handoff loading/auditing for all 100 tiles.
    handoff = json.loads(Path(report["handoff"]).read_text())
    regions, q, observations, partitions = annual_archive.audit_source(handoff, DEST / "analysis", .8)
    parity = compare_first(ee, inputs)
    ledger = batch.load_ledger(DEST / "batch/pixel_tables/2022.json")
    parquet_bytes = sum(p.stat().st_size for p in (DEST / "imagery/2022/pixel_tables").glob("**/*.parquet"))
    receipt_bytes = sum(p.stat().st_size for p in (DEST / "imagery/2022/pixel_tables").glob("**/*.json"))
    scope = json.loads((DEST / "scope.json").read_text())
    result = {"status": "passed", **scope, "parquet_bytes": parquet_bytes, "receipt_bytes": receipt_bytes,
              "download_seconds": sum(r.get("download_seconds", 0) for r in ledger),
              "local_write_seconds": sum(r.get("local_write_seconds", 0) for r in ledger),
              "eecu_seconds": sum(r.get("eecu_seconds", 0) for r in ledger),
              "tasks": len(ledger), "passes": passes,
              "audited_quality_rows": len(q), "audited_daily_rows": len(observations),
              "parity": parity,
              "scope_warning": "First 100 tiles are not necessarily representative of all 18,012."}
    save_json(DEST / "benchmark.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
