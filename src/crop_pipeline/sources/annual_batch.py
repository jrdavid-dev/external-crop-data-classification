"""Nonblocking annual collection through temporary Earth Engine table assets."""

import json
import time
from pathlib import Path

from crop_pipeline.imagery import batch_export as batch
from crop_pipeline.imagery.accelerated import AcceleratedRetriever
from crop_pipeline.imagery import quality, pixel_tables, timelines, batch_progress
from crop_pipeline.sources import annual


def collect_batch_year(ee, source_plan, year, inputs, attempt_folder=None,
                       download_workers=2, intervals_per_request=6, *, project,
                       tile_start=0, tile_count=None, archive_format="rasters"):
    retriever = AcceleratedRetriever(
        ee, int(year), inputs["imagery_folder"], inputs["report_folder"], 40
    )
    regions = annual.read_regions(inputs)
    end = len(regions) if tile_count is None else min(len(regions), tile_start + tile_count)
    if tile_start >= len(regions) and regions:
        raise ValueError(f"tile_start {tile_start} is outside {len(regions)} tiles")
    selected = regions[tile_start:end]
    sliced = tile_start != 0 or tile_count is not None
    suffix = f"_tiles_{tile_start}_{end}" if sliced else ""
    table_mode = archive_format == "pixel_tables"
    ledger_root = Path(source_plan["run_folder"]) / "batch"
    if table_mode:
        ledger_root /= "pixel_tables"
    ledger = ledger_root / f"{year}{suffix}.json"
    selected_ids = {r["properties"]["fragment_id"] for r in selected} if sliced else set()
    # An overlapping active range must be resumed through its original ledger.
    for other in ledger.parent.glob(f"{year}*.json"):
        if other == ledger or other.name.endswith("_coverage.json"):
            continue
        for entry in batch.load_ledger(other):
            if entry.get("collected") or entry.get("state") in batch.FAILED_STATES:
                continue
            covered = set(entry.get("request_fingerprints", {}))
            if not sliced or selected_ids & covered:
                raise ValueError(f"Overlapping jobs are active in {other}; resume that tile range first.")
    print(f"{source_plan['source']} {year}: selecting tiles {tile_start + 1}–{end} "
          f"of {len(regions)}", flush=True)
    inventory_started = time.monotonic()
    plans = []
    for index, region in enumerate(selected, 1):
        plans.append(retriever.inventory_region(region))
        if index % 25 == 0 or index == len(selected):
            print(f"Scene inventories: {index}/{len(selected)} "
                  f"({time.monotonic() - inventory_started:.1f}s)", flush=True)
    for plan in plans:
        plan["download_intervals"] = quality.interval_bins(int(year))
    by_id = {plan["fragment_id"]: plan for plan in plans}
    temp_folder = (f"projects/{project}/assets/crop_pipeline_batch/"
                   f"{source_plan['source']}_{source_plan['revision']}_{year}{suffix}"
                   f"{'_pixel_tables' if table_mode else ''}")

    def timeline_folder(plan):
        return Path(inputs["timeline_folder"]) / "tiles" / plan["fragment_id"]

    def daily(entry, table):
        for region_id in entry["request_fingerprints"]:
            rows = table[table["region_id"] == region_id] if len(table) else table
            batch.write_daily_checkpoints(
                by_id, [region_id], rows, timeline_folder(by_id[region_id]), 0.8
            )
        return "daily checkpoints verified"

    def composites(entry, table):
        batch.write_composites(retriever, by_id, table)
        requested = set(entry["items"])
        for plan, intervals in batch.pending_composites(retriever, list(by_id.values())):
            if any(batch.work_item("composite", plan, interval) in requested
                   for interval in intervals):
                raise ValueError("Batch export omitted or corrupted requested composites")
        return "composites verified"

    def tables(entry, table):
        count = pixel_tables.write_tables(retriever, by_id, table, list(entry["request_fingerprints"]))
        return f"{count} selected-pixel records verified"

    handlers = {"daily": daily, "pixel_table" if table_mode else "composite": tables if table_mode else composites}
    batch.collect(ee, project, ledger, handlers, plans_by_id=by_id)
    missing_daily = [plan for plan in plans if batch.missing_daily_checkpoints(
        [plan], timeline_folder(plan))]
    if table_mode:
        for plan in plans:
            pixel_tables.import_existing_rasters(retriever, plan)
        # Multiple tiles share each daily export, but retain their established checkpoint paths.
        batch.submit_daily(ee, retriever, plans, inputs["timeline_folder"], ledger,
                           temp_folder, int(year), checkpoint_folder=timeline_folder)
        batch.submit_composites(ee, retriever, plans, ledger, temp_folder, int(year), selected_only=True)
        missing_images = pixel_tables.pending(retriever, plans)
    else:
        for plan in missing_daily:
            batch.submit_daily(ee, retriever, [plan], timeline_folder(plan), ledger,
                               temp_folder, int(year))
        batch.submit_composites(ee, retriever, plans, ledger, temp_folder, int(year))
        missing_images = batch.pending_composites(retriever, plans)
    progress = batch_progress.snapshot(plans, missing_images, missing_daily,
                                       batch.load_ledger(ledger), table_mode)
    batch_progress.print_snapshot(progress)
    batch_progress.save_snapshot(ledger.parent / "progress" / f"{ledger.stem}.json", progress)
    selection_complete = not missing_daily and not missing_images
    coverage_path = ledger.parent / f"{year}_coverage.json"
    coverage = set(json.loads(coverage_path.read_text())) if coverage_path.exists() else set()
    if selection_complete:
        batch.remove_temp_folder(ee, ledger, temp_folder)
        coverage.update(plan["fragment_id"] for plan in plans)
        quality.save_json(coverage_path, sorted(coverage))
        all_ids = {r["properties"]["fragment_id"] for r in regions}
        if not sliced or all_ids <= coverage:
            # Receipts only trigger verification; the established checks decide completion.
            if table_mode:
                result = finalize_tables(ee, source_plan, year, inputs, retriever)
            else:
                result = annual.collect_year(ee, source_plan, year, inputs, attempt_folder,
                                             download_workers, intervals_per_request)
            return {**result, "progress": progress}
    pending = {plan["fragment_id"]: len(intervals) for plan, intervals in missing_images}
    for plan in missing_daily:
        pending.setdefault(plan["fragment_id"], 0)
    message = ("selected range completed; choose the next tile_start"
               if selection_complete else "selected range pending; rerun the same range later")
    print(f"{source_plan['source']} {year}: {message}. Ledger: {ledger}", flush=True)
    unselected = len(regions) - len(plans)
    retry_queue = [{"year": str(year), "region_id": region_id,
                    "status": "batch_pending", "pending_image_count": count}
                   for region_id, count in pending.items()]
    if unselected:
        retry_queue.append({"year": str(year), "status": "outside_selected_range",
                            "tile_count": unselected})
    return {
        "progress": progress,
        "selected_batch": {"tile_start": tile_start, "tile_end": end,
                           "tiles": len(plans),
                           "status": "completed" if selection_complete else "pending",
                           "ledger": str(ledger)},
        "year": str(year), "planned_tiles": len(regions),
        "completed_tiles": len(plans) - len(pending), "downloaded_images": 0,
        "reused_images": progress["half_month_periods"]["verified_local"],
        "pending_images": progress["half_month_periods"]["pending"],
        "retry_queue": retry_queue,
    }


def finalize_tables(ee, source_plan, year, inputs, retriever):
    """Verify all annual partitions and daily summaries before publishing a handoff."""
    from crop_pipeline.sources import pilot_sources
    from crop_pipeline.windows.seasons import file_sha256
    index = {}
    regions = annual.read_regions(inputs)
    for region in regions:
        plan = retriever.inventory_region(region)
        plan["download_intervals"] = quality.interval_bins(int(year))
        root = retriever.imagery_folder / "pixel_tables"
        if not pixel_tables.reusable(root, plan, deep=True):
            raise ValueError(f"Annual pixel table incomplete: {plan['fragment_id']}")
        # No network evaluation is allowed here: require the current exact daily checkpoint.
        folder = Path(inputs["timeline_folder"]) / "tiles" / plan["fragment_id"]
        if batch.missing_daily_checkpoints([plan], folder):
            raise ValueError(f"Daily checkpoint incomplete: {plan['fragment_id']}")
        timelines.retrieve_observed_timelines([plan], retriever.mask_acquisition, ee, folder, 0.8)
        _, receipt = pixel_tables.paths(root, plan)
        index[plan["fragment_id"]] = {"path": str(receipt.resolve()), "sha256": file_sha256(receipt)}
    handoff = Path(source_plan["run_folder"]) / "handoffs" / f"{year}_pixel_tables.json"
    manifest = pilot_sources.save_source_manifest(
        handoff, f"{source_plan['source']}_{year}", int(year), inputs["field_folder"],
        inputs["imagery_folder"], retriever.imagery_folder / "pixel_tables",
        inputs["timeline_folder"], source_plan["provenance"])
    manifest.update(collection=annual.COLLECTION_NAME, source_family=source_plan["source"],
                    collection_revision=source_plan["revision"],
                    archive_period=f"{year}-01-01/{int(year)+1}-01-01",
                    candidate_target_per_crop=None, usable_target_per_crop=None,
                    archive_format=pixel_tables.FORMAT, pixel_table_index=index,
                    pixel_table_root=str((retriever.imagery_folder / "pixel_tables").resolve()))
    quality.save_json(handoff, manifest)
    return {"year": str(year), "planned_tiles": len(regions), "completed_tiles": len(regions),
            "downloaded_images": 0, "reused_images": len(regions) * 24,
            "pending_images": 0, "retry_queue": [], "handoff": str(handoff.resolve())}
