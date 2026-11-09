"""Compare one real parcel/month through EE batch assets and direct retrieval.

Run repeatedly to collect a pending export. All output is isolated from production.
"""
import json
import time
import sys
from pathlib import Path

import ee
import numpy as np
import rasterio

from crop_pipeline.common import interval_bins, save_json
from crop_pipeline.imagery import batch_export as batch, timelines
from crop_pipeline.imagery.retrieval import SatelliteRetriever

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "sen4agrinet_work/live_batch_validation"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    project = json.loads((ROOT / "settings.local.json").read_text())["earth_engine_project"]
    ee.Initialize(project=project)
    direct = SatelliteRetriever(ee, 2022, DEST / "direct/images", DEST / "direct/reports")
    exported = SatelliteRetriever(ee, 2022, DEST / "batch/images", DEST / "batch/reports")
    plan_path = DEST / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        references = sorted((ROOT / "sen4agrinet_work/annual_archive/francecrops").glob(
            "runs/full_year_all_parcels/*/fields/2022/sampled_download_regions.geojson"))
        if not references:
            raise FileNotFoundError("Prepare FranceCrops 2022 references first")
        region = json.loads(references[0].read_text())["features"][0]
        plan = direct.inventory_region(region)
        # Two half-month composites, with all their real source acquisitions.
        plan["download_intervals"] = [i for i in interval_bins(2022)
                                      if i["start"].startswith("2022-06")]
        plan["acquisitions"] = [r for r in plan["acquisitions"]
                                if "2022-06-01" <= r["date"] < "2022-07-01"]
        if not timelines.matched_dates(plan):
            raise ValueError("Selected month has no matched acquisitions")
        save_json(plan_path, plan)
    if "--diagnose" in sys.argv:
        from crop_pipeline.imagery import sampling_mask
        selection = sampling_mask.load_selection(plan)
        grid = selection["grid"]
        footprint = sampling_mask.grid_footprint(ee, grid)
        coordinates = ee.Image.pixelCoordinates(ee.Projection(grid["crs"], grid["transform"]))
        selected = sampling_mask.selection_image(ee, grid, selection["selected_ids"])
        params = dict(geometry=footprint, crs=grid["crs"], crsTransform=grid["transform"], maxPixels=10000)
        print("Coordinates:", coordinates.reduceRegion(ee.Reducer.minMax(), **params).getInfo())
        print("Selection:", selected.reduceRegion(ee.Reducer.sum(), **params).getInfo())
        print("Pixel IDs:", selection["selected_ids"][:20])
        record = next(r for r in plan["acquisitions"] if r.get("cloud_probability_id"))
        image = ee.Image(direct.mask_acquisition(ee.Image(record["image_id"]), ee.Image(record["cloud_probability_id"])))
        print("One date band counts:", image.reduceRegion(ee.Reducer.count(), **params).getInfo())
        print("Selected band counts:", image.updateMask(selected).reduceRegion(ee.Reducer.count(), **params).getInfo())
        return
    print("Validation scope:", plan["fragment_id"], plan["settings"]["grid"],
          len(timelines.matched_dates(plan)), "dates", flush=True)
    ledger = DEST / "ledger.json"
    asset_folder = f"projects/{project}/assets/crop_pipeline_batch/live_validation_20261105"
    batch.submit_daily(ee, exported, [plan], DEST / "batch/timelines", ledger, asset_folder, 2022)
    batch.submit_composites(ee, exported, [plan], ledger, asset_folder, 2022)
    print("Submitted/tracked:", [(r["task_id"], r["kind"]) for r in batch.load_ledger(ledger)], flush=True)
    # The original synchronous implementation, independently evaluated and saved.
    started = time.monotonic()
    direct_rows = timelines.retrieve_observed_timelines(
        [plan], direct.mask_acquisition, ee, DEST / "direct/timelines", 0.8)
    results = direct.download_region(plan)
    if any(r["status"] == "failed" for r in results):
        raise RuntimeError(f"Direct retrieval failed: {results}")
    print("Direct results ready in", round(time.monotonic() - started, 2), "seconds", flush=True)
    by_id = {plan["fragment_id"]: plan}

    def daily(entry, table):
        batch.write_daily_checkpoints(by_id, list(by_id), table, DEST / "batch/timelines", 0.8)
        return "daily saved"

    def composites(entry, table):
        batch.write_composites(exported, by_id, table)
        if batch.pending_composites(exported, [plan]):
            raise ValueError("Incomplete exported composite table")
        return "composites saved"

    report = batch.collect(ee, project, ledger, {"daily": daily, "composite": composites}, plans_by_id=by_id)
    print(json.dumps(report, indent=2), flush=True)
    if not all(r["collected"] for r in report):
        for entry in batch.load_ledger(ledger):
            if entry.get("state") in batch.FAILED_STATES:
                print(ee.data.getTaskStatus(entry["task_id"]), flush=True)
        print("PENDING: rerun this script to collect and compare", flush=True)
        return
    batch_rows = timelines.retrieve_observed_timelines(
        [plan], exported.mask_acquisition, ee, DEST / "batch/timelines", 0.8)
    comparisons = []
    if not any(row["valid_pixels"] > 0 for row in direct_rows):
        raise ValueError("Daily comparison requires nonempty valid observations")
    for left, right in zip(direct_rows, batch_rows, strict=True):
        for key in left:
            if key in {"NDVI", "NDRE", "NDMI"} and left[key] is not None:
                np.testing.assert_allclose(left[key], right[key], rtol=1e-7, atol=1e-8)
            else:
                assert left[key] == right[key], (key, left, right)
    for interval in plan["download_intervals"]:
        arrays = []
        metadata = []
        for retriever in (direct, exported):
            path = retriever.imagery_folder / plan["fragment_id"] / batch.settings_fingerprint(plan["settings"]) / (interval["name"] + ".tif")
            with rasterio.open(path) as image:
                arrays.append(image.read())
                metadata.append((image.crs, image.transform, image.nodata, image.descriptions, image.dtypes))
        assert metadata[0] == metadata[1], metadata
        np.testing.assert_array_equal(arrays[0] == direct.nodata_value, arrays[1] == direct.nodata_value)
        np.testing.assert_allclose(arrays[0], arrays[1], rtol=1e-6, atol=1e-3)
        comparisons.append({"interval": interval["name"], "shape": list(arrays[0].shape),
                            "max_absolute_difference": float(np.max(np.abs(arrays[0] - arrays[1]))),
                            "exact_equal": bool(np.array_equal(*arrays))})
    result = {"status": "passed", "region_id": plan["fragment_id"],
              "daily_rows": len(direct_rows),
              "daily_rows_with_valid_pixels": sum(r["valid_pixels"] > 0 for r in direct_rows),
              "daily_rows_eligible": sum(r["eligible"] for r in direct_rows),
              "daily_exact_equal": direct_rows == batch_rows,
              "summary_recipe": timelines.SUMMARY_RECIPE,
              "composites": comparisons, "tasks": report,
              "daily_tolerance": {"rtol": 1e-7, "atol": 1e-8},
              "raster_tolerance": {"rtol": 1e-6, "atol": 1e-3}}
    save_json(DEST / "comparison.json", result)
    batch.remove_temp_folder(ee, ledger, asset_folder)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
