"""Small dated field-sample summaries; no additional imagery downloads."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from crop_pipeline.imagery.quality import save_json, write_table


def retrieve_observed_timelines(plans, mask_acquisition, ee, destination, minimum_fraction=0.8):
    """Checkpoint each region's daily indices on its exact selected-pixel grid."""
    destination = Path(destination)
    all_rows, failures = [], []
    for plan in plans:
        region_id = plan["fragment_id"]
        region = plan["fragment"]
        properties = region["properties"]
        settings = plan["settings"]
        fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
        path = destination / "observed_checkpoints" / f"{region_id}_{fingerprint}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            all_rows.extend(payload["rows"])
            print(region_id, "observed summaries reused")
            continue
        try:
            matched = [row for row in plan["acquisitions"] if row.get("cloud_probability_id")]
            dates = sorted(set(row["date"] for row in matched))
            footprint = ee.Geometry(region["geometry"], proj="EPSG:4326", geodesic=False)
            grid = settings["grid"]
            features = []
            for observation_date in dates:
                granules = sorted([row for row in matched if row["date"] == observation_date], key=lambda row: row["image_index"])
                images = [mask_acquisition(ee.Image(row["image_id"]), ee.Image(row["cloud_probability_id"])) for row in granules]
                daily = ee.ImageCollection.fromImages(images).mosaic()
                indices = []
                # Explicit ratios preserve negative reflectance values and mask zero denominators.
                for name, other in [("NDVI", "B4"), ("NDRE", "B5"), ("NDMI", "B11")]:
                    left, right = daily.select("B8"), daily.select(other)
                    denominator = left.add(right)
                    indices.append(left.subtract(right).divide(denominator).updateMask(denominator.neq(0)).rename(name))
                image = ee.Image.cat(indices)
                reducer = ee.Reducer.median().combine(ee.Reducer.count(), sharedInputs=True)
                statistics = image.reduceRegion(reducer=reducer, geometry=footprint,
                    crs=grid["crs"], crsTransform=grid["transform"], maxPixels=1000000)
                features.append(ee.Feature(None, statistics.set("date", observation_date)))
            payload = ee.FeatureCollection(features).getInfo()
            rows = []
            for feature in payload["features"]:
                values = feature["properties"]
                count = min(int(values.get(name + "_count", 0) or 0) for name in ["NDVI", "NDRE", "NDMI"])
                fraction = count / properties["selected_pixels"]
                if fraction > 1 + 1e-6:
                    raise ValueError("Region reduction included more pixels than the sampling mask")
                rows.append({"region_id": region_id, "partition_group": properties["partition_group"],
                    "crop": properties["crop"], "date": values["date"], "source_kind": "observed_daily",
                    "valid_pixels": count, "valid_fraction": fraction, "eligible": fraction >= minimum_fraction,
                    **{name: values.get(name + "_median") for name in ["NDVI", "NDRE", "NDMI"]}})
            save_json(path, {"settings": settings, "retrieved_at": datetime.now(timezone.utc).isoformat(), "rows": rows})
            all_rows.extend(rows)
            print(region_id, len(rows), "observed daily summaries")
        except Exception as error:
            failures.append({"region_id": region_id, "error": str(error)})
            save_json(destination / "observed_errors.json", failures)
            print(region_id, "observed summary failed:", error)
    save_json(destination / "observed_timelines.json", all_rows)
    save_json(destination / "observed_errors.json", failures)
    write_table(destination / "observed_timelines.csv", all_rows,
        ["region_id", "partition_group", "crop", "date", "source_kind", "valid_pixels", "valid_fraction", "eligible", "NDVI", "NDRE", "NDMI"])
    return all_rows
