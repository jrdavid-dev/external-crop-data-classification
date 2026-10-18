"""Shared Sentinel-2 acquisition rules for both external crop-reference sources."""

import hashlib
import json
import math
from datetime import date
from pathlib import Path

import rasterio
import requests
from affine import Affine
from pyproj import CRS, Transformer
from shapely.geometry import shape
from shapely.ops import transform as transform_geometry

from crop_pipeline.imagery.quality import interval_bins, save_json


REQUEST_VERSION = 4
RESAMPLING_POLICY = {
    "B3": "native",
    "B4": "native",
    "B8": "native",
    "B2": "native",
    "B5": "bilinear",
    "B11": "bilinear",
    "quality_masks": "nearest",
}


class SatelliteRetriever:
    """Keep both datasets on identical bands, masks, grids and composite definitions."""

    def __init__(
        self, earth_engine, year, imagery_folder, report_folder, cloud_threshold=40
    ):
        if not 0 < cloud_threshold <= 100:
            raise ValueError("Cloud probability cutoff must be between 0 and 100.")
        self.ee = earth_engine
        self.year = int(year)
        self.imagery_folder = Path(imagery_folder)
        self.report_folder = Path(report_folder)
        self.collection_name = "COPERNICUS/S2_SR_HARMONIZED"
        self.start_date = date(self.year, 1, 1).isoformat()
        self.end_date = date(self.year + 1, 1, 1).isoformat()
        self.output_resolution = 10
        self.buffer_metres = 20
        self.spectral_bands = ["B3", "B4", "B5", "B8", "B11", "B2"]
        self.output_bands = self.spectral_bands + ["clear_count"]
        self.nodata_value = -9999
        self.excluded_scl_classes = [0, 1, 3, 8, 9, 10, 11]
        self.cloud_probability_collection = "COPERNICUS/S2_CLOUD_PROBABILITY"
        self.cloud_probability_threshold = cloud_threshold

    def make_download_grid(self, fragment):
        """Create a fixed rectangular grid in the fragment's verified source CRS."""
        source_crs = CRS.from_user_input(fragment["properties"]["source_crs"])
        if not source_crs.is_projected or any(
            (
                axis.unit_name.lower() not in ["metre", "meter"]
                for axis in source_crs.axis_info
            )
        ):
            raise ValueError(
                "The download grid requires a verified projected CRS in metres."
            )
        geometry = shape(fragment["geometry"])
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("Invalid candidate geometry.")
        projector = Transformer.from_crs("EPSG:4326", source_crs, always_xy=True)
        projected = transform_geometry(projector.transform, geometry)
        west, south, east, north = projected.bounds
        left = (
            math.floor((west - self.buffer_metres) / self.output_resolution)
            * self.output_resolution
        )
        right = (
            math.ceil((east + self.buffer_metres) / self.output_resolution)
            * self.output_resolution
        )
        bottom = (
            math.floor((south - self.buffer_metres) / self.output_resolution)
            * self.output_resolution
        )
        top = (
            math.ceil((north + self.buffer_metres) / self.output_resolution)
            * self.output_resolution
        )
        width = int(round((right - left) / self.output_resolution))
        height = int(round((top - bottom) / self.output_resolution))
        return {
            "crs": source_crs.to_string(),
            "transform": [
                self.output_resolution,
                0,
                left,
                0,
                -self.output_resolution,
                top,
            ],
            "width": width,
            "height": height,
        }

    def request_settings(self, fragment):
        """Settings fingerprint prevents reuse of files from a different request."""
        return {
            "collection": self.collection_name,
            "start": self.start_date,
            "end": self.end_date,
            "bands": self.output_bands,
            "nodata": self.nodata_value,
            "grid": self.make_download_grid(fragment),
            "geometry": fragment["geometry"],
            "resampling": RESAMPLING_POLICY,
            "partition_group": fragment["properties"].get("partition_group"),
            "sampling_mask": fragment["properties"].get("sampling_mask"),
            "reflectance_scale_to_apply_later": 0.0001,
            "version": REQUEST_VERSION,
            "compositing": "median of cloud-screened daily mosaics in calendar half-month bins",
            "excluded_scl_classes": self.excluded_scl_classes,
            "cloud_probability_collection": self.cloud_probability_collection,
            "cloud_probability_threshold": self.cloud_probability_threshold,
            "missing_cloud_probability": "exclude acquisition, report missing match",
            "scene_edge_masks": ["B8A", "B9"],
        }

    def checksum(self, path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def check_raster(self, path, grid):
        """Check dimensions, CRS, transform, band count and sample readability."""
        with rasterio.open(path) as raster:
            if raster.count != len(self.output_bands):
                raise ValueError("Unexpected downloaded band count.")
            if (raster.width, raster.height) != (grid["width"], grid["height"]):
                raise ValueError("Downloaded raster dimensions do not match the plan.")
            if CRS.from_user_input(raster.crs) != CRS.from_user_input(grid["crs"]):
                raise ValueError("Downloaded CRS does not match the plan.")
            if not raster.transform.almost_equals(Affine(*grid["transform"])):
                raise ValueError("Downloaded raster transform does not match the plan.")
            if any((dtype != "float32" for dtype in raster.dtypes)):
                raise ValueError(
                    "Expected float32 medians and clear-observation counts."
                )
            raster.read(
                window=rasterio.windows.Window(
                    0, 0, min(16, raster.width), min(16, raster.height)
                )
            )

    def list_acquisitions(self, fragment):
        """Fetch only small metadata for original granules intersecting a fragment."""
        footprint = self.ee.Geometry(
            fragment["geometry"], proj="EPSG:4326", geodesic=False
        )
        collection = (
            self.ee.ImageCollection(self.collection_name)
            .filterBounds(footprint)
            .filterDate(self.start_date, self.end_date)
        )

        def describe(image):
            image = self.ee.Image(image)
            return self.ee.Feature(
                None,
                {
                    "image_index": image.get("system:index"),
                    "timestamp_ms": image.get("system:time_start"),
                    "date": self.ee.Date(image.get("system:time_start")).format(
                        "YYYY-MM-dd"
                    ),
                    "tile": image.get("MGRS_TILE"),
                    "scene_cloud_percent": image.get("CLOUDY_PIXEL_PERCENTAGE"),
                },
            )

        metadata = self.ee.FeatureCollection(
            collection.toList(collection.size()).map(describe)
        ).getInfo()
        rows = []
        for feature in metadata["features"]:
            row = feature["properties"]
            row["image_id"] = self.collection_name + "/" + row["image_index"]
            rows.append(row)
        return sorted(rows, key=lambda row: (row["timestamp_ms"], row["image_index"]))

    def find_cloud_probability_matches(self, fragment):
        """Match original granules by exact system:index, not nearest date."""
        footprint = self.ee.Geometry(
            fragment["geometry"], proj="EPSG:4326", geodesic=False
        )
        clouds = (
            self.ee.ImageCollection(self.cloud_probability_collection)
            .filterBounds(footprint)
            .filterDate(self.start_date, self.end_date)
        )
        return set(clouds.aggregate_array("system:index").getInfo())

    def mask_acquisition(self, image, cloud_probability_image):
        """Combine SCL, cloud probability and valid-data masks BEFORE medians."""
        image = self.ee.Image(image)
        scl = image.select("SCL")
        excluded = scl.remap(
            self.excluded_scl_classes, [1] * len(self.excluded_scl_classes), 0
        )
        probability = self.ee.Image(cloud_probability_image).select("probability")
        low_cloud_score = probability.lt(self.cloud_probability_threshold)
        reflectance = image.select(self.spectral_bands)
        all_bands_valid = reflectance.mask().reduce(self.ee.Reducer.min())
        edges_valid = image.select("B8A").mask().And(image.select("B9").mask())
        valid = excluded.eq(0).And(scl.mask()).And(all_bands_valid)
        valid = valid.And(low_cloud_score).And(probability.mask()).And(edges_valid)
        masked_image = image.updateMask(valid)
        # Resample source observations, not a composite with no native projection.
        # Restrict interpolation to continuous 20 m bands; masks remain categorical.
        native_bands = masked_image.select(["B3", "B4", "B8", "B2"])
        coarse_bands = masked_image.select(["B5", "B11"]).resample("bilinear")
        aligned_bands = coarse_bands.reproject(crs=image.select("B3").projection())
        reflectance = native_bands.addBands(aligned_bands).select(self.spectral_bands)
        complete_support = reflectance.mask().reduce(self.ee.Reducer.min())
        return reflectance.updateMask(complete_support).copyProperties(
            image, ["system:time_start", "system:index"]
        )

    def make_composite(self, fragment, interval, granules):
        """Use at most one observation per date, then take the semi-monthly median."""
        matched_granules = [row for row in granules if row.get("cloud_probability_id")]
        dates = sorted(set((row["date"] for row in matched_granules)))
        daily_images = []
        for observation_date in dates:
            same_date = sorted(
                [row for row in matched_granules if row["date"] == observation_date],
                key=lambda row: row["image_index"],
            )
            images = [
                self.mask_acquisition(
                    self.ee.Image(row["image_id"]),
                    self.ee.Image(row["cloud_probability_id"]),
                )
                for row in same_date
            ]
            daily = self.ee.ImageCollection.fromImages(images).mosaic()
            daily_images.append(daily)
        if daily_images:
            daily_collection = self.ee.ImageCollection.fromImages(daily_images)
            median = daily_collection.median().select(self.spectral_bands)
            clear_count = (
                daily_collection.select("B3")
                .count()
                .rename("clear_count")
                .unmask(0, sameFootprint=False)
            )
            result = median.unmask(self.nodata_value, sameFootprint=False).addBands(
                clear_count
            )
        else:
            result = self.ee.Image.constant(
                [self.nodata_value] * len(self.spectral_bands) + [0]
            ).rename(self.output_bands)
        return result.toFloat()

    def inventory_region(self, region, refresh=False):
        """Cache original granule metadata using the complete current request fingerprint."""
        region_id = region["properties"]["fragment_id"]
        settings = json.loads(json.dumps(self.request_settings(region)))
        path = self.report_folder / "acquisition_inventory" / f"{region_id}.json"
        saved = json.loads(path.read_text()) if path.exists() else {}
        if saved.get("settings") == settings and not refresh:
            return saved
        acquisitions = self.list_acquisitions(region)
        cloud_indices = self.find_cloud_probability_matches(region)
        for acquisition in acquisitions:
            image_index = acquisition["image_index"]
            acquisition["cloud_probability_id"] = (
                self.cloud_probability_collection + "/" + image_index
                if image_index in cloud_indices
                else None
            )
        plan = {
            "fragment_id": region_id,
            "fragment": region,
            "settings": settings,
            "acquisitions": acquisitions,
        }
        save_json(path, plan)
        return plan

    def build_plans(self, regions, refresh=False):
        plans, errors = [], []
        for region in regions:
            try:
                plans.append(self.inventory_region(region, refresh))
            except Exception as error:
                errors.append(
                    {
                        "fragment_id": region["properties"]["fragment_id"],
                        "error": str(error),
                    }
                )
        missing_matches = [
            {
                "fragment_id": plan["fragment_id"],
                "image_id": row["image_id"],
                "reason": "No exact cloud-probability match; excluded from compositing",
            }
            for plan in plans
            for row in plan["acquisitions"]
            if not row.get("cloud_probability_id")
        ]
        save_json(
            self.report_folder / "missing_cloud_probability.json", missing_matches
        )
        save_json(
            self.report_folder / "composite_plan.json",
            {
                "intervals": interval_bins(self.year),
                "planned_composites": len(plans) * 24,
                "failed_inventory": errors,
                "bands": self.output_bands,
                "granules_without_cloud_match": len(missing_matches),
                "cloud_probability_threshold": self.cloud_probability_threshold,
            },
        )
        return plans, errors

    def download_composites(self, plans):
        """Checkpoint individual files, preserving failures and reusing verified downloads."""
        results = []
        for plan in plans:
            results.extend(self.download_region(plan))
        summary = {
            "downloaded": sum(row["status"] == "downloaded" for row in results),
            "reused": sum(row["status"] == "reused" for row in results),
            "failed": sum(row["status"] == "failed" for row in results),
            "results": results,
        }
        save_json(self.report_folder / "last_composite_run.json", summary)
        return summary

    def download_region(self, plan):
        settings = plan["settings"]
        grid = settings["grid"]
        expected = plan["fragment"]["properties"]["sampling_grid"]
        if grid != expected:
            raise ValueError(
                "Sampling-mask and retrieval grids differ; regenerate the pilot."
            )
        if (
            max(grid["width"], grid["height"]) >= 10000
            or grid["width"] * grid["height"] * 28 >= 30 * 1024**2
        ):
            raise ValueError("Region exceeds direct-download size limits.")
        region_id = plan["fragment_id"]
        request_id = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode()
        ).hexdigest()[:16]
        folder = self.imagery_folder / region_id / request_id
        folder.mkdir(parents=True, exist_ok=True)
        path = (
            self.report_folder / "composite_progress" / f"{region_id}_{request_id}.json"
        )
        manifest = (
            json.loads(path.read_text())
            if path.exists()
            else {"fragment_id": region_id, "settings": settings, "files": {}}
        )
        results = []
        # Timing-first plans retain the unchanged composites but request only qualified bins.
        intervals = plan.get("download_intervals", interval_bins(self.year))
        allowed_intervals = {
            interval["name"]: interval for interval in interval_bins(self.year)
        }
        if any(
            allowed_intervals.get(interval["name"]) != interval
            for interval in intervals
        ):
            raise ValueError(
                "Selected download intervals must match full calendar half-month bins."
            )
        for interval in intervals:
            result = self.download_interval(plan, interval, folder, manifest)
            save_json(path, manifest)
            results.append(result)
        return results

    def reusable_file(self, path, previous, grid):
        if not previous or previous.get("status") != "complete" or not path.is_file():
            return False
        try:
            self.check_raster(path, grid)
            return self.checksum(path) == previous["sha256"]
        except Exception:
            return False

    def download_interval(self, plan, interval, folder, manifest):
        destination = folder / f"{interval['name']}.tif"
        previous = manifest["files"].get(interval["name"])
        if self.reusable_file(destination, previous, plan["settings"]["grid"]):
            return {
                "region_id": plan["fragment_id"],
                "name": interval["name"],
                "status": "reused",
            }
        granules = [
            row
            for row in plan["acquisitions"]
            if interval["start"] <= row["date"] < interval["end"]
        ]
        record = {
            **interval,
            "path": str(destination.resolve()),
            "source_image_ids": [row["image_id"] for row in granules],
            "available_dates": sorted({row["date"] for row in granules}),
            "cloud_screened_available_dates": sorted(
                {row["date"] for row in granules if row.get("cloud_probability_id")}
            ),
        }
        temporary = destination.with_suffix(".tif.part")
        try:
            self.fetch_composite(plan, interval, granules, temporary)
            record.update(self.finish_raster(temporary))
            temporary.replace(destination)
            status = "downloaded"
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            temporary.unlink(missing_ok=True)
            status = "failed"
        manifest["files"][interval["name"]] = record
        print(plan["fragment_id"], interval["name"], status)
        return {
            "region_id": plan["fragment_id"],
            "name": interval["name"],
            "status": status,
        }

    def fetch_composite(self, plan, interval, granules, destination):
        grid = plan["settings"]["grid"]
        composite = self.make_composite(plan["fragment"], interval, granules)
        url = composite.getDownloadURL(
            {
                "name": interval["name"],
                "format": "GEO_TIFF",
                "filePerBand": False,
                "crs": grid["crs"],
                "crs_transform": grid["transform"],
                "dimensions": [grid["width"], grid["height"]],
            }
        )
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            self.write_response(response, destination)
        self.check_raster(destination, grid)

    @staticmethod
    def write_response(response, destination):
        with Path(destination).open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                stream.write(chunk)

    def finish_raster(self, path):
        with rasterio.open(path, "r+") as raster:
            raster.nodata = self.nodata_value
            for band_number, band in enumerate(self.output_bands, start=1):
                raster.set_band_description(band_number, band)
            counts = raster.read(len(self.output_bands))
        return {
            "status": "complete",
            "sha256": self.checksum(path),
            "bytes": path.stat().st_size,
            "pixels_with_observations": int((counts > 0).sum()),
            "maximum_clear_dates": int(counts.max()),
            "empty_after_screening": not bool((counts > 0).any()),
        }
