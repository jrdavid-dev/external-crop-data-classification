"""Versioned full-year archives, independent of crop-window experiments.

No function authenticates or submits work unless collection is explicitly enabled.
The archive contains calendar half-month medians, not raw Sentinel-2 granules.
"""

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as transform_geometry, unary_union
from shapely.strtree import STRtree

from crop_pipeline.imagery import quality, timelines
from crop_pipeline.imagery.retrieval import (
    SatelliteRetriever,
    REQUEST_VERSION,
    RESAMPLING_POLICY,
)
from crop_pipeline.sources import cropharvest, pilot_sources, sampling
from crop_pipeline.windows.seasons import file_sha256

COLLECTION_NAME = "full_year_all_parcels"
SOURCE_NAMES = ("sen4agrinet", "francecrops", "cropharvest")
TARGET_CROPS = ("carrot", "cabbage", "potato")


def archive_root(settings, report_folder):
    """An external SSD can be selected without moving reference data or old runs."""
    default = Path(report_folder).parent / "annual_archive"
    return Path(settings.get("annual_archive_root", default)).expanduser().resolve()


def load_sen4agrinet_references(field_root):
    """Use every indexed observation year; never silently equate a local subset with upstream coverage."""
    paths = sorted(Path(field_root).glob("*/download_candidates.geojson"))
    if not paths:
        raise FileNotFoundError(
            "Saved field inventory is missing: */download_candidates.geojson"
        )
    features = []
    for path in paths:
        records = json.loads(path.read_text())["features"]
        features.extend(
            normalize_sen4agrinet_feature(record, int(path.parent.name))
            for record in records
        )
    return features, {str(path.resolve()): file_sha256(path) for path in paths}


def normalize_sen4agrinet_feature(feature, year):
    feature = json.loads(json.dumps(feature))
    properties = feature["properties"]
    parcel_id = str(properties["parcel_id"])
    properties.update(
        year=year,
        partition_group=f"sen4agrinet|{properties['country']}|{year}|{parcel_id}",
    )
    properties.setdefault("source_split", "train")
    return feature


def grouped_footprints(features):
    """Union fragments before tiling so one parcel remains one statistical group."""
    grouped = defaultdict(list)
    for feature in features:
        properties = feature["properties"]
        if properties.get("crop") in TARGET_CROPS:
            grouped[
                (
                    int(properties["year"]),
                    properties["source_crs"],
                    properties["partition_group"],
                )
            ].append(feature)
    return [merge_footprints(records) for _, records in sorted(grouped.items())]


def merge_footprints(records):
    properties = dict(records[0]["properties"])
    properties["source_fragments"] = sorted(
        {record["properties"]["fragment_id"] for record in records}
    )
    if len({record["properties"]["crop"] for record in records}) != 1:
        raise ValueError(f"Conflicting crop labels for {properties['partition_group']}")
    geometries = [shape(record["geometry"]) for record in records]
    if any(
        geometry.is_empty
        or not geometry.is_valid
        or geometry.geom_type not in {"Polygon", "MultiPolygon"}
        for geometry in geometries
    ):
        raise ValueError(
            f"Invalid reference footprint: {properties['partition_group']}"
        )
    return {
        "type": "Feature",
        "geometry": mapping(unary_union(geometries)),
        "properties": properties,
    }


def projected_cohorts(features):
    cohorts = defaultdict(list)
    for feature in grouped_footprints(features):
        properties = feature["properties"]
        projector = Transformer.from_crs(
            "EPSG:4326", properties["source_crs"], always_xy=True
        )
        geometry = transform_geometry(projector.transform, shape(feature["geometry"]))
        cohorts[(properties["year"], properties["source_crs"])].append(
            (feature, geometry)
        )
    return cohorts


def uncontested_interior(index, records, spatial_index, inward_buffer_metres):
    """Exclude competing parcel claims rather than counting the same location twice."""
    geometry = records[index][1]
    interior = (
        geometry.buffer(-inward_buffer_metres) if inward_buffer_metres else geometry
    )
    neighbors = [
        records[int(other)][1]
        for other in spatial_index.query(geometry)
        if int(other) != index
    ]
    return interior.difference(unary_union(neighbors)) if neighbors else interior


def tile_bounds(geometry, tile_pixels):
    west, south, east, north = geometry.bounds
    tile_size = tile_pixels * 10
    left, bottom = (
        math.floor(west / tile_size) * tile_size,
        math.floor(south / tile_size) * tile_size,
    )
    for horizontal in range(left, math.ceil(east), tile_size):
        for vertical in range(bottom, math.ceil(north), tile_size):
            yield box(
                horizontal, vertical, horizontal + tile_size, vertical + tile_size
            )


def save_tile(feature, interior, bounds, field_folder, tile_number):
    clipped = interior.intersection(bounds)
    if clipped.is_empty:
        return None
    properties = dict(feature["properties"])
    projector = Transformer.from_crs(
        properties["source_crs"], "EPSG:4326", always_xy=True
    )
    # Tile rectangles bound network requests, while the mask retains only interior parcel cells.
    geometry = transform_geometry(projector.transform, clipped)
    identifier = hashlib.sha256(
        f"{properties['partition_group']}|{tile_number}|{bounds.bounds}".encode()
    ).hexdigest()[:24]
    region = {
        "type": "Feature",
        "geometry": mapping(geometry),
        "properties": {
            **properties,
            "fragment_id": f"annual_{identifier}",
            "archive_tile": tile_number,
        },
    }
    grid = sampling.download_grid(region)
    mask = rasterize(
        [(mapping(clipped), 1)],
        out_shape=(grid["height"], grid["width"]),
        transform=Affine(*grid["transform"]),
        dtype="uint8",
    )
    if not mask.any():
        return None
    path = (
        Path(field_folder)
        / "balanced_pilot/sampling_masks"
        / f"annual_{identifier}.tif"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=grid["width"],
        height=grid["height"],
        count=1,
        dtype="uint8",
        crs=grid["crs"],
        transform=Affine(*grid["transform"]),
        compress="deflate",
    ) as raster:
        raster.write(mask, 1)
        raster.set_band_description(1, "eligible_interior_pixel")
    region["properties"].update(
        selected_pixels=int(mask.sum()),
        sampling_mask=str(path.resolve()),
        sampling_grid=grid,
    )
    return region


def prepare_cohort(records, field_folder, inward_buffer_metres, tile_pixels):
    spatial_index = STRtree([geometry for _, geometry in records])
    regions, exclusions = [], []
    for index, (feature, _) in enumerate(records):
        interior = uncontested_interior(
            index, records, spatial_index, inward_buffer_metres
        )
        prepared = prepare_parcel_tiles(feature, interior, field_folder, tile_pixels)
        regions.extend(prepared)
        if not prepared:
            exclusions.append(
                {
                    "partition_group": feature["properties"]["partition_group"],
                    "reason": "no_uncontested_interior_pixel",
                }
            )
    return regions, exclusions


def prepare_parcel_tiles(feature, interior, field_folder, tile_pixels):
    if interior.is_empty:
        return []
    return [
        region
        for tile_number, bounds in enumerate(tile_bounds(interior, tile_pixels))
        if (region := save_tile(feature, interior, bounds, field_folder, tile_number))
        is not None
    ]


def prepare_source(
    source_name, features, root, provenance, inward_buffer_metres=10, tile_pixels=128
):
    """Prepare masks and exact raster estimates locally; never query Earth Engine."""
    if source_name not in SOURCE_NAMES:
        raise ValueError("Unknown annual source family")
    if not 1 <= tile_pixels <= 256 or inward_buffer_metres < 0:
        raise ValueError("Use tiles of 1–256 pixels and a nonnegative inward buffer.")
    selected = [
        feature
        for feature in features
        if feature["properties"].get("crop") in TARGET_CROPS
    ]
    if not selected:
        raise ValueError(f"No eligible target-crop references for {source_name}")
    configuration = {
        "source": source_name,
        "provenance": provenance,
        "footprints_sha256": hashlib.sha256(
            json.dumps(selected, sort_keys=True).encode()
        ).hexdigest(),
        "collection": COLLECTION_NAME,
        "inward_buffer_metres": inward_buffer_metres,
        "tile_pixels": tile_pixels,
        "cloud_probability_threshold": 40,
        "request_version": REQUEST_VERSION,
        "resampling": RESAMPLING_POLICY,
        "archive_recipe": "24_half_month_medians_and_observed_daily_indices",
        "helper_sha256": file_sha256(Path(__file__)),
        "retrieval_sha256": file_sha256(
            Path(__file__).parents[1] / "imagery/retrieval.py"
        ),
        "timelines_sha256": file_sha256(Path(timelines.__file__)),
        "sampling_sha256": file_sha256(Path(sampling.__file__)),
    }
    run_folder, revision = cropharvest.initialize_run(
        Path(root) / source_name, COLLECTION_NAME, configuration
    )
    year_regions, exclusions = defaultdict(list), []
    for (year, _), records in projected_cohorts(selected).items():
        regions, excluded = prepare_cohort(
            records,
            run_folder / "fields" / str(year),
            inward_buffer_metres,
            tile_pixels,
        )
        year_regions[year].extend(regions)
        exclusions.extend(excluded)
    save_reference_footprints(run_folder, selected)
    years = save_year_inputs(run_folder, year_regions)
    if not years:
        raise ValueError(f"No eligible interior cells remain for {source_name}")
    estimate = estimate_storage(years, run_folder)
    payload = {
        "source": source_name,
        "revision": revision,
        "run_folder": str(run_folder),
        "provenance": provenance,
        "years": years,
        "estimate": estimate,
        "exclusions": exclusions,
    }
    quality.save_json(run_folder / "collection_plan.json", payload)
    return payload


def save_reference_footprints(run_folder, features):
    """Keep original outlines so saved examples can distinguish parcels from archive tiles."""
    by_year = defaultdict(list)
    for feature in features:
        by_year[int(feature["properties"]["year"])].append(feature)
    for year, records in by_year.items():
        quality.save_json(
            run_folder / "fields" / str(year) / "source_footprints.geojson",
            {"type": "FeatureCollection", "features": records},
        )


def save_year_inputs(run_folder, year_regions):
    years = {}
    for year, regions in sorted(year_regions.items()):
        if not regions:
            continue
        field_folder = run_folder / "fields" / str(year)
        quality.save_json(
            field_folder / "sampled_download_regions.geojson",
            {"type": "FeatureCollection", "features": regions},
        )
        years[str(year)] = {
            "field_folder": str(field_folder),
            "imagery_folder": str(run_folder / "imagery" / str(year)),
            "report_folder": str(run_folder / "reports" / str(year)),
            "timeline_folder": str(run_folder / "timelines" / str(year)),
            "regions": len(regions),
            "pixels_per_crop": dict(count_pixels_by_crop(regions)),
        }
    return years


def count_pixels_by_crop(regions):
    counts = Counter()
    for region in regions:
        counts[region["properties"]["crop"]] += region["properties"]["selected_pixels"]
    return counts


def estimate_storage(years, root):
    regions = [region for inputs in years.values() for region in read_regions(inputs)]
    grid_cells = sum(
        region["properties"]["sampling_grid"]["width"]
        * region["properties"]["sampling_grid"]["height"]
        for region in regions
    )
    raw_bytes = grid_cells * 24 * 7 * 4
    return {
        "parcels": len({region["properties"]["partition_group"] for region in regions}),
        "tiles": len(regions),
        "composites": len(regions) * 24,
        "raw_raster_bytes": raw_bytes,
        "raw_raster_gib": round(raw_bytes / 1024**3, 3),
        "recommended_free_bytes": raw_bytes * 2,
        "available_bytes": shutil.disk_usage(root).free,
        "note": "Uncompressed float32 rasters; allow additional disk for summaries, features and models. Compression varies.",
    }


def read_regions(inputs):
    return json.loads(
        (Path(inputs["field_folder"]) / "sampled_download_regions.geojson").read_text()
    )["features"]


def collect_source(plan, report_folder, project, enabled=False, authenticate=False):
    """Use EE for masks/medians and checkpoint local rasters; disabled means zero network calls."""
    if not enabled:
        return {
            "status": "disabled",
            "source": plan["source"],
            "estimate": plan["estimate"],
        }
    if not project:
        raise ValueError("Set earth_engine_project before enabling collection.")
    if (
        shutil.disk_usage(plan["run_folder"]).free
        < plan["estimate"]["recommended_free_bytes"]
    ):
        raise OSError(
            "Insufficient free archive space; select annual_archive_root on a larger drive."
        )
    import ee

    if authenticate:
        ee.Authenticate()
    ee.Initialize(project=project)
    handoffs = [
        collect_year(ee, plan, year, inputs)
        for year, inputs in sorted(plan["years"].items())
    ]
    return register_source(report_folder, plan, handoffs)


def collect_year(earth_engine, source_plan, year, inputs):
    """One bounded tile at a time keeps annual rasters and dated summaries resumable."""
    retriever = SatelliteRetriever(
        earth_engine, int(year), inputs["imagery_folder"], inputs["report_folder"], 40
    )
    regions = read_regions(inputs)
    for region in regions:
        collect_region(retriever, region, inputs)
    handoff_path = Path(source_plan["run_folder"]) / "handoffs" / f"{year}.json"
    manifest = pilot_sources.save_source_manifest(
        handoff_path,
        f"{source_plan['source']}_{year}",
        int(year),
        inputs["field_folder"],
        inputs["imagery_folder"],
        Path(inputs["report_folder"]) / "composite_progress",
        inputs["timeline_folder"],
        source_plan["provenance"],
    )
    manifest.update(
        collection=COLLECTION_NAME,
        source_family=source_plan["source"],
        collection_revision=source_plan["revision"],
        archive_period=f"{year}-01-01/{int(year)+1}-01-01",
        candidate_target_per_crop=None,
        usable_target_per_crop=None,
    )
    quality.save_json(handoff_path, manifest)
    return handoff_path


def collect_region(retriever, region, inputs):
    plan = retriever.inventory_region(region)
    plan["download_intervals"] = quality.interval_bins(retriever.year)
    rows = timelines.retrieve_observed_timelines(
        [plan],
        retriever.mask_acquisition,
        retriever.ee,
        Path(inputs["timeline_folder"]) / "tiles" / plan["fragment_id"],
        0.8,
    )
    checkpoint = list(
        (
            Path(inputs["timeline_folder"])
            / "tiles"
            / plan["fragment_id"]
            / "observed_checkpoints"
        ).glob("*.json")
    )
    if not checkpoint:
        raise RuntimeError(
            f"Daily summaries incomplete for {plan['fragment_id']}; resume collection."
        )
    results = retriever.download_region(plan)
    sampling_result = {
        "downloaded": sum(row["status"] == "downloaded" for row in results),
        "reused": sum(row["status"] == "reused" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "results": results,
    }
    from crop_pipeline.imagery.bilinear import verify_download_completion

    verify_download_completion(
        [plan],
        sampling_result,
        [region["properties"]["crop"]],
        Path(inputs["report_folder"]) / "completion" / f"{plan['fragment_id']}.json",
    )
    return rows


def registry_path(report_folder, source):
    return (
        Path(report_folder).parent
        / "source_handoffs"
        / COLLECTION_NAME
        / f"{source}.json"
    )


def register_source(report_folder, plan, handoffs):
    """Publish a pointer only after every eligible year and tile has completed."""
    if len(handoffs) != len(plan["years"]):
        raise ValueError("All planned years must complete before registration.")
    bundle = {
        "collection": COLLECTION_NAME,
        "source": plan["source"],
        "revision": plan["revision"],
        "handoffs": [
            {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
            for path in handoffs
        ],
    }
    path = registry_path(report_folder, plan["source"])
    quality.save_json(path, bundle)
    return {"status": "completed", "registry": str(path), "source": plan["source"]}


def resolve_collection(report_folder):
    """Require all three completed sources and derive a separate, content-specific analysis."""
    registry_paths = [registry_path(report_folder, source) for source in SOURCE_NAMES]
    missing = [str(path) for path in registry_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required full-year source handoffs missing: " + ", ".join(missing)
        )
    paths = [
        path
        for source, registry in zip(SOURCE_NAMES, registry_paths)
        for path in validate_bundle(source, registry)
    ]
    digest = hashlib.sha256(
        "".join(file_sha256(path) for path in registry_paths).encode()
    ).hexdigest()[:16]
    return paths, f"{COLLECTION_NAME}_{digest}"


def validate_bundle(source, path):
    bundle = json.loads(path.read_text())
    if (
        bundle.get("collection") != COLLECTION_NAME
        or bundle.get("source") != source
        or not bundle.get("handoffs")
    ):
        raise ValueError(f"Invalid annual source bundle: {path}")
    for entry in bundle["handoffs"]:
        if file_sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Modified annual handoff: {entry['path']}")
    return [Path(entry["path"]) for entry in bundle["handoffs"]]
