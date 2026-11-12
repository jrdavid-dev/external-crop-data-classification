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
    REQUEST_VERSION,
    RESAMPLING_POLICY,
)
from crop_pipeline.imagery.accelerated import AcceleratedRetriever as SatelliteRetriever
from crop_pipeline.sources import cropharvest, pilot_sources, sampling
from crop_pipeline.sources import annual_identity, collection_pass, preparation_cache
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


def valid_request_geometry(geometry):
    """Remove collapsed edges introduced by clipping/reprojection, keeping polygon area.

    Request outlines bound retrieval only; eligible pixels are rasterized from the
    original projected parcel intersection, never from this repaired outline.
    """
    if geometry.is_valid and geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    # A hull uses existing vertices, preserves collapsed boundary extrema, and
    # encloses all original pieces. It is only a retrieval boundary: the original
    # projected intersection below still defines every eligible pixel.
    result = geometry.convex_hull
    if result.is_empty or result.area == 0:
        return None
    if not result.is_valid or result.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("Could not construct a valid polygonal imagery request outline")
    return result


def save_tile(feature, interior, bounds, field_folder, tile_number):
    clipped = interior.intersection(bounds)
    if clipped.is_empty:
        return None
    properties = dict(feature["properties"])
    projector = Transformer.from_crs(
        properties["source_crs"], "EPSG:4326", always_xy=True
    )
    # Tile rectangles bound network requests, while the mask retains only interior parcel cells.
    geometry = valid_request_geometry(transform_geometry(projector.transform, clipped))
    if geometry is None:
        return None
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


def prepare_selected_cohort(records, field_folder, inward_buffer_metres, tile_pixels, candidate_groups):
    """Rasterize only candidates, but screen overlaps against every reference parcel."""
    spatial_index = STRtree([geometry for _, geometry in records])
    regions, exclusions = [], []
    for index, (feature, _) in enumerate(records):
        if feature["properties"]["partition_group"] not in candidate_groups:
            continue
        interior = uncontested_interior(index, records, spatial_index, inward_buffer_metres)
        prepared = prepare_parcel_tiles(feature, interior, field_folder, tile_pixels)
        regions.extend(prepared)
        if not prepared:
            exclusions.append({"partition_group": feature["properties"]["partition_group"],
                               "reason": "no_uncontested_interior_pixel"})
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
    source_name, features, root, provenance, inward_buffer_metres=10, tile_pixels=128,
    candidate_groups=None,
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
        "planning_sha256": annual_identity.planning_fingerprint(Path(__file__)),
        "retrieval_sha256": file_sha256(
            Path(__file__).parents[1] / "imagery/retrieval.py"
        ),
        "timelines_sha256": file_sha256(Path(timelines.__file__)),
        "sampling_sha256": file_sha256(Path(sampling.__file__)),
    }
    if candidate_groups is not None:
        import inspect
        candidate_groups = set(candidate_groups)
        available = {feature["properties"]["partition_group"] for feature in selected}
        if not candidate_groups or candidate_groups - available:
            raise ValueError("Candidate groups must identify eligible reference parcels")
        configuration["candidate_groups"] = sorted(candidate_groups)
        configuration["candidate_planning_sha256"] = hashlib.sha256(
            inspect.getsource(prepare_selected_cohort).encode()).hexdigest()
    run_folder, revision = annual_identity.initialize_archive(
        Path(root) / source_name, COLLECTION_NAME, configuration
    )
    cached = preparation_cache.load_prepared(run_folder)
    if cached is not None:
        cached["estimate"]["available_bytes"] = shutil.disk_usage(run_folder).free
        return cached
    year_regions, exclusions = defaultdict(list), []
    for (year, _), records in projected_cohorts(selected).items():
        if candidate_groups is None:
            regions, excluded = prepare_cohort(
                records, run_folder / "fields" / str(year), inward_buffer_metres, tile_pixels)
        else:
            regions, excluded = prepare_selected_cohort(
                records, run_folder / "fields" / str(year), inward_buffer_metres,
                tile_pixels, candidate_groups)
        year_regions[year].extend(regions)
        exclusions.extend(excluded)
    save_reference_footprints(run_folder, [feature for feature in selected
        if candidate_groups is None or feature["properties"]["partition_group"] in candidate_groups])
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
    preparation_cache.save_receipt(payload)
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


def collect_source(
    plan, report_folder, project, enabled=False, authenticate=False,
    download_workers=2, intervals_per_request=6,
    collection_backend="direct", tile_start=0, tile_count=None,
    archive_format="rasters", submit_only=False,
):
    """Use EE for masks/medians and checkpoint local rasters; disabled means zero network calls."""
    if type(tile_start) is not int or tile_start < 0 or (
        tile_count is not None and (type(tile_count) is not int or tile_count <= 0)
    ):
        raise ValueError("Use a nonnegative tile_start and a positive tile_count or None.")
    if collection_backend != "batch" and (tile_start or tile_count is not None):
        raise ValueError("Tile ranges require collection_backend='batch'.")
    if submit_only and collection_backend != "batch":
        raise ValueError("Submit-only mode requires the batch backend.")
    if archive_format not in {"rasters", "pixel_tables"}:
        raise ValueError("archive_format must be rasters or pixel_tables")
    if archive_format == "pixel_tables" and collection_backend != "batch":
        raise ValueError("Pixel tables require the batch backend")
    if collection_backend not in {"direct", "batch"}:
        raise ValueError("collection_backend must be direct or batch")
    if not 1 <= download_workers <= 4 or not 1 <= intervals_per_request <= 24:
        raise ValueError("Use 1–4 download workers and 1–24 intervals per request.")
    if not enabled:
        return {
            "status": "disabled",
            "source": plan["source"],
            "estimate": plan["estimate"],
        }
    if not project:
        raise ValueError("Set earth_engine_project before enabling collection.")
    if archive_format == "pixel_tables":
        # Allow for raw table columns, transfer buffers, Parquet and receipts.
        # This is a conservative planning allowance, not a compressed-size prediction.
        selected_pixels = sum(
            region["properties"]["selected_pixels"]
            for inputs in plan["years"].values()
            for region in read_regions(inputs)[tile_start:None if tile_count is None else tile_start + tile_count]
        )
        required_space = selected_pixels * 24 * 128 + 64 * 1024**2
    else:
        required_space = collection_pass.estimated_required_space(plan)
    if shutil.disk_usage(plan["run_folder"]).free < required_space:
        raise OSError(
            "Insufficient free archive space; select annual_archive_root on a larger drive."
        )
    import ee

    if authenticate:
        ee.Authenticate()
    ee.Initialize(project=project)
    attempt_folder = collection_pass.start_attempt(plan, report_folder)
    quality.save_json(
        attempt_folder / "runtime_code.json",
        {
            "collector_sha256": file_sha256(Path(__file__)),
            "daily_summaries_sha256": file_sha256(Path(timelines.__file__)),
            "collection_backend": collection_backend, "archive_format": archive_format,
            "tile_start": tile_start, "tile_count": tile_count,
            "pixel_tables_sha256": file_sha256(
                Path(__file__).parents[1] / "imagery/pixel_tables.py"
            ),
            "batch_transport_sha256": file_sha256(
                Path(__file__).parents[1] / "imagery/batch_export.py"
            ),
            "batch_orchestration_sha256": file_sha256(
                Path(__file__).with_name("annual_batch.py")
            ),
            "download_workers": download_workers,
            "intervals_per_request": intervals_per_request,
            "transport_sha256": file_sha256(
                Path(__file__).parents[1] / "imagery/accelerated.py"
            ),
            "scene_cache_sha256": file_sha256(
                Path(__file__).parents[1] / "imagery/scene_cache.py"
            ),
        },
    )
    from crop_pipeline.sources.annual_batch import collect_batch_year

    year_collector = collect_batch_year if collection_backend == "batch" else collect_year
    reports = [
        year_collector(
            ee, plan, year, inputs, attempt_folder,
            download_workers, intervals_per_request,
            **({"project": project, "tile_start": tile_start, "tile_count": tile_count,
                "archive_format": archive_format, **({"submit_only": True} if submit_only else {})}
               if collection_backend == "batch" else {}),
        )
        for year, inputs in sorted(plan["years"].items())
    ]
    result = collection_pass.finish_attempt(
        plan, report_folder, attempt_folder, reports
    )
    if result["status"] != "completed":
        return result
    registration = register_source(
        report_folder, plan, [Path(report["handoff"]) for report in reports]
    )
    quality.save_json(
        collection_pass.state_path(report_folder, plan["source"]),
        {**result, **registration},
    )
    return {**result, **registration}


def collect_year(
    earth_engine, source_plan, year, inputs, attempt_folder=None,
    download_workers=2, intervals_per_request=6,
):
    """Tiles remain serial; bounded concurrent image groups reduce request overhead."""
    retriever = SatelliteRetriever(
        earth_engine, int(year), inputs["imagery_folder"], inputs["report_folder"], 40
    )
    retriever.download_workers = download_workers
    retriever.intervals_per_request = intervals_per_request
    regions = read_regions(inputs)
    outcomes = [
        collect_tile_safely(retriever, region, inputs, year, attempt_folder)
        for region in regions
    ]
    retry_queue = [
        {"year": str(year), **outcome}
        for outcome in outcomes
        if outcome["status"] != "complete"
    ]
    report = {
        "year": str(year),
        "planned_tiles": len(regions),
        "completed_tiles": len(regions) - len(retry_queue),
        "downloaded_images": sum(outcome["downloaded"] for outcome in outcomes),
        "reused_images": sum(outcome["reused"] for outcome in outcomes),
        "pending_images": sum(len(outcome["pending_images"]) for outcome in outcomes),
        "retry_queue": retry_queue,
    }
    if retry_queue:
        return report
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
    return {**report, "handoff": str(handoff_path.resolve())}


def collect_region(retriever, region, inputs):
    plan = retriever.inventory_region(region)
    plan["download_intervals"] = quality.interval_bins(retriever.year)
    errors = []
    timeline_folder = Path(inputs["timeline_folder"]) / "tiles" / plan["fragment_id"]
    daily_complete = collect_daily_safely(retriever, plan, timeline_folder, errors)
    results = collect_images_safely(retriever, plan, errors)
    sampling_result = {
        "downloaded": sum(row["status"] == "downloaded" for row in results),
        "reused": sum(row["status"] == "reused" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "results": results,
    }
    pending_images = collection_pass.image_failures(
        plan, results, inputs["report_folder"]
    )
    images_complete = verify_tile_completion(plan, sampling_result, inputs, errors)
    return {
        "region_id": plan["fragment_id"],
        "status": "complete" if daily_complete and images_complete else "pending_retry",
        "daily_complete": daily_complete,
        "downloaded": sampling_result["downloaded"],
        "reused": sampling_result["reused"],
        "pending_images": pending_images,
        "errors": errors,
    }


def collect_daily_safely(retriever, plan, timeline_folder, errors):
    try:
        timelines.retrieve_observed_timelines(
            [plan], retriever.mask_acquisition, retriever.ee, timeline_folder, 0.8
        )
        if not list((timeline_folder / "observed_checkpoints").glob("*.json")):
            raise RuntimeError("Daily summaries incomplete; no checkpoint was produced")
        return True
    except Exception as error:
        errors.append({"stage": "daily_summaries", "error": str(error)})
        return False


def collect_images_safely(retriever, plan, errors):
    try:
        return retriever.download_region(plan)
    except Exception as error:
        errors.append({"stage": "imagery", "error": str(error)})
        return []


def verify_tile_completion(plan, summary, inputs, errors):
    from crop_pipeline.imagery.bilinear import verify_download_completion

    try:
        verify_download_completion(
            [plan],
            summary,
            [plan["fragment"]["properties"]["crop"]],
            Path(inputs["report_folder"])
            / "completion"
            / f"{plan['fragment_id']}.json",
        )
        return True
    except Exception as error:
        errors.append({"stage": "completion", "error": str(error)})
        return False


def collect_tile_safely(retriever, region, inputs, year, attempt_folder):
    try:
        result = collect_region(retriever, region, inputs)
    except Exception as error:
        result = {
            "region_id": region["properties"]["fragment_id"],
            "status": "pending_retry",
            "daily_complete": False,
            "downloaded": 0,
            "reused": 0,
            "pending_images": [
                {
                    "interval": interval["name"],
                    "error": "Tile planning or collection failed",
                }
                for interval in quality.interval_bins(retriever.year)
            ],
            "errors": [{"stage": "tile", "error": str(error)}],
        }
    collection_pass.write_event(attempt_folder, year, result)
    return result


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
    states = [
        collection_pass.state_path(report_folder, source) for source in SOURCE_NAMES
    ]
    pending = [
        str(path)
        for path in states
        if path.is_file() and json.loads(path.read_text()).get("status") != "completed"
    ]
    if pending:
        raise RuntimeError(
            "Full-year source collections are still pending; finish or retry them before window selection: "
            + ", ".join(pending)
        )
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
