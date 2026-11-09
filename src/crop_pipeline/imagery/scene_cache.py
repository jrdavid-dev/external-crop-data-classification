"""Reuse scene metadata across nearby tiles, retaining exact footprint filtering."""

import hashlib
import json
import math

from pyproj import Transformer
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as transform_geometry

from crop_pipeline.common import save_json


def query_region(region):
    """Projected 10 km blocks keep cache keys stable for neighboring parcels."""
    source_crs = region["properties"]["source_crs"]
    projector = Transformer.from_crs("EPSG:4326", source_crs, always_xy=True)
    west, south, east, north = transform_geometry(
        projector.transform, shape(region["geometry"])
    ).bounds
    # A centimetre of padding makes block selection stable at projection roundoff boundaries.
    bounds = (
        math.floor((west - 0.01) / 10000) * 10000,
        math.floor((south - 0.01) / 10000) * 10000,
        (math.floor((east + 0.01) / 10000) + 1) * 10000,
        (math.floor((north + 0.01) / 10000) + 1) * 10000,
    )
    # Densification avoids excluding tile corners when transforming a block boundary.
    boundary = box(*bounds).segmentize(100)
    inverse = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    geometry = mapping(transform_geometry(inverse.transform, boundary))
    return geometry, {"crs": source_crs, "bounds": bounds}


def inventory_region(retriever, region, refresh=False):
    settings = json.loads(json.dumps(retriever.request_settings(region)))
    region_id = region["properties"]["fragment_id"]
    path = retriever.report_folder / "acquisition_inventory" / f"{region_id}.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    if saved.get("settings") == settings and not refresh:
        return saved
    metadata = shared_metadata(retriever, region, refresh)
    footprint = shape(region["geometry"])
    cloud_indices = {
        feature["properties"]["image_index"]
        for feature in metadata["cloud_features"]
        if shape(feature["geometry"]).intersects(footprint)
    }
    acquisitions = [
        matched_acquisition(row, footprint, cloud_indices)
        for row in metadata["features"]
    ]
    acquisitions = [row for row in acquisitions if row is not None]
    acquisitions = sorted(
        acquisitions, key=lambda row: (row["timestamp_ms"], row["image_index"])
    )
    plan = {
        "fragment_id": region_id,
        "fragment": region,
        "settings": settings,
        "acquisitions": acquisitions,
    }
    save_json(path, plan)
    return plan


def shared_metadata(retriever, region, refresh):
    geometry, block = query_region(region)
    identity = {
        "version": 2,
        "block": block,
        "start": retriever.start_date,
        "end": retriever.end_date,
        "collection": retriever.collection_name,
        "clouds": retriever.cloud_probability_collection,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = retriever.report_folder / "shared_scene_inventory" / f"{digest}.json"
    if path.exists() and not refresh:
        saved = json.loads(path.read_text())
        if saved.get("identity") == json.loads(json.dumps(identity)):
            return saved["metadata"]
    metadata = fetch_metadata(retriever, geometry)
    save_json(path, {"identity": identity, "metadata": metadata})
    return metadata


def fetch_metadata(retriever, geometry):
    earth_engine = retriever.ee
    footprint = earth_engine.Geometry(geometry, proj="EPSG:4326", geodesic=False)
    collection = (
        earth_engine.ImageCollection(retriever.collection_name)
        .filterBounds(footprint)
        .filterDate(retriever.start_date, retriever.end_date)
    )
    clouds = (
        earth_engine.ImageCollection(retriever.cloud_probability_collection)
        .filterBounds(footprint)
        .filterDate(retriever.start_date, retriever.end_date)
    )

    def describe_cloud(image):
        image = earth_engine.Image(image)
        return earth_engine.Feature(
            image.geometry(), {"image_index": image.get("system:index")}
        )

    cloud_metadata = earth_engine.FeatureCollection(
        clouds.toList(clouds.size()).map(describe_cloud)
    ).getInfo()
    cloud_indices = {
        feature["properties"]["image_index"] for feature in cloud_metadata["features"]
    }

    def describe(image):
        image = earth_engine.Image(image)
        return earth_engine.Feature(
            image.geometry(),
            {
                "image_index": image.get("system:index"),
                "timestamp_ms": image.get("system:time_start"),
                "date": earth_engine.Date(image.get("system:time_start")).format(
                    "YYYY-MM-dd"
                ),
                "tile": image.get("MGRS_TILE"),
                "scene_cloud_percent": image.get("CLOUDY_PIXEL_PERCENTAGE"),
            },
        )

    metadata = earth_engine.FeatureCollection(
        collection.toList(collection.size()).map(describe)
    ).getInfo()
    for feature in metadata["features"]:
        properties = feature["properties"]
        image_index = properties["image_index"]
        properties["image_id"] = retriever.collection_name + "/" + image_index
        properties["cloud_probability_id"] = (
            retriever.cloud_probability_collection + "/" + image_index
            if image_index in cloud_indices
            else None
        )
    metadata["cloud_features"] = cloud_metadata["features"]
    return metadata


def matched_acquisition(feature, footprint, cloud_indices):
    if not shape(feature["geometry"]).intersects(footprint):
        return None
    # A cloud image that only overlaps a nearby tile is not a valid exact match here.
    properties = dict(feature["properties"])
    if properties["image_index"] not in cloud_indices:
        properties["cloud_probability_id"] = None
    return properties
