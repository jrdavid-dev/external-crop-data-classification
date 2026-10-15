"""FranceCrops cabbage footprints for a consistent, uninterpolated imagery pilot."""

import json
import math
from pathlib import Path

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from shapely import wkb
from shapely.geometry import mapping, shape
from shapely.ops import transform as transform_geometry

from crop_workflow import save_json
from pilot_sampling import stable_rank

DATASET_NAME = "saget-antoine/francecrops"
DEFAULT_REVISION = "6b6c51b780d8579138593f92738ed52f2528096e"
SOURCE_YEAR = 2022
TABLE_FILES = {
    "class_map": "data/protocol/class_map.parquet",
    "metadata": "data/protocol/metadata.parquet",
    "geolocation": "data/geolocation/geolocation.parquet",
}


def download_reference_tables(destination, revision=DEFAULT_REVISION):
    """Fetch labels and footprints only; the large interpolated arrays are unnecessary."""
    from huggingface_hub import hf_hub_download

    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Pin FranceCrops to a full immutable dataset revision.")
    paths = {}
    for name, filename in TABLE_FILES.items():
        path = Path(destination) / filename
        if not path.is_file():
            path = Path(
                hf_hub_download(
                    repo_id=DATASET_NAME,
                    repo_type="dataset",
                    revision=revision,
                    filename=filename,
                    local_dir=str(destination),
                )
            )
        paths[name] = path
    return paths


def load_cabbage_parcels(table_paths, revision=DEFAULT_REVISION, audit_path=None):
    """Join explicit CHU training labels to WGS84 geometry; never infer cabbage from a curve."""
    import pyarrow.parquet as parquet

    classes = parquet.read_table(table_paths["class_map"]).to_pylist()
    matching_classes = [row for row in classes if row["class_code"] == "CHU"]
    if len(matching_classes) != 1 or matching_classes[0]["class_name_en"].casefold() != "cabbage":
        raise ValueError("FranceCrops class map does not identify exactly one CHU cabbage class.")
    cabbage_label = int(matching_classes[0]["y"])
    labels = parquet.read_table(
        table_paths["metadata"], filters=[("split", "=", "train"), ("y", "=", cabbage_label)]
    ).to_pylist()
    parcel_ids = [str(row["parcel_id"]) for row in labels]
    if not parcel_ids or len(parcel_ids) != len(set(parcel_ids)):
        raise ValueError("Cabbage training labels are empty or contain duplicate parcel IDs.")
    geometry_table = parquet.read_table(
        table_paths["geolocation"], filters=[("parcel_id", "in", parcel_ids)]
    )
    validate_geolocation_crs(geometry_table.schema.metadata)
    geometry_rows = geometry_table.to_pylist()
    joined_ids = [str(row["parcel_id"]) for row in geometry_rows]
    if set(joined_ids) != set(parcel_ids) or len(joined_ids) != len(set(joined_ids)):
        raise ValueError("Cabbage parcel geometry join is incomplete or duplicated.")
    features, exclusions = [], []
    for row in geometry_rows:
        try:
            features.append(parcel_feature(row, revision, cabbage_label))
        except ValueError as error:
            exclusions.append({"parcel_id": str(row["parcel_id"]), "reason": str(error)})
    if audit_path is not None:
        save_json(
            audit_path,
            {
                "training_cabbage_parcels": len(labels),
                "valid_footprints": len(features),
                "excluded_footprints": exclusions,
            },
        )
    if not features:
        raise ValueError("No valid cabbage footprints remain.")
    return features


def validate_geolocation_crs(metadata):
    if not metadata or b"geo" not in metadata:
        raise ValueError("GeoParquet CRS metadata is required before interpreting coordinates.")
    geographic = json.loads(metadata[b"geo"])
    coordinate_system = geographic["columns"]["geometry"].get("crs")
    if coordinate_system is None or not CRS.from_user_input(coordinate_system).equals(
        CRS.from_epsg(4326), ignore_axis_order=True
    ):
        raise ValueError("FranceCrops geometry must explicitly declare WGS84 coordinates.")


def parcel_feature(row, revision, cabbage_label):
    geometry = wkb.loads(bytes(row["geometry"]))
    if (
        geometry.is_empty
        or not geometry.is_valid
        or geometry.geom_type not in {"Polygon", "MultiPolygon"}
    ):
        raise ValueError(f"Invalid parcel footprint: {row['parcel_id']}")
    longitude, latitude = geometry.centroid.coords[0]
    if not -6 <= longitude <= 10 or not 41 <= latitude <= 52:
        raise ValueError("Parcel is outside metropolitan France; inspect coordinates and CRS.")
    zone = int(math.floor((longitude + 180) / 6)) + 1
    parcel_id = str(row["parcel_id"])
    return {
        "type": "Feature",
        "geometry": mapping(geometry),
        "properties": {
            "fragment_id": f"francecrops_2022_{parcel_id}",
            "parcel_id": parcel_id,
            "partition_group": f"francecrops|FR|2022|{parcel_id}",
            "crop": "cabbage",
            "country": "FR",
            "year": SOURCE_YEAR,
            "source_crs": f"EPSG:{32600 + zone}",
            "dataset": DATASET_NAME,
            "dataset_revision": revision,
            "source_split": "train",
            "source_class_code": "CHU",
            "source_class_id": cabbage_label,
            "label_status": "reported",
            "label_evidence": f"FranceCrops {revision}, IGN RPG 2022 declared crop CHU",
        },
    }


def parcel_pixel_candidates(feature, maximum_pixels=100, seed=42, inward_buffer_metres=10):
    """Use interior 10 m cells to reduce boundary mixing; keep at most one field quota."""
    properties = feature["properties"]
    projector = Transformer.from_crs("EPSG:4326", properties["source_crs"], always_xy=True)
    geometry = transform_geometry(projector.transform, shape(feature["geometry"])).buffer(
        -inward_buffer_metres
    )
    if geometry.is_empty:
        return []
    west, south, east, north = geometry.bounds
    left, top = math.floor(west / 10) * 10, math.ceil(north / 10) * 10
    width, height = math.ceil((east - left) / 10), math.ceil((top - south) / 10)
    if width < 1 or height < 1 or width * height > 5000000:
        return []
    transform = Affine(10, 0, left, 0, -10, top)
    mask = rasterize(
        [(mapping(geometry), 1)], out_shape=(height, width), transform=transform, dtype="uint8"
    )
    raster_rows, columns = np.where(mask == 1)
    generator = np.random.default_rng(int(stable_rank(properties["parcel_id"], seed)[:16], 16))
    chosen = generator.choice(len(columns), size=min(maximum_pixels, len(columns)), replace=False)
    return [
        {
            "source_crs": properties["source_crs"],
            "x": left + int(columns[index]) * 10 + 5,
            "y": top - int(raster_rows[index]) * 10 - 5,
            "crop": "cabbage",
            "partition_group": properties["partition_group"],
            "source_fragments": [properties["fragment_id"]],
        }
        for index in sorted(chosen)
    ]


def build_cabbage_catalog(features, seed=42):
    """Exclude overlapping field claims so the candidate count reflects unique locations."""
    records = [row for feature in features for row in parcel_pixel_candidates(feature, seed=seed)]
    by_location = {}
    conflicting = set()
    for row in records:
        key = (row["source_crs"], row["x"], row["y"])
        previous = by_location.setdefault(key, row)
        if previous["partition_group"] != row["partition_group"]:
            conflicting.add(key)
    return [row for key, row in by_location.items() if key not in conflicting]
