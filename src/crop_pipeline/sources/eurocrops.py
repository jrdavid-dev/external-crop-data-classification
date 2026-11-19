"""Read declared Dutch carrot production parcels from the local EuroCrops release."""

import json
from pathlib import Path

import pyarrow.parquet as parquet
from pyproj import CRS, Transformer
from shapely import from_wkb
from shapely.geometry import mapping
from shapely.ops import transform

SOURCE = "eurocrops"
YEAR = 2021
CARROT_PRODUCTION_CODES = {"2717", "2783", "2785"}
CABBAGE_PRODUCTION_CODES = {"2759", "2761", "2789"}


def load_carrots(path):
    return _load_crop(path, "carrot", CARROT_PRODUCTION_CODES)


def load_cabbage(path):
    return _load_crop(path, "cabbage", CABBAGE_PRODUCTION_CODES)


def _load_crop(path, crop, production_codes):
    path = Path(path)
    if path.name != "NL_2021.parquet":
        raise ValueError("Provide the Dutch EuroCrops NL_2021.parquet file")
    data = parquet.ParquetFile(path)
    geometry_metadata = json.loads(data.schema_arrow.metadata[b"geo"])["columns"]["geometry"]
    if geometry_metadata["encoding"] != "WKB" or not geometry_metadata.get("crs"):
        raise ValueError("EuroCrops requires WKB geometry with a declared CRS")
    projector = Transformer.from_crs(CRS.from_user_input(geometry_metadata["crs"]), 4326, always_xy=True)
    features = {}
    seen = set()
    rejected = set()
    for batch in data.iter_batches(columns=["cropfield", "original_code", "geometry"]):
        for row in batch.to_pylist():
            code, identifier = str(row["original_code"]), row["cropfield"]
            if identifier is None:
                continue
            parcel = str(identifier)
            if parcel in seen:
                rejected.add(parcel)
            seen.add(parcel)
            if code not in production_codes:
                continue
            geometry = from_wkb(row["geometry"], on_invalid="ignore")
            if geometry is None or geometry.is_empty or not geometry.is_valid or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                rejected.add(parcel)
                continue
            geometry = transform(projector.transform, geometry)
            west, south, east, north = geometry.bounds
            if not (3 <= west <= east <= 8 and 50 <= south <= north <= 54):
                rejected.add(parcel)
                continue
            group = f"{SOURCE}|NL|{YEAR}|{parcel}"
            features[parcel] = {
                "type": "Feature", "geometry": mapping(geometry),
                "properties": {
                    "fragment_id": group, "partition_group": group, "parcel_id": parcel,
                    "crop": crop, "country": "NL", "year": YEAR,
                    "source_crs": "EPSG:32631", "source_split": "train",
                    "source_class_code": code, "label_status": "reported",
                    "label_evidence": f"EuroCrops Netherlands {YEAR} declared production code {code}",
                },
            }
    return [features[key] for key in sorted(features.keys() - rejected)]
