"""Load Danish 2023 Marker cabbage declarations as stable parcel polygons."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryFile

import pyarrow.parquet as pq
import requests
from pyproj import CRS, Transformer
from shapely import from_wkb
from shapely.geometry import mapping
from shapely.ops import transform

from crop_pipeline.common import save_json
from crop_pipeline.sources.cropharvest import utm_crs

MARKER_2023 = "https://data.source.coop/fiboa/denmark/denmark.parquet"
CABBAGE_CODES = ("408", "418")


def cabbage_features(rows, crs):
    """Use normalized geometry for identity because Marker IDs repeat."""
    to_lonlat = Transformer.from_crs(crs, "OGC:CRS84", always_xy=True)
    features = {}
    for row in rows:
        if str(row["crop_code"]) not in CABBAGE_CODES or row["geometry"] is None:
            continue
        geom = from_wkb(row["geometry"])
        if geom.is_empty or not geom.is_valid or geom.geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        geom = geom.normalize()
        parcel = hashlib.sha256(geom.wkb).hexdigest()[:24]
        group = f"denmark_marker|DK|2023|{parcel}"
        if group in features:
            continue
        geom = transform(to_lonlat.transform, geom)
        centroid = geom.centroid
        features[group] = {
            "type": "Feature", "geometry": mapping(geom), "properties": {
                "fragment_id": group, "partition_group": group, "parcel_id": parcel,
                "crop": "cabbage", "country": "DK", "year": 2023, "source_split": "train",
                "source_crs": utm_crs(centroid.x, centroid.y), "label_status": "reported",
                "label_evidence": "Marker 2023 declared white or red cabbage (408/418)",
            },
        }
    return [features[key] for key in sorted(features)]


def load_cabbage(path):
    """Cache the filtered reference snapshot for repeatable collection runs."""
    path = Path(path)
    if not path.exists():
        print("Downloading the Marker 2023 GeoParquet (about 400 MB, first run only).")
        with TemporaryFile() as stream:
            with requests.get(MARKER_2023, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                for block in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(block)
            stream.seek(0)
            parquet = pq.ParquetFile(stream)
            geometry = json.loads(parquet.schema_arrow.metadata[b"geo"])["columns"]["geometry"]
            crs = CRS.from_user_input(geometry.get("crs", "OGC:CRS84"))
            rows = pq.read_table(
                stream, columns=["crop_code", "geometry"],
                filters=[("crop_code", "in", list(CABBAGE_CODES))],
            ).to_pylist()
        features = cabbage_features(rows, crs)
        if not features:
            raise ValueError("No valid Danish cabbage polygons found")
        save_json(path, {"type": "FeatureCollection", "source_url": MARKER_2023,
                         "matching_records": len(rows), "features": features})
    return json.loads(path.read_text())
