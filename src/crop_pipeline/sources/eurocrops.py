"""Read declared Dutch carrot production parcels from the local EuroCrops release."""

import hashlib
import json
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from pathlib import Path

import pyarrow.parquet as parquet
from pyproj import CRS, Transformer
from shapely import from_wkb
from shapely.geometry import mapping
from shapely.ops import transform

SOURCE = "eurocrops"
YEAR = 2021
TOP_UP_YEAR = 2020
TOP_UP_URL = "https://zenodo.org/records/10118572/files/NL_2020.zip"
TOP_UP_MD5 = "823da32d28695b8b016740449391c0db"
CARROT_PRODUCTION_CODES = {"2717", "2783", "2785"}
CABBAGE_PRODUCTION_CODES = {"2759", "2761", "2789"}


def load_carrots(path):
    path = Path(path)
    if path.name == "NL_2020.zip":
        import pyogrio

        codes = ", ".join(f"'{code}'" for code in sorted(CARROT_PRODUCTION_CODES))
        layer = pyogrio.read_dataframe(
            f"/vsizip/{path.resolve()}/NL_2020_EC21.shp",
            columns=["objectid", "gewascode"], where=f"gewascode IN ({codes})",
        )
        if layer.crs is None:
            raise ValueError("EuroCrops requires a declared CRS")
        projector = Transformer.from_crs(layer.crs, 4326, always_xy=True)
        rows = ((str(int(identifier)), str(code), geometry)
                for identifier, code, geometry in layer.dropna(subset=["objectid"]).itertuples(index=False, name=None))
        return _crop_features(rows, "carrot", CARROT_PRODUCTION_CODES, TOP_UP_YEAR, projector)
    return _load_crop(path, "carrot", CARROT_PRODUCTION_CODES)


def cache_top_up_archive(path):
    """Download the published archive once and publish only checksum-matching bytes."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".eurocrops-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with urlopen(TOP_UP_URL, timeout=60) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        with temporary.open("rb") as stream:
            if hashlib.file_digest(stream, "md5").hexdigest() != TOP_UP_MD5:
                raise ValueError("EuroCrops 2020 archive checksum differs from the published release")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


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
    rows = (
        (str(row["cropfield"]), str(row["original_code"]),
         from_wkb(row["geometry"], on_invalid="ignore")
         if str(row["original_code"]) in production_codes else None)
        for batch in data.iter_batches(columns=["cropfield", "original_code", "geometry"])
        for row in batch.to_pylist()
        if row["cropfield"] is not None
    )
    return _crop_features(rows, crop, production_codes, YEAR, projector)


def _crop_features(rows, crop, production_codes, year, projector):
    features = {}
    seen = set()
    rejected = set()
    for parcel, code, geometry in rows:
        if parcel in seen:
            rejected.add(parcel)
        seen.add(parcel)
        if code not in production_codes:
            continue
        if geometry is None or geometry.is_empty or not geometry.is_valid or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            rejected.add(parcel)
            continue
        geometry = transform(projector.transform, geometry)
        west, south, east, north = geometry.bounds
        if not (3 <= west <= east <= 8 and 50 <= south <= north <= 54):
            rejected.add(parcel)
            continue
        group = f"{SOURCE}|NL|{year}|{parcel}"
        # The release identifies fields within each year's table.
        parcel_id = parcel if year == YEAR else f"{year}:{parcel}"
        features[parcel] = {
            "type": "Feature", "geometry": mapping(geometry),
            "properties": {
                "fragment_id": group, "partition_group": group, "parcel_id": parcel_id,
                "crop": crop, "country": "NL", "year": year,
                "source_crs": "EPSG:32631", "source_split": "train",
                "source_class_code": code, "label_status": "reported",
                "label_evidence": f"EuroCrops Netherlands {year} declared production code {code}",
            },
        }
    return [features[key] for key in sorted(features.keys() - rejected)]
