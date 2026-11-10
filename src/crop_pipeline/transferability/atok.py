"""The Atok image cube, KML fields and provisional labelled samples."""

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cached_property

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely import transform as shapely_transform, union_all
from shapely.geometry import Polygon, mapping

from crop_pipeline.training import external_combined as ext

from .config import ATOK_FLOOR, CLASSES, INTERVAL, LABEL_YEARS, MIN_CLEAR, VARS, YEARS

KML_NS = {"k": "http://www.opengis.net/kml/2.2"}
FIELDS = ["B1", "B2", "Area 2", "Area 3", "Area 5"]
INTERVAL_COLUMNS = [f"bin_{b:02d}_{v}" for b in range(1, 9) for v in VARS]
assert INTERVAL_COLUMNS == INTERVAL, "column order differs from the external tables"

# Provisional labels from the advisor-described rotations (notebook 08 transcription). A window is labelled only
# when the calendar names one crop for its whole planting-to-harvest season of at least 4 months.
LABEL_ROWS = [  # field, crop, window start month, label basis
    ("B1", "carrot", 3, "rotation, undated"),
    ("Area 2", "potato", 3, "rotation, undated"),
    ("Area 2", "cabbage", 7, "rotation, undated"),
    ("Area 2", "carrot", 11, "rotation, undated"),
    ("Area 3", "cabbage", 3, "year-round, no season"),
    ("Area 3", "cabbage", 7, "year-round, no season"),
    ("Area 3", "cabbage", 11, "year-round, no season"),
]
# Unlabelled fields shown in the inventory: (field, crop)
OTHER_CROP_FIELDS = (("B2", "lettuce"), ("Area 5", "ornamentals / seedbeds"))

# Half-month slots: (year, month, half); slot dates are mid-half-month
SLOTS = [(y, m, h) for y in YEARS for m in range(1, 13) for h in (1, 2)]
SLOT_DATE = pd.to_datetime([f"{y}-{m:02d}-{8 if h == 1 else 23}" for y, m, h in SLOTS])
SLOT_YEAR = np.array([y for y, _, _ in SLOTS])


def read_kml(path):
    rows = []
    for folder in ET.parse(path).getroot().iter(f"{{{KML_NS['k']}}}Folder"):
        for placemark in folder.findall("k:Placemark", KML_NS):
            coordinates = placemark.find(".//k:coordinates", KML_NS).text.split()
            rows.append({"name": placemark.find("k:name", KML_NS).text,
                         "geometry": Polygon([tuple(map(float, c.split(",")[:2])) for c in coordinates])})
    return pd.DataFrame(rows)


@dataclass
class AtokGrid:
    """The cached ROI cube (half-months, bands, H, W) and the KML fields on its pixel grid."""

    cube: np.ndarray
    composites: int
    crs: object
    transform: object  # window transform of the ROI grid
    height: int
    width: int
    roi_grid: object  # ROI polygon in grid CRS
    roi_mask: np.ndarray
    polygons: pd.DataFrame  # labelled fields, indexed by name, with grid_geometry
    interiors: dict  # field -> pixels at least 10 m inside the field
    field_area_ha: dict
    lon: float
    lat: float

    @cached_property
    def clear_all(self):
        """(half-months, H, W): all five bands valid."""
        return np.isfinite(self.cube).all(axis=1)

    def raster_mask(self, geometry, erode=0.0):
        shape = geometry.buffer(-erode) if erode else geometry
        return np.zeros((self.height, self.width), bool) if shape.is_empty else geometry_mask(
            [mapping(shape)], out_shape=(self.height, self.width), transform=self.transform, invert=True)

    def to_pixel(self, x, y):
        return ~self.transform * (x, y)


def load_grid(paths):
    """Load the ROI cube cached by notebook 08 or 09 and rasterise the KML fields onto it."""
    available = {i: paths.imagery_root / str(y) / f"Sentinel2_CSPlus_{y}_{m:02d}_H{h}.tif" for i, (y, m, h) in enumerate(SLOTS)}
    available = {i: p for i, p in available.items() if p.is_file()}
    if not available:
        raise FileNotFoundError(f"No composites under {paths.imagery_root}; check atok_imagery_root.")
    signature = json.loads(json.dumps(["nodata_masked_v2"] + sorted((str(p), p.stat().st_size) for p in available.values())))
    cube = next((np.load(p)["cube"] for p in paths.cube_caches
                 if p.is_file() and json.loads(str(np.load(p)["signature"])) == signature), None)
    if cube is None:
        raise FileNotFoundError("No matching ROI cube cache; run notebook 08 or 09 first to build it.")
    with rasterio.open(next(iter(available.values()))) as reference:
        crs, transform, shape = reference.crs, reference.transform, reference.shape
    kml = read_kml(paths.kml)
    to_grid = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    kml["grid_geometry"] = [shapely_transform(g, lambda xy: np.column_stack(to_grid.transform(xy[:, 0], xy[:, 1])))
                            for g in kml["geometry"]]
    roi_grid = kml.loc[kml["name"] == "Region of Interest", "grid_geometry"].iloc[0]
    window = from_bounds(*roi_grid.bounds, transform=transform).round_offsets().round_lengths()
    window = window.intersection(Window(0, 0, shape[1], shape[0]))
    height, width = int(window.height), int(window.width)
    if cube.shape[2:] != (height, width):
        raise ValueError("Cached cube does not match the ROI grid.")
    polygons = kml[~kml["name"].isin(["Atok", "Region of Interest"])].set_index("name")
    lon, lat = to_lonlat.transform(*roi_grid.centroid.coords[0])
    grid = AtokGrid(cube=cube, composites=len(available), crs=crs, transform=rasterio.windows.transform(window, transform),
                    height=height, width=width, roi_grid=roi_grid, roi_mask=None, polygons=polygons, interiors={},
                    field_area_ha={name: geom.area / 1e4 for name, geom in polygons["grid_geometry"].items()},
                    lon=lon, lat=lat)
    grid.roi_mask = grid.raster_mask(roi_grid)
    grid.interiors = {name: grid.raster_mask(polygons.loc[name, "grid_geometry"], erode=10) & grid.roi_mask for name in FIELDS}
    return grid


def window_features(grid, rr, cc, start, min_clear=MIN_CLEAR):
    """72 interval features for pixels (rr, cc) over the 8 half-months from slot `start`; also returns the clear counts."""
    block = grid.cube[start:start + 8][:, :, rr, cc].transpose(2, 0, 1).astype(float)  # (pixels, 8, 5 bands)
    clear = np.isfinite(block).all(-1)
    counts = clear.sum(1)
    keep = counts >= min_clear
    block, clear = block[keep], clear[keep]
    steps = np.arange(8)
    for n in np.nonzero(~clear.all(1))[0]:
        for b in range(block.shape[2]):
            block[n, :, b] = np.interp(steps, steps[clear[n]], block[n, clear[n], b])
    b4, b5, b8, b11 = (block[..., i] for i in (1, 2, 3, 4))
    indices = np.stack([(b8 - b4) / (b8 + b4), (b8 - b5) / (b8 + b5), (b8 - b11) / (b8 + b11),
                        1.5 * (b8 - b4) / (b8 + b4 + 0.5)], -1)  # SAVI with L = 0.5, as for the external rows
    values = np.concatenate([block, indices], -1).reshape(len(block), len(INTERVAL_COLUMNS))
    return pd.DataFrame(values, columns=INTERVAL_COLUMNS), keep, counts[keep]


def labelled_samples(grid):
    """Pixel-windows of the provisional label rows, down to ATOK_FLOOR clear composites.

    Returns (all samples, samples with >= MIN_CLEAR clear composites, per-window counts).
    """
    frames, windows = [], []
    for field, crop, month, basis in LABEL_ROWS:
        rr, cc = np.nonzero(grid.interiors[field])
        for year in LABEL_YEARS:
            features, keep, counts = window_features(grid, rr, cc, SLOTS.index((year, month, 1)), ATOK_FLOOR)
            windows.append({"field": field, "crop": crop, "year": year, "start_month": month,
                            "pixel_windows": len(rr), "kept": int((counts >= MIN_CLEAR).sum())})
            if keep.any():
                frames.append(features.assign(crop=crop, field=field, year=year, start_month=month, label_basis=basis,
                                              row=rr[keep], col=cc[keep], n_clear=counts))
    samples = pd.concat(frames, ignore_index=True)
    samples["partition_group"] = ("atok|" + samples["field"] + "|" + samples["year"].astype(str) + "|"
                                  + samples["start_month"].astype(str))
    samples["sample_id"] = samples["partition_group"] + "|" + samples["row"].astype(str) + "_" + samples["col"].astype(str)
    samples = ext.add_derived_features(samples)  # nominal-spacing shape features, identical to the external rows
    samples["label_status"] = "provisional"
    analysed = samples[samples["n_clear"] >= MIN_CLEAR].reset_index(drop=True)
    return samples, analysed, pd.DataFrame(windows)


def sample_units(atok):
    """The independent units are fields, not pixels."""
    return pd.DataFrame({crop: {
        "independent fields": part["field"].nunique(),
        "fields": ", ".join(sorted(part["field"].unique())),
        "field-seasons (field x calendar window)": part.groupby(["field", "start_month"]).ngroups,
        "field-season-years": part["partition_group"].nunique(),
        "pixel-windows (not independent)": len(part),
        "verified (dated, field-checked) fields": 0,
    } for crop, part in atok.groupby("crop")}).T.reindex(CLASSES)


def cropland_mask(grid, landuse_path):
    """NAMRIA ELUM CY-2023 'Crops' polygons within the ROI, on the grid."""
    landuse = gpd.read_file(landuse_path, bbox=gpd.GeoSeries([grid.roi_grid], crs=grid.crs), engine="pyogrio").to_crs(grid.crs)
    landuse = landuse[landuse.intersects(grid.roi_grid)]
    return grid.raster_mask(union_all(landuse.loc[landuse["sub_elu"] == "Crops", "geometry"].to_numpy())) & grid.roi_mask


def compare_land_activity(grid, cropland, land_activity_path):
    """Agreement with notebook 09's active-cropland map, or None when it is unavailable."""
    if not land_activity_path.is_file():
        return None
    with rasterio.open(land_activity_path) as source:
        land_activity = source.read(1)
    if land_activity.shape != (grid.height, grid.width):
        return None
    active = land_activity == 0
    return {"elum_crops_active": float(active[cropland].mean()),
            "active_inside_elum_crops": float(cropland[active & grid.roi_mask].mean())}


def clear_share_by_half_month(grid, last_year=2025):
    """Mean ROI clear share for each of the 24 half-months of the year, up to last_year."""
    roi_clear = grid.clear_all[:, grid.roi_mask].mean(1)
    within = SLOT_YEAR <= last_year
    return pd.Series(roi_clear[within]).groupby(np.arange(within.sum()) % 24).mean().to_numpy()


def monthly_clear_share(clear_by_half):
    return {m: float(clear_by_half[2 * m - 2:2 * m].mean()) for m in range(1, 13)}
