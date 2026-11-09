"""Reproduce the saved binary training-pixel mask exactly on its imagery grid."""

import hashlib
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS


def load_selection(plan):
    """Reject grid/count mismatches before any Earth Engine reduction is submitted."""
    properties = plan["fragment"]["properties"]
    grid = plan["settings"]["grid"]
    if properties.get("sampling_grid") != grid:
        raise ValueError(
            "Daily-summary sampling grid differs from the imagery request grid."
        )
    path = Path(properties.get("sampling_mask", ""))
    if not path.is_file():
        raise FileNotFoundError(f"Daily-summary sampling mask is missing: {path}")
    with rasterio.open(path) as raster:
        validate_raster_grid(raster, grid)
        values = raster.read(1)
    if not np.isin(values, [0, 1]).all():
        raise ValueError(
            "Daily-summary sampling mask must contain only binary 0/1 values."
        )
    identifiers = np.flatnonzero(values == 1).tolist()
    if not identifiers or len(identifiers) != int(properties["selected_pixels"]):
        raise ValueError(
            "Daily-summary sampling-mask count differs from selected_pixels."
        )
    return {
        "grid": grid,
        "selected_ids": identifiers,
        "selected_pixels": len(identifiers),
        "mask_sha256": hashlib.sha256(values.astype("uint8").tobytes()).hexdigest(),
    }


def validate_raster_grid(raster, grid):
    if raster.count != 1 or raster.shape != (grid["height"], grid["width"]):
        raise ValueError("Daily-summary sampling-mask dimensions differ from imagery.")
    if CRS.from_user_input(raster.crs) != CRS.from_user_input(grid["crs"]):
        raise ValueError("Daily-summary sampling-mask CRS differs from imagery.")
    if not raster.transform.almost_equals(Affine(*grid["transform"])):
        raise ValueError("Daily-summary sampling-mask transform differs from imagery.")


def selection_image(earth_engine, grid, selected_ids):
    """Integer pixel IDs avoid polygon-edge coverage and rasterization differences."""
    projection = earth_engine.Projection(grid["crs"], grid["transform"])
    coordinates = earth_engine.Image.pixelCoordinates(projection)
    # EE reports pixel centres (0.5, 1.5, ...), while saved mask IDs
    # use zero-based integer row/column indices.
    columns, rows = coordinates.select("x").floor(), coordinates.select("y").floor()
    pixel_ids = rows.multiply(grid["width"]).add(columns)
    identifiers = earth_engine.List(selected_ids)
    selected = pixel_ids.remap(
        identifiers, earth_engine.List.repeat(1, identifiers.size()), 0
    ).eq(1)
    # Bounds prevent a column outside the grid aliasing a valid flattened pixel ID.
    inside = columns.gte(0).And(columns.lt(grid["width"]))
    inside = inside.And(rows.gte(0)).And(rows.lt(grid["height"]))
    return selected.And(inside).rename("selected_pixel")


def grid_footprint(earth_engine, grid):
    transform = Affine(*grid["transform"])
    west, north = transform * (0, 0)
    east, south = transform * (grid["width"], grid["height"])
    return earth_engine.Geometry.Rectangle(
        [west, south, east, north], proj=grid["crs"], geodesic=False
    )
