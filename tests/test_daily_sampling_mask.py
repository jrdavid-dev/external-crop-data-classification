"""Local numeric EE substitute verifies pixel identity without any remote requests."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import rasterio
from affine import Affine

from crop_pipeline.imagery import sampling_mask, timelines


class NumericImage:
    def __init__(self, bands, engine=None):
        self.bands = (
            bands if isinstance(bands, dict) else {"value": np.ma.asarray(bands)}
        )
        self.engine = engine

    def single(self):
        return next(iter(self.bands.values()))

    def select(self, name):
        return NumericImage({name: self.bands[name]}, self.engine)

    def operation(self, other, function):
        other = other.single() if isinstance(other, NumericImage) else other
        return NumericImage(function(self.single(), other), self.engine)

    def floor(self):
        return NumericImage(np.floor(self.single()), self.engine)

    def multiply(self, other):
        return self.operation(other, np.multiply)

    def add(self, other):
        return self.operation(other, np.add)

    def subtract(self, other):
        return self.operation(other, np.subtract)

    def divide(self, other):
        return self.operation(other, np.ma.divide)

    def eq(self, other):
        return self.operation(other, np.equal)

    def neq(self, other):
        return self.operation(other, np.not_equal)

    def gte(self, other):
        return self.operation(other, np.greater_equal)

    def lt(self, other):
        return self.operation(other, np.less)

    def And(self, other):
        return self.operation(other, np.logical_and)

    def remap(self, source, targets, default):
        values = np.full(self.single().shape, default)
        for identifier, target in zip(source, targets):
            values[self.single() == identifier] = target
        return NumericImage(values, self.engine)

    def rename(self, name):
        return NumericImage({name: self.single()}, self.engine)

    def updateMask(self, selection):
        valid = np.ma.asarray(selection.single()).filled(0).astype(bool)
        return NumericImage(
            {
                name: np.ma.masked_where(~valid, values)
                for name, values in self.bands.items()
            },
            self.engine,
        )

    def reduceRegion(self, **arguments):
        self.engine.reductions.append(arguments)
        statistics = {}
        for name, values in self.bands.items():
            data = np.ma.asarray(values).compressed()
            statistics[name + "_count"] = len(data)
            statistics[name + "_median"] = float(np.median(data)) if len(data) else None
        return NumericDictionary(statistics)


class NumericDictionary(dict):
    def set(self, name, value):
        return NumericDictionary({**self, name: value})


class NumericList(list):
    def size(self):
        return len(self)

    @staticmethod
    def repeat(value, count):
        return NumericList([value] * count)


class NumericReducer:
    def unweighted(self):
        return self

    def combine(self, other, sharedInputs):
        return self


class NumericImageFactory:
    def __init__(self, engine):
        self.engine = engine

    def __call__(self, identifier):
        return self.engine.image

    def pixelCoordinates(self, projection):
        height, width = self.engine.height, self.engine.width
        rows, columns = np.indices((height + 2, width + 2))
        return NumericImage({"x": columns - 0.5, "y": rows - 0.5}, self.engine)

    def cat(self, images):
        return NumericImage(
            {name: values for image in images for name, values in image.bands.items()},
            self.engine,
        )


class NumericEarthEngine:
    def __init__(self, width=3, height=3, obscure_first_pixel=False):
        self.width, self.height, self.reductions = width, height, []
        bands = {
            name: np.ma.array(np.full((height + 2, width + 2), value), mask=False)
            for name, value in [("B8", 0.6), ("B4", 0.2), ("B5", 0.3), ("B11", 0.4)]
        }
        if obscure_first_pixel:
            for values in bands.values():
                values.mask[1, 1] = True
        self.image = NumericImage(bands, self)
        self.Image = NumericImageFactory(self)
        self.List = NumericList
        self.Projection = lambda crs, transform: (crs, transform)
        self.Geometry = SimpleNamespace(
            Rectangle=lambda coordinates, **arguments: (coordinates, arguments)
        )
        self.Reducer = SimpleNamespace(median=NumericReducer, count=NumericReducer)
        self.ImageCollection = SimpleNamespace(
            fromImages=lambda images: SimpleNamespace(mosaic=lambda: images[0])
        )
        self.Feature = lambda geometry, properties: {"properties": properties}
        self.FeatureCollection = lambda features: SimpleNamespace(
            getInfo=lambda: {"features": features}
        )


def mask_plan(folder):
    path = Path(folder) / "mask.tif"
    grid = {
        "crs": "EPSG:32631",
        "transform": [10, 0, 500000, 0, -10, 5200000],
        "width": 3,
        "height": 3,
    }
    mask = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype="uint8")
    write_mask(path, grid, mask)
    return {
        "fragment_id": "region",
        "fragment": {
            "geometry": {"type": "Polygon", "coordinates": []},
            "properties": {
                "crop": "carrot",
                "partition_group": "parcel",
                "selected_pixels": 4,
                "sampling_grid": grid,
                "sampling_mask": str(path),
            },
        },
        "settings": {"grid": grid, "version": 4},
        "acquisitions": [
            {
                "date": "2022-04-04",
                "image_id": "scene",
                "image_index": "one",
                "cloud_probability_id": "cloud",
            }
        ],
    }


def write_mask(path, grid, values):
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
    ) as raster:
        raster.write(values, 1)


class DailySamplingMaskTests(unittest.TestCase):
    def test_daily_counts_use_only_saved_pixels_and_preserve_cloud_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = mask_plan(temporary)
            engine = NumericEarthEngine(obscure_first_pixel=True)
            rows = timelines.retrieve_observed_timelines(
                [plan],
                lambda image, probability: image,
                engine,
                Path(temporary) / "summaries",
            )
            self.assertEqual(rows[0]["valid_pixels"], 3)
            self.assertEqual(rows[0]["valid_fraction"], 0.75)
            self.assertFalse(rows[0]["eligible"])
            self.assertAlmostEqual(rows[0]["NDVI"], 0.5)
            reduction = engine.reductions[0]
            self.assertEqual(reduction["crs"], plan["settings"]["grid"]["crs"])
            self.assertEqual(
                reduction["crsTransform"], plan["settings"]["grid"]["transform"]
            )

    def test_all_clear_counts_equal_mask_and_no_outside_grid_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = mask_plan(temporary)
            engine = NumericEarthEngine()
            selection = sampling_mask.load_selection(plan)
            image = sampling_mask.selection_image(
                engine, selection["grid"], selection["selected_ids"]
            )
            self.assertEqual(int(image.single().sum()), 4)
            self.assertFalse(image.single()[0].any())
            self.assertFalse(image.single()[:, 0].any())
            rows = timelines.retrieve_observed_timelines(
                [plan],
                lambda image, probability: image,
                engine,
                Path(temporary) / "summaries",
            )
            self.assertEqual(rows[0]["valid_pixels"], 4)
            self.assertTrue(rows[0]["eligible"])

    def test_grid_shape_crs_transform_and_count_are_checked_before_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = mask_plan(temporary)
            plan = copy.deepcopy(original)
            plan["fragment"]["properties"]["selected_pixels"] = 5
            with self.assertRaisesRegex(ValueError, "count"):
                sampling_mask.load_selection(plan)
            plan = copy.deepcopy(original)
            plan["fragment"]["properties"]["sampling_grid"] = {**plan["settings"]["grid"], "width": 4}
            with self.assertRaisesRegex(ValueError, "sampling grid"):
                sampling_mask.load_selection(plan)
            grid = original["settings"]["grid"]
            shifted = {**grid, "transform": [10, 0, 500010, 0, -10, 5200000]}
            write_mask(
                Path(temporary) / "mask.tif", shifted, np.ones((3, 3), dtype="uint8")
            )
            with self.assertRaisesRegex(ValueError, "transform"):
                sampling_mask.load_selection(original)

    def test_old_checkpoints_and_aggregate_outputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = mask_plan(root)
            destination = root / "summaries"
            old_fingerprint = hashlib.sha256(
                json.dumps(plan["settings"], sort_keys=True).encode()
            ).hexdigest()[:16]
            old = (
                destination / "observed_checkpoints" / f"region_{old_fingerprint}.json"
            )
            old.parent.mkdir(parents=True)
            old.write_text('{"old_results": true}')
            previous = destination / "observed_timelines.json"
            previous.write_text('[{"old_results": true}]')
            old_bytes, previous_bytes = old.read_bytes(), previous.read_bytes()
            engine = NumericEarthEngine()
            rows = timelines.retrieve_observed_timelines(
                [plan], lambda image, probability: image, engine, destination
            )
            self.assertEqual(old.read_bytes(), old_bytes)
            backups = list(
                (destination / "summary_history").glob("*/observed_timelines.json")
            )
            self.assertTrue(
                any(path.read_bytes() == previous_bytes for path in backups)
            )
            self.assertEqual(
                len(list(old.parent.glob(f"*{timelines.SUMMARY_RECIPE}*.json"))), 1
            )
            with patch.object(
                timelines,
                "daily_features",
                side_effect=AssertionError("Corrected checkpoint must be reused"),
            ):
                self.assertEqual(
                    timelines.retrieve_observed_timelines(
                        [plan], lambda image, probability: image, engine, destination
                    ),
                    rows,
                )

    def test_mask_and_quality_changes_have_distinct_daily_fingerprints(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = mask_plan(temporary)
            selection = sampling_mask.load_selection(plan)
            original = timelines.summary_contract(plan, selection, 0.8)
            path = timelines.checkpoint_path(temporary, "region", original)
            changed_fraction = timelines.summary_contract(plan, selection, 0.9)
            self.assertNotEqual(
                path, timelines.checkpoint_path(temporary, "region", changed_fraction)
            )
            changed_mask = np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]], dtype="uint8")
            write_mask(Path(temporary) / "mask.tif", selection["grid"], changed_mask)
            changed = timelines.summary_contract(
                plan, sampling_mask.load_selection(plan), 0.8
            )
            self.assertNotEqual(
                path, timelines.checkpoint_path(temporary, "region", changed)
            )

    def test_overcount_of_any_index_still_fails_instead_of_being_clamped(self):
        properties = {
            "selected_pixels": 4,
            "crop": "carrot",
            "partition_group": "parcel",
        }
        values = {
            "date": "2022-04-04",
            "NDVI_count": 5,
            "NDRE_count": 3,
            "NDMI_count": 2,
        }
        with self.assertRaisesRegex(ValueError, "more pixels"):
            timelines.row_from_statistics(values, properties, "region", 0.8)

    def test_bad_local_mask_stops_before_any_server_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = mask_plan(temporary)
            plan["fragment"]["properties"]["selected_pixels"] = 20
            with patch.object(
                timelines,
                "daily_features",
                side_effect=AssertionError("No query allowed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "sampling-mask count"):
                    timelines.retrieve_observed_timelines(
                        [plan], None, None, Path(temporary) / "summaries"
                    )
            self.assertFalse(
                list(
                    (Path(temporary) / "summaries").glob("observed_checkpoints/*.json")
                )
            )
