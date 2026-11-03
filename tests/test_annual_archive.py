"""Full-year collection stays independent from timing and remains opt-in."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import rasterio
from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform as transform_geometry

from crop_pipeline.imagery import quality
from crop_pipeline.sources import annual, pilot_sources, sampling
from crop_pipeline.windows import annual_archive, seasons
from test_season_workflow import feature_fixture


def reference_feature(parcel_id="field", left=500000, width=300, crop="carrot"):
    projector = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    geometry = transform_geometry(
        projector.transform, box(left, 5200000, left + width, 5200300)
    )
    return {
        "type": "Feature",
        "geometry": mapping(geometry),
        "properties": {
            "fragment_id": parcel_id,
            "parcel_id": parcel_id,
            "crop": crop,
            "country": "FR",
            "year": 2020,
            "source_crs": "EPSG:32631",
            "partition_group": "sen4agrinet|FR|2020|" + parcel_id,
        },
    }


class AnnualArchiveTests(unittest.TestCase):
    def test_all_cells_are_retained_with_bounded_tiles_and_one_parcel_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = annual.prepare_source(
                "sen4agrinet",
                [reference_feature()],
                temporary,
                {},
                inward_buffer_metres=0,
                tile_pixels=16,
            )
            inputs = plan["years"]["2020"]
            regions = annual.read_regions(inputs)
            self.assertGreater(
                sum(region["properties"]["selected_pixels"] for region in regions), 100
            )
            self.assertGreater(len(regions), 1)
            self.assertEqual(
                len({region["properties"]["partition_group"] for region in regions}), 1
            )
            self.assertEqual(plan["estimate"]["composites"], len(regions) * 24)
            locations = set()
            for region in regions:
                grid = region["properties"]["sampling_grid"]
                self.assertLessEqual(
                    grid["width"], 22
                )  # Buffer plus WGS84 round-trip rounding.
                with rasterio.open(region["properties"]["sampling_mask"]) as raster:
                    rows, columns = (raster.read(1) == 1).nonzero()
                    current = {
                        raster.transform * (int(column) + 0.5, int(row) + 0.5)
                        for row, column in zip(rows, columns)
                    }
                self.assertFalse(locations.intersection(current))
                locations.update(current)
            self.assertEqual(len(locations), 900)

    def test_competing_parcel_claims_are_removed_but_same_parcel_fragments_merge(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = reference_feature()
            duplicate = json.loads(json.dumps(original))
            plan = annual.prepare_source(
                "sen4agrinet",
                [original, duplicate],
                temporary,
                {},
                inward_buffer_metres=0,
            )
            self.assertEqual(plan["estimate"]["parcels"], 1)
            overlap = reference_feature("other", left=500150)
            competing = annual.prepare_source(
                "sen4agrinet",
                [original, overlap],
                temporary,
                {},
                inward_buffer_metres=0,
            )
            self.assertEqual(
                sum(
                    inputs["pixels_per_crop"]["carrot"]
                    for inputs in competing["years"].values()
                ),
                900,
            )
            self.assertNotEqual(plan["run_folder"], competing["run_folder"])
            self.assertTrue(Path(plan["run_folder"]).is_dir())

    def test_disabled_collection_never_imports_or_initializes_earth_engine(self):
        with patch.dict("sys.modules", {"ee": None}):
            result = annual.collect_source(
                {"source": "sen4agrinet", "estimate": {}}, "/absent", "", enabled=False
            )
        self.assertEqual(result["status"], "disabled")

    def test_collection_requests_all_24_bins_without_timing_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            retriever = MagicMock(year=2020)
            retriever.inventory_region.return_value = {
                "fragment_id": "region",
                "settings": {},
            }
            retriever.download_region.return_value = [
                {
                    "status": "downloaded",
                    "region_id": "region",
                    "name": interval["name"],
                }
                for interval in quality.interval_bins(2020)
            ]
            checkpoint = root / "tiles/region/observed_checkpoints/test.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("{}")
            with (
                patch.object(
                    annual.timelines, "retrieve_observed_timelines", return_value=[]
                ),
                patch(
                    "crop_pipeline.imagery.bilinear.verify_download_completion"
                ) as completion,
            ):
                annual.collect_region(
                    retriever,
                    {"properties": {"crop": "carrot"}},
                    {"timeline_folder": str(root), "report_folder": str(root)},
                )
            submitted = retriever.download_region.call_args.args[0]
            self.assertEqual(
                submitted["download_intervals"], quality.interval_bins(2020)
            )
            completion.assert_called_once()

    def test_missing_daily_checkpoint_stops_before_images_and_handoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            retriever = MagicMock(year=2020)
            retriever.inventory_region.return_value = {"fragment_id": "region"}
            with patch.object(
                annual.timelines, "retrieve_observed_timelines", return_value=[]
            ):
                with self.assertRaisesRegex(RuntimeError, "Daily summaries incomplete"):
                    annual.collect_region(retriever, {}, {"timeline_folder": temporary})
            retriever.download_region.assert_not_called()

    def test_all_three_bundles_required_and_modified_handoff_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "inventory"
            with self.assertRaisesRegex(FileNotFoundError, "cropharvest"):
                annual.resolve_collection(report)
            for source in annual.SOURCE_NAMES:
                path = root / f"{source}.json"
                path.write_text("{}")
                annual.register_source(
                    report,
                    {"source": source, "revision": "v1", "years": {"2020": {}}},
                    [path],
                )
            paths, name = pilot_sources.resolve_collection(
                report, annual.COLLECTION_NAME
            )
            self.assertEqual(len(paths), 3)
            self.assertTrue(name.startswith(annual.COLLECTION_NAME))
            paths[0].write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "Modified annual handoff"):
                annual.resolve_collection(report)

    def test_disk_backed_features_match_original_quality_rules(self):
        window, quality_rows, pixels = feature_fixture()
        windows = [window]
        pixels = [{**row, "B2": 0.2} for row in pixels]
        pixels.pop()  # One pixel must remain excluded in both implementations.
        # The fixture returns one season, including one complete and one incomplete pixel.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pixels.parquet"
            annual_archive.write_pixel_partition(path, pixels)
            rows = annual_archive.PixelPartitions({windows[0]["region_id"]: path})
            original = seasons.build_features(windows, pixels, quality_rows)
            partitioned = seasons.build_features(windows, rows, quality_rows)
            self.assertEqual(original, partitioned)

    def test_full_study_retention_has_no_500_pixel_cap(self):
        rows = [
            {
                "crop": "carrot",
                "partition_group": "field",
                "source_crs": "EPSG:32631",
                "x": index * 10,
                "y": 0,
                "window_status": "satellite_estimated",
                "season_key": "season",
                "window_start": "2020-04-01",
            }
            for index in range(600)
        ]
        _, selected, summary = sampling.trim_usable_pixels(
            [], rows, target_per_crop=None, crops=("carrot",)
        )
        self.assertEqual(len(selected), 600)
        self.assertEqual(summary[0]["status"], "all_usable_retained")

    def test_real_completion_gate_rejects_one_missing_annual_bin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            region = {"properties": {"crop": "carrot", "selected_pixels": 900}}
            retriever = MagicMock(year=2020)
            retriever.inventory_region.return_value = {
                "fragment_id": "region",
                "fragment": region,
            }
            results = [
                {"status": "reused", "region_id": "region", "name": interval["name"]}
                for interval in quality.interval_bins(2020)
            ]
            retriever.download_region.return_value = results
            checkpoint = root / "tiles/region/observed_checkpoints/test.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("{}")
            inputs = {"timeline_folder": str(root), "report_folder": str(root)}
            with patch.object(
                annual.timelines, "retrieve_observed_timelines", return_value=[]
            ):
                annual.collect_region(retriever, region, inputs)
                results[-1]["status"] = "failed"
                with self.assertRaisesRegex(ValueError, "all planned intervals"):
                    annual.collect_region(retriever, region, inputs)

    def test_francecrops_includes_potato_and_preserves_reserved_roles(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet
        from pyproj import CRS
        from crop_pipeline.sources import francecrops

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                name: root / f"{name}.parquet"
                for name in ("class_map", "metadata", "geolocation")
            }
            parquet.write_table(
                arrow.Table.from_pylist(
                    [
                        {"y": 4, "class_code": "CHU", "class_name_en": "cabbage"},
                        {"y": 12, "class_code": "PTC", "class_name_en": "potato"},
                    ]
                ),
                paths["class_map"],
            )
            labels = [
                {"parcel_id": "cabbage", "y": 4, "split": "train"},
                {"parcel_id": "potato", "y": 12, "split": "test_france"},
            ]
            parquet.write_table(arrow.Table.from_pylist(labels), paths["metadata"])
            geometries = arrow.Table.from_pylist(
                [
                    {
                        "parcel_id": row["parcel_id"],
                        "geometry": box(1, 47, 1.01, 47.01).wkb,
                    }
                    for row in labels
                ]
            )
            metadata = {
                b"geo": json.dumps(
                    {
                        "columns": {
                            "geometry": {"crs": CRS.from_epsg(4326).to_json_dict()}
                        }
                    }
                ).encode()
            }
            parquet.write_table(
                geometries.replace_schema_metadata(metadata), paths["geolocation"]
            )
            features, audit = francecrops.load_target_parcels(paths, return_audit=True)
            self.assertEqual(
                {feature["properties"]["crop"] for feature in features},
                {"cabbage", "potato"},
            )
            self.assertEqual(features[1]["properties"]["source_split"], "test_france")
            self.assertEqual(audit["unavailable_target_classes"], ["carrot"])
