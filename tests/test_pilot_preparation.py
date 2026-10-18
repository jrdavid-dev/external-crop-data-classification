"""Check source joins, unique-pixel budgets, year handling and resumable downloads."""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform as transform_geometry

from crop_pipeline.imagery import quality as crop_workflow
from crop_pipeline.sources import francecrops as france
from crop_pipeline.sources import sampling
from crop_pipeline.imagery.retrieval import SatelliteRetriever, REQUEST_VERSION, RESAMPLING_POLICY


def candidate_catalog(crop="carrot", group_count=60, pixels_per_group=60):
    return [
        {
            "crop": crop,
            "partition_group": f"{crop}_{group_number}",
            "source_crs": "EPSG:32631",
            "x": 500005 + group_number * 1000 + pixel_number * 10,
            "y": 5500005,
            "source_fragments": [f"fragment_{group_number}"],
        }
        for group_number in range(group_count)
        for pixel_number in range(pixels_per_group)
    ]


def complete_rows(crop="carrot", count=1500, groups=30, status="satellite_estimated"):
    return [
        {
            "crop": crop,
            "partition_group": f"{crop}_{index % groups}",
            "source_crs": "EPSG:32631",
            "x": index * 10,
            "y": 1000,
            "season_key": "2022",
            "window_status": status,
        }
        for index in range(count)
    ]


class PixelBudgetTests(unittest.TestCase):
    def test_exact_candidate_budget_is_reproducible_and_spread_across_fields(self):
        catalog = candidate_catalog() + candidate_catalog("potato")
        selected = sampling.sample_candidate_pixels(catalog, ["carrot", "potato"])
        repeated = sampling.sample_candidate_pixels(list(reversed(catalog)), ["carrot", "potato"])
        self.assertEqual(selected, repeated)
        self.assertEqual(Counter(row["crop"] for row in selected), {"carrot": 3000, "potato": 3000})
        self.assertEqual(len({row["partition_group"] for row in selected}), 120)
        self.assertEqual(max(Counter(row["partition_group"] for row in selected).values()), 50)

    def test_candidate_shortfall_is_not_filled_by_duplicates(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            sampling.sample_candidate_pixels(candidate_catalog(group_count=20), ["carrot"])

    def test_trimming_counts_unique_locations_and_preserves_supported_priority(self):
        estimates = complete_rows()
        supported = [{**estimates[0], "window_status": "reference_supported"}]
        alternate_season = [{**row, "season_key": "2023"} for row in estimates]
        references, selected, summary = sampling.trim_usable_pixels(
            supported, estimates + alternate_season
        )
        self.assertEqual(len(references) + len(selected), 500)
        self.assertEqual(references[0]["x"], 0)
        self.assertEqual(summary[0]["available_usable_pixels"], 1500)
        self.assertEqual(summary[1]["shortfall_pixels"], 500)
        self.assertEqual(len({sampling.pixel_location(row) for row in references + selected}), 500)

    def test_trimming_is_balanced_across_fields_and_input_order_independent(self):
        rows = complete_rows()
        first = sampling.trim_usable_pixels([], rows)
        second = sampling.trim_usable_pixels([], list(reversed(rows)))
        self.assertEqual(first, second)
        counts = Counter(row["partition_group"] for row in first[1])
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_local_test_rows_are_never_trimmed(self):
        local = complete_rows(count=1200, status="reference_supported")
        registry = [{"partition_group": row["partition_group"], "source": "local"} for row in local]
        references, estimates, summary = sampling.trim_external_pilot(local, [], registry)
        self.assertEqual(references, local)
        self.assertFalse(estimates)
        self.assertEqual(summary[0]["available_usable_pixels"], 0)

    def test_conflicting_labels_at_one_pixel_raise(self):
        carrot = complete_rows(count=1)
        cabbage = [{**carrot[0], "crop": "cabbage"}]
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            sampling.trim_usable_pixels([], carrot + cabbage)

    def test_masks_match_the_shared_retrieval_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            selected = sampling.sample_candidate_pixels(
                candidate_catalog(group_count=2, pixels_per_group=10),
                ["carrot"],
                pixels_per_crop=10,
                preferred_groups=2,
            )
            regions, _ = sampling.save_candidate_pilot(selected, temporary)
            retriever = SatelliteRetriever(None, 2022, temporary, temporary)
            for region in regions:
                self.assertEqual(
                    region["properties"]["sampling_grid"], retriever.make_download_grid(region)
                )
                with rasterio.open(region["properties"]["sampling_mask"]) as raster:
                    self.assertEqual(int(raster.read(1).sum()), 5)


class FranceReferenceTests(unittest.TestCase):
    def write_tables(self, folder, duplicate=False, wrong_class=False):
        folder = Path(folder)
        paths = {
            name: folder / f"{name}.parquet" for name in ("class_map", "metadata", "geolocation")
        }
        parquet.write_table(
            arrow.Table.from_pylist(
                [
                    {
                        "y": 4,
                        "class_code": "CHU",
                        "class_name_en": "Potato" if wrong_class else "Cabbage",
                    }
                ]
            ),
            paths["class_map"],
        )
        labels = [
            {"parcel_id": "training", "split": "train", "y": 4},
            {"parcel_id": "held_out", "split": "test_france", "y": 4},
        ]
        if duplicate:
            labels.append(labels[0])
        parquet.write_table(arrow.Table.from_pylist(labels), paths["metadata"])
        geometries = arrow.Table.from_pylist(
            [
                {"parcel_id": name, "geometry": box(1, 47, 1.01, 47.01).wkb}
                for name in ("training", "held_out")
            ]
        )
        metadata = {
            b"geo": json.dumps(
                {"columns": {"geometry": {"crs": CRS.from_epsg(4326).to_json_dict()}}}
            ).encode()
        }
        parquet.write_table(geometries.replace_schema_metadata(metadata), paths["geolocation"])
        return paths

    def test_only_cabbage_training_parcels_are_joined(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.write_tables(temporary)
            features = france.load_cabbage_parcels(paths)
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0]["properties"]["parcel_id"], "training")
            self.assertEqual(features[0]["properties"]["year"], 2022)
            self.assertEqual(features[0]["properties"]["label_status"], "reported")

    def test_changed_class_map_and_duplicate_labels_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "CHU"):
                france.load_cabbage_parcels(self.write_tables(temporary, wrong_class=True))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                france.load_cabbage_parcels(self.write_tables(temporary, duplicate=True))

    def test_missing_geometry_cannot_silently_drop_joined_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.write_tables(temporary)
            parquet.write_table(
                parquet.read_table(paths["geolocation"]).slice(1), paths["geolocation"]
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                france.load_cabbage_parcels(paths)

    def test_interior_sampling_excludes_the_boundary_and_keeps_native_grid(self):
        projector = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
        geometry = transform_geometry(projector.transform, box(500000, 5200000, 500100, 5200100))
        feature = france.parcel_feature(
            {"parcel_id": "example", "geometry": geometry.wkb}, france.DEFAULT_REVISION, 4
        )
        rows = france.parcel_pixel_candidates(feature)
        self.assertEqual(len(rows), 64)
        self.assertTrue(all(500010 < row["x"] < 500090 for row in rows))


class SatelliteRetrievalTests(unittest.TestCase):
    def test_year_is_part_of_request_settings(self):
        rows = candidate_catalog(group_count=1, pixels_per_group=2)
        region = sampling.selected_region("carrot", "group", rows, 42)
        first = SatelliteRetriever(None, 2020, "images", "reports").request_settings(region)
        second = SatelliteRetriever(None, 2022, "images", "reports").request_settings(region)
        self.assertEqual(second["start"], "2022-01-01")
        self.assertEqual(second["end"], "2023-01-01")
        self.assertEqual(first["bands"], second["bands"])
        self.assertNotEqual(first, second)

    def test_partial_download_is_retried_then_checksum_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = candidate_catalog(group_count=1, pixels_per_group=2)
            regions, _ = sampling.save_candidate_pilot(rows, folder / "fields")
            retriever = SatelliteRetriever(None, 2022, folder / "images", folder / "reports")
            settings = retriever.request_settings(regions[0])
            plan = {
                "fragment_id": regions[0]["properties"]["fragment_id"],
                "fragment": regions[0],
                "settings": settings,
                "acquisitions": [],
            }
            interval = crop_workflow.interval_bins(2022)[0]
            manifest = {"files": {}}

            def write_composite(plan, interval, granules, destination):
                grid = plan["settings"]["grid"]
                with rasterio.open(
                    destination,
                    "w",
                    driver="GTiff",
                    width=grid["width"],
                    height=grid["height"],
                    count=7,
                    dtype="float32",
                    crs=grid["crs"],
                    transform=Affine(*grid["transform"]),
                ) as raster:
                    raster.write(np.ones((7, grid["height"], grid["width"]), dtype="float32"))

            with patch.object(
                retriever, "fetch_composite", side_effect=OSError("temporary failure")
            ):
                self.assertEqual(
                    retriever.download_interval(plan, interval, folder, manifest)["status"],
                    "failed",
                )
            with patch.object(retriever, "fetch_composite", side_effect=write_composite) as fetch:
                self.assertEqual(
                    retriever.download_interval(plan, interval, folder, manifest)["status"],
                    "downloaded",
                )
                self.assertEqual(
                    retriever.download_interval(plan, interval, folder, manifest)["status"],
                    "reused",
                )
                self.assertEqual(fetch.call_count, 1)
            progress = folder / "reports"
            progress.mkdir()
            manifest.update(fragment_id=plan["fragment_id"], settings=settings)
            (progress / f"{plan['fragment_id']}_request.json").write_text(json.dumps(manifest))
            quality, pixels = crop_workflow.quality_check(
                regions,
                folder / "reports",
                folder,
                folder / "fields",
                folder / "quality",
                year=2022,
            )
            self.assertEqual(quality[0]["name"], "2022_01_01_15")
            self.assertEqual(len(quality), 24)
            self.assertEqual(quality[0]["status"], "quality_pass")
            self.assertEqual(len(pixels), 2)


class SourceHandoffTests(unittest.TestCase):
    def test_source_handoff_rejects_changed_sampling(self):
        from crop_pipeline.sources import pilot_sources

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            fields = folder / "fields"
            fields.mkdir()
            region_path = fields / "sampled_download_regions.geojson"
            region_path.write_text(json.dumps({"features": []}))
            manifest = folder / "source.json"
            pilot_sources.save_source_manifest(
                manifest,
                "francecrops",
                2022,
                fields,
                folder / "images",
                folder / "progress",
                folder / "timelines",
                france.DEFAULT_REVISION,
            )
            self.assertEqual(pilot_sources.load_sources([manifest])[0]["year"], 2022)
            region_path.write_text(json.dumps({"features": ["changed"]}))
            with self.assertRaisesRegex(ValueError, "sampling changed"):
                pilot_sources.load_sources([manifest])

    def test_combined_audit_preserves_each_source_year_and_identity(self):
        from crop_pipeline.sources import pilot_sources

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            sources = []
            for name, year in [("sen4agrinet", 2020), ("francecrops", 2022)]:
                source_folder = folder / name
                source_folder.mkdir()
                region = {
                    "properties": {
                        "fragment_id": name,
                        "crop": "cabbage" if year == 2022 else "carrot",
                    }
                }
                (source_folder / "sampled_download_regions.geojson").write_text(
                    json.dumps({"features": [region]})
                )
                (source_folder / "observed_timelines.json").write_text(
                    json.dumps([{"region_id": name, "date": f"{year}-05-01"}])
                )
                sources.append(
                    {
                        "source_name": name,
                        "request_version": REQUEST_VERSION,
                        "resampling": RESAMPLING_POLICY,
                        "year": year,
                        "field_folder": str(source_folder),
                        "imagery_folder": str(source_folder),
                        "progress_folder": str(source_folder),
                        "timeline_folder": str(source_folder),
                        "cloud_threshold": 40,
                        "dataset_revision": "pinned",
                    }
                )
            with patch.object(crop_workflow, "quality_check", return_value=([], [])) as quality:
                regions, _, _ = pilot_sources.audit_sources(sources, folder / "analysis")
            self.assertEqual([call.kwargs["year"] for call in quality.call_args_list], [2020, 2022])
            self.assertEqual(
                [region["properties"]["source_year"] for region in regions], [2020, 2022]
            )
            combined = json.loads(
                (folder / "analysis/timelines/observed_timelines.json").read_text()
            )
            self.assertEqual(len(combined), 2)


class UnequalParcelCapacityTests(unittest.TestCase):
    def test_small_parcels_can_contribute_without_an_equal_quota(self):
        catalog = candidate_catalog(group_count=30, pixels_per_group=100)
        catalog += candidate_catalog(group_count=30, pixels_per_group=1, crop="small")
        for row in catalog:
            if row["crop"] == "small":
                row["x"] += 1000000
            row["crop"] = "carrot"
        selected = sampling.sample_candidate_pixels(catalog, ["carrot"])
        self.assertEqual(len(selected), 3000)
        self.assertEqual(len({row["partition_group"] for row in selected}), 60)
        self.assertLessEqual(max(Counter(row["partition_group"] for row in selected).values()), 100)


if __name__ == "__main__":
    unittest.main()
