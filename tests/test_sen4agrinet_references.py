"""Dataset-wide reference preparation without real remote requests or imagery downloads."""

import hashlib
import io
import os
from contextlib import chdir, redirect_stdout
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from netCDF4 import Dataset

from crop_pipeline.common import save_json
from crop_pipeline.sources import annual, sen4agrinet


def write_patch(
    root, year=2020, tile="31TBF", crop_code=231, parcel_id=10, country="FR"
):
    path = Path(root) / f"data/{year}/{tile}/patch_{parcel_id}.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as dataset:
        dataset.patch_year, dataset.patch_tile, dataset.patch_country_code = (
            year,
            tile,
            country,
        )
        dataset.createDimension("y", 4)
        dataset.createDimension("x", 4)
        dataset.createVariable("x", "f8", ("x",))[:] = [500005, 500015, 500025, 500035]
        dataset.createVariable("y", "f8", ("y",))[:] = [
            5200035,
            5200025,
            5200015,
            5200005,
        ]
        for name, values in [
            ("labels", np.full((4, 4), crop_code)),
            ("parcels", np.full((4, 4), parcel_id)),
        ]:
            variable = dataset.createVariable(name, "i4", ("y", "x"))
            variable.crs = "EPSG:32631"
            variable.transform = [10.0, 0.0, 500000.0, 0.0, -10.0, 5200040.0]
            variable[:] = values
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "filename": path.relative_to(root).as_posix(),
        "year": year,
        "tile": tile,
        "expected_bytes": path.stat().st_size,
        "sha256": checksum,
        "blob_id": None,
    }


def manifest_for(records):
    return {
        "dataset": sen4agrinet.DATASET_NAME,
        "revision": "a" * 40,
        "scope": "all_netcdf_files_under_data",
        "files": records,
        "years": sorted({record["year"] for record in records}),
        "year_tile_pairs": sorted(
            {f"{record['year']}/{record['tile']}" for record in records}
        ),
        "total_source_bytes": sum(record["expected_bytes"] for record in records),
    }


class DatasetReferenceTests(unittest.TestCase):
    def test_discovery_is_recursive_across_all_years_and_tiles_and_pinned_on_restart(
        self,
    ):
        hub = MagicMock()
        hub.repo_info.return_value.sha = "a" * 40
        hub.list_repo_tree.return_value = [
            SimpleNamespace(path=name, size=100, lfs={"sha256": "abc"}, blob_id=None)
            for name in [
                "data/2019/31TBF/a.nc",
                "data/2020/31TBF/b.nc",
                "data/2020/31TCF/c.nc",
                "data/README.md",
            ]
        ]
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("huggingface_hub.HfApi", return_value=hub),
        ):
            manifest = sen4agrinet.discover_release(temporary)
            hub.list_repo_tree.assert_called_once_with(
                sen4agrinet.DATASET_NAME,
                repo_type="dataset",
                revision="a" * 40,
                path_in_repo="data",
                recursive=True,
            )
            self.assertEqual(manifest["years"], [2019, 2020])
            self.assertEqual(len(manifest["files"]), 3)
            hub.repo_info.side_effect = AssertionError("Restart must reuse the pin")
            self.assertEqual(sen4agrinet.discover_release(temporary), manifest)

    def test_all_three_crops_across_years_are_extracted_and_existing_files_remain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                write_patch(root / "old", 2019, crop_code=231),
                write_patch(root / "old", 2020, crop_code=213),
                write_patch(
                    root / "old", 2020, tile="31TCF", crop_code=510, parcel_id=11
                ),
            ]
            original = {
                record["filename"]: (root / "old" / record["filename"]).read_bytes()
                for record in records
            }
            manifest = manifest_for(records)
            completed = sen4agrinet.build_reference_inventory(
                manifest, root / "new", root / "old", download_enabled=False
            )
            self.assertEqual(completed["files_scanned"], 3)
            self.assertEqual(
                set(completed["crop_counts"]), {"carrot", "cabbage", "potato"}
            )
            features, _ = annual.load_sen4agrinet_references(
                Path(completed["field_root"])
            )
            self.assertEqual(
                {feature["properties"]["year"] for feature in features}, {2019, 2020}
            )
            for record in records:
                self.assertEqual(
                    (root / "old" / record["filename"]).read_bytes(),
                    original[record["filename"]],
                )
            with patch.object(
                sen4agrinet,
                "obtain_source",
                side_effect=AssertionError("Checkpoint must skip raw retrieval"),
            ):
                repeated = sen4agrinet.build_reference_inventory(
                    manifest, root / "new", root / "old", download_enabled=False
                )
            self.assertEqual(repeated, completed)

    def test_one_missing_patch_blocks_full_inventory_and_then_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = write_patch(root / "raw")
            missing = {
                **record,
                "filename": "data/2020/31TCF/missing.nc",
                "tile": "31TCF",
            }
            manifest = manifest_for([record, missing])
            folder = sen4agrinet.run_folder(root / "refs", manifest)
            with self.assertRaisesRegex(RuntimeError, "Reference scan incomplete"):
                sen4agrinet.build_reference_inventory(
                    manifest, root / "refs", root / "raw", download_enabled=False
                )
            self.assertFalse((folder / "completed_inventory.json").exists())
            self.assertTrue(sen4agrinet.checkpoint_path(folder, record).is_file())

    def test_only_new_temporary_netcdf_is_removed_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = write_patch(root / "source")
            manifest = manifest_for([record])
            folder = sen4agrinet.run_folder(root / "refs", manifest)
            staged = folder / "staging" / record["filename"]
            staged.parent.mkdir(parents=True)
            staged.write_bytes((root / "source" / record["filename"]).read_bytes())
            with patch(
                "huggingface_hub.hf_hub_download",
                side_effect=AssertionError("Reuse verified staging"),
            ):
                sen4agrinet.process_patch(
                    record, manifest, folder, root / "absent", False, False
                )
            self.assertFalse(staged.exists())
            self.assertTrue(sen4agrinet.checkpoint_path(folder, record).is_file())
            self.assertTrue((root / "source" / record["filename"]).exists())

    def test_mixed_labels_and_cross_patch_crop_conflicts_are_audited(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                write_patch(root / "raw", crop_code=231),
                write_patch(root / "raw", tile="31TCF", crop_code=213),
            ]
            completed = sen4agrinet.build_reference_inventory(
                manifest_for(records),
                root / "refs",
                root / "raw",
                download_enabled=False,
            )
            self.assertEqual(completed["excluded_parcel_groups"], 1)
            self.assertEqual(completed["crop_counts"], {})
            self.assertEqual(
                json.loads(Path(completed["inventories"][0]["path"]).read_text())[
                    "features"
                ],
                [],
            )

    def test_invalid_grid_stops_instead_of_publishing_mislocated_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = write_patch(root)
            with Dataset(root / record["filename"], "r+") as dataset:
                dataset.variables["x"][:] += 1
            with self.assertRaisesRegex(ValueError, "Pixel centers"):
                sen4agrinet.extract_patch(root / record["filename"], record)

    def test_hash_mismatch_does_not_modify_old_raw_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = write_patch(root / "raw")
            old = root / "raw" / record["filename"]
            before = old.read_bytes()
            record["sha256"] = "0" * 64
            folder = sen4agrinet.run_folder(root / "refs", manifest_for([record]))
            with self.assertRaisesRegex(FileNotFoundError, "Missing verified source"):
                sen4agrinet.obtain_source(
                    record, manifest_for([record]), folder, root / "raw", False
                )
            self.assertEqual(old.read_bytes(), before)

    def test_explicit_scan_opt_in_builds_capped_inventory_before_satellite_stage(self):
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "notebooks/01_sen4agrinet_preparation.ipynb"
        )
        notebook = json.loads(notebook_path.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "existing_raw"
            records = [
                write_patch(raw_root, 2019, crop_code=231, parcel_id=1),
                write_patch(raw_root, 2020, crop_code=510, parcel_id=2),
            ]
            # Both years intentionally use the same coordinates; year-scoped identities remain separate.
            hub = MagicMock()
            hub.repo_info.return_value.sha = "a" * 40
            hub.list_repo_tree.return_value = [
                SimpleNamespace(
                    path=record["filename"],
                    size=record["expected_bytes"],
                    lfs={"sha256": record["sha256"]},
                    blob_id=None,
                )
                for record in records
            ]
            save_json(
                root / "settings.local.json",
                {
                    "dataset_root": str(raw_root),
                    "report_root": str(root / "reports/inventory"),
                    "annual_archive_root": str(root / "archive"),
                    "earth_engine_project": "not-used-offline",
                },
            )
            old_inventory = root / "reports/fields/2020/download_candidates.geojson"
            save_json(
                old_inventory,
                {"old": "preserve this subset without reading it as complete"},
            )
            original_subset = old_inventory.read_bytes()
            namespace = {}
            with (
                chdir(root),
                redirect_stdout(io.StringIO()),
                patch.dict(os.environ, {"S4A_ROOT": str(raw_root)}),
                patch("huggingface_hub.HfApi", return_value=hub),
                patch(
                    "huggingface_hub.hf_hub_download",
                    side_effect=AssertionError(
                        "Verified existing files must be reused"
                    ),
                ),
                patch(
                    "requests.sessions.Session.request",
                    side_effect=AssertionError("No real network"),
                ),
            ):
                for cell in notebook["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    source = (
                        "".join(cell["source"])
                        .replace(
                            "run_satellite_collection = True",
                            "run_satellite_collection = False",
                        )
                        .replace("authenticate_now = True", "authenticate_now = False")
                        .replace("reference_cached_only = True", "reference_cached_only = False")
                    )
                    exec(compile(source, str(notebook_path), "exec"), namespace)
            self.assertTrue(namespace["retrieve_reference_masks"])
            self.assertTrue(namespace["rebuild_field_inventory"])
            self.assertEqual(namespace["reference_summary"]["files_scanned"], 2)
            self.assertEqual(set(namespace["annual_plan"]["years"]), {"2019", "2020"})
            self.assertEqual(namespace["collection_result"]["status"], "disabled")
            self.assertEqual(old_inventory.read_bytes(), original_subset)

    def test_changed_inventory_fails_offline_integrity_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = write_patch(root / "raw")
            manifest = manifest_for([record])
            completed = sen4agrinet.build_reference_inventory(
                manifest, root / "refs", root / "raw", download_enabled=False
            )
            Path(completed["inventories"][0]["path"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                sen4agrinet.load_completed_inventory(
                    sen4agrinet.run_folder(root / "refs", manifest), manifest
                )
