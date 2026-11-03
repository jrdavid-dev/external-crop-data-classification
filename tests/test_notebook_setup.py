"""Clone setup can start without local settings, installed pipeline or cached tables."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from crop_pipeline import notebook_setup
from crop_pipeline.sources import francecrops


class NotebookSetupTests(unittest.TestCase):
    def test_defaults_are_created_without_overwriting_local_customizations(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, {"EE_PROJECT": ""}),
        ):
            path = Path(temporary) / "settings.local.json"
            settings = notebook_setup.ensure_local_settings(path)
            self.assertEqual(
                settings["annual_archive_root"], "./sen4agrinet_work/annual_archive"
            )
            custom = {
                **settings,
                "annual_archive_root": "/custom/archive",
                "extra": "preserve",
            }
            path.write_text(json.dumps(custom))
            before = path.read_bytes()
            self.assertEqual(notebook_setup.ensure_local_settings(path), custom)
            self.assertEqual(path.read_bytes(), before)

    def test_project_prompt_is_saved_once_and_offline_mode_does_not_prompt(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, {"EE_PROJECT": ""}),
        ):
            path = Path(temporary) / "settings.local.json"
            settings = notebook_setup.ensure_local_settings(path)
            with patch("builtins.input", side_effect=AssertionError("Offline mode")):
                self.assertEqual(
                    notebook_setup.project_for_collection(settings, path, False), ""
                )
            with patch("builtins.input", return_value="thesis-project") as prompt:
                self.assertEqual(
                    notebook_setup.project_for_collection(settings, path, True),
                    "thesis-project",
                )
                self.assertEqual(
                    notebook_setup.project_for_collection(
                        notebook_setup.read_settings(path), path, True
                    ),
                    "thesis-project",
                )
            prompt.assert_called_once()
            self.assertEqual(
                json.loads(path.read_text())["earth_engine_project"], "thesis-project"
            )

    def test_bootstrap_installs_only_missing_requirements_in_the_current_kernel(self):
        import sys

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "requirements-notebooks.txt").write_text(
                "numpy==2.5.4\nrequests==2.34.2\n"
            )
            with (
                patch.object(
                    notebook_setup, "missing_packages", return_value=["numpy==2.5.4"]
                ),
                patch.object(notebook_setup, "ensure_editable_package"),
                patch.object(notebook_setup.subprocess, "check_call") as install,
                redirect_stdout(io.StringIO()),
            ):
                notebook_setup.prepare_environment(root)
            install.assert_called_once_with(
                [sys.executable, "-m", "pip", "install", "numpy==2.5.4"]
            )

    def test_editable_install_is_only_needed_for_another_or_missing_checkout(self):
        import sys

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with (
                patch.object(
                    notebook_setup, "editable_package_root", return_value=root
                ),
                patch.object(notebook_setup.subprocess, "check_call") as install,
            ):
                notebook_setup.ensure_editable_package(root)
            install.assert_not_called()
            with (
                patch.object(
                    notebook_setup, "editable_package_root", return_value=None
                ),
                patch.object(notebook_setup.subprocess, "check_call") as install,
            ):
                notebook_setup.ensure_editable_package(root)
            install.assert_called_once_with(
                [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(root)]
            )

    def test_francecrops_fetches_only_missing_pinned_reference_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / francecrops.TABLE_FILES["class_map"]
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"preserve-existing-table")

            def fetch(**arguments):
                self.assertEqual(arguments["revision"], francecrops.DEFAULT_REVISION)
                self.assertEqual(arguments["repo_id"], francecrops.DATASET_NAME)
                path = root / arguments["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test table")
                return str(path)

            with patch(
                "huggingface_hub.hf_hub_download", side_effect=fetch
            ) as downloader:
                paths = francecrops.download_reference_tables(root)
            self.assertEqual(downloader.call_count, 2)
            self.assertEqual(existing.read_bytes(), b"preserve-existing-table")
            self.assertEqual(set(paths), set(francecrops.TABLE_FILES))
            with patch(
                "huggingface_hub.hf_hub_download",
                side_effect=AssertionError("Reuse cached files"),
            ):
                francecrops.download_reference_tables(root)

    def test_francecrops_offline_missing_table_does_not_start_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "huggingface_hub.hf_hub_download", side_effect=AssertionError("Offline")
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError, "download_missing_reference_tables"
                ):
                    francecrops.download_reference_tables(
                        temporary, download_if_missing=False
                    )

    def test_francecrops_run_all_starts_without_settings_or_cached_tables(self):
        import json
        from contextlib import chdir
        import pyarrow as arrow
        import pyarrow.parquet as parquet
        from pyproj import CRS, Transformer
        from shapely.geometry import box
        from shapely.ops import transform

        repository = Path(__file__).resolve().parents[1]
        notebook_path = repository / "notebooks/02_francecrops_preparation.ipynb"
        notebook = json.loads(notebook_path.read_text())
        labels = [
            {"parcel_id": "cabbage", "y": 4, "split": "train"},
            {"parcel_id": "potato", "y": 12, "split": "train"},
        ]
        projector = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
        geometries = [
            {
                "parcel_id": row["parcel_id"],
                "geometry": transform(
                    projector.transform,
                    box(500000 + index * 300, 5200000, 500100 + index * 300, 5200100),
                ).wkb,
            }
            for index, row in enumerate(labels)
        ]
        tables = {
            "data/protocol/class_map.parquet": arrow.Table.from_pylist(
                [
                    {"y": 4, "class_code": "CHU", "class_name_en": "cabbage"},
                    {"y": 12, "class_code": "PTC", "class_name_en": "potato"},
                ]
            ),
            "data/protocol/metadata.parquet": arrow.Table.from_pylist(labels),
            "data/geolocation/geolocation.parquet": arrow.Table.from_pylist(
                geometries
            ).replace_schema_metadata(
                {
                    b"geo": json.dumps(
                        {
                            "columns": {
                                "geometry": {"crs": CRS.from_epsg(4326).to_json_dict()}
                            }
                        }
                    ).encode()
                }
            ),
        }

        def fetch(**arguments):
            destination = Path(arguments["local_dir"]) / arguments["filename"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            parquet.write_table(tables[arguments["filename"]], destination)
            return str(destination)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace = {}
            with (
                chdir(root),
                redirect_stdout(io.StringIO()),
                patch("huggingface_hub.hf_hub_download", side_effect=fetch) as download,
                patch(
                    "requests.sessions.Session.request",
                    side_effect=AssertionError("No real network"),
                ),
            ):
                execute_source_cells(notebook, namespace, notebook_path)
            self.assertEqual(download.call_count, 3)
            self.assertTrue((root / "settings.local.json").is_file())
            self.assertEqual(namespace["annual_plan"]["estimate"]["parcels"], 2)
            self.assertEqual(namespace["collection_result"]["status"], "disabled")
            self.assertTrue(namespace["download_missing_reference_tables"])


def execute_source_cells(notebook, namespace, notebook_path):
    """Exercise reference setup with collection disabled and all real network blocked."""
    for cell in [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]:
        source = (
            "".join(cell["source"])
            .replace(
                "run_satellite_collection = True", "run_satellite_collection = False"
            )
            .replace("authenticate_now = True", "authenticate_now = False")
        )
        exec(compile(source, str(notebook_path), "exec"), namespace)


class FreshSen4AgriNetNotebookTests(unittest.TestCase):
    def test_run_all_without_raw_files_or_settings_fetches_the_full_reference_release(
        self,
    ):
        from contextlib import chdir
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from test_sen4agrinet_references import write_patch
        from crop_pipeline.sources import remote_masks

        repository = Path(__file__).resolve().parents[1]
        notebook_path = repository / "notebooks/01_sen4agrinet_preparation.ipynb"
        notebook = json.loads(notebook_path.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            simulated_server = root / "mock_upstream"
            checkout = root / "fresh_clone"
            checkout.mkdir()
            records = [
                write_patch(simulated_server, 2019, crop_code=231, parcel_id=1),
                write_patch(simulated_server, 2020, crop_code=213, parcel_id=2),
            ]
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

            def fetch_masks(record, manifest, folder, crop_codes, enabled):
                self.assertTrue(enabled)
                self.assertEqual(manifest["revision"], "a" * 40)
                return simulated_server / record["filename"], {
                    "retrieval": "http_range_masks_only",
                    "network_bytes_this_run": 0,
                }

            namespace = {}
            with (
                chdir(checkout),
                redirect_stdout(io.StringIO()),
                patch.dict(os.environ, {"S4A_ROOT": str(checkout / "no_raw_files")}),
                patch("huggingface_hub.HfApi", return_value=hub),
                patch.object(
                    remote_masks, "obtain_reference_masks", side_effect=fetch_masks
                ) as downloader,
                patch(
                    "huggingface_hub.hf_hub_download",
                    side_effect=AssertionError("Full NetCDF downloads forbidden"),
                ),
                patch(
                    "requests.sessions.Session.request",
                    side_effect=AssertionError("No real network"),
                ),
            ):
                execute_source_cells(notebook, namespace, notebook_path)
            self.assertEqual(downloader.call_count, 2)
            self.assertTrue((checkout / "settings.local.json").is_file())
            self.assertEqual(namespace["reference_summary"]["files_scanned"], 2)
            self.assertEqual(set(namespace["annual_plan"]["years"]), {"2019", "2020"})
            self.assertEqual(namespace["collection_result"]["status"], "disabled")
            self.assertFalse((checkout / "no_raw_files").exists())
