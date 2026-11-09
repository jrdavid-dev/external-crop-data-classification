"""A fresh kernel can reuse a completed full-release inventory entirely offline."""

import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import chdir, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform as transform_geometry

from crop_pipeline.common import save_json
from crop_pipeline.sources import sen4agrinet
from crop_pipeline.windows.seasons import file_sha256


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1] / "notebooks/01_sen4agrinet_preparation.ipynb"
)


def prepare_saved_fields(folder):
    projector = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    features = []
    for crop_number, crop in enumerate(["carrot", "potato"]):
        features.extend(make_crop_fields(projector, crop, crop_number))
    reference_root = folder / "reports/annual_archive/sen4agrinet_references"
    manifest = {
        "dataset": sen4agrinet.DATASET_NAME,
        "revision": "a" * 40,
        "scope": "all_netcdf_files_under_data",
        "files": [{"filename": "data/2020/31TBF/reference.nc", "expected_bytes": 100}],
        "years": [2020],
        "year_tile_pairs": ["2020/31TBF"],
        "total_source_bytes": 100,
    }
    manifest_path = reference_root / "repository_manifest.json"
    save_json(manifest_path, manifest)
    save_json(
        reference_root / "selected_release.json",
        {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
    )
    run_folder = sen4agrinet.run_folder(reference_root, manifest)
    field_folder = run_folder / "fields/2020"
    region_path = field_folder / "download_candidates.geojson"
    save_json(region_path, {"type": "FeatureCollection", "features": features})
    save_json(
        run_folder / "completed_inventory.json",
        {
            "status": "complete",
            "revision": manifest["revision"],
            "files_scanned": 1,
            "crop_counts": {
                "carrot": {"parcel_groups": 120},
                "potato": {"parcel_groups": 120},
            },
            "excluded_parcel_groups": 0,
            "field_root": str(run_folder / "fields"),
            "inventories": [
                {"path": str(region_path), "sha256": file_sha256(region_path)}
            ],
        },
    )
    settings = {
        "report_root": str(folder / "reports/inventory"),
        "dataset_root": str(folder / "absent_raw_data"),
        "imagery_root": str(folder / "images"),
        "earth_engine_project": "test-project",
    }
    (folder / "settings.local.json").write_text(json.dumps(settings))


def make_crop_fields(projector, crop, crop_number):
    features = []
    for group_number in range(120):
        left = 500000 + group_number * 200
        bottom = 5200000 + crop_number * 1000
        geometry = transform_geometry(
            projector.transform, box(left, bottom, left + 100, bottom + 100)
        )
        parcel_id = f"{crop}_{group_number}"
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": {
                    "country": "FR",
                    "year": 2020,
                    "parcel_id": parcel_id,
                    "fragment_id": parcel_id,
                    "crop": crop,
                    "source_crs": "EPSG:32631",
                },
            }
        )
    return features


def execute_without_remote_services(folder):
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    retriever = MagicMock()
    retriever.year = 2020
    retriever.build_plans.return_value = ([], [])
    retriever.download_composites.return_value = {
        "downloaded": 0,
        "reused": 0,
        "failed": 0,
    }
    namespace = {}
    forbidden = AssertionError(
        "Default Run All attempted a raw-data or network operation"
    )
    with (
        chdir(folder),
        patch.dict("os.environ", {"S4A_ROOT": str(folder / "absent_raw_data")}),
        patch.dict("sys.modules", {"ee": MagicMock()}),
        patch("huggingface_hub.HfApi.list_repo_tree", side_effect=forbidden),
        patch("huggingface_hub.hf_hub_download", side_effect=forbidden),
        patch("netCDF4.Dataset", side_effect=forbidden),
        patch("requests.sessions.Session.request", side_effect=forbidden),
        patch("subprocess.check_call", side_effect=forbidden),
        patch("importlib.reload", side_effect=lambda module: module),
        patch(
            "crop_pipeline.imagery.retrieval.SatelliteRetriever", return_value=retriever
        ),
        patch(
            "crop_pipeline.imagery.timelines.retrieve_observed_timelines",
            return_value=[],
        ),
        redirect_stdout(io.StringIO()),
    ):
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                # Exercise explicit offline reuse while collection defaults are enabled.
                source = "".join(cell["source"])
                for flag in (
                    "retrieve_reference_masks",
                    "rebuild_field_inventory",
                    "run_satellite_collection",
                    "retrieve_daily_summaries",
                    "download_satellite_imagery",
                    "authenticate_now",
                ):
                    source = source.replace(f"{flag} = True", f"{flag} = False")
                exec(
                    compile(source, str(NOTEBOOK_PATH), "exec"),
                    namespace,
                )
    return namespace


class SequentialNotebookTests(unittest.TestCase):
    def test_run_all_reuses_completed_release_with_no_raw_files_or_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            prepare_saved_fields(folder)
            namespace = execute_without_remote_services(folder)
            self.assertFalse(namespace["retrieve_reference_masks"])
            self.assertFalse(namespace["rebuild_field_inventory"])
            self.assertFalse(namespace["dataset_folder"].exists())
            plan = namespace["annual_plan"]
            self.assertEqual(namespace["candidate_parcels_per_crop"], 250)
            # The fixture has only 120 eligible parcels per crop.
            self.assertEqual(plan["estimate"]["parcels"], 240)
            self.assertEqual(
                plan["years"]["2020"]["pixels_per_crop"],
                {"carrot": 12000, "potato": 12000},
            )
            self.assertEqual(namespace["collection_result"]["status"], "disabled")
            self.assertFalse(namespace["run_satellite_collection"])

    def test_offline_mode_requires_complete_dataset_wide_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                FileNotFoundError, "No complete dataset-wide reference discovery"
            ):
                execute_without_remote_services(Path(temporary))


if __name__ == "__main__":
    unittest.main()
