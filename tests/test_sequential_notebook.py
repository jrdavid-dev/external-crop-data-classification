"""A fresh-kernel Run All must not need raw NetCDFs in the default mode."""

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


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "Sen4AgriNet_Satellite_Preparation.ipynb"


def prepare_saved_fields(folder):
    projector = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    features = []
    for crop_number, crop in enumerate(["carrot", "potato"]):
        features.extend(make_crop_fields(projector, crop, crop_number))
    field_folder = folder / "reports/fields/2020"
    field_folder.mkdir(parents=True)
    (field_folder / "download_candidates.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features})
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
    for group_number in range(60):
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
    retriever.download_composites.return_value = {"downloaded": 0, "reused": 0, "failed": 0}
    namespace = {}
    forbidden = AssertionError("Default Run All attempted a raw-data or network operation")
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
        patch("satellite_workflow.SatelliteRetriever", return_value=retriever),
        patch("observed_timelines.retrieve_observed_timelines", return_value=[]),
        redirect_stdout(io.StringIO()),
    ):
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


class SequentialNotebookTests(unittest.TestCase):
    def test_run_all_reuses_saved_fields_with_no_raw_files_or_raw_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            prepare_saved_fields(folder)
            namespace = execute_without_remote_services(folder)
            self.assertFalse(namespace["download_netcdf_files"])
            self.assertFalse(namespace["rebuild_field_inventory"])
            self.assertFalse(namespace["dataset_folder"].exists())
            self.assertEqual(
                Counter(row["crop"] for row in namespace["selected_pixels"]),
                {"carrot": 3000, "potato": 3000},
            )
            self.assertTrue(namespace["source_handoff_path"].is_file())

    def test_missing_saved_inventory_stops_before_any_optional_raw_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "Saved field inventory is missing"):
                execute_without_remote_services(Path(temporary))


if __name__ == "__main__":
    unittest.main()
