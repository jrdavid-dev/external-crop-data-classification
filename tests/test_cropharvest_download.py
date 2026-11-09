"""Missing-label retrieval is pinned, validated and never overwrites local inputs."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from crop_pipeline.sources import cropharvest


def label_bytes(identifier="example"):
    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": None, "properties": {"uid": identifier}}
            ],
        }
    ).encode()


class CropHarvestLabelDownloadTests(unittest.TestCase):
    def test_existing_valid_labels_are_reused_without_network_or_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.geojson"
            original = label_bytes()
            path.write_bytes(original)
            with patch.object(
                cropharvest,
                "download_labels_payload",
                side_effect=AssertionError("Existing labels must not be downloaded"),
            ):
                result = cropharvest.ensure_labels(path)
            self.assertEqual(result["status"], "reused")
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(path.with_suffix(".download.json").exists())

    def test_missing_labels_publish_only_after_validation_with_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested/labels.geojson"
            payload = label_bytes()
            with patch.object(
                cropharvest,
                "download_labels_payload",
                side_effect=lambda destination: destination.write_bytes(payload),
            ) as downloader:
                result = cropharvest.ensure_labels(
                    path, expected_md5=hashlib.md5(payload).hexdigest()
                )
            downloader.assert_called_once()
            self.assertEqual(result["status"], "downloaded")
            self.assertEqual(path.read_bytes(), payload)
            provenance = json.loads(path.with_suffix(".download.json").read_text())
            self.assertEqual(provenance["zenodo_record"], "10251170")
            self.assertEqual(provenance["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(list(path.parent.glob("*.part")), [])

    def test_wrong_checksum_does_not_publish_or_leave_partial_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.geojson"
            with patch.object(
                cropharvest,
                "download_labels_payload",
                side_effect=lambda destination: destination.write_bytes(label_bytes()),
            ):
                with self.assertRaisesRegex(ValueError, "MD5 checksum"):
                    cropharvest.ensure_labels(path, expected_md5="0" * 32)
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.iterdir()), [])

    def test_matching_checksum_on_html_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.geojson"
            payload = b"<html>download failed</html>"
            with patch.object(
                cropharvest,
                "download_labels_payload",
                side_effect=lambda destination: destination.write_bytes(payload),
            ):
                with self.assertRaises(ValueError):
                    cropharvest.ensure_labels(
                        path, expected_md5=hashlib.md5(payload).hexdigest()
                    )
            self.assertFalse(path.exists())

    def test_interrupted_download_cleans_its_partial_file(self):
        def interrupted(destination):
            destination.write_bytes(b"partial")
            raise requests.Timeout("simulated interruption")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.geojson"
            with patch.object(
                cropharvest, "download_labels_payload", side_effect=interrupted
            ):
                with self.assertRaises(requests.Timeout):
                    cropharvest.ensure_labels(path)
            self.assertEqual(list(path.parent.iterdir()), [])

    def test_invalid_existing_labels_are_preserved_for_investigation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.geojson"
            path.write_text("not valid labels")
            with patch.object(
                cropharvest,
                "download_labels_payload",
                side_effect=AssertionError("Do not replace existing files"),
            ):
                with self.assertRaises(ValueError):
                    cropharvest.ensure_labels(path)
            self.assertEqual(path.read_text(), "not valid labels")

    def test_offline_mode_missing_labels_stops_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(
                cropharvest,
                "download_labels_payload",
                side_effect=AssertionError("Offline mode"),
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError, "download_missing_labels"
                ):
                    cropharvest.ensure_labels(
                        Path(temporary) / "labels.geojson", download_if_missing=False
                    )

    def test_another_kernels_labels_are_not_overwritten_during_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.geojson"
            payload, competing = label_bytes("downloaded"), label_bytes("other_kernel")

            def concurrent_download(destination):
                destination.write_bytes(payload)
                path.write_bytes(competing)

            with patch.object(
                cropharvest, "download_labels_payload", side_effect=concurrent_download
            ):
                result = cropharvest.ensure_labels(
                    path, expected_md5=hashlib.md5(payload).hexdigest()
                )
            self.assertEqual(result["status"], "reused")
            self.assertEqual(path.read_bytes(), competing)

    def test_notebook_fetches_missing_labels_before_hashing_the_run(self):
        import io
        from contextlib import chdir, redirect_stdout

        repository = Path(__file__).resolve().parents[1]
        notebook_path = repository / "notebooks/03_cropharvest_preparation.ipynb"
        notebook = json.loads(notebook_path.read_text())
        payload = label_bytes()
        original_ensure = cropharvest.ensure_labels

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copied_notebook = root / "notebooks/03_cropharvest_preparation.ipynb"
            copied_notebook.parent.mkdir()
            copied_notebook.write_bytes(notebook_path.read_bytes())
            batch_path = root / "src/crop_pipeline/imagery/batch_export.py"
            batch_path.parent.mkdir(parents=True)
            batch_path.write_bytes(
                (repository / "src/crop_pipeline/imagery/batch_export.py").read_bytes()
            )
            namespace = {}
            ensure_fixture = lambda path, download_if_missing: original_ensure(
                path, download_if_missing, hashlib.md5(payload).hexdigest()
            )
            with (
                chdir(root),
                redirect_stdout(io.StringIO()),
                patch("builtins.input", return_value="fixture-project"),
                patch.object(
                    cropharvest,
                    "download_labels_payload",
                    side_effect=lambda destination: destination.write_bytes(payload),
                ),
                patch.object(cropharvest, "ensure_labels", side_effect=ensure_fixture),
            ):
                for cell in [
                    cell
                    for cell in notebook["cells"][:15]
                    if cell["cell_type"] == "code"
                ]:
                    source = "".join(cell["source"]).replace(
                        'reference_mode = "reuse"', 'reference_mode = "download_missing"'
                    )
                    # Stop after audit identity creation: this test uses a minimal
                    # GeoJSON fixture to check retrieval precedes content hashing.
                    source = source.split("label_review = cropharvest_review.review_labels(")[0]
                    exec(compile(source, str(notebook_path), "exec"), namespace)
                    if "run_folder" in namespace:
                        break
            self.assertEqual(namespace["reference_mode"], "download_missing")
            self.assertTrue(namespace["reference_options"]["download_if_missing"])
            self.assertEqual(namespace["labels_download"]["status"], "downloaded")
            self.assertEqual(
                namespace["run_configuration"]["labels_sha256"],
                hashlib.sha256(payload).hexdigest(),
            )
            self.assertTrue(namespace["run_folder"].is_dir())
