"""Protect explicit legacy reuse and honest acquisition provenance."""

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pilot_sources
from satellite_workflow import REQUEST_VERSION, RESAMPLING_POLICY


class ExistingImageryHandoffTests(unittest.TestCase):
    def test_missing_required_source_stops_before_loading_any_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            existing = folder / "sen4agrinet_bilinear_expanded.json"
            missing = folder / "francecrops_bilinear_expanded.json"
            existing.write_text("invalid JSON should never be read")
            with patch.object(pilot_sources.json, "loads") as load_manifest:
                with self.assertRaisesRegex(
                    FileNotFoundError, "francecrops_bilinear_expanded"
                ):
                    pilot_sources.load_sources([existing, missing], require_all=True)
                load_manifest.assert_not_called()
            self.assertEqual(existing.read_text(), "invalid JSON should never be read")
            self.assertFalse(missing.exists())

    def test_all_required_sources_are_loaded_and_contract_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            region_path = folder / "sampled_download_regions.geojson"
            region_path.write_text('{"features": []}')
            paths = []
            for source_name in ("sen4agrinet", "francecrops"):
                manifest_path = folder / f"{source_name}_bilinear_expanded.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "source_name": source_name,
                            "field_folder": str(folder),
                            "regions_sha256": pilot_sources.file_sha256(region_path),
                            "request_version": REQUEST_VERSION,
                            "resampling": RESAMPLING_POLICY,
                        }
                    )
                )
                paths.append(manifest_path)
            sources = pilot_sources.load_sources(paths, require_all=True)
            self.assertEqual(
                [source["source_name"] for source in sources],
                ["sen4agrinet", "francecrops"],
            )
            invalid = json.loads(paths[1].read_text())
            invalid["request_version"] = 3
            paths[1].write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "old imagery contract"):
                pilot_sources.load_sources(paths, require_all=True)

    def test_saved_contract_requires_opt_in_and_matching_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            fields = folder / "fields"
            progress = folder / "progress"
            fields.mkdir()
            progress.mkdir()
            region_path = fields / "sampled_download_regions.geojson"
            region_path.write_text(
                json.dumps({"features": [{"properties": {"fragment_id": "parcel"}}]})
            )
            (progress / "parcel.json").write_text(
                json.dumps(
                    {
                        "fragment_id": "parcel",
                        "settings": {"version": 3, "resampling": "nearest"},
                    }
                )
            )
            manifest_path = folder / "source.json"
            payload = pilot_sources.save_source_manifest(
                manifest_path,
                "sen4agrinet",
                2020,
                fields,
                folder / "images",
                progress,
                folder / "timelines",
                "revision",
            )
            self.assertEqual(payload["request_version"], 3)
            self.assertEqual(payload["resampling"], "nearest")
            with self.assertRaises(ValueError):
                pilot_sources.load_sources([manifest_path])
            sources = pilot_sources.load_sources(
                [manifest_path], allow_existing_version_3=True
            )
            self.assertEqual(sources[0]["request_version"], 3)
            # Old data must never change what a future retrieval requests.
            self.assertEqual(REQUEST_VERSION, 4)
            self.assertEqual(RESAMPLING_POLICY["B5"], "bilinear")
            region_path.write_text(json.dumps({"features": []}))
            with self.assertRaisesRegex(ValueError, "field folder"):
                pilot_sources.save_source_manifest(
                    manifest_path,
                    "sen4agrinet",
                    2020,
                    fields,
                    folder / "images",
                    progress,
                    folder / "timelines",
                    "revision",
                )

    def test_mixed_download_contracts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for index, version in enumerate([3, 4]):
                (folder / f"{index}.json").write_text(
                    json.dumps(
                        {
                            "fragment_id": "parcel",
                            "settings": {
                                "version": version,
                                "resampling": (
                                    "nearest" if version == 3 else RESAMPLING_POLICY
                                ),
                            },
                        }
                    )
                )
            with self.assertRaisesRegex(ValueError, "Mixed imagery contracts"):
                pilot_sources.save_source_manifest(
                    folder / "source.json",
                    "sen4agrinet",
                    2020,
                    folder,
                    folder / "images",
                    folder,
                    folder / "timelines",
                    "revision",
                )
