"""Selective reference reads use synthetic NetCDFs and a simulated Range server."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from netCDF4 import Dataset

from crop_pipeline.sources import remote_masks, sen4agrinet
from test_sen4agrinet_references import write_patch, manifest_for


class RangeResponse:
    def __init__(self, payload, start, end, status=206):
        self.payload = payload[start:end]
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start}-{end-1}/{len(payload)}"}
        self.body_requested = False

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        self.body_requested = True
        yield self.payload


class RangeServer:
    def __init__(self, payload, status=206):
        self.payload, self.status = payload, status
        self.responses = []

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        return False

    def get(self, url, headers, stream, timeout):
        start, end = headers["Range"].split("=")[1].split("-")
        response = RangeResponse(self.payload, int(start), int(end) + 1, self.status)
        self.responses.append(response)
        return response


def patch_with_imagery(root, crop_code=231):
    record = write_patch(root, crop_code=crop_code)
    path = Path(root) / record["filename"]
    with Dataset(path, "r+") as dataset:
        bands = dataset.createGroup("bands")
        bands.createDimension("time", 64)
        bands.createDimension("row", 128)
        bands.createDimension("column", 128)
        variable = bands.createVariable("B03", "f4", ("time", "row", "column"))
        variable[:] = np.ones((64, 128, 128), dtype="float32")
    record.update(
        expected_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return path, record


class RemoteMaskTests(unittest.TestCase):
    def test_selected_masks_match_full_netcdf_without_accessing_band_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, record = patch_with_imagery(root / "upstream")
            manifest = manifest_for([record])
            folder = sen4agrinet.run_folder(root / "refs", manifest)
            reader = remote_masks.RangeReader(
                RangeServer(source.read_bytes()),
                "https://mock/file.nc",
                source.stat().st_size,
            )
            read_variables = []
            original_read = h5py.Dataset.__getitem__

            def track_read(variable, selection):
                read_variables.append(variable.name)
                return original_read(variable, selection)

            with patch.object(h5py.Dataset, "__getitem__", track_read):
                mask_path, receipt = remote_masks.publish_reference_masks(
                    reader,
                    record,
                    folder,
                    tuple(sen4agrinet.CROP_CODES.values()),
                    manifest["revision"],
                )
            self.assertEqual(
                sen4agrinet.extract_reference_patch(mask_path, record),
                sen4agrinet.extract_patch(source, record),
            )
            self.assertFalse(any("B03" in name for name in read_variables))
            self.assertLess(receipt["network_bytes"], source.stat().st_size // 4)
            self.assertLess(mask_path.stat().st_size, source.stat().st_size // 4)
            self.assertFalse(receipt["full_source_checksum_verified"])
            self.assertTrue(receipt["target_labels_present"])

    def test_patches_without_target_labels_skip_parcel_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, record = patch_with_imagery(root / "upstream", crop_code=999)
            manifest = manifest_for([record])
            folder = sen4agrinet.run_folder(root / "refs", manifest)
            reader = remote_masks.RangeReader(
                RangeServer(source.read_bytes()),
                "https://mock/file.nc",
                source.stat().st_size,
            )
            mask_path, receipt = remote_masks.publish_reference_masks(
                reader,
                record,
                folder,
                tuple(sen4agrinet.CROP_CODES.values()),
                manifest["revision"],
            )
            with Dataset(mask_path, "r") as dataset:
                self.assertNotIn("parcels", dataset.variables)
            self.assertFalse(receipt["target_labels_present"])
            self.assertEqual(sen4agrinet.extract_reference_patch(mask_path, record), [])

    def test_ignored_range_stops_before_response_body_is_read(self):
        server = RangeServer(b"do not fetch a complete file", status=200)
        reader = remote_masks.RangeReader(
            server, "https://mock/file.nc", len(server.payload), block_bytes=8
        )
        with self.assertRaisesRegex(RuntimeError, "full NetCDF downloads are disabled"):
            reader.read(4)
        self.assertFalse(server.responses[0].body_requested)

    def test_offset_and_truncated_response_fail_instead_of_publishing(self):
        response = RangeResponse(b"0123456789", 0, 4)
        response.headers["Content-Range"] = "bytes 1-4/10"
        with self.assertRaisesRegex(ValueError, "Content-Range"):
            remote_masks.validate_range_response(response, 0, 4, 10)
        response.payload = b"01"
        with self.assertRaisesRegex(ValueError, "truncated"):
            remote_masks.read_bounded_response(response, 4)
        response.payload = b"012345"
        with self.assertRaisesRegex(ValueError, "exceeded"):
            remote_masks.read_bounded_response(response, 4)

    def test_reader_reuses_blocks_and_refuses_unbounded_reads(self):
        server = RangeServer(b"0123456789abcdef")
        reader = remote_masks.RangeReader(
            server,
            "https://mock/file.nc",
            len(server.payload),
            block_bytes=4,
            cache_blocks=2,
        )
        self.assertEqual(reader.read(3), b"012")
        reader.seek(1)
        self.assertEqual(reader.read(3), b"123")
        self.assertEqual(reader.range_requests, 1)
        reader.seek(-4, 2)
        self.assertEqual(reader.read(4), b"cdef")
        self.assertLessEqual(len(reader.blocks), 2)
        with self.assertRaisesRegex(ValueError, "Unbounded"):
            reader.read()

    def test_entire_inventory_uses_ranges_then_reuses_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, record = patch_with_imagery(root / "upstream")
            manifest = manifest_for([record])
            server = RangeServer(source.read_bytes())
            with (
                patch("requests.Session", return_value=server),
                patch(
                    "huggingface_hub.hf_hub_download",
                    side_effect=AssertionError("Full NetCDF fetch is forbidden"),
                ),
            ):
                completed = sen4agrinet.build_reference_inventory(
                    manifest, root / "refs", root / "absent", reference_only=True
                )
            self.assertEqual(completed["files_scanned"], 1)
            self.assertEqual(completed["crop_counts"]["carrot"]["parcel_groups"], 1)
            folder = sen4agrinet.run_folder(root / "refs", manifest)
            transfer = json.loads((folder / "last_scan_transfer.json").read_text())
            self.assertGreater(transfer["network_bytes"], 0)
            self.assertLess(transfer["network_bytes"], source.stat().st_size // 4)
            with patch(
                "requests.Session",
                side_effect=AssertionError("Checkpoint should avoid all network"),
            ):
                self.assertEqual(
                    sen4agrinet.build_reference_inventory(
                        manifest, root / "refs", root / "absent"
                    ),
                    completed,
                )
            self.assertEqual(
                json.loads((folder / "last_scan_transfer.json").read_text())[
                    "network_bytes"
                ],
                0,
            )

    def test_mask_cache_can_be_reused_by_a_new_extractor_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, record = patch_with_imagery(root / "upstream")
            manifest = manifest_for([record])
            folder = sen4agrinet.run_folder(root / "refs", manifest)
            with patch(
                "requests.Session", return_value=RangeServer(source.read_bytes())
            ):
                original, _ = remote_masks.obtain_reference_masks(
                    record, manifest, folder, tuple(sen4agrinet.CROP_CODES.values())
                )
            another_extractor = folder.parent / "another_extractor"
            with patch(
                "requests.Session",
                side_effect=AssertionError("Cached masks need no network"),
            ):
                reused, receipt = remote_masks.obtain_reference_masks(
                    record,
                    manifest,
                    another_extractor,
                    tuple(sen4agrinet.CROP_CODES.values()),
                    enabled=False,
                )
            self.assertEqual(original, reused)
            self.assertEqual(receipt["network_bytes_this_run"], 0)

    def test_fill_values_and_scale_offsets_preserve_the_original_mask_interpretation(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, record = patch_with_imagery(root / "upstream")
            with Dataset(source, "r+") as dataset:
                labels = dataset.variables["labels"]
                labels.set_auto_maskandscale(False)
                labels[:] = np.full((4, 4), 116, dtype="int32")
                labels[0, 0] = -99
                labels.scale_factor, labels.add_offset, labels.missing_value = (
                    2.0,
                    -1.0,
                    -99,
                )
            record.update(
                expected_bytes=source.stat().st_size,
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            manifest = manifest_for([record])
            folder = sen4agrinet.run_folder(root / "refs", manifest)
            reader = remote_masks.RangeReader(
                RangeServer(source.read_bytes()),
                "https://mock/file.nc",
                source.stat().st_size,
            )
            mask_path, _ = remote_masks.publish_reference_masks(
                reader,
                record,
                folder,
                tuple(sen4agrinet.CROP_CODES.values()),
                manifest["revision"],
            )
            original = sen4agrinet.extract_patch(source, record)
            self.assertEqual(
                sen4agrinet.extract_reference_patch(mask_path, record), original
            )
            self.assertEqual(original[0]["properties"]["crop_pixels"], 15)

    def test_nested_mask_groups_keep_their_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = write_patch(root / "original")
            source = root / "nested.nc"
            with (
                Dataset(root / "original" / record["filename"], "r") as original,
                Dataset(source, "w") as target,
            ):
                for attribute in ("patch_year", "patch_tile", "patch_country_code"):
                    target.setncattr(attribute, original.getncattr(attribute))
                annotations = target.createGroup("annotations")
                annotations.createDimension("x", 4)
                annotations.createDimension("y", 4)
                copy_fixture_variables(original, annotations)
            record.update(
                expected_bytes=source.stat().st_size,
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            manifest = manifest_for([record])
            folder = sen4agrinet.run_folder(root / "refs", manifest)
            reader = remote_masks.RangeReader(
                RangeServer(source.read_bytes()),
                "https://mock/file.nc",
                source.stat().st_size,
            )
            mask_path, _ = remote_masks.publish_reference_masks(
                reader,
                record,
                folder,
                tuple(sen4agrinet.CROP_CODES.values()),
                manifest["revision"],
            )
            self.assertEqual(
                sen4agrinet.extract_reference_patch(mask_path, record),
                sen4agrinet.extract_patch(source, record),
            )

    def test_changed_mask_cache_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, record = patch_with_imagery(root / "upstream")
            manifest = manifest_for([record])
            folder = sen4agrinet.run_folder(root / "refs", manifest)
            reader = remote_masks.RangeReader(
                RangeServer(source.read_bytes()),
                "https://mock/file.nc",
                source.stat().st_size,
            )
            mask_path, _ = remote_masks.publish_reference_masks(
                reader,
                record,
                folder,
                tuple(sen4agrinet.CROP_CODES.values()),
                manifest["revision"],
            )
            mask_path.write_bytes(b"modified cached mask")
            with patch("requests.Session", side_effect=AssertionError("Offline mode")):
                with self.assertRaisesRegex(
                    FileNotFoundError, "Missing cached reference masks"
                ):
                    remote_masks.obtain_reference_masks(
                        record,
                        manifest,
                        folder,
                        tuple(sen4agrinet.CROP_CODES.values()),
                        enabled=False,
                    )


def copy_fixture_variables(original, target):
    for name in ("x", "y", "labels", "parcels"):
        source = original.variables[name]
        variable = target.createVariable(name, source.dtype, source.dimensions)
        variable[:] = source[:]
        for attribute in source.ncattrs():
            variable.setncattr(attribute, source.getncattr(attribute))
