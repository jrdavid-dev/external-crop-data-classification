from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import box, mapping

from crop_pipeline.common import interval_bins, save_json
from crop_pipeline.imagery import annual_tif
from crop_pipeline.imagery.retrieval import REQUEST_VERSION, RESAMPLING_POLICY
from crop_pipeline.sources import adaptive, jecam_transfer, test_cohort
from crop_pipeline.windows import external
from crop_pipeline.windows.seasons import file_sha256


def completed_batch(report, source, parcels):
    is_test = source == test_cohort.TEST
    policy = test_cohort.TEST_POLICY if is_test else None
    batch = adaptive.root(report, policy) / "rounds/0001" / source
    grid = {"width": 1, "height": 1, "crs": "EPSG:32631", "transform": [10, 0, 500000, 0, -10, 5200000]}
    fields = batch / "fields"
    fields.mkdir(parents=True)
    timelines = batch / "timelines"
    regions, tiffs = [], {}
    descriptions = [f"{period['name']}__{band}"
                    for period in interval_bins(2020) for band in annual_tif.BANDS]
    for group, bounds in parcels.items():
        mask, image = fields / f"{group}_mask.tif", fields / f"{group}_annual.tif"
        options = dict(driver="GTiff", width=1, height=1, crs=grid["crs"], transform=Affine(*grid["transform"]))
        with rasterio.open(mask, "w", **options, count=1, dtype="uint8") as output:
            output.write(np.ones((1, 1, 1), dtype="uint8"))
        with rasterio.open(image, "w", **options, count=len(descriptions), dtype="float32", nodata=-9999) as output:
            values = np.tile([1000] * (len(annual_tif.BANDS) - 1) + [1], 24)
            output.write(values.astype("float32").reshape(-1, 1, 1))
            output.descriptions = tuple(descriptions)
        regions.append({
            "type": "Feature", "geometry": mapping(box(*bounds)),
            "properties": {"fragment_id": group, "partition_group": group, "crop": "carrot",
                           "selected_pixels": 1, "sampling_grid": grid, "sampling_mask": str(mask)},
        })
        tiffs[group] = {"path": str(image), "sha256": file_sha256(image)}
        save_json(timelines / "tiles" / group / "observed_timelines.json", [])
    region_path = fields / "sampled_download_regions.geojson"
    save_json(region_path, {"features": regions})
    save_json(fields / "source_footprints.geojson", {"features": regions})
    handoff = batch / "handoff.json"
    save_json(handoff, {
        "source_name": f"{source}_2020", "year": 2020, "dataset_revision": "fixture",
        "request_version": REQUEST_VERSION, "resampling": RESAMPLING_POLICY,
        "archive_format": "selected_pixel_parquet_v1", "field_folder": str(fields),
        "annual_tif_root": str(fields), "annual_tif_index": tiffs,
        "timeline_folder": str(timelines), "regions_sha256": file_sha256(region_path),
    })
    selection = {
        "source": source, "round": 1, "groups": list(parcels),
        "physical_keys": [f"{source}||{group}" for group in parcels],
        "assignments": dict.fromkeys(parcels, "test" if is_test else "train"),
    }
    save_json(batch / "selection.json", selection)
    save_json(batch / "state.json", {"status": "completed"})
    save_json(batch / "completed.json", {"source": source, "round": 1,
                                         "handoffs": [{"path": str(handoff), "sha256": file_sha256(handoff)}]})
    return batch


class JecamTransferTests(unittest.TestCase):
    def test_imported_batch_audits_without_sender_and_excludes_overlap(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sender = root / "sender/inventory"
            receiver = root / "receiver/inventory"
            completed_batch(sender, test_cohort.TEST, {"overlap": (1, 1, 3, 3), "keep": (4, 4, 5, 5)})
            development = completed_batch(receiver, "sen4agrinet", {"sen": (0, 0, 2, 2)})
            completed_batch(receiver, "francecrops", {"fra": (6, 6, 7, 7)})
            original = (development / "completed.json").read_bytes()

            exports = jecam_transfer.export_completed(sender, root / "share")
            shutil.rmtree(sender.parent)
            [imported] = jecam_transfer.import_completed(receiver, root / "share")
            self.assertEqual(jecam_transfer.import_completed(receiver, exports[0]), [imported])
            self.assertEqual((development / "completed.json").read_bytes(), original)

            quality = external.audit(receiver, root / "analysis")
            saved = jecam_transfer.read(quality)
            self.assertEqual(saved["overlap_exclusions"], ["overlap"])
            self.assertEqual(set(saved["pixel_paths"]), {"keep", "sen", "fra"})
            self.assertEqual(jecam_transfer.read(quality.parent / "jecam_overlap.json"),
                             {"excluded_parcels": ["overlap"]})
            self.assertTrue((imported / "year_0000/tiffs/000000.tif").is_file())

    def test_checksum_and_conflict_leave_existing_pending_batch_unchanged(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sender, receiver = root / "sender/inventory", root / "receiver/inventory"
            completed_batch(sender, test_cohort.TEST, {"same": (1, 1, 3, 3)})
            pending = completed_batch(receiver, test_cohort.TEST, {"same": (1, 1, 3, 3)})
            (pending / "completed.json").unlink()
            state = pending / "state.json"
            save_json(state, {"status": "pending_retry", "task": "keep"})
            original = state.read_bytes()
            [bundle] = jecam_transfer.export_completed(sender, root / "share")
            with self.assertRaisesRegex(ValueError, "conflicts with an existing parcel"):
                jecam_transfer.import_completed(receiver, bundle)
            image = bundle / "year_0000/tiffs/000000.tif"
            image.write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                jecam_transfer.import_completed(receiver, bundle)
            self.assertEqual(state.read_bytes(), original)
            self.assertEqual(list(pending.parent.iterdir()), [pending])
