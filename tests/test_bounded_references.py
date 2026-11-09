import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crop_pipeline.sources import bounded_references, sen4agrinet
from test_sen4agrinet_references import write_patch, manifest_for


class BoundedReferenceTests(unittest.TestCase):
    def test_stops_resumes_and_keeps_full_inventory_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            records = [write_patch(folder/'raw', crop_code=code, parcel_id=i+1)
                       for i, code in enumerate([231, 213, 510, 231, 213, 510])]
            manifest = manifest_for(records)
            # Force an easily inspected patch order; process_patch still uses real validation/extraction.
            with patch.object(bounded_references, 'ordered_patches', return_value=iter(records)):
                result = bounded_references.prepare(manifest, folder/'references', folder/'raw', parcels_per_crop=1)
            self.assertEqual(result['files_scanned'], 3)
            self.assertEqual(result['stop_reason'], 'parcel_targets_reached')
            self.assertEqual(result['scope'], 'bounded_training_reference_subset')
            self.assertEqual(result['eligible_physical_parcels'], {'cabbage': 1, 'carrot': 1, 'potato': 1})
            self.assertFalse((sen4agrinet.run_folder(folder/'references', manifest)/'completed_inventory.json').exists())
            with patch.object(sen4agrinet, 'process_patch', side_effect=AssertionError('network')):
                cached = bounded_references.prepare(manifest, folder/'references', folder/'raw', parcels_per_crop=1, scan_enabled=False)
            self.assertEqual(cached, result)
            inventory = Path(result['inventories'][0]['path'])
            inventory.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                bounded_references.prepare(manifest, folder/'references', folder/'raw', parcels_per_crop=1)

    def test_exhaustion_and_repeated_physical_parcels(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            records = [write_patch(folder/'raw', year=year, parcel_id=1) for year in [2019, 2020]]
            result = bounded_references.prepare(manifest_for(records), folder/'references', folder/'raw', parcels_per_crop=2)
            self.assertEqual(result['stop_reason'], 'catalogue_exhausted')
            self.assertEqual(result['eligible_physical_parcels']['carrot'], 1)
            self.assertEqual(result['deficits'], {'carrot': 1, 'cabbage': 2, 'potato': 2})

    def test_failed_patch_does_not_publish_and_successful_checkpoint_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            records = [write_patch(folder/'raw', parcel_id=1), write_patch(folder/'raw', parcel_id=2)]
            manifest = manifest_for(records)
            real_process = sen4agrinet.process_patch
            def interrupted(record, *args, **kwargs):
                if record == records[1]:
                    raise RuntimeError('interrupted')
                return real_process(record, *args, **kwargs)
            with patch.object(bounded_references, 'ordered_patches', side_effect=lambda *a: iter(records)), \
                 patch.object(sen4agrinet, 'process_patch', side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    bounded_references.prepare(manifest, folder/'references', folder/'raw', parcels_per_crop=2)
            self.assertFalse(list((folder/'references').rglob('completed_inventory.json')))
            with patch.object(bounded_references, 'ordered_patches', return_value=iter(records)):
                result = bounded_references.prepare(manifest, folder/'references', folder/'raw', parcels_per_crop=2)
            transfer = json.loads((Path(result['field_root']).parent/'last_scan_transfer.json').read_text())
            self.assertEqual(transfer['patches_reused_from_checkpoints'], 1)

    def test_mixed_and_held_out_references_do_not_fill_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            records = [write_patch(folder/'raw', crop_code=code, parcel_id=i+1)
                       for i, code in enumerate([231, 213, 510])]
            real_process = sen4agrinet.process_patch
            def altered(record, *args, **kwargs):
                result = real_process(record, *args, **kwargs)
                if record == records[0]:
                    result['features'][0]['properties']['mixed_label_parcel'] = True
                if record == records[1]:
                    result['features'][0]['properties']['source_split'] = 'test'
                import hashlib
                result['features_sha256'] = hashlib.sha256(json.dumps(result['features'], sort_keys=True).encode()).hexdigest()
                from crop_pipeline.common import save_json
                save_json(sen4agrinet.checkpoint_path(args[1], record), result)
                return result
            with patch.object(sen4agrinet, 'process_patch', side_effect=altered):
                result = bounded_references.prepare(manifest_for(records), folder/'references', folder/'raw', parcels_per_crop=1)
            self.assertEqual(result['eligible_physical_parcels'], {'carrot': 0, 'cabbage': 0, 'potato': 1})
            self.assertEqual(result['excluded_parcel_groups'], 1)

    def test_cap_publishes_available_parcels_without_waiting_for_cabbage(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            records = [write_patch(folder/'raw', crop_code=code, parcel_id=i+1)
                       for i, code in enumerate([231, 510, 231, 510])]
            with patch.object(bounded_references, 'ordered_patches', return_value=iter(records)):
                result = bounded_references.prepare(manifest_for(records), folder/'references', folder/'raw',
                    parcels_per_crop=10, target_crops=('carrot', 'potato'), max_patches=2)
            self.assertEqual(result['files_scanned'], 2)
            self.assertEqual(result['stop_reason'], 'patch_cap_reached')
            self.assertEqual(result['deficits'], {'carrot': 9, 'potato': 9})
            self.assertEqual(result['status'], 'complete')

    def test_cached_only_never_fetches_missing_or_corrupt_patches(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            records = [write_patch(folder/'raw', crop_code=code, parcel_id=i+1)
                       for i, code in enumerate([231, 510, 231])]
            manifest = manifest_for(records)
            checkpoint_folder = sen4agrinet.run_folder(folder/'references', manifest)
            sen4agrinet.process_patch(records[1], manifest, checkpoint_folder, folder/'raw', False, False)
            corrupt = sen4agrinet.checkpoint_path(checkpoint_folder, records[0])
            corrupt.parent.mkdir(parents=True, exist_ok=True)
            corrupt.write_text('{}')
            with patch.object(sen4agrinet, 'process_patch', side_effect=AssertionError('must not fetch')):
                result = bounded_references.prepare(manifest, folder/'references', folder/'raw',
                    target_crops=('carrot', 'potato'), max_patches=350, cached_only=True)
            self.assertEqual(result['files_scanned'], 1)
            self.assertEqual(result['stop_reason'], 'cached_checkpoints_exhausted')
            self.assertEqual(result['eligible_physical_parcels']['potato'], 1)
            self.assertEqual(json.loads((Path(result['field_root']).parent/'last_scan_transfer.json').read_text())['network_bytes'], 0)

    def test_patch_order_is_stable_and_interleaves_strata(self):
        records = [{'filename': f'data/{year}/{tile}/{i}.nc', 'year': year, 'tile': tile}
                   for year, tile in [(2019, 'A'), (2020, 'B'), (2020, 'C')] for i in range(3)]
        ordered = list(bounded_references.ordered_patches(records))
        self.assertEqual(ordered, list(bounded_references.ordered_patches(list(reversed(records)))))
        self.assertEqual(len({(r['year'], r['tile']) for r in ordered[:3]}), 3)
