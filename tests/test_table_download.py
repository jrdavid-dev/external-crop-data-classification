import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from crop_pipeline.imagery import table_download, batch_export


def fake_ee(responses):
    state = SimpleNamespace(max_retries=5)
    data = SimpleNamespace(_get_state=lambda: state,
                           setMaxRetries=lambda value: setattr(state, 'max_retries', value),
                           setDeadline=Mock(), computeFeatures=Mock(side_effect=responses),
                           deleteAsset=Mock())
    return SimpleNamespace(data=data, FeatureCollection=lambda asset: asset)


def page(value, token=None):
    result = {'features': [{'properties': {'value': value, 'missing': None}}]}
    if token:
        result['nextPageToken'] = token
    return result


class TableDownloadTests(unittest.TestCase):
    def test_resume_and_complete_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            first = fake_ee([page(1.23456789, 'next'), requests.Timeout('timed out')])
            with self.assertRaises(RuntimeError):
                table_download.download(first, 'asset', directory, retries=0)
            self.assertEqual(first.data._get_state().max_retries, 5)
            self.assertFalse(json.loads((Path(directory)/'manifest.json').read_text())['complete'])
            second = fake_ee([page(2.5)])
            frame = table_download.download(second, 'asset', directory)
            self.assertEqual(frame.value.tolist(), [1.23456789, 2.5])
            self.assertTrue(frame.missing.isna().all())
            self.assertEqual(second.data.computeFeatures.call_args.args[0],
                             {'expression': 'asset', 'pageSize': 5000, 'pageToken': 'next'})
            second.data.setDeadline.assert_called_once_with(120000)
            cached = fake_ee([])
            self.assertTrue(table_download.download(cached, 'asset', directory).equals(frame))
            cached.data.computeFeatures.assert_not_called()
            (Path(directory)/'page_00001.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                table_download.download(cached, 'asset', directory)

    @patch('crop_pipeline.imagery.table_download.time.sleep')
    def test_visible_retry(self, sleep):
        ee = fake_ee([requests.Timeout(), page(3)])
        self.assertEqual(table_download.download(ee, 'asset').value.tolist(), [3])
        self.assertEqual(ee.data.computeFeatures.call_count, 2)
        sleep.assert_called_once_with(2)
        self.assertEqual(ee.data._get_state().max_retries, 5)

    def test_nontransient_and_repeated_token(self):
        ee = fake_ee([ValueError('permission denied')])
        with self.assertRaises(RuntimeError):
            table_download.download(ee, 'asset')
        self.assertEqual(ee.data.computeFeatures.call_count, 1)
        with tempfile.TemporaryDirectory() as directory:
            ee = fake_ee([page(1, 'same'), page(2, 'same')])
            with self.assertRaisesRegex(ValueError, 'repeated'):
                table_download.download(ee, 'asset', directory)
            manifest = json.loads((Path(directory)/'manifest.json').read_text())
            self.assertEqual(len(manifest['pages']), 1)

    def test_collect_failure_preserves_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory)/'ledger.json'
            entry = {'kind': 'pixel_table', 'task_id': 'task', 'asset_id': 'asset', 'year': '2022'}
            batch_export.save_ledger(ledger, [entry])
            ee = fake_ee([])
            with patch.object(batch_export, 'task_state', return_value=('SUCCEEDED', 0)), \
                 patch.object(batch_export, 'read_table', side_effect=RuntimeError('timeout')):
                with self.assertRaises(RuntimeError):
                    batch_export.collect(ee, 'project', ledger, {'pixel_table': Mock()})
            ee.data.deleteAsset.assert_not_called()
            self.assertFalse(batch_export.load_ledger(ledger)[0].get('collected', False))
