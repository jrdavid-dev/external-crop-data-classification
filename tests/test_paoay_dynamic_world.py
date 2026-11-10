"""Dynamic World inputs preserve upstream data, masks and probability semantics."""
from contextlib import chdir, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from affine import Affine
import numpy as np
import rasterio

from crop_pipeline import progress
from crop_pipeline.analysis_workflow import resolve_stage_pointer, write_stage_pointer
from crop_pipeline.paoay import dynamic_world as dw, imagery
from test_paoay_imagery import reference_fixture, acquisitions, fake_fetch

REPOSITORY = Path(__file__).resolve().parents[1]


def write_world(path, grid, empty=False):
    data = np.full((10, grid['height'], grid['width']), 1/9, dtype='float32')
    data[-1] = 1
    if empty:
        data[:-1], data[-1] = dw.NODATA, 0
    with rasterio.open(path, 'w', driver='GTiff', width=grid['width'], height=grid['height'], count=10,
                       dtype='float32', crs=grid['crs'], transform=Affine(*grid['transform']), nodata=dw.NODATA) as raster:
        raster.write(data)
        raster.descriptions = tuple(dw.BANDS)


def world_acquisitions(retriever):
    return [{'image_index': 'scene_' + row['start'], 'date': row['start'],
             'image_id': dw.COLLECTION + '/scene_' + row['start'],
             'dynamicworld_algorithm_version': 'fixture-model', 'qa_algorithm_version': 'fixture-mask'}
            for row in retriever.request['period']['intervals']]


def world_fetch(retriever, interval, granules, path):
    write_world(path, retriever.request['settings']['grid'], empty=not bool(granules))


class PaoayDynamicWorldTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = reference_fixture(self.root / 'inputs')
        request = imagery.prepare_request(self.reference, '2025-07-01', self.root / 'sentinel2')
        with patch.object(imagery.PaoayRetriever, 'list_acquisitions', acquisitions), \
             patch.object(imagery.PaoayRetriever, 'find_cloud_probability_matches', lambda retriever, fragment: {row['image_index'] for row in acquisitions(retriever, fragment)}), \
             patch.object(imagery.PaoayRetriever, 'fetch_composite', fake_fetch):
            self.imagery_handoff = imagery.collect(request, 'download_missing', object())
        self.upstream = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.request = dw.prepare_request(self.imagery_handoff, self.root / 'dynamic_world')

    def download(self, request=None, fetch=world_fetch, inventory=world_acquisitions):
        with patch.object(dw.DynamicWorldRetriever, 'list_acquisitions', inventory), \
             patch.object(dw.DynamicWorldRetriever, 'fetch_composite', fetch):
            return dw.collect(request or self.request, 'download_missing', object())

    def test_plan_is_local_and_inherits_exact_dates_grid_and_nine_probabilities(self):
        with patch('socket.socket.connect', side_effect=AssertionError('No network')):
            source = imagery.load_imagery_bundle(self.imagery_handoff)['manifest']
            self.assertEqual(self.request['period'], source['period'])
            self.assertEqual(self.request['settings']['grid'], source['grid'])
            self.assertEqual(len(dw.cache_status(self.request)), 8)
        self.assertFalse(self.request['folder'].exists())
        self.assertEqual(dw.BANDS, dw.PROBABILITIES + ['observation_count'])
        self.assertNotIn('label', dw.BANDS)
        self.assertIsNone(self.request['settings']['probability_threshold'])

    def test_reuse_has_no_network_and_does_not_create_download_files(self):
        with patch('socket.socket.connect', side_effect=AssertionError('No network')):
            with self.assertRaisesRegex(FileNotFoundError, '8 of 8'):
                dw.collect(self.request)
        self.assertFalse(self.request['folder'].exists())
        with self.assertRaisesRegex(ValueError, 'initialized'):
            dw.collect(self.request, 'download_missing')

    def test_download_then_rerun_preserves_all_bytes_and_attribution(self):
        handoff = self.download()
        bundle = dw.load_dynamic_world_bundle(handoff)
        before = {p: p.read_bytes() for p in self.request['folder'].rglob('*') if p.is_file()}
        with patch('socket.socket.connect', side_effect=AssertionError('No network')), \
             patch.object(dw.DynamicWorldRetriever, 'fetch_composite', side_effect=AssertionError('No download')):
            self.assertEqual(dw.collect(self.request), handoff)
        self.assertEqual({p: p.read_bytes() for p in self.upstream}, self.upstream)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertEqual(len(bundle['rasters']), 8)
        self.assertEqual(bundle['manifest']['summary']['empty_bins'], 0)
        self.assertEqual(bundle['manifest']['license'], 'CC-BY-4.0')
        self.assertIn('World Resources Institute', bundle['manifest']['attribution'])
        self.assertEqual(bundle['manifest']['summary']['any_bin_observed_pixels'], bundle['manifest']['summary']['roi_pixels'])

    def test_interrupted_transfer_resumes_only_missing_pairs(self):
        calls = []
        def interrupted(retriever, interval, granules, path):
            calls.append(interval['name'])
            if len(calls) == 3:
                path.write_bytes(b'partial')
                raise OSError('Interrupted')
            world_fetch(retriever, interval, granules, path)
        with self.assertRaises(OSError):
            self.download(fetch=interrupted)
        self.assertEqual(sum(row['status'] == 'verified local' for row in dw.cache_status(self.request)), 2)
        before = {p: p.read_bytes() for p in self.request['folder'].rglob('*') if p.is_file()}
        calls.clear()
        def resumed(retriever, interval, granules, path):
            calls.append(interval['name'])
            world_fetch(retriever, interval, granules, path)
        with patch.object(dw.DynamicWorldRetriever, 'list_acquisitions', side_effect=AssertionError('Reuse metadata')), \
             patch.object(dw.DynamicWorldRetriever, 'fetch_composite', resumed):
            dw.collect(self.request, 'download_missing', object())
        self.assertEqual(len(calls), 6)
        self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_empty_inventory_stays_nodata_without_substituting_other_dates(self):
        bundle = dw.load_dynamic_world_bundle(self.download(inventory=lambda retriever: []))
        self.assertEqual(bundle['manifest']['summary']['empty_bins'], 8)
        self.assertEqual(bundle['manifest']['summary']['any_bin_observed_pixels'], 0)
        with rasterio.open(bundle['rasters'][0]) as raster:
            self.assertTrue((raster.read(1) == dw.NODATA).all())
            self.assertTrue((raster.read(10) == 0).all())

    def test_full_raster_validation_rejects_bad_probabilities_counts_and_alignment(self):
        item = self.request['period']['intervals'][0]
        path = self.root / 'invalid.tif'
        for case in ('range', 'sum', 'count', 'missing', 'grid', 'nan'):
            with self.subTest(case=case):
                write_world(path, self.request['settings']['grid'])
                with rasterio.open(path, 'r+') as raster:
                    if case == 'grid':
                        raster.transform = raster.transform * Affine.translation(1, 0)
                    else:
                        band_number = 10 if case in ('count', 'missing') else 9
                        band = raster.read(band_number)
                        band[-1, -1] = {'range': 2, 'sum': .2, 'count': .5, 'missing': 0, 'nan': np.nan}.get(case, 0)
                        raster.write(band, band_number)
                with self.assertRaises(ValueError):
                    dw.inspect_raster(path, self.request, item, [item['start']])

    def test_corrupt_or_unpaired_cached_files_are_never_replaced(self):
        handoff = self.download()
        raster = dw.load_dynamic_world_bundle(handoff)['rasters'][0]
        raster.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            dw.collect(self.request, 'download_missing', object())
        self.assertEqual(raster.read_bytes(), b'changed')
        other = dw.prepare_request(self.imagery_handoff, self.root / 'other')
        path, receipt = dw._paths(other, other['period']['intervals'][0])
        path.parent.mkdir(parents=True)
        receipt.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            dw.collect(other, 'download_missing', object())
        self.assertEqual(receipt.read_text(), '{}')

    def test_upstream_change_and_tampered_inventory_block_resume(self):
        def interrupted(*args):
            raise OSError('Interrupted')
        with self.assertRaises(OSError):
            self.download(fetch=interrupted)
        inventory = self.request['folder'] / 'acquisitions.json'
        payload = json.loads(inventory.read_text()); payload['acquisitions'][0]['date'] = '2024-01-01'
        inventory.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            dw.collect(self.request, 'download_missing', object())
        self.imagery_handoff.write_text(self.imagery_handoff.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'handoff changed'):
            dw.collect(self.request)

    def test_manifest_incomplete_inventory_or_different_grid_blocks_downstream(self):
        handoff = self.download()
        original = json.loads(handoff.read_text())
        changed = deepcopy(original)
        changed['files'].pop(next(iter(changed['files'])))
        handoff.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'exactly eight'):
            dw.load_dynamic_world_bundle(handoff)
        changed = deepcopy(original)
        changed['identity']['settings']['grid']['width'] += 1
        changed['grid'] = changed['identity']['settings']['grid']
        changed['request_id'] = imagery._digest(changed['identity'])[:20]
        handoff.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'dates/grid differ'):
            dw.load_dynamic_world_bundle(handoff)

    def test_direct_download_uses_upstream_grid_and_writes_labelled_probability_bands(self):
        interval = self.request['period']['intervals'][0]
        source = self.root / 'http-source.tif'
        write_world(source, self.request['settings']['grid'])
        payload = source.read_bytes()
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [payload[:100], payload[100:]]
        composite = MagicMock()
        composite.getDownloadURL.return_value = 'https://example.invalid/fixture'
        retriever = dw.DynamicWorldRetriever(None, self.request)
        output = self.root / 'http-result.tif'
        with patch.object(retriever, 'make_composite', return_value=composite), \
             patch.object(dw.requests, 'get', return_value=response):
            retriever.fetch_composite(interval, [], output)
        parameters = composite.getDownloadURL.call_args.args[0]
        grid = self.request['settings']['grid']
        self.assertEqual(parameters['crs'], grid['crs'])
        self.assertEqual(parameters['crs_transform'], grid['transform'])
        self.assertEqual(parameters['dimensions'], [grid['width'], grid['height']])
        self.assertFalse(parameters['filePerBand'])
        self.assertGreater(dw.inspect_raster(output, self.request, interval)['observed_roi_pixels'], 0)

    def test_bounds_and_metadata_validation_precede_downloads(self):
        bundle = imagery.load_imagery_bundle(self.imagery_handoff)
        bundle['manifest']['grid'] = {**bundle['manifest']['grid'], 'width': 2000, 'height': 2000}
        with patch.object(imagery, 'load_imagery_bundle', return_value=bundle):
            with self.assertRaisesRegex(ValueError, 'tiling'):
                dw.prepare_request(self.imagery_handoff, self.root / 'huge')
        with self.assertRaisesRegex(ValueError, 'metadata'):
            self.download(inventory=lambda retriever: world_acquisitions(retriever) * 2)
        self.assertFalse(self.request['folder'].exists())

    def test_notebook_run_all_uses_verified_cache_and_missing_previous_stage_leaves_pending(self):
        root = self.root
        (root / 'pyproject.toml').write_text('[project]\nname="fixture"\n')
        (root / 'src/crop_pipeline').mkdir(parents=True)
        paoay = root / 'sen4agrinet_work/paoay'
        source_pointer = paoay / 'sentinel2/imagery_stage.json'
        write_stage_pointer(source_pointer, self.imagery_handoff)
        request = dw.prepare_request(self.imagery_handoff, paoay / 'dynamic_world')
        self.download(request)
        notebook = json.loads((REPOSITORY / 'notebooks/paoay/archive/eight_notebook_workflow/09_paoay_dynamic_world.ipynb').read_text())
        def run():
            namespace = {}
            try:
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        exec(compile(''.join(cell['source']), 'paoay-dynamic-world-notebook', 'exec'), namespace)
            finally:
                progress.finish_notebook()
            return namespace
        with chdir(root), redirect_stdout(io.StringIO()), \
             patch('socket.socket.connect', side_effect=AssertionError('No network')), \
             patch('subprocess.check_call', side_effect=AssertionError('No installation')):
            namespace = run()
            self.assertEqual(resolve_stage_pointer(namespace['dynamic_world_pointer']), namespace['dynamic_world_handoff'])
            write_stage_pointer(source_pointer, status='pending', reason='Notebook 08 incomplete')
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                run()
        with self.assertRaisesRegex(RuntimeError, 'pending'):
            resolve_stage_pointer(namespace['dynamic_world_pointer'])


class ArrayImage:
    """Tiny masked-array EE fixture to test the actual compositing arithmetic."""
    def __init__(self, data, bands=dw.PROBABILITIES):
        self.data, self.bands = np.ma.asarray(data), list(bands)
    def select(self, bands):
        bands = [bands] if isinstance(bands, str) else bands
        return ArrayImage(self.data[[self.bands.index(band) for band in bands]], bands)
    def mask(self):
        return ArrayImage((~np.ma.getmaskarray(self.data)).astype(float), self.bands)
    def reduce(self, reducer):
        return ArrayImage(np.ma.min(self.data, axis=0, keepdims=True), ['mask'])
    def updateMask(self, mask):
        return ArrayImage(np.ma.array(self.data, mask=np.ma.getmaskarray(self.data) | np.broadcast_to(mask.data == 0, self.data.shape)), self.bands)
    def reproject(self, **kwargs):
        return self
    def unmask(self, value, **kwargs):
        return ArrayImage(self.data.filled(value), self.bands)
    def rename(self, bands):
        return ArrayImage(self.data, [bands] if isinstance(bands, str) else bands)
    def addBands(self, other):
        return ArrayImage(np.ma.concatenate([self.data, other.data]), self.bands + other.bands)
    def toFloat(self):
        return self


class ArrayCollection:
    def __init__(self, images):
        self.images = images
    def mosaic(self):
        result = self.images[0].data.copy()
        for image in self.images[1:]:
            result = np.ma.where(np.ma.getmaskarray(image.data), result, image.data)
        return ArrayImage(result)
    def mean(self):
        return ArrayImage(np.ma.mean(np.ma.stack([image.data for image in self.images]), axis=0))
    def select(self, band):
        return ArrayCollection([image.select(band) for image in self.images])
    def count(self):
        return ArrayImage(np.ma.count(np.ma.stack([image.data for image in self.images]), axis=0), ['count'])


class DynamicWorldCompositeTests(unittest.TestCase):
    def test_no_acquisitions_make_explicit_nodata_probabilities_and_zero_counts(self):
        factory = SimpleNamespace(constant=lambda values: ArrayImage(np.asarray(values)[:, None, None], dw.BANDS))
        ee = SimpleNamespace(Image=factory)
        request = {'settings': {'grid': {}}}
        result = dw.DynamicWorldRetriever(ee, request).make_composite([])
        self.assertEqual(result.bands, dw.BANDS)
        np.testing.assert_array_equal(result.data[:-1], np.full((9, 1, 1), dw.NODATA))
        self.assertEqual(result.data[-1, 0, 0], 0)

    def test_same_date_tiles_count_once_and_means_keep_probability_vectors(self):
        def image(a, b, mask=False):
            values = np.array([a, b] + [0] * 7, dtype=float)[:, None, None]
            masks = np.zeros_like(values, dtype=bool)
            masks[-1] = mask
            return ArrayImage(np.ma.array(values, mask=masks))
        images = {'a': image(1, 0), 'b': image(0, 1), 'c': image(1, 0), 'masked': image(.5, .5, True)}
        factory = lambda key: images[key]
        ee = SimpleNamespace(Image=factory, ImageCollection=SimpleNamespace(fromImages=ArrayCollection),
                             Reducer=SimpleNamespace(min=lambda: 'min'))
        request = {'settings': {'grid': {'crs': 'EPSG:32651', 'transform': [10, 0, 0, 0, -10, 0]}}}
        granules = [{'image_id': key, 'image_index': key, 'date': day} for key, day in
                    [('a', '2025-07-01'), ('b', '2025-07-01'), ('c', '2025-07-02'), ('masked', '2025-07-03')]]
        result = dw.DynamicWorldRetriever(ee, request).make_composite(granules).data[:, 0, 0]
        np.testing.assert_allclose(result[:2], [.5, .5])
        self.assertEqual(result[-1], 2)
        self.assertAlmostEqual(result[:-1].sum(), 1)


if __name__ == '__main__':
    unittest.main()
