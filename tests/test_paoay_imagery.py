"""Local Paoay imagery contracts, resume behavior and notebook execution."""
from contextlib import chdir, redirect_stdout
from datetime import date
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from affine import Affine
import numpy as np
import rasterio

from crop_pipeline import progress
from crop_pipeline.analysis_workflow import write_stage_pointer, resolve_stage_pointer
from crop_pipeline.imagery.retrieval import SatelliteRetriever, RESAMPLING_POLICY
from crop_pipeline.paoay import imagery, references

REPOSITORY = Path(__file__).resolve().parents[1]


def reference_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    kmz = root / 'roi.kmz'
    with zipfile.ZipFile(kmz, 'w') as archive:
        archive.writestr('doc.kml', '''<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><name>Region of Interest</name><Polygon><outerBoundaryIs><LinearRing><coordinates>120.70,16.60 120.703,16.60 120.703,16.603 120.70,16.603 120.70,16.60</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Document></kml>''')
    records, sources = references.read_inputs([kmz])
    return references.export_reference_bundle(records, sources, root / 'references')


def write_raster(path, grid, empty=False):
    data = np.full((7, grid['height'], grid['width']), 1500, dtype='float32')
    data[-1] = 1
    if empty:
        data[:-1] = -9999
        data[-1] = 0
    with rasterio.open(path, 'w', driver='GTiff', width=grid['width'], height=grid['height'], count=7,
                       dtype='float32', crs=grid['crs'], transform=Affine(*grid['transform']), nodata=-9999) as raster:
        raster.write(data)
        raster.descriptions = tuple(imagery.BANDS)


def acquisitions(retriever, fragment):
    return [{'image_index': 'scene_' + interval['start'], 'date': interval['start'],
             'image_id': retriever.collection_name + '/scene_' + interval['start'], 'timestamp_ms': 1}
            for year in (2024, 2025, 2026) for interval in imagery.interval_bins(year)
            if retriever.start_date <= interval['start'] < retriever.end_date]


def fake_fetch(retriever, plan, interval, granules, path):
    write_raster(path, plan['settings']['grid'])


class PaoayImageryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = reference_fixture(self.root / 'inputs')
        self.request = imagery.prepare_request(self.reference, '2025-07-01', self.root / 'sentinel2')

    def download(self, request=None, fetch=fake_fetch):
        request = request or self.request
        with patch.object(imagery.PaoayRetriever, 'list_acquisitions', acquisitions), \
             patch.object(imagery.PaoayRetriever, 'find_cloud_probability_matches', lambda retriever, fragment: {row['image_index'] for row in acquisitions(retriever, fragment)}), \
             patch.object(imagery.PaoayRetriever, 'fetch_composite', fetch):
            return imagery.collect(request, 'download_missing', earth_engine=object())

    def test_calendar_crosses_year_and_rejects_partial_or_future_periods(self):
        period = imagery.four_month_period('2025-11-01', date(2026, 3, 1))
        self.assertEqual(period['end'], '2026-03-01')
        self.assertEqual(len(period['intervals']), 8)
        self.assertEqual(period['intervals'][4]['name'], '2026_01_01_15')
        for start in ('2025-07-02', '2025-7-01', None, '2016-01-01', '2026-11-01'):
            with self.subTest(start=start), self.assertRaises(ValueError):
                imagery.four_month_period(start, date(2026, 11, 10))

    def test_plan_is_local_and_preserves_the_shared_training_recipe(self):
        with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
            request = imagery.prepare_request(self.reference, '2025-07-01', self.root / 'sentinel2')
            self.assertEqual(imagery.cache_status(request)[0]['status'], 'missing')
        self.assertFalse(request['folder'].exists())
        self.assertIs(imagery.PaoayRetriever.mask_acquisition, SatelliteRetriever.mask_acquisition)
        self.assertIs(imagery.PaoayRetriever.make_composite, SatelliteRetriever.make_composite)
        settings = request['settings']
        self.assertEqual(settings['resampling'], RESAMPLING_POLICY)
        self.assertEqual(settings['cloud_probability_threshold'], 40)
        self.assertEqual(settings['bands'], imagery.BANDS)
        self.assertEqual(settings['grid']['transform'][:2], [10, 0])
        self.assertEqual(settings['start'], '2025-07-01')
        changed = imagery.prepare_request(self.reference, '2025-08-01', self.root / 'sentinel2')
        self.assertNotEqual(request['folder'], changed['folder'])

    def test_reuse_never_connects_or_writes_when_files_are_missing(self):
        with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
            with self.assertRaisesRegex(FileNotFoundError, '8 of 8'):
                imagery.collect(self.request)
        self.assertFalse(self.request['folder'].exists())
        with self.assertRaisesRegex(ValueError, 'initialized'):
            imagery.collect(self.request, 'download_missing')

    def test_download_and_local_rerun_preserve_inputs_and_every_cached_byte(self):
        original = self.reference.read_bytes()
        handoff = self.download()
        folder = self.request['folder']
        before = {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}
        with patch.object(imagery.PaoayRetriever, 'fetch_composite', side_effect=AssertionError('No download')), \
             patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
            self.assertEqual(imagery.collect(self.request), handoff)
            bundle = imagery.load_imagery_bundle(handoff)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertEqual(self.reference.read_bytes(), original)
        self.assertEqual(len(bundle['rasters']), 8)
        self.assertEqual(bundle['manifest']['summary']['all_eight_bins_clear_fraction'], 1)
        self.assertEqual(bundle['manifest']['spectral_scale_to_apply_later'], .0001)
        self.assertFalse(bundle['manifest']['clear_counts_scaled'])

    def test_interrupted_download_resumes_only_missing_without_replacing_completed_files(self):
        calls = []
        def fail_third(retriever, plan, interval, granules, path):
            calls.append(interval['name'])
            if len(calls) == 3:
                path.write_bytes(b'partial transfer')
                raise OSError('Interrupted transfer')
            fake_fetch(retriever, plan, interval, granules, path)
        with self.assertRaisesRegex(OSError, 'Interrupted'):
            self.download(fetch=fail_third)
        completed = [p for p in self.request['folder'].rglob('*') if p.is_file()]
        before = {p: p.read_bytes() for p in completed}
        self.assertEqual(sum(row['status'] == 'verified local' for row in imagery.cache_status(self.request)), 2)
        resumed = []
        def resume(retriever, plan, interval, granules, path):
            resumed.append(interval['name'])
            fake_fetch(retriever, plan, interval, granules, path)
        with patch.object(imagery.PaoayRetriever, 'list_acquisitions', side_effect=AssertionError('Reuse inventory')), \
             patch.object(imagery.PaoayRetriever, 'fetch_composite', resume):
            imagery.collect(self.request, 'download_missing', object())
        self.assertEqual(len(resumed), 6)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertFalse(list(self.request['folder'].rglob('*.part')))

    def test_corrupt_or_unpaired_files_are_preserved_and_block_reuse_or_download(self):
        handoff = self.download()
        raster = imagery.load_imagery_bundle(handoff)['rasters'][0]
        raster.write_bytes(b'changed')
        for mode in ('reuse', 'download_missing'):
            with self.assertRaisesRegex(ValueError, 'changed'):
                imagery.collect(self.request, mode, object())
        self.assertEqual(raster.read_bytes(), b'changed')
        other = imagery.prepare_request(self.reference, '2025-08-01', self.root / 'sentinel2')
        path, _ = imagery._paths(other, other['period']['intervals'][0])
        path.parent.mkdir(parents=True)
        path.write_bytes(b'unpaired file')
        with self.assertRaisesRegex(ValueError, 'Incomplete cached pair'):
            imagery.collect(other, 'download_missing', object())
        self.assertEqual(path.read_bytes(), b'unpaired file')

    def test_empty_bins_stay_nodata_and_are_reported_as_input_coverage_only(self):
        def empty_fetch(retriever, plan, interval, granules, path):
            write_raster(path, plan['settings']['grid'], empty=True)
        bundle = imagery.load_imagery_bundle(self.download(fetch=empty_fetch))
        summary = bundle['manifest']['summary']
        self.assertEqual(summary['empty_bins'], 8)
        self.assertEqual(summary['all_eight_bins_clear_pixels'], 0)
        self.assertIn('Not measured', summary['local_accuracy'])
        with rasterio.open(bundle['rasters'][0]) as raster:
            self.assertTrue((raster.read(1) == -9999).all())
            self.assertTrue((raster.read(7) == 0).all())

    def test_cross_year_download_and_full_payload_checks(self):
        request = imagery.prepare_request(self.reference, '2025-11-01', self.root / 'sentinel2')
        bundle = imagery.load_imagery_bundle(self.download(request))
        self.assertIn('2026_02_16_end', bundle['rasters'][-1].stem)
        raster = bundle['rasters'][0]
        with rasterio.open(raster, 'r+') as ds:
            counts = ds.read(7); counts[-1, -1] = 0.5; ds.write(counts, 7)
        with self.assertRaisesRegex(ValueError, 'daily observation counts'):
            imagery.inspect_raster(raster, request)

    def test_manifest_inventory_and_reference_dependencies_cannot_silently_change(self):
        handoff = self.download()
        original = handoff.read_text()
        manifest = json.loads(original)
        manifest['files'].pop(next(iter(manifest['files'])))
        handoff.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'exactly eight'):
            imagery.load_imagery_bundle(handoff)
        handoff.write_text(original)
        self.reference.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'dependency changed'):
            imagery.load_imagery_bundle(handoff)

    def test_grid_size_and_spectral_support_fail_before_publication(self):
        grid = {"crs": "EPSG:32651", "transform": [10, 0, 250000, 0, -10, 1850000],
                "width": 2000, "height": 2000}
        with patch.object(imagery.PaoayRetriever, 'make_download_grid', return_value=grid):
            with self.assertRaisesRegex(ValueError, 'tiling'):
                imagery.prepare_request(self.reference, '2025-07-01', self.root / 'huge')
        def inconsistent(retriever, plan, interval, granules, path):
            write_raster(path, plan['settings']['grid'])
            with rasterio.open(path, 'r+') as ds:
                band = ds.read(6); band[-1, -1] = -9999; ds.write(band, 6)
        with self.assertRaisesRegex(ValueError, 'support'):
            self.download(fetch=inconsistent)
        self.assertFalse(list(self.request['folder'].rglob('*.tif')))
        self.assertFalse((self.request['folder'] / 'imagery_handoff.json').exists())

    def test_changed_request_or_cached_inventory_blocks_resume(self):
        self.request['settings']['cloud_probability_threshold'] = 99
        with self.assertRaisesRegex(ValueError, 'changed after planning'):
            imagery.collect(self.request, 'download_missing', object())
        self.request = imagery.prepare_request(self.reference, '2025-07-01', self.root / 'sentinel2')
        def interrupted(*args):
            raise OSError('Interrupted')
        with self.assertRaises(OSError):
            self.download(fetch=interrupted)
        inventory = self.request['folder'] / 'acquisitions.json'
        saved = json.loads(inventory.read_text())
        saved['acquisitions'][0]['date'] = '2024-01-01'
        inventory.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            imagery.collect(self.request, 'download_missing', object())
        self.assertEqual(json.loads(inventory.read_text()), saved)

    def test_notebook_run_all_from_verified_local_cache_and_pending_on_missing(self):
        root = self.root
        (root / 'pyproject.toml').write_text('[project]\nname="fixture"\n')
        (root / 'src/crop_pipeline').mkdir(parents=True)
        paoay = root / 'sen4agrinet_work/paoay'
        write_stage_pointer(paoay / 'references/reference_stage.json', self.reference)
        request = imagery.prepare_request(self.reference, '2024-01-01', paoay / 'sentinel2')
        self.download(request)
        notebook = json.loads((REPOSITORY / 'notebooks/paoay/archive/eight_notebook_workflow/08_paoay_sentinel2.ipynb').read_text())
        namespace = {}
        with chdir(root), redirect_stdout(io.StringIO()), \
             patch('socket.socket.connect', side_effect=AssertionError('No network')), \
             patch('subprocess.check_call', side_effect=AssertionError('No installation')):
            try:
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        exec(compile(''.join(cell['source']), 'paoay-imagery-notebook', 'exec'), namespace)
            finally:
                progress.finish_notebook()
        self.assertEqual(resolve_stage_pointer(namespace['imagery_pointer']), namespace['imagery_handoff'])
        # A new missing period must not leave the previous successful stage selected.
        namespace = {}
        with chdir(root), redirect_stdout(io.StringIO()):
            try:
                with self.assertRaisesRegex(FileNotFoundError, 'missing'):
                    for cell in notebook['cells']:
                        if cell['cell_type'] == 'code':
                            exec(compile(''.join(cell['source']), 'paoay-imagery-notebook', 'exec'), namespace)
                            if 'period_start' in namespace and 'request' not in namespace:
                                namespace['period_start'] = '2025-08-01'
                                namespace['imagery_mode'] = 'reuse'  # Exercise the explicit local-only option.
            finally:
                progress.finish_notebook()
        with self.assertRaisesRegex(RuntimeError, 'pending'):
            resolve_stage_pointer(namespace['imagery_pointer'])


if __name__ == '__main__':
    unittest.main()
