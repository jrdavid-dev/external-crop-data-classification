"""Land-cover guidance retains uncertainty and never filters upstream crop inputs."""
from contextlib import chdir, redirect_stdout
from datetime import date, timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio

from crop_pipeline import progress
from crop_pipeline.analysis_workflow import resolve_stage_pointer, write_stage_pointer
from crop_pipeline.paoay import land_cover as lc, dynamic_world as dw, imagery
import test_paoay_dynamic_world as fixtures

REPOSITORY = Path(__file__).resolve().parents[1]


def two_dates(retriever):
    rows = fixtures.world_acquisitions(retriever)
    extra = []
    for row in rows:
        day = (date.fromisoformat(row['date']) + timedelta(days=1)).isoformat()
        extra.append({**row, 'date': day, 'image_index': row['image_index'] + '_b', 'image_id': row['image_id'] + '_b'})
    return rows + extra


def patterned_fetch(retriever, interval, granules, path):
    grid = retriever.request['settings']['grid']
    fixtures.write_world(path, grid)
    with rasterio.open(path, 'r+') as raster:
        data = raster.read()
        stripes = np.indices((grid['height'], grid['width']))[1] % 8
        data[-1] = 2
        for stripe, code in [(0, 0), (1, 4), (2, 7), (3, 6), (4, 4), (7, 4 if interval['start'][5:7] == '07' else 1)]:
            data[:-1, stripes == stripe] = .025
            data[code, stripes == stripe] = .8
        data[-1, stripes == 4] = 1
        data[:-1, stripes == 5] = -9999
        data[-1, stripes == 5] = 0
        raster.write(data)


class GuidanceRuleTests(unittest.TestCase):
    def test_all_states_missing_water_bare_built_and_outside_are_distinct(self):
        data = np.full((10, 1, 9), 1/9, dtype='float32')
        data[-1] = 2
        for index, code in [(0, 0), (1, 4), (2, 7), (3, 6), (4, 4)]:
            data[:-1, 0, index] = .025
            data[code, 0, index] = .8
        data[-1, 0, 4] = 1
        data[:-1, 0, 5], data[-1, 0, 5] = -9999, 0
        data[:-1, 0, 7] = 0
        data[0:2, 0, 7] = .5
        roi = np.ones((1, 9), dtype=bool); roi[0, -1] = False
        result = lc.classify(data, roi, lc.DEFAULT_RULES)
        np.testing.assert_array_equal(result[5, 0], [4, 3, 4, 4, 1, 0, 2, 2, -9999])
        self.assertEqual(result[0, 0, 0], 0)  # Water is class zero, not missing evidence.
        self.assertEqual(result[0, 0, 5], -9999)
        self.assertEqual(result[0, 0, 2], 7)  # Bare retained without inventing a fallow class.
        self.assertEqual(result[0, 0, 3], 6)
        self.assertTrue((result[:, 0, -1] == -9999).all())

    def test_boundaries_and_ties_are_explicit_even_with_zero_margin_rule(self):
        data = np.full((10, 1, 2), .05, dtype='float64')
        data[-1] = 2; data[4, 0, 0] = .6
        data[:-1, 0, 1] = 0; data[:2, 0, 1] = .5
        result = lc.classify(data, np.ones((1, 2), dtype=bool), lc.DEFAULT_RULES)
        self.assertEqual(result[5, 0, 0], 3)
        rules = {**lc.DEFAULT_RULES, 'minimum_top_probability': .5, 'minimum_margin': 0}
        self.assertEqual(lc.classify(data, np.ones((1, 2), dtype=bool), rules)[5, 0, 1], 2)

    def test_completely_unobserved_bin_stays_unknown_without_inventing_a_class(self):
        data = np.full((10, 2, 2), -9999, dtype='float32'); data[-1] = 0
        roi = np.ones((2, 2), dtype=bool)
        result = lc.classify(data, roi, lc.DEFAULT_RULES)
        self.assertTrue((result[:4] == -9999).all())
        self.assertTrue((result[4:] == 0).all())

    def test_invalid_rules_and_probability_support_are_rejected(self):
        for key, values in [('minimum_dates', [True, 0, 1.5, 17]), ('minimum_top_probability', [np.nan, -1, 2, True]),
                            ('minimum_margin', [float('inf'), -1, False])]:
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    lc.validate_rules({**lc.DEFAULT_RULES, key: value})
        with self.assertRaises(ValueError):
            lc.validate_rules({'minimum_dates': 2})
        data = np.full((10, 1, 1), .2); data[-1] = 2
        with self.assertRaisesRegex(ValueError, 'support'):
            lc.classify(data, np.ones((1, 1), dtype=bool), lc.DEFAULT_RULES)


class PaoayLandCoverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        reference = fixtures.reference_fixture(self.root / 'inputs')
        image_request = imagery.prepare_request(reference, '2025-07-01', self.root / 'sentinel2')
        with patch.object(imagery.PaoayRetriever, 'list_acquisitions', fixtures.acquisitions), \
             patch.object(imagery.PaoayRetriever, 'find_cloud_probability_matches', lambda retriever, fragment: {row['image_index'] for row in fixtures.acquisitions(retriever, fragment)}), \
             patch.object(imagery.PaoayRetriever, 'fetch_composite', fixtures.fake_fetch):
            self.imagery_handoff = imagery.collect(image_request, 'download_missing', object())
        world_request = dw.prepare_request(self.imagery_handoff, self.root / 'dynamic_world')
        with patch.object(dw.DynamicWorldRetriever, 'list_acquisitions', two_dates), \
             patch.object(dw.DynamicWorldRetriever, 'fetch_composite', patterned_fetch):
            self.world_handoff = dw.collect(world_request, 'download_missing', object())
        self.upstream = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.request = lc.prepare_request(self.world_handoff, self.root / 'land_cover')

    def test_build_and_rerun_are_local_immutable_and_leave_all_crop_inputs_untouched(self):
        with patch('socket.socket.connect', side_effect=AssertionError('No network')):
            handoff = lc.build(self.request)
            bundle = lc.load_land_cover_bundle(handoff)
            before = {p: p.read_bytes() for p in bundle['folder'].rglob('*') if p.is_file()}
            self.assertEqual(lc.build(self.request), handoff)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertEqual({p: p.read_bytes() for p in self.upstream}, self.upstream)
        self.assertEqual(bundle['manifest']['downstream_policy'], 'guidance_only_preserve_all_valid_crop_model_inputs')
        self.assertEqual(len(bundle['rasters']), 8)
        source = dw.load_dynamic_world_bundle(self.world_handoff)['manifest']
        self.assertEqual(bundle['manifest']['grid'], source['grid'])
        self.assertEqual(bundle['manifest']['period'], source['period'])
        for path in bundle['rasters']:
            with rasterio.open(path) as raster:
                self.assertEqual(list(raster.descriptions), lc.BANDS)
                self.assertEqual(raster.nodata, -9999)

    def test_area_accounting_includes_unknowns_once_per_bin_and_preserves_rotations(self):
        bundle = lc.load_land_cover_bundle(lc.build(self.request))
        roi_count = bundle['manifest']['roi_pixels']
        for row in bundle['coverage']:
            self.assertEqual(sum(int(row[label]) for label in lc.STATES.values()), roi_count)
        for interval in bundle['manifest']['period']['intervals']:
            rows = [row for row in bundle['class_areas'] if row['interval'] == interval['name']]
            self.assertEqual(sum(int(row['pixels']) for row in rows), roi_count)
            self.assertAlmostEqual(sum(float(row['area_ha']) for row in rows), roi_count * .01)
            self.assertAlmostEqual(sum(float(row['roi_percent']) for row in rows), 100)
        categories = {row['category'] for row in bundle['class_areas']}
        self.assertIn('bare', categories); self.assertNotIn('fallow', categories)
        with rasterio.open(bundle['rasters'][0]) as first, rasterio.open(bundle['rasters'][-1]) as last:
            self.assertTrue(((first.read(1) == 4) & (last.read(1) == 1)).any())

    def test_threshold_changes_create_separate_results_without_replacing_prior_files(self):
        first = lc.build(self.request)
        before = first.read_bytes()
        strict = lc.prepare_request(self.world_handoff, self.root / 'land_cover', {**lc.DEFAULT_RULES, 'minimum_top_probability': .9})
        second = lc.build(strict)
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), before)
        bundle = lc.load_land_cover_bundle(second)
        self.assertTrue(all(int(row['Crop support']) == 0 and int(row['Other-cover support']) == 0 for row in bundle['coverage']))

    def test_failed_build_publishes_nothing_and_preserves_prior_result(self):
        first = lc.build(self.request)
        strict = lc.prepare_request(self.world_handoff, self.root / 'land_cover', {**lc.DEFAULT_RULES, 'minimum_top_probability': .9})
        original = lc.classify
        calls = []
        def interrupted(*args):
            calls.append(1)
            if len(calls) == 3:
                raise OSError('Interrupted local build')
            return original(*args)
        with patch.object(lc, 'classify', interrupted), self.assertRaises(OSError):
            lc.build(strict)
        self.assertFalse(strict['folder'].exists())
        lc.load_land_cover_bundle(first)
        lc.load_land_cover_bundle(lc.build(strict))

    def test_changed_output_dependency_and_incomplete_inventory_are_blocked(self):
        handoff = lc.build(self.request)
        bundle = lc.load_land_cover_bundle(handoff)
        original = handoff.read_text()
        manifest = json.loads(original)
        manifest['files'].pop('coverage.csv')
        handoff.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'exactly eight'):
            lc.load_land_cover_bundle(handoff)
        handoff.write_text(original)
        raster = bundle['rasters'][0]; raster.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            lc.build(self.request)
        self.assertEqual(raster.read_bytes(), b'changed')
        self.world_handoff.write_text(self.world_handoff.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'handoff changed'):
            lc.build(self.request)

    def test_preview_has_all_eight_periods_and_explicit_uncertainty_legend(self):
        import matplotlib.pyplot as plt
        bundle = lc.load_land_cover_bundle(lc.build(self.request))
        figure = lc.plot_guidance(bundle)
        self.assertEqual(len(figure.axes), 8)
        self.assertEqual(figure.axes[0].get_title(), "Jul 1–15, 2025")
        self.assertEqual(len(figure.legends[0].get_texts()), 5)
        figure.savefig(self.root / 'preview.png')
        plt.close(figure)

    def test_notebook_run_all_and_pending_previous_stage_are_local(self):
        root = self.root
        (root / 'pyproject.toml').write_text('[project]\nname="fixture"\n')
        (root / 'src/crop_pipeline').mkdir(parents=True)
        paoay = root / 'sen4agrinet_work/paoay'
        source_pointer = paoay / 'dynamic_world/dynamic_world_stage.json'
        write_stage_pointer(source_pointer, self.world_handoff)
        notebook = json.loads((REPOSITORY / 'notebooks/paoay/archive/eight_notebook_workflow/10_paoay_land_cover.ipynb').read_text())
        def run():
            namespace = {}
            try:
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        exec(compile(''.join(cell['source']), 'paoay-land-cover-notebook', 'exec'), namespace)
                        namespace['show_preview'] = False
            finally:
                progress.finish_notebook()
            return namespace
        with chdir(root), redirect_stdout(io.StringIO()), \
             patch('socket.socket.connect', side_effect=AssertionError('No network')), \
             patch('subprocess.check_call', side_effect=AssertionError('No installation')):
            namespace = run()
            self.assertEqual(resolve_stage_pointer(namespace['land_cover_pointer']), namespace['land_cover_handoff'])
            write_stage_pointer(source_pointer, status='pending', reason='Notebook 09 incomplete')
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                run()
        with self.assertRaisesRegex(RuntimeError, 'pending'):
            resolve_stage_pointer(namespace['land_cover_pointer'])


if __name__ == '__main__':
    unittest.main()
