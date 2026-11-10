"""Paoay features match the training recipe and preserve independent evidence/data."""
from contextlib import chdir, redirect_stdout
from copy import deepcopy
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio

from crop_pipeline import progress
from crop_pipeline.analysis_workflow import write_stage_pointer, resolve_stage_pointer
from crop_pipeline.features import experiments
from crop_pipeline.imagery.retrieval import REQUEST_VERSION, RESAMPLING_POLICY
from crop_pipeline.paoay import features, imagery, dynamic_world as dw, land_cover, references
from crop_pipeline.windows import seasons
import test_paoay_imagery as satellite
import test_paoay_dynamic_world as world

REPOSITORY = Path(__file__).resolve().parents[1]


def model_fixture(root, design='interval_and_shape'):
    folder = root / 'analysis/test/selected_model/fixture'
    folder.mkdir(parents=True)
    feature_handoff = root / 'training_feature_handoff.json'
    feature_handoff.write_text(json.dumps({'format_version': 1, 'kind': 'local_parcel_feature_experiments',
        'feature_policy': experiments.FEATURE_POLICY, 'feature_sets': experiments.feature_sets(),
        'identity': {'implementation_sha256': references.checksum(inspect.getfile(experiments))}}))
    # Deliberately not executable: feature preparation must never deserialize the model.
    (folder / 'model.joblib').write_bytes(b'fixture model bytes; not to be deserialized')
    recipe = {'status': 'unvalidated_external_parcel_prototype', 'analysis_mode': 'pilot',
        'selected_design': design, 'selection': {'status': 'selected_on_validation', 'selected_design': design},
        'feature_columns': experiments.feature_sets()[design], 'feature_policy': dict(experiments.FEATURE_POLICY),
        'window_convention': 'four_calendar_months_eight_half_month_bins_end_exclusive',
        'window_recipe': {'minimum_valid_fraction': .8, 'composites': 8, 'missing_values': 'exclude_no_imputation'},
        'quality': {'minimum_valid_fraction': .8, 'finite_features_required': True, 'missing_values': 'exclude_no_imputation'},
        'source_feature_manifest_sha256': references.checksum(feature_handoff),
        'imagery': {'collection': 'COPERNICUS/S2_SR_HARMONIZED', 'request_version': REQUEST_VERSION,
            'resampling': RESAMPLING_POLICY, 'spectral_scale': .0001, 'collection_bands': imagery.BANDS[:-1],
            'feature_bands': list(seasons.FEATURE_BANDS), 'bands': list(seasons.FEATURE_BANDS),
            'clear_counts_scaled': False, 'output_resolution_metres': 10, 'input_nodata_value': -9999,
            'cloud_probability_threshold': 40, 'cloud_probability_collection': 'COPERNICUS/S2_CLOUD_PROBABILITY',
            'excluded_scl_classes': [0, 1, 3, 8, 9, 10, 11], 'cloud_probability_rule': 'probability < threshold',
            'missing_cloud_probability': 'exclude', 'scene_edges': 'valid_B8A_and_B9',
            'compositing': 'same_date_mosaics_then_calendar_half_month_medians',
            'valid_data_rule': 'all six collection bands must be valid before compositing',
            'nodata_handling': 'mask nodata before reflectance scaling'}}
    (folder / 'recipe.json').write_text(json.dumps(recipe))
    manifest = {'format_version': 1, 'kind': 'paoay_inference_model_package', 'status': recipe['status'],
        'identity': {'model_sha256': references.checksum(folder / 'model.joblib'), 'recipe': recipe},
        'feature_handoff': str(feature_handoff),
        'files': {name: {'sha256': references.checksum(folder / name)} for name in ('model.joblib', 'recipe.json')}}
    (folder / 'model_manifest.json').write_text(json.dumps(manifest))
    return folder / 'model_manifest.json'


def spectral_fetch(retriever, plan, interval, granules, path):
    satellite.write_raster(path, plan['settings']['grid'])
    month = int(interval['start'][5:7])
    position = (month - 7) * 2 + (int(interval['start'][8:10]) == 16)
    near_infrared = [1000, 2000, 3500, 5000, 4500, 3500, 2500, 1500][position]
    with rasterio.open(path, 'r+') as raster:
        data = raster.read()
        for index, value in enumerate([900, 1000, 1300, near_infrared, 700, 600, 1]):
            data[index] = value
        raster.write(data)


def input_fixture(root, fetch=spectral_fetch):
    reference = satellite.reference_fixture(root / 'inputs')
    request = imagery.prepare_request(reference, '2025-07-01', root / 'sentinel2')
    with patch.object(imagery.PaoayRetriever, 'list_acquisitions', satellite.acquisitions), \
         patch.object(imagery.PaoayRetriever, 'find_cloud_probability_matches', lambda retriever, fragment: {row['image_index'] for row in satellite.acquisitions(retriever, fragment)}), \
         patch.object(imagery.PaoayRetriever, 'fetch_composite', fetch):
        image_handoff = imagery.collect(request, 'download_missing', object())
    request = dw.prepare_request(image_handoff, root / 'dynamic_world')
    with patch.object(dw.DynamicWorldRetriever, 'list_acquisitions', world.world_acquisitions), \
         patch.object(dw.DynamicWorldRetriever, 'fetch_composite', world.world_fetch):
        world_handoff = dw.collect(request, 'download_missing', object())
    guidance = land_cover.build(land_cover.prepare_request(world_handoff, root / 'land_cover'))
    return guidance


class PaoayFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.guidance = input_fixture(self.root)
        self.model = model_fixture(self.root)
        self.upstream = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.request = features.prepare_request(self.guidance, self.model, self.root / 'features')

    def test_unique_model_selection_requires_explicit_choice_if_ambiguous(self):
        self.assertEqual(features.find_model(self.root / 'analysis'), self.model)
        with self.assertRaises(FileNotFoundError):
            features.find_model(self.root / 'absent')
        model_fixture(self.root / 'extra')
        with self.assertRaisesRegex(ValueError, 'Multiple'):
            features.find_model(self.root)
        self.assertEqual(features.find_model(self.root, self.model), self.model)

    def test_exact_training_parity_scaling_shape_time_and_order_without_loading_model(self):
        with patch('joblib.load', side_effect=AssertionError('No executable model load')), \
             patch('socket.socket.connect', side_effect=AssertionError('No network')):
            handoff = features.build(self.request)
            bundle = features.load_feature_bundle(handoff)
        recipe, image = features._verify(self.request)
        roi, cube, coverage = features._coverage(image, recipe)
        row, column = int(bundle['rows'][0]), int(bundle['columns'][0])
        period, grid = image['manifest']['period'], image['manifest']['grid']
        base = {'field_id': 'paoay_roi', 'season_key': period['start']+'/'+period['end'],
                'window_start': period['start'], 'window_end': period['end'], 'source_crs': grid['crs'],
                'x': bundle['x'][0], 'y': bundle['y'][0]}
        for number, raw in enumerate(cube[:, :, row, column], start=1):
            pixel = {band: float(value)*.0001 for band, value in zip(imagery.BANDS[:-1], raw[:-1])}
            pixel.update(valid=True, clear_count=float(raw[-1]))
            base.update({f'bin_{number:02d}_{name}': value for name, value in seasons.pixel_features(pixel).items()})
        expected = experiments.enhance_window(base, recipe['feature_policy'])
        np.testing.assert_allclose(bundle['matrix'][0], [expected[name] for name in recipe['feature_columns']], atol=1e-12, rtol=0)
        self.assertEqual(bundle['manifest']['feature_columns'], experiments.feature_sets()['interval_and_shape'])
        self.assertAlmostEqual(bundle['matrix'][0, recipe['feature_columns'].index('bin_01_B4')], .1)
        self.assertAlmostEqual(bundle['matrix'][0, recipe['feature_columns'].index('bin_02_NDVI')], 1/3)
        self.assertEqual(expected['bin_01_clear_count'], 1)
        self.assertEqual(bundle['sample_ids'][0], expected['sample_id'])
        self.assertFalse(bundle['manifest']['land_cover_filter_applied'])
        # Source Dynamic World has one date, so ALL pixels lack supported land cover.
        self.assertTrue(all(int(row['Crop support']) == 0 for row in land_cover.load_land_cover_bundle(self.guidance)['coverage']))
        self.assertEqual(bundle['manifest']['row_count'], int(roi.sum()))

    def test_rerun_preserves_every_input_and_prior_output_byte(self):
        handoff = features.build(self.request)
        before = {p: p.read_bytes() for p in self.request['folder'].rglob('*') if p.is_file()}
        self.assertEqual(features.build(self.request), handoff)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertEqual({p: p.read_bytes() for p in self.upstream}, self.upstream)

    def test_all_supported_designs_preserve_the_saved_order_and_same_pixel_pool(self):
        counts = []
        for design, width in [('interval_only', 72), ('shape_only', 48)]:
            model = model_fixture(self.root / design, design)
            request = features.prepare_request(self.guidance, model, self.root / 'features')
            bundle = features.load_feature_bundle(features.build(request))
            self.assertEqual(bundle['matrix'].shape[1], width)
            self.assertEqual(bundle['manifest']['feature_columns'], experiments.feature_sets()[design])
            counts.append(bundle['manifest']['row_count'])
        self.assertEqual(counts[0], counts[1])

    def test_missing_bin_or_zero_index_denominator_never_gets_imputed(self):
        cube = np.full((8, 7, 1, 1), 1000, dtype='float32'); cube[:, -1] = 1
        period = imagery.four_month_period('2025-07-01')
        grid = {'crs': 'EPSG:32651', 'transform': [10, 0, 250000, 0, -10, 1850000]}
        cube[0, :-1] = -9999; cube[0, -1] = 0
        self.assertEqual(features.pixel_row(cube, 0, 0, period, grid, experiments.FEATURE_POLICY), (None, 1))
        cube[0, :-1] = 1000; cube[0, -1] = 1; cube[0, 1] = -1000
        self.assertEqual(features.pixel_row(cube, 0, 0, period, grid, experiments.FEATURE_POLICY), (None, 2))

    def test_failed_roi_coverage_publishes_audit_but_blocks_prediction(self):
        def sparse(retriever, plan, interval, granules, path):
            spectral_fetch(retriever, plan, interval, granules, path)
            if interval['name'].endswith('07_01_15'):
                roi = imagery._roi_mask({'settings': plan['settings'], 'fragment': plan['fragment']})
                positions = np.argwhere(roi)[:int(roi.sum()*.3)]
                with rasterio.open(path, 'r+') as raster:
                    data = raster.read()
                    data[:-1, positions[:, 0], positions[:, 1]] = -9999
                    data[-1, positions[:, 0], positions[:, 1]] = 0
                    raster.write(data)
        guidance = input_fixture(self.root / 'sparse', sparse)
        request = features.prepare_request(guidance, self.model, self.root / 'features')
        handoff = features.build(request)
        bundle = features.load_feature_bundle(handoff, require_ready=False)
        self.assertEqual(bundle['manifest']['status'], 'pending_inputs')
        self.assertEqual(bundle['matrix'].shape, (0, 120))
        self.assertGreater(next(int(row['pixels']) for row in bundle['audit'] if row['status'] == 'Failed ROI composite coverage'), 0)
        with self.assertRaisesRegex(ValueError, 'No ready'):
            features.load_feature_bundle(handoff)

    def test_reordered_columns_and_incompatible_scaling_are_rejected(self):
        loaded = features.load_model_recipe(self.model)
        recipe = deepcopy(loaded['recipe'])
        recipe['imagery']['spectral_scale'] = 1
        _, image, _ = features._inputs(self.guidance)
        with self.assertRaisesRegex(ValueError, 'Unsupported model imagery'):
            features.validate_imagery(recipe, image)
        manifest = json.loads(self.model.read_text())
        recipe = manifest['identity']['recipe']
        recipe['feature_columns'] = list(reversed(recipe['feature_columns']))
        recipe_path = self.model.parent / 'recipe.json'
        recipe_path.write_text(json.dumps(recipe))
        manifest['files']['recipe.json']['sha256'] = references.checksum(recipe_path)
        self.model.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'design/order'):
            features.load_model_recipe(self.model)

    def test_changed_formula_provenance_and_model_bytes_are_blocked(self):
        with patch.object(experiments, 'feature_sets', return_value={}):
            with self.assertRaises(ValueError):
                features.load_model_recipe(self.model)
        source = Path(json.loads(self.model.read_text())['feature_handoff'])
        source.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'provenance changed'):
            features.load_model_recipe(self.model)
        (self.model.parent / 'model.joblib').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'package changed'):
            features.load_model_recipe(self.model)

    def test_saved_pixel_coordinates_are_checked_before_mapping(self):
        handoff = features.build(self.request)
        matrix_path = handoff.parent / 'features.npz'
        with np.load(matrix_path, allow_pickle=False) as arrays:
            values = {name: arrays[name] for name in arrays.files}
        values['x'][0] += 10
        np.savez_compressed(matrix_path, **values)
        manifest = json.loads(handoff.read_text())
        manifest['files']['features.npz']['sha256'] = references.checksum(matrix_path)
        handoff.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'pixel centres'):
            features.load_feature_bundle(handoff)

    def test_failed_build_is_atomic_and_changed_saved_outputs_are_never_replaced(self):
        with patch.object(features, 'pixel_row', side_effect=OSError('Interrupted')), self.assertRaises(OSError):
            features.build(self.request)
        self.assertFalse(self.request['folder'].exists())
        handoff = features.build(self.request)
        matrix = handoff.parent / 'features.npz'; matrix.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'output changed'):
            features.build(self.request)
        self.assertEqual(matrix.read_bytes(), b'changed')

    def test_notebook_run_all_is_local_and_previous_pending_stage_blocks_it(self):
        root = self.root
        (root / 'pyproject.toml').write_text('[project]\nname="fixture"\n')
        (root / 'src/crop_pipeline').mkdir(parents=True)
        (root / 'settings.local.json').write_text(json.dumps({'report_root': './inventory'}))
        source_pointer = root / 'paoay/land_cover/land_cover_stage.json'
        write_stage_pointer(source_pointer, self.guidance)
        notebook = json.loads((REPOSITORY / 'notebooks/paoay/archive/eight_notebook_workflow/11_paoay_features.ipynb').read_text())
        def run():
            namespace = {}
            try:
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        exec(compile(''.join(cell['source']), 'paoay-feature-notebook', 'exec'), namespace)
            finally:
                progress.finish_notebook()
            return namespace
        with chdir(root), redirect_stdout(io.StringIO()), patch('socket.socket.connect', side_effect=AssertionError('No network')), \
             patch('joblib.load', side_effect=AssertionError('No model load')), \
             patch('sklearn.ensemble.RandomForestClassifier.fit', side_effect=AssertionError('No fitting')):
            namespace = run()
            self.assertEqual(resolve_stage_pointer(namespace['feature_pointer']), namespace['feature_handoff'])
            write_stage_pointer(source_pointer, status='pending', reason='Notebook 10 incomplete')
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                run()
        with self.assertRaisesRegex(RuntimeError, 'pending'):
            resolve_stage_pointer(namespace['feature_pointer'])


if __name__ == '__main__':
    unittest.main()
