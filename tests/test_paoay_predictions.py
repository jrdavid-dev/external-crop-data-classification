"""Fixed-model inference preserves class mapping, nodata, model bytes and provenance."""
from contextlib import chdir, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier

from crop_pipeline import progress
from crop_pipeline.analysis_workflow import write_stage_pointer, resolve_stage_pointer
from crop_pipeline.paoay import features, predictions, references
import test_paoay_features as fixtures

REPOSITORY = Path(__file__).resolve().parents[1]


def fitted_model(root):
    manifest_path = fixtures.model_fixture(root)
    manifest = json.loads(manifest_path.read_text())
    recipe = manifest['identity']['recipe']
    recipe.update(study_classes=['carrot', 'cabbage', 'potato'], class_codes={'carrot': 1, 'cabbage': 2, 'potato': 3}, nodata_code=0)
    columns = recipe['feature_columns']
    classifier = RandomForestClassifier(n_estimators=3, bootstrap=False, random_state=42, n_jobs=1)
    classifier.fit(np.zeros((6, len(columns))), ['carrot', 'cabbage', 'potato'] * 2)
    configuration = {'feature_policy': recipe['feature_policy'], 'study_classes': recipe['study_classes']}
    joblib.dump({'classifier': classifier, 'feature_columns': columns, 'configuration': configuration}, manifest_path.parent / 'model.joblib')
    (manifest_path.parent / 'recipe.json').write_text(json.dumps(recipe))
    manifest['identity']['model_sha256'] = references.checksum(manifest_path.parent / 'model.joblib')
    manifest['files'] = {name: {'sha256': references.checksum(manifest_path.parent/name)} for name in ('recipe.json', 'model.joblib')}
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


class PaoayPredictionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.guidance = fixtures.input_fixture(self.root)
        self.model = fitted_model(self.root)
        self.feature_request = features.prepare_request(self.guidance, self.model, self.root / 'features')
        self.feature_handoff = features.build(self.feature_request)
        self.upstream = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.request = predictions.prepare_request(self.feature_handoff, self.root / 'predictions')

    def test_class_order_mapping_ties_and_chunk_size_match_the_fixed_classifier(self):
        classifier, recipe = predictions.load_classifier(self.model)
        source = features.load_feature_bundle(self.feature_handoff)
        first = predictions.predict_matrix(classifier, source['matrix'], recipe, chunk_size=31)
        second = predictions.predict_matrix(classifier, source['matrix'], recipe, chunk_size=4096)
        np.testing.assert_array_equal(first['probabilities'], second['probabilities'])
        expected = [recipe['class_codes'][name] for name in classifier.predict(source['matrix'])]
        np.testing.assert_array_equal(first['class_codes'], expected)
        self.assertTrue(first['tied'].all())
        self.assertTrue((first['class_codes'] == 2).all())  # Learned alphabetic order wins ties, not output-band order.
        def unequal(matrix):
            raw = np.full((len(matrix), 3), .1)
            for index in range(len(matrix)):
                raw[index, index % 3] = .8
            return raw
        with patch.object(classifier, 'predict_proba', unequal):
            result = predictions.predict_matrix(classifier, source['matrix'][:3], recipe)
        np.testing.assert_array_equal(result['class_codes'], [2, 1, 3])
        np.testing.assert_allclose(result['probabilities'], [[.1, .8, .1], [.8, .1, .1], [.1, .1, .8]])

    def test_build_is_inference_only_and_every_model_input_and_result_byte_is_preserved(self):
        with patch('sklearn.ensemble.RandomForestClassifier.fit', side_effect=AssertionError('No refitting')), \
             patch('joblib.dump', side_effect=AssertionError('Do not save a model')), \
             patch('socket.socket.connect', side_effect=AssertionError('No network')):
            handoff = predictions.build(self.request, chunk_size=101)
            before = {p: p.read_bytes() for p in self.request['folder'].rglob('*') if p.is_file()}
            with patch('joblib.load', side_effect=AssertionError('Cache reuse must not deserialize')):
                self.assertEqual(predictions.build(self.request), handoff)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertEqual({p: p.read_bytes() for p in self.upstream}, self.upstream)
        bundle = predictions.load_prediction_bundle(handoff)
        self.assertFalse(bundle['manifest']['model_refitted'])
        self.assertFalse(bundle['manifest']['land_cover_filter_applied'])
        self.assertFalse(bundle['manifest']['confidence_threshold_applied'])
        self.assertEqual(sum(int(row['pixels']) for row in bundle['summary']), bundle['manifest']['row_count'])

    def test_rasters_preserve_pixel_centres_codes_probability_labels_and_nodata(self):
        bundle = predictions.load_prediction_bundle(predictions.build(self.request))
        source = features.load_feature_bundle(self.feature_handoff)
        grid = source['manifest']['grid']
        ready = np.zeros((grid['height'], grid['width']), dtype=bool)
        ready[source['rows'], source['columns']] = True
        with rasterio.open(bundle['folder']/'crop_class.tif') as raster:
            data = raster.read(1)
            self.assertEqual(raster.nodata, 0)
            np.testing.assert_array_equal(data[ready], bundle['class_codes'])
            self.assertTrue((data[~ready] == 0).all())
        with rasterio.open(bundle['folder']/'crop_probabilities.tif') as raster:
            self.assertEqual(raster.descriptions[:3], ('probability_carrot', 'probability_cabbage', 'probability_potato'))
            data = raster.read()
            np.testing.assert_allclose(data[:3, source['rows'], source['columns']].T, bundle['probabilities'])
            self.assertTrue((data[:, ~ready] == -9999).all())

    def test_invalid_probabilities_or_chunk_sizes_are_rejected(self):
        classifier, recipe = predictions.load_classifier(self.model)
        matrix = features.load_feature_bundle(self.feature_handoff)['matrix'][:2]
        for size in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, 'chunk_size'):
                predictions.predict_matrix(classifier, matrix, recipe, size)
        for values in ([[.5, .5]], [[.5, .5, .5]], [[np.nan, 0, 1]], [[-1, 1, 1]]):
            raw = np.tile(values, (len(matrix), 1))
            with patch.object(classifier, 'predict_proba', return_value=raw), self.assertRaisesRegex(ValueError, 'probabilities'):
                predictions.predict_matrix(classifier, matrix, recipe)

    def test_pending_features_block_before_model_deserialization(self):
        request = features.prepare_request(self.guidance, self.model, self.root/'empty_features')
        with patch.object(features, 'pixel_row', return_value=(None, 3)):
            pending = features.build(request)
        with patch('joblib.load', side_effect=AssertionError('No model load')):
            with self.assertRaisesRegex(ValueError, 'No ready'):
                predictions.prepare_request(pending, self.root/'predictions')

    def test_fitted_artifact_schema_mismatch_is_rejected(self):
        artifact = joblib.load(self.model.parent/'model.joblib')
        artifact['feature_columns'] = artifact['feature_columns'][::-1]
        with patch.object(predictions.joblib, 'load', return_value=artifact), self.assertRaisesRegex(ValueError, 'feature order'):
            predictions.load_classifier(self.model)
        artifact['feature_columns'] = artifact['feature_columns'][::-1]
        artifact['classifier'].classes_ = np.array(['carrot', 'cabbage', 'other'])
        with patch.object(predictions.joblib, 'load', return_value=artifact), self.assertRaisesRegex(ValueError, 'dimensions/classes'):
            predictions.load_classifier(self.model)

    def test_interrupted_inference_publishes_nothing_and_does_not_modify_model(self):
        before = (self.model.parent/'model.joblib').read_bytes()
        original = RandomForestClassifier.predict_proba
        calls = []
        def interrupted(classifier, matrix):
            calls.append(1)
            if len(calls) == 2:
                raise OSError('Interrupted inference')
            return original(classifier, matrix)
        with patch.object(RandomForestClassifier, 'predict_proba', interrupted), self.assertRaises(OSError):
            predictions.build(self.request, chunk_size=101)
        self.assertFalse(self.request['folder'].exists())
        self.assertEqual((self.model.parent/'model.joblib').read_bytes(), before)
        predictions.load_prediction_bundle(predictions.build(self.request))

    def test_changed_outputs_dependencies_and_grid_are_not_silently_reused(self):
        handoff = predictions.build(self.request)
        manifest = json.loads(handoff.read_text())
        manifest['grid']['width'] += 1
        handoff.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'schema/dates/grid'):
            predictions.build(self.request)
        self.feature_handoff.write_text(self.feature_handoff.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'dependency changed'):
            predictions.build(self.request)

    def test_corrupt_prediction_and_misplaced_raster_values_are_blocked(self):
        handoff = predictions.build(self.request)
        raster_path = handoff.parent/'crop_class.tif'
        with rasterio.open(raster_path, 'r+') as raster:
            data = raster.read(1); data[:] = 0; raster.write(data, 1)
        with self.assertRaisesRegex(ValueError, 'Saved prediction changed'):
            predictions.load_prediction_bundle(handoff)
        manifest = json.loads(handoff.read_text())
        manifest['files']['crop_class.tif']['sha256'] = references.checksum(raster_path)
        handoff.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'feature pixel locations'):
            predictions.load_prediction_bundle(handoff)

    def test_preview_and_notebook_run_all_are_local_without_refitting(self):
        import matplotlib.pyplot as plt
        bundle = predictions.load_prediction_bundle(predictions.build(self.request))
        figure = predictions.plot_predictions(bundle)
        self.assertEqual(len(figure.axes), 3)
        figure.savefig(self.root/'preview.png'); plt.close(figure)
        root = self.root
        (root/'pyproject.toml').write_text('[project]\nname="fixture"\n')
        (root/'src/crop_pipeline').mkdir(parents=True)
        (root/'settings.local.json').write_text(json.dumps({'report_root': './inventory'}))
        source_pointer = root/'paoay/features/feature_stage.json'
        write_stage_pointer(source_pointer, self.feature_handoff)
        notebook = json.loads((REPOSITORY/'notebooks/paoay/archive/eight_notebook_workflow/12_paoay_predictions.ipynb').read_text())
        def run():
            namespace = {}
            try:
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        exec(compile(''.join(cell['source']), 'paoay-prediction-notebook', 'exec'), namespace)
                        namespace['show_preview'] = False
            finally:
                progress.finish_notebook()
            return namespace
        with chdir(root), redirect_stdout(io.StringIO()), \
             patch('socket.socket.connect', side_effect=AssertionError('No network')), \
             patch('sklearn.ensemble.RandomForestClassifier.fit', side_effect=AssertionError('No fitting')):
            namespace = run()
            self.assertEqual(resolve_stage_pointer(namespace['prediction_pointer']), namespace['prediction_handoff'])
            write_stage_pointer(source_pointer, status='pending', reason='Notebook 11 incomplete')
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                run()
        with self.assertRaisesRegex(RuntimeError, 'pending'):
            resolve_stage_pointer(namespace['prediction_pointer'])


if __name__ == '__main__':
    unittest.main()
