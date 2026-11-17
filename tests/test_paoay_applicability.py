"""Model applicability flags pixels unlike the training data and never changes predictions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crop_pipeline.paoay import applicability, references


def training():
    rng = np.random.default_rng(0)
    centres = np.array([[0., 0., 5.], [1., 1., 5.], [2., 0., 5.], [0., 2., 5.]])
    matrix = np.vstack([centre + np.r_[rng.normal(0, .05, 2), 0] for centre in centres for _ in range(6)])
    groups = np.repeat(['a', 'b', 'c', 'd'], 6)
    return matrix, groups


class ApplicabilityMathTests(unittest.TestCase):
    def test_training_di_ignores_the_same_parcel_and_far_pixels_fall_outside(self):
        matrix, groups = training()
        reference = applicability.training_reference(matrix, groups, [.5, .5, 0.])
        # Pixels of one parcel are near-duplicates; only other parcels may be neighbours.
        self.assertGreater(reference['training_di'].min(), .3)
        self.assertAlmostEqual(reference['threshold'], reference['training_di'].max())
        new = np.array([matrix[0], [1., 1., 99.], [30., 30., 5.]])
        index = applicability.dissimilarity(reference, new, chunk_size=2)
        self.assertAlmostEqual(index[0], 0.)
        # The third predictor is constant in training and has zero importance: it cannot move a pixel outside.
        self.assertLess(index[1], reference['threshold'])
        self.assertGreater(index[2], reference['threshold'])
        self.assertEqual(len(applicability.dissimilarity(reference, np.empty((0, 3)))), 0)

    def test_outliers_do_not_stretch_the_threshold(self):
        matrix, groups = training()
        matrix = np.vstack([matrix, [[40., 40., 5.]]])
        groups = np.append(groups, 'isolated')
        reference = applicability.training_reference(matrix, groups, [.5, .5, 0.])
        self.assertLess(reference['threshold'], reference['training_di'].max())

    def test_invalid_training_inputs_are_refused(self):
        matrix, groups = training()
        for arguments in ((matrix, groups, [1., 1.]), (matrix, groups, [0., 0., 0.]), (matrix, groups, [1., -1., 0.]),
                          (matrix, np.repeat('one', len(matrix)), [1., 1., 1.]),
                          (np.where(np.arange(matrix.size).reshape(matrix.shape) == 0, np.nan, matrix), groups, [1., 1., 1.])):
            with self.assertRaises(ValueError):
                applicability.training_reference(*arguments)


class ApplicabilityBundleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        matrix, groups = training()
        validation = matrix[:4] + .01
        self.columns = ['f1', 'f2', 'f3']
        rows = ([{'partition_group': g, 'partition': 'train', **dict(zip(self.columns, map(float, v)))} for v, g in zip(matrix, groups)]
                + [{'partition_group': 'v', 'partition': 'validation', **dict(zip(self.columns, map(float, v)))} for v in validation])
        run = self.root / 'run'
        (run / 'models/fixed_full').mkdir(parents=True)
        self.rows = run / 'features_development.json'
        self.rows.write_text(json.dumps({'rows': {'fixed_full': rows}}))
        snapshot = run / 'models/development.json'
        snapshot.write_text(json.dumps({'feature_sha256': references.checksum(self.rows)}))
        self.model = run / 'models/fixed_full/model_manifest.json'
        self.manifest = {'experiment_cohort': 'full', 'identity': {'recipe': {'window_recipe': {'type': 'fixed'}}},
                         'feature_handoff': str(snapshot),
                         'development_split': {**{g: 'train' for g in 'abcd'}, 'v': 'validation'}}
        self.model.write_text(json.dumps(self.manifest))
        self.feature_handoff = self.root / 'feature_handoff.json'
        self.feature_handoff.write_text('{}')
        pixels = np.array([[0., 0., 5.], [1., 1., 5.], [30., 30., 5.]])
        self.feature = {'manifest': {'model_manifest': str(self.model), 'period': {'start': '2024-01-01', 'end': '2024-05-01'},
                                     'grid': {'crs': 'EPSG:32651', 'transform': [10, 0, 0, 0, -10, 20], 'width': 2, 'height': 2}},
                        'matrix': pixels, 'rows': np.array([0, 0, 1]), 'columns': np.array([0, 1, 0])}
        classifier = type('Classifier', (), {'feature_importances_': np.array([.5, .5, 0.])})()
        for target, value in (('crop_pipeline.paoay.features.load_feature_bundle', lambda *a, **k: self.feature),
                              ('crop_pipeline.paoay.predictions.load_classifier', lambda *a: (classifier, {'feature_columns': self.columns}))):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_training_rows_are_found_by_checksum_or_reported_unavailable(self):
        self.assertEqual(applicability.locate_training_rows(self.model), (self.rows, 'fixed_full', None))
        self.rows.write_text(json.dumps({'rows': {'fixed_full': []}}))
        path, cohort, reason = applicability.locate_training_rows(self.model)
        self.assertIsNone(path)
        self.assertIn('not found', reason)
        with self.assertRaises(FileNotFoundError):
            applicability.prepare_request(self.feature_handoff, self.root / 'out')
        self.model.write_text(json.dumps({**self.manifest, 'experiment_cohort': None}))
        self.assertIn('does not record', applicability.locate_training_rows(self.model)[2])

    def test_bundle_flags_far_pixels_without_touching_inputs_and_is_reused(self):
        before = {path: path.read_bytes() for path in (self.rows, self.model, self.feature_handoff)}
        request = applicability.prepare_request(self.feature_handoff, self.root / 'out')
        handoff = applicability.build(request)
        bundle = applicability.load_bundle(handoff)
        np.testing.assert_array_equal(bundle['status'], [[1, 1], [2, 0]])
        self.assertTrue(np.isnan(bundle['index'][1, 1]))
        manifest = bundle['manifest']
        self.assertEqual((manifest['study_pixels'], manifest['study_pixels_inside'], manifest['training_pixels'],
                          manifest['training_parcel_groups'], manifest['validation_pixels']), (3, 2, 24, 4, 4))
        self.assertIs(manifest['predictions_changed'], False)
        self.assertEqual(len(bundle['training_di']), 24)
        self.assertEqual(applicability.build(request), handoff)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        prediction = {'manifest': {'feature_handoff': str(self.feature_handoff), 'class_codes': {'carrot': 1, 'potato': 3}},
                      'class_codes': np.array([1, 3, 3]), 'top_probability': np.array([.7, .8, .5])}
        table = {row['raw_prediction']: row for row in applicability.by_prediction(bundle, prediction)}
        self.assertEqual((table['Carrot']['inside_applicability'], table['Potato']['pixels'], table['Potato']['inside_fraction']), (1, 2, .5))
        (handoff.parent / 'summary.csv').write_text('changed')
        with self.assertRaises(ValueError):
            applicability.load_bundle(handoff)

    def test_rows_that_disagree_with_the_recorded_split_are_refused(self):
        request = applicability.prepare_request(self.feature_handoff, self.root / 'out')
        self.model.write_text(json.dumps({**self.manifest, 'development_split': {**self.manifest['development_split'], 'a': 'validation'}}))
        request['identity']['model_manifest_sha256'] = references.checksum(self.model)
        request['bundle_id'] = applicability.imagery._digest(request['identity'])[:20]
        with self.assertRaises(ValueError):
            applicability.build(request)


if __name__ == '__main__':
    unittest.main()
