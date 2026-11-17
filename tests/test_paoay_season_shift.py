"""Season comparison describes curve differences and never changes predictions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crop_pipeline.paoay import references, season_shift

COLUMNS = [name for index in season_shift.INDICES for name in season_shift.bin_columns(index)]
PERIOD = {'start': '2024-01-01', 'end': '2024-05-01', 'intervals': [
    {'start': f'2024-{month:02d}-{day:02d}'} for month in (1, 2, 3, 4) for day in (1, 16)]}


def curve(shift=0, level=0.):
    return level + .5 + .3 * np.sin((np.arange(8) - shift) * np.pi / 4)


class SeasonMathTests(unittest.TestCase):
    def test_lag_sign_limit_and_tie_breaking(self):
        later = season_shift.best_lag(curve(), curve(2))
        self.assertEqual(later['lag'], 2)
        self.assertAlmostEqual(later['rmse'], 0)
        self.assertGreater(later['rmse_same_bins'], .1)
        self.assertEqual(season_shift.best_lag(curve(2), curve())['lag'], -2)
        self.assertEqual(season_shift.best_lag(np.full(8, .4), np.full(8, .4))['lag'], 0)
        # Shifts beyond three bins leave fewer than five shared bins and are not tested.
        self.assertLessEqual(abs(season_shift.best_lag(curve(), curve(6))['lag']), 3)
        with self.assertRaises(ValueError):
            season_shift.best_lag(curve(), curve()[:7])

    def test_summary_labels_and_design_support(self):
        summary = season_shift.summarize(np.vstack([curve(), curve(level=.2), curve(level=.4)]))
        np.testing.assert_allclose(summary['median'], curve(level=.2))
        self.assertEqual(summary['pixels'], 3)
        self.assertEqual(season_shift.bin_labels(PERIOD)[:3], ['Jan 1–15', 'Jan 16–end', 'Feb 1–15'])
        self.assertTrue(season_shift.available(COLUMNS))
        self.assertFalse(season_shift.available(['shape_NDVI_amplitude']))
        flat = season_shift.compare(np.full(8, .2), curve())
        self.assertEqual(flat['correlation'], '')
        self.assertEqual(flat['study_peak_bin'], 3)


class SeasonBundleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        rng = np.random.default_rng(1)
        def pixel(base):
            return dict(zip(COLUMNS, map(float, np.tile(base, 4) + rng.normal(0, .01, 32))))
        rows = ([{'partition_group': 'a', 'partition': 'train', 'crop': 'carrot', **pixel(curve())} for _ in range(25)]
                + [{'partition_group': 'b', 'partition': 'train', 'crop': 'potato', **pixel(curve(1))} for _ in range(25)]
                + [{'partition_group': 'v', 'partition': 'validation', 'crop': 'carrot', **pixel(curve(level=5))} for _ in range(25)])
        run = self.root / 'run'
        (run / 'models/fixed_full').mkdir(parents=True)
        (run / 'features_development.json').write_text(json.dumps({'rows': {'fixed_full': rows}}))
        (run / 'models/development.json').write_text(json.dumps({'feature_sha256': references.checksum(run / 'features_development.json')}))
        self.model = run / 'models/fixed_full/model_manifest.json'
        self.model.write_text(json.dumps({'experiment_cohort': 'full', 'identity': {'recipe': {'window_recipe': {'type': 'fixed'}}},
                                          'feature_handoff': str(run / 'models/development.json'),
                                          'development_split': {'a': 'train', 'b': 'train', 'v': 'validation'}}))
        for name in ('feature', 'prediction', 'imagery', 'reference'):
            (self.root / f'{name}_handoff.json').write_text(json.dumps({'name': name}))
        matrix = np.vstack([np.tile(curve(2), 4) + rng.normal(0, .01, 32) for _ in range(60)])
        feature = {'manifest': {'model_manifest': str(self.model), 'period': PERIOD, 'feature_columns': COLUMNS,
                                'imagery_handoff': str(self.root / 'imagery_handoff.json'),
                                'grid': {'crs': 'EPSG:32651', 'transform': [10, 0, 0, 0, -10, 60], 'width': 10, 'height': 6}},
                   'matrix': matrix, 'rows': np.repeat(np.arange(6), 10), 'columns': np.tile(np.arange(10), 6)}
        prediction = {'manifest': {'feature_handoff': str(self.root / 'feature_handoff.json'), 'class_codes': {'carrot': 1, 'cabbage': 2, 'potato': 3}},
                      'class_codes': np.r_[np.full(30, 1), np.full(25, 3), np.full(5, 2)]}
        mask = np.zeros((6, 10), dtype=bool); mask[:3] = True
        for target, value in (('crop_pipeline.paoay.features.load_feature_bundle', lambda *a, **k: feature),
                              ('crop_pipeline.paoay.predictions.load_prediction_bundle', lambda *a: prediction),
                              ('crop_pipeline.paoay.imagery.load_imagery_bundle', lambda *a: {'manifest': {'reference_handoff': str(self.root / 'reference_handoff.json')}}),
                              ('crop_pipeline.paoay.references.load_reference_bundle', lambda *a: {}),
                              ('crop_pipeline.paoay.crop_baseline.reference_mask', lambda *a: mask)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_bundle_reports_the_shift_per_training_crop_and_is_reused(self):
        request = season_shift.prepare_request(self.root / 'prediction_handoff.json', self.root / 'out')
        handoff = season_shift.build(request)
        bundle = season_shift.load_bundle(handoff)
        manifest = bundle['manifest']
        # Too few predicted-cabbage pixels and no cabbage training pixels: neither becomes a curve.
        self.assertEqual(manifest['study_groups'], {'All study-area pixels': 60, 'Reference areas': 30,
                                                    'Predicted carrot': 30, 'Predicted potato': 25})
        self.assertIs(manifest['predictions_changed'], False)
        self.assertEqual({row['group'] for row in bundle['curves'] if row['source'] == 'training'}, {'carrot', 'potato'})
        lags = {(row['crop'], row['study_group']): int(row['best_lag_bins']) for row in bundle['comparison'] if row['index'] == 'NDVI'}
        self.assertEqual(lags[('carrot', 'Reference areas')], 2)
        self.assertEqual(lags[('potato', 'All study-area pixels')], 1)
        self.assertNotIn(('cabbage', 'Reference areas'), lags)
        # Validation pixels sit at a very different level and must not enter the training curve.
        training = [float(row['median']) for row in bundle['curves'] if (row['source'], row['group'], row['index']) == ('training', 'carrot', 'NDVI')]
        self.assertLess(max(training), 1)
        display = season_shift.comparison_rows(bundle)
        self.assertEqual([row['crop'] for row in display], ['Carrot', 'Potato'])
        self.assertTrue(display[0]['best_shift'].startswith('+2 bins (+30 days)'))
        self.assertEqual(season_shift.build(request), handoff)
        (handoff.parent / 'curves.csv').write_text('changed')
        with self.assertRaises(ValueError):
            season_shift.load_bundle(handoff)


if __name__ == '__main__':
    unittest.main()
