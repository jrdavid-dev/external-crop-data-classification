import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from crop_pipeline.visualization import demo


class DemoVisualTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_coverage_keeps_cohorts_and_units_separate(self):
        rows = [dict(cohort=c, crop='carrot', partition='train', usable_parcel_groups='2', usable_pixels='40')
                for c in ('fixed_full', 'aligned_full')]
        fig = demo.plot_usable_samples(rows)
        self.assertEqual(len(fig.axes), 4)
        self.assertEqual(fig.axes[0].patches[0].get_width(), 2)
        self.assertEqual(fig.axes[1].patches[0].get_width(), 40)
        self.assertEqual(rows[0]['usable_pixels'], '40')

    def test_confusion_filters_population_and_keeps_reference_axes(self):
        rows = [dict(test_population=p, cohort='fixed_full', actual='carrot', predicted='potato', parcels=n)
                for p,n in [('jecam','3'),('supplement','99')]]
        fig = demo.plot_confusion(rows)
        matrix = fig.axes[0].images[0].get_array()
        self.assertEqual(matrix[0,2], 3)
        self.assertEqual(matrix.sum(), 3)
        self.assertEqual(fig.axes[0].get_ylabel(), 'Reference crop')
        with self.assertRaises(ValueError):
            demo.plot_confusion(rows, 'missing')

    def test_validation_uses_saved_candidates_including_zero(self):
        fig = demo.plot_validation({'fixed_full': {'selected_min_samples_leaf': 4, 'candidates': [
            {'min_samples_leaf': 2, 'metrics': {'macro_f1': 0}},
            {'min_samples_leaf': 4, 'metrics': {'macro_f1': .6}}]}})
        self.assertEqual([b.get_width() for b in fig.axes[0].patches], [0, .6])
