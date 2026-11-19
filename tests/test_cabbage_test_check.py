import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from shapely.geometry import box, mapping

from crop_pipeline import cabbage_test_check as checker
from crop_pipeline.common import save_json
from crop_pipeline.sources import adaptive, test_cohort
from crop_pipeline.windows import seasons


def footprint(group, crop='cabbage', x=0):
    return {'type': 'Feature', 'geometry': mapping(box(x, 0, x + 1, 1)),
            'properties': {'partition_group': group, 'fragment_id': group,
                           'crop': crop, 'source_year': 2021, 'selected_pixels': 1}}


def pixel(group, x):
    return {'partition_group': group, 'region_id': group, 'source_crs': 'EPSG:32631',
            'x': x, 'y': 1, 'crop': 'cabbage', 'partition': 'test'}


class CabbageCheckTests(unittest.TestCase):
    def test_discovery_includes_supplements_pending_and_verifies_hashes(self):
        with TemporaryDirectory() as folder:
            report = Path(folder) / 'inventory'
            campaign = adaptive.root(report, test_cohort.CABBAGE_SUPPLEMENT_POLICY)
            batch = campaign / 'rounds/0001/cropharvest'
            manifest = Path(folder) / 'handoff.json'
            save_json(manifest, {'fixture': True})
            save_json(batch / 'state.json', {'status': 'completed'})
            save_json(batch / 'completed.json', {'handoffs': [
                {'path': str(manifest), 'sha256': seasons.file_sha256(manifest)}]})
            save_json(campaign / 'rounds/0001/next/state.json', {'status': 'pending'})
            with patch.object(checker.pilot_sources, 'load_sources', return_value=[{'source_name': 'test'}]):
                cohorts = checker.discover(report)
                supplement = next(c for c in cohorts if c['campaign'] == campaign.name)
                self.assertEqual(len(supplement['sources']), 1)
                self.assertEqual(len(supplement['batches']), 2)
                self.assertTrue(any(c['cohort'] == 'jecam' for c in cohorts))
                self.assertTrue(any(c['cohort'] == 'denmark_marker_cabbage_2023_v1' for c in cohorts))
                manifest.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'handoff changed'):
                    checker.discover(report)

    def test_changed_collection_does_not_publish_a_success_receipt(self):
        with TemporaryDirectory() as temporary:
            with patch.object(checker, 'discover', return_value=[]), \
                 patch.object(test_cohort, 'completed_sources', return_value=[]), \
                 patch.object(test_cohort, 'source_footprints', return_value=[]), \
                 patch.object(checker, 'collection_signature', side_effect=[{}, {'changed': 'hash'}]):
                with self.assertRaisesRegex(RuntimeError, 'Collection changed'):
                    checker.run(temporary)
            self.assertFalse(list(Path(temporary).rglob('check_receipt.json')))

    def test_counts_cap_deduplicate_match_and_include_zeros(self):
        rows = [pixel('a', i) for i in range(30)]
        result = checker.summarize({'jecam': {'fixed': rows + rows, 'aligned': rows[:5]},
                                    'supplement': {'fixed': rows, 'aligned': []},
                                    'missing': {'fixed': [], 'aligned': []}})
        def find(cohort, recipe, population):
            return next(r for r in result if (r['test_cohort'], r['window'], r['population'])
                        == (cohort, recipe, population))
        self.assertEqual(find('all_test_cohorts', 'fixed', 'full')['usable_pixels'], 20)
        self.assertEqual(find('all_test_cohorts', 'aligned', 'matched')['usable_pixels'], 5)
        self.assertEqual(find('all_test_cohorts', 'fixed', 'matched')['usable_pixels'], 5)
        self.assertEqual(find('missing', 'fixed', 'full')['pixel_shortfall'], 400)

    def test_run_filters_cabbage_screens_overlaps_and_preserves_primary_state(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / 'pipeline_outputs/preprocessing/current_quality.json'
            save_json(primary, {'preserve': True})
            original = primary.read_bytes()
            features = {'dev': [footprint('dev', x=0)],
                        'jecam': [footprint('overlap', x=0), footprint('a', x=3),
                                  footprint('carrot', crop='carrot', x=8)],
                        'supplement': [footprint('duplicate', x=3), footprint('b', x=6)]}
            cohorts = [{'campaign': name, 'cohort': name, 'sources': [{'name': name}],
                        'batches': [], 'status': 'completed_inputs_available'}
                       for name in ('jecam', 'supplement')]
            def audit(analysis):
                analysis.regions = features[analysis.sources[0]['name']]
                analysis.quality_rows = []
                analysis.composite_rows = []
                analysis.pixel_rows = type('Pixels', (), {'paths': {}})()
            def select(quality, report):
                saved = json.loads(quality.read_text())
                self.assertTrue(all(r['properties']['partition'] == 'test' for r in saved['regions']))
                rows = [pixel(r['properties']['partition_group'], 1) for r in saved['regions']]
                save_json(quality.parent / 'windows.json', {'fixed': rows, 'aligned': rows})
                return quality.parent / 'windows.json'
            with patch.object(checker, 'discover', return_value=cohorts), \
                 patch.object(test_cohort, 'completed_sources', return_value=[{'name': 'dev'}]), \
                 patch.object(test_cohort, 'source_footprints', side_effect=lambda sources: features[sources[0]['name']]), \
                 patch.object(checker.WindowAnalysis, 'audit', audit), \
                 patch.object(checker.external, 'select', side_effect=select), \
                 patch.object(checker.external, 'export_features'), \
                 patch('requests.sessions.Session.request', side_effect=AssertionError('No network')):
                output = checker.run(root)
            rows = seasons.read_table(output / 'parcel_outcomes.csv')
            self.assertEqual({r['parcel_group'] for r in rows}, {'overlap', 'a', 'duplicate', 'b'})
            self.assertEqual({r['parcel_group'] for r in rows if r['overlap_excluded'] == 'True'}, {'overlap', 'duplicate'})
            counts = seasons.read_table(output / 'window_counts.csv')
            combined = next(r for r in counts if r['test_cohort'] == 'all_test_cohorts' and r['window'] == 'fixed' and r['population'] == 'full')
            self.assertEqual(combined['usable_parcels'], '2')
            self.assertEqual(primary.read_bytes(), original)
            self.assertEqual(json.loads((output / 'check_receipt.json').read_text())['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
