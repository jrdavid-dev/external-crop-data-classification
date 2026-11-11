"""Five-notebook batch 1 reuses existing recipes and preserves manual reviews."""
from contextlib import chdir, redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crop_pipeline import progress
from crop_pipeline.analysis_workflow import resolve_stage_pointer, write_stage_pointer
from crop_pipeline.paoay import imagery, dynamic_world, references, reference_suitability
import test_paoay_imagery as sentinel
import test_paoay_dynamic_world as world
import test_paoay_references as reference_fixtures

ROOT = Path(__file__).resolve().parents[1]


class BatchOneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'pyproject.toml').write_text('[project]\nname="fixture"\n')
        (self.root/'src/crop_pipeline').mkdir(parents=True)
        self.reference = sentinel.reference_fixture(self.root/'inputs')
        self.paoay = self.root/'sen4agrinet_work/paoay'
        write_stage_pointer(self.paoay/'references/reference_stage.json', self.reference)
        self.request = imagery.prepare_request(self.reference, '2024-01-01', self.paoay/'sentinel2')
        with patch.object(imagery.PaoayRetriever, 'list_acquisitions', sentinel.acquisitions), \
             patch.object(imagery.PaoayRetriever, 'find_cloud_probability_matches', lambda retriever, fragment: {r['image_index'] for r in sentinel.acquisitions(retriever, fragment)}), \
             patch.object(imagery.PaoayRetriever, 'fetch_composite', sentinel.fake_fetch):
            self.image = imagery.collect(self.request, 'download_missing', object())
        self.world_request = dynamic_world.prepare_request(self.image, self.paoay/'dynamic_world')

    def populate_world(self):
        with patch.object(dynamic_world.DynamicWorldRetriever, 'list_acquisitions', world.world_acquisitions), \
             patch.object(dynamic_world.DynamicWorldRetriever, 'fetch_composite', world.world_fetch):
            return dynamic_world.collect(self.world_request, 'download_missing', object())

    def run_notebook(self, changes=None):
        notebook = json.loads((ROOT/'notebooks/paoay/02_satellite_inputs.ipynb').read_text())
        namespace = {}
        try:
            with chdir(self.root), redirect_stdout(io.StringIO()):
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        exec(compile(''.join(cell['source']), 'combined-satellite-notebook', 'exec'), namespace)
                        if changes and 'request' not in namespace:
                            namespace.update(changes)
        finally:
            progress.finish_notebook()
        return namespace

    def test_combined_cache_reuse_preserves_bytes_and_publishes_both_dependencies(self):
        self.populate_world()
        before = {p:p.read_bytes() for p in self.paoay.rglob('*') if p.is_file() and not p.name.endswith('_stage.json')}
        with patch('socket.socket.connect', side_effect=AssertionError('No network')), \
             patch('joblib.load', side_effect=AssertionError('No model loading')):
            result = self.run_notebook()
        self.assertEqual(resolve_stage_pointer(result['imagery_pointer']), self.image.resolve())
        self.assertEqual(resolve_stage_pointer(result['dynamic_world_pointer']), result['dynamic_world_handoff'])
        self.assertEqual(result['world_request']['period'], result['request']['period'])
        self.assertEqual(result['world_request']['settings']['grid'], result['request']['settings']['grid'])
        self.assertEqual({p:p.read_bytes() for p in before}, before)

    def test_missing_second_source_leaves_both_pointers_pending(self):
        self.populate_world()
        # Start with successful selections, then choose a separate empty world cache.
        write_stage_pointer(self.paoay/'sentinel2/imagery_stage.json', self.image)
        write_stage_pointer(self.paoay/'dynamic_world/dynamic_world_stage.json', self.world_request['folder']/'dynamic_world_handoff.json')
        with self.assertRaisesRegex(FileNotFoundError, 'missing'):
            self.run_notebook({'dynamic_world_mode':'reuse', 'dynamic_world_root':self.paoay/'separate_world'})
        for pointer in ('sentinel2/imagery_stage.json', 'dynamic_world/dynamic_world_stage.json'):
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                resolve_stage_pointer(self.paoay/pointer)

    def test_only_missing_source_downloads_with_one_earth_engine_initialization(self):
        (self.root/'settings.local.json').write_text(json.dumps({'earth_engine_project':'fixture-project'}))
        fake_ee = SimpleNamespace(Initialize=lambda **kwargs:None, EEException=RuntimeError)
        with patch.dict('sys.modules', {'ee':fake_ee}), \
             patch('crop_pipeline.notebook_setup.configure_certificates'), \
             patch.object(imagery.PaoayRetriever, 'fetch_composite', side_effect=AssertionError('Sentinel must reuse')), \
             patch.object(dynamic_world.DynamicWorldRetriever, 'list_acquisitions', world.world_acquisitions), \
             patch.object(dynamic_world.DynamicWorldRetriever, 'fetch_composite', autospec=True, side_effect=world.world_fetch) as fetch:
            result = self.run_notebook({'imagery_mode':'download_missing','dynamic_world_mode':'download_missing'})
        self.assertEqual(fetch.call_count, 8)
        self.assertEqual(resolve_stage_pointer(result['dynamic_world_pointer']), result['dynamic_world_handoff'])

    def test_suitability_review_is_manual_preserves_edits_and_rejects_duplicates(self):
        records, sources = references.read_inputs(reference_fixtures.fixture(self.root/'suitability_inputs'))
        self.reference = references.export_reference_bundle(records, sources, self.root/'suitability_references')
        before = {p:p.read_bytes() for p in self.reference.parent.iterdir() if p.is_file()}
        path, rows = reference_suitability.prepare_review(self.reference)
        self.assertTrue(all(r['area_type']=='unreviewed' for r in rows))
        rows[0].update(area_type='rotation_or_mixed', temporal_evidence='calendar_only', review_notes='Preserve this edit')
        def write(values):
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=reference_suitability.COLUMNS);writer.writeheader();writer.writerows(values)
        write(rows); edited=path.read_bytes()
        _, restored=reference_suitability.prepare_review(self.reference)
        self.assertEqual(restored[0]['review_notes'],'Preserve this edit')
        self.assertEqual(path.read_bytes(), edited)
        self.assertEqual({p:p.read_bytes() for p in before},before)
        write(rows+[rows[0]])
        with self.assertRaisesRegex(ValueError,'once'):
            reference_suitability.prepare_review(self.reference)

    def test_archived_notebooks_match_inventory(self):
        import hashlib
        folder=ROOT/'notebooks/paoay/archive/eight_notebook_workflow'
        inventory=json.loads((folder/'inventory.json').read_text())
        self.assertEqual(len(inventory),8)
        for name,digest in inventory.items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),digest)
