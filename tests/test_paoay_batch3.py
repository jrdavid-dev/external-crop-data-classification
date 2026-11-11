"""Field review freezes maps, preserves edits and never invents historical validation."""
from contextlib import chdir,redirect_stdout
import csv
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from crop_pipeline import progress
from crop_pipeline.analysis_workflow import resolve_stage_pointer,write_stage_pointer
from crop_pipeline.paoay import crop_baseline,field_review,maps,review,screening
import test_paoay_final_workflow as fixtures

ROOT=Path(__file__).resolve().parents[1]


class FieldReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.FinalWorkflowTests(methodName='runTest')
        self.addCleanup(self.fixture.doCleanups);self.fixture.setUp()
        self.root=self.fixture.root
        self.map=maps.build(self.fixture.map_request)
        self.baseline=crop_baseline.build(crop_baseline.prepare_request(self.fixture.prediction,self.root/'crop_baseline'))
        self.flags=screening.build(screening.prepare_request(self.baseline,self.map,self.root/'screening'))
        self.context=review.build(review.prepare_request(self.map,self.root/'review'))
        self.inputs=maps.context(self.fixture.prediction)
        self.field_root=self.root/'field_review'
        self.observations,self.corrections=field_review.prepare_records(self.inputs['reference_path'],self.field_root)
        self.area=next(r for r in self.inputs['reference']['records'] if r['name']=='B1')
        self.upstream={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def write_observations(self, rows):
        field_review._write(self.observations,rows,field_review.OBSERVATION_COLUMNS)

    def observation(self,**changes):
        row={key:'' for key in field_review.OBSERVATION_COLUMNS}
        row.update(observation_id='visit-001',reference_id=self.area['reference_id'],observation_date='2027-01-10',
                   latitude='16.6005',longitude='120.7005',observed_crop='lettuce',surface_state='crop',
                   crop_stage='growing',evidence_type='field_visit',evidence_reference='photo-001',
                   observer='Field team',review_status='unreviewed')
        row.update(changes);return row

    def request(self):
        return field_review.prepare_request(self.baseline,self.flags,self.context,self.field_root)

    def test_blank_records_freeze_map_without_inventing_observations_or_corrections(self):
        with patch('socket.socket.connect',side_effect=AssertionError('No network')), \
             patch('joblib.load',side_effect=AssertionError('No model loading')):
            request=self.request();handoff=field_review.build(request);bundle=field_review.load_bundle(handoff)
            saved={p:p.read_bytes() for p in bundle['folder'].iterdir()}
            self.assertEqual(field_review.build(request),handoff)
        self.assertEqual(bundle['manifest']['observation_count'],0)
        self.assertEqual(bundle['manifest']['correction_proposal_count'],0)
        self.assertFalse(bundle['manifest']['corrected_map_created'])
        self.assertFalse(bundle['manifest']['accuracy_evaluated'])
        self.assertEqual((bundle['folder']/'preliminary_crop_map.tif').read_bytes(),(self.baseline.parent/'crop_baseline.tif').read_bytes())
        self.assertEqual({p:p.read_bytes() for p in self.upstream},self.upstream)
        self.assertEqual({p:p.read_bytes() for p in saved},saved)
        checklist=(bundle['folder']/'field_check_plan.csv').read_text()
        self.assertNotIn('preliminary_prediction',checklist)
        (bundle['folder']/'report.md').write_text('corrupt')
        with self.assertRaisesRegex(ValueError,'changed'):
            field_review.load_bundle(handoff)

    def test_january_visit_stays_different_period_and_unmodelled_crop_stays_explicit(self):
        self.write_observations([self.observation()])
        field_review._write(self.corrections,[{'correction_id':'proposal-001','observation_id':'visit-001',
                             'proposed_label':'lettuce','decision':'proposed','reason':'January photo; different map period'}],field_review.CORRECTION_COLUMNS)
        request=self.request();handoff=field_review.build(request);bundle=field_review.load_bundle(handoff)
        row=bundle['observations'][0]
        self.assertEqual(row['temporal_relation'],'different_period')
        self.assertEqual(row['observed_crop'],'lettuce')
        self.assertEqual(row['location_status'],'mapped_pixel')
        self.assertEqual(bundle['manifest']['correction_proposal_count'],1)
        self.assertFalse(bundle['manifest']['accuracy_evaluated'])
        self.assertFalse(bundle['manifest']['corrected_map_created'])
        edited=self.observations.read_bytes()
        field_review.prepare_records(self.inputs['reference_path'],self.field_root)
        self.assertEqual(self.observations.read_bytes(),edited)
        # Later source edits create a new snapshot; old frozen review remains readable.
        self.write_observations([self.observation(observation_id='visit-002')])
        with self.assertRaisesRegex(ValueError,'changed'):
            field_review.build(request)
        self.assertEqual(field_review.load_bundle(handoff)['manifest']['observation_count'],1)
        # Correction now references a removed observation; report must stop, not guess.
        with self.assertRaisesRegex(ValueError,'existing observation'):
            self.request()
        field_review._write(self.corrections,[],field_review.CORRECTION_COLUMNS)
        self.assertNotEqual(self.request()['bundle_id'],request['bundle_id'])

    def test_invalid_dates_gps_duplicate_ids_and_unsupported_temporal_claims_are_rejected(self):
        cases=[([self.observation(observation_date='January 2027')],'YYYY-MM-DD'),
               ([self.observation(latitude='nan')],'range'),
               ([self.observation(longitude='121')],'outside'),
               ([self.observation(review_status='supports_map_window')],'entire mapped period'),
               ([self.observation(evidence_start='2025-07-01',evidence_end='2025-10-31',review_status='supports_map_window')],'field visit'),
               ([self.observation(),self.observation()],'unique ID')]
        for rows,message in cases:
            with self.subTest(message=message):
                self.write_observations(rows)
                with self.assertRaisesRegex(ValueError,message):self.request()
        # Explicit farmer recall can document the window, without manufacturing accuracy.
        self.write_observations([self.observation(evidence_type='farmer_recall',evidence_start='2025-07-01',
                                                evidence_end='2025-10-31',review_status='supports_map_window')])
        bundle=field_review.load_bundle(field_review.build(self.request()))
        self.assertEqual(bundle['observations'][0]['temporal_relation'],'covers_map_window')
        self.assertFalse(bundle['manifest']['accuracy_evaluated'])

    def test_new_notebook_runs_locally_and_pending_map_stops_report_fallback(self):
        (self.root/'pyproject.toml').write_text('[project]\nname="fixture"\n')
        (self.root/'src/crop_pipeline').mkdir(parents=True)
        (self.root/'settings.local.json').write_text(json.dumps({'report_root':'./inventory'}))
        paoay=self.root/'paoay'
        for name,path in [('maps/map_stage.json',self.map),('crop_baseline/crop_baseline_stage.json',self.baseline),('screening/screening_stage.json',self.flags)]:
            write_stage_pointer(paoay/name,path)
        notebook=json.loads((ROOT/'notebooks/paoay/05_reference_field_review.ipynb').read_text())
        def execute():
            namespace={}
            try:
                for cell in notebook['cells']:
                    if cell['cell_type']=='code':exec(compile(''.join(cell['source']),'field-review-notebook','exec'),namespace)
            finally:progress.finish_notebook()
            return namespace
        with chdir(self.root),redirect_stdout(io.StringIO()), \
             patch('socket.socket.connect',side_effect=AssertionError('No network')), \
             patch('joblib.load',side_effect=AssertionError('No model loading')):
            result=execute()
            self.assertEqual(resolve_stage_pointer(result['field_pointer']),result['field_handoff'])
            write_stage_pointer(paoay/'screening/screening_stage.json',status='pending',reason='New screening incomplete')
            with self.assertRaisesRegex(RuntimeError,'pending'):execute()
        for name in ('review/review_stage.json','field_review/field_review_stage.json'):
            with self.assertRaisesRegex(RuntimeError,'pending'):resolve_stage_pointer(paoay/name)


class FiveNotebookWorkflowTests(unittest.TestCase):
    def test_all_five_entry_points_chain_with_simulated_retrieval_and_fixed_model(self):
        import tempfile
        from types import SimpleNamespace
        from crop_pipeline.paoay import imagery,dynamic_world
        import test_paoay_predictions as models
        import test_paoay_imagery as satellite
        import test_paoay_dynamic_world as world
        import test_paoay_references as refs
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'pyproject.toml').write_text('[project]\nname="fixture"\n')
            (root/'src/crop_pipeline').mkdir(parents=True)
            (root/'settings.local.json').write_text(json.dumps({'earth_engine_project':'fixture-project'}))
            kmz=root/'data/paoay'
            refs.write_kmz(kmz/'00 - Region of Interest-2.kmz',[
                refs.placemark('Region of Interest',geometry=refs.polygon(120.7,16.6,120.703,16.603)),
                refs.placemark('Area 1','Crop rotation; year unconfirmed',geometry=refs.polygon(120.7,16.6,120.7015,16.603))])
            refs.write_kmz(kmz/'Crop Area.kmz',[
                refs.placemark('B2','Lettuce all year',geometry=refs.polygon(120.7015,16.6,120.703,16.603))])
            model=models.fitted_model(root)
            original_model={p:p.read_bytes() for p in model.parent.iterdir() if p.is_file()}
            fake_ee=SimpleNamespace(Initialize=lambda **kwargs:None,EEException=RuntimeError)
            names=['01_reference_review.ipynb','02_satellite_inputs.ipynb','03_model_application.ipynb',
                   '04_maps.ipynb','05_reference_field_review.ipynb']
            with chdir(root),redirect_stdout(io.StringIO()),patch.dict('sys.modules',{'ee':fake_ee}), \
                 patch('crop_pipeline.notebook_setup.configure_certificates'), \
                 patch('socket.socket.connect',side_effect=AssertionError('No network')), \
                 patch('sklearn.ensemble.RandomForestClassifier.fit',side_effect=AssertionError('No fitting')), \
                 patch.object(imagery.PaoayRetriever,'list_acquisitions',satellite.acquisitions), \
                 patch.object(imagery.PaoayRetriever,'find_cloud_probability_matches',lambda retriever,fragment:{r['image_index'] for r in satellite.acquisitions(retriever,fragment)}), \
                 patch.object(imagery.PaoayRetriever,'fetch_composite',satellite.fake_fetch), \
                 patch.object(dynamic_world.DynamicWorldRetriever,'list_acquisitions',world.world_acquisitions), \
                 patch.object(dynamic_world.DynamicWorldRetriever,'fetch_composite',world.world_fetch):
                for name in names:
                    notebook=json.loads((ROOT/'notebooks/paoay'/name).read_text());namespace={}
                    try:
                        for cell in notebook['cells']:
                            if cell['cell_type']=='code':
                                exec(compile(''.join(cell['source']),name,'exec'),namespace)
                                namespace['show_preview']=False
                                if name.startswith('02') and 'request' not in namespace:
                                    namespace.update(imagery_mode='download_missing',dynamic_world_mode='download_missing')
                                if name.startswith('03') and 'request' not in namespace:namespace['model_manifest']=str(model)
                    finally:progress.finish_notebook()
                final=field_review.load_bundle(resolve_stage_pointer(namespace['field_pointer']))
            self.assertEqual(final['manifest']['observation_count'],0)
            self.assertFalse(final['manifest']['accuracy_evaluated'])
            self.assertEqual({p:p.read_bytes() for p in original_model},original_model)
