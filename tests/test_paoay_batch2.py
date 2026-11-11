"""Combined application/map entry points preserve the model and keep screening reversible."""
from contextlib import chdir, redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from crop_pipeline import progress
from crop_pipeline.analysis_workflow import resolve_stage_pointer,write_stage_pointer
from crop_pipeline.paoay import maps,screening,crop_baseline,dynamic_world,land_cover,features
import test_paoay_final_workflow as fixture

ROOT=Path(__file__).resolve().parents[1]


class SpectralScreeningTests(unittest.TestCase):
    def test_temporal_screen_does_not_reject_a_field_with_bare_and_green_dates(self):
        names=['B3','B4','B5','B8','B11','B2','clear_count']
        cube=np.full((8,7,1,4),1000.,dtype=float);cube[:,-1]=1
        cube[:,names.index('B8'),0,1]=5000  # Green at every date.
        cube[-1,names.index('B8'),0,2]=5000  # Bare followed by green: no weak flag.
        cube[0,-1,0,3]=0  # Missing observation is not noncrop evidence.
        classes=np.array([[6,5,7,0]]);status=np.array([[1,1,1,4]])
        result=screening.diagnose(cube,names,np.ones((1,4),bool),classes,status,.5,screening.RULES)
        flags=result[-1].astype(int)
        np.testing.assert_array_equal(flags,[[1|4,8,32,16]])
        self.assertAlmostEqual(result[1,0,1],2/3,places=6)
        self.assertEqual(result[-2,0,3],7)
        np.testing.assert_array_equal(cube[:,-1,0,0],np.ones(8))

    def test_invalid_rule_values_are_rejected(self):
        for value in (float('nan'),2,True):
            with self.assertRaises(ValueError):
                screening.validate_rules({'maximum_ndvi_below':value})


class CombinedNotebookTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture.FinalWorkflowTests(methodName='runTest')
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.root=self.fixture.root
        (self.root/'pyproject.toml').write_text('[project]\nname="fixture"\n')
        (self.root/'src/crop_pipeline').mkdir(parents=True)
        (self.root/'settings.local.json').write_text(json.dumps({'report_root':'./inventory'}))
        self.paoay=self.root/'paoay'
        self.inputs=maps.context(self.fixture.prediction)
        guidance=self.inputs['guidance']['manifest']
        self.world=Path(guidance['dynamic_world_handoff'])
        world=dynamic_world.load_dynamic_world_bundle(self.world)
        self.image=Path(world['manifest']['imagery_handoff'])
        write_stage_pointer(self.paoay/'sentinel2/imagery_stage.json',self.image)
        write_stage_pointer(self.paoay/'dynamic_world/dynamic_world_stage.json',self.world)

    def execute(self,name):
        notebook=json.loads((ROOT/'notebooks/paoay'/name).read_text());namespace={}
        with chdir(self.root),redirect_stdout(io.StringIO()):
            try:
                for cell in notebook['cells']:
                    if cell['cell_type']=='code':
                        exec(compile(''.join(cell['source']),name,'exec'),namespace)
                        if 'model_manifest' in namespace and 'request' not in namespace:
                            namespace['model_manifest']=str(self.fixture.model)
                        namespace['show_preview']=False
            finally:progress.finish_notebook()
        return namespace

    def test_combined_notebooks_execute_locally_reuse_and_preserve_all_upstream_bytes(self):
        before=dict(self.fixture.upstream)
        with patch('socket.socket.connect',side_effect=AssertionError('No network')), \
             patch('sklearn.ensemble.RandomForestClassifier.fit',side_effect=AssertionError('No fitting')):
            application=self.execute('03_model_application.ipynb')
            self.assertEqual(resolve_stage_pointer(application['prediction_pointer']),application['prediction_handoff'])
            with patch('joblib.load',side_effect=AssertionError('Cached predictions must reuse')):
                again=self.execute('03_model_application.ipynb')
                result=self.execute('04_maps.ipynb')
                repeated=self.execute('04_maps.ipynb')
        self.assertEqual(application['prediction_handoff'],again['prediction_handoff'])
        self.assertEqual(result['screening_handoff'],repeated['screening_handoff'])
        self.assertEqual(resolve_stage_pointer(result['crop_pointer']),result['crop_baseline_handoff'])
        bundle=screening.load_bundle(result['screening_handoff'])
        baseline=crop_baseline.load_bundle(result['crop_baseline_handoff'])
        self.assertEqual(bundle['manifest']['policy'],'flags_only_no_hard_exclusions')
        self.assertFalse(bundle['manifest']['model_refitted'])
        self.assertEqual(bundle['data'].shape[0],18)
        np.testing.assert_array_equal(result['baseline_bundle']['data'],baseline['data'])
        self.assertTrue((bundle['data'][:,~baseline['roi']]==screening.NODATA).all())
        self.assertEqual({p:p.read_bytes() for p in before},before)
        (bundle['folder']/'boundary_flags.csv').write_text('corrupt')
        with self.assertRaisesRegex(ValueError,'changed'):
            screening.load_bundle(result['screening_handoff'])

    def test_zero_ready_features_stop_before_inference_and_leave_pointers_pending(self):
        pending = {'manifest': {'status':'pending_inputs', 'roi_pixels':1, 'row_count':0},
                   'coverage':[], 'audit':[], 'matrix':np.empty((0,120))}
        with patch.object(features,'build',return_value=self.root/'pending_feature_handoff.json'), \
             patch.object(features,'load_feature_bundle',return_value=pending), \
             patch('crop_pipeline.paoay.predictions.build',side_effect=AssertionError('No inference')):
            with self.assertRaisesRegex(RuntimeError,'No ready feature pixels'):
                self.execute('03_model_application.ipynb')
        for pointer in ('features/feature_stage.json','predictions/prediction_stage.json'):
            with self.assertRaisesRegex(RuntimeError,'pending'):
                resolve_stage_pointer(self.paoay/pointer)

    def test_pending_sources_block_new_predictions_and_maps(self):
        self.execute('03_model_application.ipynb')
        write_stage_pointer(self.paoay/'sentinel2/imagery_stage.json',status='pending',reason='New imagery incomplete')
        with self.assertRaisesRegex(RuntimeError,'pending'):
            self.execute('03_model_application.ipynb')
        for pointer in ('features/feature_stage.json','predictions/prediction_stage.json'):
            with self.assertRaisesRegex(RuntimeError,'pending'):
                resolve_stage_pointer(self.paoay/pointer)
        with self.assertRaisesRegex(RuntimeError,'pending'):
            self.execute('04_maps.ipynb')
        for pointer in ('maps/map_stage.json','screening/screening_stage.json','crop_baseline/crop_baseline_stage.json'):
            with self.assertRaisesRegex(RuntimeError,'pending'):
                resolve_stage_pointer(self.paoay/pointer)
