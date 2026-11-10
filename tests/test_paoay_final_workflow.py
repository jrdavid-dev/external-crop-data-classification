"""Map assembly and notes review retain uncertainty, temporal context and raw outputs."""
from contextlib import chdir, redirect_stdout
from datetime import date, timedelta
import csv
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
from crop_pipeline.paoay import maps, review, features, predictions, references, crop_baseline
import test_paoay_predictions as prediction_fixtures
import test_paoay_features as feature_fixtures
import test_paoay_land_cover as cover_fixtures
import test_paoay_references as reference_fixtures

REPOSITORY = Path(__file__).resolve().parents[1]
ORIGINAL_WORLD_ACQUISITIONS = feature_fixtures.world.world_acquisitions


def two_dates(retriever):
    rows = ORIGINAL_WORLD_ACQUISITIONS(retriever)
    return rows + [{**r, 'date': (date.fromisoformat(r['date'])+timedelta(days=1)).isoformat(),
                    'image_index': r['image_index']+'_b', 'image_id': r['image_id']+'_b'} for r in rows]


def reference_input(root):
    root.mkdir(parents=True, exist_ok=True)
    p = reference_fixtures.polygon
    placemark = reference_fixtures.placemark
    hole = '<innerBoundaryIs><LinearRing><coordinates>120.7005,16.601 120.701,16.601 120.701,16.602 120.7005,16.602 120.7005,16.601</coordinates></LinearRing></innerBoundaryIs>'
    path = reference_fixtures.write_kmz(root/'references.kmz', [
        placemark('Region of Interest', geometry=p(120.7,16.6,120.703,16.603)),
        placemark('Area 2', 'Cabbage July-October. Multiple crop rotations.', geometry=p(120.7,16.6,120.7015,16.603).replace('</Polygon>',hole+'</Polygon>')),
        placemark('B2', 'Lettuce all year.', geometry=p(120.7015,16.6,120.703,16.603)),
        placemark('Mixed', 'Cabbage seedbeds year-round. Mixed crops.', geometry=p(120.7008,16.6006,120.7022,16.6024)),
        placemark('B1', 'Carrot March-June.', geometry=p(120.7002,16.6002,120.7008,16.6008)),
        placemark('Outside', 'Crop not confirmed.', geometry=p(120.71,16.61,120.72,16.62)),
        placemark('Invalid', 'Cabbage all year.', geometry='<Point><coordinates>120.701,16.601</coordinates></Point>'),
        placemark('Tiny', 'Unconfirmed.', geometry=p(120.700001,16.600001,120.700002,16.600002))])
    records, sources = references.read_inputs([path])
    return references.export_reference_bundle(records, sources, root/'reference_bundle')


class MapRuleTests(unittest.TestCase):
    def test_missing_water_bare_weak_crops_and_sources_stay_distinct(self):
        roi = np.ones((1,9), dtype=bool); roi[0,-1] = False
        top = np.array([[6,7,0,4,4,4,4,4,6]])
        state = np.array([[4,4,4,3,3,3,2,3,4]])
        crop = np.array([[1,1,1,2,0,3,1,2,1]])
        score = np.array([[.8,.8,.8,.8,-9999,.5,.9,.8,.9]])
        margin = np.array([[.4,.4,.4,.4,-9999,.05,.5,0,.5]])
        result = maps.combine(roi,top,state,crop,score,margin,maps.RULES)
        np.testing.assert_array_equal(result[0,0],[6,7,0,2,0,0,0,0,0])
        np.testing.assert_array_equal(result[1,0],[1,1,4,1,3,2,2,2,0])
        np.testing.assert_array_equal(result[2,0],[1,1,7,2,3,4,5,4,0])
        self.assertNotIn('Other crop', maps.CLASSES.values())
        self.assertEqual(len(maps.CLASSES),7)
        self.assertEqual(maps.CLASSES[4],'Other cultivated vegetation')
        self.assertEqual(maps.CLASSES[7],'Bare soil / fallow')
        self.assertFalse((result[0] == 4).any())

    def test_period_summary_requires_enough_support_and_preserves_temporal_conflict(self):
        roi = np.ones((1,3), dtype=bool)
        tops = np.full((8,1,3),4)
        states = np.full((8,1,3),3)
        tops[-1,0,1], states[-1,0,1] = 7,4
        states[3:,0,2] = 2
        top,state,count,conflict = maps.stable_cover(tops,states,roi,4)
        np.testing.assert_array_equal(count,[[8,8,3]])
        result = maps.combine(roi,top,state,np.ones((1,3)),np.full((1,3),.8),np.full((1,3),.4),maps.RULES,count,conflict)
        np.testing.assert_array_equal(result[0],[[1,0,0]])
        np.testing.assert_array_equal(result[1],[[1,2,2]])
        np.testing.assert_array_equal(result[2],[[2,6,5]])

    def test_all_dynamic_world_types_map_only_to_supported_thesis_classes(self):
        roi=np.ones((1,9),dtype=bool)
        top=np.arange(9).reshape(1,9)
        state=np.full((1,9),4);state[0,4]=3
        result=maps.combine(roi,top,state,np.ones((1,9)),np.full((1,9),.9),np.full((1,9),.8),maps.RULES)
        np.testing.assert_array_equal(result[0,0],[0,5,0,0,1,5,6,7,0])
        np.testing.assert_array_equal(result[1,0],[4,1,4,4,1,1,1,1,4])
        self.assertFalse((result[0] == 4).any())

    def test_no_observations_missing_crop_inputs_and_weak_evidence_are_separate(self):
        roi=np.ones((1,4),dtype=bool)
        top=np.full((8,1,4),4)
        state=np.broadcast_to(np.array([0,1,2,3]),(8,1,4)).copy()
        t,s,count,conflict=maps.stable_cover(top,state,roi,4)
        result=maps.combine(roi,t,s,np.zeros((1,4)),np.full((1,4),-9999),np.full((1,4),-9999),maps.RULES,count,conflict)
        np.testing.assert_array_equal(result[0],[[0,0,0,0]])
        np.testing.assert_array_equal(result[1],[[3,2,2,3]])
        np.testing.assert_array_equal(result[2],[[8,5,5,3]])

    def test_woody_group_agrees_across_trees_and_shrubs_without_false_temporal_conflict(self):
        roi=np.ones((1,1),dtype=bool)
        top=np.array([1,5]*4).reshape(8,1,1)
        state=np.full((8,1,1),4)
        t,s,count,conflict=maps.stable_cover(top,state,roi,4)
        self.assertFalse(conflict.any())
        result=maps.combine(roi,t,s,np.zeros((1,1)),np.zeros((1,1)),np.zeros((1,1)),maps.RULES,count,conflict)
        self.assertEqual(result[0,0,0],5)
        self.assertEqual(result[1,0,0],1)

    def test_invalid_rules_are_rejected(self):
        for changes in ({'minimum_supported_bins':0},{'minimum_supported_bins':True},{'minimum_crop_probability':np.nan},{'minimum_crop_margin':2}):
            with self.assertRaises(ValueError):
                maps.validate_rules({**maps.RULES,**changes})


class FinalWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        with patch.object(feature_fixtures.satellite,'reference_fixture',reference_input), \
             patch.object(feature_fixtures.world,'world_acquisitions',two_dates), \
             patch.object(feature_fixtures.world,'world_fetch',cover_fixtures.patterned_fetch):
            guidance = feature_fixtures.input_fixture(self.root)
        self.model = prediction_fixtures.fitted_model(self.root)
        feature_handoff = features.build(features.prepare_request(guidance,self.model,self.root/'features'))
        self.prediction = predictions.build(predictions.prepare_request(feature_handoff,self.root/'predictions'))
        self.upstream = {p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.map_request = maps.prepare_request(self.prediction,self.root/'maps')

    def test_crop_baseline_reuses_scores_without_model_loading_and_preserves_inputs(self):
        request = crop_baseline.prepare_request(self.prediction, self.root/'crop_baseline')
        with patch('socket.socket.connect', side_effect=AssertionError('No network')), \
             patch('joblib.load', side_effect=AssertionError('No model loading')):
            handoff = crop_baseline.build(request)
            bundle = crop_baseline.load_bundle(handoff)
            before = {p: p.read_bytes() for p in bundle['folder'].iterdir()}
            self.assertEqual(crop_baseline.build(request), handoff)
        self.assertEqual({p: p.read_bytes() for p in self.upstream}, self.upstream)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        inputs = maps.context(self.prediction)
        expected = np.zeros(inputs['roi'].shape, dtype=bool)
        for record in inputs['reference']['records']:
            if record['reference_id'] != inputs['reference']['manifest']['roi_id']:
                mask, _ = review.area_mask(record, bundle['manifest']['grid'], inputs['roi'])
                expected |= mask
        np.testing.assert_array_equal(bundle['roi'], expected)
        self.assertTrue((inputs['roi'] & ~expected).any())
        self.assertTrue((bundle['data'][:, ~expected] == 0).all())
        self.assertEqual(bundle['manifest']['spatial_scope'], 'reference_polygon_union')
        self.assertFalse(bundle['manifest']['model_refitted'])
        self.assertFalse(bundle['manifest']['land_cover_filter_applied'])
        self.assertEqual(sum(int(r['pixels']) for r in bundle['summary']), int(bundle['roi'].sum()))
        self.assertTrue((bundle['data'][1] == 2).any())  # Tied scores are Unknown.
        self.assertEqual(len(bundle['summary']), 5)
        handoff.parent.joinpath('crop_baseline_areas.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            crop_baseline.build(request)

    def make_review(self):
        self.map_handoff = maps.build(self.map_request)
        request = review.prepare_request(self.map_handoff,self.root/'review')
        return request,review.build(request)

    def test_maps_account_for_every_pixel_keep_conflicts_and_preserve_all_source_bytes(self):
        with patch('socket.socket.connect',side_effect=AssertionError('No network')), \
             patch('joblib.load',side_effect=AssertionError('No model loading')):
            handoff = maps.build(self.map_request)
            bundle = maps.load_map_bundle(handoff)
            before = {p:p.read_bytes() for p in bundle['folder'].rglob('*') if p.is_file()}
            self.assertEqual(maps.build(self.map_request),handoff)
        self.assertEqual({p:p.read_bytes() for p in self.upstream},self.upstream)
        self.assertEqual({p:p.read_bytes() for p in before},before)
        self.assertEqual(len(bundle['rasters']),9)
        for name in bundle['rasters']:
            classified = next(int(r['pixels']) for r in bundle['statuses'] if r['map']==name and int(r['code'])==1)
            self.assertEqual(sum(int(r['pixels']) for r in bundle['areas'] if r['map']==name),classified)
            self.assertEqual(sum(int(r['pixels']) for r in bundle['statuses'] if r['map']==name),bundle['manifest']['roi_pixels'])
            class_rows = [r for r in bundle['areas'] if r['map']==name]
            self.assertEqual(len(class_rows),7)
            other = next(r for r in class_rows if int(r['code'])==4)
            self.assertEqual(int(other['pixels']),0)
            self.assertIn('not_assessed',other['assignment_status'])
            self.assertEqual(sum(int(r['pixels']) for r in bundle['sources'] if r['map']==name),bundle['manifest']['roi_pixels'])
        with rasterio.open(bundle['rasters']['period_summary']) as raster:
            data = raster.read()
            roi = maps.context(self.prediction)['roi']
            np.testing.assert_array_equal(raster.read(4, masked=True).mask, ~roi)
        self.assertFalse(list(bundle['folder'].rglob('*.msk')))
        self.assertTrue((data[2]==6).any())
        self.assertTrue((data[1]==2).any())  # Tied crop scores remain unknown, not class 4.
        self.assertTrue((data[1]==4).any())  # Water is excluded, not woody vegetation.
        self.assertFalse((data[0]==4).any())
        self.assertTrue((data[0]==6).any())
        self.assertTrue((data[0]==7).any())

    def test_seven_class_loader_rejects_invented_other_cultivated_assignments(self):
        handoff=maps.build(self.map_request)
        path=handoff.parent/'rasters/period_summary.tif'
        with rasterio.open(path,'r+') as raster:
            data=raster.read()
            location=tuple(np.argwhere(data[1] == 1)[0])
            data[0][location]=4
            raster.write(data)
        manifest=json.loads(handoff.read_text())
        manifest['files']['rasters/period_summary.tif']['sha256']=references.checksum(path)
        handoff.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'statuses are inconsistent'):
            maps.load_map_bundle(handoff)

    def test_legacy_map_and_report_remain_readable_and_never_get_relabelled(self):
        from affine import Affine
        from crop_pipeline.paoay import imagery
        inputs=maps.context(self.prediction)
        roi=inputs['roi'];grid=inputs['crop']['manifest']['grid'];period=inputs['crop']['manifest']['period']
        identity={'version':1,'prediction_sha256':references.checksum(self.prediction),'rules':maps.RULES,'implementation_sha256':'historical_fixture'}
        key=imagery._digest(identity)[:20]
        folder=self.root/'legacy_maps'/key;(folder/'rasters').mkdir(parents=True)
        names=[r['name'] for r in period['intervals']]+['period_summary']
        for name in names:
            values=np.zeros((3,*roi.shape),dtype='uint8');values[0,roi]=4;values[1,roi]=4;values[2,roi]=8
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),rasterio.open(folder/'rasters'/f'{name}.tif','w',driver='GTiff',
                    height=grid['height'],width=grid['width'],count=3,dtype='uint8',crs=grid['crs'],transform=Affine(*grid['transform']),nodata=0) as raster:
                raster.write(values);raster.descriptions=tuple(maps.LEGACY_BANDS);raster.write_mask(roi.astype('uint8')*255)
        maps._write_table(folder/'class_areas.csv',[{'map':n,'class':'Cropland: crop type unresolved','code':4,'pixels':int(roi.sum())} for n in names])
        maps._write_table(folder/'sources.csv',[{'map':n,'source':'Dynamic World crops; crop score unresolved','code':4,'pixels':int(roi.sum())} for n in names])
        manifest={'format_version':1,'kind':'paoay_combined_maps','status':'prepared','identity':identity,'bundle_id':key,
            'prediction_handoff':str(self.prediction),'period':period,'grid':grid,'bands':maps.LEGACY_BANDS,
            'class_codes':maps.LEGACY_CLASSES,'source_codes':maps.LEGACY_SOURCES,'raw_crop_predictions_preserved':True,
            'roi_pixels':int(roi.sum()),'pixel_area_ha':.01,
            'files':{p.relative_to(folder).as_posix():{'sha256':references.checksum(p)} for p in folder.rglob('*') if p.is_file()}}
        legacy=folder/'map_handoff.json';legacy.write_text(json.dumps(manifest))
        before={p:p.read_bytes() for p in folder.rglob('*') if p.is_file()}
        loaded=maps.load_map_bundle(legacy)
        self.assertEqual(loaded['manifest']['class_codes']['4'],'Cropland: crop type unresolved')
        with self.assertRaisesRegex(ValueError,'Run notebook 13'):
            review.prepare_request(legacy,self.root/'review')
        calendar=self.root/'legacy_calendar.csv';calendar.write_text('reference_id,crop,months,scope,source_excerpt\n')
        ident={'version':1,'map_sha256':references.checksum(legacy),'calendar_sha256':references.checksum(calendar),'implementation_sha256':'historical_review'}
        old=self.root/'legacy_review';old.mkdir()
        for name in ('area_overview.csv','area_composition.csv','raw_crop_areas.csv','calendar_context.csv'):
            (old/name).write_text('reference_id,name\n')
        (old/'calendar_snapshot.csv').write_bytes(calendar.read_bytes());(old/'report.md').write_text('Historical review; unverified notes only.\n')
        old_manifest={'format_version':1,'kind':'paoay_result_review','status':'prepared','identity':ident,'bundle_id':imagery._digest(ident)[:20],
            'map_handoff':str(legacy),'calendar_file':str(calendar),'period':period,'grid':grid,'area_count':7,
            'independent_accuracy_evaluated':False,'evidence_policy':'unverified_notes_and_explicit_calendar_context_only',
            'files':{p.name:{'sha256':references.checksum(p)} for p in old.iterdir()}}
        old_handoff=old/'review_handoff.json';old_handoff.write_text(json.dumps(old_manifest))
        old_bytes={p:p.read_bytes() for p in old.iterdir()}
        self.assertEqual(review.load_review_bundle(old_handoff)['manifest']['format_version'],1)
        modern=maps.build(self.map_request)
        self.assertEqual(maps.load_map_bundle(modern)['manifest']['format_version'],2)
        self.assertEqual({p:p.read_bytes() for p in before},before)
        self.assertEqual({p:p.read_bytes() for p in old_bytes},old_bytes)

    def test_rule_changes_make_separate_maps_and_failed_or_corrupt_results_are_preserved(self):
        handoff = maps.build(self.map_request)
        strict = maps.prepare_request(self.prediction,self.root/'maps',{**maps.RULES,'minimum_supported_bins':8})
        with patch.object(maps,'combine',side_effect=OSError('Interrupted')),self.assertRaises(OSError):
            maps.build(strict)
        self.assertFalse(strict['folder'].exists())
        maps.load_map_bundle(handoff)
        self.assertNotEqual(maps.build(strict),handoff)
        raster = handoff.parent/'rasters/period_summary.tif'; raster.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Saved map changed'):
            maps.build(self.map_request)
        self.assertEqual(raster.read_bytes(),b'changed')

    def test_default_review_keeps_original_notes_unknowns_holes_and_overlapping_areas(self):
        request,handoff = self.make_review()
        bundle = review.load_review_bundle(handoff)
        self.assertFalse(bundle['manifest']['independent_accuracy_evaluated'])
        self.assertEqual(bundle['manifest']['area_count'],7)
        self.assertTrue(all(r['status']=='manual_review_required' for r in bundle['calendar']))
        overview = {r['name']:r for r in bundle['overview']}
        self.assertEqual(overview['Invalid']['status'],'invalid_geometry')
        self.assertEqual(overview['Outside']['status'],'no_roi_pixel_centres')
        self.assertEqual(overview['Tiny']['status'],'no_roi_pixel_centres')
        self.assertIn('seedbeds',overview['Mixed']['original_notes'])
        for row in bundle['overview']:
            rows = [r for r in bundle['composition'] if r['reference_id']==row['reference_id'] and r['map']=='period_summary']
            self.assertEqual(sum(int(r['pixels']) for r in rows),int(row['classified_pixels']))
            states = [r for r in bundle['statuses'] if r['reference_id']==row['reference_id'] and r['map']=='period_summary']
            self.assertEqual(sum(int(r['pixels']) for r in states),int(row['roi_pixels']))
            raw = [r for r in bundle['raw_areas'] if r['reference_id']==row['reference_id']]
            self.assertEqual(sum(int(r['pixels']) for r in raw),int(row['roi_pixels']))
            self.assertAlmostEqual(float(row['mapped_area_ha']),int(row['roi_pixels'])*.01)
        context = maps.context(self.prediction)
        area = next(r for r in context['reference']['records'] if r['name']=='Area 2')
        with_hole,_ = review.area_mask(area,context['crop']['manifest']['grid'],context['roi'])
        solid = dict(area); solid['geometry'] = {**area['geometry'],'coordinates':area['geometry']['coordinates'][:1]}
        without_hole,_ = review.area_mask(solid,context['crop']['manifest']['grid'],context['roi'])
        self.assertLess(with_hole.sum(),without_hole.sum())
        text = (bundle['folder']/'report.md').read_text()
        self.assertIn('does not measure independent local accuracy',text)
        self.assertIn('overlapping polygons',text)
        self.assertEqual({p:p.read_bytes() for p in self.upstream},self.upstream)

    def fill_calendar(self,path):
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            values = {'Area 2':('cabbage','7;8;9;10','crop_calendar','Cabbage July-October'),
                      'B1':('carrot','3;4;5;6','crop_calendar','Carrot March-June'),
                      'B2':('lettuce','1;2;3;4;5;6;7;8;9;10;11;12','crop_calendar','Lettuce all year'),
                      'Mixed':('cabbage','1;2;3;4;5;6;7;8;9;10;11;12','seedbed','Cabbage seedbeds year-round')}.get(row['name'])
            if values:
                row.update(zip(('crop','months','scope','source_excerpt'),values))
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=review.CALENDAR_COLUMNS);writer.writeheader();writer.writerows(rows)

    def test_explicit_calendar_interpretations_are_context_only_and_edits_create_new_snapshots(self):
        request,old_handoff = self.make_review()
        old_bytes = old_handoff.read_bytes()
        self.fill_calendar(Path(request['calendar_file']))
        edited = Path(request['calendar_file']).read_bytes()
        with self.assertRaisesRegex(ValueError,'Calendar input changed'):
            review.build(request)
        new_request = review.prepare_request(self.map_handoff,self.root/'review')
        handoff = review.build(new_request)
        self.assertNotEqual(handoff,old_handoff)
        self.assertEqual(old_handoff.read_bytes(),old_bytes)
        review.load_review_bundle(old_handoff)
        bundle=review.load_review_bundle(handoff)
        statuses={r['name']:r['status'] for r in bundle['calendar']}
        self.assertEqual(statuses['Area 2'],'calendar_overlaps_window')
        self.assertEqual(statuses['B1'],'calendar_outside_window')
        self.assertEqual(statuses['B2'],'crop_not_modelled')
        self.assertEqual(statuses['Mixed'],'context_only_seedbed')
        self.assertFalse(bundle['manifest']['independent_accuracy_evaluated'])
        self.assertEqual(Path(new_request['calendar_file']).read_bytes(),edited)

    def test_invalid_calendar_rows_are_rejected_without_guessing_or_overwriting(self):
        request,handoff = self.make_review()
        path=Path(request['calendar_file'])
        self.fill_calendar(path)
        original=path.read_text()
        for field,value in [('months','13'),('months','7;7'),('months','July-October'),('scope','verified_ground_truth'),('source_excerpt','invented observation')]:
            with self.subTest(field=field,value=value):
                path.write_text(original)
                with path.open(newline='') as stream:
                    rows=list(csv.DictReader(stream))
                target=next(r for r in rows if r['name']=='Area 2');target[field]=value
                with path.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=review.CALENDAR_COLUMNS);writer.writeheader();writer.writerows(rows)
                with self.assertRaises(ValueError):
                    review.prepare_request(self.map_handoff,self.root/'review')
        review.load_review_bundle(handoff)  # Immutable report survives edits to optional annotations.

    def test_failed_review_or_changed_snapshot_is_not_published_or_replaced(self):
        self.map_handoff=maps.build(self.map_request)
        request=review.prepare_request(self.map_handoff,self.root/'review')
        with patch.object(review,'area_mask',side_effect=OSError('Interrupted')),self.assertRaises(OSError):
            review.build(request)
        self.assertFalse(request['folder'].exists())
        handoff=review.build(request)
        report=handoff.parent/'report.md';report.write_text('changed')
        with self.assertRaisesRegex(ValueError,'Saved review changed'):
            review.build(request)
        self.assertEqual(report.read_text(),'changed')

    def test_both_notebooks_run_all_locally_and_pending_inputs_block_fallback(self):
        import matplotlib.pyplot as plt
        map_bundle=maps.load_map_bundle(maps.build(self.map_request))
        figure=maps.plot_maps(map_bundle)
        self.assertEqual(len(figure.axes),9)
        self.assertEqual(len(figure.legends),2)
        self.assertEqual(len(figure.legends[0].get_texts()),7)
        self.assertEqual([t.get_text() for t in figure.legends[1].get_texts()],['Unknown','Missing data','Excluded cover'])
        figure.savefig(self.root/'preview.png');plt.close(figure)
        root=self.root
        (root/'pyproject.toml').write_text('[project]\nname="fixture"\n')
        (root/'src/crop_pipeline').mkdir(parents=True)
        (root/'settings.local.json').write_text(json.dumps({'report_root':'./inventory'}))
        paoay=root/'paoay'
        prediction_pointer=paoay/'predictions/prediction_stage.json'
        write_stage_pointer(prediction_pointer,self.prediction)
        def run(name):
            notebook=json.loads((REPOSITORY/'notebooks/paoay/archive/eight_notebook_workflow'/name).read_text())
            namespace={}
            try:
                for cell in notebook['cells']:
                    if cell['cell_type']=='code':
                        exec(compile(''.join(cell['source']),name,'exec'),namespace)
                        namespace['show_preview']=False
            finally:
                progress.finish_notebook()
            return namespace
        with chdir(root),redirect_stdout(io.StringIO()), \
             patch('socket.socket.connect',side_effect=AssertionError('No network')), \
             patch('joblib.load',side_effect=AssertionError('No model loading')), \
             patch('sklearn.ensemble.RandomForestClassifier.fit',side_effect=AssertionError('No fitting')):
            first=run('13_paoay_maps.ipynb')
            second=run('14_paoay_review.ipynb')
            self.assertEqual(resolve_stage_pointer(first['map_pointer']),first['map_handoff'])
            self.assertEqual(resolve_stage_pointer(second['review_pointer']),second['review_handoff'])
            write_stage_pointer(prediction_pointer,status='pending',reason='Notebook 12 incomplete')
            with self.assertRaisesRegex(RuntimeError,'pending'):
                run('13_paoay_maps.ipynb')
            with self.assertRaisesRegex(RuntimeError,'pending'):
                run('14_paoay_review.ipynb')
        with self.assertRaisesRegex(RuntimeError,'pending'):
            resolve_stage_pointer(second['review_pointer'])


if __name__=='__main__':
    unittest.main()
