"""Freeze preliminary maps and keep field observations/correction proposals separate."""
import csv
from datetime import date
import json
from pathlib import Path
import shutil
import tempfile

from pyproj import Transformer
import rasterio
from shapely.geometry import Point, shape

from crop_pipeline.paoay import crop_baseline, imagery, maps, references, reference_suitability, review, screening

OBSERVATION_COLUMNS = ['observation_id', 'reference_id', 'observation_date', 'latitude', 'longitude',
                       'observed_crop', 'surface_state', 'crop_stage', 'evidence_type',
                       'evidence_start', 'evidence_end', 'evidence_reference', 'observer', 'review_status', 'notes']
CORRECTION_COLUMNS = ['correction_id', 'observation_id', 'proposed_label', 'decision', 'reason']
SURFACES = {'crop', 'bare', 'built', 'water', 'woody', 'mixed', 'unknown'}
EVIDENCE = {'field_visit', 'farmer_recall', 'historical_record'}
REVIEW_STATES = {'unreviewed', 'context_only', 'supports_map_window'}


def _read(path, columns):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError(f'Unexpected columns in {Path(path).name}; existing edits preserved')
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('Incomplete or extra CSV fields; existing edits preserved')
    return [{key:value.strip() for key,value in row.items()} for row in rows]


def _write(path, rows, columns, exclusive=False):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def prepare_records(reference_handoff, output_root):
    """One-time editable templates, with no predictions or invented field observations."""
    reference = references.load_reference_bundle(reference_handoff)
    folder = Path(output_root)/'records'/reference['manifest']['bundle_id']
    folder.mkdir(parents=True, exist_ok=True)
    observation_file, correction_file = folder/'observations.csv', folder/'corrections.csv'
    if not observation_file.exists():
        _write(observation_file, [{'reference_id':r['reference_id'], 'review_status':'unreviewed'}
               for r in reference['records'] if r['reference_id'] != reference['manifest']['roi_id']], OBSERVATION_COLUMNS, True)
    if not correction_file.exists():
        _write(correction_file, [], CORRECTION_COLUMNS, True)
    return observation_file, correction_file


def _date(text):
    try:
        value = date.fromisoformat(text)
        if value.isoformat() != text:
            raise ValueError('Use YYYY-MM-DD')
        return value
    except (TypeError, ValueError) as error:
        raise ValueError('Field dates must use YYYY-MM-DD') from error


def assess_observations(path, reference, baseline):
    known = {r['reference_id']:r for r in reference['records'] if r['reference_id'] != reference['manifest']['roi_id']}
    period = baseline['manifest']['period']
    start, end = _date(period['start']), _date(period['end'])  # end exclusive
    grid = baseline['manifest']['grid']
    project = Transformer.from_crs('EPSG:4326',grid['crs'],always_xy=True)
    records, summaries, ids = [], [], set()
    for row in _read(path,OBSERVATION_COLUMNS):
        active = any(row[k] for k in OBSERVATION_COLUMNS if k not in {'reference_id','review_status'})
        if not active:
            if row['reference_id'] not in known or row['review_status'] not in {'','unreviewed'}:
                raise ValueError('Blank field template needs a known reference and unreviewed status')
            continue
        identifier = row['observation_id']
        if not identifier or identifier in ids or row['reference_id'] not in known:
            raise ValueError('Each observation needs a unique ID and known reference')
        ids.add(identifier)
        observed = _date(row['observation_date'])
        if (row['surface_state'] not in SURFACES or row['evidence_type'] not in EVIDENCE
                or row['review_status'] not in REVIEW_STATES or not row['evidence_reference'] or not row['observer']
                or (row['surface_state']=='crop' and not row['observed_crop'])):
            raise ValueError('Record surface, crop when applicable, evidence type/reference, observer and review status')
        if bool(row['evidence_start']) != bool(row['evidence_end']):
            raise ValueError('Provide both evidence dates or leave both blank')
        first,last = (_date(row['evidence_start']),_date(row['evidence_end'])) if row['evidence_start'] else (observed,observed)
        if first > last:
            raise ValueError('Evidence start is after evidence end')
        relation = 'different_period' if last < start or first >= end else 'partial_window_overlap'
        if first <= start and last >= date.fromordinal(end.toordinal()-1):
            relation = 'covers_map_window'
        if row['review_status']=='supports_map_window' and (relation!='covers_map_window' or not row['evidence_start']):
            raise ValueError('Map-window support requires explicit evidence covering the entire mapped period')
        if row['evidence_type']=='field_visit' and row['evidence_start'] and (first!=observed or last!=observed):
            raise ValueError('A field visit documents its visit date; record historical evidence separately')
        if bool(row['latitude']) != bool(row['longitude']):
            raise ValueError('Provide both GPS coordinates or neither')
        location, prediction, pixel_status = 'not_located', '', ''
        if row['latitude']:
            try:
                latitude,longitude = float(row['latitude']),float(row['longitude'])
            except ValueError as error:
                raise ValueError('GPS coordinates must be numeric') from error
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                raise ValueError('GPS coordinates out of range')
            area = known[row['reference_id']]
            if area['geometry_status']!='valid' or not shape(area['geometry']).covers(Point(longitude,latitude)):
                raise ValueError('GPS point lies outside its named reference polygon')
            x,y=project.transform(longitude,latitude)
            r,c=rasterio.transform.rowcol(rasterio.Affine(*grid['transform']),x,y)
            if 0<=r<grid['height'] and 0<=c<grid['width'] and baseline['roi'][r,c]:
                location='mapped_pixel'
                code=int(baseline['data'][0,r,c]);status=int(baseline['data'][1,r,c])
                prediction=crop_baseline.CROPS.get(code,'')
                pixel_status=crop_baseline.STATUSES[status]
            else:
                location='outside_mapped_pixel_centres'
        records.append(row)
        summaries.append({'observation_id':identifier,'reference_id':row['reference_id'],
                          'name':known[row['reference_id']]['name'],'observation_date':row['observation_date'],
                          'observed_crop':row['observed_crop'],'surface_state':row['surface_state'],
                          'temporal_relation':relation,'review_status':row['review_status'],
                          'location_status':location,'preliminary_prediction':prediction,'preliminary_status':pixel_status,
                          'interpretation':'point/context record; no automatic correctness or whole-polygon label'})
    return records,summaries


def assess_corrections(path, observations):
    known={r['observation_id']:r for r in observations}; ids=set(); rows=[]
    for row in _read(path,CORRECTION_COLUMNS):
        if not any(row.values()): continue
        if (not row['correction_id'] or row['correction_id'] in ids or row['observation_id'] not in known
                or not row['proposed_label'] or not row['reason'] or row['decision'] not in {'proposed','reviewed','rejected'}):
            raise ValueError('Correction proposals need a unique ID, existing observation, label, decision and reason')
        ids.add(row['correction_id']);rows.append(row)
    return rows


def _sources(request, mutable=True):
    if imagery._digest(request['identity'])[:20] != request['bundle_id']:
        raise ValueError('Field review request changed')
    for key in ('baseline_handoff','screening_handoff','review_handoff'):
        if references.checksum(request[key])!=request['identity'][key+'_sha256']:
            raise ValueError('Frozen map/review dependency changed')
    baseline=crop_baseline.load_bundle(request['baseline_handoff'])
    flags=screening.load_bundle(request['screening_handoff'])
    context=review.load_review_bundle(request['review_handoff'])
    if (flags['manifest']['baseline_handoff']!=request['baseline_handoff']
            or flags['manifest']['map_handoff']!=context['manifest']['map_handoff']
            or context['manifest'].get('primary_crop_method')!='three_crop_with_unknown'
            or baseline['manifest']['identity']['rules']!=context['manifest']['identity']['crop_score_rules']):
        raise ValueError('Field review inputs do not refer to the same preliminary map and rules')
    inputs=maps.context(baseline['manifest']['prediction_handoff'])
    if mutable:
        for key in ('observation_file','correction_file','suitability_file'):
            if references.checksum(request[key])!=request['identity'][key+'_sha256']:
                raise ValueError('Editable records changed; prepare a new review snapshot')
    return baseline,flags,context,inputs


def prepare_request(baseline_handoff,screening_handoff,review_handoff,output_root,observation_file=None,correction_file=None):
    baseline=crop_baseline.load_bundle(baseline_handoff)
    inputs=maps.context(baseline['manifest']['prediction_handoff'])
    observation_template,correction_template=prepare_records(inputs['reference_path'],output_root)
    suitability_file,_=reference_suitability.prepare_review(inputs['reference_path'])
    paths={'baseline_handoff':baseline_handoff,'screening_handoff':screening_handoff,'review_handoff':review_handoff,
           'observation_file':observation_file or observation_template,'correction_file':correction_file or correction_template,
           'suitability_file':suitability_file}
    identity={'version':1,'implementation_sha256':references.checksum(__file__),
              **{key+'_sha256':references.checksum(path) for key,path in paths.items()}}
    key=imagery._digest(identity)[:20]
    request={'identity':identity,'bundle_id':key,**{key:str(Path(path).resolve()) for key,path in paths.items()},
             'folder':Path(output_root)/'bundles'/key}
    baseline,flags,context,inputs=_sources(request)
    records,_=assess_observations(request['observation_file'],inputs['reference'],baseline)
    assess_corrections(request['correction_file'],records)
    return request


def build(request, iterator=None):
    baseline,flags,context,inputs=_sources(request)
    folder=request['folder'];handoff=folder/'field_review_handoff.json'
    if folder.exists():
        load_bundle(handoff);return handoff
    observations,assessments=assess_observations(request['observation_file'],inputs['reference'],baseline)
    corrections=assess_corrections(request['correction_file'],observations)
    areas=[r for r in inputs['reference']['records'] if r['reference_id']!=inputs['reference']['manifest']['roi_id']]
    suitability=_read(request['suitability_file'],reference_suitability.COLUMNS)
    by_id={r['reference_id']:r for r in suitability}
    plan=[]
    for area in (iterator(areas) if iterator else areas):
        entry=by_id[area['reference_id']]
        plan.append({'reference_id':area['reference_id'],'name':area['name'],'area_type':entry['area_type'],
                     'temporal_evidence':entry['temporal_evidence'],'geometry_status':area['geometry_status'],
                     'visit_priority':'manual selection; include uncertain and accepted areas',
                     'record':'date; GPS; actual crop/cover; growth stage; photo/evidence; observer; rotation history',
                     'original_notes':area['notes']})
    folder.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.field-review-',dir=folder.parent) as temporary:
        staging=Path(temporary)/request['bundle_id'];staging.mkdir()
        copies={'preliminary_crop_map.tif':baseline['folder']/'crop_baseline.tif',
                'preliminary_crop_areas.csv':context['folder']/'crop_baseline_areas.csv',
                'screening_flags.csv':flags['folder']/'boundary_flags.csv',
                'calendar_context.csv':context['folder']/'calendar_context.csv',
                'calendar_snapshot.csv':context['folder']/'calendar_snapshot.csv',
                'reference_suitability_snapshot.csv':request['suitability_file'],
                'observations_snapshot.csv':request['observation_file'],'corrections_snapshot.csv':request['correction_file']}
        for name,path in copies.items():shutil.copyfile(path,staging/name)
        _write(staging/'field_check_plan.csv',plan,['reference_id','name','area_type','temporal_evidence','geometry_status','visit_priority','record','original_notes'])
        assessment_columns=['observation_id','reference_id','name','observation_date','observed_crop','surface_state','temporal_relation','review_status','location_status','preliminary_prediction','preliminary_status','interpretation']
        _write(staging/'observation_review.csv',assessments,assessment_columns)
        lines=['# Paoay reference and field-check review','',
               f"Imagery: {baseline['manifest']['period']['start']} to {baseline['manifest']['period']['end']} (end exclusive).",'',
               'The preliminary map is frozen here. Crop scores and screening flags are not ground truth. No original prediction is corrected or used as a training label.',
               f"Recorded observations: {len(observations)}. Correction proposals: {len(corrections)}. No corrected raster is created.",'',
               '## Predictions by boundary','',
               'Zero counts for invalid, outside or subpixel references do not establish crop absence. Overlapping boundaries are reviewed separately, not pooled as independent samples.','',
               '| Boundary | Carrot | Cabbage | Potato | Unknown | Missing |','| --- | ---: | ---: | ---: | ---: | ---: |']
        for area in areas:
            values={r['category']:float(r['roi_percent']) for r in context['crop_baseline'] if r['reference_id']==area['reference_id']}
            name=area['name'].replace('|','\\|').replace('\n',' ')
            labels=[*crop_baseline.CROPS.values(),crop_baseline.STATUSES[2],crop_baseline.STATUSES[3]]
            lines.append('| '+name+' | '+' | '.join(f"{values.get(label,0):.1f}%" for label in labels)+' |')
        lines+=['','## January field checks','',
                'Use field_check_plan.csv for area context; it intentionally omits predictions. Select visits across accepted and uncertain areas, not only suspected errors. This is a practical checklist, not a statistical sampling design.',
                'Record observations before viewing predictions where practical. A GPS point does not label the entire polygon. Mixed crops and seedbeds remain explicit.',
                'A January visit records January conditions. It cannot directly verify this historical map. Record farmer recall or historical documents separately with the period they support.',
                'Observation dates alone never establish four-month crop occupancy. supports_map_window requires explicit evidence covering the full mapped window and remains a manual evidence claim, not an accuracy result.',
                'Keep corrections as separate proposals with evidence. Neither a reviewed proposal nor agreement with a corrected product measures original-model accuracy. A later corrected raster needs explicit spatial scope and a separate output.',
                'No accuracy, confusion matrix or F1 is computed from these records.','',
                '## Files','',
                '- preliminary_crop_map.tif: frozen boundary-limited model map.',
                '- preliminary_crop_areas.csv, screening_flags.csv: crop summaries and overlapping review flags.',
                '- calendar_context.csv and calendar_snapshot.csv: unverified calendar context.',
                '- field_check_plan.csv: visit checklist without predicted labels.',
                '- observation_review.csv: dated evidence context and point-level preliminary predictions.',
                '- *_snapshot.csv: immutable copies; edit the separate source CSVs, then rerun 11.','',
                '## Original references','']
        for area in areas:
            lines += ['### '+area['name'].replace('\n',' '),'', *['> '+line for line in (area['notes'] or 'No original notes.').splitlines()],'']
        lines+=['## Deferred work','','- TODO: pooled other-crop model.','- TODO: individual-crop model with grouped reporting.','- Hard exclusions remain disabled pending screening validation.','']
        (staging/'report.md').write_text('\n'.join(lines),encoding='utf-8')
        _sources(request)
        manifest={'format_version':1,'kind':'paoay_field_review','status':'prepared',
                  **{key:request[key] for key in ('identity','bundle_id','baseline_handoff','screening_handoff','review_handoff','observation_file','correction_file','suitability_file')},
                  'period':baseline['manifest']['period'],'grid':baseline['manifest']['grid'],
                  'observation_count':len(observations),'correction_proposal_count':len(corrections),
                  'accuracy_evaluated':False,'corrected_map_created':False,'model_refitted':False,
                  'files':{path.name:{'sha256':references.checksum(path)} for path in staging.iterdir()}}
        (staging/'field_review_handoff.json').write_text(json.dumps(manifest,indent=2)+'\n');staging.rename(folder)
    load_bundle(handoff);return handoff


def load_bundle(path):
    path=Path(path);m=json.loads(path.read_text())
    expected={'preliminary_crop_map.tif','preliminary_crop_areas.csv','screening_flags.csv','calendar_context.csv','calendar_snapshot.csv',
              'reference_suitability_snapshot.csv','observations_snapshot.csv','corrections_snapshot.csv','field_check_plan.csv','observation_review.csv','report.md'}
    if (m.get('format_version')!=1 or m.get('kind')!='paoay_field_review' or m.get('status')!='prepared'
            or any(m.get(key) is not False for key in ('accuracy_evaluated','corrected_map_created','model_refitted'))
            or set(m.get('files',{}))!=expected):
        raise ValueError('Unsupported or incomplete field review')
    baseline,flags,context,inputs=_sources(m,mutable=False)
    if any(m[key]!=baseline['manifest'][key] for key in ('period','grid')):
        raise ValueError('Field review dates/grid changed')
    for name,record in m['files'].items():
        if references.checksum(path.parent/name)!=record['sha256']:
            raise ValueError('Saved field review changed; files preserved')
    for name,key in (('observations_snapshot.csv','observation_file'),('corrections_snapshot.csv','correction_file'),('reference_suitability_snapshot.csv','suitability_file')):
        if references.checksum(path.parent/name)!=m['identity'][key+'_sha256']:
            raise ValueError('Frozen editable-record snapshot differs from request')
    if references.checksum(path.parent/'preliminary_crop_map.tif')!=baseline['manifest']['files']['crop_baseline.tif']['sha256']:
        raise ValueError('Frozen preliminary map differs from model output')
    records,expected_assessment=assess_observations(path.parent/'observations_snapshot.csv',inputs['reference'],baseline)
    corrections=assess_corrections(path.parent/'corrections_snapshot.csv',records)
    if len(records)!=m['observation_count'] or len(corrections)!=m['correction_proposal_count']:
        raise ValueError('Field record counts differ from frozen snapshots')
    with (path.parent/'observation_review.csv').open(newline='') as stream:assessment=list(csv.DictReader(stream))
    if assessment!=expected_assessment:
        raise ValueError('Observation review differs from frozen evidence')
    return {'manifest':m,'folder':path.parent,'observations':assessment,'crop_areas':context['crop_baseline'],'screening':flags['rows']}
