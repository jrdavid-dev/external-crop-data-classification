"""Area/calendar feasibility review; unverified KMZ notes never become test labels."""

import csv
import io
import json
from pathlib import Path
import tempfile

from affine import Affine
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.features import geometry_mask
from shapely import transform
from shapely.geometry import shape, mapping

from crop_pipeline.paoay import maps, imagery, references, crop_baseline

VERSION = 4
CALENDAR_COLUMNS = ['reference_id', 'name', 'crop', 'months', 'scope', 'source_excerpt', 'review_notes', 'original_notes']
SCOPES = {'crop_calendar', 'mixed_area', 'seedbed', 'unconfirmed'}


def _calendar_rows(path, records):
    known = {r['reference_id']: r for r in records}
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'reference_id','crop','months','scope','source_excerpt'}.issubset(reader.fieldnames or []):
            raise ValueError('Calendar CSV needs reference_id, crop, months, scope and source_excerpt')
        rows = list(reader)
    parsed, keys = [], set()
    normalize = lambda text: ' '.join(text.casefold().split())
    for row in rows:
        if not any((row.get(key) or '').strip() for key in ('crop','months','scope','source_excerpt')):
            continue
        key, crop = row['reference_id'].strip(), row['crop'].strip().casefold()
        if key not in known or not crop or row['scope'].strip() not in SCOPES or not row['source_excerpt'].strip():
            raise ValueError('Calendar entry needs a known area, crop, documented scope and source excerpt')
        try:
            months = [int(value.strip()) for value in row['months'].split(';')]
        except ValueError as error:
            raise ValueError('Calendar months must be semicolon-separated numbers 1–12') from error
        if not months or len(set(months)) != len(months) or any(not 1 <= month <= 12 for month in months):
            raise ValueError('Calendar months must be unique numbers 1–12')
        if normalize(row['source_excerpt']) not in normalize(known[key]['notes']):
            raise ValueError('Calendar source excerpt must occur in the original KMZ notes')
        if (key, crop) in keys:
            raise ValueError('Combine repeated area/crop calendars into one month list')
        keys.add((key, crop))
        parsed.append({'reference_id': key, 'crop': crop, 'months': sorted(months), 'scope': row['scope'].strip(),
                       'source_excerpt': row['source_excerpt'], 'review_notes': row.get('review_notes', '')})
    return parsed


def _context(map_handoff):
    bundle = maps.load_map_bundle(map_handoff)
    inputs = maps.context(bundle['manifest']['prediction_handoff'])
    areas = [r for r in inputs['reference']['records'] if r['reference_id'] != inputs['reference']['manifest']['roi_id']]
    return bundle, inputs, areas


def prepare_request(map_handoff, output_root, calendar_file=None):
    bundle, inputs, areas = _context(map_handoff)
    if bundle['manifest']['format_version'] != maps.VERSION:
        raise ValueError('Run notebook 13 to create seven-class maps before preparing a new review')
    output_root = Path(output_root)
    template = output_root/'calendars'/(references.checksum(inputs['reference_path'])[:20]+'.csv')
    if not template.exists():
        template.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=CALENDAR_COLUMNS); writer.writeheader()
        writer.writerows({'reference_id': r['reference_id'], 'name': r['name'], 'original_notes': r['notes']} for r in areas)
        with template.open('x', newline='', encoding='utf-8') as stream:
            stream.write(buffer.getvalue())
    path = Path(calendar_file) if calendar_file is not None else template
    before = references.checksum(path)
    _calendar_rows(path, areas)
    if references.checksum(path) != before:
        raise ValueError('Calendar file changed while being read')
    identity = {'version': VERSION, 'map_sha256': references.checksum(map_handoff), 'calendar_sha256': before,
                'implementation_sha256': references.checksum(__file__),
                'crop_spatial_scope': 'reference_polygon_union',
                'crop_score_rules': {'minimum_top_probability': bundle['manifest']['identity']['rules']['minimum_crop_probability'],
                                     'minimum_margin': bundle['manifest']['identity']['rules']['minimum_crop_margin']}, 'crop_baseline_sha256': references.checksum(crop_baseline.__file__)}
    key = imagery._digest(identity)[:20]
    return {'bundle_id': key, 'identity': identity, 'map_handoff': str(Path(map_handoff).resolve()),
            'calendar_file': str(path.resolve()), 'calendar_template': template,
            'folder': output_root/'bundles'/key}


def area_mask(record, grid, roi):
    if record['geometry_status'] != 'valid' or record['geometry'] is None:
        return np.zeros(roi.shape, dtype=bool), 'invalid_geometry'
    project = Transformer.from_crs('EPSG:4326', grid['crs'], always_xy=True).transform
    geometry = transform(shape(record['geometry']), project, interleaved=False)
    mask = geometry_mask([mapping(geometry)], out_shape=roi.shape, transform=Affine(*grid['transform']), invert=True) & roi
    return mask, 'included' if mask.any() else 'no_roi_pixel_centres'


def _verify(request, check_calendar=True):
    if imagery._digest(request['identity'])[:20] != request['bundle_id']:
        raise ValueError('Review request changed after planning')
    if references.checksum(request['map_handoff']) != request['identity']['map_sha256']:
        raise ValueError('Combined-map dependency changed')
    if check_calendar and references.checksum(request['calendar_file']) != request['identity']['calendar_sha256']:
        raise ValueError('Calendar input changed; prepare a new review request')
    return _context(request['map_handoff'])


def _write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def build(request, iterator=None):
    bundle, inputs, areas = _verify(request)
    folder, handoff = request['folder'], request['folder']/'review_handoff.json'
    if folder.exists():
        load_review_bundle(handoff)
        return handoff
    if bundle['manifest']['format_version'] != maps.VERSION or request['identity']['version'] != VERSION:
        raise ValueError('Replan the review using notebook 13 seven-class maps')
    calendars = _calendar_rows(request['calendar_file'], areas)
    roi, grid = inputs['roi'], bundle['manifest']['grid']
    primary_mask = crop_baseline.reference_mask(inputs['reference'], grid, roi)
    primary, _ = crop_baseline.arrays(inputs['crop'], inputs['feature'], request['identity']['crop_score_rules'], primary_mask)
    pixel_ha = bundle['manifest']['pixel_area_ha']
    crop = inputs['crop']; crop_manifest = crop['manifest']
    raw_codes = np.zeros(roi.shape, dtype='uint8')
    raw_codes[inputs['feature']['rows'], inputs['feature']['columns']] = crop['class_codes']
    matrices, map_statuses = {}, {}
    for name, path in bundle['rasters'].items():
        with rasterio.open(path) as raster:
            matrices[name] = raster.read(1)
            map_statuses[name] = raster.read(2)
    overview, compositions, comparisons, raw_areas, status_areas, baseline_areas = [], [], [], [], [], []
    window_months = {int(item['start'][5:7]) for item in bundle['manifest']['period']['intervals']}
    items = iterator(areas) if iterator else areas
    for record in items:
        mask, status = area_mask(record, grid, roi)
        total = int(mask.sum()); ready = int((mask & (raw_codes > 0)).sum())
        overview.append({'reference_id': record['reference_id'], 'name': record['name'], 'status': status,
            'roi_pixels': total, 'mapped_area_ha': total*pixel_ha, 'raw_crop_pixels': ready,
            'raw_crop_roi_percent': ready/total*100 if total else 0, 'original_notes': record['notes'],
            'evidence_status': 'unverified_calendar_context_not_ground_truth',
            **{label.lower().replace(' ', '_')+'_pixels': int((mask & (map_statuses['period_summary'] == code)).sum())
               for code, label in maps.STATUSES.items() if code}})
        baseline_areas.extend({'reference_id': record['reference_id'], 'name': record['name'], **r}
                              for r in crop_baseline.area_rows(primary, mask, pixel_ha))
        for target in list(crop_manifest['class_codes'])+['Missing crop input']:
            code = crop_manifest['class_codes'].get(target, 0)
            count = int((mask & (raw_codes == code)).sum())
            raw_areas.append({'reference_id': record['reference_id'], 'name': record['name'], 'category': target,
                'pixels': count, 'area_ha': count*pixel_ha, 'area_roi_percent': count/total*100 if total else 0,
                'predicted_percent': count/ready*100 if ready and code else ''})
        for name, data in matrices.items():
            for code, label in maps.CLASSES.items():
                if code:
                    count = int((mask & (data == code)).sum())
                    compositions.append({'reference_id': record['reference_id'], 'name': record['name'], 'map': name,
                        'category': label, 'pixels': count, 'area_ha': count*pixel_ha, 'area_roi_percent': count/total*100 if total else 0,
                        'assignment_status': maps.CLASS_ASSIGNMENT[code]})
            for code, label in maps.STATUSES.items():
                if code:
                    count = int((mask & (map_statuses[name] == code)).sum())
                    status_areas.append({'reference_id': record['reference_id'], 'name': record['name'], 'map': name,
                        'status': label, 'pixels': count, 'area_ha': count*pixel_ha, 'area_roi_percent': count/total*100 if total else 0})
        entries = [r for r in calendars if r['reference_id'] == record['reference_id']]
        if not entries:
            comparisons.append({'reference_id': record['reference_id'], 'name': record['name'], 'crop': '',
                'calendar_months': '', 'months_overlapping_window': '', 'scope': '', 'source_excerpt': '',
                'status': 'manual_review_required', 'window_crop_roi_percent': '', 'window_crop_predicted_percent': '',
                'period_summary_crop_roi_percent': '', 'review_notes': ''})
        for entry in entries:
            target = entry['crop']; overlaps = sorted(window_months & set(entry['months']))
            if entry['scope'] != 'crop_calendar':
                decision = 'context_only_'+entry['scope']
            elif target not in crop_manifest['class_codes']:
                decision = 'crop_not_modelled'
            elif not total:
                decision = status
            else:
                decision = 'calendar_overlaps_window' if overlaps else 'calendar_outside_window'
            raw_count = int((mask & (raw_codes == crop_manifest['class_codes'].get(target, -1))).sum())
            assembled_code = {'carrot': 1, 'cabbage': 2, 'potato': 3}.get(target, -1)
            summary_count = int((mask & (matrices['period_summary'] == assembled_code)).sum())
            comparisons.append({'reference_id': record['reference_id'], 'name': record['name'], 'crop': target,
                'calendar_months': ';'.join(map(str, entry['months'])), 'months_overlapping_window': ';'.join(map(str, overlaps)),
                'scope': entry['scope'], 'source_excerpt': entry['source_excerpt'], 'status': decision,
                'window_crop_roi_percent': raw_count/total*100 if total and target in crop_manifest['class_codes'] else '',
                'window_crop_predicted_percent': raw_count/ready*100 if ready and target in crop_manifest['class_codes'] else '',
                'period_summary_crop_roi_percent': summary_count/total*100 if total and assembled_code > 0 else '',
                'review_notes': entry['review_notes']})
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.review-', dir=folder.parent) as temporary:
        staging = Path(temporary)/request['bundle_id']; staging.mkdir()
        _write_csv(staging/'area_overview.csv', overview, ['reference_id','name','status','roi_pixels','mapped_area_ha','raw_crop_pixels','raw_crop_roi_percent','original_notes','evidence_status','classified_pixels','unknown_pixels','missing_data_pixels','excluded_cover_pixels'])
        _write_csv(staging/'area_composition.csv', compositions, ['reference_id','name','map','category','pixels','area_ha','area_roi_percent','assignment_status'])
        _write_csv(staging/'crop_baseline_areas.csv', baseline_areas, ['reference_id','name','category','kind','pixels','area_ha','roi_percent'])
        _write_csv(staging/'area_statuses.csv', status_areas, ['reference_id','name','map','status','pixels','area_ha','area_roi_percent'])
        _write_csv(staging/'calendar_context.csv', comparisons, ['reference_id','name','crop','calendar_months','months_overlapping_window','scope','source_excerpt','status','window_crop_roi_percent','window_crop_predicted_percent','period_summary_crop_roi_percent','review_notes'])
        _write_csv(staging/'raw_crop_areas.csv', raw_areas, ['reference_id','name','category','pixels','area_ha','area_roi_percent','predicted_percent'])
        (staging/'calendar_snapshot.csv').write_bytes(Path(request['calendar_file']).read_bytes())
        summary_lines = ['# Paoay mapping feasibility review', '',
            f"Period: {bundle['manifest']['period']['start']} to {bundle['manifest']['period']['end']} (end exclusive).", '',
            '## Interpretation', '', 'This report describes model predictions, coverage and calendar context. It does not measure independent local accuracy.',
            'The KMZ notes do not confirm which crop occupied each polygon in this historical year. No confusion matrix, accuracy, precision, recall or F1 is computed.',
            'Crop labels use the entire four-month window; half-month maps use different broad-cover contexts, not new dated crop observations.',
            'Primary crop result: the current three-crop model with Unknown crop type and separate missing input. No other-crop training is implemented.',
            f"Crop score cutoffs: {request['identity']['crop_score_rules']['minimum_top_probability']:.2f} top probability and {request['identity']['crop_score_rules']['minimum_margin']:.2f} margin; ties are Unknown. Scores are not calibrated local correctness probabilities.",
            'The primary crop map is restricted to the union of reference polygons inside the ROI; outside is masked, not Unknown.',
            'The primary crop view is conditional on the three trained crops and does not screen non-crop land. Treat it separately from the conservative Dynamic World context map.',
            'The secondary context map retains seven thesis classes with separate unknown, missing and excluded statuses.',
            'Other cultivated vegetation is not assessed by these inputs. Its unassigned area must not be interpreted as absence.',
            'Trees/shrubs form forest or woody vegetation. Supported water, grass, flooded vegetation and snow/ice remain excluded cover.',
            'Broad-cover and crop score thresholds are illustrative. Bare supports the bare-soil/fallow group but does not confirm fallow.', '',
            '## Coverage and sources', '', f"Study ROI pixel centres: {int(roi.sum())}.",
            f"Raw crop predictions: {crop_manifest['row_count']}; missing crop inputs: {int(roi.sum())-crop_manifest['row_count']}.",
            f"Reference areas: {len(areas)}. Overlapping area pairs in reference audit: {len(inputs['reference']['overlaps'])}.",
            'Areas are summarized independently, clipped to the ROI; overlapping polygons are not independent samples or pooled into a test score.',
            'Pixel-centre area estimates can differ from surveyed parcel area. Small, invalid or outside polygons remain explicitly flagged.', '',
            '## Review tables', '', '- crop_baseline_areas.csv: PRIMARY three-crop + Unknown area summaries; missing inputs stay separate.',
            '- area_overview.csv: coverage and original notes.',
            '- raw_crop_areas.csv: unfiltered three-crop predictions and missing-input areas.',
            '- area_composition.csv: seven thesis class areas, with assignment availability.',
            '- area_statuses.csv: classified, unknown, missing-data and excluded-cover areas, separately for each map.',
            '- calendar_context.csv: optional manually entered calendar expectations beside window-level predictions.',
            '- calendar_snapshot.csv: exact calendar input; editable source remains separate.', '',
            'Calendar overlap describes only time-window compatibility; it is not a correctness score. Missing expectations stay manual_review_required.', '',
            '## Original area notes', '']
        for row in overview:
            summary_lines += [f"### {row['name']}", '', f"Status: {row['status']}; ROI pixels: {row['roi_pixels']}; raw crop inputs: {row['raw_crop_roi_percent']:.1f}%.", '',
                              *['> '+line for line in (row['original_notes'] or 'No source notes.').splitlines()], '']
        summary_lines += ['## TODO: deferred training comparisons', '',
            '- [ ] Train a separate four-class model pooling selected, explicitly labelled non-target crops.',
            '- [ ] Train individual crop labels, then group known non-target predictions for reporting.',
            'Both require balanced crop/source sampling, preserved parcel splits and crop-species holdout checks. Neither is implemented or launched. Unknown must not be relabelled as other crops.', '',
            '## Evidence that would strengthen later validation', '',
            'Dated crop observations, verified field boundaries and planting/harvest timing would support a separate evaluation protocol. The current report remains a feasibility/calendar-context review.', '']
        (staging/'report.md').write_text('\n'.join(summary_lines), encoding='utf-8')
        _verify(request)
        files = ['crop_baseline_areas.csv','area_overview.csv','area_composition.csv','area_statuses.csv','raw_crop_areas.csv','calendar_context.csv','calendar_snapshot.csv','report.md']
        manifest = {'format_version': VERSION, 'kind': 'paoay_result_review', 'status': 'prepared',
            'bundle_id': request['bundle_id'], 'identity': request['identity'], 'map_handoff': request['map_handoff'],
            'calendar_file': request['calendar_file'], 'period': bundle['manifest']['period'], 'grid': grid,
            'primary_crop_method': 'three_crop_with_unknown', 'deferred_methods': crop_baseline.DEFERRED_METHODS,
            'area_count': len(areas), 'class_codes': maps.CLASSES, 'status_codes': maps.STATUSES, 'class_assignment': maps.CLASS_ASSIGNMENT,
            'independent_accuracy_evaluated': False, 'evidence_policy': 'unverified_notes_and_explicit_calendar_context_only',
            'files': {name: {'sha256': references.checksum(staging/name)} for name in files}}
        (staging/'review_handoff.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        staging.rename(folder)
    load_review_bundle(handoff)
    return handoff


def load_review_bundle(path):
    path = Path(path); manifest = json.loads(path.read_text())
    expected = {'area_overview.csv','area_composition.csv','raw_crop_areas.csv','calendar_context.csv','calendar_snapshot.csv','report.md'}
    version = manifest.get('format_version')
    legacy = version == 1
    if version in (3, 4):
        expected.add('crop_baseline_areas.csv')
    if not legacy:
        expected.add('area_statuses.csv')
    if (manifest.get('format_version') not in {1, 2, 3, VERSION} or manifest.get('kind') != 'paoay_result_review'
            or manifest.get('status') != 'prepared' or manifest.get('independent_accuracy_evaluated') is not False
            or manifest.get('evidence_policy') != 'unverified_notes_and_explicit_calendar_context_only'
            or set(manifest.get('files', {})) != expected):
        raise ValueError('Unsupported or incomplete result-review handoff')
    if not legacy and (manifest.get('class_codes') != {str(k): v for k, v in maps.CLASSES.items()}
            or manifest.get('status_codes') != {str(k): v for k, v in maps.STATUSES.items()}
            or manifest.get('class_assignment') != {str(k): v for k, v in maps.CLASS_ASSIGNMENT.items()}):
        raise ValueError('Seven-class review schema changed')
    request = {key: manifest[key] for key in ('bundle_id','identity','map_handoff','calendar_file')}
    bundle, inputs, areas = _verify(request, check_calendar=False)  # Saved snapshot survives subsequent user edits.
    if not legacy and bundle['manifest']['format_version'] != maps.VERSION:
        raise ValueError('Seven-class review requires seven-class maps')
    if manifest['period'] != bundle['manifest']['period'] or manifest['grid'] != bundle['manifest']['grid']:
        raise ValueError('Review dates/grid differ from maps')
    for name, record in manifest['files'].items():
        if references.checksum(path.parent/name) != record['sha256']:
            raise ValueError(f'Saved review changed: {name}; existing files preserved')
    if references.checksum(path.parent/'calendar_snapshot.csv') != manifest['identity']['calendar_sha256']:
        raise ValueError('Calendar snapshot differs from the frozen review input')
    tables = {}
    for name in ['area_overview.csv','area_composition.csv','raw_crop_areas.csv','calendar_context.csv'] + ([] if legacy else ['area_statuses.csv']) + (['crop_baseline_areas.csv'] if version in (3, 4) else []):
        with (path.parent/name).open(newline='', encoding='utf-8') as stream:
            tables[name] = list(csv.DictReader(stream))
    return {'manifest': manifest, 'folder': path.parent, 'overview': tables['area_overview.csv'],
            'crop_baseline': tables.get('crop_baseline_areas.csv', []), 'composition': tables['area_composition.csv'], 'statuses': tables.get('area_statuses.csv', []), 'raw_areas': tables['raw_crop_areas.csv'], 'calendar': tables['calendar_context.csv']}
