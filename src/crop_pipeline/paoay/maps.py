"""Assemble attributed maps without changing the crop-only or land-cover inputs."""

import csv
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile

from affine import Affine
import numpy as np
import rasterio

from crop_pipeline.paoay import predictions, features, land_cover, dynamic_world, imagery, references

VERSION = 2
RULES = {"minimum_crop_probability": .6, "minimum_crop_margin": .1, "minimum_supported_bins": 4}
CLASSES = {1: "Carrot", 2: "Cabbage", 3: "Potato", 4: "Other cultivated vegetation",
           5: "Forest or woody vegetation", 6: "Built-up area", 7: "Bare soil / fallow"}
STATUSES = {0: "Outside ROI", 1: "Classified", 2: "Unknown", 3: "Missing data", 4: "Excluded cover"}
SOURCES = {0: "Outside ROI", 1: "Dynamic World broad cover", 2: "Dynamic World crops + crop model",
           3: "Dynamic World crops; crop inputs missing", 4: "Dynamic World crops; crop score unresolved",
           5: "Insufficient broad-cover support", 6: "Conflicting broad cover over time",
           7: "Supported cover outside thesis scheme", 8: "No broad-cover observations"}
BANDS = ["classification", "status", "source", "supported_bins"]
DW_CODES = {1: 5, 5: 5, 6: 6, 7: 7}
EXCLUDED_DW = [0, 2, 3, 8, 9]  # 9 is the period-summary group for supported excluded cover.
CLASS_ASSIGNMENT = {code: ("not_assessed_no_independent_non_target_evidence" if code == 4 else "predicted") for code in CLASSES}
# Historical version-1 bundles remain readable without rewriting or relabelling them.
LEGACY_CLASSES = {0: "Outside ROI", 1: "Carrot", 2: "Cabbage", 3: "Potato", 4: "Cropland: crop type unresolved",
           5: "Water", 6: "Trees", 7: "Grass", 8: "Flooded vegetation", 9: "Shrub and scrub",
           10: "Built", 11: "Bare surface (fallow unverified)", 12: "Snow and ice", 13: "Unresolved cover"}
LEGACY_SOURCES = {0: "Outside ROI", 1: "Dynamic World broad cover", 2: "Dynamic World crops + crop model",
           3: "Dynamic World crops; crop inputs missing", 4: "Dynamic World crops; crop score unresolved",
           5: "Insufficient broad-cover support", 6: "Conflicting broad cover over time"}
LEGACY_BANDS = ["classification", "source", "supported_bins"]


def validate_rules(rules):
    if set(rules) != set(RULES):
        raise ValueError("Use the three documented map assembly rules")
    for key in ("minimum_crop_probability", "minimum_crop_margin"):
        v = rules[key]
        if type(v) not in (float, int) or not np.isfinite(v) or not 0 <= v <= 1:
            raise ValueError("Crop score rules must be finite values between zero and one")
    if type(rules["minimum_supported_bins"]) is not int or not 1 <= rules["minimum_supported_bins"] <= 8:
        raise ValueError("minimum_supported_bins must be an integer between one and eight")
    return dict(rules)


def context(prediction_handoff):
    crop = predictions.load_prediction_bundle(prediction_handoff)
    feature = features.load_feature_bundle(crop["manifest"]["feature_handoff"])
    guidance = land_cover.load_land_cover_bundle(feature["manifest"]["guidance_handoff"])
    world = dynamic_world.load_dynamic_world_bundle(guidance["manifest"]["dynamic_world_handoff"])
    image = imagery.load_imagery_bundle(world["manifest"]["imagery_handoff"])
    reference_path = Path(image["manifest"]["reference_handoff"])
    reference = references.load_reference_bundle(reference_path)
    if any(crop["manifest"][key] != guidance["manifest"][key] for key in ("period", "grid")):
        raise ValueError("Crop and broad-cover dates/grid differ")
    settings = world["manifest"]["identity"]["settings"]
    roi = imagery._roi_mask({"settings": settings, "fragment": {"geometry": settings["geometry"]}})
    return {"crop": crop, "feature": feature, "guidance": guidance, "reference": reference,
            "reference_path": reference_path, "roi": roi}


def prepare_request(prediction_handoff, output_root, rules=None):
    inputs = context(prediction_handoff)
    identity = {"version": VERSION, "prediction_sha256": references.checksum(prediction_handoff),
                "rules": validate_rules(RULES if rules is None else rules), "implementation_sha256": references.checksum(__file__)}
    key = imagery._digest(identity)[:20]
    return {"bundle_id": key, "identity": identity, "prediction_handoff": str(Path(prediction_handoff).resolve()),
            "folder": Path(output_root) / "bundles" / key}


def _verify(request):
    if imagery._digest(request["identity"])[:20] != request["bundle_id"]:
        raise ValueError("Map request changed after planning")
    if references.checksum(request["prediction_handoff"]) != request["identity"]["prediction_sha256"]:
        raise ValueError("Prediction dependency changed")
    validate_rules(request["identity"]["rules"])
    return context(request["prediction_handoff"])


def combine(roi, top_class, guidance, crop_codes, top_probability, margin, rules, support_count=None, conflict=None):
    """Seven thesis classes; unknown/missing/excluded remain orthogonal statuses."""
    rules = validate_rules(rules)
    shape = roi.shape
    if any(array.shape != shape for array in (top_class, guidance, crop_codes, top_probability, margin)):
        raise ValueError("Map inputs must share the same grid")
    result = np.zeros((4, *shape), dtype="uint8")
    result[1, roi], result[2, roi] = 2, 5
    missing = roi & (guidance == 0)
    result[1, missing], result[2, missing] = 3, 8
    supported = roi & np.isin(guidance, [3, 4])
    result[3] = (supported.astype('uint8') if support_count is None else np.where(roi, support_count, 0))
    for dw_code, combined_code in DW_CODES.items():
        mask = roi & (guidance == 4) & (top_class == dw_code)
        result[0, mask], result[1, mask], result[2, mask] = combined_code, 1, 1
    excluded = roi & (guidance == 4) & np.isin(top_class, EXCLUDED_DW)
    result[1, excluded], result[2, excluded] = 4, 7
    cropland = roi & (guidance == 3) & (top_class == 4)
    result[1, cropland] = np.where(crop_codes[cropland] == 0, 3, 2)
    result[2, cropland] = np.where(crop_codes[cropland] == 0, 3, 4)
    accepted = (cropland & np.isin(crop_codes, [1, 2, 3]) & np.isfinite(top_probability) & np.isfinite(margin)
                & (top_probability >= rules["minimum_crop_probability"]) & (margin > 0) & (margin >= rules["minimum_crop_margin"]))
    result[0, accepted], result[1, accepted], result[2, accepted] = crop_codes[accepted], 1, 2
    if conflict is not None:
        result[0, roi & conflict], result[1, roi & conflict], result[2, roi & conflict] = 0, 2, 6
    return result


def stable_cover(tops, states, roi, minimum_bins):
    # Agreement is evaluated at the thesis cover-group level: trees/shrubs are woody,
    # and supported water/grass/flooded vegetation/snow are outside the class scheme.
    grouped = np.where(tops == 5, 1, tops)
    grouped = np.where(np.isin(grouped, [0, 2, 3, 8]), 9, grouped)
    supported = np.isin(states, [3, 4]) & roi
    observed = np.isin(states, [1, 2, 3, 4]) & roi
    count = supported.sum(axis=0).astype('uint8')
    lowest = np.where(supported, grouped, 99).min(axis=0)
    highest = np.where(supported, grouped, -1).max(axis=0)
    conflict = (count > 0) & (lowest != highest) & roi
    enough = (count >= minimum_bins) & ~conflict & roi
    summary_state = np.where(observed.any(axis=0), 2, 0).astype('int16')
    summary_state[enough] = np.where(lowest[enough] == 4, 3, 4)
    return lowest, summary_state, count, conflict


def _write_table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def build(request, iterator=None):
    inputs = _verify(request)
    folder = request["folder"]
    handoff = folder / "map_handoff.json"
    if folder.exists():
        load_map_bundle(handoff)
        return handoff
    if request["identity"]["version"] != VERSION:
        raise ValueError("Replan map assembly using the seven-class format")
    crop, feature, guidance, roi = [inputs[key] for key in ("crop", "feature", "guidance", "roi")]
    grid, period = crop["manifest"]["grid"], crop["manifest"]["period"]
    codes = np.zeros(roi.shape, dtype='uint8')
    score, margin = np.full(roi.shape, -9999.), np.full(roi.shape, -9999.)
    rows, columns = feature["rows"], feature["columns"]
    for name, target in (("carrot", 1), ("cabbage", 2), ("potato", 3)):
        selected = crop["class_codes"] == crop["manifest"]["class_codes"][name]
        codes[rows[selected], columns[selected]] = target
    score[rows, columns], margin[rows, columns] = crop["top_probability"], crop["probability_margin"]
    rules = request["identity"]["rules"]
    transform = Affine(*grid["transform"])
    hectare = abs(transform.a*transform.e-transform.b*transform.d)/10000
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.maps-', dir=folder.parent) as temporary:
        staging = Path(temporary) / request["bundle_id"]
        (staging / 'rasters').mkdir(parents=True)
        tops, states, area_rows, source_rows, status_rows = [], [], [], [], []
        def save(name, data):
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(staging/'rasters'/f'{name}.tif', 'w', driver='GTiff', height=grid['height'], width=grid['width'],
                    count=4, dtype='uint8', crs=grid['crs'], transform=transform, nodata=0, compress='deflate') as raster:
                raster.write(data); raster.descriptions = tuple(BANDS)
                raster.write_mask(roi.astype('uint8')*255)  # Valid zero support counts inside ROI; no external mask file.
            for code, label in CLASSES.items():
                if code:
                    n = int((data[0] == code).sum())
                    area_rows.append({'map': name, 'class': label, 'code': code, 'pixels': n, 'area_ha': n*hectare, 'roi_percent': n/roi.sum()*100, 'assignment_status': CLASS_ASSIGNMENT[code]})
            for code, label in SOURCES.items():
                if code:
                    source_rows.append({'map': name, 'source': label, 'code': code, 'pixels': int((data[2] == code).sum())})
            for code, label in STATUSES.items():
                if code:
                    n = int((data[1] == code).sum())
                    status_rows.append({'map': name, 'status': label, 'code': code, 'pixels': n,
                                        'area_ha': n*hectare, 'roi_percent': n/roi.sum()*100})
        items = list(zip(period['intervals'], guidance['rasters']))
        for interval, path in (iterator(items) if iterator else items):
            with rasterio.open(path) as raster:
                raw = raster.read()
            tops.append(raw[0]); states.append(raw[5])
            save(interval['name'], combine(roi, raw[0], raw[5], codes, score, margin, rules))
        top, state, count, conflict = stable_cover(np.asarray(tops), np.asarray(states), roi, rules['minimum_supported_bins'])
        save('period_summary', combine(roi, top, state, codes, score, margin, rules, count, conflict))
        _write_table(staging/'class_areas.csv', area_rows); _write_table(staging/'sources.csv', source_rows)
        _write_table(staging/'statuses.csv', status_rows)
        _verify(request)
        files = [*sorted((staging/'rasters').glob('*.tif')), staging/'class_areas.csv', staging/'sources.csv', staging/'statuses.csv']
        manifest = {'format_version': VERSION, 'kind': 'paoay_combined_maps', 'status': 'prepared',
            'bundle_id': request['bundle_id'], 'identity': request['identity'], 'prediction_handoff': request['prediction_handoff'],
            'period': period, 'grid': grid, 'class_codes': CLASSES, 'source_codes': SOURCES, 'status_codes': STATUSES, 'bands': BANDS,
            'class_assignment': CLASS_ASSIGNMENT, 'unassigned_class_code': 0,
            'roi_pixels': int(roi.sum()), 'pixel_area_ha': hectare,
            'raw_crop_predictions_preserved': True, 'summary_policy': 'minimum_supported_bins_and_unanimous_supported_thesis_cover_groups',
            'crop_time_scope': 'one_four_month_model_label_conditional_on_half_month_broad_cover_not_dated_crop_observations',
            'rules_status': 'illustrative_unvalidated_not_selected_using_local_scores',
            'limitations': ['Predictions, not local accuracy or ground truth', 'Bare predicts the bare-soil/fallow group but does not confirm fallow',
                'Other cultivated vegetation is not assessed; weak target-crop scores do not establish this class',
                'Unknown, missing data and excluded cover are statuses, not thesis classes',
                'Temporal broad-cover conflicts remain unresolved in the period summary; inspect the eight individual maps'],
            'attribution': guidance['manifest']['attribution'], 'license': guidance['manifest']['license'],
            'copernicus_notice': guidance['manifest'].get('copernicus_notice'), 'source_url': dynamic_world.SOURCE_URL,
            'files': {p.relative_to(staging).as_posix(): {'sha256': references.checksum(p)} for p in files}}
        (staging/'map_handoff.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        staging.rename(folder)
    load_map_bundle(handoff)
    return handoff


def load_map_bundle(path):
    path = Path(path); manifest = json.loads(path.read_text())
    version = manifest.get('format_version')
    legacy = version == 1
    classes, sources, bands = (LEGACY_CLASSES, LEGACY_SOURCES, LEGACY_BANDS) if legacy else (CLASSES, SOURCES, BANDS)
    if (version not in {1, VERSION} or manifest.get('kind') != 'paoay_combined_maps'
            or manifest.get('status') != 'prepared' or manifest.get('bands') != bands
            or manifest.get('class_codes') != {str(k): v for k, v in classes.items()}
            or manifest.get('source_codes') != {str(k): v for k, v in sources.items()}
            or manifest.get('raw_crop_predictions_preserved') is not True):
        raise ValueError('Unsupported or inconsistent map handoff')
    if not legacy and (manifest.get('status_codes') != {str(k): v for k, v in STATUSES.items()}
            or manifest.get('class_assignment') != {str(k): v for k, v in CLASS_ASSIGNMENT.items()}
            or manifest.get('unassigned_class_code') != 0):
        raise ValueError('Seven-class map status/assignment schema changed')
    inputs = _verify({key: manifest[key] for key in ('bundle_id', 'identity', 'prediction_handoff')})
    if any(manifest[key] != inputs['crop']['manifest'][key] for key in ('period', 'grid')):
        raise ValueError('Combined-map dates/grid differ from source predictions')
    names = [x['name'] for x in manifest['period']['intervals']] + ['period_summary']
    table_names = ['class_areas.csv', 'sources.csv'] + ([] if legacy else ['statuses.csv'])
    expected = {f'rasters/{name}.tif' for name in names} | set(table_names)
    if set(manifest.get('files', {})) != expected:
        raise ValueError('Combined map inventory must contain nine rasters and its summary tables')
    roi, grid = inputs['roi'], manifest['grid']
    totals = {}
    for name, record in manifest['files'].items():
        if references.checksum(path.parent/name) != record['sha256']:
            raise ValueError(f'Saved map changed: {name}; existing files preserved')
        if name.endswith('.tif'):
            with rasterio.open(path.parent/name) as raster:
                if (raster.count != len(bands) or (raster.height, raster.width) != roi.shape
                        or raster.crs != rasterio.crs.CRS.from_user_input(grid['crs'])
                        or not raster.transform.almost_equals(Affine(*grid['transform']))
                        or list(raster.descriptions) != bands or raster.nodata != 0 or set(raster.dtypes) != {'uint8'}):
                    raise ValueError('Combined map raster differs from the saved grid/schema')
                data = raster.read()
                if not np.array_equal(raster.dataset_mask() > 0, roi):
                    raise ValueError('Combined-map validity mask differs from the ROI')
            source_band, count_band = (1, 2) if legacy else (2, 3)
            if (np.any(data[:, ~roi] != 0) or np.any(data[count_band] > 8)
                    or not np.isin(data[source_band, roi], [k for k in sources if k]).all()):
                raise ValueError('Invalid combined-map codes or outside-ROI values')
            if legacy:
                if not np.isin(data[0, roi], [k for k in classes if k]).all():
                    raise ValueError('Invalid legacy combined-map classes')
            else:
                classified = roi & (data[1] == 1)
                if (not np.isin(data[1, roi], [1, 2, 3, 4]).all()
                        or not np.isin(data[0, classified], list(CLASSES)).all()
                        or np.any(data[0, ~classified] != 0) or np.any(data[0] == 4)):
                    raise ValueError('Thesis classes and unknown/missing/excluded statuses are inconsistent')
                totals[Path(name).stem] = int(classified.sum())
    tables = {}
    for name in table_names:
        with (path.parent/name).open(newline='', encoding='utf-8') as stream:
            tables[name] = list(csv.DictReader(stream))
    if not legacy:
        for name in names:
            areas = [r for r in tables['class_areas.csv'] if r['map'] == name]
            states = [r for r in tables['statuses.csv'] if r['map'] == name]
            if (len(areas) != 7 or {int(r['code']) for r in areas} != set(CLASSES)
                    or sum(int(r['pixels']) for r in areas) != totals[name]
                    or len(states) != 4 or {int(r['code']) for r in states} != {1, 2, 3, 4}
                    or sum(int(r['pixels']) for r in states) != int(roi.sum())
                    or next(int(r['pixels']) for r in states if int(r['code']) == 1) != totals[name]):
                raise ValueError('Seven-class and separate status area accounting disagree')
    return {'manifest': manifest, 'folder': path.parent, 'areas': tables['class_areas.csv'], 'sources': tables['sources.csv'],
            'statuses': tables.get('statuses.csv', []),
            'rasters': {name: path.parent/'rasters'/f'{name}.tif' for name in names}}


def plot_maps(bundle):
    """Seven study-class legend plus a separate data-status legend; legacy preview retained."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    legacy = bundle['manifest']['format_version'] == 1
    if legacy:
        colors = ['#ed9137','#4b9a65','#8b70b1','#b6d38a','#419bdf','#397d49','#b9c56a','#7a87c6','#dfc35a','#c4281b','#a59b8f','#b39fe1','#dedede']
    else:
        colors = ['#ed9137','#4b9a65','#8b70b1','#b6d38a','#397d49','#c4281b','#a59b8f', '#e8c873','#d9d9d9','#8cacc6']
    figure, axes = plt.subplots(3, 3, figsize=(13, 12), layout='constrained')
    for axis, (name, path) in zip(axes.flat, bundle['rasters'].items()):
        with rasterio.open(path) as raster:
            data = raster.read(1, masked=True)
            if not legacy:
                states = raster.read(2)
                data = data.copy()
                for status, display_code in ((2, 8), (3, 9), (4, 10)):
                    data[(states == status) & ~np.ma.getmaskarray(data)] = display_code
        axis.imshow(data, cmap=ListedColormap(colors).with_extremes(bad='white'), norm=BoundaryNorm(np.arange(.5, len(colors)+1), len(colors)), interpolation='nearest')
        if name == 'period_summary':
            title = 'Conservative period summary'
        else:
            interval = next(x for x in bundle['manifest']['period']['intervals'] if x['name'] == name)
            start, end = date.fromisoformat(interval['start']), date.fromisoformat(interval['end'])-timedelta(days=1)
            title = f'{start:%b} {start.day}–{end.day}, {start.year}'
        axis.set_title(title, fontsize=10); axis.set_axis_off()
    figure.suptitle('Combined prediction guidance | crop identities use the entire four-month window' if legacy else
        'Seven thesis classes | crop identities use the full four-month window\nOther cultivated vegetation: not assessed. Bare signal does not confirm fallow.', fontsize=11)
    if legacy:
        figure.legend(handles=[Patch(color=colors[code-1], label=label) for code, label in LEGACY_CLASSES.items() if code], loc='outside lower center', ncol=3, fontsize=8)
    else:
        figure.legend(handles=[Patch(color=colors[code-1], label=label) for code, label in CLASSES.items()],
                      title='Thesis classes', loc='outside lower left', ncol=3, fontsize=8)
        figure.legend(handles=[Patch(color=colors[i], label=STATUSES[code]) for i, code in ((7, 2), (8, 3), (9, 4))],
                      title='Data status (not classes)', loc='outside lower right', ncol=1, fontsize=8)
    return figure
