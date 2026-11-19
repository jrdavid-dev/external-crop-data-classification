"""Multi-year land activity (active cropland, perennial vegetation, bare/built) for the Atok study area.

Ported from the archived Atok cropping-pattern notebook (https://github.com/jrdavid-dev/external-crop-data-classification/tree/c1fe80c/notebooks/archive/14_atok_unsupervised.ipynb):
growth cycles are detected in 2019-2026 half-month Sentinel-2 composites, eight pixel summaries are
clustered with a stable K-means, and clusters are named by fixed rules. The result is descriptive and
unvalidated: it never changes the frozen crop predictions and is not crop/noncrop ground truth.
NAMRIA ELUM 2023 polygons provide an independent check of the non-crop side.
"""
import csv
import json
from pathlib import Path
import tempfile

from affine import Affine
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

from crop_pipeline.paoay import crop_baseline, imagery, maps, references

VERSION = 2
KIND = 'paoay_land_activity'
BANDS = ('B3', 'B4', 'B5', 'B8', 'B11')
CYCLE_VARS = ('NDVI', 'NDRE', 'NDMI', 'B5', 'B11')
RULES = {'first_year': 2019, 'last_year': 2026, 'minimum_clear_half_months': 60, 'cycle_prominence': 0.15,
         'minimum_cycle_half_months': 4, 'maximum_cycle_half_months': 16, 'minimum_cycle_clear_share': 0.6,
         'k_values': [2, 3, 4, 5, 6, 7, 8], 'stable_ari': 0.8, 'seed': 42,
         'reference_erode_m': 10.0, 'landuse_erode_m': 5.0}
LAND_CLASSES = {1: 'active cropland', 2: 'mixed / low activity', 3: 'perennial vegetation', 4: 'bare / built',
                5: 'insufficient clear observations'}
# One palette for both map panels, so no colour means two different things.
LAND_COLORS = {1: '#d7301f', 2: '#fdd835', 3: '#1b5e20', 4: '#424242', 5: '#e0e0e0'}
CROP_COLORS = {'Carrot': '#ff8f00', 'Cabbage': '#9ccc65', 'Potato': '#795548',
               'Unknown crop type': '#9e9e9e', 'Missing input': '#ffffff'}
NOT_CROPLAND_COLOR = '#cfd8e3'
SUMMARY_FEATURES = ('ndvi_p10', 'ndvi_p50', 'ndvi_p90', 'cycles_per_year', 'mean_cycle_amplitude',
                    'bare_share', 'ndmi_median', 'b11_median')
LANDUSE_TYPES = {  # NAMRIA ELUM 2023 sub-class -> type, as in the archived Atok notebooks
    'General Production Forest': 'forest', 'General Protection Forest': 'forest', 'National Park': 'forest',
    'General Production Agriculture': 'forest',  # closed forest on alienable land (ELUM metadata)
    'Residential': 'built-up', 'General Institutional': 'built-up', 'Commercial': 'built-up',
    'General Tourism': 'built-up', 'Parks and Recreation': 'built-up', 'Road Network': 'built-up',
    'Pasture': 'pasture', 'Rivers': 'water', 'Crops': 'crops'}
NON_CROP_TYPES = ('forest', 'built-up', 'pasture')
FILES = ('land_activity.tif', 'pixel_summaries.tif', 'k_selection.csv', 'cluster_centers.csv',
         'reference_area_check.csv', 'namria_check.csv', 'crop_map_by_land_activity.csv',
         'monthly_activity.csv', 'four_month_windows.csv')
NODATA = -9999.


def validate_rules(rules):
    if set(rules) != set(RULES):
        raise ValueError('Land-activity rules must use exactly the documented keys')
    if not (rules['first_year'] <= rules['last_year'] and rules['minimum_clear_half_months'] > 0
            and 0 < rules['minimum_cycle_half_months'] <= rules['maximum_cycle_half_months']
            and 0 <= rules['minimum_cycle_clear_share'] <= 1 and 0 < rules['stable_ari'] <= 1
            and rules['k_values'] and min(rules['k_values']) >= 2):
        raise ValueError('Invalid land-activity rules')
    return json.loads(json.dumps(rules))


def half_month_slots(first_year, last_year):
    """(year, month, half) slots and their nominal mid dates (8th and 23rd)."""
    slots = [(y, m, h) for y in range(first_year, last_year + 1) for m in range(1, 13) for h in (1, 2)]
    dates = np.array([f'{y}-{m:02d}-{"08" if h == 1 else "23"}' for y, m, h in slots], dtype='datetime64[D]')
    return slots, dates


def composite_path(root, slot):
    year, month, half = slot
    return Path(root) / str(year) / f'Sentinel2_CSPlus_{year}_{month:02d}_H{half}.tif'


def composite_inventory(root, rules):
    slots, _ = half_month_slots(rules['first_year'], rules['last_year'])
    found = [(index, composite_path(root, slot)) for index, slot in enumerate(slots)]
    found = [(index, path) for index, path in found if path.is_file()]
    if not found:
        raise FileNotFoundError(f'No Sentinel2_CSPlus half-month composites under {root}')
    return [{'slot': index, 'file': str(path.relative_to(root)), 'sha256': references.checksum(path)}
            for index, path in found]


def read_cube(root, inventory, grid, rules):
    """Reflectance cube (slots, bands, rows, cols) on the crop-map grid; -9999, NaN and <= 0 are missing."""
    slots, _ = half_month_slots(rules['first_year'], rules['last_year'])
    target = Affine(*grid['transform'])
    cube = np.full((len(slots), len(BANDS), grid['height'], grid['width']), np.nan, dtype='float32')
    for item in inventory:
        path = Path(root) / item['file']
        with rasterio.open(path) as source:
            if (source.crs != rasterio.crs.CRS.from_user_input(grid['crs']) or list(source.descriptions) != list(BANDS)
                    or abs(source.transform.a - target.a) > 1e-9 or abs(source.transform.e - target.e) > 1e-9):
                raise ValueError(f'{path.name} uses a different CRS, resolution or band order')
            bounds = rasterio.transform.array_bounds(grid['height'], grid['width'], target)
            window = from_bounds(*bounds, transform=source.transform)
            offsets = (window.col_off, window.row_off)
            if any(abs(v - round(v)) > 1e-6 for v in offsets) or window.col_off < 0 or window.row_off < 0 \
                    or window.col_off + grid['width'] > source.width or window.row_off + grid['height'] > source.height:
                raise ValueError(f'{path.name} does not cover the crop-map grid on aligned pixels')
            window = rasterio.windows.Window(round(window.col_off), round(window.row_off), grid['width'], grid['height'])
            values = source.read(window=window).astype('float32')
            nodata = source.nodata if source.nodata is not None else NODATA
            values[(values == nodata) | ~np.isfinite(values) | (values <= 0)] = np.nan
            cube[item['slot']] = values / 10000.
    return cube


def add_indices(bands):
    """Append NDVI, NDRE and NDMI to (..., B3, B4, B5, B8, B11)."""
    b4, b5, b8, b11 = bands[..., 1], bands[..., 2], bands[..., 3], bands[..., 4]
    with np.errstate(divide='ignore', invalid='ignore'):
        extra = np.stack([(b8 - b4) / (b8 + b4), (b8 - b5) / (b8 + b5), (b8 - b11) / (b8 + b11)], -1)
    return np.concatenate([bands, extra], -1)


def smooth(values):
    """Interpolate inside gaps of up to two half-months, then a 3-step running median (per column)."""
    import pandas as pd
    frame = pd.DataFrame(values)
    return frame.interpolate(limit=2, limit_area='inside', axis=0).rolling(3, center=True, min_periods=2).median().to_numpy()


def cycle_bounds(ndvi, clear_column, rules):
    """(start, peak, end) slots of growth cycles in one smoothed NDVI series (archived notebook 14 rule)."""
    from scipy.signal import find_peaks
    finite = np.isfinite(ndvi)
    if finite.sum() < 20:
        return []
    usable = np.where(finite, ndvi, np.nanmin(ndvi))
    peaks = find_peaks(usable, prominence=rules['cycle_prominence'], distance=4)[0]
    bounds = []
    for index, peak in enumerate(peaks):
        if not finite[peak]:
            continue
        previous_peak = peaks[index - 1] if index > 0 else -1
        next_peak = peaks[index + 1] if index + 1 < len(peaks) else len(ndvi)
        left = np.arange(max(0, peak - 12, previous_peak + 1), peak)
        right = np.arange(peak + 1, min(len(ndvi), peak + 13, next_peak))
        left, right = left[finite[left]], right[finite[right]]
        if not len(left) or not len(right):
            continue
        start, end = left[np.argmin(ndvi[left])], right[np.argmin(ndvi[right])]
        if (rules['minimum_cycle_half_months'] <= end - start <= rules['maximum_cycle_half_months']
                and clear_column[start:end + 1].mean() >= rules['minimum_cycle_clear_share']):
            bounds.append((start, peak, end))
    return bounds


def pixel_summaries(series, clear, rules, cycles_out=None):
    """Eight per-pixel summaries from (slots, pixels, 8) series; cycles need finite values of all cycle variables.

    If ``cycles_out`` is a list, (pixel, start, peak, end) of each counted cycle is appended to it.
    """
    variables = list(BANDS) + ['NDVI', 'NDRE', 'NDMI']
    smoothed = np.stack([smooth(series[:, :, variables.index(v)]) for v in CYCLE_VARS], -1)
    count, amplitude = np.zeros(series.shape[1]), np.zeros(series.shape[1])
    for pixel in range(series.shape[1]):
        for start, peak, end in cycle_bounds(smoothed[:, pixel, 0], clear[:, pixel], rules):
            if np.isfinite(smoothed[[start, peak, end], pixel, :]).all():
                count[pixel] += 1
                amplitude[pixel] += smoothed[peak, pixel, 0] - smoothed[start, pixel, 0]
                if cycles_out is not None:
                    cycles_out.append((pixel, start, peak, end))
    first, last = clear.argmax(0), len(clear) - 1 - clear[::-1].argmax(0)
    years = np.maximum((last - first + 1) / 24, 0.5)
    ndvi = series[:, :, variables.index('NDVI')]
    with np.errstate(invalid='ignore'):
        values = {'ndvi_p10': np.nanpercentile(ndvi, 10, 0), 'ndvi_p50': np.nanpercentile(ndvi, 50, 0),
                  'ndvi_p90': np.nanpercentile(ndvi, 90, 0), 'cycles_per_year': count / years,
                  'mean_cycle_amplitude': amplitude / np.maximum(count, 1),
                  'bare_share': np.nanmean(np.where(clear, ndvi < 0.3, np.nan), 0),
                  'ndmi_median': np.nanmedian(series[:, :, variables.index('NDMI')], 0),
                  'b11_median': np.nanmedian(series[:, :, variables.index('B11')], 0)}
    return np.column_stack([values[name] for name in SUMMARY_FEATURES])


def choose_kmeans(z, rules):
    """K by silhouette among stable K (seed and 80% subsample ARI >= stable_ari), as in the archived notebook."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    seed = rules['seed']
    sample = np.random.default_rng(seed).choice(len(z), min(5000, len(z)), replace=False)
    table = []
    for k in rules['k_values']:
        if k >= len(z):
            continue
        labels = [KMeans(k, n_init=10, random_state=s).fit(z).predict(z) for s in range(5)]
        seed_ari = np.mean([adjusted_rand_score(labels[i], labels[j]) for i in range(5) for j in range(i + 1, 5)])
        rng = np.random.default_rng(seed)
        sub_ari = np.mean([adjusted_rand_score(labels[0], KMeans(k, n_init=10, random_state=0).fit(
            z[rng.choice(len(z), int(0.8 * len(z)), replace=False)]).predict(z)) for _ in range(3)])
        table.append({'k': k, 'silhouette': float(silhouette_score(z[sample], labels[0][sample])),
                      'seed_ari': float(seed_ari), 'subsample_ari': float(sub_ari)})
    if not table:
        raise ValueError('Too few pixels to cluster')
    stable = [row for row in table if row['seed_ari'] >= rules['stable_ari'] and row['subsample_ari'] >= rules['stable_ari']]
    chosen = max(stable or table, key=lambda row: row['silhouette'])['k']
    for row in table:
        row['stable'] = row in stable
        row['chosen'] = row['k'] == chosen
    return KMeans(chosen, n_init=10, random_state=seed).fit(z).predict(z), table


def land_name(center):
    """Fixed naming rules applied to cluster medians."""
    if center['ndvi_p90'] < 0.35:
        return 'bare / built'
    if center['ndvi_p10'] >= 0.5 and center['cycles_per_year'] < 1:
        return 'perennial vegetation'
    if center['cycles_per_year'] >= 1 and center['mean_cycle_amplitude'] >= 0.2:
        return 'active cropland'
    return 'mixed / low activity'


def classify(cube, mask, rules, cycles_out=None):
    """Land-activity codes and summaries for mask pixels of a (slots, bands, rows, cols) reflectance cube.

    If ``cycles_out`` is a list, (row, col, start_slot, peak_slot, end_slot) of every detected cycle is appended.
    """
    rules = validate_rules(rules)
    from sklearn.preprocessing import StandardScaler
    rows, cols = np.nonzero(mask)
    series = add_indices(np.moveaxis(cube[:, :, rows, cols], 1, -1))
    clear = np.isfinite(series).all(-1)
    enough = clear.sum(0) >= rules['minimum_clear_half_months']
    codes = np.zeros(mask.shape, dtype='uint8')
    codes[mask] = 5
    summaries = np.full((len(SUMMARY_FEATURES), *mask.shape), NODATA, dtype='float32')
    if enough.sum() < 3:
        raise ValueError('Fewer than three pixels have enough clear half-months')
    found = []
    values = pixel_summaries(series[:, enough], clear[:, enough], rules, found)
    if cycles_out is not None:
        kept_rows, kept_cols = rows[enough], cols[enough]
        cycles_out.extend((int(kept_rows[p]), int(kept_cols[p]), int(s), int(k), int(e)) for p, s, k, e in found)
    labels, table = choose_kmeans(StandardScaler().fit_transform(values), rules)
    centers = []
    lookup = {name: code for code, name in LAND_CLASSES.items()}
    for cluster in sorted(set(labels.tolist())):
        center = dict(zip(SUMMARY_FEATURES, np.median(values[labels == cluster], 0).tolist()))
        centers.append({'cluster': cluster, 'name': land_name(center), 'pixels': int((labels == cluster).sum()), **center})
    names = {row['cluster']: row['name'] for row in centers}
    codes[rows[enough], cols[enough]] = [lookup[names[label]] for label in labels]
    summaries[:, rows[enough], cols[enough]] = values.T
    return codes, summaries, table, centers


def share_rows(codes, masks):
    """Share of each land-activity class inside each named mask."""
    out = []
    for name, extra, mask in masks:
        inside = codes[mask]
        n = int(mask.sum())
        assessed = inside[inside != 5]
        row = {**extra, 'name': name, 'pixels': n, 'assessed_pixels': int(len(assessed))}
        for code, label in LAND_CLASSES.items():
            if code != 5:
                row[label] = float((assessed == code).mean()) if len(assessed) else 0.
        out.append(row)
    return out


def polygon_mask(geometry, grid, erode_m=0.):
    shape_ = geometry.buffer(-erode_m) if erode_m else geometry
    if shape_.is_empty:
        return np.zeros((grid['height'], grid['width']), dtype=bool)
    from shapely.geometry import mapping
    return geometry_mask([mapping(shape_)], out_shape=(grid['height'], grid['width']),
                         transform=Affine(*grid['transform']), invert=True)


def reference_check(codes, reference, grid, roi, rules):
    from pyproj import Transformer
    from shapely import transform
    from shapely.geometry import shape
    project = Transformer.from_crs('EPSG:4326', grid['crs'], always_xy=True).transform
    masks = []
    for record in references.reference_areas(reference):
        if record['geometry_status'] != 'valid' or record['geometry'] is None:
            continue
        geometry = transform(shape(record['geometry']), project, interleaved=False)
        mask = polygon_mask(geometry, grid, rules['reference_erode_m']) & roi
        if mask.any():
            masks.append((record['name'], {'reference_id': record['reference_id']}, mask))
    return share_rows(codes, masks)


def landuse_check(codes, landuse_path, grid, roi, rules):
    """Share of each land-activity class inside NAMRIA ELUM types (polygons shrunk inward)."""
    import geopandas as gpd
    from shapely import union_all
    from shapely.geometry import box
    bounds = rasterio.transform.array_bounds(grid['height'], grid['width'], Affine(*grid['transform']))
    frame = gpd.read_file(landuse_path, bbox=gpd.GeoSeries([box(*bounds)], crs=grid['crs']),
                          engine='pyogrio').to_crs(grid['crs'])
    frame = frame.assign(type=frame['sub_elu'].map(LANDUSE_TYPES)).dropna(subset=['type'])
    masks = []
    for kind in (*NON_CROP_TYPES, 'water', 'crops'):
        shapes = frame.loc[frame['type'] == kind, 'geometry'].to_numpy()
        if len(shapes):
            mask = polygon_mask(union_all(shapes), grid, rules['landuse_erode_m']) & roi
            if mask.any():
                masks.append((kind, {'polygons': int(len(shapes))}, mask))
    rows = share_rows(codes, masks)
    for row in rows:
        row['outside active cropland'] = 1 - row['active cropland'] if row['assessed_pixels'] else 0.
    return rows


def crop_map_rows(codes, baseline, pixel_ha):
    """Frozen crop-baseline categories split by land-activity class (descriptive; predictions unchanged)."""
    classification, status = baseline['data']
    roi = baseline['roi']
    categories = [(label, roi & (status == 1) & (classification == code)) for code, label in crop_baseline.CROPS.items()]
    categories += [(crop_baseline.STATUSES[code], roi & (status == code)) for code in (2, 3)]
    rows = []
    for label, category in categories:
        total = int(category.sum())
        for code, land in LAND_CLASSES.items():
            count = int((category & (codes == code)).sum())
            rows.append({'crop_map_category': label, 'land_activity': land, 'pixels': count, 'area_ha': count * pixel_ha,
                         'share_of_category': count / total if total else 0.})
    return rows


MONTHS = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')


def window_label(start_month):
    return f'{MONTHS[start_month - 1]}–{MONTHS[(start_month + 2) % 12]}'


def seasonal_tables(cube, codes, cycles, rules, available_slots=None):
    """Month-by-month clear observations and cycle timing in active cropland, plus every four-month window.

    A four-month window is eight consecutive half-month composites. Its complete share is the share of active
    cropland pixels with all eight clear, which is what the current fixed model needs. ``available_slots`` are
    the slots with a composite file; a file that is cloudy everywhere still counts (as not clear). Without it,
    a slot counts as available when any pixel is clear.
    """
    slots, _ = half_month_slots(rules['first_year'], rules['last_year'])
    months = np.array([month for _, month, _ in slots])
    clear = np.isfinite(cube).all(axis=1)
    if available_slots is None:
        available = clear.any(axis=(1, 2))
    else:
        available = np.zeros(len(slots), dtype=bool)
        available[list(available_slots)] = True
    active = codes == 1
    assessed = np.isin(codes, [1, 2, 3, 4])
    cropland_cycles = [(start, peak) for row, col, start, peak, _ in cycles if active[row, col]]
    starts = np.bincount([months[start] for start, _ in cropland_cycles], minlength=13)[1:]
    peaks = np.bincount([months[peak] for _, peak in cropland_cycles], minlength=13)[1:]
    total = max(len(cropland_cycles), 1)
    monthly = []
    for month in range(1, 13):
        chosen = (months == month) & available
        monthly.append({'month': MONTHS[month - 1], 'composites': int(chosen.sum()),
                        'clear_share_active_cropland': float(clear[chosen][:, active].mean()) if chosen.any() and active.any() else float('nan'),
                        'clear_share_study_area': float(clear[chosen][:, assessed].mean()) if chosen.any() and assessed.any() else float('nan'),
                        'cycle_start_share': starts[month - 1] / total, 'cycle_peak_share': peaks[month - 1] / total})
    windows = []
    for index, (year, month, half) in enumerate(slots):
        if half != 1 or index + 8 > len(slots):
            continue
        selected = np.arange(index, index + 8)
        present = int(available[selected].sum())
        complete = clear[selected][:, active].all(axis=0).mean() if present == 8 and active.any() else float('nan')
        windows.append({'year': year, 'start_month': month, 'window': window_label(month),
                        'composites_available': present, 'active_cropland_complete_share': float(complete)})
    return monthly, windows


def window_summary(windows):
    """For each start month: median and best complete share over years with all eight composites."""
    rows = []
    for month in range(1, 13):
        values = [(row['active_cropland_complete_share'], row['year']) for row in windows
                  if int(row['start_month']) == month and int(row['composites_available']) == 8]
        shares = [float(value) for value, _ in values]
        best = max(values, key=lambda item: float(item[0])) if values else (float('nan'), None)
        rows.append({'window': window_label(month), 'years_with_all_composites': len(values),
                     'median_complete_share': float(np.median(shares)) if shares else float('nan'),
                     'best_complete_share': float(best[0]), 'best_year': best[1]})
    return rows


def prepare_request(baseline_handoff, imagery_root, landuse_path, output_root, rules=None):
    rules = validate_rules(RULES if rules is None else rules)
    crop_baseline.load_bundle(baseline_handoff)
    inventory = composite_inventory(imagery_root, rules)
    identity = {'version': VERSION, 'baseline_sha256': references.checksum(baseline_handoff),
                'composites': inventory, 'landuse_sha256': references.checksum(landuse_path), 'rules': rules,
                'implementation_sha256': references.checksum(__file__)}
    key = imagery._digest(identity)[:20]
    return {'bundle_id': key, 'identity': identity, 'baseline_handoff': str(Path(baseline_handoff).resolve()),
            'imagery_root': str(Path(imagery_root).resolve()), 'landuse_path': str(Path(landuse_path).resolve()),
            'folder': Path(output_root) / 'bundles' / key}


def _verify(request):
    if imagery._digest(request['identity'])[:20] != request['bundle_id']:
        raise ValueError('Land-activity request changed after planning')
    if references.checksum(request['baseline_handoff']) != request['identity']['baseline_sha256']:
        raise ValueError('Crop-baseline dependency changed')
    if references.checksum(request['landuse_path']) != request['identity']['landuse_sha256']:
        raise ValueError('Land-use map changed')
    return crop_baseline.load_bundle(request['baseline_handoff'])


def _write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build(request):
    baseline = _verify(request)
    folder = request['folder']
    handoff = folder / 'land_activity_handoff.json'
    if folder.exists():
        load_bundle(handoff)
        return handoff
    rules = request['identity']['rules']
    for item in request['identity']['composites']:
        if references.checksum(Path(request['imagery_root']) / item['file']) != item['sha256']:
            raise ValueError(f"Composite changed since planning: {item['file']}")
    grid = baseline['manifest']['grid']
    inputs = maps.context(baseline['manifest']['prediction_handoff'])
    roi = inputs['roi']
    cube = read_cube(request['imagery_root'], request['identity']['composites'], grid, rules)
    cycles = []
    codes, summaries, table, centers = classify(cube, roi, rules, cycles)
    monthly, windows = seasonal_tables(cube, codes, cycles, rules,
                                       [item['slot'] for item in request['identity']['composites']])
    affine = Affine(*grid['transform'])
    pixel_ha = abs(affine.a * affine.e - affine.b * affine.d) / 10000
    tables = {'k_selection.csv': table, 'cluster_centers.csv': centers,
              'reference_area_check.csv': reference_check(codes, inputs['reference'], grid, roi, rules),
              'namria_check.csv': landuse_check(codes, request['landuse_path'], grid, roi, rules),
              'crop_map_by_land_activity.csv': crop_map_rows(codes, baseline, pixel_ha),
              'monthly_activity.csv': monthly, 'four_month_windows.csv': windows}
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.land-activity-', dir=folder.parent) as temporary:
        staging = Path(temporary) / request['bundle_id']
        staging.mkdir()
        profile = dict(driver='GTiff', height=grid['height'], width=grid['width'], crs=grid['crs'],
                       transform=affine, compress='deflate')
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(staging / 'land_activity.tif', 'w', count=1, dtype='uint8', nodata=0, **profile) as raster:
                raster.write(codes, 1)
                raster.descriptions = ('land_activity',)
                raster.write_mask(roi.astype('uint8') * 255)
            with rasterio.open(staging / 'pixel_summaries.tif', 'w', count=len(SUMMARY_FEATURES), dtype='float32',
                               nodata=NODATA, **profile) as raster:
                raster.write(summaries)
                raster.descriptions = SUMMARY_FEATURES
        for name, rows in tables.items():
            _write_csv(staging / name, rows)
        _verify(request)
        manifest = {'format_version': VERSION, 'kind': KIND, 'status': 'prepared',
                    **{key: request[key] for key in ('bundle_id', 'identity', 'baseline_handoff', 'imagery_root', 'landuse_path')},
                    'grid': grid, 'land_classes': LAND_CLASSES, 'summary_features': list(SUMMARY_FEATURES),
                    'composites_used': len(request['identity']['composites']),
                    'chosen_k': next(row['k'] for row in table if row['chosen']),
                    'crop_predictions_changed': False,
                    'method_source': 'https://github.com/jrdavid-dev/external-crop-data-classification/tree/c1fe80c/notebooks/archive/14_atok_unsupervised.ipynb (land activity, NAMRIA check)',
                    'interpretation': ('Unsupervised, rule-named land activity from 2019-2026 composites; descriptive '
                                       'and unvalidated. Not crop/noncrop ground truth; crop predictions are unchanged.'),
                    'files': {name: {'sha256': references.checksum(staging / name)} for name in FILES}}
        (staging / 'land_activity_handoff.json').write_text(json.dumps(manifest, indent=2) + '\n')
        staging.rename(folder)
    load_bundle(handoff)
    return handoff


def load_bundle(path):
    path = Path(path)
    m = json.loads(path.read_text())
    if (m.get('format_version') != VERSION or m.get('kind') != KIND or m.get('status') != 'prepared'
            or m.get('crop_predictions_changed') is not False or set(m.get('files', {})) != set(FILES)
            or m.get('land_classes') != {str(k): v for k, v in LAND_CLASSES.items()}):
        raise ValueError('Unsupported land-activity bundle')
    baseline = _verify(m)
    if m['grid'] != baseline['manifest']['grid']:
        raise ValueError('Land-activity grid differs from the crop baseline')
    for name, record in m['files'].items():
        if references.checksum(path.parent / name) != record['sha256']:
            raise ValueError('Saved land-activity file changed; preserved')
    with rasterio.open(path.parent / 'land_activity.tif') as raster:
        codes = raster.read(1)
        if (raster.crs != rasterio.crs.CRS.from_user_input(m['grid']['crs'])
                or not raster.transform.almost_equals(Affine(*m['grid']['transform']))
                or not set(np.unique(codes).tolist()) <= {0, *LAND_CLASSES}):
            raise ValueError('Land-activity raster differs from its manifest')
    tables = {}
    for name in FILES:
        if name.endswith('.csv'):
            with (path.parent / name).open(newline='') as stream:
                tables[name[:-4]] = list(csv.DictReader(stream))
    return {'manifest': m, 'folder': path.parent, 'codes': codes, 'baseline': baseline, 'tables': tables}


def headlines(bundle):
    """Weighted summary shares from the saved tables."""
    def weighted(rows, column):
        rows = [row for row in rows if int(row['assessed_pixels'])]
        total = sum(int(row['assessed_pixels']) for row in rows)
        return sum(float(row[column]) * int(row['assessed_pixels']) for row in rows) / total if total else float('nan')
    tables = bundle['tables']
    split = tables['crop_map_by_land_activity']
    def category_share(categories):
        selected = [row for row in split if row['crop_map_category'] in categories]
        total = sum(int(row['pixels']) for row in selected)
        cropland = sum(int(row['pixels']) for row in selected if row['land_activity'] == 'active cropland')
        return cropland / total if total else float('nan')
    return {'reference_area_pixels_in_active_cropland': weighted(tables['reference_area_check'], 'active cropland'),
            'namria_noncrop_pixels_outside_active_cropland': weighted(
                [row for row in tables['namria_check'] if row['name'] in NON_CROP_TYPES], 'outside active cropland'),
            'namria_crops_pixels_in_active_cropland': weighted(
                [row for row in tables['namria_check'] if row['name'] == 'crops'], 'active cropland'),
            'crop_predictions_in_active_cropland': category_share(set(crop_baseline.CROPS.values())),
            'unknown_in_active_cropland': category_share({crop_baseline.STATUSES[2]})}


def plot_months(bundle):
    """Monthly clear observations and cycle timing in active cropland, and complete share of every window."""
    import matplotlib.pyplot as plt
    monthly = bundle['tables']['monthly_activity']
    windows = bundle['tables']['four_month_windows']
    figure, axes = plt.subplots(1, 2, figsize=(18, 6), layout='constrained', gridspec_kw={'width_ratios': [1, 1.25]})
    axis = axes[0]
    x = np.arange(12)
    axis.axvspan(-.5, 3.5, color='#fff3c4', zorder=0, label='Current model window (Jan–Apr)')
    clear = [float(row['clear_share_active_cropland']) for row in monthly]
    axis.bar(x, clear, color='#78909c', width=.7, label='Clear observations in active cropland')
    axis.plot(x, [float(row['cycle_start_share']) for row in monthly], '-o', color='#2e7d32', lw=2, label='Crop cycles starting (green-up)')
    axis.plot(x, [float(row['cycle_peak_share']) for row in monthly], '-s', color='#c62828', lw=2, label='Crop cycles peaking')
    axis.set_xticks(x, MONTHS)
    axis.set_ylim(0, 1)
    axis.set_ylabel('Share')
    axis.set_title('When is it clear, and when do crops grow? (2019–2026)', fontsize=13, fontweight='bold')
    axis.legend(loc='upper center', bbox_to_anchor=(.5, -.08), ncol=2, frameon=False, fontsize=10)
    axis.spines[['top', 'right']].set_visible(False)

    years = sorted({int(row['year']) for row in windows})
    grid = np.full((len(years), 12), np.nan)
    for row in windows:
        if int(row['composites_available']) == 8:
            grid[years.index(int(row['year'])), int(row['start_month']) - 1] = float(row['active_cropland_complete_share'])
    axis = axes[1]
    image = axis.imshow(np.ma.masked_invalid(grid), cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    axis.set_facecolor('#eeeeee')
    for (r, c), value in np.ndenumerate(grid):
        if np.isfinite(value):
            axis.text(c, r, f'{value:.0%}', ha='center', va='center', fontsize=8,
                      color='white' if value > .55 else '#212121')
        else:
            axis.text(c, r, '–', ha='center', va='center', fontsize=8, color='#757575')
    axis.add_patch(plt.Rectangle((-.5, -.5), 1, len(years), fill=False, edgecolor='#e65100', lw=2.5))
    axis.set_xticks(range(12), [window_label(m) for m in range(1, 13)], rotation=45, ha='right')
    axis.set_yticks(range(len(years)), years)
    axis.set_title('Active-cropland pixels with all 8 half-months clear, by four-month window', fontsize=13, fontweight='bold')
    figure.colorbar(image, ax=axis, shrink=.8, label='Complete share  (– = composite file missing)')
    return figure


def plot(bundle):
    """Land activity, the frozen crop map, and the crop map restricted to active cropland (display only)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    codes = bundle['codes']
    roi = codes > 0
    figure, axes = plt.subplots(1, 3, figsize=(22, 8.5), layout='constrained')
    edge = dict(edgecolor='#616161', linewidth=.6)

    def legend(axis, handles):
        axis.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.01), ncol=3, frameon=False,
                    fontsize=10, handlelength=1.6, handleheight=1.2, columnspacing=1.4)

    present = [code for code in LAND_CLASSES if (codes == code).any()]
    land_colors = [LAND_COLORS[code] for code in LAND_CLASSES]
    axes[0].imshow(np.ma.array(codes, mask=~roi), cmap=ListedColormap(land_colors).with_extremes(bad='white'),
                   norm=BoundaryNorm(np.arange(.5, len(LAND_CLASSES) + 1), len(LAND_CLASSES)), interpolation='nearest')
    axes[0].set_title('Land activity, 2019–2026', fontsize=13, fontweight='bold')
    legend(axes[0], [Patch(facecolor=LAND_COLORS[code], label=LAND_CLASSES[code].capitalize(), **edge) for code in present])

    classification, status = bundle['baseline']['data']
    crop_roi = bundle['baseline']['roi']
    labels = [*crop_baseline.CROPS.values(), crop_baseline.STATUSES[2], crop_baseline.STATUSES[3]]
    values = classification.astype(float)
    values[status == 2] = 4
    values[status == 3] = 5
    crop_colors = [CROP_COLORS[label] for label in labels] + [NOT_CROPLAND_COLOR]
    cropland_only = values.copy()
    cropland_only[codes != 1] = 6
    for axis, data, title in ((axes[1], values, 'Crop map, Jan–Apr 2024'),
                              (axes[2], cropland_only, 'Crop map inside active cropland only')):
        axis.imshow(np.ma.array(data, mask=~crop_roi), cmap=ListedColormap(crop_colors).with_extremes(bad='white'),
                    norm=BoundaryNorm(np.arange(.5, 7), 6), interpolation='nearest')
        axis.contour(crop_roi.astype(float), levels=[.5], colors='#616161', linewidths=.6)
        axis.set_title(title, fontsize=13, fontweight='bold')
    shown = [label for value, label in enumerate(labels, 1) if (crop_roi & (values == value)).any()]
    legend(axes[1], [Patch(facecolor=CROP_COLORS[label], label=label, **edge) for label in shown])
    kept = [label for value, label in enumerate(labels, 1) if (crop_roi & (cropland_only == value)).any()]
    legend(axes[2], [Patch(facecolor=CROP_COLORS[label], label=label, **edge) for label in kept]
           + [Patch(facecolor=NOT_CROPLAND_COLOR, label='Not active cropland (hidden)', **edge)])
    for axis in axes:
        axis.set_axis_off()
    return figure
