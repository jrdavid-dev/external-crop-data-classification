"""Retrieve Sentinel-2 L2A for the decoded polygon and export to Google Drive.

Install: python -m pip install earthengine-api shapely pyproj
List scenes:
python get_field_sentinel2.py --project YOUR_PROJECT --source-crs EPSG:4326 --authenticate
Export one acquisition near a date:
python get_field_sentinel2.py --project YOUR_PROJECT --source-crs EPSG:4326 --date 2022-06-15 --export

Supply --source-crs only after confirming the geometry's coordinate system.
Exports are asynchronous Earth Engine tasks, not direct local downloads.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import ee
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import mapping
from shapely.ops import transform

BANDS = ['B3', 'B4', 'B5', 'B8', 'B11']


def prepare(image):
    scl = image.select('SCL')
    # Retain bare soil: it matters for green-up detection. SCL screening is not perfect.
    bad = scl.eq(0).Or(scl.eq(1)).Or(scl.eq(3)).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    valid = bad.Not().updateMask(scl.mask()).rename('clear_mask')
    reflectance = image.select(BANDS + ['B2']).multiply(0.0001).updateMask(valid)
    result = reflectance
    for name, left, right in [('NDVI', 'B8', 'B4'), ('NDRE', 'B8', 'B5'), ('NDMI', 'B8', 'B11')]:
        a, b = reflectance.select(left), reflectance.select(right)
        denominator = a.add(b)
        index = a.subtract(b).divide(denominator).updateMask(denominator.neq(0)).rename(name)
        result = result.addBands(index)
    return result.addBands(valid).copyProperties(image, ['system:time_start', 'system:index', 'MGRS_TILE', 'CLOUDY_PIXEL_PERCENTAGE'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', required=True, help='Registered Earth Engine Google Cloud project ID')
    ap.add_argument('--source-crs', required=True, help='Confirmed WKT source CRS, e.g. EPSG:4326')
    ap.add_argument('--polygon', default=str(Path(__file__).resolve().parent / 'decoded_geometry' / 'polygon.wkt'))
    ap.add_argument('--start', default='2022-01-01', help='Inclusive start date')
    ap.add_argument('--end', default='2023-01-01', help='Exclusive end date')
    ap.add_argument('--date', help='Choose nearest qualifying actual acquisition to YYYY-MM-DD')
    ap.add_argument('--min-clear', type=float, default=0.8, help='Minimum SCL clear-field fraction for export, default 0.8; exploratory setting')
    ap.add_argument('--authenticate', action='store_true')
    ap.add_argument('--export', action='store_true', help='Submit one masked acquisition to Google Drive')
    ap.add_argument('--drive-folder', default='Benguet_External_Sentinel2')
    ap.add_argument('--output', default='sentinel2_field')
    args = ap.parse_args()
    if args.export and not args.date: ap.error('--export requires --date')
    if not 0 <= args.min_clear <= 1: ap.error('--min-clear must be between 0 and 1')
    start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    if start >= end: ap.error('--end must be later than --start')
    path = Path(args.polygon)
    if not path.exists(): ap.error(f'WKT file does not exist: {path}; run decode_pasted_geometry.py first')
    polygon = wkt.loads(path.read_text(encoding='utf-8'))
    if polygon.is_empty or not polygon.is_valid or polygon.geom_type not in ['Polygon', 'MultiPolygon']:
        ap.error('Expected a valid nonempty polygon')
    polygon = transform(Transformer.from_crs(args.source_crs, 'EPSG:4326', always_xy=True).transform, polygon)
    xmin, ymin, xmax, ymax = polygon.bounds
    if not (-180 <= xmin <= xmax <= 180 and -90 <= ymin <= ymax <= 90):
        ap.error('Invalid geographic bounds; confirm the source CRS')
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.authenticate: ee.Authenticate()
    ee.Initialize(project=args.project)
    field = ee.Geometry(mapping(polygon), proj='EPSG:4326', geodesic=False)
    source = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(field).filterDate(args.start, args.end)
    collection = source.map(prepare)

    def record(image):
        image = ee.Image(image)
        clear = image.select('clear_mask').reduceRegion(ee.Reducer.mean(), field, scale=20, maxPixels=10000000).get('clear_mask')
        return ee.Feature(None, {'image_index': image.get('system:index'),
            'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
            'timestamp_ms': image.get('system:time_start'), 'tile': image.get('MGRS_TILE'),
            'scene_cloud_percent': image.get('CLOUDY_PIXEL_PERCENTAGE'), 'field_clear_fraction': clear})

    # Only small metadata is transferred to the computer, not full imagery arrays.
    info = ee.FeatureCollection(collection.toList(collection.size()).map(record)).getInfo()
    rows = sorted([f['properties'] for f in info['features']], key=lambda r: r['timestamp_ms'])
    columns = ['image_index', 'date', 'timestamp_ms', 'tile', 'scene_cloud_percent', 'field_clear_fraction']
    with (out / 'acquisitions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    (out / 'field.geojson').write_text(json.dumps({'type': 'Feature', 'geometry': mapping(polygon), 'properties': {}}, indent=2))
    run = {'collection': 'COPERNICUS/S2_SR_HARMONIZED', 'source_crs': args.source_crs,
           'start_inclusive': args.start, 'end_exclusive': args.end, 'scenes_found': len(rows),
           'mask': 'SCL invalid, saturated, shadow, cloud, cirrus and snow classes excluded',
           'min_clear': args.min_clear, 'warning': 'SCL-only masking requires visual QA; threshold is not a validated growth-detection rule.'}
    print(f'Found {len(rows)} actual image granules. Metadata: {out / "acquisitions.csv"}')
    if args.date:
        target = datetime.fromisoformat(args.date)
        candidates = [r for r in rows if r.get('field_clear_fraction') is not None and r['field_clear_fraction'] >= args.min_clear]
        if not candidates:
            print('No scenes meet the clear-field threshold. No export submitted.')
        else:
            chosen = min(candidates, key=lambda r: (abs((datetime.fromisoformat(r['date']) - target).days), -r['field_clear_fraction']))
            run['selected_scene'] = chosen
            distance = abs((datetime.fromisoformat(chosen['date']) - target).days)
            print(f"Selected actual date: {chosen['date']} ({distance} days from request); clear fraction: {chosen['field_clear_fraction']:.3f}")
            image = ee.Image(collection.filter(ee.Filter.eq('system:index', chosen['image_index'])).first())
            if args.export:
                original = ee.Image(source.filter(ee.Filter.eq('system:index', chosen['image_index'])).first())
                projection = original.select('B3').projection().getInfo()
                prefix = 'field_S2_' + chosen['image_index'].replace('/', '_')
                task = ee.batch.Export.image.toDrive(image=image.select(BANDS + ['NDVI', 'NDRE', 'NDMI', 'B2', 'clear_mask']).clip(field).toFloat(),
                    description=prefix, folder=args.drive_folder, fileNamePrefix=prefix, region=field,
                    crs=projection['crs'], crsTransform=projection['transform'], maxPixels=100000000,
                    fileFormat='GeoTIFF', formatOptions={'cloudOptimized': True})
                task.start()
                run['export_task_id'] = task.id
                run['export_status_at_submission'] = task.status()
                run['export_band_order'] = BANDS + ['NDVI', 'NDRE', 'NDMI', 'B2', 'clear_mask']
                print(f'Task submitted: {task.id}. Drive folder: {args.drive_folder}')
                print('Check tasks at https://code.earthengine.google.com/tasks ; submission does not confirm completion.')
    (out / 'run_metadata.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    print('Exports are individual acquisitions; eight semi-monthly composites and green-up detection remain separate steps.')


if __name__ == '__main__':
    main()
