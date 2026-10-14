"""Map one FranceCrops parcel from its geolocation Parquet.

Install: python -m pip install pyarrow shapely pyproj earthengine-api
python visualize_francecrops_map.py geolocation.parquet --parcel ID --crs EPSG:4326
Add --project REGISTERED_EE_PROJECT --date 2022-06-15 for dated Sentinel-2.
CRS must be confirmed from the geolocation source, not guessed.
"""
import argparse
import json
from datetime import datetime,timedelta
from pathlib import Path

import pyarrow.parquet as pq
from shapely import wkb,wkt
from shapely.geometry import shape,mapping
from shapely.ops import transform
from pyproj import Transformer


def load_geometry(value):
    if isinstance(value,(bytes,bytearray,memoryview)):
        return wkb.loads(bytes(value))
    if isinstance(value,dict): return shape(value)
    if isinstance(value,str):
        if value.lstrip().startswith('{'): return shape(json.loads(value))
        try: return wkb.loads(value,hex=True)
        except Exception: return wkt.loads(value)
    raise ValueError('Unsupported geometry representation; expected WKB, WKT or GeoJSON')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path',help='Geolocation parquet file or folder')
    ap.add_argument('--parcel',required=True,help='ID from sample_metadata.json or parcel_timeseries report')
    ap.add_argument('--crs',required=True,help='Confirmed source CRS, e.g. EPSG:4326')
    ap.add_argument('--geometry-column',default='geometry')
    ap.add_argument('--label',default='Declared crop parcel',help='Name confirmed from benchmark/class-map, e.g. Cabbage')
    ap.add_argument('--project',help='Registered Earth Engine project for dated Sentinel-2')
    ap.add_argument('--date',help='Target acquisition date YYYY-MM-DD; requires --project')
    ap.add_argument('--days',type=int,default=15,help='Search either side of target date (default: 15)')
    ap.add_argument('--authenticate',action='store_true',help='Run interactive Earth Engine authentication')
    ap.add_argument('--output',default='francecrops_inspection')
    args=ap.parse_args()
    if bool(args.project)!=bool(args.date): ap.error('Supply both --project and --date for dated imagery')
    root=Path(args.path)
    if not root.exists(): ap.error(f'Path not found: {root}')
    files=[root] if root.is_file() else sorted(root.rglob('*.parquet'))
    geom=None; source=None
    for file in files:
        pf=pq.ParquetFile(file)
        if not {'parcel_id',args.geometry_column}.issubset(pf.schema_arrow.names):continue
        for batch in pf.iter_batches(batch_size=2048,columns=['parcel_id',args.geometry_column]):
            for row in batch.to_pylist():
                if str(row['parcel_id'])==args.parcel:
                    geom=load_geometry(row[args.geometry_column]); source=str(file.resolve()); break
            if geom is not None:break
        if geom is not None:break
    if geom is None:ap.error('Parcel not found in a geolocation file. Check ID and geometry-column name.')
    if geom.is_empty or not geom.is_valid or geom.geom_type not in ['Polygon','MultiPolygon']:
        ap.error('Geometry must be a valid nonempty Polygon or MultiPolygon')
    converter=Transformer.from_crs(args.crs,'EPSG:4326',always_xy=True)
    geom=transform(converter.transform,geom)
    left,bottom,right,top=geom.bounds
    if not (-180<=left<=right<=180 and -90<=bottom<=top<=90):ap.error('Invalid geographic bounds; verify source CRS')
    feature={'type':'Feature','geometry':mapping(geom),'properties':{'parcel_id':args.parcel,'label':args.label}}
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'selected_parcel.geojson').write_text(json.dumps(feature,indent=2),encoding='utf-8')
    satellite_url=None; info={'parcel_id':args.parcel,'label':args.label,'source':source,'source_crs':args.crs,
        'warning':'Label is supplied by user; verify against benchmark y. Annual declaration does not confirm crop on acquisition date.'}
    if args.project:
        import ee
        if args.authenticate:ee.Authenticate()
        ee.Initialize(project=args.project)
        target=datetime.fromisoformat(args.date)
        region=ee.Geometry(mapping(geom))
        collection=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region).filterDate(
            (target-timedelta(days=args.days)).isoformat(),(target+timedelta(days=args.days+1)).isoformat())
        # Rank by clear fraction over the field first, then proximity to target date.
        def score(image):
            scl=image.select('SCL')
            bad=scl.eq(1).Or(scl.eq(3)).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
            clear=bad.Not().updateMask(scl.mask())
            frac=clear.reduceRegion(ee.Reducer.mean(),region,20,maxPixels=10000000).get('SCL')
            frac=ee.Number(ee.Algorithms.If(frac,frac,0))
            distance=ee.Number(image.get('system:time_start')).subtract(ee.Date(args.date).millis()).abs().divide(86400000)
            return image.set({'field_clear_fraction':frac,'inspection_score':frac.multiply(1000).subtract(distance)})
        ranked=collection.map(score).sort('inspection_score',False)
        if ranked.size().getInfo()==0:ap.error('No Sentinel-2 scenes in search interval; increase --days')
        image=ee.Image(ranked.first())
        acquisition=ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        info.update(acquisition_date=acquisition,image_id=image.get('system:index').getInfo(),
                    field_clear_fraction=image.get('field_clear_fraction').getInfo(),
                    selection='Highest SCL clear-field fraction, then date proximity; inspect visually')
        satellite_url=image.getMapId({'bands':['B4','B3','B2'],'min':0,'max':3000})['tile_fetcher'].url_format
        title=f'Sentinel-2 L2A acquisition {acquisition}; clear fraction {info["field_clear_fraction"]:.2f}'
    else:
        title='Context satellite basemap — acquisition date unknown; not evidence of 2022 crop presence'
    config={'feature':feature,'satellite':satellite_url,'title':title}
    html='''<!doctype html><html><head><meta charset="utf-8"><title>FranceCrops parcel inspection</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>body{margin:0;font-family:Arial}#header{padding:14px}#map{height:80vh}.note{font-size:14px;color:#555}</style>
</head><body><div id="header"><strong id="title"></strong><p id="label"></p>
<p class="note">Magenta outline = declared parcel. Toggle the fill to compare the imagery. Annual labels do not verify crop identity on this date.</p></div><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const config=CONFIG;
document.getElementById('title').textContent=config.title;
document.getElementById('label').textContent=config.feature.properties.label+' — parcel '+config.feature.properties.parcel_id;
const map=L.map('map');
const context=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{attribution:'Esri and imagery providers',maxZoom:19});
const base={'Context imagery (date unknown)':context};
if(config.satellite){base['Dated Sentinel-2']=L.tileLayer(config.satellite,{attribution:'Copernicus Sentinel-2 / Google Earth Engine',maxZoom:19}).addTo(map)}else{context.addTo(map)}
const outline=L.geoJSON(config.feature,{style:{color:'#ff00ff',weight:3,fillOpacity:0}}).addTo(map);
const fill=L.geoJSON(config.feature,{style:{color:'#ff00ff',weight:0,fillOpacity:.3}});
outline.bindPopup(config.feature.properties.label+' | '+config.feature.properties.parcel_id);
map.fitBounds(outline.getBounds().pad(.8));
L.control.layers(base,{'Parcel outline':outline,'Label fill':fill},{collapsed:false}).addTo(map);
</script></body></html>'''.replace('CONFIG',json.dumps(config).replace('<','\\u003c'))
    (out/'parcel_map.html').write_text(html,encoding='utf-8')
    (out/'parcel_map_metadata.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(f'Open: {out / "parcel_map.html"}')
    print(title)
    if args.project:print('Earth Engine map access is temporary; rerun when the imagery token expires.')


if __name__=='__main__':main()
