"""Reversible spectral/land-cover review flags; never infer non-target crop labels."""
import csv
import json
from pathlib import Path
import tempfile

from affine import Affine
import numpy as np
import rasterio

from crop_pipeline.paoay import crop_baseline, features, imagery, maps, references

RULES = {'maximum_ndvi_below': 0.2}
FLAGS = {1: 'Weak vegetation across all eight bins', 2: 'Supported water context',
         4: 'Supported built-up context', 8: 'Supported woody context',
         16: 'Incomplete spectral evidence', 32: 'Supported bare context (not verified fallow)'}
INDICES = ('NDVI', 'NDRE', 'NDMI', 'SAVI')
BANDS = [f'{index}_{statistic}' for index in INDICES for statistic in ('minimum', 'maximum', 'mean', 'amplitude')] + ['valid_bins', 'review_flags']
NODATA = -9999.


def validate_rules(rules):
    if (set(rules) != set(RULES) or type(rules['maximum_ndvi_below']) not in (int, float)
            or not np.isfinite(rules['maximum_ndvi_below']) or not -1 <= rules['maximum_ndvi_below'] <= 1):
        raise ValueError('Use a finite NDVI diagnostic threshold from -1 to 1')
    return dict(rules)


def diagnose(cube, names, mask, context_classes, context_status, soil_adjustment, rules):
    """Summarize observed indices only; these diagnostics never modify crop scores."""
    rules = validate_rules(rules)
    if cube.shape[0] != 8 or cube.shape[2:] != mask.shape:
        raise ValueError('Screening requires eight aligned bins')
    raw = {name: cube[:, names.index(name)] for name in ('B2','B3','B4','B5','B8','B11','clear_count')}
    valid = (raw['clear_count'] >= 1) & mask
    for name, value in raw.items():
        valid &= np.isfinite(value) & (value != NODATA)
    reflectance = {name: value*.0001 for name,value in raw.items() if name != 'clear_count'}
    nir, red = reflectance['B8'], reflectance['B4']
    with np.errstate(divide='ignore', invalid='ignore'):
        indices = [(nir-red)/(nir+red),
                   (nir-reflectance['B5'])/(nir+reflectance['B5']),
                   (nir-reflectance['B11'])/(nir+reflectance['B11']),
                   (1+soil_adjustment)*(nir-red)/(nir+red+soil_adjustment)]
    for values in indices:
        valid &= np.isfinite(values)
    count = valid.sum(axis=0)
    stats = []
    for values in indices:
        minimum = np.where(valid, values, np.inf).min(axis=0)
        maximum = np.where(valid, values, -np.inf).max(axis=0)
        mean = np.divide(np.where(valid,values,0).sum(axis=0),count,out=np.zeros(mask.shape),where=count>0)
        with np.errstate(invalid='ignore'):
            amplitude = maximum-minimum
        stats.extend(np.where(count>0,value,NODATA) for value in (minimum,maximum,mean,amplitude))
    flags = np.zeros(mask.shape, dtype='uint8')
    flags[mask & (count < 8)] |= 16
    flags[mask & (count == 8) & (stats[1] < rules['maximum_ndvi_below'])] |= 1
    supported = mask & (context_status == 1)
    for code, bit in ((6,4),(5,8),(7,32)):
        flags[supported & (context_classes == code)] |= bit
    # The thesis context groups water with other excluded types; water must be supplied separately.
    return np.asarray(stats + [count, flags], dtype='float32')


def prepare_request(baseline_handoff, map_handoff, output_root, rules=None):
    baseline = crop_baseline.load_bundle(baseline_handoff)
    context = maps.load_map_bundle(map_handoff)
    if (baseline['manifest']['prediction_handoff'] != context['manifest']['prediction_handoff']
            or baseline['manifest']['format_version'] != 2 or context['manifest']['format_version'] != maps.VERSION):
        raise ValueError('Use matching boundary-limited crop and current context bundles')
    identity = {'version':1, 'baseline_sha256':references.checksum(baseline_handoff),
                'map_sha256':references.checksum(map_handoff), 'rules':validate_rules(RULES if rules is None else rules),
                'implementation_sha256':references.checksum(__file__)}
    key=imagery._digest(identity)[:20]
    return {'bundle_id':key, 'identity':identity, 'baseline_handoff':str(Path(baseline_handoff).resolve()),
            'map_handoff':str(Path(map_handoff).resolve()), 'folder':Path(output_root)/'bundles'/key}


def _inputs(request):
    if imagery._digest(request['identity'])[:20] != request['bundle_id']:
        raise ValueError('Screening request changed')
    for key,sha in (('baseline_handoff','baseline_sha256'),('map_handoff','map_sha256')):
        if references.checksum(request[key]) != request['identity'][sha]:
            raise ValueError('Screening dependency changed; existing files preserved')
    baseline=crop_baseline.load_bundle(request['baseline_handoff'])
    context=maps.load_map_bundle(request['map_handoff'])
    if (baseline['manifest']['format_version'] != 2
            or context['manifest']['format_version'] != maps.VERSION
            or baseline['manifest']['prediction_handoff'] != context['manifest']['prediction_handoff']):
        raise ValueError('Screening requires matching current crop/context predictions')
    return baseline,context


def build(request):
    baseline,context=_inputs(request)
    folder=request['folder'];handoff=folder/'screening_handoff.json'
    if folder.exists():
        load_bundle(handoff)
        return handoff
    inputs=maps.context(baseline['manifest']['prediction_handoff'])
    feature=inputs['feature']
    image=imagery.load_imagery_bundle(feature['manifest']['imagery_handoff'])
    cubes=[]
    for path in image['rasters']:
        with rasterio.open(path) as raster:
            cubes.append(raster.read())
    cube=np.stack(cubes)
    recipe=features.load_model_recipe(feature['manifest']['model_manifest'])['recipe']
    with rasterio.open(context['rasters']['period_summary']) as raster:
        classes,status=raster.read(1),raster.read(2)
    data=diagnose(cube,imagery.BANDS,baseline['roi'],classes,status,
                  recipe['feature_policy']['savi_soil_adjustment'],request['identity']['rules'])
    # Water requires repeated explicit water support, never the grouped excluded-cover code.
    waters,states,supports=[],[],[]
    for path in inputs['guidance']['rasters']:
        with rasterio.open(path) as raster:
            waters.append(raster.read(1)==0);states.append(raster.read(6)==4)
            supports.append(np.isin(raster.read(6),[3,4]))
    water=np.asarray(waters)&np.asarray(states)
    supported=np.asarray(supports)
    stable_water=(water.sum(axis=0)>=context['manifest']['identity']['rules']['minimum_supported_bins']) & ~(supported & ~np.asarray(waters)).any(axis=0)
    flags=data[-1].astype('uint8');flags[baseline['roi'] & stable_water] |= 2;data[-1]=flags
    grid=baseline['manifest']['grid'];transform=Affine(*grid['transform'])
    pixel_ha=abs(transform.a*transform.e-transform.b*transform.d)/10000
    rows=[]
    for record in inputs['reference']['records']:
        if record['reference_id']==inputs['reference']['manifest']['roi_id']:continue
        from crop_pipeline.paoay.review import area_mask
        mask,_=area_mask(record,grid,baseline['roi'])
        for bit,label in FLAGS.items():
            count=int((mask & ((flags & bit)>0)).sum())
            rows.append({'reference_id':record['reference_id'],'name':record['name'],'flag':label,
                         'pixels':count,'area_ha':count*pixel_ha,'boundary_percent':100*count/int(mask.sum()) if mask.any() else 0})
    data[:,~baseline['roi']]=NODATA
    folder.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.screening-',dir=folder.parent) as temporary:
        staging=Path(temporary)/request['bundle_id'];staging.mkdir()
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(staging/'screening.tif','w',driver='GTiff',
                height=grid['height'],width=grid['width'],count=len(BANDS),dtype='float32',crs=grid['crs'],
                transform=transform,nodata=NODATA,compress='deflate') as raster:
            raster.write(data);raster.descriptions=tuple(BANDS);raster.write_mask(baseline['roi'].astype('uint8')*255)
        with (staging/'boundary_flags.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['reference_id','name','flag','pixels','area_ha','boundary_percent']);writer.writeheader();writer.writerows(rows)
        manifest={'format_version':1,'kind':'paoay_screening_flags','status':'prepared',
                  **{key:request[key] for key in ('bundle_id','identity','baseline_handoff','map_handoff')},
                  'policy':'flags_only_no_hard_exclusions','model_refitted':False,'bands':BANDS,'flags':FLAGS,
                  'grid':grid,'period':baseline['manifest']['period'],
                  'interpretation':'Illustrative review flags, not validated noncrop labels; flag areas overlap.',
                  'files':{name:{'sha256':references.checksum(staging/name)} for name in ('screening.tif','boundary_flags.csv')}}
        _inputs(request)
        (staging/'screening_handoff.json').write_text(json.dumps(manifest,indent=2)+'\n');staging.rename(folder)
    load_bundle(handoff)
    return handoff


def load_bundle(path):
    path=Path(path);m=json.loads(path.read_text())
    if (m.get('format_version')!=1 or m.get('kind')!='paoay_screening_flags' or m.get('status')!='prepared'
            or m.get('policy')!='flags_only_no_hard_exclusions' or m.get('model_refitted') is not False
            or m.get('bands')!=BANDS or m.get('flags')!={str(k):v for k,v in FLAGS.items()} or set(m.get('files',{}))!={'screening.tif','boundary_flags.csv'}):
        raise ValueError('Unsupported screening bundle')
    baseline,context=_inputs(m)
    validate_rules(m['identity']['rules'])
    if any(m[key]!=baseline['manifest'][key] for key in ('grid','period')):
        raise ValueError('Screening grid or dates changed')
    for name,record in m['files'].items():
        if references.checksum(path.parent/name)!=record['sha256']:raise ValueError('Saved screening changed; preserved')
    with rasterio.open(path.parent/'screening.tif') as raster:
        data=raster.read()
        if (raster.descriptions!=tuple(BANDS) or raster.crs!=rasterio.crs.CRS.from_user_input(m['grid']['crs'])
                or not raster.transform.almost_equals(Affine(*m['grid']['transform']))
                or not np.array_equal(raster.dataset_mask()>0,baseline['roi'])
                or data.shape!=(len(BANDS),*baseline['roi'].shape)):
            raise ValueError('Screening raster differs from source boundaries')
    mask=baseline['roi']
    if (not np.isfinite(data).all() or np.any(data[:,~mask]!=NODATA)
            or np.any((data[-2,mask]<0)|(data[-2,mask]>8))
            or np.any(data[-2,mask]!=np.floor(data[-2,mask]))
            or np.any((data[-1,mask]<0)|(data[-1,mask]>63))
            or np.any(data[-1,mask]!=np.floor(data[-1,mask]))):
        raise ValueError('Invalid screening values or outside-boundary pixels')
    with (path.parent/'boundary_flags.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    return {'manifest':m,'folder':path.parent,'data':data,'rows':rows}
