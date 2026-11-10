"""Selective three-crop baseline, separate from Dynamic World fusion and other-crop training."""

import csv
import json
from pathlib import Path
import tempfile

from affine import Affine
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from pyproj import Transformer
from shapely import transform
from shapely.geometry import shape, mapping

from crop_pipeline.paoay import predictions, features, imagery, references, maps

RULES = {"minimum_top_probability": .6, "minimum_margin": .1}
CROPS = {1: "Carrot", 2: "Cabbage", 3: "Potato"}
STATUSES = {0: "Outside mapped boundaries", 1: "Crop-model prediction", 2: "Unknown crop type", 3: "Missing input"}
DEFERRED_METHODS = ["pooled_four_class_model", "individual_crop_model_grouped_for_reporting"]


def validate_rules(rules):
    if set(rules) != set(RULES) or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in rules.values()):
        raise ValueError("Use finite crop score cutoffs between zero and one")
    return dict(rules)


def reference_mask(reference, grid, roi):
    """Union reference polygons inside the study ROI, retaining holes and pixel centres."""
    selected = np.zeros(roi.shape, dtype=bool)
    project = Transformer.from_crs('EPSG:4326', grid['crs'], always_xy=True).transform
    for record in reference['records']:
        if (record['reference_id'] == reference['manifest']['roi_id']
                or record['geometry_status'] != 'valid' or record['geometry'] is None):
            continue
        geometry = transform(shape(record['geometry']), project, interleaved=False)
        selected |= geometry_mask([mapping(geometry)], out_shape=roi.shape,
                                  transform=Affine(*grid['transform']), invert=True)
    return selected & roi


def arrays(crop, feature, rules=None, spatial_mask=None):
    rules = validate_rules(RULES if rules is None else rules)
    with rasterio.open(feature['folder']/'pixel_status.tif') as raster:
        roi = raster.read(1) > 0
    if spatial_mask is not None:
        roi &= spatial_mask
    classification = np.zeros(roi.shape, dtype='uint8')
    status = np.zeros_like(classification); status[roi] = 3
    rows, columns = feature['rows'], feature['columns']
    score, margin = crop['top_probability'], crop['probability_margin']
    accepted = (score >= rules['minimum_top_probability']) & (margin >= rules['minimum_margin']) & (margin > 0)
    status[rows, columns] = np.where(accepted, 1, 2)
    for name, code in (('carrot', 1), ('cabbage', 2), ('potato', 3)):
        chosen = accepted & (crop['class_codes'] == crop['manifest']['class_codes'][name])
        classification[rows[chosen], columns[chosen]] = code
    classification[~roi] = 0
    status[~roi] = 0
    return np.asarray([classification, status]), roi


def area_rows(data, mask, pixel_ha):
    n = int(mask.sum())
    rows = []
    for code, label in CROPS.items():
        count = int((mask & (data[0] == code)).sum())
        rows.append({'category': label, 'kind': 'crop_model_prediction', 'pixels': count, 'area_ha': count*pixel_ha, 'roi_percent': count/n*100 if n else 0})
    for code in (2, 3):
        count = int((mask & (data[1] == code)).sum())
        rows.append({'category': STATUSES[code], 'kind': 'data_status', 'pixels': count, 'area_ha': count*pixel_ha, 'roi_percent': count/n*100 if n else 0})
    return rows


def prepare_request(prediction_handoff, output_root, rules=None):
    predictions.load_prediction_bundle(prediction_handoff)
    identity = {'version': 2, 'spatial_scope': 'reference_polygon_union', 'prediction_sha256': references.checksum(prediction_handoff),
                'rules': validate_rules(RULES if rules is None else rules), 'implementation_sha256': references.checksum(__file__)}
    key = imagery._digest(identity)[:20]
    return {'bundle_id': key, 'identity': identity, 'prediction_handoff': str(Path(prediction_handoff).resolve()),
            'folder': Path(output_root)/'bundles'/key}


def _source(request):
    if imagery._digest(request['identity'])[:20] != request['bundle_id'] or references.checksum(request['prediction_handoff']) != request['identity']['prediction_sha256']:
        raise ValueError('Crop baseline input changed; plan again')
    crop = predictions.load_prediction_bundle(request['prediction_handoff'])
    feature = features.load_feature_bundle(crop['manifest']['feature_handoff'])
    return crop, feature


def build(request):
    crop, feature = _source(request)
    folder = request['folder']; handoff = folder/'crop_baseline_handoff.json'
    if folder.exists():
        load_bundle(handoff)
        return handoff
    inputs = maps.context(request['prediction_handoff'])
    mask = reference_mask(inputs['reference'], crop['manifest']['grid'], inputs['roi'])
    if not mask.any():
        raise ValueError('No reference-polygon pixel centres inside the study ROI')
    data, roi = arrays(crop, feature, request['identity']['rules'], mask)
    grid = crop['manifest']['grid']; affine = Affine(*grid['transform'])
    hectare = abs(affine.a*affine.e-affine.b*affine.d)/10000
    summary = area_rows(data, roi, hectare)
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.crop-baseline-', dir=folder.parent) as temporary:
        staging = Path(temporary)/request['bundle_id']; staging.mkdir()
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(staging/'crop_baseline.tif','w',driver='GTiff',
                height=grid['height'],width=grid['width'],count=2,dtype='uint8',crs=grid['crs'],transform=affine,nodata=0,compress='deflate') as raster:
            raster.write(data);raster.descriptions=('classification','status');raster.write_mask(roi.astype('uint8')*255)
        with (staging/'crop_baseline_areas.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        _source(request)
        manifest={'format_version':2,'kind':'three_crop_with_unknown','status':'prepared','identity':request['identity'],'bundle_id':request['bundle_id'],
            'prediction_handoff':request['prediction_handoff'],'grid':grid,'period':crop['manifest']['period'],'class_codes':CROPS,'status_codes':STATUSES,
            'spatial_scope':'reference_polygon_union','land_cover_filter_applied':False,'model_refitted':False,'deferred_methods':DEFERRED_METHODS,
            'interpretation':'Conditional three-crop model scores; no crop/noncrop discrimination or independent local accuracy',
            'files':{name:{'sha256':references.checksum(staging/name)} for name in ('crop_baseline.tif','crop_baseline_areas.csv')}}
        (staging/'crop_baseline_handoff.json').write_text(json.dumps(manifest,indent=2)+'\n');staging.rename(folder)
    load_bundle(handoff)
    return handoff


def load_bundle(path):
    path=Path(path);m=json.loads(path.read_text())
    if (m.get('format_version') not in (1, 2) or m.get('kind')!='three_crop_with_unknown' or m.get('status')!='prepared'
            or m.get('model_refitted') is not False or m.get('land_cover_filter_applied') is not False
            or m.get('class_codes')!={str(k):v for k,v in CROPS.items()} or m.get('status_codes')!={str(k):v for k,v in ({**STATUSES, 0: 'Outside ROI'} if m.get('format_version') == 1 else STATUSES).items()}
            or set(m.get('files',{}))!={'crop_baseline.tif','crop_baseline_areas.csv'}):
        raise ValueError('Unsupported crop baseline bundle')
    crop,feature=_source({k:m[k] for k in ('identity','bundle_id','prediction_handoff')})
    if any(m[key]!=crop['manifest'][key] for key in ('grid','period')):raise ValueError('Crop baseline grid/dates changed')
    for name,record in m['files'].items():
        if references.checksum(path.parent/name)!=record['sha256']:raise ValueError('Saved crop baseline changed; files preserved')
    mask = None
    if m['format_version'] == 2:
        if m.get('spatial_scope') != 'reference_polygon_union' or m['identity'].get('spatial_scope') != 'reference_polygon_union':
            raise ValueError('Crop baseline boundary scope changed')
        inputs = maps.context(m['prediction_handoff'])
        mask = reference_mask(inputs['reference'], m['grid'], inputs['roi'])
    expected,roi=arrays(crop,feature,m['identity']['rules'],mask);g=m['grid']
    with rasterio.open(path.parent/'crop_baseline.tif') as raster:
        if (raster.count!=2 or raster.crs!=rasterio.crs.CRS.from_user_input(g['crs']) or not raster.transform.almost_equals(Affine(*g['transform']))
                or raster.descriptions!=('classification','status') or not np.array_equal(raster.read(),expected)
                or not np.array_equal(raster.dataset_mask()>0,roi)):
            raise ValueError('Crop baseline differs from verified predictions')
    with (path.parent/'crop_baseline_areas.csv').open(newline='') as stream:summary=list(csv.DictReader(stream))
    if sum(int(r['pixels']) for r in summary)!=int(roi.sum()):raise ValueError('Crop baseline area accounting changed')
    return {'manifest':m,'folder':path.parent,'data':expected,'roi':roi,'summary':summary}


def plot(bundle):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Patch
    values=bundle['data'][0].copy();status=bundle['data'][1]
    values[status==2]=4;values[status==3]=5
    colors=['#ed9137','#4b9a65','#8b70b1','#e8c873','#d9d9d9']
    figure,axis=plt.subplots(figsize=(9,7),layout='constrained')
    axis.imshow(np.ma.array(values,mask=~bundle['roi']),cmap=ListedColormap(colors).with_extremes(bad='white'),norm=BoundaryNorm(np.arange(.5,6),5),interpolation='nearest')
    axis.set_axis_off();axis.set_title('Three-crop baseline + Unknown: reference boundaries only\nConditional crop-model predictions; notes are unverified',fontsize=11)
    figure.legend(handles=[Patch(color=color,label=label) for color,label in zip(colors,[*CROPS.values(),STATUSES[2],STATUSES[3]])],loc='outside lower center',ncol=3,fontsize=9)
    return figure
