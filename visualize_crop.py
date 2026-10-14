"""Inspect a local S4A patch; outputs are exploratory, not cloud-masked training data.

python visualize_crop.py
python visualize_crop.py patch.nc --crop potato --parcel 123 --start 2020-05-01
Dependencies: numpy netCDF4 matplotlib
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from netCDF4 import Dataset, num2date

BANDS = ['B03', 'B04', 'B05', 'B08', 'B11']
CODES = {'carrot': 231, 'potato': 510, 'cabbage': 213}
# Original taxonomy: https://github.com/Orion-AI-Lab/S4A/blob/main/utils/encodings_en.py
LABEL_NAMES = {0: 'Background / unlabeled', 120: 'Maize', 150: 'Barley', 170: 'Oats',
    213: 'Cabbages', 230: 'Root bulb or tuberous vegetables', 231: 'Carrots',
    330: 'Grapes', 351: 'Apples', 352: 'Apricots', 353: 'Cherries',
    354: 'Peaches and nectarines', 355: 'Pears and quinces', 361: 'Almonds',
    438: 'Sunflower', 442: 'Olives', 510: 'Potatoes', 770: 'Peas',
    912: 'Permanent grass crops', 975: 'Fallow land',
    981: 'Other crops temporary', 998: 'Unknown crops'}


def label_picture(out, rgb, labels, parcels, selected, date):
    codes = sorted(int(c) for c in np.unique(labels[np.isfinite(labels)]) if c != 0)
    colors = {code: plt.get_cmap('tab20')(i % 20) for i,code in enumerate(codes)}
    colors.update({231: (1,0,1,1), 213: (0,1,1,1), 510: (1,.65,0,1)})
    rgba = np.zeros((*labels.shape,4))
    for code in codes:
        rgba[labels == code] = (*colors[code][:3], .38)
    edge = np.zeros(labels.shape,dtype=bool)
    edge[:-1,:] |= parcels[:-1,:] != parcels[1:,:]
    edge[:,:-1] |= parcels[:,:-1] != parcels[:,1:]
    edge &= np.isfinite(parcels) & (parcels > 0)
    boundaries = np.zeros_like(rgba); boundaries[edge] = (1,1,1,.85)
    fig,axs = plt.subplots(1,3,figsize=(20,8))
    for ax in axs:
        ax.imshow(rgb,interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
    axs[0].set_title('Satellite RGB without labels')
    for ax in axs[1:]:
        ax.imshow(rgba,interpolation='nearest'); ax.imshow(boundaries,interpolation='nearest')
    axs[1].set_title('All labels: transparent colors; white parcel edges')
    axs[2].set_title('Zoom around selected crop parcel')
    yy,xx=np.where(selected)
    margin=20
    axs[2].set_xlim(max(0,xx.min()-margin),min(labels.shape[1]-.5,xx.max()+margin))
    axs[2].set_ylim(min(labels.shape[0]-.5,yy.max()+margin),max(0,yy.min()-margin))
    for code in codes:
        ys,xs=np.where(labels==code)
        # One anchor per class; the legend explains every colored pixel.
        axs[1].text(xs[len(xs)//2],ys[len(ys)//2],str(code),fontsize=7,color='white',
                    bbox=dict(facecolor='black',alpha=.65,pad=1))
    axs[2].text(np.median(xx),np.median(yy),'Selected parcel',fontsize=9,color='white',
                ha='center',bbox=dict(facecolor='black',alpha=.7,pad=2))
    handles=[Patch(facecolor=colors[c],label=f'{c}: {LABEL_NAMES.get(c,"Unmapped category")} ({int((labels==c).sum()):,} pixels)') for c in codes]
    fig.legend(handles=handles,loc='lower center',ncol=4,fontsize=9,frameon=False)
    fig.suptitle(f'{date:%Y-%m-%d} | annual declared labels, not date-specific crop confirmation')
    fig.tight_layout(rect=(0,.24,1,.94)); fig.savefig(out/'05_satellite_labels.png',dpi=180); plt.close(fig)
    with (out/'label_legend.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['code','name','pixels'])
        for c in [0]+codes: writer.writerow([c,LABEL_NAMES.get(c,'Unmapped category'),int((labels==c).sum())])


def read(ds, name):
    return np.ma.asarray(ds.groups[name].variables[name][:], dtype=float).filled(np.nan)


def align(array, shape):
    # Native S4A grids have integer resolution ratios; nearest neighbour preserves values.
    if array.shape[-2:] == shape:
        return array
    factors = [shape[i] // array.shape[-2+i] for i in range(2)]
    if any(shape[i] != array.shape[-2+i]*factors[i] for i in range(2)):
        raise ValueError('Unsupported grid sizes; requires explicit geospatial resampling')
    return np.repeat(np.repeat(array, factors[0], axis=-2), factors[1], axis=-1)


def ratio(a, b):
    return np.divide(a-b, a+b, out=np.full_like(a, np.nan), where=np.isfinite(a+b) & ((a+b) != 0))


def indices(data):
    return {'NDVI': ratio(data['B08'], data['B04']),
            'NDRE': ratio(data['B08'], data['B05']),
            'NDMI': ratio(data['B08'], data['B11'])}


def stretch(rgb):
    result = np.zeros_like(rgb)
    for k in range(3):
        a = rgb[..., k]
        finite = a[np.isfinite(a)]
        if len(finite):
            lo, hi = np.percentile(finite, [2, 98])
            result[..., k] = np.clip((a-lo)/max(hi-lo, 1e-9), 0, 1)
    return np.nan_to_num(result)


def overlay(ax, selected):
    if selected.any() and not selected.all():
        ax.contour(selected.astype(int), levels=[0.5], colors='magenta', linewidths=.7)
    ax.set_xticks([]); ax.set_yticks([])


def add_month(year, month, offset):
    value = year*12+month-1+offset
    return datetime(value//12, value%12+1, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file', nargs='?', help='Default: the only .nc file in the current folder')
    ap.add_argument('--crop', choices=CODES, default='carrot')
    ap.add_argument('--parcel', type=int, help='Default: largest visible parcel for selected crop')
    ap.add_argument('--date', help='Nearest acquisition to YYYY-MM-DD; default: middle acquisition')
    ap.add_argument('--start', help='Four-month window start YYYY-MM-01')
    ap.add_argument('--output', default='visual_inspection')
    args = ap.parse_args()
    candidates = list(Path.cwd().glob('*.nc'))
    if not args.file and len(candidates) != 1:
        ap.error('Specify a filename when the current folder has zero or multiple .nc files')
    path = Path(args.file) if args.file else candidates[0]
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    with Dataset(path) as ds:
        labels = read(ds, 'labels'); parcels = read(ds, 'parcels')
        crop = labels == CODES[args.crop]
        ids, sizes = np.unique(parcels[crop & np.isfinite(parcels) & (parcels > 0)], return_counts=True)
        if not len(ids):
            ap.error(f'No valid {args.crop} parcels found under original encoding')
        pid = args.parcel if args.parcel is not None else int(ids[np.argmax(sizes)])
        selected = crop & (parcels == pid)
        if not selected.any():
            ap.error('Selected parcel has no pixels of the selected crop')
        inventory = [{'parcel_id': int(i), 'crop_pixels': int(s)} for i,s in zip(ids,sizes)]
        data = {}
        dates = None
        band_metadata = {}
        for band in BANDS:
            group = ds.groups[band]; var = group.variables[band]
            tv = group.variables['time']
            decoded = num2date(tv[:], tv.units, calendar=getattr(tv, 'calendar', 'standard'))
            current_dates = [datetime(d.year,d.month,d.day,d.hour,d.minute,d.second) for d in decoded]
            if dates is not None and current_dates != dates:
                raise ValueError('Band acquisition dates differ; explicit temporal alignment required')
            dates = current_dates
            data[band] = align(read(ds, band), labels.shape)
            band_metadata[band] = {a: str(var.getncattr(a)) for a in var.ncattrs()}
        rgb_blue = align(read(ds,'B02'), labels.shape) if 'B02' in ds.groups else None
        blue_dates_match = False
        if rgb_blue is not None:
            tv = ds.groups['B02'].variables['time']
            bd = num2date(tv[:],tv.units,calendar=getattr(tv,'calendar','standard'))
            blue_dates_match = [str(d) for d in bd] == [str(d) for d in decoded]
        metadata = {a: str(ds.getncattr(a)) for a in ds.ncattrs()}
    # Files can store acquisitions out of chronological order.
    order = np.argsort(np.array(dates, dtype='datetime64[s]'))
    dates = [dates[int(i)] for i in order]
    data = {name: array[order] for name, array in data.items()}
    if blue_dates_match:
        rgb_blue = rgb_blue[order]
    derived = indices(data)
    with (out/'parcel_inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['parcel_id','crop_pixels']); w.writeheader(); w.writerows(inventory)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    axs[0].imshow(labels,cmap='tab20',interpolation='nearest'); axs[0].set_title('All label codes (categorical colors)')
    axs[1].imshow(crop,cmap='gray'); axs[1].set_title(f'{args.crop} mask, code {CODES[args.crop]}')
    axs[2].imshow(selected,cmap='gray'); axs[2].set_title(f'Selected parcel {pid}: {selected.sum()} crop pixels')
    for ax in axs: overlay(ax,selected)
    fig.tight_layout(); fig.savefig(out/'01_masks.png',dpi=160); plt.close(fig)
    target=datetime.fromisoformat(args.date) if args.date else dates[len(dates)//2]
    at=min(range(len(dates)),key=lambda i:abs((dates[i]-target).total_seconds()))
    if blue_dates_match:
        rgb=stretch(np.stack([data['B04'][at],data['B03'][at],rgb_blue[at]],axis=-1))
        label_picture(out,rgb,labels,parcels,selected,dates[at])
    fields={**data,**derived}
    fig,axs=plt.subplots(2,4,figsize=(16,8))
    for ax,(name,array) in zip(axs.flat,fields.items()):
        image=array[at]; finite=image[np.isfinite(image)]
        lo,hi=(-1,1) if name in derived else (np.percentile(finite,[2,98]) if len(finite) else (0,1))
        im=ax.imshow(image,vmin=lo,vmax=hi,cmap='RdYlGn' if name in derived else 'viridis')
        overlay(ax,selected); ax.set_title(name); fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle(f'{path.name} | {dates[at]} | magenta = selected crop pixels'); fig.tight_layout()
    fig.savefig(out/'02_bands_indices.png',dpi=160); plt.close(fig)
    records=[]
    for i,date in enumerate(dates):
        row={'date':date.isoformat(),'parcel_id':pid,'crop_pixels':int(selected.sum())}
        for name,array in fields.items():
            values=array[i][selected]; values=values[np.isfinite(values)]
            row[name+'_median']=float(np.median(values)) if len(values) else None
            row[name+'_valid_pixels']=len(values)
        records.append(row)
    with (out/'parcel_timeseries.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    fig,axs=plt.subplots(2,1,figsize=(13,8),sharex=True)
    for name in BANDS: axs[0].plot(dates,[r[name+'_median'] for r in records],'.-',label=name)
    for name in derived: axs[1].plot(dates,[r[name+'_median'] for r in records],'.-',label=name)
    axs[0].set_ylabel('Stored band values'); axs[1].set_ylabel('Index (median of pixel indices)')
    for ax in axs: ax.legend(); ax.grid(alpha=.3)
    if args.start:
        start=datetime.fromisoformat(args.start)
        if start.day != 1: ap.error('--start must be the first day of a month')
        end=add_month(start.year,start.month,4)
        for ax in axs: ax.axvspan(start,end,color='gold',alpha=.2)
    fig.suptitle(f'{args.crop}, parcel {pid} | NOT cloud masked'); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(out/'03_timeline.png',dpi=160); plt.close(fig)
    # Contact sheets show every acquisition, each with its own display stretch.
    for first in range(0,len(dates),12):
        fig,axs=plt.subplots(3,4,figsize=(14,10))
        for ax in axs.flat: ax.axis('off')
        for ax,i in zip(axs.flat,range(first,min(first+12,len(dates)))):
            if blue_dates_match:
                rgb=np.stack([data['B04'][i],data['B03'][i],rgb_blue[i]],axis=-1)
                ax.imshow(stretch(rgb)); mode='RGB'
            else:
                rgb=np.stack([data['B08'][i],data['B04'][i],data['B03'][i]],axis=-1)
                ax.imshow(stretch(rgb)); mode='false color'
            overlay(ax,selected); ax.set_title(dates[i].strftime('%Y-%m-%d'))
        fig.suptitle(f'{mode} | magenta = selected crop pixels | display stretch varies by date')
        fig.tight_layout(); fig.savefig(out/f'04_acquisitions_{first+1:03d}.png',dpi=140); plt.close(fig)
    windows=[]
    for year in sorted({d.year for d in dates}):
        for month in range(1,13):
            start=add_month(year,month,0); end=add_month(year,month,4)
            if start>max(dates) or end<=min(dates): continue
            bins=[]
            for offset in range(4):
                ms=add_month(year,month,offset); me=add_month(year,month,offset+1)
                middle=ms.replace(day=16)
                for a,b in [(ms,middle),(middle,me)]:
                    bins.append(sum(a<=d<b for d in dates))
            windows.append({'start':start.date().isoformat(),'end_exclusive':end.date().isoformat(),
                            'occupied_bins':sum(n>0 for n in bins),'observations_per_bin':bins})
    (out/'inspection_metadata.json').write_text(json.dumps({'source':str(path.resolve()),'crop':args.crop,
        'parcel':pid,'source_metadata':metadata,'band_metadata':band_metadata,'four_month_windows':windows},indent=2))
    if args.start:
        window=next(w for w in windows if w['start']==args.start)
        start=datetime.fromisoformat(args.start)
        composites={name:[] for name in BANDS}
        for offset in range(4):
            ms=add_month(start.year,start.month,offset); me=add_month(start.year,start.month,offset+1)
            for a,b in [(ms,ms.replace(day=16)),(ms.replace(day=16),me)]:
                take=[i for i,d in enumerate(dates) if a<=d<b]
                for name in BANDS:
                    composites[name].append(np.nanmedian(data[name][take],axis=0) if take else np.full(labels.shape,np.nan))
        composites={name:np.stack(v) for name,v in composites.items()}
        composites.update(indices(composites))
        np.savez_compressed(out/'exploratory_8_composites.npz',**composites,labels=labels,selected_mask=selected)
        print('Selected window:',window)
    notes=[f'File: {path.resolve()}',f'Crop: {args.crop}; parcel: {pid}; crop pixels: {selected.sum()}',
           f'Acquisitions: {len(dates)}; observed months: {len({(d.year,d.month) for d in dates})}',
           f'Dates: {min(dates)} to {max(dates)}',
           'NDVI=(B08-B04)/(B08+B04); NDRE=(B08-B05)/(B08+B05); NDMI=(B08-B11)/(B08+B11).',
           '20 m bands are repeated to the 10 m label grid; this does not add spatial detail.',
           'No cloud/shadow masking or L1C-to-L2A correction is performed. Outputs are for inspection.',
           'Annual crop labels do not prove planting/harvest dates or repeated crop cycles.',
           'Band metadata is preserved for checking CRS, transforms, scaling and offsets.',
           'Four-month window bin counts are acquisition availability, not clear-pixel coverage.',
           'Optional composites use medians of stored bands, then indices of those medians.',
           'Do not use exploratory composites as final training inputs without cloud masking and processing harmonization.',
           'Keep every pixel and window from an independent field in the same data partition.',
           'Inspect parcel_inventory.csv to choose another parcel with --parcel ID.',
           'Four-month candidates (start: occupied semi-monthly bins / 8):']
    notes += [f"{w['start']}: {w['occupied_bins']}/8; counts={w['observations_per_bin']}" for w in windows]
    (out/'README.txt').write_text('\n'.join(notes),encoding='utf-8')
    print('\n'.join(notes[:5])); print(f'Results: {out}')


if __name__=='__main__':
    main()
