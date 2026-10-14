"""Count FranceCrops benchmark parcels and inspect selected pixel time series.

Install: python -m pip install numpy pyarrow matplotlib
Run: python inspect_francecrops.py "E:\\ground_truth\\FranceCrops\\data\\benchmark"
One file: python inspect_francecrops.py "path\\train.parquet" --crop cabbage
Input arrays contain interpolated values; this is exploratory inspection.
"""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

NAMES = ['Spring oat','Winter durum wheat','Winter soft wheat','Non-fodder beet / Swiss chard',
         'Cabbage','Winter rapeseed','Faba bean','Fiber flax','Maize','Winter barley',
         'Permanent grassland','Spring pea','Table potato','Ryegrass','Winter rye',
         'Sorghum','Soybean','Buckwheat','Sunflower','Winter triticale']
CROPS = {'cabbage':4,'potato':12}
BAND_ORDER = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
INPUTS = ['B3','B4','B5','B8','B11']
DATES = [datetime(2022,2,1)+timedelta(days=5*i) for i in range(60)]


def ratio(a,b):
    return np.divide(a-b,a+b,out=np.full_like(a,np.nan),where=np.isfinite(a+b)&((a+b)!=0))


def inspect_sample(row, source, out, start):
    x=np.asarray(row['x'],dtype=np.float32)
    if x.shape != (100,60,12):
        raise ValueError(f"Expected (100,60,12), got {x.shape}; check dataset version")
    x=x/10000.0
    bands={name:x[:,:,BAND_ORDER.index(name)] for name in INPUTS}
    values={**bands,'NDVI':ratio(bands['B8'],bands['B4']),
            'NDRE':ratio(bands['B8'],bands['B5']),'NDMI':ratio(bands['B8'],bands['B11'])}
    with (out/'sample_timeseries.csv').open('w',newline='') as f:
        fields=['date']+[f'{n}_{s}' for n in values for s in ['p25','median','p75']]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i,date in enumerate(DATES):
            record={'date':date.date().isoformat()}
            for name,a in values.items():
                for stat,v in zip(['p25','median','p75'],np.nanpercentile(a[:,i],[25,50,75])):
                    record[f'{name}_{stat}']=float(v)
            w.writerow(record)
    fig,axs=plt.subplots(2,1,figsize=(13,8),sharex=True)
    for ax,names in [(axs[0],INPUTS),(axs[1],['NDVI','NDRE','NDMI'])]:
        for name in names:
            q=np.nanpercentile(values[name],[25,50,75],axis=0)
            line,=ax.plot(DATES,q[1],'.-',label=name)
            ax.fill_between(DATES,q[0],q[2],alpha=.10,color=line.get_color())
        ax.legend(); ax.grid(alpha=.3)
    axs[0].set_ylabel('L2A reflectance'); axs[1].set_ylabel('Index')
    axs[1].set_xlabel('Shading = middle 50% of sampled pixels; values include interpolation')
    if start:
        a=datetime.fromisoformat(start)
        end=month_offset(a,4)
        for ax in axs: ax.axvspan(a,end,color='gold',alpha=.15)
    fig.suptitle(f"{NAMES[int(row['y'])]} parcel {row['parcel_id']} | FranceCrops 2022")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out/'sample_timeline.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(16,8))
    for ax,(name,a) in zip(axs.flat,values.items()):
        im=ax.imshow(a,aspect='auto',interpolation='nearest',
                     vmin=-1 if name not in INPUTS else None,vmax=1 if name not in INPUTS else None,
                     cmap='RdYlGn' if name not in INPUTS else 'viridis')
        ax.set_title(name); ax.set_xlabel('Five-day time step'); ax.set_ylabel('Sampled pixel index')
        fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle('Pixel time-series heatmaps — rows are NOT geographic image rows')
    fig.tight_layout(); fig.savefig(out/'sample_heatmaps.png',dpi=140); plt.close(fig)
    meta={'file':str(source),'parcel_id':str(row['parcel_id']),'class_id':int(row['y']),
          'class_name':NAMES[int(row['y'])],'shape':list(x.shape),'band_order':BAND_ORDER,
          'date_start':str(DATES[0].date()),'date_end':str(DATES[-1].date()),
          'warning':'Dates are aligned and include interpolated observations; original clear counts are unavailable in x.'}
    if start:
        a=datetime.fromisoformat(start)
        if a.day!=1: raise ValueError('--start must be YYYY-MM-01')
        end=month_offset(a,4)
        if a<DATES[0] or end>DATES[-1]+timedelta(days=1):
            raise ValueError('Requested window is not fully within the supplied timeline')
        composite={name:[] for name in INPUTS}; counts=[]; bounds=[]
        for offset in range(4):
            ms=month_offset(a,offset); me=month_offset(a,offset+1)
            for left,right in [(ms,ms.replace(day=16)),(ms.replace(day=16),me)]:
                take=[i for i,d in enumerate(DATES) if left<=d<right]
                counts.append(len(take)); bounds.append([str(left.date()),str(right.date())])
                for name in INPUTS:
                    composite[name].append(np.nanmedian(bands[name][:,take],axis=1) if take else np.full(100,np.nan))
        composite={n:np.stack(v,axis=1) for n,v in composite.items()}
        composite.update(NDVI=ratio(composite['B8'],composite['B4']),
                         NDRE=ratio(composite['B8'],composite['B5']),
                         NDMI=ratio(composite['B8'],composite['B11']))
        feature_names=[f'{n}_t{i+1}' for i in range(8) for n in values]
        matrix=np.column_stack([composite[n][:,i] for i in range(8) for n in values])
        with (out/'exploratory_64_features.csv').open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['parcel_id','pixel_index','class_id']+feature_names)
            for i,features in enumerate(matrix): w.writerow([row['parcel_id'],i,row['y']]+features.tolist())
        meta['window']={'start':start,'end_exclusive':str(end.date()),'bin_bounds':bounds,
                        'aligned_steps_per_bin':counts,'includes_interpolated_values':True}
    (out/'sample_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')


def month_offset(date,offset):
    month=date.year*12+date.month-1+offset
    return datetime(month//12,month%12+1,1)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path',nargs='?',default='FranceCrops',help='One benchmark parquet or folder recursively searched')
    ap.add_argument('--crop',choices=CROPS,default='cabbage')
    ap.add_argument('--parcel',help='Parcel ID to visualize; default: first matching parcel')
    ap.add_argument('--start',help='Optional four-month start YYYY-MM-01, e.g. 2022-04-01')
    ap.add_argument('--output',default='francecrops_inspection')
    args=ap.parse_args(); root=Path(args.path)
    if not root.exists(): ap.error(f'Input does not exist: {root}')
    files=[root] if root.is_file() else sorted(root.rglob('*.parquet'))
    if not files: ap.error('No parquet files found')
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    totals=Counter(); by_file=[]; errors=[]; skipped=[]; chosen=None; chosen_file=None
    # Read only labels/IDs for totals; load one x row group later for visualization.
    for file in files:
        try:
            pf=pq.ParquetFile(file); schema=pf.schema_arrow.names
            if not {'x','y','parcel_id'}.issubset(schema):
                skipped.append(str(file)); continue
            counts=Counter()
            for batch in pf.iter_batches(batch_size=2048,columns=['y','parcel_id']):
                for row in batch.to_pylist():
                    y=int(row['y'])
                    if y<0 or y>=len(NAMES): raise ValueError(f'Unexpected class ID {y}')
                    counts[y]+=1
            totals.update(counts)
            by_file.extend({'file':str(file),'class_id':y,'class_name':NAMES[y],'parcel_rows':n} for y,n in sorted(counts.items()))
            if chosen is None and counts[CROPS[args.crop]]:
                for group in range(pf.num_row_groups):
                    lookup=pf.read_row_group(group,columns=['y','parcel_id']).to_pylist()
                    index=next((i for i,r in enumerate(lookup) if int(r['y'])==CROPS[args.crop] and
                                (args.parcel is None or str(r['parcel_id'])==args.parcel)),None)
                    if index is not None:
                        chosen=pf.read_row_group(group,columns=['x','y','parcel_id']).slice(index,1).to_pylist()[0]
                        chosen_file=file; break
            print(f'{file.name}: {sum(counts.values()):,} parcel rows; cabbage={counts[4]:,}, potato={counts[12]:,}',flush=True)
        except Exception as exc:
            errors.append({'file':str(file),'error':f'{type(exc).__name__}: {exc}'})
            print(f'Failed: {file}: {exc}',flush=True)
    with (out/'class_counts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['file','class_id','class_name','parcel_rows']); w.writeheader(); w.writerows(by_file)
    summary={'counts':{NAMES[y]:n for y,n in sorted(totals.items())},'files_scanned':len(files),
             'files_skipped_nonbenchmark':skipped,'errors':errors,'counts_are':'parcel rows, not independently deduplicated fields',
             'carrot':'Not present in the published 20-class benchmark'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if totals:
        fig,ax=plt.subplots(figsize=(12,7)); keys=sorted(totals)
        ax.barh([NAMES[k] for k in keys],[totals[k] for k in keys]); ax.set_xlabel('Parcel rows in scanned benchmark files')
        fig.tight_layout(); fig.savefig(out/'class_distribution.png',dpi=150); plt.close(fig)
    sample_error=None
    if chosen is not None:
        try: inspect_sample(chosen,chosen_file,out,args.start)
        except Exception as exc: sample_error=str(exc); print('Sample inspection failed:',exc)
    else: print('No matching sample found; check --crop, --parcel and downloaded files.')
    notes=['FranceCrops inspection',f'Input: {root.resolve()}',f'Parcel row totals: {summary["counts"]}',
           f'Errors: {len(errors)}; skipped helper files: {len(skipped)}',
           'Counts are parcel rows; duplicate downloads/splits are not deduplicated.',
           'Only files with x, y, parcel_id are scanned; metadata/class_map/geolocation helpers are skipped.',
           'Cabbage class=4; table potato class=12. Carrot is absent from the benchmark class list.',
           'The 100 sampled pixels are not a spatial image grid. Do not reshape them into an RGB satellite picture.',
           'Values are L2A scaled by 10000 and include linearly interpolated missing observations.',
           'Aligned five-day step counts are NOT counts of original cloud-free satellite observations.',
           'Use geolocation/RPG boundaries to retrieve original L2A imagery for overlays and reliable growth inspection.',
           'No verified planting/harvest dates or validated green-up detector are produced.',
           'Optional 64-feature table is exploratory: half-month medians include interpolation.',
           'Do not use parcel IDs as model features. Keep pixels/windows of each field in one partition.',
           'Dataset card: https://huggingface.co/datasets/saget-antoine/francecrops']
    if sample_error: notes.append('Sample error: '+sample_error)
    (out/'summary.txt').write_text('\n'.join(notes),encoding='utf-8')
    print(f'Reports saved to {out}')
    return 1 if errors or sample_error or not totals else 0


if __name__=='__main__':
    raise SystemExit(main())
