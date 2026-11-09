# Notebook workflow

Notebooks **01–03** share six sections. Review setup, then **Restart Kernel → Run All**. Opening a notebook starts no work.

| Section | What happens |
| --- | --- |
| 1. Setup and configuration | Choose reference mode, review the shared 100-per-crop round limit and local paths |
| 2. Obtain references | Reuse valid local inputs, or obtain missing inputs when enabled |
| 3. Validate references | Apply the source's label, identity, geometry and date checks; report exclusions |
| 4. Prepare the collection round | Resume original saved plans and add permitted candidates; prepare masks and show the combined workload |
| 5. Collect annual imagery | Submit or collect cloud jobs, verify results and publish annual TIFFs |
| 6. Summary and next action | Show parcels by crop, saved tiles, cloud states, paths and what to run next |

## Reference modes

The same `reference_mode` choices apply to every source:

- **`reuse` (default):** use local references without new reference downloads. Satellite collection remains controlled by `run_satellite_collection`.
- **`download_missing`:** obtain missing reference inputs. Use this for a fresh clone. Sen4AgriNet reads selected masks within an explicit patch cap, not whole imagery NetCDFs.
- **`offline`:** use local references and installed dependencies only. No package installation, sign-in or satellite calls occur. Missing inputs produce an error.

Setup can install missing dependencies in the first two modes. Existing settings and source inputs are preserved. An offline/disabled inspection does not overwrite the saved state of pending jobs.

## Source differences

| Notebook | Crops selected | Reference preparation | Edge handling |
| --- | --- | --- | --- |
| [01 Sen4AgriNet](../notebooks/01_sen4agrinet_preparation.ipynb) | Carrot, potato | Pinned catalogue and bounded label/parcel-mask extraction | Declared crop-mask cells; **0 m additional buffer** |
| [02 FranceCrops](../notebooks/02_francecrops_preparation.ipynb) | Cabbage, table potato | Pinned class-map, label and footprint tables | **10 m inward buffer** on parcel polygons |
| [03 CropHarvest](../notebooks/03_cropharvest_preparation.ipynb) | Available carrot, cabbage, potato | Pinned labels; source, date, duplicate and geometry audit | **10 m inward buffer** on parcel polygons |

These existing scientific rules remain different deliberately; shared headings do not imply equivalent boundary uncertainty. No source's checks are relaxed to meet targets. See [methodology](methodology.md).

## Resume and assess

1. Run the source notebooks. Rerun while jobs are pending; inspect errors before retrying. Submitted cloud jobs continue after Run All returns, but downloading and TIFF creation need a local run.
2. Existing selections, plans, ledgers and locked partitions remain authoritative. Older batches above 100 are preserved. Notebook layout or summary changes do not replace their plans. New settings apply to new selections; mask repair uses the original configuration and stops if it cannot reproduce it safely.
3. Once launched batches finish, run **04 Four-month windows**. It can assess partial source coverage, combines completed rounds and reports usable samples and deficits.
4. Rerun sources for deficient crop/partition combinations, then 04. Continue to **05 Feature engineering → 06 Model training** only when `Targets met: True`.

Per crop, initial targets are **1,200 training pixels from 60 parcels**, **400 validation pixels from 20 parcels**, and **400 reserved-test pixels from 20 parcels**, with at most 20 retained pixels per parcel. These budgets do not guarantee accuracy.

If Sen4AgriNet runs out of reference candidates, increasing the imagery batch limit will not help. Deliberately increase `reference_target_parcels_per_crop` and, if needed, `reference_scan_max_patches`, then use `download_missing`. Earlier selections/jobs stay intact; an empty selection is retained as an audit record when an expanded pool supplies a supplemental batch.

## Results and troubleshooting

The final summary lists batch states, reported verified tiles, queued/running/failed jobs, campaign/TIFF/log paths and a next action. Unavailable counts are shown as unavailable. `completed` confirms saved collection files; notebook 04 still checks timing and quality. `pending_retry` requires a later collection/retry; `disabled` is preparation only; a skipped source explains whether targets are met or candidates are exhausted.

Default annual TIFFs are under `sen4agrinet_work/annual_tifs/<source>/<year>/<campaign>/round_<number>/`. Each tile stores 24 half-month periods × seven measurements. Sampling masks and Parquet transport caches are separate. Use the printed paths for the exact run.

For missing references, check the mode and paths. For errors, inspect the detail log and traceback. Keep quality thresholds unchanged when usable yield is low. Revised windows can reuse annual histories; cross-year windows need explicit adjacent-year imagery.
