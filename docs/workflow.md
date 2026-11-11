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
| [01 Sen4AgriNet](../notebooks/preprocessing/02_sen4agrinet_preparation.ipynb) | Carrot, potato | Pinned catalogue and bounded label/parcel-mask extraction | Declared crop-mask cells; **0 m additional buffer** |
| [02 FranceCrops](../notebooks/preprocessing/03_francecrops_preparation.ipynb) | Cabbage, table potato | Pinned class-map, label and footprint tables | **10 m inward buffer** on parcel polygons |
| [03 CropHarvest](../notebooks/preprocessing/04_cropharvest_test_preparation.ipynb) | Available carrot, cabbage, potato | Pinned labels; source, date, duplicate and geometry audit | **10 m inward buffer** on parcel polygons |

These existing scientific rules remain different deliberately; shared headings do not imply equivalent boundary uncertainty. No source's checks are relaxed to meet targets. See [methodology](methodology.md).

## Resume and assess

1. Run the source notebooks. Rerun while jobs are pending; inspect errors before retrying. Submitted cloud jobs continue after Run All returns, but downloading and TIFF creation need a local run.
2. Existing selections, plans, ledgers and locked partitions remain authoritative. Older batches above 100 are preserved. Notebook layout or summary changes do not replace their plans. New settings apply to new selections; mask repair uses the original configuration and stops if it cannot reproduce it safely.
3. Once launched batches finish, run **04 Four-month windows**. It can assess partial source coverage, combines completed rounds and reports usable samples and deficits.
4. Rerun sources for deficient crop/partition combinations when more data is wanted. For **05–06**, use the same collection and analysis mode as 04: `pilot` permits unmet collection budgets; `strict` requires `Targets met: True`. Both keep complete-data checks and locked partitions.

Per crop, initial targets are **1,200 training pixels from 60 parcels**, **400 validation pixels from 20 parcels**, and **400 reserved-test pixels from 20 parcels**, with at most 20 retained pixels per parcel. These budgets do not guarantee accuracy.

If Sen4AgriNet runs out of reference candidates, increasing the imagery batch limit will not help. Deliberately increase `reference_target_parcels_per_crop` and, if needed, `reference_scan_max_patches`, then use `download_missing`. Earlier selections/jobs stay intact; an empty selection is retained as an audit record when an expanded pool supplies a supplemental batch.

## Analysis and training: notebooks 04–06

Each notebook uses concise setup, input checks, processing, saved outputs and a final summary. They read local files; no imagery downloads are needed.

| Notebook | Process | Handoff |
| --- | --- | --- |
| 04 Four-month windows | Audit imagery, assess timing, select complete windows, report usable counts and deficits | `window_handoff.json` |
| 05 Feature engineering | Verify window inputs and compute the 72/48/120-predictor designs once | `feature_handoff.json` |
| 06 Model training | Verify features, train three forests, select on validation, evaluate external holdout | Selected model, recipe, metrics and checksums |

`analysis_mode = "pilot"` is the default. It reports unmet collection targets but requires finite complete windows, at least five usable source parcel groups per crop, and every crop in each locked train/validation/test split. It does not reassign parcels to make a split pass. Strict mode enforces the original budgets; advanced scientific production retains its separate reviewed/frozen-input requirements.

Timing, eight-bin completeness, imagery quality and the 20-pixel-per-parcel cap remain unchanged. Reference-supported and satellite-estimated windows remain distinguishable. All feature designs use the same samples; notebook 06 reads notebook 05’s saved values directly. Validation macro-F1 selects the design before test scoring. Low accuracy does not block a pilot result.

Handoffs and model inputs are content-keyed and hash-verified. Changed inputs create separate artifacts; previous imagery, jobs, partitions and results stay intact. The selected-model package records the preprocessing recipe needed for later Paoay inference. Local mapping and comparison with dated KMZ ground truth are a later stage; existing notebooks 12–15 remain separate experiments.

## Results and troubleshooting

The final summary lists batch states, reported verified tiles, queued/running/failed jobs, campaign/TIFF/log paths and a next action. Unavailable counts are shown as unavailable. `completed` confirms saved collection files; notebook 04 still checks timing and quality. `pending_retry` requires a later collection/retry; `disabled` is preparation only; a skipped source explains whether targets are met or candidates are exhausted.

Default annual TIFFs are under `sen4agrinet_work/annual_tifs/<source>/<year>/<campaign>/round_<number>/`. Each tile stores 24 half-month periods × seven measurements. Sampling masks and Parquet transport caches are separate. Use the printed paths for the exact run.

For missing references, check the mode and paths. For errors, inspect the detail log and traceback. Keep quality thresholds unchanged when usable yield is low. Use pilot mode for a smaller qualifying experiment; do not fabricate missing values, move locked parcels or mark production reviews complete. Revised windows can reuse annual histories; cross-year windows need explicit adjacent-year imagery.

## Paoay application

The [five-notebook Paoay workflow 07–11](paoay_workflow.md) is implemented: references → matching satellite inputs → exact features/fixed-model predictions → boundary-limited maps and screening flags → reference/field review. Follow the stage pointers in order. Raw probabilities, model/input files and original outputs remain preserved. Screening is flags only; Unknown and missing inputs stay separate. Notebook 11 freezes the preliminary map and prepares separate observations/correction proposals; calendar notes and later January visits do not establish historical accuracy. See the [field-check guide](paoay_field_checks.md). Original entry points are archived; older experiments remain 12–15.
