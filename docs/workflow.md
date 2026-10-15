# Run the prototype

Install [the requirements and local settings](../README.md#setup), select `.venv` as the kernel, and run notebooks from the repository folder. The active collection and analysis run is **`bilinear_expanded`**. Historical runs remain in their own folders.

## Run order

| Order | Notebook | Completion check |
|---|---|---|
| 1 | [Sen4AgriNet preparation](../Sen4AgriNet_Satellite_Preparation.ipynb) | Save `sen4agrinet_bilinear_expanded.json` |
| 2 | [FranceCrops preparation](../FranceCrops_Satellite_Preparation.ipynb) | Save `francecrops_bilinear_expanded.json` |
| 3 | [Four-month windows](../Crop_Four_Month_Windows.ipynb) | Inspect usable pixel counts and shortfalls for all crops |
| 4 | [Feature engineering](../Crop_Feature_Engineering.ipynb) | Inspect the three designs and formal export status |
| 5 | [Model training](../Crop_Model_Training.ipynb) | Save parcel splits, models, predictions and evaluation metrics |

The two source notebooks may run in separate kernels concurrently. Complete authentication one at a time. Wait for both source notebooks to finish successfully before starting window preparation. Restart downstream kernels and run their cells in order.

## Source preparation

For the existing saved references, leave Sen4AgriNet NetCDF downloading and field rebuilding disabled. Both source notebooks currently enable daily-summary retrieval, satellite downloading and authentication. Missing cached reference files must be resolved before collection.

| Crop | Candidate target | Maximum candidates per source parcel |
|---|---:|---:|
| Carrot | 3,400 | 100 |
| Cabbage | 7,000 | 100 |
| Potato | 10,000 | 100 |

The source notebooks check candidate capacity before network access, retrieve actual-date daily summaries, apply the timing checks, then download qualified half-month median composites. B5/B11 use bilinear alignment before compositing; quality masks use nearest-neighbor resampling.

Each crop must retain **more than 3,000 sampled candidate pixels after timing filtering** before raster downloads begin. A second check requires completed files for that coverage before exporting the handoff. This counts sampling footprints covered by files, not guaranteed usable feature pixels. A shortfall stops collection; do not relax quality rules to pass it.

Timing-qualified cycles are scheduled independently. Complete in-year cycles remain scheduled when another cycle crosses the request-year boundary. Unsupported cycles are logged with their dates and `adjacent_year_imagery_required` in `download_selection.json`. A region may appear in both scheduled plans and pending-cycle diagnostics. Cross-year windows are not clipped, and adjacent-year downloads are not added automatically.

## Four-month windows and usable pixels

The window notebook requires both source handoffs and rejects version 3 imagery. It audits saved images, assesses supported rises, and chooses the earliest qualified four-month window containing complete pixels per parcel. All fragments of a parcel share its selected date range.

Each usable pixel requires all eight half-month intervals and 64 finite base features. No missing-data filling is applied. After selecting windows, retain up to **500 usable unique pixels per crop** across the two window policies. Report shortages instead of duplicating samples. See [methodology](methodology.md) for exact checks and interpretation.

## Features and training

Feature engineering prepares 72 interval, 120 interval-plus-shape and 48 shape predictors. Formal exports may remain `pending_inputs` while field identities, reviews or development freezes are missing. Leave scientific freeze flags unchanged until their evidence is available.

The default training mode, `existing_parcel_exploratory`, reads complete pixel tables directly from `bilinear_expanded` and computes these designs internally. It does not require a prepared formal feature handoff. It requires all three crops and at least five source parcel groups per crop, then locks a deterministic parcel-separated split before training local 1,000-tree Random Forests.

Validation macro-F1 selects the design before test scoring. Keep test results out of model-development decisions. Production Earth Engine evaluation has separate independent-local-data and freeze requirements; see [training](training.md) and [feature definitions](feature_experiments.md).

## Important files

Paths are relative to the parent of your configured `report_root`, normally `sen4agrinet_work/`.

| Path | Contents |
|---|---|
| `source_handoffs/<source>_bilinear_expanded.json` | Acquisition paths and provenance |
| `analysis/bilinear_expanded/source_inputs.json` | Combined source provenance |
| `analysis/bilinear_expanded/seasons/automatic_timing_audit.csv` | Timing passes and failures |
| `analysis/bilinear_expanded/seasons/cycle_selection_audit.csv` | Candidate windows and selected cycles |
| `analysis/bilinear_expanded/seasons/usable_pixel_selection.csv` | Retained counts and crop shortfalls |
| `analysis/bilinear_expanded/seasons/features/complete_features.csv` | Selected reference-supported pixel features |
| `analysis/bilinear_expanded/seasons/estimated_features/complete_features.csv` | Selected qualified estimated-window features |
| `analysis/bilinear_expanded/seasons/existing_parcel_exploratory_bilinear/` | Locked split, models, predictions, metrics and package |

Imagery stays under each configured imagery root in a separate `bilinear_expanded` folder. Existing verified downloads are reused; missing or damaged files can be retried. Keep exported bundle files together so hash verification succeeds.

## Common problems

| Symptom | Action |
|---|---|
| Certificate verification fails | Restart the kernel and run from the certificate-setup cell before authentication; it loads trusted certificates without disabling HTTPS verification |
| Required handoff missing | Finish the named source notebook before window processing |
| Timing-qualified count is too low | Inspect timing rejection reasons and available parcel capacity; add suitable references rather than weakening checks |
| Few complete usable pixels | Inspect missing intervals, quality losses and cycle-selection audits separately |
| Formal partition counts are zero | Formal field-registry/review requirements may be pending; this is separate from the local exploratory split |
| Training says inputs changed | Use a new analysis/output run; do not replace a locked holdout |
| Earth Engine throttling or temporary failure | Retry using the saved checkpoints; running one source at a time can reduce contention |

A successful prototype demonstrates repeatable data handoffs and saved evaluation. It does not guarantee a target accuracy or establish local Benguet performance.
