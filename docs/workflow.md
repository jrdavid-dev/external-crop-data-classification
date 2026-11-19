# Run the external classification workflow

Start with [project design](project_design.md) and [setup](fresh_clone.md).
Notebook numbering restarts inside each folder. Use a fresh project kernel for
each notebook. Read the settings before Run All; collection notebooks can submit
cloud jobs.

## Choose the input route

- **Collect:** review Collection 01, run the source notebooks needed below, then 08.
  Rerun pending batches with the same settings until downloads are complete.
- **Use a teammate's imagery:** follow [export/import](sharing_imagery.md), skip
  collection, and start Preprocessing 01. Import copies TIFFs and preserves labels,
  masks, provenance and partition assignments; no Earth Engine login is needed.

## Notebook order and handoffs

| Order | Notebook | Reads | Publishes / next step |
| --- | --- | --- | --- |
| Review | [Collection 01](../notebooks/data_collection/01_external_reference_audit.ipynb) | Local references and saved selections | Source audit; choose required collections |
| Collect development | Collection [02 Sen4AgriNet](../notebooks/data_collection/02_sen4agrinet_collection.ipynb), [03 FranceCrops](../notebooks/data_collection/03_francecrops_collection.ipynb), [04 EuroCrops](../notebooks/data_collection/04_eurocrops_collection.ipynb) | Source references and saved batch plans | Annual TIFFs, masks, timelines, completed receipts and locked assignments |
| Collect primary test | [Collection 05](../notebooks/data_collection/05_cropharvest_test_collection.ipynb), `collection_cohort = "jecam"` | CropHarvest references | Separate primary test campaign |
| Optional supplements | Collection 05 supplement modes; [06 California](../notebooks/data_collection/06_california_carrot_test_supplement.ipynb); [07 Denmark](../notebooks/data_collection/07_denmark_cabbage_test_supplement.ipynb) | References and overlap-screen inputs | Separate supplemental test campaigns |
| Summarize | [Collection 08](../notebooks/data_collection/08_dataset_summary.ipynb) | Saved collection records | Metadata snapshot; does not certify usable imagery |
| 1 | [Preprocessing 01](../notebooks/preprocessing/01_imagery_quality.ipynb) | Completed collection or imported batches | `quality.json`: imagery audit and collection snapshot |
| 2 | [Preprocessing 02](../notebooks/preprocessing/02_window_selection.ipynb) | Verified quality handoff | `windows.json`: fixed/aligned qualifying windows |
| 3 | [Preprocessing 03](../notebooks/preprocessing/03_feature_engineering.ipynb) | Verified windows handoff | `features.json`, `features_development.json`, `features_test.json`, counts and shortfalls |
| 4 | [Training 01](../notebooks/training/01_train_and_validate.ipynb) | Development features only | `training.json`: fitted candidates and validation selection |
| 5 | [Training 02](../notebooks/training/02_freeze_models.ipynb) | Verified training handoff | `models/models_stage.json`: frozen copies; no refitting |
| 6 | [Testing 01](../notebooks/testing/01_external_test_evaluation.ipynb) | Frozen models and matching test features | `scores.json`: parcel predictions, scores and majority baseline |
| 7 | [Testing 02](../notebooks/testing/02_results_and_diagnostics.ipynb) | Verified scores handoff | `reports/diagnostics/report.json`, `results.md`, crop tables and confusion matrices |

Sen4AgriNet and FranceCrops are required development sources in the current audit;
EuroCrops contributes completed carrot batches when present. Supplemental
collections are optional. Local
quota notebooks are diagnostics outside the main handoff chain. Paoay has a
[separate workflow](paoay_workflow.md) and remains deferred scientifically.

## Defaults, snapshots and reruns

Blank input paths resolve the current successful stage pointer under
`pipeline_outputs/<stage>/current_<name>.json`; they do not scan folders by date.
A pending pointer blocks fallback to an older success. Explicit inputs select an
exact historical handoff; use the matching development/test feature pair.
Preprocessing verifies its receipts and collection snapshot. Training/testing
validate the feature contract and matching experiment provenance.

Preprocessing 01 creates a new audit. Successful unchanged outputs in 02–03 are
verified and reused. Partial or changed preprocessing outputs require a new 01
run. Training 01 and Testing 01 create new run folders on deliberate reruns;
Training 02 and Testing 02 reuse verified completed outputs. Existing outputs
and campaign identities remain preserved.

After adding or importing data, finish pending collection and rerun Preprocessing
01–03. Then train/freeze new models and evaluate the matching test export.
Alternatively, continue using a complete older feature snapshot explicitly;
training/testing do not read live collection state.

## Read results correctly

Use `usable_samples.csv` for final development/primary JECAM counts, and
`test_population_coverage.csv` for supplements and the combined pool. Candidate
counts, complete-window counts and final capped feature counts measure different
stages. Inspect shortfalls per crop, partition and recipe. Full and matched
cohorts overlap; never sum them.

Select model settings using validation parcel macro-F1 only. Report JECAM,
each supplement and the combined population separately. Missing test crops make
three-class aggregates unavailable. Retain coverage, per-crop support and the
training-majority baseline with every score. See [research basis](research_basis.md)
and [training/testing details](workflow.md#training-and-external-testing).

Use [output layout](storage_layout.md#organized-outputs) to locate files and [demo notes](notebook_demo.md)
to present them. Local fixture tests verify software behavior; a completed run on
the chosen data is still required for scientific results.

## Training and external testing

Training 01 fits and selects candidates under `pipeline_outputs/training/run_<id>/validation/`.
Training 02 copies selected fitted packages into that run's `models/` without
refitting. Testing 01 saves a new testing run's input snapshot, parcel predictions,
per-population scores and training-majority baseline; Testing 02 saves diagnostics.
Frozen packages and diagnostics are reused only after their receipts verify.
Only load trusted local joblib files.

Blank paths use the current verified stage pointer. A newly exported test snapshot
does not automatically match older models: choose its matching older
`features_test.json` explicitly when using an earlier model. Saved provenance,
window metadata and quality snapshots must agree. Training/testing read saved
feature exports, not live downloads or collection state; a complete older snapshot
can be used while a separate collection continues.

Each development cohort needs carrot, cabbage and potato in both training and
validation. The active notebooks report budget shortfalls; they do not expose
the retired pooled workflow's `analysis_mode = "pilot"/"strict"` switch. Missing
test crops suppress three-class aggregates. Keep scientific constraints unchanged
when reporting a shortfall. Model settings and scoring definitions are in
[methodology](methodology.md#model-selection-and-scoring).

## Progress and troubleshooting

Where a notebook uses the shared progress display, counts describe its named
work units; nested detail belongs in saved logs. Unknown-length work has no
invented percentage. ETAs learn from measured throughput or comparable completed
runs; authentication, cache reuse, cloud queues and retries change runtime.
Pending/failed stages are not successful timing profiles. Notebook interfaces
can differ; the handoff table above is the supported run sequence.

Shared progress logs/timing records live under
`<parent-of-report_root>/progress/<notebook-name>/runs/`. Exact result and log
paths are printed by the entry point. Forest completion is a model-level unit,
not a per-tree progress claim.

| Situation | Action |
| --- | --- |
| Collection returns pending/retry | Inspect saved task errors/states; rerun the same source settings. Queued/running tasks are not automatically failures. |
| Collection changes during preprocessing | Finish it, then start a new Preprocessing 01 audit; preserve the earlier attempt. |
| Partial/changed stage artifacts | Use a fresh upstream audit or training/evaluation run as appropriate; do not overwrite or bypass receipts. |
| Model/test snapshot mismatch | Select the test export paired with the frozen models, or train/freeze from the new development export. |
| Unmet quotas | Report crop/partition/recipe shortfalls and plan legitimate collection top-ups. Do not change labels, partitions or quality checks. |
| Old preprocessing receipt rejects current code | Preserve it and rerun Preprocessing 01–03. A data move is not a new scientific result. |

## Verification

For local software checks, run:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The notebook integration test uses tiny TIFF fixtures, mocked imagery audit/window
selection and small forests to verify handoffs, collection isolation and snapshot
rejection. Separate tests cover the processing and transfer helpers. These checks
do not replace a completed experiment on the chosen data or establish crop accuracy.
Dated past test totals belong in [history](history/README.md), not a live run guide.
