# Training and external testing notebooks

Run preprocessing 01–03 first, then:

| Notebook | Purpose | Output |
| --- | --- | --- |
| `training/01_train_and_validate` | Fit training parcels; choose leaf size with validation parcel macro-F1 | `training/run_<id>/validation/` |
| `training/02_freeze_models` | Copy selected fitted models into verified packages, without refitting | same run's `models/` |
| `testing/01_external_test_evaluation` | Score matching CropHarvest test features; save predictions and training-majority baseline | `testing/run_<id>/predictions/` and `reports/` |
| `testing/02_results_and_diagnostics` | Summarize saved test scores, per-crop metrics, coverage and confusion matrices | same run's `reports/diagnostics/` |

All paths above are under `pipeline_outputs/`. Training and testing run folders also retain their input feature JSON and checksum receipts. Blank notebook inputs use the latest successfully completed stage; explicit handoff paths select an earlier run. Testing must use the same quality/window snapshot as training. A newly completed preprocessing run does not automatically match older models: select the older `features_test.json` explicitly when appropriate.

Training and evaluation deliberately create a new run on every execution, preserving all previous files. Freeze and diagnostics reuse verified completed outputs. Interrupted/modified outputs are never overwritten; rerun training/evaluation to create a fresh run if necessary. Pending stage pointers prevent accidentally falling back to an earlier successful attempt. Only use trusted local joblib packages.

## Existing methodology retained

Three classes: carrot, cabbage, potato. Development sources are Sen4AgriNet, FranceCrops and EuroCrops; CropHarvest JECAM remains external test only. Parcel partitions remain locked. Each training/validation cohort needs all three classes. Missing test classes suppress three-class macro-F1 and balanced accuracy.

The forests retain 1,000 trees, seed 42, square-root feature sampling and 50% bootstrap sampling. Full cohorts choose minimum leaf size 2 or 4 using validation only; matched cohorts retain leaf size 2. Features, confidence/parcel weighting and window/quality rules are unchanged. Selected draft packages and candidate validation metrics are saved in `validation/`; `models/` contains their published copies. Freeze performs no additional fitting. Test predictions average probabilities per parcel, as before.

The baseline predicts the most common training parcel crop (ties follow first occurrence in saved training rows). It is evaluated on the same test parcels as each corresponding model. Full cohorts can differ in coverage; matched cohorts hold locations constant. Diagnostics report whether macro-F1 is at least 0.50, exceeds the baseline, and each crop has F1 of at least 0.50. Model selection uses validation only. Unmet collection budgets remain visible in coverage metrics.

## Safe alongside collection

These notebooks read exported feature JSON, not live downloads, batch queues or collection state. Completed preprocessing snapshots remain usable while further collection runs. No files are moved or deleted. Collection layout refreshes preserve these new native output folders. Preprocessing itself still requires its existing collection-completion checks; training does not bypass those checks to create a feature snapshot.
