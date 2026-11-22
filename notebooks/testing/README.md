# External testing

Run `01_external_test_evaluation.ipynb`, then `02_results_and_diagnostics.ipynb` after freezing models. Outputs go to `pipeline_outputs/testing/run_<id>/`. See [handoffs and safeguards](../../docs/workflow.md#training-and-external-testing). Each notebook runs in a fresh kernel. The report gives separate rows for primary JECAM, each registered supplement, and the combined pool. Single-crop supplements report per-crop metrics and coverage without three-class macro-F1.

## Batch 1 report from frozen parcel predictions

Testing 02 uses the frozen Batch 1 run when it is available locally. It displays
JECAM first, then California carrot, Denmark cabbage, non-JECAM potato, and
the combined pool, with usable pixel and parcel counts beside F1 and its intervals. Each notebook run writes
a separate report directory beside the frozen run. Set `scores_handoff` in
Testing 02 to review standard diagnostics from another saved run.


From the repository root, with the saved testing run available locally, run:

```bash
.venv/bin/python -m crop_pipeline.training.external_report \
    pipeline_outputs/testing/run_81a0d0a2812e \
    pipeline_outputs/testing/run_81a0d0a2812e_batch1_revised --revised-scope
```

The command reads the saved parcel CSVs, scores and training-majority reference
reports. Use a new output directory outside the frozen run for each invocation.
Outputs include `report.md`, `parcel_metrics.csv`, `confusion_matrices.csv`,
`paired_macro_f1.csv` and `methods.json`. The default seed is 42 and the default
number of repetitions is 10,000. The combined interval conditions on each
model's observed campaign and reference-crop counts. Blank estimates and
intervals in the CSVs indicate unavailable values.

## Batch 2 model comparisons

[Testing 03](03_model_comparisons.ipynb) selects a single half-month composite
using spatial cross-validation in the matched development training split. It
freezes that comparator and an eight-bin spectral-bands-only comparator, then
scores both on the same saved external parcels as their frozen eight-bin
counterparts. It writes recipes, validation and external predictions, F1 tables,
and a report to a separate Batch 2 output directory.

The notebook also completes the matched three-way feature comparison with
72 interval, 48 shape-only, and 120 interval-plus-shape predictors. It
reuses and verifies the archived interval-plus-shape forests and fits the
missing shape-only forests before external scoring. The three-way tables
are saved under `shape_comparison/` in the Batch 2 output directory.

The final section of Testing 03 compares eight-bin indices only (32),
bands only (40), and bands plus indices (72). After it runs, use
[Training 03](../training/03_matched_model_tuning.ipynb) for development-only
hyperparameter selection and the paired comparison with the original models
on the saved external parcels.

## Batch 3 error analysis and diagnostics

[Testing 04](04_error_analysis_and_diagnostics.ipynb) reads the original frozen
models and saved feature, prediction, reference-audit, and imagery-quality files.
It writes parcel errors and support, imagery coverage, the training cabbage
trace, reconstructed internal OOB diagnostics, grouped impurity importance, and
hard pixel and parcel votes to a separate Batch 3 directory.

## Annual model test and paired comparison

[Testing 05](05_annual_external_test.ipynb) scores the frozen annual model from
Training 04 once on JECAM, each supplement and the combined pool, in the same
layout as Testing 01. It then compares the annual model with each original
window model on the parcels that model scored, using the parcel bootstrap of the
external report:

```bash
.venv/bin/python -m crop_pipeline.training.paired_comparison \
    pipeline_outputs/annual/test/run_<id> annual pipeline_outputs/testing/run_<id> \
    pipeline_outputs/reports/<new_folder> [--revised-scope]
```

The external parcels were already scored for the window models, so this
comparison is retrospective. [Paoay 08](../paoay/08_annual_maps.ipynb) applies the
frozen annual model to Paoay and the wider Atok boundary.
