# Classification implementation handoff

**Start with Batch 1.** The dataset is final for this round. The quantitative study evaluates frozen, externally trained models on labelled external parcels. Paoay is a later preliminary application. The four external batches below require no new data downloads.

Read [AGENTS.md](../AGENTS.md) first and work solo. Use the [revised classification methods and deliverables](classification_methods_and_deliverables.md) as the study specification. This handoff records planned work. Only the starting evidence described below has been verified.

## Starting evidence

| Saved stage | Inputs and results to reuse |
| --- | --- |
| Preprocessing | [Feature receipt](../pipeline_outputs/preprocessing/external_xxu0iz4o/features/features_stage_receipt.json) and [usable parcels and pixels](../pipeline_outputs/preprocessing/external_xxu0iz4o/features/usable_samples.csv) |
| Training and freezing | [Validation candidates](../pipeline_outputs/training/run_fa40f5d6bb31/validation/validation_candidates.json) and [four frozen model packages](../pipeline_outputs/training/run_fa40f5d6bb31/models/models_stage.json) |
| External testing | [Parcel predictions](../pipeline_outputs/testing/run_81a0d0a2812e/predictions/), [training-majority baseline](../pipeline_outputs/testing/run_81a0d0a2812e/reports/majority_baseline.json), and [existing results](../pipeline_outputs/testing/run_81a0d0a2812e/reports/diagnostics/results.md) |
| Dataset interpretation | [Source and crop register](classification_dataset_register.md), [external parcel audit](../pipeline_outputs/test_audit_2026-11-19/report.md), and [development error review](../pipeline_outputs/development_error_review_2026-11-19/report.md) |

Fixed-full training has 269 carrot, 449 cabbage and 358 potato parcels. Green-up-aligned-full has 107, 66 and 87. Each matched model uses the same 72 carrot, 50 cabbage and 77 potato parcels and the same pixel locations, with features from its own window. Keep all four cohorts, the locked splits, the imagery gates and the existing archives. Generated run artifacts are local and ignored by Git. A fresh checkout needs the corresponding data handoff.

## Batch 1 — Complete external metrics and uncertainty

**Implement**

1. Extend reporting from the saved parcel predictions. The active [training and testing workflow](../src/crop_pipeline/training/notebook_workflow.py) connects Testing 01–02 to the run. Write new reporting outputs separately from sealed run files.
2. Report primary JECAM first, then California carrot, Denmark cabbage, non-JECAM cabbage and non-JECAM carrot/potato separately. Compute the combined result from their pooled parcel rows, including JECAM once. Show both full models with their own denominators and both matched models on the same parcel IDs.
3. Retain confusion matrices, support, producer’s accuracy (recall), user’s accuracy (precision), per-crop F1 and macro-F1. Add parcel overall accuracy, Cohen’s kappa where defined, and 95% intervals. Resample whole parcels. Use the same resampled IDs for the model and its frozen training-majority baseline when estimating their macro-F1 difference.
4. Record the bootstrap seed, repetitions, sampling strata and treatment of undefined replicates. Agree whether the combined interval conditions on the observed campaign/class mixture.

**Outputs** — Machine-readable parcel metrics and interval tables, paired baseline differences, and a short results report.

**Done when** — The original 24 model/population confusion matrices and all available point scores reproduce. Three-class macro-F1 stays unavailable for incomplete class coverage. Report whether macro-F1 reaches 0.50, exceeds majority and has a paired difference interval entirely above zero. Flag each crop below F1 0.50 or whose interval extends below it. As a cross-check, matched JECAM macro-F1 rounds to 0.493 fixed and 0.222 aligned. This batch requires no fitting.

## Batch 2 — Complete the model comparisons

**Implement** — Fit the missing single-date RF and spectral-bands-only RF on the frozen development split. Decide whether the single-date baseline means one acquisition or one half-month composite before implementation, and document any raw-input requirement. Choose the timing rule using development evidence. For the bands-only baseline, use B3, B4, B5, B8 and B11 across the eight intervals. Hold the evaluation parcels, weights and other relevant RF settings constant within each comparison. Freeze each comparator before external scoring.

Reuse the [three-way matched development comparison](../pipeline_outputs/training/run_fa40f5d6bb31/matched_geographic_feature_comparison.json) for 72 interval, 48 shape and 120 combined predictors. [Selected external shape comparisons](../pipeline_outputs/training/run_fa40f5d6bb31/shape_external_comparison/external_comparison.json) also exist. If the manuscript claims a three-way external comparison, complete the missing shape-only scoring on identical parcels.

**Outputs** — Comparator recipes, frozen artifacts, predictions and a classwise and aggregate comparison table identifying the evaluation stage.

**Done when** — A reader can identify the exact date or bin, predictor list, split and shared parcel IDs for each comparison. If compatible inputs for an actual single acquisition are unavailable locally, record the limitation or agree on the composite definition. Describe new comparisons as evaluations on an already examined benchmark.

## Batch 3 — Explain errors, coverage and forest behavior

**Implement** — Join predictions to saved source, geography, crop, window, observation date and pixel support. Summarize crop confusions, one-pixel parcels, clear observations and missing coverage. Inspect representative errors for mixed fields, boundaries, terrain shadow and possible crop transitions where evidence permits. Preserve unknown causes. For matched JECAM, 0/83 reference observation dates fall inside the fixed window and 62/83 inside the aligned window.

Trace training cabbage with the [cycle-selection audit](../data/pipeline_work/analysis/external/external_xxu0iz4o/seasons/cycle_selection_audit.csv). All training cabbage comes from FranceCrops. Of 449 fixed-full cabbage parcels, 51 also qualified for aligned-full, 117 reached cycle selection without a complete usable window and 281 did not reach that stage. Another 15 qualified only for aligned-full. The retired Dutch cabbage supplement was a separate test campaign.

Export grouped impurity importance by band/index and time bin. The active forests use 1,000 trees, seed 42, Gini, square-root feature sampling and a weighted half-size training bootstrap. Their saved estimators have OOB scoring disabled. Implement the internal OOB diagnostic using omitted training samples and document how it was obtained. Preserve the original fitted models and disclose the optimistic effect of same-field pixel dependence.

**Outputs** — Campaign/crop error and support tables, imagery-loss summaries with unknown legacy counts, grouped importance, and an identified OOB diagnostic.

**Done when** — Counts reconcile to the frozen cohorts and example errors trace to parcel IDs. Clearly distinguish per-tree training draws, internal OOB error and Batch 1 external parcel intervals. Report each finding against the model that produced it.

## Batch 4 — Package the external study for submission

**Implement** — Finalize the dataset register and create a run manifest linking the frozen feature snapshots, split identities, model hashes, recipes, predictions, tables and software versions. Document the known mismatch between saved preprocessing source hashes and the current preprocessing code. Provide a reproducible reporting entry point from the saved snapshots. Choose the model for later application using the documented development-validation rule. Decisions informed by previous test inspection must be identified as retrospective.

**Outputs** — Reproducibility manifest, verified reporting instructions, named model package, and manuscript-ready external methods, results and limitations.

**Done when** — A teammate with the local data handoff can reproduce the external tables, verify model and input hashes, and identify the selected model and window. Report JECAM as the primary benchmark and the combined pool as overlapping evidence. Record quota gaps and uncertainty alongside scores. Final venue selection, author sign-off and submission are separate decisions. Retain the submission receipt when the paper is submitted.

## Deferred batch — Preliminary Paoay application

**Resume after** — The external model package and imagery window are designated. Decide whether January–April 2024 is essential. The [historical map receipt](../data/pipeline_work/paoay/predictions/bundles/53f0e0e9bbd8075558d9/prediction_handoff.json) traces to an older 120-predictor pilot. The active fixed external recipes require a May start, and the [saved period audit](../data/pipeline_work/paoay/period_search/comparison_20261110T033142Z/recommendation.md) found no examined May-start window meeting the existing whole-ROI gate.

**Implement** — Verify authoritative boundaries and compatible imagery. Apply the designated frozen package with its exact feature order, scaling and window. Preserve the pilot separately. Report clear observations, the training 5th–95th percentile range check and the flag for more than 20% of predictors outside range. Decide whether to export hard-tree vote strength or document the probability-based amendment. The saved pilot uses maximum mean probability and a margin.

**Outputs** — Preliminary 10 m prediction raster, confidence and coverage diagnostics, area table, and provenance linking the raster to the externally evaluated model.

**Done when** — Model and imagery receipts match, and accepted crop area, withheld area and missing imagery reconcile to the mapped extent. Calendar notes provide context only. Local ground-truth validation and local accuracy remain deferred until independent, dated field-level occupancy evidence is available.
