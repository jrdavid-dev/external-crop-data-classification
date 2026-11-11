# Training and evaluation

Run **04 Four-month windows → 05 Feature engineering → 06 Model training**. All three use the same `analysis_collection` (`adaptive_training` by default) and explicit `analysis_mode`.

## Current external evaluation

Run [preprocessing 05–07](../notebooks/preprocessing) after completing the development and dedicated CropHarvest JECAM imagery collections. Use the printed paths for [external evaluation 01–02](../notebooks/external_evaluation). Train and tune on locked Sen4AgriNet and FranceCrops development parcels. Score CropHarvest with the frozen packages. The controlled comparison uses the same locations in both recipes and the full cohorts show each recipe's coverage. A missing test crop suppresses the three-class summary.

Use the `fixed_full` model manifest for January–April Paoay imagery. Below is the documentation for the archived pooled pilot workflow.

## Pilot and strict modes

- **`analysis_mode = "pilot"` (default):** train with qualifying available data while reporting unmet collection budgets. Low accuracy does not block completion.
- **`analysis_mode = "strict"`:** require the original adaptive pixel and parcel targets before feature export or training.

Both modes keep the same timing, imagery-quality, complete-data and partition checks. Every crop needs at least **five usable source parcel groups**, and carrot, cabbage and potato must each be present in training, validation and test. This is an execution minimum, not evidence that the sample is sufficient for a final study.

Whole parcels retain their locked partition. Missing identities, conflicting crop labels, shared locations across groups and incomplete/nonfinite features stop the workflow. Do not move reserved parcels to make a split pass. Pending launched source batches must finish before analysis.

## Notebook 06

1. Load the exact feature handoff exported by notebook 05.
2. Verify hashes, predictor schemas, sample identities and locked partitions.
3. Fit the three existing local Random Forest feature experiments.
4. Select the design using **validation macro-F1**, before test scoring.
5. Save held-out predictions, metrics and models.
6. Package the selected model and input recipe for later Paoay testing.

Features are computed once in notebook 05, not recomputed differently in training. All designs use the same samples and partitions. Forests retain 1,000 trees, seed 42, square-root feature selection, minimum leaf size 2 and 50% bootstrap samples. Each crop receives equal total training weight, shared equally among its source parcels; confidence allocates each parcel's share across pixels.

Report accuracy, macro-F1, crop-level precision/recall/F1, confusion matrices and parcel counts together. Test results describe held-out external source labels, not independently measured Paoay accuracy. Do not use test scores to change windows, discard samples or switch feature designs.

## Saved results

The notebooks print their exact content-keyed output folders. Changed inputs create a separate artifact; existing data, locked partitions and earlier models are preserved.

Completed local runs are reused only after their experiment receipt and artifact checksums verify. Incomplete or changed runs stop instead of overwriting results; choose a new `training_run_name` for a separate attempt.

Saved results include feature provenance, parcel assignments, configuration, training weights, validation/test predictions, feature selection, model files, metrics and figures. The selected-model package contains the model, ordered predictors, crop codes, imagery/scaling rules, four-month window and feature recipe, source references and checksums. It is a **pilot model for evaluation**, not a claim of local accuracy. Only load trusted joblib files.

The next project stage is to prepare matching four-month Paoay imagery, classify it with this fixed model, and compare predictions with dated local ground truth. That mapping/evaluation stage is separate from notebooks 01–06; the existing experimental notebooks 12–15 are not their automatic continuation.

## Separate production path

Strict collection budgets are not the same as scientific production readiness. The advanced production path retains reviewed field identities, supported local crop-cycle evidence, frozen window/feature decisions, verified formal bundles and independent local test fields. Its Earth Engine comparisons remain separate from pilot models.

Do not mark reviews or settings frozen merely to clear a pending status. Production compares external-only, local-only and combined training on a shared independent local test set; local scikit-learn and Earth Engine bagging are not identical.
