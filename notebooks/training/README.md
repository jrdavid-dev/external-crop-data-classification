# Training

Run `01_train_and_validate.ipynb`, then `02_freeze_models.ipynb` after preprocessing 03. Outputs go to `pipeline_outputs/training/run_<id>/`. See [handoffs and safeguards](../../docs/workflow.md#training-and-external-testing). Each notebook runs in a fresh kernel.

For an optional geographic validation diagnostic, pass the saved development
features, their run's validation directory, and a new output path:

```text
.venv/bin/python -m crop_pipeline.training.geographic_validation FEATURES_DEVELOPMENT_JSON VALIDATION_DIR REPORT_JSON
```

The diagnostic refits the selected 72-feature forest settings in five folds
within the locked training split. Folds hold out complete one-degree site cells
and merge cells linked by a shared parcel. The report includes per-fold
predictions and pooled parcel scores. Adjacent cells have no buffer between them.

## Annual harmonic recipe

[Training 04](04_annual_recipe.ipynb) adds a third model family beside the fixed
and aligned window models, which stay unchanged as the baseline. It builds
whole-year Sentinel-2 harmonic predictors, with Sentinel-1 radar sampled at the
same pixels, from the verified Preprocessing 01 quality handoff (up to 20 pixels
per parcel). Selection uses development views only and runs in three steps:

1. Recipe: Sentinel-2 only, Sentinel-2 plus radar, and two green-up variants
   (the external study's NDVI detector run on the annual series, adding cycle
   timing or measuring harmonic phases from green-up) are scored with the
   default forest.
2. Classifier settings: on the chosen recipe, each family is tuned by spatial
   CV in the training split with the external study's grids: Random Forest
   one-setting changes and leaf size 4, weighted logistic regression, and
   gradient-boosted trees (XGBoost when it loads, otherwise scikit-learn's
   histogram gradient boosting with the same grid; `selection.json` records
   the backend).
3. Family: the highest selection score wins, ties to the Random Forest.

The chosen recipe and classifier are frozen on the training split with the
main pipeline's training weights. Outputs go to `pipeline_outputs/annual/<stage>/run_<id>/`.
Score the frozen model in [Testing 05](../testing/05_annual_external_test.ipynb).

The same notebook re-scores the window-model follow-ups (shape predictors,
parcel medians, one-setting forest changes and the matched ensemble) on
development data with `crop_pipeline.training.window_variants`.
