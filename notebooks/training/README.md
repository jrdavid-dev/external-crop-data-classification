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
per parcel). Selection uses development views only: Sentinel-2 only and
Sentinel-2 plus radar are scored with the default Random Forest, and the
higher selection score (mean of spatial CV and dataset-holdout macro-F1)
wins. The chosen recipe is frozen on the training split with the main
pipeline's forest settings and training weights. Outputs go to
`pipeline_outputs/annual/<stage>/run_<id>/`. Score the frozen model in
[Testing 05](../testing/05_annual_external_test.ipynb).

Alternatives that were tried and not adopted (the external study's window
follow-ups, green-up and curve-shape recipes, other classifiers, short-cycle
radar) live in [`experiments/`](../../experiments/README.md) with their results.

### Running the annual notebooks on another machine

Run them in order, each in a fresh kernel: Training 04 → Testing 05 →
Paoay 08. They need:

- **Preprocessing 01** (the quality handoff) and **Training 01–02 /
  Testing 01** (the window models compared against). The notebooks read
  the current pointers, so they use whatever dataset your preprocessing
  contains. Your own data gives your own results; nothing needs importing.
- **Earth Engine access** (`earth_engine_project` in `settings.local.json`)
  the first time: Training 04 samples Sentinel-1 at the selected pixels
  (about 260 small requests) and Paoay 08 fetches the 2024 Sentinel-1 and
  WorldCover grids. Everything is cached under `pipeline_outputs/annual/cache/`;
  reruns make no requests. No exports or cloud assets are created.
- **Paoay inputs** for Paoay 08: the KMZ files, `atok_imagery_root`
  composites and the Paoay 06 land-activity map.
Runtime on a laptop: about 5 minutes for Training 04 (plus the first
Sentinel-1 sampling), a few minutes for the other two. Each run writes new `run_<id>` folders and updates the
`current_*` pointers under `pipeline_outputs/annual/`; earlier runs stay.
