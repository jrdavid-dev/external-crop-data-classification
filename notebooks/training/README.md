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

[04_aligned_gap_validation.ipynb](04_aligned_gap_validation.ipynb) compares the
20-day and 25-day green-up onset gaps using the aligned-matched Random Forest
recipe. It includes completed development downloads, preserves existing split
assignments, and scores both models on identical locked validation features.
Models, membership, predictions, and metrics go to a separate training run.
