# Training and evaluation

The prototype checks that data can move reliably from selected windows to saved models and evaluation. Better data can be added later. Accuracy is useful for diagnosing the pipeline; it is not yet evidence of performance in Benguet.

## Current files: exploratory accuracy evaluation

Run [Model Training](../notebooks/05_model_training.ipynb) in a fresh `.venv` kernel with the saved defaults:

```python
training_mode = "existing_parcel_exploratory"
analysis_run_name = "bilinear_expanded"
```

This mode reads the new run's complete pixel tables and builds features locally. It does not download imagery or contact Earth Engine. Formal feature handoffs may remain pending; they are not required for this mode.

### How the data is split

Each crop needs at least **five source parcel groups**. This is an execution minimum, not a recommendation for a final study.

| Split | Approximate share of parcel groups | Purpose |
|---|---:|---|
| Training | 60% | Fit the models |
| Validation | 20% | Choose the feature design |
| Test | 20% | Evaluate held-out predictions |

All pixels and fragments from one source parcel stay together. Pixel percentages may differ because parcels have different sizes. Missing identities, conflicting labels and shared locations across groups stop training.

The split uses seed 42 and is saved with input-file hashes before fitting. If inputs change, use a new output folder rather than replacing the locked split.

### What gets trained

Three local Random Forests compare the [feature designs](feature_experiments.md). Each uses 1,000 trees, seed 42, square-root feature selection, minimum leaf size 2 and 50% bootstrap samples. Training weights reduce domination by parcels with many pixels.

Validation macro-F1 selects the design before test scoring. All designs use the same test samples. Do not choose a design or tune settings using test scores.

### What gets saved

Outputs are under `analysis/bilinear_expanded/seasons/existing_parcel_exploratory_bilinear/`, relative to the parent of `report_root`.

| File | Purpose |
|---|---|
| `exploratory_input.json` | Input hashes and locked group assignments |
| `parcel_split.csv` | Parcel membership in each split |
| `partitioned_features.csv` | Features with their split assignments |
| `configuration.json` | Recorded training settings |
| `feature_selection.json` | Validation-based design selection |
| `experiment_metrics.json` | Scores, crop-level results and confusion matrices |
| `<design>/model.joblib` | Fitted model and ordered predictors |
| `<design>/*_predictions.csv` | Validation and test predictions |
| `training_package/` | File inventory and provenance for the run |

Read accuracy, macro-F1, per-crop precision/recall and parcel counts together. Nearby parcels can remain similar despite separate identities. Cabbage also comes from a different source/year, which can influence scores. These are external-source results, not independent local accuracy.

## Local data and field partitions

This section applies to the later **production** study. Source notebooks do not automatically collect or label local evaluation data.

Prepare local observations with the same imagery and feature rules. Before the window notebook's field-partition step, add the local reference-supported feature rows once and record their input paths in the export provenance.

Complete `seasons/review/field_registry.csv` using the generated template. Keep every fragment, season and pixel of the same reviewed field in one partition. Sources are `external` or `local`; partitions are `train`, `validation` or `test`. Independent local test fields require confirmed crop labels and reference-supported windows.

## Production requirements

Use `training_mode = "production"` only when the following are ready:

1. Reviewed field identities and independent local test labels.
2. Supported crop-cycle evidence in `season_reviews.csv` for reference-supported windows.
3. A frozen window configuration with recorded development evidence.
4. A frozen feature recipe and its development evidence.
5. Hash-verified feature bundles and all configured crops in the required splits.

Use the generated templates for exact evidence fields. Do not mark a decision frozen merely to clear a pending status. Choose settings using development data, then re-export windows and features if their inputs change.

Production uses Earth Engine to compare external-only, local-only and combined training across the three designs: nine models. The shared local test set remains separate from fitting and design selection. Estimated external windows may support exploratory training, but cannot become independent local test truth. Earth Engine's bootstrap weighting differs from local scikit-learn weighting; the two implementations are not identical.
