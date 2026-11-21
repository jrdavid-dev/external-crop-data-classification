# Tuning

Exploratory Random Forest hyperparameter tuning on development parcels only.
Run after Preprocessing 03, in order, each in a fresh kernel:

1. `01_validation_grid_search.ipynb` — grid over `max_features`, `min_samples_leaf`,
   `max_samples` and `max_depth`; fit on training parcels, score validation parcel macro-F1.
2. `02_spatial_cross_validation.ipynb` — baseline and top settings rechecked with five
   site-cell folds inside the training split.
3. `03_seed_stability_and_summary.ipynb` — baseline and tuned settings refit with
   1,000 trees over five seeds.

Outputs go to new folders under `pipeline_outputs/tuning/`. The notebooks never read
test features, change stage pointers or replace models frozen by Training 02.
Sample weighting and the 72 interval predictors are unchanged. Code lives in
`model_aza/training/tuning.py`.
