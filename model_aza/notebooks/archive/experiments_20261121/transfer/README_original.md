# Transfer experiments

Exploratory experiments aimed at better geographic and cross-source transfer.
They use development parcels only (training and validation partitions) and the
full cohorts only; matched cohorts are too small to tune. Run in order after
Preprocessing 03, each in a fresh kernel:

1. `01_spatial_cv_selection.ipynb` — select Random Forest settings by spatial
   cross-validation (five site-cell folds in the training split) instead of the
   locked validation split.
2. `02_source_confounding_and_holdout.ipynb` — crop by source and site-cell counts,
   how well features predict the source within a crop, and leave-one-source-out scores.
3. `03_source_robust_features.ipynb` — feature variants: 72 interval predictors,
   indices only, per-pixel temporally standardized series, shape features, and
   indices plus shape.
4. `04_training_augmentation.ipynb` — training-only augmentation: one-bin seasonal
   shifts and ±10% band gains with recomputed indices.
5. `05_window_comparison_and_summary.ipynb` — fixed and aligned windows with the
   current and combined configurations at 1,000 trees.

6. `06_fixed_start_month_sweep.ipynb` — fixed four-month windows starting in each
   month from January to September, rebuilt from the Preprocessing 01 quality handoff
   with the unchanged window builder, validity rule, shared-location exclusion and
   20-pixel cap. Development regions only; test regions are dropped before windows
   are built. Only capped rows are held in memory and no windows or feature exports
   are written. Includes a check that the pipeline's start month reproduces the
   Preprocessing 03 fixed_full development pixels. Independent of notebooks 01–05.

Notebooks 07–10 test ways to reduce reliance on source and region signatures.
Each compares an option against two bases: the plain 72 interval predictors and
the best configuration so far from notebook 04 (its `selected.json`). Every
option is fitted inside each training fold only, then scored by spatial CV,
locked validation, leave-one-source-out and a per-crop holdout (one crop of one
source held out while that source's other crops stay in training).

7. `07_source_balanced_weights.ipynb` — weights that equalize sources within each
   crop, then parcels within each source.
8. `08_anchor_crop_calibration.ipynb` — additive per-source offsets estimated from
   crops two sources share (potato, carrot). With the current chain of sources
   (EuroCrops–carrot–Sen4AgriNet–potato–FranceCrops), holdouts that remove the
   bridge crop are reported as not scorable.
9. `09_within_source_pairwise.ipynb` — binary forests trained within one source
   (Sen4AgriNet carrot/potato, FranceCrops cabbage/potato), combined into three-crop
   probabilities with potato linking the pairs.
10. `10_source_feature_screening.ipynb` — drop the predictors that identify the
    source more than the crop, ranked inside each training fold.

Notebooks 01–05 and 07–10 each read the previous notebook's `selected.json` (blank input uses the
latest run). Outputs go to new folders under `pipeline_outputs/transfer/`. The
notebooks never read test features, change stage pointers or replace models frozen
by Training 02. Selecting by spatial CV, changing predictors or augmenting training
data would be methodology changes; record them separately before adopting any.

Not covered here: adding a second development source for cabbage needs new
collection. Code lives in `model_aza/training/transfer.py`.

13. `13_paoay_start_month_maps.ipynb` — exploratory: one fixed-window model per start
    month (Jan–Apr … Sep–Dec, development parcels only), applied to Paoay pixels whose
    eight half-months are clear in the local Cloud Score+ composites, in each window's
    clearest year. Reports clear-window coverage of Paoay cropland and crop shares of
    confident pixels next to the PSA Benguet 2024 shares. Code:
    `model_aza/training/start_month_maps.py`.
