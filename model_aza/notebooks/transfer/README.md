# Transfer experiments: fixed-window start months

Two exploratory notebooks on which four-month window to use. They use development
parcels only (training and validation partitions); outputs go to new folders under
`pipeline_outputs/transfer/`. They never read test features, change stage pointers or
replace models frozen by Training 02.

- `06_fixed_start_month_sweep.ipynb` — fixed four-month windows starting in each
  month from January to September, rebuilt from the Preprocessing 01 quality handoff
  with the unchanged window builder, validity rule, shared-location exclusion and
  20-pixel cap. Test regions are dropped before windows are built. Only capped rows
  are held in memory and no windows or feature exports are written. Includes a check
  that the pipeline's start month reproduces the Preprocessing 03 fixed_full
  development pixels. Code: `model_aza/training/start_month_sweep.py`.
- `13_paoay_start_month_maps.ipynb` — one fixed-window model per start month
  (Jan–Apr … Sep–Dec, development parcels only), applied to Paoay pixels whose eight
  half-months are clear in the local Cloud Score+ composites, in each window's
  clearest year. Reports clear-window coverage of Paoay cropland and crop shares of
  confident pixels next to the PSA Benguet 2024 shares; reads notebook 06's scores.
  Code: `model_aza/training/start_month_maps.py`.

The other transfer notebooks (01–05 and 07–12: spatial CV selection, source
confounding, source-robust features, augmentation, source weights, anchor
calibration, pairwise models, feature screening, the final validation/test check and
the Paoay aligned comparison) are archived with their findings in
[`archive/experiments_20261121`](../archive/experiments_20261121/README.md); their
original guide is kept there as `transfer/README_original.md`. Code for all of them
remains in `model_aza/training/transfer.py`.
