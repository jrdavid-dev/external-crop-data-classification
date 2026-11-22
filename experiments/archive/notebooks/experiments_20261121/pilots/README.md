# Pilots

Exploratory pilots of five proposed enhancements, run on a data subset to keep
them fast. Code lives in `experiments/pilots/`; outputs go to new folders
under `pipeline_outputs/pilots/`. Nothing here changes stage pointers, frozen
models, partitions or collection state.

Run in order, each in a fresh kernel:

0. `00_pilot_subset_and_downloads.ipynb` — samples up to 40 training and 15
   validation parcels per crop and source (locked partitions kept) plus all primary
   JECAM test parcels, 10 pixels each, from the verified Preprocessing 01 handoff.
   Reads their annual 24 half-month Sentinel-2 series, then makes small direct Earth
   Engine requests (no export tasks): Sentinel-1 half-month VV/VH, ERA5-Land monthly
   temperature per parcel, and Sentinel-1 plus temperature for the Paoay study area.
1. `01_evaluation_protocol.ipynb` — the fixed May–August predictors scored four
   ways: locked random split, spatial CV, dataset holdout and the JECAM test.
2. `02_harmonic_features.ipynb` — two-harmonic regression on the clear half-months of
   the year (full, and a shift-invariant form) against the fixed window.
3. `03_sentinel1_and_spatial_feature_selection.ipynb` — Sentinel-1 harmonics alone and
   with Sentinel-2, then spatial forward feature selection over variable groups.
4. `04_climate_stratification.ipynb` — growing degree days (base 5 °C) and monthly
   temperature descriptors; training on the climate-nearest half of parcels or with
   soft climate-similarity weights, scored on JECAM and dataset holdout.
5. `05_paoay_harmonic_gmm.ipynb` — harmonic coverage of Paoay pixels, classifier
   predictions, Gaussian-mixture clusters (BIC, stability) and a comparison with
   crop statistics from a local CSV (`crop,area_ha`). The default,
   `data/paoay/benguet_vegetable_area_2024.csv` (not in Git), holds PSA's 2024 Benguet
   area harvested for cabbage, carrot and potato (Situation of Selected High Value
   Vegetable Crops in Benguet 2024, Table 3). These are province-wide annual totals,
   not Atok or Paoay areas.

Pilot simplifications: the per-pixel clear flag replaces the pipeline's 80% parcel
coverage rule; 300-tree forests averaged over three seeds; climate is monthly
ERA5-Land at about 9 km, which smooths highland temperatures.
