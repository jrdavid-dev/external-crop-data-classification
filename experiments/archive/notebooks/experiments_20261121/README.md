# Archived experiments (20–21 November 2026)

Experiment notebooks whose results are recorded in the
[slides](../../../../pipeline_outputs/reports/slides_20261121/slides.md) and that are either superseded by the
[improved pipeline v1](../../../notebooks/improved_v1/README.md) or gave no gain. They are kept for reference and can still run
from here: each finds the repository root by looking for `pyproject.toml`. Their outputs stay in
`pipeline_outputs/transfer/` and `pipeline_outputs/pilots/`; their code stays in
`experiments/transfer/transfer.py` and `experiments/pilots/` (the improved pipeline reuses the
pilots code). Relative links inside the notebooks may point to their old folders.

The active experiment notebooks are `experiments/notebooks/tuning/` 01–03, `experiments/notebooks/transfer/` 06 and 13,
`experiments/notebooks/improved_v1/` 01–05 and `experiments/notebooks/mixed_training/` 01.

## Transfer: why validation overstates transfer (`transfer/`)

The notebooks chain through each run's `selected.json`: 01 → 02/03 → 04 → 05, 07–10 and 11 → 12.

| Notebook | What it found |
|---|---|
| 01 Spatial CV selection | Spatial CV scores far below the locked validation split; settings barely matter |
| 02 Source confounding and holdout | Within one crop, features identify the source dataset 82–97 % of the time; leave-one-source-out carrot recall 0.02–0.24 |
| 03 Source-robust features | Index-only and standardized features did not remove the source signal |
| 04 Training augmentation | Time-shifted and brightened training copies helped the aligned model most |
| 05 Window comparison and summary | Fixed vs aligned windows across all views |
| 07 Source-balanced weights | No gain |
| 08 Anchor-crop calibration | No gain |
| 09 Within-source pairwise | No gain |
| 10 Source feature screening | No gain; the source-revealing predictors also carry crop information |
| 11 Validation and external test | Final check, test used once: aligned model on JECAM 0.25 → 0.41 with augmentation |
| 12 Paoay aligned comparison | Only 1.9 % of Paoay pixels get a qualified green-up window |

## Pilots: new features on a subset, then full data (`pilots/`)

00 builds the subset; 01–08 use it; 09 builds the full data used by 10–12. See `pilots/README.md`.

| Notebook | What it found |
|---|---|
| 00 Subset and downloads | ~3,500-pixel subset and Sentinel-1 / ERA5 downloads |
| 01 Evaluation protocol | Four views (random split, spatial CV, dataset holdout, JECAM); dataset holdout is the hardest |
| 02 Harmonic features | Whole-year harmonics: coverage 52 % → 98 %, better dataset holdout |
| 03 Sentinel-1 and spatial selection | Radar gives the best dataset holdout, not a better JECAM score |
| 04 Climate stratification | GDD matching did not help (no tropical training data) |
| 05 Paoay harmonic GMM | Unsupervised clusters for Paoay |
| 06 Phase-invariant harmonics, EVI2 | Removing timing hurt: timing carries crop information |
| 07 Normalized SAR | Radar ratios better on JECAM, worse in Europe |
| 08 Unsupervised log-ratio | Many cluster-to-crop assignments fit the PSA totals equally well |
| 09 Full data build | All parcels, 5 pixels each, Sentinel-1 for every pixel |
| 10 Full data evaluation | Harmonics + radar: spatial CV 0.86, dataset holdout 0.69, JECAM 0.41 |
| 11 Full data Paoay | 100 % Paoay coverage, ~99 % potato |
| 12 Filters, seasonality, masks | 3×3 median and ω = 1.5 options; cropland masks agree on 61–82 % of pixels |
