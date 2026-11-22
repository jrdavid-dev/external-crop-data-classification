# Experiments

Ideas that were tried alongside the main pipeline (`src/crop_pipeline/`, `notebooks/`). Each one
was judged on development data first (spatial CV within the training split and holding out one
development dataset at a time; the selection score is their mean). The main pipeline holds the
reported window models; everything here, including the annual recipe, is kept with its result so
the comparison stays honest.
Nothing here changes the main pipeline's code, partitions, stage pointers or frozen models. It
reads them, and writes its own outputs to `pipeline_outputs/` (ignored by Git).

## Index

External scores (JECAM macro-F1, combined pool) were computed after the test parcels had already
been seen for the window models, so they are retrospective. Numbers are for the expanded dataset
(Jose's plus 1,500 extra development parcels) unless marked; the annual Random Forest scores
JECAM 0.395 on it and 0.348 on Jose's original dataset.

| Experiment | Folder | Question | Result | Adopted |
|---|---|---|---|---|
| Annual recipe | `annual/` | Does a whole-year harmonic model (Sentinel-2, optional Sentinel-1) beat the window models? | JECAM 0.395, combined pool 0.588 (rerun 22 Nov 2026, `pipeline_outputs/annual/test/run_de290eb775e9`); paired comparison against each window model is retrospective | Kept as an experiment; the window models stay the reported baseline |
| Pilots | `pilots/` | Which annual features (harmonics, Sentinel-1, ERA5, masks, clustering) are worth a full run? | Harmonic curves and Sentinel-1 helped on development data | Yes, for the annual recipe |
| Improved pipeline v1 | `improved_v1/` | First end-to-end annual pipeline (5 pixels/parcel, fit on train + validation) | JECAM 0.374, combined 0.541; superseded by `experiments.annual.workflow` (20 pixels/parcel, locked training split) | Superseded |
| Transfer and start month | `transfer/` | Does a different fixed window, source weighting, calibration or augmentation transfer better? | Features identify the source dataset 82–97% of the time within a crop; source weights, calibration, pairwise models and screening gave no gain; augmentation lifted the aligned model on JECAM 0.25 → 0.41 (test already used); May–Sep windows score best in Europe but are 1–6% clear in Atok | No |
| Tuning | `tuning/` | Grid, spatial CV and seed stability for the window forests | Five-seed validation: fixed-full +0.018, aligned-full −0.001, fixed-matched unchanged, aligned-matched +0.031 with seed s.d. 0.08; no setting helped consistently | No |
| Window-model follow-ups | `window_followups/` | Do the external study's follow-ups (shape predictors, parcel medians, forest settings, 50/50 ensemble) help all four window models? | None improved all four; parcel medians worse everywhere (−0.04 to −0.06) | No |
| Green-up recipes | `greenup/` | Does the external study's NDVI green-up detector help the annual model? | Timing features 0.782, phases from green-up 0.753 vs 0.789 (selection score) | No |
| Curve shape (A/B/C) | `curve_shape/` | Do the external study's curve-shape descriptors help the annual forest? | C vs A on JECAM −0.013 [−0.085, +0.056] (Jose's data +0.005); shape only worse on the combined pool | No |
| Other classifiers | `classifiers/` | Logistic regression and gradient-boosted trees (XGBoost) with the external study's grids | Higher development scores (0.804, 0.811 vs 0.802) but worse on JECAM (0.201, 0.300 vs 0.395) on both datasets | No (recorded as a negative finding) |
| Short-cycle radar | `short_cycle/` | Describe one Sentinel-1 crop cycle instead of the whole year | Selection 0.720 vs 0.789; flips the Paoay map from potato to cabbage | No |
| Mixed training | `mixed_training/` | Do labels from the test places help a new place? | Same-place labels raised held-out scores 0.56 → 0.71; other places did not help a new place (0.53) | Motivates local labels |
| Field-visit plan | `field_plan/` | Stratified points and field sheets for the January 2027 visit | 65 points, 60 reserves, KMZ reference points | Planning tool |
| Walkthrough | `walkthrough/` | Figures and summaries for the walkthrough notebook | Reads current outputs only | Presentation |

## Layout

| Path | Contents |
|---|---|
| `<topic>/` | Code for each experiment above |
| `notebooks/` | Experiment notebooks: `annual/`, `improved_v1/`, `mixed_training/`, `transfer/`, `tuning/`, `walkthrough/` |
| `tests/` | Tests for the experiment code |
| `archive/notebooks/` | Superseded experiment notebooks (21 Nov) with their findings |
| `archive/removed_test_cohort_supplement_matching.patch` | A fix for re-collecting the carrot/potato supplement, not applied to the main pipeline yet; apply with `git apply` if agreed |

Experiment recipes register themselves with the annual recipe when imported
(`experiments.annual.workflow.register_recipes`), so their frozen models can be tested and mapped
with its stages; they are never selection defaults. Non-forest classifiers are frozen with
`workflow.freeze(..., estimator=...)`.

## Running

Notebooks find the repository root and add both `src/` and the root to `sys.path`. Tests:

```bash
.venv/bin/python -m unittest discover -s experiments/tests -t . -v
```

XGBoost (`classifiers/`) needs the OpenMP runtime on macOS (`brew install libomp`); without it the
comparison uses scikit-learn's gradient boosting with the same grid and records which ran.

Who did what is recorded in the Git history; most of this folder is Aza's work building on the
external study's code in `crop_pipeline`.

## Rerun on 22 November 2026

Every annual-family experiment rerun on the same inputs with `python -m experiments.annual.rerun_experiments`
(summary: `pipeline_outputs/experiments/rerun_20261122/experiments.json`; Atok application checks:
`.../application/`, from `python -m experiments.annual.application_review`). External scores are
retrospective. The Atok columns are label-free checks, not accuracy: confident share of cropland,
total-variation distance of the confident crop mix from PSA's 2024 Benguet shares (lower is closer),
and share of pixels outside the training range.

| Experiment | Selection score | Locked validation | JECAM | Combined | Atok confident | Atok mix distance | Atok outside range | Atok crop mix (carrot/cabbage/potato) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A annual S2+S1 | 0.789 | 0.923 | 0.395 | 0.588 | 0.32 | 0.62 | 0.60 | 0/0/100 |
| S2 only | 0.732 | 0.896 | 0.259 | 0.478 | 0.31 | 0.62 | 0.55 | 1/0/99 |
| curve shape B | 0.731 | 0.914 | 0.326 | 0.469 | 0.21 | 0.61 | 0.04 | 1/1/98 |
| curve shape B+radar | 0.761 | 0.932 | 0.338 | 0.563 | 0.18 | 0.63 | 0.04 | 0/0/100 |
| curve shape C | 0.761 | 0.939 | 0.382 | 0.579 | 0.23 | 0.63 | 0.12 | 0/0/100 |
| green-up timing | 0.782 | 0.926 | 0.395 | 0.609 | 0.32 | 0.62 | 0.59 | 0/0/100 |
| green-up referenced | 0.753 | 0.916 | 0.332 | 0.641 | 0.12 | 0.53 | 0.63 | 0/10/90 |
| short-cycle radar | 0.720 | 0.851 | 0.303 | 0.574 | 0.31 | 0.49 | 0.76 | 8/88/4 |
| short-cycle radar + S2 | 0.771 | 0.920 | 0.360 | 0.611 | 0.16 | 0.53 | 0.73 | 1/9/90 |
| RF tuned (half_features) | 0.802 | 0.933 | 0.318 | 0.616 | 0.37 | 0.61 | 0.60 | 0/2/98 |
| logistic regression (C=1) | 0.804 | 0.922 | 0.201 | 0.582 | 0.79 | 0.32 | 0.60 | 7/24/69 |
| gradient boosting (depth=6,rate=0.05,trees=300) | 0.814 | 0.942 | 0.292 | 0.608 | 0.69 | 0.31 | 0.60 | 7/25/68 |
