# Experiments

Ideas that were tried alongside the main pipeline (`src/crop_pipeline/`, `notebooks/`). Each one
was judged on development data first (spatial CV within the training split and holding out one
development dataset at a time; the selection score is their mean). Only what was adopted lives in
the main pipeline; everything else is kept here with its result so the comparison stays honest.
Nothing here changes the main pipeline's code, partitions, stage pointers or frozen models. It
reads them, and writes its own outputs to `pipeline_outputs/` (ignored by Git).

## Index

External scores (JECAM macro-F1, combined pool) were computed after the test parcels had already
been seen for the window models, so they are retrospective. Numbers are for the expanded dataset
(Jose's plus 1,500 extra development parcels) unless marked; the annual Random Forest scores
JECAM 0.395 on it and 0.348 on Jose's original dataset.

| Experiment | Folder | Question | Result | Adopted |
|---|---|---|---|---|
| Pilots | `pilots/` | Which annual features (harmonics, Sentinel-1, ERA5, masks, clustering) are worth a full run? | Harmonic curves and Sentinel-1 helped on development data | Yes: the basis of the annual model |
| Improved pipeline v1 | `improved_v1/` | First end-to-end annual pipeline (5 pixels/parcel, fit on train + validation) | JECAM 0.374, combined 0.541; superseded by `crop_pipeline.training.annual_workflow` (20 pixels/parcel, locked training split) | Superseded |
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
| `notebooks/` | Experiment notebooks: `improved_v1/`, `mixed_training/`, `transfer/`, `tuning/`, `walkthrough/` |
| `tests/` | Tests for the experiment code |
| `archive/notebooks/` | Superseded experiment notebooks (21 Nov) with their findings |
| `archive/removed_test_cohort_supplement_matching.patch` | A fix for re-collecting the carrot/potato supplement, not applied to the main pipeline yet; apply with `git apply` if agreed |

Experiment recipes register themselves with the main pipeline when imported
(`annual_workflow.register_recipes`), so their frozen models can be tested and mapped with the
pipeline's own stages; they are never selection defaults. Non-forest classifiers are frozen with
`annual_workflow.freeze(..., estimator=...)`.

## Running

Notebooks find the repository root and add both `src/` and the root to `sys.path`. Tests:

```bash
.venv/bin/python -m unittest discover -s experiments/tests -t . -v
```

XGBoost (`classifiers/`) needs the OpenMP runtime on macOS (`brew install libomp`); without it the
comparison uses scikit-learn's gradient boosting with the same grid and records which ran.

Who did what is recorded in the Git history; most of this folder is Aza's work building on the
external study's code in `crop_pipeline`.
