# Feature designs

A feature is a number the model uses to distinguish crops. The window notebook first saves **64 base features per pixel**: five bands and three vegetation indices across eight half-month intervals.

This stage adds SAVI and measurements of how vegetation changes over the window.

## Three designs

| Design | Information used | Features |
|---|---|---:|
| `interval_only` | Five bands and four indices in each of eight intervals | 72 |
| `interval_and_shape` | Interval measurements plus changes over time | 120 |
| `shape_only` | Changes over time for the four indices | 48 |

Bands are B3, B4, B5, B8 and B11. Indices are NDVI, NDRE, NDMI and SAVI. B2 and clear-observation counts are used for inspection, not classification.

SAVI uses scaled reflectance and the recorded soil adjustment `L = 0.5`:

```text
SAVI = (1 + L) × (B8 − B4) / (B8 + B4 + L)
```

All designs use the same samples and partitions. Missing values or invalid index denominators cause an error; one design does not silently receive a different dataset.

## What “shape” means

Each index contributes 12 features:

| Features | Meaning |
|---|---|
| Seven differences | Change between consecutive intervals |
| Two slopes | Trend before and after the first maximum, in index units per day |
| Peak interval | First interval containing the maximum, numbered 1–8 |
| Amplitude | Maximum minus minimum |
| Integral | Trapezoidal area between the first and last interval midpoints |

Slopes and integrals use actual calendar-bin midpoints. A peak at a boundary has an unavailable phase slope recorded as zero, with a separate metadata flag. Ties use the first maximum. These describe numerical patterns; they do not verify planting or harvest dates.

## How to run it

Run [Feature Engineering](../notebooks/04_feature_engineering.ipynb) after the window notebook. It reads `analysis/bilinear_expanded/seasons/` and exports formal feature bundles inside that run.

| Window bundle | Feature bundle |
|---|---|
| `training_input/` | `feature_input/` |
| `exploratory_training_input/` | `exploratory_feature_input/` |

Each feature bundle contains `experiment_features.csv`, `configuration.json`, `coverage.json`, `window_input_provenance.json` and `feature_handoff.json`. Keep these files together for hash verification.

Formal exports require version 4 imagery: bilinear B5/B11 before compositing, with categorical quality masks preserved. Old imagery remains separate.

A formal bundle can be inspected while review/freeze requirements are pending. **Local exploratory training builds these designs directly from complete pixel tables**, so pending formal exports do not block the prototype. See [training](training.md) for the distinction.

## Compare designs fairly

Select the design using validation macro-F1 before test scoring. Ties prefer fewer features, then the design name. Keep all three designs on the same rows and partitions; do not use test scores to change windows or discard samples.

A changed feature recipe requires a new feature export. Changes to imagery, windows or selected pixels require a new window export first. Preserve prior model results in their original run folders.
