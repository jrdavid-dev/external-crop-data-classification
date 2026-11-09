# Feature preparation

Notebook **04** saves complete four-month windows with **64 base features per pixel**: five spectral bands and NDVI, NDRE and NDMI across eight half-month intervals. Notebook **05** adds SAVI and temporal-shape measurements, then exports the exact feature bundle consumed by **06**.

## Three designs

| Design | Information used | Predictors |
| --- | --- | ---: |
| `interval_only` | Five bands and four indices in eight intervals | 72 |
| `shape_only` | Temporal shape of the four indices | 48 |
| `interval_and_shape` | Both designs combined | 120 |

Bands are B3, B4, B5, B8 and B11. Indices are NDVI, NDRE, NDMI and SAVI. B2 and clear-observation counts support inspection, not classification.

SAVI uses scaled reflectance and the recorded soil adjustment `L = 0.5`:

```text
SAVI = (1 + L) × (B8 − B4) / (B8 + B4 + L)
```

Each index contributes seven consecutive differences, slopes before and after the first maximum, peak interval, amplitude and trapezoidal integral. Slopes/integrals use actual calendar-bin midpoints. A boundary peak has its unavailable phase slope set to zero with a separate metadata flag. Ties use the first maximum. These describe the measurements, not verified planting or harvesting dates.

## Notebook 05

1. Select the same collection and analysis mode as notebook 04.
2. Load its explicit `window_handoff.json`.
3. Verify inputs, complete values, identities and locked partitions.
4. Compute the three designs on one common sample set.
5. Export `feature_handoff.json` with feature rows, recipe and hashed provenance.
6. Show counts, readiness and the path to use in notebook 06.

Outputs are immutable, content-keyed artifacts. Changed windows, samples or feature policy create a separate export; earlier files stay intact. Missing values or invalid index denominators stop export rather than silently removing different samples for each design.

`analysis_mode = "pilot"` reports unmet collection budgets without falsely declaring production reviews or settings frozen. `"strict"` requires the adaptive collection targets. Both preserve complete-data and parcel-separation requirements. Formal reviewed/frozen feature bundles remain a separate advanced production path.

Notebook 06 reads these saved predictors directly. It selects the feature design using validation macro-F1 before test scoring; all designs share the same rows and partitions. See [training](training.md) and [workflow](workflow.md).
