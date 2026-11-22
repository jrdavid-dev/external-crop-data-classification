# Mixed training experiment

Question: does the model transfer better when part of the external test places
(Madagascar, Senegal, Réunion, Antilles, France, Denmark, California) is added to
the European training parcels?

| Notebook | What it does |
|---|---|
| `01_mixed_training.ipynb` | Splits each place's test parcels in half by 0.1° blocks, compares `europe_only`, `europe_plus_added`, `added_only` and `new_place` on the held-out half (three split seeds), then maps Atok with models trained on all external parcels |

- Code: `experiments/mixed_training/mixing.py`; tests: `experiments/tests/test_mixing.py`.
- Input: the improved pipeline's current inputs (`pipeline_outputs/improved/current_inputs.json`).
- Output: `pipeline_outputs/mixed_training/experiment_*` and `paoay_*`, each with a receipt.
- Features are the improved pipeline's selected set (`s2_harmonic_all_radar`); nothing here selects features or settings.
- The improved pipeline's own test scores remain the final results of the Europe-trained model.
  Models trained on every external parcel have no independent test left; their Atok maps are exploratory.

## Weighted mixed training (22 November 2026)

`python -m experiments.mixed_training.weighted <new output folder>` reruns the idea on the current annual
inputs (recipe A predictors, 300-tree forests for every model) with extra importance for the added parcels:

1. Whole 0.1° blocks per place (never split by crop) go to the added or the held-out half.
2. Every model is scored on the held-out half only.
3. The weight of added parcels (1×, 2×, 5×) is chosen by block-wise CV inside the added half; the
   held-out half is scored once per weight afterwards, for transparency.
4. New place: development plus the added half of the other places, scored on one place's held-out half.
5. Three split seeds.

Results: `pipeline_outputs/mixed_training/weighted_20261122/` (`summary.csv`, `accuracy_by_place.csv`,
`experiment.json`). Held-out macro-F1, mean over seeds 1/2/3:

| Model | All held-out | JECAM held-out | Tropical places |
|---|---:|---:|---:|
| Europe only | 0.574 | 0.294 | 0.295 |
| Europe + added, weight 1 | 0.813 | 0.665 | 0.628 |
| Europe + added, weight 2 | 0.817 | 0.638 | 0.605 |
| Europe + added, weight 5 | 0.819 | 0.616 | 0.598 |
| New place (chosen weight) | 0.584 | 0.290 | 0.274 |

CV chose weight 5, 2 and 1 for the three seeds (CV differences under 0.011), so no weight is reliably best.
Same-place labels help a lot; extra weight does not add to it and lowers JECAM; a new place gains nothing.
