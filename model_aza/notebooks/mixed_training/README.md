# Mixed training experiment

Question: does the model transfer better when part of the external test places
(Madagascar, Senegal, Réunion, Antilles, France, Denmark, California) is added to
the European training parcels?

| Notebook | What it does |
|---|---|
| `01_mixed_training.ipynb` | Splits each place's test parcels in half by 0.1° blocks, compares `europe_only`, `europe_plus_added`, `added_only` and `new_place` on the held-out half (three split seeds), then maps Atok with models trained on all external parcels |

- Code: `model_aza/improved/mixing.py`; tests: `model_aza/tests/test_mixing.py`.
- Input: the improved pipeline's current inputs (`pipeline_outputs/improved/current_inputs.json`).
- Output: `pipeline_outputs/mixed_training/experiment_*` and `paoay_*`, each with a receipt.
- Features are the improved pipeline's selected set (`s2_harmonic_all_radar`); nothing here selects features or settings.
- The improved pipeline's own test scores remain the final results of the Europe-trained model.
  Models trained on every external parcel have no independent test left; their Atok maps are exploratory.
