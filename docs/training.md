# Training and evaluation

Run [four-month window preparation](../Crop_Four_Month_Windows.ipynb) before opening [model training](../Crop_Model_Training.ipynb). Features alone are not sufficient: model comparisons also require field identities, frozen settings and independent local test labels.

## Required preparation

Step 9 creates templates under `analysis/pilot_3000/seasons/review/`:

| Template | Completed file | Required for |
|---|---|---|
| `field_registry.template.csv` | `field_registry.csv` | Assigning independent fields and partitions |
| `frozen_configuration.template.json` | `frozen_configuration.json` | Reference-supported acceptance and model training |
| `season_reviews.template.csv` | `season_reviews.csv` | Reference-supported acceptance or optional human timing overrides |

Automatic exploratory timing does not require a completed manual review CSV. Templates can be regenerated without replacing completed files.

## Local data and field partitions

Use this registry schema:

```csv
partition_group,field_id,source,partition,identity_evidence
source_parcel_1,external_field_1,external,train,Reviewed parcel identity and boundary
local_parcel_1,local_field_1,local,train,Reviewed local field boundary
local_parcel_2,local_field_2,local,test,Reserved independent labeled field
```

This is a schema example, not an adequate experiment. Allowed sources are `external` and `local`; partitions are `train`, `validation` and `test`.

Keep every fragment, season and pixel from the same independent field in the same source and partition. Do not randomly split pixels within a field. Final test fields must be local, independently labeled, and have confirmed crop labels and reference-supported windows. Resolve field overlap across coordinate systems during identity review; coordinate checks alone do not establish independence.

Process local data with the same imagery and feature rules. Before Step 13, combine its reference-supported feature rows with the selected external `feature_rows`:

```python
local_feature_rows = season_workflow.read_table(local_complete_features_path)
feature_rows = feature_rows + local_feature_rows
```

Add local rows only once per run. Include their input paths and losses in Step 14's provenance. The source notebooks do not automatically collect or label local test data.

## Freeze the configuration

Record the development decision in `development_evidence`; keep `detector_parameters` equal to the settings used to generate candidates. Specify the study classes, quality threshold, timing policy, random seed, confidence weights and forest parameters, then set `frozen` to `true`.

Record dataset revisions, source terms, map reference period and limitations. The supported policies are `window_rule=onset_month_plus_three` and `missing_data_policy=complete_only`. Choose rules using development fields, never final test performance.

## Reference evidence and optional overrides

For `accept_reference`, supply `season_id`, `crop`, `label_status`, `label_evidence`, `timing_evidence`, `cycle_evidence`, `reviewer` and `review_notes`. Retain the template's candidate ID and detector fingerprint. If onset spans two months, also supply `resolved_onset_month` and `month_resolution_evidence`.

An optional `accept_estimate` override needs a season ID, reviewer, notes and timing evidence; month-boundary resolution also needs supporting evidence. It remains an estimated window with reported crop labels. `reject` excludes the candidate; legacy `accept` means reference-supported acceptance.

Nonempty evidence fields are not proof of scientific correctness. A crop declaration or dated visit does not automatically verify the entire four-month cycle. Changed detector settings or candidates invalidate earlier reviews.

## Export and choose model inputs

Run preprocessing Steps 9–14 again after updating reviews or configuration. Step 14 exports:

| Bundle | Allowed rows |
|---|---|
| `seasons/training_input/` | Reference-supported windows |
| `seasons/exploratory_training_input/` | Supported rows plus timing-qualified external training estimates |

Each bundle contains `training_handoff.json`, `partitioned_features.csv`, `configuration.json` and `coverage.json`. Keep them together. Training verifies file hashes; re-export after changing bundle inputs.

The training notebook defaults to the primary bundle. To use qualified satellite estimates, set `handoff_path` to the exploratory bundle explicitly. Exploratory outputs are saved under `seasons/exploratory_estimated/`. Estimated windows cannot become local training or test truth.

An empty export is marked `pending_inputs`; it does not fit a model.

## Fit and assess the models

Run the training notebook sequentially in a fresh kernel. The current design trains external-only, local-only and combined Random Forests for both 40- and 64-feature inputs: six models total.

**Every configured crop must appear in external training, local training and the shared local test set.** The combined experiment reuses the same local training rows as the local-only experiment. Validation rows are reserved and excluded from fitting and final scoring.

A field's confidence budget is divided across its rows and seasons. Default confidence weights distinguish confirmed and reported labels; they are development settings, not probabilities of correctness. Weighting cannot fix incorrect labels or missing classes.

Inspect per-crop precision, recall and F1, macro-F1, balanced accuracy and the confusion matrix. Report independent-field counts alongside pixel counts because pixels within a field are correlated. Preserve models, predictions, configuration and the generated run manifests. Field-level uncertainty analysis and full-area mapping are separate work.
