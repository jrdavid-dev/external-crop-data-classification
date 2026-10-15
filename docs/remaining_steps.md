# Review and train: Steps 9–15

These steps turn **reviewed field-seasons** into reproducible model comparisons.
The implementation is in [season_workflow.py](../season_workflow.py), with
executable cells and explanations in [the notebook](../Sen4AgriNet_Workflow_Outline.ipynb).

## 1. Prepare the inputs

Install `requirements-notebooks.txt` in the notebook environment, including
scikit-learn and joblib. Run Steps 6–8 to obtain quality records, scaled
selected-pixel bands, candidates and detector settings.

Run Step 9. It creates three templates in
`sen4agrinet_work/analysis/2020/seasons/review/`:

| Template | Copy to | Purpose |
|---|---|---|
| `season_reviews.template.csv` | `season_reviews.csv` | Review each candidate cycle |
| `field_registry.template.csv` | `field_registry.csv` | Record independent fields and partitions |
| `frozen_configuration.template.json` | `frozen_configuration.json` | Record settings and development decisions |

Templates can be regenerated. Your completed copies are preserved. Work with
the entire candidate ID as text; it links a review to one exact detector outcome.
Changed candidates or detector settings require another review.

## 2. Review field-seasons

Use `pending`, `reject` or `accept` in `decision`. Unsupported outcomes can stay
pending while the rest of the workflow produces coverage and exclusion reports.

For an accepted row, complete these fields:

| Column | What to provide |
|---|---|
| `season_id` | Stable identifier for the crop cycle; shared across fragments of that cycle |
| `crop` | Crop supported by evidence; must match the source region label |
| `label_status` | `confirmed` or `reported` |
| `label_evidence` | Specific dated crop record or declaration and its reference period |
| `timing_evidence` | Evidence supporting the estimated green-up interval |
| `cycle_evidence` | Why the crop label applies to this cycle and four-month window |
| `reviewer`, `review_notes` | Who reviewed it and why it was accepted |
| `resolved_onset_month` | `YYYY-MM` if onset uncertainty crosses months |
| `month_resolution_evidence` | Why that month is supported |

Keep candidate IDs and detector fingerprints unchanged. Detector-rejected
outcomes cannot be accepted directly; correct development settings or inputs and
rerun detection. Ambiguous or provisional candidates need explicit evidence for
the selected cycle. A reviewer is responsible for the evidence's validity;
nonempty text alone does not establish scientific correctness.

Optionally supply both `reference_greenup_start` and `reference_greenup_end`
as ISO dates from independent green-up evidence. The audit reports the distance
between the estimated and reference intervals: zero means overlap. Do not use
planting dates as if they were observed green-up dates. Without this reference,
the timing result is labeled `review_only`.

## 3. Freeze the development rules

Complete `development_evidence` with the development report or decision record.
Keep `detector_parameters` equal to the Step 8 settings used for the candidates.
Set the study classes, minimum valid fraction, confidence weights, forest
parameters and random seed before final evaluation. Then set `frozen` to `true`.

Also record `dataset_revisions`, `source_terms`, `map_reference_period` and
`limitations` so the run package explains the intended study. These descriptive
records are user-supplied; the software cannot verify source permissions or
field evidence. Do not claim a three-class experiment without a cabbage source
and cabbage examples in every required training/test set.

The supported primary rule is `onset_month_plus_three`; the missing-data policy
is `complete_only`. A window beginning October 2020 ends on February 1, 2021
(exclusive). It needs six bins from 2020 and two from 2021. Missing next-year
data are reported rather than replaced with 2020 observations.

## 4. Build features and inspect losses

Run Steps 9–12 again after completing reviews. Features use already-scaled
reflectance from Step 6, with no additional scaling or temporal median.
Feature order is `bin_01_B3` through `bin_08_NDMI`; each bin includes five
bands and three indices. The band-only comparison uses the corresponding
40 columns. Coordinates, counts and evidence remain metadata.

Every selected pixel needs all 64 finite values and passing quality in all eight
bins. Zero denominators, invalid clear counts, missing bins and low coverage
cause exclusion. The `reconstructed` flag is always false in this baseline.

| Report | How to use it |
|---|---|
| `season_review_audit.csv` | Inspect every candidate's acceptance or exclusion reason |
| `accepted_seasons.csv` / `.json` | Check crop evidence and four-month dates |
| `features/complete_features.csv` | Inspect eligible pixel windows |
| `features/feature_exclusions.csv` | Identify missing bins for known pixel locations |
| `features/sample_losses.csv` | Compare selected, accepted, excluded and unlocated counts by crop/region/group |
| `features/bin_coverage.csv` | Inspect usable selected pixels in every requested bin |

Counts refer to pixel windows: the same pixel in two reviewed seasons contributes
two rows. Loss records preserve fields with no usable pixel records. Field size
here means the selected-pixel budget, not a measured full-field area.

## Local data and field partitions

Reserve independent local Benguet test fields before selecting models or rules.
Complete the field registry with these columns:

```csv
partition_group,field_id,source,partition,identity_evidence
ES|2020|parcel_1,external_field_1,external,train,Reviewed source geometry and parcel records
PH|local|field_1,benguet_field_1,local,train,Reviewed independent field boundary
PH|local|field_2,benguet_field_2,local,test,Reserved independent field boundary
```

This illustrates the schema, not an adequate class-balanced experiment.
Allowed sources are `external` and `local`; partitions are `train`, `validation`
and `test`. All fragments, seasons and windows of one field share one partition
and source. Test records must be local and have confirmed field-season labels.

Process local imagery with the same bands, masks, grid checks, scaling, detector,
review and feature functions. Use globally unique region IDs. Combine the local
and external feature rows before Step 13, for example:

```python
local_feature_rows = season_workflow.read_table(local_complete_features_path)
combined_feature_rows = feature_rows + local_feature_rows
partitioned_rows = season_workflow.assign_partitions(
    combined_feature_rows, field_registry
)
```

When using the notebook, assign that combined list to `feature_rows` before
Step 13, and include local input files and loss reports in Step 15's provenance
and coverage records. Keep the original external list when rerunning to avoid
appending the same local rows twice.

Exact coordinate duplicates in the same CRS cannot claim different field IDs.
Duplicate field-season pixels are rejected. Resolve overlapping geometries and
duplicates across CRSs during identity review; coordinate checks alone cannot
establish independence. Different disjoint fragments can share a reviewed field
ID, so long as their partition/source assignments agree.

## 5. Compare the models

Step 14 trains fresh external-only, local-only and combined forests for both
40 and 64 features. Each training source and the shared local test set must
contain every configured class. The combined experiment reuses exactly the same
local training rows as the local-only experiment. Validation rows are excluded
from both training and final test scoring.

One field's confidence budget is shared across all its rows and seasons:
`label_confidence / field_row_count`. This controls repeated pixels' influence.
Mixed-status rows share the same denominator, so weaker evidence cannot increase
the total field budget. Class weights, when configured, multiply these weights.

Each feature design exports three `.joblib` model bundles, test predictions,
settings and `metrics.json`. Bundles contain the classifier, ordered feature
columns and configuration. Scores include per-crop precision/recall/F1, macro-F1,
balanced accuracy and confusion matrices in configured class order. Pixel metrics
are correlated within fields; sample counts report independent fields too.

Inspect both predeclared feature comparisons as final results. Do not choose
parameters or a winning feature design using final test performance. Field-level
uncertainty estimation and tuning on development folds remain separate analyses.

## 6. Preserve the run

Step 15 creates `package/run_manifest.json` and `package/RUN_README.md` with
configuration, package versions, file hashes, sample counts and detector/quality
losses. It lists missing files explicitly. Only files created by the current
model cell are listed as model artifacts.

Preserve the data, reviews and generated outputs separately from Git. Mapping
coverage stays `not_measured` until an actual mapping run measures it. No imagery
export is launched by these steps, and no export completion is inferred.

## Verification

Run the processing and synthetic model tests from the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

On Windows, use `.\.venv\Scripts\python.exe` instead. Synthetic tests check
calendar boundaries, evidence gates, feature completeness, field leakage,
model exports and provenance. They do not validate real crop identities or
phenology accuracy.

## Optional development checks

The notebook also provides two checks under `seasons/experiments/`:

- **Coverage:** compare 70%, 80% and 90% valid-pixel fractions on saved summaries,
  preserving original quality flags. This does not change the pixel cloud masks
  or measure remaining cloud contamination.
- **Window shifts:** construct eight consecutive calendar bins one bin earlier
  or later, alongside the primary window. Variants preserve the primary dates,
  label the shifted dates and export complete features, exclusions and losses.

Use these on development fields before freezing the main experiment. A shifted
window is a sensitivity design, not four whole calendar months. Missing imagery
at either boundary remains missing. The optional cells do not refit or choose a
model from test results. Feature and training-source comparisons are already
implemented in Step 14.
