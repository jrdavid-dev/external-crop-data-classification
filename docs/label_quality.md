# Crop evidence, timing and training weights

Crop identity and vegetation timing are separate questions. Satellite curves show growth, but do not independently prove which crop grew in a field.

## Field-season evidence

Benguet fields can rotate crops and have multiple cycles. A general calendar suggests what is plausible; it does not identify a crop in a particular field-season.

The adviser supplied this target-crop context:

| Crop | Planting / production context | Harvest context |
|---|---|---|
| Carrot | April–October | June–December |
| Cabbage | April–September | June–December |
| White potato | April–May and September–October | June–August and December |

These are adviser-provided statements, not independently verified field-season records. Their dates overlap. Do not label a crop from a calendar alone or force all agricultural fields into three classes. Preserve other-crop and unknown records.

S4A annual declarations provide parcel-specific evidence. Their reference period and relevance to each selected cycle still need checking. Spectral agreement is not independent crop confirmation. Green-up is not necessarily planting; decline is not necessarily harvest.

## What to record

| Record | Purpose |
|---|---|
| Field ID and location with CRS | Identify fields and possible duplicates |
| Year and season ID | Separate successive cycles |
| Reported crop and source | Preserve the original declaration or account |
| Evidence type and date | Distinguish visits, records, interviews and general calendars |
| Planting/harvest ranges | Keep timing uncertainty visible |
| Label status and review notes | Preserve conflicts and alternatives |
| Estimated green-up interval | Separate satellite estimates from recorded planting |
| Accepted label/window and reason | Document the final decision |

Keep personal farmer/contact information separate from modeling tables when not needed for analysis.

## Label statuses

| Status | Evidence | Use |
|---|---|---|
| Confirmed | Reliable dated field-season record or crop-identifying observation | Trusted training; independent records can support testing |
| Reported | Specific account or declaration accepted under documented rules | Training candidate with uncertainty |
| Tentative | Mainly a calendar, usual rotation or model inference | Separate; optional experiment |
| Unknown | Multiple plausible crops or insufficient identification | Exclude from supervised target-class training |

A dated visit confirms what was present on that date, not the entire four-month window. These statuses are evidence categories, not probabilities. Keep declarations distinguishable from interviews even when their status matches.

## Optional confidence weighting

Random Forest accepts per-row `sample_weight`. Weighting can reduce weaker labels' influence; it cannot repair a wrong label or establish truth.

Proposed development examples are confirmed 1.0, reported 0.5, tentative 0.1 or excluded, and unknown excluded. These are experimental weights, not validated confidence probabilities. Test a small predefined set of policies on development fields and freeze the choice before final testing.

Pixels within fields are correlated. Control field contribution as well as pixel weights:

1. Cap pixels per field and give them the field's confidence weight.
2. Allocate a field confidence budget `c` across `n` rows as `c / n`.

Define whether budgets cover all rows/windows of a field or separate seasons. Repeated windows must not silently multiply influence.

## Confidence is not class balance

Confidence weighting addresses evidence; class balancing addresses representation. Random Forest class weights and sample weights multiply, so examine effective weights when using both.

No weighting method replaces missing classes or enough independent fields. Do not label unknown samples to balance counts. Equal pilot counts still need completeness and crop-cycle checks.

## Evaluation

Hold partitions, windows, preprocessing and independent test fields fixed. Start with the strongest labels. Compare adding reported labels with equal weights against the same samples with reduced confidence weights. Keep class balancing fixed to isolate that effect.

Report pixel and independent-field counts by crop/status, quality losses, per-crop precision/recall/F1, macro-F1, balanced accuracy and a confusion matrix. Where possible, estimate uncertainty by resampling fields rather than correlated pixels.

Predictions can prioritize review but are not independent label confirmation. If used as training labels, call them pseudo-labels and evaluate separately. Never test against the same guessed labels used for training.

## Next actions

Collect specific field-season evidence, check declaration meaning, reserve independent confirmed Benguet test fields and validate timing. Accept windows only when crop-to-cycle correspondence is supported.

See the [study plan](study_plan.md) and [workflow](workflow.md). Confidence weighting and final training are planned work, not completed results.
