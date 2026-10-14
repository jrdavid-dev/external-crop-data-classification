# Benguet crop labels: confidence, records, and training weights

Discussion notes prepared October 13, 2026.

## The problem

Benguet has multiple cropping and crop rotation. A general crop calendar tells us which crops are plausible during a period, but does not identify the crop in a particular field and growing cycle.

The adviser supplied the following context:

| Crop | Planting / production context | Harvest context |
|---|---|---|
| Carrot | Planting April–October | June–December |
| Cabbage | Planting April–September | June–December |
| White potato | Planting April–May and September–October | June–August and December |
| Radish | Local production documented in 2025 | Monthly timing insufficiently documented |
| Chinese cabbage | December production documented locally | Full season not established |
| Broccoli | July production documented locally | Full season not established |
| Celery | July production documented locally | Full season not established |
| Snap bean | July production documented locally | Full season not established |

These statements are adviser-provided context, not independently verified field-season records. A production observation in a month does not establish planting date or full cycle duration. Do not apply a documented year's observation automatically to every year.

Carrot, cabbage, and potato calendars overlap. A field greening up in April could contain any of them, or another crop. Do not force every agricultural field into the three target classes. Preserve other-crop and unknown records for assessing the intended mapping scope.

## Crop identity and growth timing are different questions

- **Crop identity:** What crop grew in this field during this cycle?
- **Growth timing:** When did vegetation establish, peak, and decline?

Cloud-masked satellite timelines help answer the second question. They do not independently prove the first. Green-up is not necessarily planting, and canopy decline is not necessarily harvest.

S4A and FranceCrops provide parcel-specific crop declarations, which are stronger evidence than a general calendar. Their annual labels still require checking against the declaration's meaning/reference period and the selected cycle. They can be used as potentially noisy external reference labels; they are not automatically independently confirmed ground truth.

No global model reliably identifying all three target vegetables everywhere has been verified in this project. A model prediction may help prioritize review, but is not independent confirmation of a declaration, particularly when the model learned from that same source.

## How to identify a field-season label

1. Assign a field location or boundary and a stable field ID.
2. Ask the farmer or adviser which crop was grown in that specific field and year/cycle.
3. Record approximate planting and harvest ranges. Exact days are not required to start, but uncertainty must remain visible.
4. Ask whether the account describes a remembered field-season, a dated observation, an administrative declaration, or only the usual rotation.
5. Compare the account with cloud-masked imagery and its growth timeline. Record agreement or conflict without treating spectral agreement as independent crop identification.
6. If several crops or cycles remain plausible, retain the alternatives in notes and leave the accepted label empty or provisional.
7. Review conflicting evidence instead of automatically accepting the most confident-sounding source.

Use the crop calendar as a plausibility check, not a label generator. Assess crop identity and date precision separately: a crop can be well identified while its planting month is uncertain.

## Label statuses

These are operational project categories. Define and apply the evidence criteria consistently; they are not numerical confidence probabilities.

| Status | Evidence | Recommended use |
|---|---|---|
| Confirmed | Specific field-season supported by a reliable dated record or direct crop-identifying observation | Trusted training; independently verified records can support testing |
| Reported | Farmer/adviser identifies the specific field-season, without independent corroboration; or a source declaration accepted under documented rules | Training candidate with uncertainty documented |
| Tentative | Crop inferred mainly from a calendar, usual rotation, or model prediction | Keep separate; optional later experiment |
| Unknown | Multiple plausible crops or insufficient crop-identification evidence | Unlabeled; exclude from supervised target-class training |

A dated visit confirms what was present on that date; it does not automatically confirm the entire four-month window. Keep source-specific administrative declarations distinguishable from local interviews even if they receive the same experimental status.

## What to record

Use a spreadsheet or table with at least:

| Column | Meaning |
|---|---|
| field_id | Stable identifier for the independent field |
| location_or_boundary | GPS point or polygon, with CRS recorded |
| year | Year associated with the evidence |
| cycle_id | Distinguishes growing cycles within the same field |
| crop_label | Accepted crop, or empty if unresolved |
| candidate_crops | Alternatives when identity remains uncertain |
| planting_start / planting_end | Reported or observed planting range, if available |
| harvest_start / harvest_end | Reported or observed harvest range, if available |
| evidence_source | Farmer, adviser, dated visit, source declaration, model, etc. |
| evidence_details | What was said/observed and which field-season it refers to |
| label_status | Confirmed, reported, tentative, or unknown |
| date_evidence_status | Whether timing is recorded, reported, inferred, or unknown |
| satellite_growth_interval | Estimated vegetation-cycle interval |
| agreement_or_conflict | How imagery relates to the account |
| window_start / window_end | Selected four-month input dates |
| crop_to_window_support | Evidence linking the crop identity to the selected cycle/window |
| review_date | When the record was assessed |
| training_eligibility | Accept, review, or exclude, with reason |
| training_weight | Experimental weight assigned under the chosen policy |

### Examples

| Field | Crop | Evidence | Status |
|---|---|---|---|
| F001 | Carrot | Farmer recalls carrot planted in April 2025; imagery broadly agrees | Reported |
| F002 | Empty / unknown | April green-up; carrot, cabbage and potato all fit the calendar | Unknown |
| F003 | Cabbage | Dated crop-identifying field visit | Confirmed for the observation date; assess window relevance separately |

Keep personally identifying farmer/contact information separate from modeling tables when it is not needed for analysis.

## Giving stronger labels more training weight

Random Forest can accept a weight for each training row through scikit-learn's `sample_weight` argument to `fit`. Higher-weight rows contribute more to weighted tree fitting. Weighting reduces the influence of uncertain labels; it does not correct an incorrect label or establish its truth.

Suggested starting settings to evaluate:

| Status | Example relative weight |
|---|---:|
| Confirmed | 1.0 |
| Reported | 0.5 |
| Tentative | 0.1, or exclude from the initial model |
| Unknown | Exclude |

These values are proposed experimental settings, not research-validated probabilities. A weight of 0.5 does not mean the label has a 50% probability of being correct. Select a small predefined set of weight policies on development data, then freeze the chosen policy before final testing.

Sample weights do not guarantee a proportional change in predictions or a proportional number of samples per tree. Their practical effect must be measured. If explicit weighted resampling is introduced, describe it as a separate method.

## Control field contributions as well as pixel weights

Pixels within a field are correlated. Thousands of tentative pixels can collectively outweigh a few confirmed fields despite lower per-pixel weights.

Two reasonable approaches are:

1. Sample a comparable, capped number of pixels per field, then assign the field's confidence weight to those rows.
2. Allocate a total weight budget per field and distribute it across its sampled rows.

For the second approach, if field confidence weight is c and the field contributes n rows, a proposed row weight is **c / n**. If several overlapping windows are included, define whether n covers all rows/windows for that independent field or a fixed budget split across its field-seasons. Do not let repeated windows silently multiply its influence.

These are proposed design choices to evaluate. A cap may be easier for the initial experiment. Report both pixel counts and independent-field counts by crop, source, and confidence status.

## Confidence weighting and class balancing

Label confidence and class imbalance are different issues:

- Confidence weighting reduces the influence of less-supported labels.
- Class weighting or sampling addresses unequal representation of cabbage, carrot and potato.

In scikit-learn Random Forest, `class_weight` and supplied sample weights multiply together. A large class weight could increase the influence of tentative minority-class rows. Inspect the effective weights and class/field totals; do not assume that enabling both automatically solves the problem.

No balancing policy can substitute for missing crop classes or sufficient independent fields. Do not give unknown crops a target label just to balance counts.

## Recommended experiments

Keep the field partitions, preprocessing, windows and independent test set fixed across comparisons.

| Experiment | Training samples | Purpose |
|---|---|---|
| A: trusted baseline | Confirmed only | Establish performance with strongest available evidence |
| B: add reported labels | Confirmed + reported, with equal confidence weights | Measure the effect of additional reported data |
| C: weight reported labels | Same samples as B, with lower weights for reported rows | Isolate the effect of confidence weighting |
| D: optional tentative extension | Same as C plus low-weight tentative samples | Test whether provisional labels help or hurt |

Hold any class-balancing method fixed when comparing B and C. Evaluate weight settings using field-separated development data. Keep final independently confirmed Benguet test fields untouched and separate from training/selection.

Report per-crop precision, recall and F1, macro-F1, balanced accuracy and a confusion matrix. Report independent-field counts and quality/coverage exclusions. Where feasible, estimate uncertainty using fields rather than treating pixels as independent observations.

This confidence experiment is additional to the thesis's main external-only, local-only and combined-data comparison. Begin with a manageable baseline rather than testing every source, feature and weight combination at once.

## Using model predictions for review

A model trained on supported examples can flag uncertain cases and prioritize farmer/adviser follow-up. Save the predicted crop, scores, model version and evidence status separately from the original label.

- Agreement with a source declaration is not independent confirmation.
- High model confidence can still be wrong in a new region.
- If predictions become provisional training labels, call them pseudo-labels and evaluate that as a separate experiment.
- Do not evaluate accuracy against the same guessed labels used to train the model.

## Immediate next steps

1. Create the field-season evidence table.
2. Ask the adviser for specific field locations and remembered/recorded crop seasons, not only regional calendars.
3. Review the temporal meaning of S4A/FranceCrops declarations.
4. Build a small supported reference set and reserve independent confirmed Benguet fields for testing.
5. Apply the cloud-masked growth detector to estimate timing, preserving uncertainty.
6. Select windows only when crop-to-cycle correspondence is sufficiently supported under documented rules.
7. Train the trusted baseline, then compare reported-label inclusion and weighting.
8. Keep tentative and unknown samples separate until their role is explicitly evaluated.

The project can use imperfect external declarations without verifying every farmer individually, provided their provenance and limitations are clear and final performance is assessed on independent, well-supported local evidence.

## Related project files

- `Benguet_External_Crop_Data_Plan.md`
- `Benguet_S4A_to_Four_Month_Workflow.md`
- `Sen4AgriNet_Workflow_Outline.ipynb`
- `FranceCrops_Workflow_Outline.ipynb`

This document records the proposed project approach. The adviser calendar and example weight values are not independent validation of crop identity, planting dates or weighting effectiveness.
