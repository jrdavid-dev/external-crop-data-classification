# Project plan and deliverable checklist

## Current classification workflow

```text
Sen4AgriNet + FranceCrops              CropHarvest JECAM
        Train and tune                  Independent test
               │                               │
               ▼                               ▼
         Freeze models ──────────────► Score frozen models
               │
               └── January–April model ──► Paoay prototype map
```

All three paths use the same Sentinel‑2 cloud and shadow masking, half-month composites, bands, and indices.

| Step | What happens | What we get |
|---|---|---|
| **1. Prepare the parcels** | Use Sen4AgriNet for carrot and potato and FranceCrops for cabbage and potato. Reserve CropHarvest JECAM for testing and check that its parcels don’t overlap the training parcels. | Separate development and test sets. Training imagery is in progress, and JECAM imagery is pending. |
| **2. Turn images into usable inputs** | Mask clouds and shadows, then combine clear observations into one composite per half-month. Check which parcels and intervals have enough clear data. | A coverage report and a record of missing or excluded data. |
| **3. Build two four-month views** | The **fixed** view uses January–April for every parcel. The **aligned** view starts around detected green-up, so a parcel that greens up in March uses March–June. | Two sets of model inputs, each with eight half-month composites. |
| **4. Train, freeze, and test** | Tune the Random Forests on the development sources, freeze them, and score CropHarvest. Compare both methods on the same usable pixel locations and report each method’s coverage separately. | Parcel-level scores showing which crops were confused and how many parcels could be classified. |
| **5. Update the Paoay map** | Apply the frozen January–April model to Paoay’s January–April 2024 imagery. Display the predicted crops, uncertainty, and gaps in coverage. | A preliminary local map. An earlier prototype exists, and the revised map awaits the new model. |

## Deliverable checklist

Checked items are done. Open items are marked **In progress**, **Needs review**, or **Pending**.

## 1. Prepare and assess the project datasets

- [x] Identify training sources for carrot, cabbage, and potato and screen CropHarvest JECAM test candidates.
- [x] Import the existing Paoay reference areas and preserve their crop calendar notes.
- [ ] **In progress** — Finish assembling the training and test imagery.
- [ ] **Needs review** — Profile candidate and usable parcels by crop, location, and season. Confirm class imbalance, label-to-imagery date matches, and any boundaries that still need digitizing.
- [ ] **Needs review** — Confirm the availability and coverage of soil, terrain, climate, and land-cover layers for the broader project.

## 2. Implement remote-sensing preprocessing and features

- [x] Implement Sentinel‑2 cloud and shadow masking, half-month composites, and B3, B4, B5, B8, B11, NDVI, NDRE, and NDMI features.
- [x] Implement coverage and observation loss reporting for new imagery. Older imagery has limited loss diagnostics.
- [ ] **Needs review** — Verify the new imagery diagnostics on a small live sample and summarize clear observations, missing intervals, cloud and shadow losses, and usable coverage.

## 3. Implement the crop-classification prototype

- [x] Implement parcel-level split and exclusion rules, separate CropHarvest scoring, and the Random Forest comparison of fixed and green-up-aligned windows.
- [x] Build an earlier preliminary Paoay map and verify map assembly locally.
- [ ] **In progress** — Complete the independent evaluation and refresh the Paoay prototype with the frozen January–April model.
- [ ] **Needs review** — Check whether JECAM meets **20 usable parcels and 400 usable pixels per crop**. Report shortfalls, per-crop results, and the geographic concentration of the test parcels.

## 4. Implement the crop-suitability prototype

- [ ] **Needs review** — Confirm the input layers, crop criteria, AHP weights, and limiting-factor rules.
- [ ] **Pending** — Apply AHP/WLC and limiting factors to produce preliminary **S1, S2, S3, and N** maps. Track this as a separate project workstream.

## 5. Identify data limitations and validation needs

- [x] Identify the main classification limitations, including uneven geographic coverage, missing imagery, crop rotations, and limited dated Paoay references.
- [ ] **Needs review** — Audit EuroCrops v2.0 as a possible parcel source and check whether independent field references are available for USDA CDL.
- [ ] **Needs review** — Develop the provisional January 2027 Paoay field plan. Confirm access, candidate fields, timing, and how to collect dated evidence that matches the imagery season.
- [ ] **Pending** — Use the prototype results to prioritize improvements for classification and suitability mapping.

## 6. Submit a journal or conference paper

- [ ] **Needs review** — Choose the venue and agree on the research question, evidence to include, and submission timeline.
- [ ] **Pending** — Draft and submit the paper. Submission is required for THSST2 and is outside the prototype grading checklist.

## Open TODOs at a glance

- [ ] **Check the CropHarvest test set.** Confirm at least 20 usable parcels and 400 usable pixels per crop. Note any shortfalls or concentration at one site.
- [ ] **Check backup datasets.** Count labelled EuroCrops parcels by crop and look for independent field records to use alongside USDA CDL.
- [ ] **Plan the January 2027 Paoay visits.** Confirm which fields we can access, when to visit, and how to record each crop and its dates.
