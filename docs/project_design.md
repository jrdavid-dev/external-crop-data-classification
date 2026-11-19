# Project design and deliverables

Read this first when resuming work on Project SAKAi. It records the agreed classification design and separates implementation from evidence still needed. Use saved run receipts and reports for current results.

## Scope

The active classification experiment distinguishes **carrot, cabbage, and potato**. It trains on external parcel data and tests on an independent external holdout. Paoay application is on hold for this experiment. The January 2027 field visit should record crops and dated observations that are available then. Keep those observations separate from interpretation of the preliminary January–April 2024 map.

## Classification deliverables

| Deliverable | Design and entry points | Evidence and status |
| --- | --- | --- |
| **1. Reference data and imagery collection** | [Data collection 01–08](../notebooks/data_collection/README.md) audits labels, collects source-specific annual Sentinel-2 imagery, and reports progress. Development uses Sen4AgriNet (mostly potato, some carrot), FranceCrops (potato and cabbage), and EuroCrops (carrot). Primary external testing uses CropHarvest JECAM for all three crops. | Collection and reporting are implemented. Save source, crop, year, parcel identity, batch state and TIFF provenance. Collection counts are candidate counts. |
| **2. Imagery quality, windows, and features** | [Preprocessing 01–03](../notebooks/preprocessing/README.md) checks imagery and spatial independence, builds four-month windows, and exports separate development and test features. | Implemented. The feature-stage receipt and `usable_samples.csv` record the checked export and quota shortfalls. |
| **3. Classification and external evaluation** | [Training 01–02 and testing 01–02](workflow.md#training-and-external-testing). Train and select on development only, freeze fitted models, then score the matching JECAM and supplemental test features. | Implemented and verified with test fixtures. The project evidence is a completed run on the selected data, with validation results, frozen recipes, external predictions and diagnostics. |

In implementation terms, **batch 1** organizes collection, **batch 2** organizes preprocessing, **batch 3** trains and freezes models, and **batch 4** evaluates them. These batch numbers refer to implementation work, independently of the batch slots used to collect imagery.

## What we compare

- **Fixed window:** May–August for each observation year (May 1 through September 1, end exclusive). **Aligned window:** four months from the selected green-up start. Both use eight half-month bins.
- **Full cohorts:** each recipe uses its own qualifying locations. **Matched cohorts:** both recipes use the same qualifying locations, allowing a comparison with sample locations held constant. Keep coverage and results distinct.
- Both recipes use the ordered **72 interval predictors** from five Sentinel-2 bands (B3, B4, B5, B8, B11) and four indices (NDVI, NDRE, NDMI, SAVI). Use the saved model recipe for exact scaling, feature order, and quality rules. The older 48- and 120-predictor experiments are historical. [Methods](methodology.md)
- Use the existing Random Forest settings and weighting. Select minimum leaf size on **validation parcel macro-F1** for each full cohort (2 or 4); matched cohorts use leaf size 2. Testing averages pixel probabilities per parcel. The baseline predicts the most common training parcel crop.
- Report macro-F1, crop precision/recall/F1, parcel counts, coverage, and confusion matrices. The report flags macro-F1 ≥ 0.50, strictly above the baseline, and F1 ≥ 0.50 for every crop. Incomplete crop coverage makes the three-class aggregate unavailable. External scores describe this external holdout.

## Invariants and remaining work

- Preserve locked parcel assignments, collection ledgers, TIFFs, and source provenance. Screen test footprints against development. Exclude pixel locations shared by different development parcel groups within each recipe before matching and capping, including repeated farmland across years.
- Keep the 80% validity requirement, eight-bin completeness, finite features, and 20-pixel parcel cap. Targets per crop are **60 parcels / 1,200 pixels** in training and **20 parcels / 400 pixels** each in validation and testing. Assess targets per recipe and split. Pilot runs can report shortfalls; reaching the targets does not establish accuracy.
- Report primary JECAM, each supplemental campaign and the combined external pool. Screen supplements against development and primary test footprints before scoring.
- Before the final experiment, check selected top-up completion, rerun preprocessing after collection changes, and inspect current counts and receipts before training. Record any remaining quota gaps and geographic concentration in the final evaluation.
- **Broader project deliverables:** crop suitability (AHP/WLC, limiting factors, preliminary S1/S2/S3/N maps), data limitations and validation planning, and eventual paper submission are tracked separately from this classification implementation. [Paoay field checks](paoay_workflow.md#field-observations)

## Deferred Paoay experiments

Keep partial-coverage mapping and field inference separate from the preliminary baseline. If resumed, report observed and missing area separately, and test any field-level inference on labelled fields with simulated missing coverage before applying it locally. Other-crop classes and hard noncrop exclusions require their own validation. Preserve the preliminary map and record correction proposals separately. [Paoay workflow](paoay_workflow.md)
