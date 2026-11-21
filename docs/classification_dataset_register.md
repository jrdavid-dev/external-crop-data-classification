# Classification dataset register and crop-label crosswalk

**External classification study · Final run register · 21 November 2026**

The three target crops are carrot, cabbage and potato. They are documented in Paoay and Cattubo, Atok, Benguet. This register covers the frozen external study. Paoay application is deferred. [PIDS Atok study](https://pidswebs.pids.gov.ph/CDN/document/pidsrp2303.pdf)

## Dataset register

Counts below are distinct reference parcel groups within each project split in the saved **matched** cohort. Parcel groups are the unit of scoring and resampling. Both window recipes use these locations. Full-cohort counts differ. A declaration gives a reported crop and year. A survey date gives a field observation. Neither alone establishes occupancy throughout a four-month window. The rationales below explain the scientific role each source can support.

| Source and label evidence | Geography and years in matched data | Project split and parcel counts | Reason for inclusion and transfer limit |
|---|---|---|---|
| Sen4AgriNet, crop-coded parcel masks | Spain 2019–20, France 2019 | Training 25 carrot, 66 potato. Validation 6 carrot, 31 potato. | Provides two target crops under one labelling system across two countries. Within-source comparisons help reveal geographic dependence. European calendars and declared masks offer limited evidence about Paoay rotations and plot boundaries. |
| FranceCrops, declared crops joined to parcel geometry | France, 2022 | Training 50 cabbage, 11 potato. Validation 13 cabbage, 8 potato. | Supplies an explicit cabbage class alongside table potato. All matched training cabbage comes from this one source, so apparent cabbage discrimination can depend on source or geography. |
| EuroCrops, Dutch production declarations | Netherlands, 2021 | Training 47 carrot. Validation 11 carrot. | Adds a second source for carrot so recognition can be compared with Sen4AgriNet. Its parcels give annual evidence and are geographically concentrated. |
| CropHarvest JECAM, field observations | Madagascar 2019, Senegal 2021 | Primary external test. Madagascar 37 carrot, 3 cabbage, 35 potato. Senegal 8 cabbage. | Madagascar provides the closest documented farming-system comparison among the selected campaigns: tropical central highlands, small fields and observations of all three target crops. Differences in hemisphere and crop timing matter. Senegal contributes a second cabbage setting. Neither campaign measures Philippine accuracy. [JECAM site study](https://essd.copernicus.org/articles/13/5951/2021/index.html) |
| California Department of Water Resources, single-crop survey polygons | California, USA, 2023 | Supplemental test, 62 carrot. | Tests recognition of independently mapped carrots. A single-crop test provides little evidence about false carrot assignments in other local crops. |
| Denmark Marker, declared cabbage polygons | Denmark, 2023 | Supplemental test, 15 cabbage. | Broadens the cabbage check beyond FranceCrops development and the small JECAM cabbage cohort. White/red cabbage declarations warrant a local check against the intended Paoay cabbage definition. |
| CropHarvest non-JECAM cabbage campaign | Île-de-France, Martinique, Réunion, 2019 | Archived original-scope supplement. Five cabbage parcels, with 3, 1 and 1 respectively. | The revised reported scope excludes this campaign after score review. The frozen predictions and original-scope results retain it. |
| CropHarvest non-JECAM carrot/potato campaign | Île-de-France and Réunion, 2019 | Reported potato supplement. Île-de-France 9 potato and Réunion 6 potato. The original scope also contains 1 Île-de-France carrot. | The revised scope retains the potato reference rows and excludes carrot reference rows from this campaign. Potato use types and both sites remain identifiable. |
| Paoay reference polygons and associated crop/calendar notes | Paoay, Atok. Historical map window January–April 2024. Notes have their own evidence periods. | Deferred application context. Seven reference areas. Zero independent crop observations recorded in the frozen field review. | Defines the target landscape and helps locate rotations, mixed areas and potential field visits. Local class counts and accuracy are still unknown. Count a verified field–crop cycle separately from a named reference area. |
| EuroCrops Dutch cabbage, retired campaign | Netherlands, 2021 | Attempted test supplement, excluded from active exports. | Retain the campaign record. The team reports insufficient clear-pixel support. Attach the batch quality evidence before publishing an exact failure count or attributing every exclusion to cloud. |

### Reported and archived test populations

The primary test is JECAM. Results then show California carrot, Denmark cabbage and non-JECAM potato separately, followed by the combined pool, which includes JECAM once per model. This was a post hoc reporting revision after reviewing scores. The original full-scope evaluation retains the non-JECAM cabbage campaign and non-JECAM carrot reference rows. Frozen splits, imagery gates and model predictions remain available in the original archive.

| Matched split | Carrot parcels / pixels | Cabbage parcels / pixels | Potato parcels / pixels | Total parcels / pixels |
|---|---:|---:|---:|---:|
| Development training | 72 / 1,223 | 50 / 961 | 77 / 1,127 | 199 / 3,311 |
| Locked development validation | 17 / 316 | 13 / 260 | 39 / 552 | 69 / 1,128 |
| Primary JECAM | 37 / 115 | 11 / 93 | 35 / 137 | 83 / 345 |
| California carrot | 62 / 1,218 | — | — | 62 / 1,218 |
| Denmark cabbage | — | 15 / 264 | — | 15 / 264 |
| Non-JECAM potato | — | — | 15 / 229 | 15 / 229 |
| Revised combined test | 99 / 1,333 | 26 / 357 | 50 / 366 | 175 / 2,056 |
| Archived original combined test | 100 / 1,353 | 31 / 405 | 50 / 366 | 181 / 2,124 |

The full models have different eligible parcels. Fixed-full and aligned-full have 148 and 89 JECAM parcels, respectively, and 341 and 186 revised combined parcels. The test target is 20 parcels and 400 usable pixels per crop. All JECAM models fall short of the pixel target for each crop. For matched JECAM, cabbage also falls short of the parcel target. In the revised combined pool, matched cabbage and potato pass the parcel target and fall short of the pixel target. [Coverage and scores](classification_external_study.md#results)

JECAM supplies the closest documented highland comparison through its Madagascar site. All 83 paired JECAM observation dates fall outside the fixed May–August window. The aligned window contains the date for 24 of 37 carrot, 6 of 11 cabbage and 32 of 35 potato parcels. Date overlap provides timing context. Annual declarations and survey dates seldom establish the entire crop cycle.

**Spatial distribution.** Based on matched training site-cell centres, selected Dutch, French and Spanish training sites are roughly 10,000–11,200 km from Paoay. JECAM field coordinates place the Madagascar and Senegal test sites about 9,000 and 14,200 km away. The approximate distances describe geographic separation. Quantitative similarity in elevation, field size, cloud cover and season still needs local measurements.

**Supporting spatial data.** Sentinel-2 Level-2A surface reflectance supplies the predictor imagery. Paoay boundaries define the analysis extent. Dynamic World and NAMRIA land-use/land-cover inputs provide context for local map review. These products supply no independent species labels. Record the imagery request, dates, quality masks and reference geometry version separately from each crop label. [Paoay workflow](paoay_workflow.md)

## Crop-label crosswalk

| Target | Source-native values in the reported scope or original archive | Interpretation to preserve |
|---|---|---|
| Carrot | Sen4AgriNet **231**. EuroCrops NL **2717, 2783, 2785**. JECAM **carrot**. California DWR **T6**, single-crop records. Archived non-JECAM CropHarvest **carotte**. | All map to carrot for scoring. Keep Dutch production code, original survey text and evidence date in the register. |
| Cabbage | FranceCrops **CHU**. JECAM **cabbage**. Denmark Marker **408/418**, the campaign's declared white/red cabbage code set. Archived non-JECAM CropHarvest **chou**. | All map to cabbage in the existing experiment. Retain generic cabbage wording whenever the source leaves the subtype unspecified. The saved Danish audit identifies the accepted code pair collectively. |
| Potato | Sen4AgriNet **510**. FranceCrops **PTC** (table potato). JECAM **potato**. Non-JECAM CropHarvest **pomme de terre de consommation** (consumption potato) and **pomme de terre féculière** (starch potato). | Both French use types are pooled for the three-class experiment. In the matched supplement, 12 are consumption potato and 3 starch potato. Preserve the original use type for interpreting transfer to local white potato. |

**Inactive mappings.** Sen4AgriNet recognizes cabbage code **213**, but it supplies no cabbage in the active development cohort. Dutch cabbage codes **2759, 2761, 2789** belong to the retired campaign. Keep these apart from active source–crop combinations and from observed Paoay labels.

## Deferred local application decisions and publication checks

| Decision | Proposed handling |
|---|---|
| Which local species and subtypes count? | Confirm head cabbage and white potato terminology with local agricultural staff or growers. Record Chinese cabbage, pechay, broccoli, lettuce, sweet potato and other named crops verbatim. Default them outside the three targets unless the team documents a justified equivalence. Distinguish an actually observed other crop from an unidentified crop. |
| What establishes a usable local label? | Require location, actual date, observed crop, evidence source and spatial scope. Record planting/harvest or dated occupancy evidence where available. Agree on a primary cycle-matching rule and an ambiguous category before viewing local predictions. |
| What is one independent local sample? | Use one delineated field or homogeneous bed during a documented crop cycle. Retain farm and physical-field IDs across visits. Keep points as points if boundaries cannot be supported. Agree how to flag mixed plots, rotations, seedbeds and boundary pixels. |
| What comparisons can be claimed? | Use the reported scope above and retain the full original results. Predeclare a separate local test if application resumes. Collect comparable field-size, elevation, seasonal and clear-coverage summaries before claiming more specific agroecological similarity. |
| What evidence is outstanding? | Attach the Dutch cabbage quality receipt, verify source versions and original Danish code-level evidence for publication, and inventory any independent dated local observations held outside this repository. Keep historical 2024 and prospective field evidence separate. |

## Audit trail

The counts can be reproduced from the ignored working outputs in the project checkout. Preserve these with the run handoff.

- Development split, country, year and crop: `pipeline_outputs/development_error_review_2026-11-19/parcel_diagnostics.json`, fixed matched, geographic out-of-fold training and locked validation.
- External scope, label text, site, year and field identity: `pipeline_outputs/test_audit_2026-11-19/parcel_audit.json`, fixed matched. Original cohort totals: `pipeline_outputs/preprocessing/external_xxu0iz4o/features/usable_samples.csv`. Revised test totals: `pipeline_outputs/testing/run_81a0d0a2812e_batch1_revised/parcel_metrics.csv`.
- Mappings: [Sen4AgriNet](../src/crop_pipeline/sources/sen4agrinet.py), [FranceCrops](../src/crop_pipeline/sources/francecrops.py), [EuroCrops](../src/crop_pipeline/sources/eurocrops.py), [Denmark](../src/crop_pipeline/sources/denmark.py), and [collection documentation](collection.md).
- Local evidence: `data/pipeline_work/paoay/field_review/bundles/929cda2b0290abe9b68e/report.md`. Retired Dutch campaign role: `data/pipeline_work/retired_campaigns/ba5a6dfb78cd/policy.json`.

- Exact saved data and archive revisions are recorded per source handoff in `pipeline_outputs/preprocessing/external_xxu0iz4o/quality/source_inputs.json`; the collection snapshot and source-input ledger are checksummed in the [external run manifest](../pipeline_outputs/testing/run_81a0d0a2812e_batch4/study_manifest.json).
