# Classification dataset register and crop-label crosswalk

**Deliverable 1 · Draft for team review · 20 November 2026**

The application area is Barangay Paoay, Atok, Benguet, Philippines. The Atok Municipal Agricultural Office identified Paoay and Cattubo as major producers of carrot, cabbage and potato. This supports the choice of target crops. Transferability remains an empirical question because crop identity can be shared across sites while growing seasons, field size, terrain and clear imagery differ. [PIDS Atok study](https://pidswebs.pids.gov.ph/CDN/document/pidsrp2303.pdf)

## Dataset register

Counts below are independent parcel groups within each project split in the saved **matched** cohort. Both window recipes use these locations. Full-cohort counts differ. A declaration gives a reported crop and year. A survey date gives a field observation. Neither alone establishes occupancy throughout a four-month window. The rationales below explain the scientific role each source can support.

| Source and label evidence | Geography and years in matched data | Project split and parcel counts | Reason for inclusion and transfer limit |
|---|---|---|---|
| Sen4AgriNet, crop-coded parcel masks | Spain 2019–20, France 2019 | Training 25 carrot, 66 potato. Validation 6 carrot, 31 potato. | Provides two target crops under one labelling system across two countries. Within-source comparisons help reveal geographic dependence. European calendars and declared masks offer limited evidence about Paoay rotations and plot boundaries. |
| FranceCrops, declared crops joined to parcel geometry | France, 2022 | Training 50 cabbage, 11 potato. Validation 13 cabbage, 8 potato. | Supplies an explicit cabbage class alongside table potato. All matched training cabbage comes from this one source, so apparent cabbage discrimination can depend on source or geography. |
| EuroCrops, Dutch production declarations | Netherlands, 2021 | Training 47 carrot. Validation 11 carrot. | Adds a second source for carrot so recognition can be compared with Sen4AgriNet. Its parcels give annual evidence and are geographically concentrated. |
| CropHarvest JECAM, field observations | Madagascar 2019, Senegal 2021 | Primary external test. Madagascar 37 carrot, 3 cabbage, 35 potato. Senegal 8 cabbage. | Madagascar provides the closest documented farming-system comparison among the selected campaigns: tropical central highlands, small fields and observations of all three target crops. Differences in hemisphere and crop timing matter. Senegal contributes a second cabbage setting. Neither campaign measures Philippine accuracy. [JECAM site study](https://essd.copernicus.org/articles/13/5951/2021/index.html) |
| California Department of Water Resources, single-crop survey polygons | California, USA, 2023 | Supplemental test, 62 carrot. | Tests recognition of independently mapped carrots. A single-crop test provides little evidence about false carrot assignments in other local crops. |
| Denmark Marker, declared cabbage polygons | Denmark, 2023 | Supplemental test, 15 cabbage. | Broadens the cabbage check beyond FranceCrops development and the small JECAM cabbage cohort. White/red cabbage declarations warrant a local check against the intended Paoay cabbage definition. |
| CropHarvest non-JECAM cabbage campaign | Île-de-France, Martinique, Réunion, 2019 | Supplemental test, 5 cabbage: 3, 1 and 1 respectively. | Adds other sources of the original label *chou*. The small groups and annual date placeholders limit site-specific and seasonal interpretation. |
| CropHarvest non-JECAM carrot/potato campaign | Île-de-France and Réunion, 2019 | Supplemental test. Île-de-France 1 carrot and 9 potato. Réunion 6 potato. | Checks transfer across declared potato production types and locations. Only one matched carrot survives. Keep subtypes and sites visible. |
| Paoay reference polygons and associated crop/calendar notes | Paoay, Atok. Historical map window January–April 2024. Notes have their own evidence periods. | Local context and field planning. Seven reference areas. Zero independent crop observations recorded in the frozen field review. | Defines the target landscape and helps locate rotations, mixed areas and potential field visits. Local class counts and accuracy are still unknown. Count a verified field–crop cycle separately from a named reference area. |
| EuroCrops Dutch cabbage, retired campaign | Netherlands, 2021 | Attempted test supplement, excluded from active exports. | Retain the campaign record. The team reports insufficient clear-pixel support. Attach the batch quality evidence before publishing an exact failure count or attributing every exclusion to cloud. |

**Matched totals.** Training has 72 carrot, 50 cabbage and 77 potato. Locked validation has 17, 13 and 39 respectively. External testing has 100, 31 and 50 respectively, including JECAM. JECAM supplies the strongest highland comparison for carrot and potato, but only three of its cabbage parcels come from Madagascar. All 83 matched JECAM observations fall outside the fixed May–August window. The aligned window contains the observation date for 24 of 37 carrot, 6 of 11 cabbage and 32 of 35 potato parcels. Date overlap is useful evidence, though it cannot identify planting and harvest dates.

**Spatial distribution.** Based on matched training site-cell centres, selected Dutch, French and Spanish training sites are roughly 10,000–11,200 km from Paoay. JECAM field coordinates place the Madagascar and Senegal test sites about 9,000 and 14,200 km away. The approximate distances describe geographic separation. Quantitative similarity in elevation, field size, cloud cover and season still needs local measurements.

**Supporting spatial data.** Sentinel-2 Level-2A surface reflectance supplies the predictor imagery. Paoay boundaries define the analysis extent. Dynamic World and NAMRIA land-use/land-cover inputs provide context for local map review. These products supply no independent species labels. Record the imagery request, dates, quality masks and reference geometry version separately from each crop label. [Paoay workflow](paoay_workflow.md)

## Crop-label crosswalk

| Target | Source-native values used or retained by the active campaign | Interpretation to preserve |
|---|---|---|
| Carrot | Sen4AgriNet **231**. EuroCrops NL **2717, 2783, 2785**. JECAM **carrot**. California DWR **T6**, single-crop records. Non-JECAM CropHarvest **carotte**. | All map to carrot for scoring. Keep Dutch production code, original survey text and evidence date in the register. |
| Cabbage | FranceCrops **CHU**. JECAM **cabbage**. Denmark Marker **408/418**, the campaign's declared white/red cabbage code set. Non-JECAM CropHarvest **chou**. | All map to cabbage in the existing experiment. Retain generic cabbage wording whenever the source leaves the subtype unspecified. The saved Danish audit identifies the accepted code pair collectively. |
| Potato | Sen4AgriNet **510**. FranceCrops **PTC** (table potato). JECAM **potato**. Non-JECAM CropHarvest **pomme de terre de consommation** (consumption potato) and **pomme de terre féculière** (starch potato). | Both French use types are pooled for the three-class experiment. In the matched supplement, 12 are consumption potato and 3 starch potato. Preserve the original use type for interpreting transfer to local white potato. |

**Inactive mappings.** Sen4AgriNet recognizes cabbage code **213**, but it supplies no cabbage in the active development cohort. Dutch cabbage codes **2759, 2761, 2789** belong to the retired campaign. Keep these apart from active source–crop combinations and from observed Paoay labels.

## Decisions before sign-off

| Decision | Proposed handling |
|---|---|
| Which local species and subtypes count? | Confirm head cabbage and white potato terminology with local agricultural staff or growers. Record Chinese cabbage, pechay, broccoli, lettuce, sweet potato and other named crops verbatim. Default them outside the three targets unless the team documents a justified equivalence. Distinguish an actually observed other crop from an unidentified crop. |
| What establishes a usable local label? | Require location, actual date, observed crop, evidence source and spatial scope. Record planting/harvest or dated occupancy evidence where available. Agree on a primary cycle-matching rule and an ambiguous category before viewing local predictions. |
| What is one independent local sample? | Use one delineated field or homogeneous bed during a documented crop cycle. Retain farm and physical-field IDs across visits. Keep points as points if boundaries cannot be supported. Agree how to flag mixed plots, rotations, seedbeds and boundary pixels. |
| What comparisons can be claimed? | Retain JECAM as the previously examined external benchmark and the supplements as named diagnostics. Predeclare a separate local test. Collect comparable field-size, elevation, seasonal and clear-coverage summaries before claiming more specific agroecological similarity. |
| What evidence is outstanding? | Attach the Dutch cabbage quality receipt, verify source versions and original Danish code-level evidence for publication, and inventory any independent dated local observations held outside this repository. Keep historical 2024 and prospective field evidence separate. |

## Audit trail

The counts can be reproduced from the ignored working outputs in the project checkout. Preserve these with the run handoff.

- Development split, country, year and crop: `pipeline_outputs/development_error_review_2026-11-19/parcel_diagnostics.json`, fixed matched, geographic out-of-fold training and locked validation.
- External scope, label text, site, year and field identity: `pipeline_outputs/test_audit_2026-11-19/parcel_audit.json`, fixed matched. Cohort totals: `pipeline_outputs/preprocessing/external_xxu0iz4o/features/usable_samples.csv`.
- Mappings: [Sen4AgriNet](../src/crop_pipeline/sources/sen4agrinet.py), [FranceCrops](../src/crop_pipeline/sources/francecrops.py), [EuroCrops](../src/crop_pipeline/sources/eurocrops.py), [Denmark](../src/crop_pipeline/sources/denmark.py), and [collection documentation](collection.md).
- Local evidence: `data/pipeline_work/paoay/field_review/bundles/929cda2b0290abe9b68e/report.md`. Retired Dutch campaign role: `data/pipeline_work/retired_campaigns/ba5a6dfb78cd/policy.json`.
