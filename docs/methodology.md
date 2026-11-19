# Data and processing rules

For the current experiment, start with [project design](project_design.md) and
[research basis and limits](research_basis.md). The classifier uses **72 interval
predictors**: five bands plus NDVI, NDRE, NDMI and SAVI in eight bins.
The 64-feature description below refers to the base window-completeness stage;
older feature experiments are historical.

These rules explain what a usable sample means. The exact timing cutoffs are development choices; they have not been validated as crop-specific planting or harvesting rules.

## Reference sources

| Dataset/cohort | Role and crop labels | Observation years |
| --- | --- | --- |
| Sen4AgriNet | Development potato/carrot from bounded reference scans | Recorded source years |
| FranceCrops | Development `CHU` cabbage and `PTC` table potato | 2022 |
| EuroCrops | Additional development carrot, local Dutch declarations | 2021 |
| CropHarvest JECAM | Primary external test for carrot/cabbage/potato | Resolved label years |
| Non-JECAM CropHarvest | Separate carrot/potato and cabbage test supplements | Resolved label years |
| California DWR | Supplemental T6 single-crop carrot pilot | 2023 |
| Denmark Marker | Supplemental declared white/red cabbage (408/418) | 2023 |

FranceCrops is pinned to revision `6b6c51b780d8579138593f92738ed52f2528096e`. Labels are joined to parcel boundaries by `parcel_id`; invalid boundaries are excluded. Its original validation/test parcels can be archived but remain outside model-development exports. A 10 m inward buffer reduces boundary mixing.

Preserve source revisions and attribution. FranceCrops credits include Antoine Saget (CC BY 4.0), Copernicus Sentinel-2 and IGN RPG 2022. Follow Sen4AgriNet's source terms as well.

An annual crop label identifies the reported crop, but does not prove which crop occupied every detected cycle.

## Candidate sampling and imagery

Collection uses bounded, source-specific parcel batches. Batch limits, crop filters and extra slots are workload settings in the notebooks; they are not one universal 50-parcel rule. Retain at most 20 usable pixels per parcel after window assessment. The optional full-catalogue workflow uses every eligible parcel and interior pixel in supplied inventories. Same-parcel fragments share one group. Contested locations are excluded, and bounded tiles preserve the original parcel identity. FranceCrops and CropHarvest retain a 10 m inward buffer; Sen4AgriNet retains its existing declared crop-mask cells with no additional inward buffer. This is an existing source-specific rule, not evidence of equivalent boundary uncertainty; adding erosion would change selected pixels and require a separately versioned methodology change.

The external imagery collection uses Sentinel-2 surface reflectance from `COPERNICUS/S2_SR_HARMONIZED`:

- Mask unsuitable SCL classes, cloud probability at the configured cutoff of 40, and invalid scene edges. Missing cloud-probability matches are not treated as clear.
- Align B5/B11 to the 10 m grid using bilinear interpolation before compositing. Quality masks use nearest-neighbor resampling.
- Mosaic observations by date, then take the median across dates in each calendar half-month.
- Scale spectral values by 0.0001 before calculating indices. Do not scale observation counts.

Resampling does not create new spatial detail. Missing observations remain explicit.

## Four-month windows

Daily summaries retain actual observation dates for timing checks. Half-month median composites provide stable model inputs.

A supported rise in April gives an April–July window: April 1 through August 1, with the end excluded. Each window needs **eight half-month intervals**. Timing is evaluated after full-year collection. Cross-year windows remain incomplete without adjacent-year imagery; valid in-year alternatives are still eligible.

| Outcome | Meaning |
|---|---|
| Reference-supported | Specific crop-cycle and timing evidence supports the window |
| Satellite-estimated | The timing passes automatic checks or an explicit review override; crop-cycle relevance remains unverified |
| Unresolved | The available observations do not support a stable timing estimate |

A vegetation rise is not a verified planting date; a decline is not a verified harvest date.

## Automatic timing assessment

Each cycle is checked separately. Exploratory timing does not require manual curve review.

| Check | Current rule |
|---|---|
| Observation coverage | At least 80% valid selected pixels for usable observations |
| Onset | Observations bracket the threshold within 20 days and one month |
| Baseline | At least two usable observations in the preceding 30 days |
| Growth support | At least two later observations above threshold within 30 days; at least 75% of usable confirmation observations above threshold |
| Support gaps | At most 20 days between growth-confirming observations |
| NDRE agreement | Median post-onset increase of at least 0.02 |
| Sensitivity | Match the same cycle at 15%, 20% and 25% amplitude; stable onset month and peaks within 15 days |

NDMI change is diagnostic, not a required increase. Passing timing checks establishes algorithmic consistency, not independently validated crop phenology. Amplitude-based detection has precedent in [Copernicus phenology methods](https://land.copernicus.eu/en/products/vegetation?tab=vegetation_phenology_and_productivity_parameters); that does not validate these exact cutoffs.

## Complete features and final selection

For each parcel, evaluate qualified windows chronologically. Keep the **earliest window with at least one complete usable pixel**; try the next if the earlier window has none. All fragments share that date range. Do not combine cycles to fill gaps or select the window with the most pixels.

A complete pixel needs five bands—B3, B4, B5, B8, B11—and NDVI, NDRE and NDMI in every interval: **64 finite base features**. Missing bins, low coverage, invalid observation counts and zero index denominators cause exclusion. No imputation is used.

For adaptive collection, deduplicate usable locations and apply the per-parcel sampling budget and locked partitions. Historical full-catalogue experiments used different selection budgets; those are not the active capped export. Reference-supported rows take precedence for duplicate locations. Preserve registered local evaluation rows and upstream reserved roles. Legacy pilot budgets are historical and do not define a separate mode in the active notebooks.

Timing, cycle-selection and usable-selection audits explain losses. Source collection and window selection are separate; changing the latter can reuse the annual archive. See [collection](collection.md) and the [transport reference](reference/transport.md).

## Study scope

This is a three-class pipeline for known carrot, cabbage and potato parcels. General land-cover mapping needs additional classes. Source/year differences and nearby-parcel similarity can influence accuracy. Independent local evaluation is needed before making claims about Benguet.

## Observation loss accounting

New external selected-pixel exports record the number of dates with B3 support in matched granules before screening, clear dates, cloud or shadow rejection, and other rejection. A date with any surviving granule counts as clear. If none survives, a cloud or shadow flag on a supported granule takes precedence over other reasons. Granule matches and acquisition dates are reported separately from pixel-date counts. Legacy files can still be read and show unavailable diagnostics as null. New Paoay downloads use the same daily screening definitions. Their receipts store totals over ROI pixel-dates and the fraction rejected by cloud or shadow. The seven base archive measurements stay compatible with saved rasters; classifier predictors remain the five bands and four indices listed above. Previously downloaded intervals retain acquisition matching and usable coverage, and their cloud and shadow loss remains unavailable.

## Classifier features

Each of eight bins contributes B3, B4, B5, B8, B11 and four indices: 72 ordered
predictors. The 64-value base completeness stage precedes the eight SAVI values.
Use scaled reflectance for these formulas and the saved recipe for feature order.

| Index | Formula |
| --- | --- |
| NDVI | `(B8 − B4) / (B8 + B4)` |
| NDRE | `(B8 − B5) / (B8 + B5)` |
| NDMI | `(B8 − B11) / (B8 + B11)` |
| SAVI | `(1 + L) × (B8 − B4) / (B8 + B4 + L)`, with saved `L = 0.5` |

Missing/nonfinite inputs and invalid denominators are not imputed. Research
attribution and the limits of these choices are in [research basis](research_basis.md).

## Model selection and scoring

Random Forest settings are 1,000 trees, seed 42, square-root feature sampling and
50% bootstrap sampling. Full cohorts choose minimum leaf size 2 or 4 on validation
parcel macro-F1; matched cohorts retain leaf size 2. Training weights give crops
equal total weight, then parcels equal shares within crop; confidence distributes
a parcel's share across its pixels (confirmed 1.0, reported 0.5 before normalization).

Both window recipes use the ordered 72 interval predictors. The older 48-shape
and 120-combined recipes are retained as [historical feature context](history/feature_experiments.md),
not alternative designs selected by the current notebooks. Fixed windows are
May–August; aligned windows follow the saved green-up recipe. Old exports lacking
fixed-start metadata retain their January-start interpretation and must be paired
with matching model/input recipes.

Test scoring averages pixel probabilities within each parcel. The baseline
predicts the most common training parcel crop; ties follow first occurrence in
the saved training rows. Evaluate it on the same test parcels as each model.
Report precision, recall, F1, parcel support, coverage and confusion matrices;
missing test crops suppress three-class macro-F1 and balanced accuracy.

The reporting criteria require macro-F1 ≥ 0.50, strictly above the baseline, and
F1 ≥ 0.50 for every crop. They are project criteria, not proof of usefulness.
Report JECAM, each supplement and the combined pool separately. Validation selects
settings; external scores never tune models, windows or collection. Local Paoay
accuracy still requires independent dated evidence.
