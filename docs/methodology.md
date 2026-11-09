# Data and processing rules

These rules explain what a usable sample means. The exact timing cutoffs are development choices; they have not been validated as crop-specific planting or harvesting rules.

## Reference sources

| Dataset | Labels used | Year |
|---|---|---:|
| [Sen4AgriNet](https://huggingface.co/datasets/orion-ai-lab/S4A) | All indexed target-crop parcels | Indexed observation years |
| [FranceCrops](https://huggingface.co/datasets/saget-antoine/francecrops) | `CHU` cabbage and `PTC` table-potato parcels | 2022 |

FranceCrops is pinned to revision `6b6c51b780d8579138593f92738ed52f2528096e`. Labels are joined to parcel boundaries by `parcel_id`; invalid boundaries are excluded. Its original validation/test parcels can be archived but remain outside model-development exports. A 10 m inward buffer reduces boundary mixing.

Preserve source revisions and attribution. FranceCrops credits include Antoine Saget (CC BY 4.0), Copernicus Sentinel-2 and IGN RPG 2022. Follow Sen4AgriNet's source terms as well.

An annual crop label identifies the reported crop, but does not prove which crop occupied every detected cycle.

## Candidate sampling and imagery

The default adaptive workflow selects up to 100 candidate parcels per available crop per source per round, then retains at most 20 usable pixels per parcel after window assessment. The optional full-catalogue workflow uses every eligible parcel and interior pixel in supplied inventories. Same-parcel fragments share one group. Contested locations are excluded, and bounded tiles preserve the original parcel identity. FranceCrops and CropHarvest retain a 10 m inward buffer; Sen4AgriNet retains its existing declared crop-mask cells with no additional inward buffer. This is an existing source-specific rule, not evidence of equivalent boundary uncertainty; adding erosion would change selected pixels and require a separately versioned methodology change.

All three sources use Sentinel-2 surface reflectance from `COPERNICUS/S2_SR_HARMONIZED`:

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

For adaptive collection, deduplicate usable locations and apply the per-parcel sampling budget and locked partitions. The optional full-catalogue workflow retains every unique usable external pixel across both policies. Reference-supported rows take precedence for duplicate locations. Preserve registered local evaluation rows and upstream reserved roles. The old 500-pixel pilot cap applies only when deliberately reopening a pilot collection.

Timing, cycle-selection and usable-selection audits explain losses. Source collection and window selection are separate; changing the latter can reuse the annual archive. See [full-year collection](full_year_collection.md).

## Study scope

This is a three-class pipeline for known carrot, cabbage and potato parcels. General land-cover mapping needs additional classes. Source/year differences and nearby-parcel similarity can influence accuracy. Independent local evaluation is needed before making claims about Benguet.

## Saved preprocessing examples

Section 15 of the window notebook displays one deterministic example per crop using saved files only: reference/sample locations, post-mask pixel eligibility, daily timelines with the selected four-month window, and eight half-month composites. PNGs and the example-selection manifest are saved in the selected analysis folder’s `preprocessing_visuals/`. Original unmasked scenes are not stored, so the screening view is not a before/after cloud-removal comparison.
