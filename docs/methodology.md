# Data and processing rules

These rules explain what a usable sample means. The exact timing cutoffs are development choices; they have not been validated as crop-specific planting or harvesting rules.

## Reference sources

| Dataset | Labels used | Year |
|---|---|---:|
| [Sen4AgriNet](https://huggingface.co/datasets/orion-ai-lab/S4A) | Carrot and potato | 2020 |
| [FranceCrops](https://huggingface.co/datasets/saget-antoine/francecrops) | `CHU` cabbage labels from its training split | 2022 |

FranceCrops is pinned to revision `6b6c51b780d8579138593f92738ed52f2528096e`. Labels are joined to parcel boundaries by `parcel_id`; invalid boundaries are excluded. Its original validation/test parcels are not sampled. A 10 m inward buffer reduces boundary mixing.

Preserve source revisions and attribution. FranceCrops credits include Antoine Saget (CC BY 4.0), Copernicus Sentinel-2 and IGN RPG 2022. Follow Sen4AgriNet's source terms as well.

An annual crop label identifies the reported crop, but does not prove which crop occupied every detected cycle.

## Candidate sampling and imagery

Targets are **3,400 carrot, 7,000 cabbage and 10,000 potato candidates**, with at most **100 per source parcel group**. Sampling favours broad parcel coverage. Only selected sampling-mask pixels can enter training; surrounding pixels in downloaded images are excluded.

Both sources use Sentinel-2 surface reflectance from `COPERNICUS/S2_SR_HARMONIZED`:

- Mask unsuitable SCL classes, cloud probability at the configured cutoff of 40, and invalid scene edges. Missing cloud-probability matches are not treated as clear.
- Align B5/B11 to the 10 m grid using bilinear interpolation before compositing. Quality masks use nearest-neighbor resampling.
- Mosaic observations by date, then take the median across dates in each calendar half-month.
- Scale spectral values by 0.0001 before calculating indices. Do not scale observation counts.

Resampling does not create new spatial detail. Missing observations remain explicit.

## Four-month windows

Daily summaries retain actual observation dates for timing checks. Half-month median composites provide stable model inputs.

A supported rise in April gives an April–July window: April 1 through August 1, with the end excluded. Each window needs **eight half-month intervals**. Cross-year cycles are logged separately during collection; they do not discard valid in-year alternatives or trigger adjacent-year downloads.

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

Preserve the complete pool in `all_complete_features.csv`, then select up to **500 usable unique external pixels per crop** across both window policies, spreading selection across parcels. Reference-supported rows take precedence for duplicate locations. Registered local evaluation rows remain preserved. Report any shortfall instead of duplicating pixels.

The timing, cycle-selection and usable-selection audits explain losses at each stage. Budget trimming is separate from a quality failure. More candidates do not guarantee 500 usable pixels.

## Scope of the prototype

This is a three-class pipeline for known carrot, cabbage and potato parcels. General land-cover mapping needs additional classes. Source/year differences and nearby-parcel similarity can influence accuracy. Independent local evaluation is needed before making claims about Benguet.
