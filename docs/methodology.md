# Processing rules and limitations

## Reference sources

| Source | Labels used | Reference year |
|---|---|---|
| [Sen4AgriNet](https://huggingface.co/datasets/orion-ai-lab/S4A) | Carrot and potato parcel labels | 2020 |
| [FranceCrops](https://huggingface.co/datasets/saget-antoine/francecrops) | `CHU` cabbage labels from the training split | 2022 |

FranceCrops is pinned to revision `6b6c51b780d8579138593f92738ed52f2528096e`. The loader verifies the class map, joins labels to WGS84 parcel footprints by `parcel_id`, and excludes invalid footprints with an audit. Its validation and test parcels are not sampled. A 10 m inward buffer reduces boundary mixing.

Annual administrative crop declarations are reported labels. They do not independently establish which crop occupied every date in a selected four-month cycle. Satellite curves cannot confirm crop identity by themselves.

Attribute FranceCrops / Antoine Saget (CC BY 4.0), Copernicus Sentinel-2 and IGN RPG 2022 as described in the source release. Preserve Sen4AgriNet's dataset revision and source terms as well.

## Candidate sampling and imagery

Select **3,000 unique candidate pixels per crop**. Prefer 60 geographically spread parcel groups; add groups when small parcels limit capacity, with at most 100 candidates per group. Parcel groups are conservative identity groups, not automatically verified independent fields.

Both source notebooks use the same Sentinel-2 processing:

- Surface reflectance from `COPERNICUS/S2_SR_HARMONIZED` on fixed 10 m grids.
- SCL, exact cloud-probability matches with a cutoff of 40, and valid scene-edge masks applied before compositing.
- One mosaic per acquisition date, followed by a median over dates in each calendar half-month.
- Six stored spectral bands: B3, B4, B5, B8, B11 and B2, plus `clear_count`.
- Missing spectral values remain nodata; empty and failed intervals stay explicit.

Scale valid spectral values by 0.0001 before feature calculation. Never scale observation counts. Resampling originally 20 m bands onto a 10 m grid does not add spatial detail. Only pixels marked in the sampling masks belong to the pilot.

## Four-month windows

A window begins on the first day of the supported green-up month and includes the next three calendar months. An October start ends on February 1 of the following year, exclusive. The window requires **eight half-month bins**; missing adjacent-year imagery is not substituted with another year's data.

| Outcome | Meaning |
|---|---|
| Reference-supported | Recorded crop-cycle and timing evidence supports the selected window |
| Satellite-estimated | Timing is supported by algorithmic checks or an explicit human override; crop-cycle relevance remains unverified |
| Unresolved | Support is insufficient, competing cycles exist, or the onset month cannot be established |

Green-up is an estimated growth transition, not a verified planting date. A decline is not automatically a harvest date.

## Automatic timing assessment

Manual curve review is not required for exploratory estimates. The default rules require:

| Check | Development rule |
|---|---|
| Initial candidate | One supported actual-date cycle with onset inside one month |
| Observation coverage | Initially at least 80% valid selected pixels |
| Onset support | Actual observations bracket the NDVI threshold; onset gap at most 20 days |
| Nearby baseline | At least two usable observations in the preceding 30 days |
| Sustained growth | At least two later observations above threshold within 30 days; at least 75% of usable observations above threshold |
| Support gaps | At most 20 days between growth-confirming observations |
| NDRE agreement | Median post-onset increase of at least 0.02 |
| Threshold stability | Unique supported cycle at 15%, 20% and 25% relative amplitude, same onset month, peak dates within 15 days |

NDMI change is diagnostic, not a required increase. Passing estimates retain `timing_review_status=automated_pass`, `timing_validation=algorithmic_consistency_only` and `label_relevance=unverified`.

These cutoffs are development choices, not universal vegetable thresholds. Freeze them before final evaluation. Amplitude-based phenology and uncertainty flags have a methodological basis in [Copernicus phenology products](https://land.copernicus.eu/en/products/vegetation?tab=vegetation_phenology_and_productivity_parameters), but that does not validate this project's NDVI/NDRE rules for the target crops.

## Complete features and final selection

Each usable pixel needs five bands—B3, B4, B5, B8 and B11—and three indices—NDVI, NDRE and NDMI—in all eight bins: **64 finite features**. The band-only comparison uses 40 features. B2 and clear counts support inspection and quality checks; they are not classifier features.

Missing bins, low coverage, invalid counts and zero index denominators cause exclusion. The baseline does not fill or reconstruct missing observations.

After quality filtering:

1. Preserve every complete window in `all_complete_features.csv`.
2. Keep one window per physical pixel, preferring reference-supported windows for duplicate locations.
3. Select reproducibly across parcel groups, up to **1,000 external pixels per crop across both window policies together**.
4. Preserve registered local evaluation rows and record crop-level shortfalls in `usable_pixel_selection.csv`.

`complete_features.csv` contains selected rows. Quality-loss reports describe the pool before trimming; budget trimming is not a cloud-quality failure. Three thousand candidates do not guarantee one thousand usable pixels.

## Scope of the prototype

A three-class classifier assigns inputs to carrot, cabbage or potato. Evaluate it on parcels known to contain those crops. General land-cover mapping requires additional classes and labeled examples.

Different regions and years can introduce differences unrelated to crop identity. Independent local evaluation is necessary. Passing software tests or timing checks is not evidence of classification accuracy.
