# Research basis and limits

The sources below motivate methods; they do not validate this project's exact
settings or establish accuracy on its data. See [project design](project_design.md)
for the current experiment and saved run receipts for measured results.

| Method or claim | Evidence | What this project can claim |
| --- | --- | --- |
| Random Forest on satellite time series | [Pelletier et al. (2016)](https://doi.org/10.1016/j.rse.2016.10.010) studies RF robustness for large-area land-cover mapping. | RF is an established comparison method; these crops and settings still require evaluation. |
| Keep parcel groups separate; assess geographic transfer | [Roberts et al. (2017)](https://doi.org/10.1111/ecog.02881) explains validation for structured observations. | Grouping and overlap screening reduce specific leakage risks. Nearby fields and source differences can still bias estimates. |
| Compare calendar and green-up timing | [Nyborg et al. (2022), TimeMatch](https://arxiv.org/abs/2111.02682) investigates temporal shifts across regions. | Timing shifts can matter. Our four-month alignment is a project experiment, not an implementation of TimeMatch or verified planting dates. |
| Sentinel-2 reflectance and resolution | [Earth Engine surface-reflectance catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) documents scaling and native bands. | Reflectance uses the documented scale; resampling B5/B11 to 10 m creates no new detail. |
| SAVI | [Huete (1988)](https://doi.org/10.1016/0034-4257(88)90106-X) introduces a soil-adjusted vegetation index. | SAVI has a published basis; its contribution here must be measured. |
| Dynamic World context | [Brown et al. (2022)](https://doi.org/10.1038/s41597-022-01307-4) describes nine-class land-cover probabilities. | Context and screening, not carrot/cabbage/potato labels or local ground truth. |

## Project choices to report explicitly

The 80% validity cutoff, eight complete half-month bins, four-month duration,
20-pixel parcel cap, fixed May–August period, green-up thresholds, sample budgets,
RF settings and F1 ≥ 0.50 reporting criteria are project choices. These papers do
not establish that those values are optimal. Keep them fixed when presenting a
run; record methodology changes separately.

Validation parcel macro-F1 selects the saved model settings. External testing
is reserved for evaluation. Report crop precision, recall, F1, parcel support,
coverage and the training-majority baseline. Missing classes make a three-class
aggregate unavailable. Full cohorts can differ in coverage; matched cohorts
hold locations constant. A test score is evidence about that sampled population,
not all agriculture or Paoay accuracy.

A spectral rise indicates estimated vegetation timing, not independently
verified sowing or harvest. Land-activity clusters and suitability rules are
exploratory project rules until independently validated. Historical outputs in
notebooks belong to their original runs; rerunning is required for current data.
