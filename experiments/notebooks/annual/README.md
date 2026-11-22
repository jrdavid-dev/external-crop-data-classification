# Annual harmonic recipe

An experiment to improve on the window models, which remain the reported
baseline. Whole-year Sentinel-2 harmonic predictors, with Sentinel-1 radar
sampled at the same pixels, need no four-month window or green-up date
(at least 12 clear half-months with no gap over three months). Code lives in
[`experiments/annual/`](../../annual/).

Run in order, each in a fresh kernel, after Preprocessing 01 and the window
models' Training 01–02 and Testing 01:

1. [`01_annual_recipe.ipynb`](01_annual_recipe.ipynb) builds the inputs from the
   verified Preprocessing 01 quality handoff (up to 20 pixels per parcel). It keeps its
   own train/validation split: the lock each parcel group got at collection (seed 42),
   covering every development parcel. The window models instead re-split, in
   Preprocessing 03, only the parcels with a qualifying window, so the two families'
   validation parcels differ. It selects
   on development views only and freezes. Sentinel-2 only and Sentinel-2 plus
   radar are scored with the default Random Forest; the higher selection score
   (mean of spatial CV within the training split and dataset-holdout macro-F1)
   wins. The chosen recipe is frozen on the training split with the main
   pipeline's forest settings and training weights.
2. [`02_annual_external_test.ipynb`](02_annual_external_test.ipynb) scores the
   frozen model once on JECAM, each supplement and the combined pool, then
   compares it with each window model on the parcels that model scored, using the
   parcel bootstrap of the external report. The external parcels were already
   scored for the window models, so this comparison is retrospective. The same
   comparison from the command line:

   ```bash
   .venv/bin/python -m experiments.annual.paired_comparison \
       pipeline_outputs/annual/test/run_<id> annual pipeline_outputs/testing/run_<id> \
       pipeline_outputs/reports/<new_folder> [--revised-scope]
   ```

3. [`03_annual_maps.ipynb`](03_annual_maps.ipynb) applies the frozen model to
   Paoay (cropland from the Paoay 06 land-activity map) and the wider Atok
   boundary (ESA WorldCover 2021 cropland).

**Needs:** Earth Engine access (`earth_engine_project` in `settings.local.json`)
the first time: 01 samples Sentinel-1 at the selected pixels (about 260 small
requests) and 03 fetches the 2024 Sentinel-1 and WorldCover grids. Everything is
cached under `pipeline_outputs/annual/cache/`; reruns make no requests, and no
exports or cloud assets are created. 03 also needs the Paoay KMZ files,
`atok_imagery_root` composites and the Paoay 06 land-activity map.

Each run writes new `pipeline_outputs/annual/<stage>/run_<id>/` folders with
sealed receipts and updates the `current_*` pointers there; earlier runs stay and
the window runs are never changed. Runtime on a laptop: about 5 minutes for 01
(plus the first Sentinel-1 sampling), a few minutes for the other two.

The green-up, curve-shape, other-classifier and short-cycle experiments build on
this recipe; their results are in the [experiments index](../../README.md).

To compare the annual experiment with the window models and Jose's comparison variants
(Testing 03 and Training 03 outputs) on the parcels they all predicted:

```bash
.venv/bin/python -m experiments.annual.compare_models pipeline_outputs/testing/run_<id> \
    pipeline_outputs/annual/test/run_<id> pipeline_outputs/reports/<new_folder>
```
