# Incremental training data and annual TIFFs

The default collection is now `adaptive_training`, not the entire 18,012-tile FranceCrops catalogue. Each source notebook selects a bounded candidate parcel batch before creating masks or requesting satellite data. Full-year histories are retained so the four-month rule can be revised locally.

## Initial targets

Targets apply separately to carrot, cabbage and potato, **after** timing, eight-bin completeness and quality checks:

| Partition | Usable pixels | Minimum distinct source parcels |
| --- | ---: | ---: |
| Training | 1,200 | 60 |
| Validation | 400 | 20 |
| Reserved test | 400 | 20 |

At most 20 unique pixels are retained per parcel. These are initial collection budgets, not a proven minimum or a promise of model accuracy. Source parcels are proxies for independent fields. Geographic/year diversity matters; a large number of neighboring pixels from a few fields cannot meet the parcel target.

## Run the collection loop

1. Restart kernels and Run All in source notebooks **01–03**. Notebooks 01 and 03 select at most **25 new eligible parcels per available crop**. Notebook 02 has an overnight setting of **100 candidates per available crop per round**, including already selected candidates. FranceCrops has no carrots. References are spread across country/year/geographic strata using deterministic selection; only upstream `train` references are eligible.
2. Rerun each source notebook to collect its existing Earth Engine jobs until that candidate batch completes. Repeated runs before assessment use the same selection and checkpoints.
3. Run **04 Four-month windows**. It combines completed batches from all rounds, applies the existing timing and quality requirements, selects usable samples, and prints pixels/parcels by crop and locked partition. Partial source coverage is allowed for assessment; outstanding launched batches must finish first.
4. If targets are short, rerun source notebooks **01–03**, then **04** again. New candidate selection is limited to deficient crop/partition combinations and previously unselected physical parcel IDs. A source without candidates for the deficits is explicitly skipped. Failed timing/coverage is not repaired by relaxing thresholds or duplicating pixels.
5. Once notebook 04 reports `Targets met: True`, run **05–06**. Those notebooks block adaptive training until the initial targets are met and pass the locked split to exploratory model preparation.

No large full-catalogue collection starts by default. A candidate parcel may yield no usable pixels or may span more than one tile. Therefore a batch is measured in parcels and the stop condition is measured in usable pixels **and parcels**, rather than raw downloaded tile count.

Source reference discovery can still inspect the pinned upstream catalogue; the limited satellite collection does not imply that reference discovery is instant. Missing dependencies, source inputs and Earth Engine credentials still require the documented setup.

## Partitions and later experiments

A deterministic hash assigns each selected physical parcel to train/validation/test with a 60/20/20 allocation before usable-window assessment. The allocation is approximate for any small batch; later top-ups fill measured deficits. All fragments of the parcel stay together, repeated observation years are not treated as new independent parcels, and earlier assignments remain reserved. Upstream held-out references are not recycled into development data. Window threshold sensitivity checks exclude adaptive test parcels.

The feedback uses completeness and counts, not model scores. Use parcel-separated validation learning curves when deciding whether to increase the initial budget. Do not repeatedly use reserved-test performance to choose windows or decide which additional samples to collect. External-source evaluation does not establish performance in Benguet.

After changing window rules, rerun notebook 04 first: retained annual measurements can supply different four-month windows. Only then top up if the revised quality-passing yield is insufficient. Changing masks, input bands or compositing rules can require new source processing. Budgets are shared in `src/crop_pipeline/sources/adaptive.py`; changing them starts a separate policy-keyed campaign.

## Organized outputs

With the default local storage setting:

```text
sen4agrinet_work/
  adaptive_training/<campaign>/
    policy.json
    feedback.json
    locked_split.json
    rounds/0001/<source>/
      selection.json
      context.json
      state.json
      completed.json
  annual_archive/adaptive_training/<campaign>/round_0001/<source>/...
  annual_tifs/<source>/<year>/<campaign>/round_0001/
    <tile>_<year>.tif
    <tile>_<year>.json
```

Each annual TIFF contains **168 labelled float32 bands**: 24 half-month periods × six spectral bands plus clear-observation count. Descriptions use `<period>__<band>`. The CRS and transform preserve the saved sampling grid. Only the saved parcel selection is populated; surrounding cells are nodata. Reflectance stays unscaled until window auditing; counts are never scaled.

TIFFs are atomically published after read-back comparison and accompanied by a checksum receipt. Notebook 04 reads the organized annual TIFFs, and previews select the requested seven bands for each period. Parquet remains the compact export transport/cache and allows TIFF reconstruction without another satellite retrieval. Source sampling masks live separately under the versioned archive. `annual_archive_root` still controls archive storage; `annual_tifs` is a sibling folder on that drive.

Source preparation verifies its masks and regenerates missing masks. Deleting all TIFFs also removes old sampling/reference masks and legacy imagery, so earlier benchmark/legacy workflows cannot be treated as intact archives afterward. Labels, code, JSON, Parquet caches and model files are not TIFFs and are not removed by TIFF-only cleanup. No cloud tasks or assets are deleted by local cleanup.

## Larger overnight FranceCrops round

Notebook 02 exposes `candidate_parcels_per_crop = 100`, targeting up to 100 cabbage and 100 potato candidate parcels in the current round. Existing selections count toward the total: a running 25-per-crop batch is retained and resumed, and a separate `overnight_100` extension adds at most 75 new parcels per crop. A second Run All reuses the same two selections and their ledgers. Campaign identity, usable-sample targets and locked partitions do not change. Source 03 retains its 25-per-crop default; source 01 has its own larger overnight setting below.

Restart the kernel and Run All in the updated notebook. Keep the laptop on until it returns normally after preparing and submitting/collecting the active batches. Submitted jobs continue in Earth Engine overnight; downloading and TIFF creation happen when the notebook runs again. The larger workload does not raise Earth Engine concurrency or guarantee completion by morning. No new live jobs are submitted merely by editing this setting.

## Capped Sen4AgriNet reference scan and submission

Notebook 01 now uses `reference_cached_only = True`, `reference_scan_max_patches = 350`, and `reference_target_crops = ("carrot", "potato")`. It consolidates at most 350 valid saved patch checkpoints and makes no new reference patch transfers. Missing or corrupt checkpoints are ignored in cache-only mode; no usable checkpoints produces a clear error instead of an empty GEE batch. Existing complete inventories can also be reused. Cabbage is supplied by the completed FranceCrops collection and is excluded from new Sen4AgriNet candidate selection. All known reference features remain available for geometry exclusions.

`candidate_parcels_per_crop = 250` is a maximum for each of carrot and potato, not a prerequisite for submission. With the last reported counts, the expected selection is up to 250 potato and about 35 carrot parcels, subject to eligibility checks and existing selections. It is no longer a 750-parcel batch. After local reference consolidation, Run All continues to existing resumable job submission without waiting for more carrot or cabbage references. Completed FranceCrops output is combined with Sen4AgriNet by notebook 04 under the same adaptive campaign; it is not downloaded again by notebook 01.

The scan order remains a deterministic seeded round robin across year/tile strata. Subset inventories are configuration-keyed and record the crop targets, patch cap, cached-only flag, scanned filenames, actual counts, deficits and stop reason (`parcel_targets_reached`, `patch_cap_reached`, `cached_checkpoints_exhausted` or `catalogue_exhausted`). Physical parcel counts deduplicate observation years; mixed/conflicting scanned parcel groups and non-training references do not fill the budget. Unseen patches have not been audited for conflicting labels or neighboring geometry, and subset coverage is recorded explicitly.

To deliberately collect more references later, turn `reference_cached_only` off and choose a patch cap. It will stop at either the requested counts or the cap and submit the available candidates, rather than scanning the whole catalogue for a rare crop. The cap limits patches, not elapsed time. Repeated Run All reuses saved selections and ledgers; increasing collection within an already saved round requires a distinct extension budget/label or the next measured-deficit round. Existing locked partitions and imagery recipes are preserved.

Restart the kernel and Run All after loading the updated notebook. Keep the laptop on through preparation and submission. Once Run All returns normally, submitted cloud jobs continue while it is off; rerun later to collect results and create annual TIFFs. Neither the candidate count nor the patch cap guarantees eight hours of GEE processing. Editing notebook settings does not itself submit jobs.

### Valid imagery request outlines

Clipping and WGS84 conversion can leave collapsed edges, self-intersections or geometry collections in a tile's retrieval outline. The planner keeps valid polygonal outlines unchanged and uses a valid enclosing hull for problematic outlines; zero-area outlines are skipped. Sampling masks still use the original projected parcel intersection, so the enclosing request does not select surrounding pixels. The planner fingerprint includes this repair, preventing reuse of invalid prepared plans. Restart the kernel and Run All to regenerate the affected preparation from saved references; earlier archives are retained. The actual 338-tile Sen4AgriNet batch was checked offline: every request outline passes geometry/grid validation and every tile retains the same selected pixel centers as before repair.
