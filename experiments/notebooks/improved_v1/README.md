# Improved pipeline

A separate pipeline that keeps the main pipeline's safeguards and adds the features
that helped in the tuning, transfer and pilot experiments. Code:
`experiments/improved_v1/workflow.py`. Outputs: new folders under
`pipeline_outputs/improved/<stage>/`, each sealed with a receipt and published as
`pipeline_outputs/improved/current_<stage>.json`. The main pipeline, its pointers and
its frozen models are not changed.

Run in order, each in a fresh kernel, after Preprocessing 01:

1. `01_inputs.ipynb` — every development and test parcel (JECAM and supplements) from the verified
   quality handoff (locked partitions), 5 pixels per parcel by default, annual
   24 half-month Sentinel-2 series, and Sentinel-1 VV/VH at the same pixels (direct
   Earth Engine requests, no export tasks; cached under `pipeline_outputs/improved/cache/`).
2. `02_model_selection.ipynb` — four candidates (Sentinel-2 harmonics; plus Sentinel-1;
   with omega 1.5; omega 1.5 plus Sentinel-1) scored on development data by random split,
   spatial CV and dataset holdout. The highest mean of spatial CV and dataset holdout wins.
3. `03_freeze_and_test.ipynb` — the selected candidate is refitted on train + validation
   with 1,000 trees, frozen, and scored once on every collected test population: primary
   JECAM, each supplemental campaign (non-JECAM CropHarvest cabbage and carrot/potato,
   California DWR carrot, Denmark Marker cabbage) and the combined pool, each with the
   training-majority baseline and the F1 criteria. Single-crop populations report
   present-crop F1 only.
4. `04_paoay_map.ipynb` — the frozen model is applied to all Paoay study pixels (most
   complete Cloud Score+ year), with the 0.60/0.10 confidence rule and the land-activity
   cropland flag. Outputs `crop_map.tif`, `probabilities.tif` and area tables.
5. `05_field_visit_plan.ipynb` — January field-visit plan from the current Paoay map:
   stratified random points by map class inside land-activity cropland (≥ 50 m apart, one
   reserve per primary point) plus points in every KMZ reference area. Writes
   `visit_points.csv`, `.kml` (Google Earth / phone), `.geojson` and `field_sheet.csv`
   (Paoay 05 observation columns plus visit ID and group). The KML and field sheet leave
   out predictions, so observers record what they see first. After the visit, the same notebook
   turns the filled sheet into stratum-weighted accuracy of the map's crop labels.
   Code: `experiments/field_plan/field_plan.py`.
6. `06_atok_boundary_map.ipynb` — the frozen model on the wider KMZ "Atok" boundary
   (about 1,789 ha), with Sentinel-1 fetched in row strips and ESA WorldCover cropland
   (the land-activity map covers only Paoay). Writes `pipeline_outputs/improved/atok/` and
   `current_atok.json`; the Paoay map from 04 is unchanged. The eastern ~19 % has no local
   Sentinel-2 composites, so it shows as "No features".

7. `07_dataset_variants.ipynb`: reruns 02 and 03 three times, once per development dataset:
   `groupmate` (his package), `yours` (your collection before the merge) and `combined`. A sealed
   `origins` stage labels each parcel group `shared`, `groupmate_only` or `yours_only`. It is built
   once from `~/Downloads/groupmate_data_latest` and the pre-merge backup under
   `sen4agrinet_work/`. A variant drops whole development parcel groups only. Test parcels, locked
   partitions and quality rules stay the same, and each variant writes its own
   `current_<stage>__<variant>.json` pointers, so `current_models.json` and the maps are untouched.
   Code: `experiments/improved_v1/datasets.py`. With the 21 Nov inputs, training parcels are 3,266
   (groupmate), 3,124 (yours) and 3,650 (combined). Your extra 1,500 parcels add only 493 here,
   because 1,007 of them are in the reserved development test split, which this pipeline does
   not use.

**Status (21 Nov 2026, proposed):** this pipeline is the recommended way to map Paoay,
because fixed four-month windows face a trade-off there (transfer notebook 13): the best
European windows (May–Sep) are 1–6 % clear in the Atok monsoon, and the only fully clear
window (Jan–Apr 2024) is the weakest in Europe. The main fixed-window pipeline stays the
baseline for comparison. The map is still potato-biased and unvalidated; use it to plan
the January visit, not as a result. The change is pending agreement with the team.

| Kept from the main pipeline | Added from the experiments |
| --- | --- |
| Verified Preprocessing 01 handoff, locked partitions | Annual harmonic features (no window, no green-up detection) |
| Crop / parcel / confidence training weights, Random Forest | Sentinel-1 backscatter and VH/VV, cross-pol ratios |
| Parcel-level probability averaging | Omega 1.5 frequency option |
| Receipts, stage pointers, single external test | Selection by spatial CV + dataset holdout, not validation alone |
| 0.60 / 0.10 confidence rule, Unknown class | Full Paoay coverage; land-activity cropland flag |

Not carried over, because the experiments showed no benefit: training augmentation for
harmonics, climate (GDD) matching, removing calendar timing, source-balanced weights,
anchor calibration, the 3×3 median filter, and clustering with crop statistics.

Limits: the per-pixel clear flag replaces the 80% parcel coverage rule; omega 1.5 looked
best on JECAM in the pilots, but selection here uses development data only; the Paoay map
has no local validation.
