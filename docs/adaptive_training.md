# Collect training data in rounds

The default `adaptive_training` workflow collects annual histories for selected parcels, then checks usable four-month windows in notebook 04. Collection is bounded; it does not download the full source catalogue by default.

## Run notebooks 01–03

| Notebook | Candidate limit | Reference preparation |
| --- | --- | --- |
| 01 Sen4AgriNet | Up to 100 carrot and 100 potato parcels per round | Local references by default; 350-patch cap |
| 02 FranceCrops | Up to 100 cabbage and 100 potato parcels per round | Reuse pinned reference tables; obtain missing inputs explicitly |
| 03 CropHarvest | Up to 100 parcels per available crop per round | Reuse labels; obtain missing inputs explicitly, then screen eligibility |

**Shared reference controls:** all three default to `reference_mode = "reuse"`. Choose `"download_missing"` on a fresh clone or `"offline"` for local-only inspection. Sen4AgriNet has a separate `reference_target_parcels_per_crop = 100` and `reference_scan_max_patches = 350`; it stops at either limit and reports available candidates and reference deficits. If the pool is exhausted, deliberately increase the reference target and, if needed, the patch cap. This does not move existing selections or jobs. Unseen patches have not been audited.

1. Review settings, then **Restart Kernel → Run All**. Setup prepares local settings and, outside offline mode, missing packages. Enter an Earth Engine-enabled project ID and sign in when prompted.
2. Keep the laptop on until Run All returns normally. Inspect reported job states and errors. Submitted cloud jobs can continue while the laptop is off.
3. Rerun the same notebook to collect existing jobs and create local annual TIFFs. Pending runs reuse selections and checkpoints. All source notebooks resume original plans and preserve earlier selections while filling their round limit.
4. Once launched batches finish, run **04 Four-month windows**. It combines completed rounds, applies timing and quality checks, and reports usable pixels, parcels and deficits. Partial source coverage is allowed; outstanding launched batches must finish first.
5. Rerun relevant source notebooks to fill measured crop/partition deficits, then run 04 again. Continue to **05–06** only when `Targets met: True`.

All source notebooks use the same 100-per-crop round limit. Existing selections count toward it; older saved batches above 100 remain intact and are resumed without adding candidates to that crop. FranceCrops has no carrots. A source with no candidates for the remaining deficits is skipped. Candidate limits do not guarantee usable yield, runtime or completion overnight.

## Usable-sample targets

Targets apply separately to carrot, cabbage and potato **after** timing, eight-bin completeness and quality checks:

| Partition | Usable pixels | Minimum distinct parcels |
| --- | ---: | ---: |
| Training | 1,200 | 60 |
| Validation | 400 | 20 |
| Reserved test | 400 | 20 |

Retain at most 20 unique pixels per parcel. These are initial sampling budgets, not evidence of crop accuracy. Do not relax checks or duplicate pixels to fill deficits.

Only eligible upstream training references enter selection. Physical parcels receive deterministic, locked train/validation/test assignments, approximately 60/20/20. Fragments and repeated years stay together. Top-ups use counts and completeness, not model scores; reserved-test performance must not guide collection or window choices. External-source results do not establish performance in Benguet.

## Saved files

Default folders:

```text
sen4agrinet_work/
  adaptive_training/<campaign>/    # selections, state, feedback, locked partitions
  annual_archive/                 # references, masks, Parquet caches, summaries
  annual_tifs/<source>/<year>/<campaign>/round_<number>/
```

Each completed tile has an annual TIFF with **168 labelled float32 bands**: 24 half-month periods × six spectral bands plus clear-observation count. Descriptions use `<period>__<band>`. Only selected parcel pixels are populated; surrounding cells are nodata. Reflectance remains unscaled until window auditing; counts are never scaled.

TIFFs are verified on read-back and have checksum receipts. Notebook 04 reads them directly. Parquet transport caches support reconstruction without another satellite retrieval; sampling masks are separate and regenerated if missing. Set `annual_archive_root` before large transfers to choose another drive; `annual_tifs` is its sibling folder.

Keep existing archives and campaign records. Changed window rules can reuse annual histories: rerun 04 before requesting top-ups. Windows crossing a year boundary need explicit adjacent-year imagery. Changed masks, bands or compositing rules may need new processing; changed sampling policy creates a separate campaign.

For setup see [fresh clone](fresh_clone.md); for imagery rules see [methodology](methodology.md). Full-catalogue collection is an [optional separate workflow](full_year_collection.md).
