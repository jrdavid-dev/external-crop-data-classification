# Collect training data in rounds

Use [collection 02–05](../notebooks/data_collection/README.md) for top-ups and Collection 06 for progress. [Preprocessing 01–03](../notebooks/preprocessing/README.md) checks the completed imagery and exports usable counts.

| Collection notebook | Candidate limit for new slots | Current top-up |
| --- | ---: | --- |
| 02 Sen4AgriNet | 50 per crop | Potato |
| 03 FranceCrops | 100 per crop | Cabbage |
| 04 EuroCrops | 100 per crop | Carrot |
| 05 CropHarvest | 50 per crop | Separate non-JECAM cabbage supplement by default |

`extra_batches` counts total extra slots in the current round. Increasing it adds slots after existing batches. Rerun with the same settings to resume pending jobs. Sen4AgriNet also has a reference pool target and patch cap. Increase these deliberately and use `download_missing` to expand an exhausted pool. [Notebook workflow](workflow.md)

## Usable-sample targets

Targets apply separately to carrot, cabbage and potato **after** timing, eight-bin completeness and quality checks:

| Partition | Usable pixels | Minimum distinct parcels |
| --- | ---: | ---: |
| Training | 1,200 | 60 |
| Validation | 400 | 20 |
| Reserved test | 400 | 20 |

Retain at most 20 unique pixels per parcel. These are initial sampling budgets, not evidence of crop accuracy. Do not relax checks or duplicate pixels to fill deficits.

Only eligible upstream training references enter selection. Physical parcels receive deterministic, locked train/validation/test assignments, approximately 60/20/20. Fragments stay with their locked group. Preprocessing excludes coordinates shared by different development parcel groups within each recipe, including cross-year aliases. Top-ups use counts and completeness, not model scores; reserved-test performance must not guide collection or window choices. External-source results describe the held-out source.

## Saved files

Default folders:

```text
sen4agrinet_work/
  adaptive_training/<campaign>/    # selections, state, feedback, locked partitions
  annual_archive/                 # references, masks, Parquet caches, summaries
  annual_tifs/<source>/<year>/<campaign>/round_<number>/
```

Each completed tile has an annual TIFF with **168 labelled float32 bands**: 24 half-month periods × six spectral bands plus clear-observation count. Descriptions use `<period>__<band>`. Only selected parcel pixels are populated; surrounding cells are nodata. Reflectance remains unscaled until window auditing; counts are never scaled.

TIFFs are verified on read-back and have checksum receipts. Preprocessing 01 reads them directly. Parquet transport caches support reconstruction without another satellite retrieval; sampling masks are separate and regenerated if missing. Set `annual_archive_root` before large transfers to choose another drive; `annual_tifs` is its sibling folder.

Keep existing archives and campaign records. Changed window rules can reuse annual histories: rerun preprocessing before requesting top-ups. Windows crossing a year boundary need explicit adjacent-year imagery. Changed masks, bands or compositing rules may need new processing; changed sampling policy creates a separate campaign.

For setup see [fresh clone](fresh_clone.md); for imagery rules see [methodology](methodology.md). Full-catalogue collection is an [optional separate workflow](full_year_collection.md).
