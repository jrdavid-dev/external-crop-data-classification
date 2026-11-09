# External crop data classification

Prepare carrot, cabbage and potato training data with **small, incremental parcel batches**. The default `adaptive_training` workflow collects annual histories for selected candidates, checks their usable four-month windows, and tops up only crops/partitions below the initial sample targets. It does not default to downloading all 18,012 FranceCrops tiles.

Initial targets **per crop**, after quality filtering:

| Partition | Usable pixels | Minimum source parcels |
| --- | ---: | ---: |
| Training | 1,200 | 60 |
| Validation | 400 | 20 |
| Reserved test | 400 | 20 |

Retain at most 20 pixels per parcel. These are sampling budgets, not a guarantee of accuracy. Keep whole parcels in one partition and assess external-to-local transfer separately.

## Setup

Open the repository and select a Python 3.12+ kernel (3.13 recommended). Source notebooks 01–03 install missing dependencies and configure local settings. All three share six workflow sections and default to `reference_mode = "reuse"`. On a fresh clone, choose `"download_missing"`; Sen4AgriNet retains an explicit reference patch cap. Use `"offline"` for local inspection without package installation or satellite calls. Enter an Earth Engine-enabled project ID and sign in when prompted. Nothing starts simply by opening a notebook.

Manual installation remains available:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebooks.txt
python -m pip install -e .
```

On Windows, use the corresponding Python launcher and `.venv\Scripts\activate`.

## Notebook loop

1. **01 Sen4AgriNet, 02 FranceCrops, 03 CropHarvest:** notebook 01 targets up to 100 carrot and 100 potato candidates per round from a capped reference subset; notebook 02 targets up to 100 cabbage and 100 potato candidates per round; notebook 03 targets up to 100 eligible parcels per available crop per round. Existing selections and submitted jobs are preserved. Rerun the same notebook to collect its pending jobs.
2. **04 Four-month windows:** combine completed batches, run the existing timing/quality checks, retain balanced usable samples, and save deficits.
3. If targets are unmet, repeat **01–03 → 04**. Previously selected parcels and locked partitions are preserved. Only deficit crop/partition combinations get new candidates.
4. When targets are met, run **05 Feature engineering → 06 Model training**. Adaptive training remains blocked until the targets are satisfied.

FranceCrops contributes cabbage/table potato, not carrots. Sources with no candidates for remaining deficits are skipped explicitly. Candidate batches are not expected to have a guaranteed usable yield.

## Annual TIFF folder

Completed imagery is organized under:

```text
sen4agrinet_work/annual_tifs/<source>/<year>/<campaign>/round_<number>/
```

Each tile has one annual TIFF with **168 labelled bands**: all 24 half-month periods × seven measurements. This preserves flexibility to revise four-month windows. Only selected parcel pixels are populated. Sampling masks are stored separately; Parquet files are retained as the compact transport/cache. Notebook 04 reads the annual TIFFs directly. Set `annual_archive_root` in local settings to use another local drive; annual TIFFs are stored in a sibling folder.

Earth Engine jobs use temporary project asset storage; Google Drive is not required. Local files are verified before temporary assets are removed. The original full-year catalogue/archive helpers and earlier explicit collection selectors remain available for deliberate use, but are not the source notebooks' default.

## Guides

- [Adaptive training and annual TIFFs](docs/adaptive_training.md): budgets, candidate selection, top-ups, locked partitions and file layout.
- [Fresh clone](docs/fresh_clone.md): environment and source setup.
- [Progress](docs/progress.md): measured local completion and task states.
- [Pixel-table transport](docs/pixel_tables.md): export/cache format and the earlier bounded benchmark.
- [Full-year catalogue archive](docs/full_year_collection.md): optional full-catalogue collection and previous transport modes.
- [Methodology](docs/methodology.md): imagery rules and timing/quality limitations.
- [Training](docs/training.md): evaluation and evidence requirements.

Raw data, generated files, credentials, local settings and environments stay out of Git. The upstream dataset licenses do not assign a license to this project's code.

## Development

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests verify software behavior and numerical transport; they do not establish crop-label or phenology accuracy.
