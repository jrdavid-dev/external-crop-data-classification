# External crop data classification

Prepare carrot, cabbage and potato training data with **small, incremental parcel batches**. The default `adaptive_training` workflow collects annual histories for selected candidates, checks their usable four-month windows, and tops up only crops/partitions below the initial sample targets. It does not default to downloading all 18,012 FranceCrops tiles.

Initial targets **per crop**, after quality filtering:

| Partition | Usable pixels | Minimum source parcels |
| --- | ---: | ---: |
| Training | 1,200 | 60 |
| Validation | 400 | 20 |
| Reserved test | 400 | 20 |

Retain at most 20 pixels per parcel. These are collection budgets, not a guarantee of accuracy. The default **pilot** analysis can train with qualifying available data while reporting unmet budgets; **strict** analysis requires the targets. Both preserve quality checks and locked parcel partitions.

## Setup

Open the repository and select a Python 3.12+ kernel (3.13 recommended). Collection notebooks 02, 03 and 05 install missing dependencies and configure local settings. Choose `reference_mode = "download_missing"` when references are missing or when deliberately expanding Sen4AgriNet’s bounded reference scan. Collection 04 uses local EuroCrops references. Use `"offline"` for local inspection without package installation or satellite calls. Enter an Earth Engine-enabled project ID and sign in when prompted. Nothing starts simply by opening a notebook.

Manual installation remains available:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebooks.txt
python -m pip install -e .
```

On Windows, use the corresponding Python launcher and `.venv\Scripts\activate`.

## Notebook workflow

Start with the [project design and deliverables](docs/project_design.md) for the current experiment, data roles, quotas, and remaining work. For a full, read-only walkthrough of the project, open the [project walkthrough](notebooks/00_project_walkthrough.ipynb). It covers basic terms, the reasons behind each choice, earlier Atok experiments, collection, preprocessing, training, external testing, the Paoay map, and every recorded decision and open issue. The shorter [crop classification walkthrough](notebooks/00_crop_classification_walkthrough.ipynb) summarises only the latest run's results. Both only read saved outputs and run in seconds.

1. [Data collection](notebooks/data_collection) audits references and collects incremental development batches from Sen4AgriNet, FranceCrops and EuroCrops, plus a separate CropHarvest JECAM test cohort. Run collection 06 for a local dataset/progress report. Then run [preprocessing 01–03](notebooks/preprocessing) to audit imagery, select windows and export development/test features. Each notebook prints its handoff paths.
2. [Training 01–02](notebooks/training) fits and validates Random Forests, then freezes the selected models. [Testing 01–02](notebooks/testing) evaluates the matching CropHarvest snapshot and reports crop metrics, confusion matrices, coverage and a training-majority baseline. See [the notebook guide](docs/external_training_notebooks.md).
3. [Paoay](notebooks/paoay) runs 01–05 from references through field review, then the optional 06 (land activity) and 07 (exploratory aligned-model map). For the January–April 2024 map, set `paoay_model_manifest` in local settings to the `fixed_full` manifest printed by Training 02. See the [Paoay guide](docs/paoay_workflow.md).
4. [Archived notebooks](notebooks/archive) retain earlier experiment outputs and provenance.

Collection notebooks 02–05 use `submit_only = False` for the current top-ups. Run All to submit missing jobs and download results. If a run reports pending jobs, rerun with the same settings to resume. `reference_mode = "reuse"` reuses references but does not disable satellite collection.

Current new-batch requests target potato in Sen4AgriNet, cabbage in FranceCrops, carrot in EuroCrops, and non-JECAM cabbage in CropHarvest. Set Collection 05 to JECAM mode for the saved carrot and potato test top-ups. These are shortage top-ups, not the full dataset scope. Existing batches retain their original crop selections, counts, identities and partitions. Review `extra_batches` before running; changing notebook locations does not reset campaign progress.

Collection 05 can run separately while development sources are collected. Its optional final export cell writes completed JECAM bundles. On the receiving machine, import the JECAM bundle using the existing collection transfer helper before starting preprocessing 01. Preprocessing then checks overlaps and continues to 02–03. Existing local completions remain usable.

[Collection 04](notebooks/data_collection/04_eurocrops_collection.ipynb) uses the local `data/external/eurocrops_v2/NL_2021.parquet` file for carrot top-ups. Preprocessing 01 automatically includes completed EuroCrops handoffs. CropHarvest remains the external test source.

The collection notebooks refresh organized output links after completion. Collection 06 saves `pipeline_outputs/collection/reports/summary_<id>/` containing a readable report, reference register, batch states and metadata manifest. This is a local metadata snapshot, not imagery-quality certification. Preprocessing 03 reports the final usable field/pixel counts by crop and partition. See the [collection guide](notebooks/data_collection/README.md).

Original collection notebook bytes and outputs are preserved in `notebooks/archive/collection_before_reorganization/`. Use the active `data_collection/` notebooks for new work.

## Annual TIFF folder

For organized access to existing downloads and results, use `pipeline_outputs/`
with collection, preprocessing, training, and testing folders. Refresh it with
`.venv/bin/python -m crop_pipeline.output_layout`. This linked layout preserves the
original files and saved paths. See [output organization](docs/output_layout.md).

Completed imagery is organized under:

```text
sen4agrinet_work/annual_tifs/<source>/<year>/<campaign>/round_<number>/
```

Each tile has one annual TIFF with **168 labelled bands**: all 24 half-month periods × seven measurements. This preserves flexibility to revise four-month windows. Only selected parcel pixels are populated. Sampling masks are stored separately; Parquet files are retained as the compact transport/cache. Preprocessing 01 reads the annual TIFFs directly. New external pixel-table exports also store three observation diagnostics per interval, for 240 bands. Older 168-band archives remain readable. Set `annual_archive_root` in local settings to use another local drive; annual TIFFs are stored in a sibling folder.

Earth Engine jobs use temporary project asset storage; Google Drive is not required. Local files are verified before temporary assets are removed. The original full-year catalogue/archive helpers and earlier explicit collection selectors remain available for deliberate use, but are not the source notebooks' default.

## Guides

- [Adaptive training and annual TIFFs](docs/adaptive_training.md): budgets, candidate selection, top-ups, locked partitions and file layout.
- [Fresh clone](docs/fresh_clone.md): environment and source setup.
- [Progress](docs/progress.md): measured local completion and task states.
- [Pixel-table transport](docs/pixel_tables.md): export/cache format and the earlier bounded benchmark.
- [Full-year catalogue archive](docs/full_year_collection.md): optional full-catalogue collection and previous transport modes.
- [Methodology](docs/methodology.md): imagery rules and timing/quality limitations.
- [Training](docs/training.md): evaluation and evidence requirements.
- [Paoay application](docs/paoay_workflow.md): the five application notebooks, screening flags and frozen reference review.

Raw data, generated files, credentials, local settings and environments stay out of Git. The upstream dataset licenses do not assign a license to this project's code.

## Development

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests verify software behavior and numerical transport; they do not establish crop-label or phenology accuracy.
