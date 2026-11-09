# Workflow guides

Start with [fresh-clone setup](fresh_clone.md), then follow the [notebook workflow](workflow.md). The default collection is `adaptive_training`: collect bounded parcel batches, assess usable four-month windows, and fill measured deficits.

| Guide | Purpose |
| --- | --- |
| [Fresh clone](fresh_clone.md) | Prepare a kernel, settings and Earth Engine access |
| [Workflow](workflow.md) | Run notebooks 01–06 and resume interrupted work |
| [Sen4AgriNet](sen4agrinet.md) | Understand notebook 01's cached references and capped scans |
| [CropHarvest](cropharvest.md) | Understand notebook 03's label checks and collection |
| [Adaptive training](adaptive_training.md) | Sampling targets, locked partitions and annual TIFFs |
| [Progress](progress.md) | Read local completion counts and cloud task states |
| [Methodology](methodology.md) | Imagery, timing and quality rules |
| [Feature designs](feature_experiments.md) | Compare the three model inputs |
| [Training](training.md) | Evaluation outputs and evidence requirements |
| [Pixel tables](pixel_tables.md) | Understand the compact imagery transport/cache |
| [Full-year catalogue](full_year_collection.md) | Use optional earlier collection modes |

A **candidate** still needs timing and quality checks. A **usable pixel** passes those checks in its selected window. A **parcel group** stays in one locked partition. A **handoff** records saved inputs for the next notebook.

Collection completion confirms saved files; it does not establish crop-label accuracy or local model performance.
