The current default workflow is [incremental adaptive training with annual TIFFs](adaptive_training.md). Earlier full-catalogue guides describe optional legacy collection modes.

# Workflow guides

The goal is a repeatable workflow: **labelled parcels → satellite observations → usable windows → features → models → saved evaluation**. Full-year collection precedes usable-window selection.

| Guide | Read it for |
|---|---|
| [Progress and estimates](progress.md) | Read progress bars, ETA limits and saved timings |
| [Fresh clone](fresh_clone.md) | Run All without preparing datasets or JSON files manually |
| [Setup](../README.md#setup) | Install dependencies and configure local folders |
| [Sen4AgriNet](sen4agrinet.md) | Collect all upstream years/tiles without overwriting old files |
| [Full-year collection](full_year_collection.md) | Understand scope, storage, switches and preservation |
| [CropHarvest](cropharvest.md) | Prepare a safe supplemental collection before window selection |
| [Workflow](workflow.md) | Run the notebooks in order, find outputs and resolve common errors |
| [Processing rules](methodology.md) | Understand sampling, cloud masks, timing and usable pixels |
| [Feature designs](feature_experiments.md) | Understand the three model inputs |
| [Training](training.md) | Understand parcel splits, model outputs and later production requirements |

## Terms used in the notebooks

| Term | Meaning |
|---|---|
| Candidate pixel | A selected location that still needs timing and quality checks |
| Usable pixel | A selected location with all required values in its chosen window |
| Parcel group | The source identity kept together when splitting data |
| Handoff | A saved file describing inputs for the next notebook |
| Pending | A requirement is not ready; inspect its explanation before proceeding |

The default collection is `full_year_all_parcels`; source notebooks are configured for Run All; nothing executes until you run their cells. Old runs remain separate. Local exploratory training works from complete pixel tables; formal production exports have additional evidence and freeze requirements.

- [Annual selected-pixel tables](pixel_tables.md): full-year histories, compact archives, editable windows and the bounded live benchmark.
