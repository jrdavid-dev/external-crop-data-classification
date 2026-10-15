# Prototype guides

The goal is a repeatable workflow: **labelled parcels → satellite observations → usable windows → features → models → saved evaluation**. Larger or better datasets can be added later.

| Guide | Read it for |
|---|---|
| [Setup](../README.md#setup) | Install dependencies and configure local folders |
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

The active run is `bilinear_expanded`. Old runs remain separate. Local exploratory training works from complete pixel tables; formal production exports have additional evidence and freeze requirements.
