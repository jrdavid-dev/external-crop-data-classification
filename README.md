# External crop data classification

Prepare labelled carrot, cabbage and potato parcels with annual Sentinel-2
imagery, compare fixed and green-up-aligned four-month windows, and evaluate
frozen Random Forests on external test data. The current pipeline uses incremental
parcel batches and reports sample shortfalls alongside results.

Start with [project design](docs/project_design.md) for the experiment and targets,
then follow [the workflow](docs/workflow.md). The [documentation index](docs/README.md)
contains ten guides, with technical references and history kept separate.

## Setup

Use Python 3.12+ in the project environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebooks.txt
python -m pip install -e .
```

On Windows Command Prompt, activate with `.venv\Scripts\activate`.
New imagery collection requires an Earth Engine-enabled project and authentication.
Imported imagery can be preprocessed, trained and evaluated locally without an
Earth Engine login. See [setup](docs/fresh_clone.md).

## Notebook workflow

| Stage | Entry points | Purpose |
| --- | --- | --- |
| Collect or import | [Collection](notebooks/data_collection/README.md) or [shared imagery](docs/sharing_imagery.md) | References, annual TIFFs, masks, timelines and locked selections |
| Preprocess | [01–03](notebooks/preprocessing/README.md) | Audit quality/overlaps, select windows, export separate development/test features |
| Train | [01–02](notebooks/training/README.md) | Fit/select on development, then freeze without refitting |
| Test | [01–02](notebooks/testing/README.md) | Score matching external features and report parcel metrics, coverage and baseline |

Choose only the source collections required for the experiment or current top-ups.
Rerun pending jobs with the same settings. Review crop filters, batch limits and
reference modes before Run All; opening a notebook starts no work.

Preprocessing reports final usable pixels and parcels per crop, split and recipe.
Training requires all three crops in training and validation; collection budgets
are reported separately from quality/class coverage. Select models with validation
only. Missing test crops suppress three-class aggregates. Software completion and
sample targets do not establish crop accuracy.

The optional [Paoay 01–05 plus optional 06–07](docs/paoay_workflow.md) is separate
and deferred for the current external experiment. Its historical January–April
2024 application requires a matching saved model recipe and independent dated
reference evidence before any local accuracy claim. Notebook 07 is an exploratory
aligned-model application with documented imagery/timing differences; it is
separate from primary external evaluation.

## Data and outputs

Authoritative working data is under **`data/pipeline_work/`**; upstream reference
files are under `data/external/`. Annual TIFFs, masks, transport caches and
collection ledgers must stay together. `pipeline_outputs/` provides collection/
preprocessing navigation and native training/testing run folders. See
[storage and organized outputs](docs/storage_layout.md) for paths and refresh commands.

To let groupmates work without downloading imagery again, use the
[export/import guide](docs/sharing_imagery.md). Send the complete package and matching
code, not TIFFs or filesystem links alone. Credentials, private settings, raw data,
generated results and environments stay out of Git. Upstream dataset terms do not
assign a license to this project's code.

## Development

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Local fixture tests verify software behavior, not scientific validity.
