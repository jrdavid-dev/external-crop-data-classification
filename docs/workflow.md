# Run the external classification workflow

Start with [the project design and deliverables](project_design.md) and [environment setup](fresh_clone.md). Run notebooks in the order below. Use each folder's numbers because numbering restarts for each stage.

| Stage | Run | Output |
| --- | --- | --- |
| Collection | [01–06](../notebooks/data_collection/README.md) | Source audits, annual TIFFs, batch records and a collection summary |
| Preprocessing | [01–03](../notebooks/preprocessing/README.md) | Quality audit, fixed/aligned windows, separate development/test features and usable counts |
| Training | [01–02](../notebooks/training/README.md) | Validation results and frozen models |
| Testing | [01–02](../notebooks/testing/README.md) | Parcel predictions, crop diagnostics, coverage and baseline comparison |

Collection 01 audits sources. Run 02–04 for development and 05 for test imagery. Collection 06 summarizes local records. Notebook 05 defaults to the separate cabbage supplement and requires completed primary JECAM and development inputs for its overlap screen. Set `collection_cohort = "jecam"` for primary JECAM. [CropHarvest guide](cropharvest.md)

- `download_missing` obtains missing references where supported. `reuse` uses local references. `offline` disables network operations in notebooks that support it. EuroCrops uses a local parcel file.
- `submit_only = True` queues exports. `False` allows submission and download. If jobs are pending when the notebook returns, rerun it with the same settings. `extra_batches` is the total number of extra slots for the round.
- Finish pending collection before starting a fresh preprocessing audit. Rerun preprocessing 01–03 after adding data. Use `usable_samples.csv` to decide which crop and split need a top-up.
- Each downstream notebook uses the prepared handoff from the preceding stage. Use explicit paths for a deliberate historical run. Pair the frozen models with the matching test feature snapshot.
- Use [the output guide](output_layout.md) to locate results and earlier runs. Preserve TIFFs, annual archives, collection ledgers, and the inputs for any current analysis.

Archived workflows use different notebook numbers and may use different experiments. The [Paoay application](paoay_workflow.md) has its own entry points.
