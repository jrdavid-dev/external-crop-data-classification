# Run the workflow

For a new clone, follow [fresh-clone Run All](fresh_clone.md): source notebooks install missing packages, create settings and obtain their inputs. A Python 3.12+ kernel and Earth Engine project/sign-in are required. Run cells sequentially. The default collection is **`full_year_all_parcels`**; collection and sign-in are configured for the next Run All, but nothing starts until cells execute.

| Order | Notebook | What it does | Completion output |
|---|---|---|---|
| 1 | [Sen4AgriNet](../notebooks/01_sen4agrinet_preparation.ipynb) | Scan every upstream year/tile; extract references and collect annual imagery | Completed source registry |
| 2 | [FranceCrops](../notebooks/02_francecrops_preparation.ipynb) | Fetch pinned tables and read all declared cabbage/table-potato parcels; preserve upstream roles | Completed source registry |
| 3 | [CropHarvest](../notebooks/03_cropharvest_preparation.ipynb) | Fetch missing labels, screen references and collect annual imagery | Completed source registry |
| 4 | [Four-month windows](../notebooks/04_four_month_windows.ipynb) | Audit saved rasters, detect timing and select complete usable windows | Usable counts, exclusions and window handoffs |
| 5 | [Feature engineering](../notebooks/05_feature_engineering.ipynb) | Prepare interval, shape and combined designs | Feature handoffs |
| 6 | [Model training](../notebooks/06_model_training.ipynb) | Fit parcel-separated models and evaluate predictions | Splits, models, metrics and figures |

The source notebooks are set for Run All. Sen4AgriNet performs full reference discovery and extraction before satellite collection. For preparation-only inspection, disable satellite collection; its `retrieve_reference_masks` flag controls reference-only transfers separately. See [full-year collection](full_year_collection.md) for switches, reference requirements, storage paths and restart behavior. Sources can run in separate kernels; authenticate one at a time and keep the archive drive connected. Finish all three before notebook 04.

## Important checks

- Sen4AgriNet now scans the entire pinned release, not the old subset. Any unscanned patch blocks completion. See [its Run All guide](sen4agrinet.md).
- `status: disabled` means satellite collection did not run; Sen4AgriNet references and CropHarvest labels have their own retrieval switches. `completed` means all planned annual files exist, not that their pixels passed quality checks.
- Notebook 04 keeps the same timing and complete-data requirements. It retains all usable pixels without a pilot cap. Missing cross-year dates remain unresolved.
- Upstream reserved splits stay outside model-development exports. Local exploratory training keeps each parcel in one split and gives each crop equal total training weight.
- Formal production exports may remain pending independent field identities, timing evidence or development freezes. External-source accuracy is not evidence of Benguet accuracy.

## Saved outputs

Source registries are under `source_handoffs/full_year_all_parcels/`, beside the configured report folder. Analysis is under `analysis/full_year_all_parcels_<source-hash>/`. Inspect:

- `source_inputs.json`: exact source contracts.
- `seasons/automatic_timing_audit.csv`: timing passes and failures.
- `seasons/cycle_selection_audit.csv`: candidate completeness and selected windows.
- `seasons/usable_pixel_selection.csv`: unique usable counts per crop.
- `seasons/features/` and `seasons/estimated_features/`: complete model-development features.
- `seasons/reserved_upstream_features.csv`: separate upstream holdouts.
- The printed training output folder: locked parcel split, models and accuracy.

Old pilot results remain in `analysis/bilinear_expanded/`. Set `analysis_collection` to that name consistently across notebooks 04–06 when revisiting them.

## Common problems

| Symptom | Action |
|---|---|
| Reference retrieval failed | Fix the displayed network or validation error, then resume Run All; incomplete inputs are not treated as crop absence |
| Certificate failure | Run the trusted-certificate cell before authentication |
| Required source handoff missing | Finish the named source collection; partial source sets cannot proceed |
| Interrupted collection or `pending_retry` | Restart Kernel → Run All with the same settings; verified files are reused and pending work retried |
| Few usable pixels | Examine timing, missing-bin and coverage exclusions separately |
| Formal partitions are zero | Inspect registry/review requirements; local exploratory splitting is separate |
| Training input changed | Create a new output run rather than replacing a locked evaluation |
