# Run the workflow

Install the requirements, configure `settings.local.json`, and select `.venv` as the kernel. Run notebooks from the repository folder so helper imports and relative paths resolve correctly.

## 1. Sen4AgriNet: carrot and potato

Open [Sen4AgriNet_Satellite_Preparation.ipynb](../Sen4AgriNet_Satellite_Preparation.ipynb).

If the saved field inventory exists, leave the first settings cell at:

```python
download_netcdf_files = False
rebuild_field_inventory = False
install_dependencies = False
```

Use **Run All**. The optional raw-data section comes first and prints intentional skip messages. The remaining cells:

1. Load `fields/2020/download_candidates.geojson`.
2. Select 3,000 candidates per crop and save masks and footprints.
3. Connect to Earth Engine, inventory acquisitions and download composites.
4. Retrieve actual-date vegetation summaries.
5. Save the source handoff.

For a first run without saved fields, enable raw downloading for the chosen tile/batch and enable field rebuilding. If raw files already exist, enable rebuilding without downloading. One batch may not supply enough crop coverage for the candidate budget.

## 2. FranceCrops: cabbage

Run [FranceCrops_Satellite_Preparation.ipynb](../FranceCrops_Satellite_Preparation.ipynb) from Step 1 through Step 9.

| Steps | Work |
|---|---|
| 1–4 | Load settings, obtain reference tables, verify cabbage labels and select 3,000 candidates |
| 5–7 | Connect to Earth Engine, inventory acquisitions and download satellite composites |
| 8 | Retrieve actual-date vegetation summaries |
| 9 | Save the source handoff |

The source tables contain labels and parcel boundaries. The large interpolated FranceCrops satellite arrays are not needed. The notebook retrieves matching Sentinel-2 imagery independently.

For either source, set `authenticate_now = True` in its retrieval settings only when login is needed. Rerun failed satellite downloads; completed files are reused after grid and checksum checks.

## 3. Four-month windows and usable pixels

After both source handoffs are saved, run [Crop_Four_Month_Windows.ipynb](../Crop_Four_Month_Windows.ipynb) from its first cell through Step 14. No Earth Engine connection or NetCDF download is needed.

| Steps | Work |
|---|---|
| 6–8 | Audit each source in its own year, construct curves and identify candidate green-up |
| 9–10 | Assess timing and save reference-supported, estimated and unresolved outcomes |
| 11–12 | Build complete features, retain up to 1,000 usable unique external pixels per crop and report losses |
| 13–14 | Apply field partitions and export primary and exploratory training bundles |

A missing source is reported and available sources can be inspected. A complete three-class experiment still needs all three crops. Field assignment can remain pending even when usable feature rows exist; see [training requirements](training.md).

## Important files

Paths below are relative to `report_root`'s parent, normally `sen4agrinet_work/`.

| Location | Contents |
|---|---|
| `fields/2020/pilot_3000/` | Sen4AgriNet selections, footprints and sampling masks |
| `l2a/2020/pilot_3000/` | Sen4AgriNet acquisition and download records |
| `analysis/2020/pilot_3000/timelines/` | Sen4AgriNet actual-date summaries |
| `francecrops/` | Cabbage source tables, selections, acquisition records and summaries |
| `source_handoffs/` | `sen4agrinet_3000.json` and `francecrops_3000.json` |
| `analysis/pilot_3000/seasons/` | Combined windows, feature tables, selection counts and training inputs |

Sen4AgriNet imagery is stored under configured `imagery_root/pilot_3000/`. Cabbage imagery uses `francecrops_imagery_root/pilot_3000/`, or defaults to `dataset_root/sentinel2_francecrops/pilot_3000/`. Previous pilot outputs remain separate.

Within the combined `seasons/` folder:

| File or folder | Meaning |
|---|---|
| `season_review_audit.csv` | Every window outcome and its reason |
| `automatic_timing_audit.csv` | Timing checks and failures |
| `features/`, `estimated_features/` | Separate supported and estimated feature tables and losses |
| `usable_pixel_selection.csv` | Available pixels, selected pixels and shortfalls by crop |
| `review/` | Field registry, review and configuration templates |
| `training_input/`, `exploratory_training_input/` | Separate model-input bundles |

## Common problems

| Message or result | Meaning and next step |
|---|---|
| Saved field inventory missing | Restore the inventory or build it from raw source files |
| NetCDF preparation skipped | Expected with default reuse settings; continue sequentially |
| Source not prepared | Complete the corresponding source notebook's final handoff cell |
| Source sampling changed | Finish acquisition for the new selection and export its handoff again |
| Failed composites | Inspect the download report and rerun retrieval |
| No actual-date summaries | Run the source notebook's summary retrieval; composite-only timing stays provisional |
| Fewer than 1,000 usable pixels | Inspect timing and completeness losses; collect more candidates if necessary |
| Zero reference-supported seasons | Independent crop-cycle evidence has not established supported windows; qualified estimates remain separate |
| Zero partitioned rows / `pending_inputs` | Field registry or eligible model inputs are still missing |

Continue with [model training](training.md) after checking the selected counts and partitions.
