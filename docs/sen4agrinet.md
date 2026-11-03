# Sen4AgriNet reference-only collection

[Notebook 01](../notebooks/01_sen4agrinet_preparation.ipynb) is configured for **Restart Kernel → Run All**. It scans every year and tile in a pinned Sen4AgriNet release, extracting eligible carrot, cabbage and potato locations **without downloading complete imagery NetCDFs**. It then collects annual Sentinel-2 composites separately through Earth Engine.

## Settings

```python
retrieve_reference_masks = True
rebuild_field_inventory = True
run_satellite_collection = True
authenticate_now = True
dataset_revision = None
```

First-run setup installs missing packages, including `h5py`, creates local settings and asks for your Earth Engine project if needed. No prior datasets or JSON files are required. See [fresh-clone setup](fresh_clone.md).

To reuse a completed reference inventory without any reference requests, disable the first two switches. Disable satellite collection and authentication too for a fully offline run with an already prepared environment.

## How it is faster

The labels and parcels are embedded in NetCDF/HDF5 files rather than supplied as separate source files. A seekable reader uses small HTTP byte ranges to access:

1. HDF5 headers and mask metadata.
2. The label mask and its coordinates.
3. The parcel mask only when target crop labels occur in the patch.

Spectral arrays are not accessed. Requested byte blocks can include a small amount of adjacent data. Verified existing local NetCDFs can also be reused without network transfer. A server that does not honor Range stops the scan **before its full response body is read**; there is no automatic full-file download fallback.

This follows supported [HDF5 file-object access](https://docs.h5py.org/en/stable/high/file.html#python-file-like-objects). Actual savings depend on upstream HDF5 layout, compression, request latency and network speed. No fixed speedup or real-data transfer size is claimed before a live scan.

## Run All steps and outputs

1. **Discover all patches.** Pin a repository release and list every year/tile. The displayed full-NetCDF size is context, not planned transfer size.
2. **Read masks selectively.** Cache compressed mask-only NetCDFs and per-patch transfer receipts. Empty target patches need no parcel payload.
3. **Validate and extract references.** Keep existing crop codes, valid-mask semantics, 10 m grids, coordinates, parcel IDs, geometry checks and ambiguous-group exclusions. Fill values and scale/offset metadata retain their existing netCDF4 interpretation.
4. **Complete the inventory.** Checkpoint every patch, including crop absence. Any failed patch blocks publication. Restarting reuses completed checkpoints; cached masks can be reused after feature-extractor changes.
5. **Print actual transfer.** `Reference bytes received this scan (MiB)` measures bytes received during that scan. It excludes Earth Engine imagery and is distinct from the full source-file size.
6. **Prepare and collect annual imagery.** Keep all eligible pixels without pilot quotas, print the satellite-storage estimate, and collect all 24 half-month composites locally. Timing selection remains in notebook 04.

The partial reader checks returned byte offsets and total file size against the pinned source. It **cannot verify the entire upstream NetCDF checksum** without fetching the whole file. Receipts record this limitation, the pinned source identity and SHA-256 of the local mask cache. Existing complete local files still require full size/hash verification before reuse.

## Storage and preservation

New references default to `<annual_archive_root>/sen4agrinet_references/`. Change `sen4agrinet_reference_root` in local settings to choose another local drive.

```text
sen4agrinet_references/
  selected_release.json
  releases/<release-sha>/repository_manifest.json
  releases/<release-sha>/reference_masks/    small masks and transfer receipts
  releases/<release-sha>/inventories/<extractor-hash>/
    checkpoints/                           extracted references per patch
    fields/<year>/                         completed reference inventories
    excluded_parcel_groups.json
    completed_inventory.json
    last_scan_transfer.json
```

Old raw files, full-download runs, field inventories, imagery and model results remain in their original folders. No full NetCDF staging is performed by the notebook's reference-only mode. Window checks, pixel quality rules and Earth Engine imagery processing remain unchanged.

After all three source notebooks finish, run notebooks 04 → 05 → 06. The January–December archive still needs an explicit adjacent-year extension for boundary-crossing windows.
