# Collection reference

Use the [workflow](workflow.md) for notebook order and [setup](fresh_clone.md) for
the environment. To use existing imagery, [import a shared package](sharing_imagery.md)
and start preprocessing instead of running collection.

## Source controls

| Collection notebook | Data role | Review before running |
| --- | --- | --- |
| 02 Sen4AgriNet | Development potato/carrot | Target crops, reference pool target, patch cap and parcel limit |
| 03 FranceCrops | Development potato/cabbage | Crop filter and parcel limit |
| 04 EuroCrops | Development carrot | Local Dutch 2021 reference path and parcel limit |
| 05 CropHarvest | Primary JECAM or separate non-JECAM supplements | `collection_cohort`, crop filter and parcel limit |
| 06 California DWR | Supplemental carrot test | Reference-download flag and parcel limit |
| 07 Denmark Marker | Supplemental cabbage test | Local reference snapshot and parcel limit |

Batch sizes and `extra_batches` are workload settings in the notebooks, not
universal scientific constants. Read their current values rather than copying a
past campaign's settings. `extra_batches` is the total number of extra slots for
the round; rerunning with the same settings resumes saved plans and jobs.

For notebooks exposing `reference_mode`, `reuse` concerns local references and
does not disable satellite collection; `download_missing` obtains missing
references; `offline` also prevents installation and satellite calls. EuroCrops
uses a local parcel file. Supplemental notebooks have their own reference-download
controls. `run_satellite_collection = False` disables satellite work, but reference
preparation can still write selections. Use Collection 08 for metadata-only review.

## Sen4AgriNet references

Collection 02 pins a release, inventories patches, and selectively reads label,
parcel and coordinate arrays through bounded HTTP ranges. Spectral arrays are
not fetched for reference extraction. Servers that ignore Range are rejected;
verified complete local NetCDFs can also be reused. Crop codes, grids, geometry,
identities, conflicts and upstream held-out roles are checked before selection.

`reference_target_crops`, `reference_target_parcels_per_crop` and
`reference_scan_max_patches` control the bounded reference scan, separately from
the imagery batch limit. Expand an exhausted pool deliberately with
`download_missing`. Extraction stops at its target or patch cap; the cap does not
bound runtime. No usable checkpoints produces an error, not an empty imagery run.

Unseen patches have not been audited. Saved coverage, deficits and stop reasons
refer to the scanned subset, not the full release. Partial reads check offsets
and source size but cannot verify the entire upstream NetCDF checksum. Local
mask caches have SHA-256 checks; received-byte counts measure reference transfers.

References default to `<annual_archive_root>/sen4agrinet_references/` with pinned
manifests and versioned checkpoints. `sen4agrinet_reference_root` can override this.
Sen4AgriNet retains its declared mask cells with no additional inward buffer.

## CropHarvest labels and cohorts

In Collection 05, use `collection_cohort = "jecam"` for primary testing.
`carrot_potato_supplement` and `cabbage_supplement` select separate non-JECAM
campaigns. Supplements are screened against development and existing test
footprints; finish the required development/primary inputs before their screen.
Preprocessing reports primary JECAM, each registered supplement and the combined
pool separately.

Missing labels are obtained in `download_missing` mode from the pinned release.
Downloads validate MD5 and GeoJSON structure before atomic publication, with a
local SHA-256 receipt. No upstream imagery/feature archive is required. Optional
country borders can be omitted; dataset names provide the recorded fallback.

The audit normalizes explicit crop aliases, excludes mixed labels, checks source
reliability, duplicate identities, polygon geometry and observation years, and
excludes upstream test labels. A January 1 placeholder needs supported year
information. Source dates are not planting dates, and reliability grades are
screening judgments, not independent accuracy evidence. Label/year audits retain
those decisions. Eligible polygon sampling uses the existing 10 m inward buffer.

For groupmates, use `python -m crop_pipeline.share import` as explained in the
[sharing guide](sharing_imagery.md), before Preprocessing 01. The optional
`jecam_export_folder` cell in Collection 05 still exports individual JECAM batches;
it is separate from the all-source package command. Preprocessing 01 does not
have a `jecam_import_folder` control.

## Locked partitions and top-ups

Development parcel assignments are deterministic and locked; fragments remain
with their physical parcel group. Primary and supplemental external-test
campaigns remain test-only. Exclude shared development coordinates within each
recipe, including repeated farmland across years. Preserve upstream reserved
roles and source provenance. Do not change policies to abandon an existing campaign.

Use the per-crop targets in [project design](project_design.md), assessed after
quality/window checks, and final shortfalls from Preprocessing 03 to plan top-ups.
Keep at most 20 usable pixels per parcel. Do not duplicate pixels or relax checks
to fill deficits. Reserved-test scores must not guide collection or window choices.

## Completion and reuse

Keep candidate selections, batch states, task ledgers, masks, TIFFs, daily timelines
and transport caches together. A pending collection pass is not completion:
rerun the same settings to retrieve results and retry unfinished work. Temporary
cloud assets are removed only after verified local collection. Do not run concurrent
writers for the same archive.

Completed collection receipts certify registered files, not final usable training
coverage. Finish pending required batches, run Collection 08 for a local snapshot,
and run Preprocessing 01–03 for overlap, imagery, windows and feature checks.
Changing windows can reuse annual histories; changed masks, bands or compositing
can require new processing. Cross-year windows need explicit adjacent-year imagery.
See [storage](storage_layout.md) and the [transport reference](reference/transport.md).
