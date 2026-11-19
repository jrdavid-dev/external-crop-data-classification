# Local data storage

> Historical record. Instructions and test totals describe their original checkpoint; do not use old notebook sequences as the current run guide. See the [current workflow](../workflow.md) and [history index](README.md). Links have been adjusted for this documentation move.

The default working root is **`data/pipeline_work/`**, shared by all sources.
`data/external/` continues to hold upstream reference downloads. Collection
notebooks resolve configured archive paths against the repository, independent
of their kernel's starting directory.

| Path under `data/pipeline_work/` | Contents |
| --- | --- |
| `annual_archive/` | Reference snapshots, selected-pixel transport caches, masks and acquisition metadata |
| `annual_tifs/` | Annual TIFFs with named bands and checksum indexes |
| `adaptive_training/` | Campaign policies, locked parcel selections, batch plans, states and completion receipts |
| `inventory/` and `progress/` | Collection reports and saved progress |
| `analysis/` | Saved quality/window/features runs and historical experiments |
| `paoay/` | Local application inputs and saved stage bundles where present |

`pipeline_outputs/` is the organized navigation/results view. Existing training
and testing runs there remain in place. `data/` and local settings stay ignored
by Git. To share data with teammates, use [the portable package](../sharing_imagery.md)
rather than copying local storage folders.

## Migration of the existing workspace

The root and nested notebook `sen4agrinet_work/` directories were consolidated
into `data/pipeline_work/`. Preflight rejects file collisions before moving data.
The migration uses same-filesystem renames, then verifies SHA-256 and size for
every moved file. The recovery inventory is saved locally under
`data/storage_migrations/<timestamp>/manifest.json` with original paths, relative
destinations and hashes. Local settings are backed up privately before editing.

A second, explicit metadata rebase removes the old compatibility links. Saved
paths now name `data/pipeline_work/` directly. File-checksum dependencies are
updated from leaves to receipts; original transfer handoff identities remain
unchanged. No parcel assignments, cloud job identities, raw arrays, TIFFs or
classifier binaries are regenerated.

The rebase stores original affected metadata in
`data/storage_migrations/rebase_<timestamp>/original_metadata.tar.gz`, plus a
checksum dependency index, before/after hashes and a migration status record.
These are relocation records, not a claim that historical runs were generated
by the current implementation. The originals remain recoverable locally. Old
portable teammate ZIPs remain independent snapshots and do not require local
compatibility links.

Migrating storage does not refresh scientific results. If an old preprocessing
receipt fails because its recorded implementation changed, start a new
Preprocessing 01–03 run; keep the old result and receipt intact. Do not edit
receipts to make a changed implementation look like the original run.

An external drive can still be configured with `annual_archive_root` in ignored
`settings.local.json`. Absolute custom paths remain supported. `report_root`
locates the inventory and its sibling campaign/analysis directories.

## Verification of this migration

On 2026-11-19, the live migration verified **168,933 files** totaling
32,371,685,890 bytes against their original SHA-256 checksums. All 115 completed
primary/supplemental handoffs and their TIFF/mask paths remained readable; core
snapshot discovery loaded 104 source/year handoffs and 5,579 locked development
assignments. The output navigation was refreshed.

The full local suite passed **485 tests**, including storage collision/alias
checks, nested-notebook path resolution, offline imagery transfer, and the
notebook handoff chain. Notebook code compiles and guide links resolve. Saved
notebook outputs and metadata were preserved. No cloud collection or full-data
training was launched; these checks do not establish scientific accuracy.

The path rebase is checked separately from the earlier physical move: active
handoffs, sampling masks, transport cache reuse and collection progress are
verified after removing the aliases. See the local rebase migration record for
its exact file list and original metadata.

## Removing the legacy aliases

The old root and notebook-directory aliases were removed after a dependency-aware
metadata rebase. **42,216 metadata files** were updated; the original backup was
verified against its original checksums before finalization. All changed metadata
was checked again for its new checksum and absence of the legacy storage name.
Output navigation links now point directly to the canonical data root.

After unlinking, all **115 completed handoffs** and **115 sampled transport-cache
reuse checks** passed. All 194 checked progress/assignment records were unchanged,
including 5,579 locked development assignments. Four frozen classifiers loaded,
and previously verified training/testing receipts remained verified. The full
suite passed **490 tests**. Existing preprocessing receipts were already
unavailable before this rebase; run Preprocessing 01–03 with the current code to
publish a fresh preprocessing snapshot.

Pixel cache filenames retain their original identities. When an absolute-path
change alters the computed name, lookup accepts an existing cache only if its
full saved contract matches; reuse still checks the data checksum and the normal
quality rules. Changed settings or corrupt cache data are not accepted. No imagery
was downloaded and no new scientific training run was launched.

Post-rebase checksum checks also confirmed that all 39,212 non-text files from
the original storage inventory and 16,479 checked native-output binaries remained
unchanged. These populations can overlap; their counts are not additive.
