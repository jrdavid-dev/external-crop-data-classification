# Share imagery with a groupmate

Send the **whole exported folder**, optionally compressed as one ZIP, through a
shared drive or USB. Keep its subfolders intact. TIFFs alone lack the labels,
masks and locked partitions needed by this workflow. The package contains only
completed batches; `package.json` lists included and pending coverage.

## Sender

From the repository root, using the project environment:

```bash
python -m crop_pipeline.share export pipeline_outputs/transfers/groupmate_data
python -m crop_pipeline.share verify pipeline_outputs/transfers/groupmate_data
```

Choose a new output folder for a later snapshot. Export copies files; it does not
move imagery, submit jobs or change collection state. ZIP the exported folder
using your file manager, then send it with the matching version of this repository
(including any current local code changes). Do not send local settings or credentials.

## Groupmate

1. Obtain the matching repository code. Create and activate a Python 3.12+ environment:

   ```bash
   python -m venv .venv
   ```

   Activate with `source .venv/bin/activate` on macOS/Linux, or
   `.venv\Scripts\activate` in Windows Command Prompt. Then install:

   ```bash
   python -m pip install -r requirements-notebooks.txt
   python -m pip install -e .
   ```

2. Extract the ZIP. From the repository root, use the path to the folder containing
   `package.json` (replace `PATH_TO_PACKAGE` below):

   ```bash
   python -m crop_pipeline.share verify "PATH_TO_PACKAGE"
   python -m crop_pipeline.share import "PATH_TO_PACKAGE"
   ```

3. Open **preprocessing 01 → 02 → 03**, then **training 01 → 02**, then
   **testing 01 → 02** using the project kernel. Skip collection notebooks.
   Import uses the default local output paths unless your own ignored
   `settings.local.json` already sets `report_root`. Earth Engine authentication
   and new imagery downloads are unnecessary for these local stages.

Repeated import of identical batches reuses their registered paths. Conflicting
parcel selections stop import; use a fresh clone for a separate experiment.
Pre-existing pending local collections still block the standard imagery audit.
A partial package may lack required sources or enough qualifying crops; the normal
checks remain active. Installing dependencies needs internet unless already cached.

## What is included

- Annual TIFFs with their original named bands and checksums.
- Sampled-region GeoJSON: crop labels, stable parcel/group IDs, grids and source properties.
- Original parcel footprints and sampling masks for overlap/quality checks.
- Daily observation timelines needed for green-up assessment.
- Source handoffs: years, source revision, imagery recipe, TIFF index and provenance.
- Saved selections and locked training/validation/test assignments.
- Bundle checksums and a package inventory. Supplements retain their own policies.

Operational cloud job state, raw upstream catalogs, transport Parquet caches,
settings and credentials are not needed for this handoff. The imported receipts
record the sender's handoff hashes while relocating paths to the receiving machine.
The importer copies imagery into the receiver's collection folders; allow space
for the extracted package plus an imported copy. Preflight verification also uses
a temporary copy before publication.

Source attribution and dataset terms still apply. Keep source revisions and
provenance with the files; consult the repository's source guides and upstream
terms before redistribution beyond the project group. Software test results do
not establish crop-label or model accuracy.
