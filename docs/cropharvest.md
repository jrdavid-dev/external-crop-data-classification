# CropHarvest source preparation

The first cell installs missing dependencies and creates local settings when absent; see [fresh-clone setup](fresh_clone.md).

Run [notebook 03](../notebooks/03_cropharvest_preparation.ipynb) before window selection. It automatically downloads missing `data/external/cropharvest/labels.geojson` when Run All executes, using the [official pinned Zenodo release 10251170](https://zenodo.org/records/10251170). Existing labels are validated and reused without overwriting them. Only the label file is fetched; the large feature and imagery archives are not requested.

The notebook keeps explicit crop aliases, mixed-label exclusions, reliable-source checks, polygon validation, duplicate screening and upstream test exclusions. Observation years must be supported by the documented export period or an explicit override. A January 1 placeholder can identify a supported season year; it is not a planting date. Decisions are saved in `labels/observation_year_audit.csv` in a versioned label-review folder.

Every eligible target parcel and resolved Sentinel-2-era reference year enters the **full-year collection**, without candidate budgets or per-parcel pixel quotas. A 10 m inward buffer and contested-location exclusion remain. There is no timing gate in this source notebook.

`download_missing_labels = True`, `run_satellite_collection = True` and `authenticate_now = True` are configured for the next Run All. Nothing starts merely by opening the notebook. To inspect files entirely offline, set all three to `False` and supply existing valid labels.

The label downloader configures trusted certificates before its request, verifies the release MD5 and GeoJSON structure, and publishes a validated file atomically. Failed downloads are cleaned up and can be retried with Run All. A `.download.json` sidecar records the source release and SHA-256. The normal label-content hash then determines the versioned processing run.

Satellite collection uses Earth Engine processing and checkpointed local GeoTIFF storage. All annual files must finish before a source registry is published. See [full-year collection](full_year_collection.md) for storage, completion and restart details.

Notebook 04 requires all three source registries and performs timing and quality selection afterward. Notebooks 05 and 06 handle features and training. Old CropHarvest runs, pilot imagery and models stay separate.
