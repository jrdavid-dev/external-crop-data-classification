"""Bounded local auditing of annual archives using the existing quality rules."""

import json
from collections import defaultdict
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet

from crop_pipeline.imagery import quality


class PixelPartitions:
    """Keep 24-bin pixel records on disk; materialize only requested regions."""

    def __init__(self, paths):
        self.paths = {identifier: Path(path) for identifier, path in paths.items()}

    def for_regions(self, identifiers):
        for identifier in identifiers:
            path = self.paths.get(identifier)
            if path is not None:
                yield from parquet.read_table(path).to_pylist()

    def __iter__(self):
        return self.for_regions(self.paths)


def write_pixel_partition(path, rows):
    """Store missing bands as nulls rather than mixed numeric/string columns."""
    cleaned = [
        {
            **row,
            **{
                band: row[band] if row[band] != "" else None
                for band in quality.STUDY_BANDS
            },
        }
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(cleaned), path, compression="zstd")


def audit_sources(sources, analysis_folder, minimum_fraction=0.8):
    """Reuse the raster auditor one tile at a time without retaining all pixel records."""
    analysis_folder = Path(analysis_folder)
    regions, quality_rows, observations, partitions = [], [], [], {}
    for source in sources:
        audited = audit_source(source, analysis_folder, minimum_fraction)
        if set(partitions).intersection(audited[3]):
            raise ValueError("Annual source region identifiers collide.")
        regions.extend(audited[0])
        quality_rows.extend(audited[1])
        observations.extend(audited[2])
        partitions.update(audited[3])
    quality.save_json(analysis_folder / "source_inputs.json", sources)
    quality.save_json(
        analysis_folder / "timelines/observed_timelines.json", observations
    )
    quality.save_json(
        analysis_folder / "quality/selected_pixel_bands.index.json",
        {identifier: str(path) for identifier, path in partitions.items()},
    )
    return regions, quality_rows, PixelPartitions(partitions)


def audit_source(source, analysis_folder, minimum_fraction):
    from crop_pipeline.sources.pilot_sources import validate_source_contract

    validate_source_contract(source)
    field_folder = Path(source["field_folder"])
    regions = json.loads(
        (field_folder / "sampled_download_regions.geojson").read_text()
    )["features"]
    table_mode = source.get("archive_format") == "selected_pixel_parquet_v1"
    manifests = {} if table_mode else manifest_paths(source["progress_folder"])
    all_quality, observed, partitions = [], [], {}
    for region in regions:
        identifier = region["properties"]["fragment_id"]
        region["properties"].update(
            source_dataset=source["source_name"],
            source_year=source["year"],
            dataset_revision=source["dataset_revision"],
            imagery_request_version=source["request_version"],
        )
        output_folder = Path(analysis_folder) / "quality/tiles" / identifier
        if table_mode:
            from crop_pipeline.imagery import pixel_tables
            from crop_pipeline.windows.seasons import file_sha256
            if source.get("annual_tif_index"):
                from crop_pipeline.imagery import annual_tif
                region_quality, pixels = annual_tif.audit_region(source, region, minimum_fraction)
            else:
                receipt = source["pixel_table_index"][identifier]
                if file_sha256(Path(receipt["path"])) != receipt["sha256"]:
                    raise ValueError("Pixel-table receipt changed after source registration")
                plan = pixel_tables.plan_from_receipt(receipt["path"])
                if plan["fragment"]["properties"]["sampling_grid"] != region["properties"]["sampling_grid"]:
                    raise ValueError("Pixel-table grid differs from source")
                region_quality, pixels = pixel_tables.audit_region(source["pixel_table_root"], plan, minimum_fraction)
        else:
            region_quality, pixels = quality.quality_check(
                [region],
                source["progress_folder"],
                source["imagery_folder"],
                field_folder,
                output_folder,
                source["cloud_threshold"],
                minimum_fraction,
                int(source["year"]),
                source["request_version"],
                progress_paths=manifests.get(identifier, []),
            )
        partition = output_folder / "pixel_bands.parquet"
        write_pixel_partition(partition, pixels)
        partitions[identifier] = partition
        all_quality.extend(region_quality)
        observed.extend(read_observations(source, identifier))
    quality.write_table(
        Path(analysis_folder)
        / "quality"
        / source["source_name"]
        / "composite_quality.csv",
        all_quality,
        list(all_quality[0]) if all_quality else [],
    )
    return regions, all_quality, observed, partitions


def manifest_paths(folder):
    indexed = defaultdict(list)
    for path in Path(folder).glob("*.json"):
        record = json.loads(path.read_text())
        indexed[record["fragment_id"]].append(path)
    return indexed


def read_observations(source, identifier):
    path = (
        Path(source["timeline_folder"])
        / "tiles"
        / identifier
        / "observed_timelines.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Annual daily-summary checkpoint missing: {path}")
    rows = json.loads(path.read_text())
    if any(row["region_id"] != identifier for row in rows):
        raise ValueError("Timeline region differs from its annual tile.")
    return rows


def build_partitioned_features(
    seasons, pixel_rows, quality_rows, minimum_fraction, bin_shift
):
    """Use the unchanged feature builder on each tile before global parcel-cycle selection."""
    from crop_pipeline.windows.seasons import build_features

    seasons_by_region = defaultdict(list)
    quality_by_region = defaultdict(list)
    for season in seasons:
        seasons_by_region[season["region_id"]].append(season)
    for row in quality_rows:
        quality_by_region[row["region_id"]].append(row)
    outputs = ([], [], [], [])
    for identifier, region_seasons in seasons_by_region.items():
        results = build_features(
            region_seasons,
            pixel_rows.for_regions([identifier]),
            quality_by_region[identifier],
            minimum_fraction,
            bin_shift,
        )
        for output, records in zip(outputs, results):
            output.extend(records)
    return outputs


def partitioned_timelines(quality_rows, pixel_rows, output_folder):
    grouped = defaultdict(list)
    for row in quality_rows:
        grouped[row["region_id"]].append(row)
    rows = []
    for identifier, region_quality in grouped.items():
        rows.extend(
            quality.composite_timelines(
                region_quality,
                pixel_rows.for_regions([identifier]),
                Path(output_folder) / "tiles" / identifier,
            )
        )
    quality.write_table(
        Path(output_folder) / "composite_timelines.csv",
        rows,
        [
            "region_id",
            "crop",
            "partition_group",
            "name",
            "start",
            "end",
            "date",
            "source_kind",
            "eligible",
            "valid_fraction",
            "valid_pixels",
            "NDVI",
            "NDRE",
            "NDMI",
            "status",
        ],
    )
    return rows
