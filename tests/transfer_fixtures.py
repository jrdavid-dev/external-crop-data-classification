"""Synthetic cached external inputs shared by transfer-analysis tests."""
from pathlib import Path

from crop_pipeline.training import external_combined as external
from test_external_combined import source_tables


def cached_v1_folder(folder):
    folder = Path(folder)
    inputs = folder / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    tables = source_tables(parcels_per_crop=12, pixels=4)
    for name, table in tables.items():
        table.to_csv(inputs / f'{name}.csv', index=False)
    data = external.add_derived_features(external.harmonize(tables))
    data, _ = external.link_split_units(data)
    data = external.add_spatial_blocks(data)
    external.lock_test_split(data, folder / 'test_split.csv')
    return folder
