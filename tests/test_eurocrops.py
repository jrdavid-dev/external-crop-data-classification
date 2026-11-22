import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import geopandas as gpd
import pyarrow as arrow
import pyarrow.parquet as parquet
from shapely.geometry import box

from crop_pipeline.sources import adaptive, eurocrops, test_cohort
from crop_pipeline.training.source_holdout import checked_rows


class EuroCropsTests(unittest.TestCase):
    def test_production_parcels_plan_as_development_and_cannot_be_test_input(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "NL_2021.parquet"
            geometry = box(5.01, 52, 5.02, 52.01)
            records = arrow.table({
                "cropfield": [1, 2, 3, 3],
                "original_code": ["2785", "2786", "2717", "1"],
                "geometry": [geometry.wkb] * 4,
            }).replace_schema_metadata({b"geo": json.dumps({
                "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}}
            }).encode()})
            parquet.write_table(records, path)
            features = eurocrops.load_carrots(path)
            self.assertEqual([item["properties"]["parcel_id"] for item in features], ["1"])
            [(plan, context)] = adaptive.prepare_round_batches(
                eurocrops.SOURCE, features, root / "archive", root / "inventory", {},
                candidate_crops=("carrot",),
            )
            self.assertEqual(plan["estimate"]["parcels"], 1)
            self.assertEqual(context["campaign"], str(adaptive.root(root / "inventory")))
            self.assertNotIn(eurocrops.SOURCE, test_cohort.REQUIRED_DEVELOPMENT)
            rows = [{"source_dataset": eurocrops.SOURCE, "partition": "train"}]
            self.assertEqual(checked_rows(rows, test=False), rows)
            with self.assertRaisesRegex(ValueError, "Unexpected source"):
                checked_rows(rows, test=True)

    def test_dutch_2020_archive_uses_declared_production_and_year_scoped_ids(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            df = gpd.GeoDataFrame(
                {"objectid": [1, 2], "gewascode": ["2783", "2784"]},
                geometry=[box(5.01, 52, 5.02, 52.01), box(5.03, 52, 5.04, 52.01)],
                crs="EPSG:4326",
            )
            df.to_file(root / "NL_2020_EC21.shp")
            archive = root / "NL_2020.zip"
            with ZipFile(archive, "w") as zipped:
                for member in root.glob("NL_2020_EC21.*"):
                    zipped.write(member, member.name)
            [feature] = eurocrops.load_carrots(archive)
            properties = feature["properties"]
            self.assertEqual(properties["partition_group"], "eurocrops|NL|2020|1")
            self.assertEqual(adaptive.physical_key(eurocrops.SOURCE, properties), "eurocrops|NL|2020:1")
            self.assertEqual(properties["source_class_code"], "2783")
