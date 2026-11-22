"""Rebuild tools for Jose's comparison notebooks: reference-audit matching rules."""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_test_reference_audit", ROOT / "tools/build_test_reference_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

SHA = "a" * 64


class ReferenceAuditTests(unittest.TestCase):
    def test_cropharvest_parcels_take_their_labels_record(self):
        labels = {("jecam", 7): {"dataset": "jecam", "index": 7, "collection_date": "2019-06-02T00:00:00"}}
        row = {"partition_group": "cropharvest|jecam|2019|jecam:7"}
        evidence = f"CropHarvest jecam label 'potato' (labels.geojson sha256:{SHA})"
        self.assertEqual(audit.raw_matches(row, evidence, labels, SHA), [labels["jecam", 7]])
        with self.assertRaises(ValueError):  # a different labels revision is refused
            audit.raw_matches(row, evidence, labels, "b" * 64)
        with self.assertRaises(ValueError):  # a parcel missing from labels.geojson is refused
            audit.raw_matches({"partition_group": "cropharvest|jecam|2019|jecam:8"}, evidence, labels, SHA)

    def test_supplements(self):
        dwr = audit.raw_matches({"partition_group": "california_dwr|US|2023|1"},
                                "DWR 2023 single crop, T6, main crop date day 184", {}, SHA)
        self.assertEqual(dwr[0]["MAIN_CROP_DATE"], 184)
        denmark = audit.raw_matches({"partition_group": "denmark_marker|DK|2023|x"}, "Marker 2023 declared cabbage", {}, SHA)
        self.assertEqual(denmark, [{"label_evidence": "Marker 2023 declared cabbage"}])


if __name__ == "__main__":
    unittest.main()
