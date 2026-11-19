import unittest
from shapely.geometry import box, mapping
from crop_pipeline.sources import adaptive, denmark, supplemental, test_cohort


class SupplementalTests(unittest.TestCase):
    def test_california_uses_only_explicit_single_crop_carrot_in_window(self):
        def feature(identifier, **changes):
            return {"geometry": mapping(box(-120, 36, -119.9, 36.1)),
                    "properties": {"UniqueID": identifier, "MAIN_CROP": "T6",
                                   "MULTIUSE": "S", "MAIN_CROP_DATE": 150, **changes}}

        rows = supplemental._california_features({
            "type": "FeatureCollection",
            "features": [feature("a"), feature("b", MULTIUSE="D"),
                         feature("c", MAIN_CROP_DATE=250), feature("d", MAIN_CROP="T4")],
        })
        self.assertEqual([r["properties"]["parcel_id"] for r in rows], ["a"])
        self.assertEqual(rows[0]["properties"]["year"], 2023)

    def test_supplement_policies_do_not_share_existing_campaigns(self):
        keys = {adaptive.policy_key(policy) for policy in [
            adaptive.DEFAULT_POLICY, test_cohort.TEST_POLICY,
            supplemental.policy_for("california_dwr"),
            supplemental.policy_for("denmark_marker"),
        ]}
        self.assertEqual(len(keys), 4)

    def test_denmark_deduplicates_geometry_even_when_field_ids_repeat(self):
        first = box(10, 55, 10.001, 55.001)
        second = box(10.01, 55, 10.011, 55.001)
        rows = [
            {"id": "3-0", "crop_code": "408", "geometry": first.wkb},
            {"id": "3-0", "crop_code": "418", "geometry": second.wkb},
            {"id": "another", "crop_code": "418", "geometry": first.reverse().wkb},
            {"id": "other", "crop_code": "409", "geometry": box(11, 55, 11.001, 55.001).wkb},
        ]
        features = denmark.cabbage_features(rows, "OGC:CRS84")
        self.assertEqual(len(features), 2)
        self.assertEqual(features, denmark.cabbage_features(reversed(rows), "OGC:CRS84"))
