import unittest

import numpy as np

from crop_pipeline.training import transfer_preprocessing as transfer


class TransferPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([0., 14., 29., 45., 60., 76., 91., 106.])
        self.row = {"partition_group": "parcel-a"}
        for i in range(8):
            for channel in transfer.CHANNELS:
                self.row[f"bin_{i + 1:02d}_{channel}"] = self.positions[i] / 120

    def test_shift_preserves_identity_and_rebuilds_shape(self):
        shifted = transfer.transform(self.row, "shift", 1, self.positions)
        self.assertEqual(shifted["partition_group"], self.row["partition_group"])
        self.assertEqual(shifted["bin_01_NDVI"], shifted["bin_02_NDVI"])
        self.assertEqual(shifted["bin_08_NDVI"], self.row["bin_07_NDVI"])
        self.assertAlmostEqual(shifted["shape_NDVI_amplitude"], self.positions[6] / 120)
        self.assertNotIn("shape_NDVI_amplitude", self.row)

    def test_centering_is_invariant_to_a_constant_offset(self):
        centered = transfer.transform(self.row, "temporal_centering", positions=self.positions)
        offset = {**self.row, **{name: self.row[name] + 0.2 for name in transfer.COLUMNS[:72]}}
        shifted = transfer.transform(offset, "temporal_centering", positions=self.positions)
        np.testing.assert_allclose([centered[n] for n in transfer.COLUMNS],
                                   [shifted[n] for n in transfer.COLUMNS], atol=1e-12)
        self.assertAlmostEqual(np.median([centered[f"bin_{i:02d}_NDVI"] for i in range(1, 9)]), 0)

    def test_repairs_ignore_hidden_values_and_keep_all_other_bins(self):
        gap = transfer.gap_bin(self.row["partition_group"]) + 1
        column = f"bin_{gap:02d}_NDVI"
        hidden = {**self.row, column: 999}
        linear = transfer.transform(hidden, "gap_interpolation", positions=self.positions)
        nearest = transfer.transform(hidden, "gap_nearest", positions=self.positions)
        self.assertAlmostEqual(linear[column], self.row[column])
        self.assertIn(nearest[column], [self.row[f"bin_{i:02d}_NDVI"] for i in (gap - 1, gap + 1)])
        self.assertEqual({key for key in transfer.COLUMNS[:72] if linear[key] != hidden[key]}, {column})
