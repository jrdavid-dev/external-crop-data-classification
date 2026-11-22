import unittest

import numpy as np

from experiments.short_cycle import short_cycle


def radar(start, rows=1, rise=6.0):
    """VH flat at -20 dB with one 8-bin cycle beginning at ``start``; VV 6 dB above VH."""
    vh = np.full((rows, 24), -20.0)
    for b in range(8):
        vh[:, (start + b) % 24] += rise * np.sin(np.pi * b / 7)
    return np.stack([vh + 6, vh], axis=-1)


class ShortCycleTests(unittest.TestCase):
    def test_window_is_placed_relative_to_the_cycle_not_the_calendar(self):
        offsets = set()
        for start in (0, 2, 10, 20):  # January, February, June, wrapping across December
            found, rise = short_cycle.cycle_start(radar(start)[..., 1])
            offsets.add((found[0] - start) % 24)
            self.assertAlmostEqual(rise[0], 6.0 * np.sin(np.pi * 3 / 7))
        self.assertEqual(len(offsets), 1)

    def test_same_cycle_in_different_months_gives_the_same_features(self):
        early, _, columns = short_cycle.s1_cycle(None, None, radar(2))
        late, ok, _ = short_cycle.s1_cycle(None, None, radar(20))
        np.testing.assert_allclose(early, late)
        self.assertTrue(ok.all())
        self.assertEqual(len(columns), 26)
        self.assertFalse(any("month" in c or "phase" in c for c in columns))

    def test_gaps_are_filled_and_sparse_pixels_rejected(self):
        values = radar(5, rows=2)
        values[0, 7, :] = np.nan          # one missing half-month: filled
        values[1, :4, :] = np.nan         # four missing: below the 22-bin rule
        X, ok, _ = short_cycle.s1_cycle(None, None, values)
        self.assertEqual(ok.tolist(), [True, False])
        self.assertTrue(np.isfinite(X[0]).all())

    def test_needs_radar(self):
        with self.assertRaises(ValueError):
            short_cycle.s1_cycle(None, None, None)


if __name__ == "__main__":
    unittest.main()
