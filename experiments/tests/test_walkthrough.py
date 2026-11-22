import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from experiments.walkthrough import metrics

CROPS = ("carrot", "cabbage", "potato")


def bell(peak, low=.2, high=.8, width=2.):
    t = np.arange(24)
    return low + (high - low) * np.exp(-((t - peak) ** 2) / (2 * width ** 2))


class CorrespondenceTests(unittest.TestCase):
    def test_cycle_is_contiguous_around_the_peak_and_spans_cloudy_bins(self):
        series = bell(12)[None]
        clear = np.ones((1, 24), bool)
        clear[0, 12] = False  # cloudy at the peak itself
        status = metrics.temporal_correspondence(series, clear)
        confirmed = np.nonzero(status[0] == metrics.IN_SEASON)[0]
        self.assertTrue(np.all(np.diff(confirmed[confirmed != 12]) <= 2))
        self.assertEqual(status[0, 12], metrics.UNAVAILABLE)
        self.assertEqual(status[0, 0], metrics.OFF_SEASON)
        self.assertTrue(confirmed.min() < 12 < confirmed.max())

    def test_flat_series_is_uncertain_and_missing_bins_are_unavailable(self):
        series = np.full((1, 24), .3)
        clear = np.ones((1, 24), bool)
        clear[0, :4] = False
        status = metrics.temporal_correspondence(series, clear)
        self.assertTrue(np.all(status[0, :4] == metrics.UNAVAILABLE))
        self.assertTrue(np.all(status[0, 4:] == metrics.UNCERTAIN))

    def test_second_smaller_cycle_is_not_counted_as_the_main_cycle(self):
        series = (bell(5) + bell(18, low=0, high=.35))[None]
        status = metrics.temporal_correspondence(series, np.ones((1, 24), bool))
        self.assertEqual(status[0, 5], metrics.IN_SEASON)
        self.assertEqual(status[0, 18], metrics.OFF_SEASON)

    def test_parcel_series_takes_the_median_of_clear_pixels(self):
        rows = [{"partition_group": "b"}, {"partition_group": "a"}, {"partition_group": "a"}]
        values = np.full((3, 24, 5), .1)
        values[:, :, 3] = [[.5], [.3], [.7]]  # NIR per pixel
        valid = np.ones((3, 24), bool)
        valid[2, 0] = False
        groups, series, clear = metrics.parcel_series(rows, values, valid)
        self.assertEqual(groups, ["a", "b"])
        ndvi = lambda nir: (nir - .1) / (nir + .1)
        self.assertAlmostEqual(series[0, 0], ndvi(.3))
        self.assertAlmostEqual(series[0, 1], np.median([ndvi(.3), ndvi(.7)]))
        self.assertTrue(clear.all())


class ScoreTests(unittest.TestCase):
    def test_per_crop_scores_and_confusion(self):
        truth = ["carrot", "carrot", "potato", "potato", "cabbage"]
        prediction = ["carrot", "potato", "potato", "potato", "potato"]
        scores = metrics.per_crop_scores(truth, prediction, CROPS).set_index("crop")
        self.assertEqual(scores.loc["carrot", "precision"], 1.)
        self.assertEqual(scores.loc["carrot", "recall"], .5)
        self.assertTrue(np.isnan(scores.loc["cabbage", "precision"]))
        self.assertEqual(scores.loc["cabbage", "f1"], 0.)
        self.assertAlmostEqual(scores.loc["potato", "f1"], 2 * 2 / (4 + 2))
        table = metrics.confusion(truth, prediction, list(CROPS))
        self.assertEqual(table.loc["cabbage", "potato"], 1)
        self.assertEqual(int(table.to_numpy().sum()), 5)

    def test_present_crop_f1_and_population_records(self):
        frame = pd.DataFrame({"crop": ["carrot", "carrot", "cabbage", "potato"],
                              "prediction": ["carrot", "potato", "cabbage", "potato"],
                              "test_cohort": ["california", "california", "jecam", "jecam"]})
        self.assertAlmostEqual(metrics.present_crop_f1(frame.crop[:2], frame.prediction[:2], CROPS), 2 / 3)
        records = metrics.population_records("m", frame[frame.test_cohort == "jecam"],
                                             {"jecam": 2, "california": 2, "combined": 4}, CROPS)
        by = {r["population"]: r for r in records}
        self.assertEqual(by["jecam"]["f1"], 1.)
        self.assertEqual(by["california"]["scored"], 0)
        self.assertTrue(np.isnan(by["california"]["f1"]))
        self.assertEqual(by["combined"]["total"], 4)

    def test_valid_share_and_sampling_are_reproducible(self):
        frame = pd.DataFrame({"group": ["x", "x", "y"]})
        valid = np.zeros((3, 24), bool)
        valid[0, 0] = valid[1, 0] = valid[2, 1] = True
        shares = metrics.valid_share(frame, valid, ("group",))
        self.assertEqual(shares.shape, (2, 24))
        self.assertEqual(shares.loc["x"].iloc[0], 1.)
        self.assertEqual(shares.loc["y"].iloc[0], 0.)
        labels, keys = ["a"] * 5 + ["b"] * 2, [f"k{i}" for i in range(7)]
        first = metrics.sample_per_group(labels, keys, 3)
        self.assertEqual(len(first), 5)
        np.testing.assert_array_equal(first, metrics.sample_per_group(labels, keys, 3))

    def test_maps_and_shares(self):
        codes = np.array([[0, 1, 3], [4, 5, 3]])
        cropland = np.array([[False, True, True], [True, True, False]])
        names = metrics.class_names(codes, metrics.IMPROVED_CODES, cropland)
        self.assertEqual(names[0, 0], "")
        self.assertEqual(names[1, 2], "Not active cropland")
        shares, count = metrics.confident_crop_shares(names[cropland], CROPS)
        self.assertEqual(count, 2)
        self.assertEqual(shares["potato"], .5)


class FeatureTests(unittest.TestCase):
    def test_harmonic_curve_recovers_a_sinusoid_through_gaps(self):
        t = (np.arange(24) + .5) / 24
        series = .5 + .3 * np.cos(2 * np.pi * t)
        clear = np.ones(24, bool)
        clear[[3, 4, 10]] = False
        x, curve, ok, coefficients = metrics.harmonic_curve(series, clear)
        self.assertTrue(ok)
        self.assertAlmostEqual(coefficients[0], .5, places=6)
        self.assertAlmostEqual(coefficients[1], .3, places=6)
        self.assertAlmostEqual(curve.max(), .8, places=3)

    def test_grouped_permutation_importance_finds_the_informative_family(self):
        rng = np.random.default_rng(0)
        y = np.repeat(["carrot", "potato"], 100)
        signal = (y == "carrot").astype(float)
        X = np.column_stack([signal + rng.normal(0, .1, 200), signal + rng.normal(0, .1, 200), rng.normal(size=200)])
        model = RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y)
        table, baseline = metrics.grouped_permutation_importance(model, X, y, ["VH_mean", "VH_amp1", "B3_mean"], repeats=3)
        self.assertGreater(baseline, .9)
        self.assertEqual(table.loc["VH", "columns"], 2)
        self.assertGreater(table.loc["VH", "mean"], table.loc["B3", "mean"])
        self.assertEqual(metrics.feature_family("VHVV_mean"), "VH/VV ratio")


class FigureSmokeTests(unittest.TestCase):
    def test_figures_render_from_synthetic_inputs(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from experiments.walkthrough import figures

        figures.style()
        rng = np.random.default_rng(1)
        rows, s2 = [], []
        for i in range(36):
            crop = CROPS[i % 3]
            partition = "test" if i % 4 == 0 else "train"
            rows.append({"partition_group": f"g{i}", "crop": crop, "partition": partition,
                         "source_dataset": "cropharvest" if partition == "test" else "sen4agrinet",
                         "test_cohort": "jecam" if partition == "test" else None, "site_cell": f"{i % 5},0",
                         "lon": float(i % 5), "lat": 45.})
            values = np.full((24, 5), .1)
            values[:, 3] = bell(4 + 6 * (i % 3), low=.12, high=.5)
            s2.append(values)
        frame, s2 = pd.DataFrame(rows), np.asarray(s2)
        valid = rng.random((36, 24)) > .2
        s1 = rng.normal(-15, 1, (36, 24, 2))
        drawn = [figures.crop_distribution(frame), figures.spectral_signatures(frame, s2, valid, s1),
                 figures.valid_observation_heatmap(metrics.valid_share(frame, valid, ("source_dataset", "crop")))]
        groups, series, clear = metrics.parcel_series(rows, s2, valid)
        drawn.append(figures.temporal_correspondence_heatmap(metrics.temporal_correspondence(series, clear),
                                                             np.asarray([r["crop"] for r in rows])))
        truth, prediction = frame.crop, np.roll(frame.crop.to_numpy(), 1)
        drawn.append(figures.confusion_panels([("m", metrics.confusion(truth, prediction, list(CROPS)))]))
        drawn.append(figures.precision_recall_f1([("m", metrics.per_crop_scores(truth, prediction, CROPS))]))
        records = metrics.population_records("m", frame.assign(prediction=prediction), {"jecam": 9, "combined": 36}, CROPS)
        drawn.append(figures.model_comparison(records))
        self.assertTrue(all(len(f.axes) for f in drawn))
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
