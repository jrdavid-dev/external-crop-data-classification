import unittest

import numpy as np

from experiments.pilots import climate, clustering, data, earth_engine, evaluation, features


def series_rows(n_per_crop=40, sources=("sen4agrinet", "francecrops"), shift=0):
    """Synthetic annual NDVI-like curves; each crop peaks at a different half-month."""
    rng = np.random.default_rng(0)
    rows, curves = [], []
    t = (np.arange(24) + .5) / 24
    for level, crop in enumerate(evaluation.CROPS):
        for i in range(n_per_crop):
            source = sources[i % len(sources)]
            partition = "train" if i % 4 else "validation"
            group = f"{crop}-{i}"
            peak = (.3 + .15 * level + shift) % 1
            curve = .2 + .5 * np.exp(-((t - peak) ** 2) / .01) + rng.normal(0, .02, 24)
            for _ in range(2):
                rows.append({"partition_group": group, "crop": crop, "partition": partition, "source_dataset": source,
                             "label_status": "reported", "site_cell": f"45,{i % 10}"})
                curves.append(curve)
    return rows, np.asarray(curves)


class SubsetTests(unittest.TestCase):
    def test_bin_index_and_sampling_keep_partitions(self):
        self.assertEqual(data.bin_index("2022_01_01_15"), 0)
        self.assertEqual(data.bin_index("2022_05_16_end"), 9)
        regions = [{"properties": {"partition_group": f"{p}{i}", "partition": p, "crop": "carrot",
                                   "source_dataset": "s", "test_cohort": "jecam" if p == "test" else None}}
                   for p in ("train", "validation", "test") for i in range(10)]
        regions.append({"properties": {"partition_group": "supp", "partition": "test", "crop": "carrot",
                                       "source_dataset": "s", "test_cohort": "denmark"}})
        chosen = data.sample_parcels(regions, {"train": 3, "validation": 2, "test": None})
        counts = {p: sum(r["properties"]["partition"] == p for r in chosen) for p in ("train", "validation", "test")}
        self.assertEqual(counts, {"train": 3, "validation": 2, "test": 10})
        self.assertEqual(chosen, data.sample_parcels(regions, {"train": 3, "validation": 2, "test": None}))


class HarmonicTests(unittest.TestCase):
    def test_fit_recovers_coefficients_and_requires_spread_clear_bins(self):
        t = (np.arange(24) + .5) / 24
        truth = np.array([.4, .2, -.1, .05, .03])
        y = truth[0] + truth[1] * np.cos(2 * np.pi * t) + truth[2] * np.sin(2 * np.pi * t) \
            + truth[3] * np.cos(4 * np.pi * t) + truth[4] * np.sin(4 * np.pi * t)
        valid = np.ones((3, 24), bool)
        valid[1, ::2] = False          # 12 clear bins, gap 1: fitted
        valid[2, :10] = False          # 10-bin gap: rejected
        coefficients, ok = features.fit_harmonics(np.tile(y, (3, 1)), valid)
        np.testing.assert_allclose(coefficients[0], truth, atol=1e-9)
        np.testing.assert_allclose(coefficients[1], truth, atol=1e-9)
        self.assertEqual(ok.tolist(), [True, True, False])
        self.assertEqual(features.longest_gap(valid).tolist(), [0, 1, 10])

    def test_invariant_terms_do_not_change_when_the_season_shifts(self):
        t = (np.arange(24) + .5) / 24
        curves = [.3 + .3 * np.cos(2 * np.pi * (t - d)) + .1 * np.cos(4 * np.pi * (t - d) + 1) for d in (0, .25)]
        coefficients, _ = features.fit_harmonics(np.asarray(curves), np.ones((2, 24), bool))
        invariant = features.harmonic_terms(coefficients, invariant=True)
        full = features.harmonic_terms(coefficients)
        np.testing.assert_allclose(invariant[0], invariant[1], atol=1e-9)
        self.assertFalse(np.allclose(full[0], full[1]))

    def test_fixed_interval_needs_all_eight_bins(self):
        cube = np.random.default_rng(0).uniform(.1, .3, (2, 24, 9))
        valid = np.ones((2, 24), bool)
        valid[1, 10] = False
        X, ok, columns = features.fixed_interval(cube, valid)
        self.assertEqual(X.shape, (2, 72))
        self.assertEqual(ok.tolist(), [True, False])
        self.assertEqual(columns[0], "bin_01_B3")
        self.assertEqual(sorted(features.column_groups(["NDVI_mean", "NDVI_amp1", "VV_mean"])), ["NDVI", "VV"])


class EvaluationTests(unittest.TestCase):
    def test_protocol_scores_harmonics_and_selection_prefers_informative_groups(self):
        rows, curves = series_rows()
        cube = curves[:, :, None]
        X, ok, columns = features.harmonic_features(cube, np.ones(curves.shape, bool), names=("NDVI",))
        noise = np.random.default_rng(1).normal(size=(len(X), 7))
        X = np.hstack([X, noise])
        columns += [f"NOISE_{i}" for i in range(7)]
        params = {**evaluation.PARAMS, "n_estimators": 30}
        self.assertGreater(evaluation.spatial_cv(X, rows, ok, params)["macro_f1"], .8)
        self.assertGreater(evaluation.random_split(X, rows, ok, params)["macro_f1"], .8)
        holdout = evaluation.dataset_holdout(X, rows, ok, params)
        self.assertEqual({r["status"] for r in holdout}, {"scored"})
        groups = {"NDVI": list(range(7)), "NOISEA": list(range(7, 10)), "NOISEB": list(range(10, 14))}
        selected, history = evaluation.forward_group_selection(X, rows, ok, groups, params, progress=None)
        self.assertIn("NDVI", selected)
        self.assertTrue(history)

    def test_zero_weight_rows_are_left_out(self):
        rows, curves = series_rows(n_per_crop=8)
        weights = np.ones(len(rows))
        weights[[r["crop"] == "potato" for r in rows]] = 0
        model = evaluation.fit(curves, rows, {**evaluation.PARAMS, "n_estimators": 5}, weights=weights)
        self.assertEqual(sorted(model.classes_), ["cabbage", "carrot"])


class ClimateAndClusterTests(unittest.TestCase):
    def test_descriptors_and_nearest_share(self):
        warm = climate.descriptors([10] * 12)
        self.assertAlmostEqual(warm["gdd_annual"], 5 * 365)
        self.assertIsNone(climate.descriptors([10] * 11))
        rows = [{"partition_group": g, "crop": c} for g, c in (("a", "carrot"), ("b", "carrot"), ("c", "potato"), ("d", "potato"))]
        weights = climate.nearest_share_weights(rows, {"a": 0., "b": 5., "c": 3., "d": 1.}, np.ones(4), keep=.5)
        self.assertEqual((weights > 0).tolist(), [True, False, False, True])
        distances = climate.distances([warm, climate.descriptors([20] * 12)], [warm])
        self.assertAlmostEqual(distances[0], 0)

    def test_gmm_finds_separated_blobs_stably(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(c, .1, (100, 2)) for c in ((0, 0), (3, 3), (0, 3))])
        model, scaler, table = clustering.select_gmm(X, components=range(1, 6))
        self.assertEqual(model.n_components, 3)
        mean, low = clustering.stability(X, 3, scaler, repeats=3)
        self.assertGreater(low, .9)

    def test_half_months_and_radar_ratio(self):
        bins = earth_engine.half_months(2024)
        self.assertEqual(len(bins), 24)
        self.assertEqual(bins[-1][1].isoformat(), "2025-01-01")
        cube = earth_engine.s1_features_cube(np.array([[[-10., -18.]]]))
        self.assertEqual(cube[0, 0].tolist(), [-10., -18., -8.])



class SecondRoundTests(unittest.TestCase):
    def test_amplitude_forms_and_which_phase_terms_survive_a_season_shift(self):
        t = (np.arange(24) + .5) / 24
        curves = [.3 + .3 * np.cos(2 * np.pi * (t - d)) + .1 * np.cos(4 * np.pi * (t - d) + 1) for d in (0, .25)]
        coefficients, _ = features.fit_harmonics(np.asarray(curves), np.ones((2, 24), bool))
        for form in ("amplitude", "invariant"):
            terms = features.harmonic_terms(coefficients, form=form)
            np.testing.assert_allclose(terms[0], terms[1], atol=1e-9)
        naive = features.harmonic_terms(coefficients, form="phase_difference")
        self.assertFalse(np.allclose(naive[0], naive[1]))
        X, ok, columns = features.harmonic_features(np.asarray(curves)[:, :, None], np.ones((2, 24), bool),
                                                    names=("EVI2",), form="amplitude")
        self.assertEqual(columns, ["EVI2_mean", "EVI2_amp1", "EVI2_amp2"])
        with self.assertRaises(ValueError):
            features.harmonic_terms(coefficients, form="unknown")

    def test_evi2_sar_ratios_and_temporal_zscore(self):
        values = np.zeros((1, 1, 5))
        values[..., data.BANDS.index("B4")], values[..., data.BANDS.index("B8")] = .1, .4
        self.assertAlmostEqual(float(features.evi2(values)[0, 0]), 2.5 * .3 / (.4 + .24 + 1))
        sar = features.sar_variables(np.array([[[-10., -20.]]]))
        np.testing.assert_allclose(sar[0, 0], [-10, -20, -10, .1 / 1.1])
        cube = np.arange(24, dtype=float).reshape(1, 24, 1)
        z = features.temporal_zscore(cube, np.ones((1, 24), bool))
        self.assertAlmostEqual(float(np.nanmean(z)), 0, places=9)
        self.assertAlmostEqual(float(np.nanstd(z)), 1, places=9)

    def test_outlier_filter_pcib_and_logratio_assignment(self):
        rng = np.random.default_rng(0)
        Z = np.vstack([rng.normal(size=(500, 4)), [[40, 40, 40, 40]]])
        keep, literal = clustering.mahalanobis_keep(Z)
        self.assertFalse(keep[-1])
        self.assertGreater(keep.mean(), .95)
        self.assertLess(literal, keep.mean())
        _, scores = clustering.principal_components(Z[keep], 3)
        self.assertLessEqual(len(set(clustering.isometric_bins(scores, bins=3))), 9)
        assignment, loss, near, total = clustering.logratio_assign({0: 40, 1: 20, 2: 40}, {"a": .4, "b": .2, "c": .4})
        self.assertAlmostEqual(loss, 0)
        self.assertEqual(assignment[1], "b")
        self.assertEqual(near, 2)  # clusters 0 and 2 have equal size, so they can swap
        _, _, near_other, _ = clustering.logratio_assign({0: 40, 1: 20, 2: 40, 3: 30}, {"a": .4, "b": .2, "c": .4}, other=True)
        self.assertGreaterEqual(near_other, 2)



class AugmentationTests(unittest.TestCase):
    def test_copies_are_used_for_training_rows_only(self):
        rows, curves = series_rows(n_per_crop=12)
        ok = np.ones(len(rows), bool)
        params = {**evaluation.PARAMS, "n_estimators": 20}
        garbage = curves.copy()
        validation = np.asarray([r["partition"] == "validation" for r in rows])
        garbage[validation] = 1e6
        clean = evaluation.random_split(curves, rows, ok, params, copies=[curves])
        dirty = evaluation.random_split(curves, rows, ok, params, copies=[garbage])
        self.assertEqual(clean["confusion_matrix"], dirty["confusion_matrix"])
        model = evaluation.fit_rows(curves, rows, np.arange(6), params, copies=[curves, curves])
        self.assertEqual(model.n_features_in_, 24)

    def test_shift_and_gain_copies(self):
        values = np.arange(24, dtype=float).reshape(1, 24, 1).repeat(5, axis=2)
        valid = np.ones((1, 24), bool)
        shifted, _ = features.shift_series(values, valid, 1)
        self.assertEqual(shifted[0, :2, 0].tolist(), [23, 0])
        copies = features.augmented_copies(lambda v, m: v[:, :, 0], values, valid, shifts=(-1, 1), gain=.1)
        self.assertEqual(len(copies), 3)
        self.assertFalse(np.allclose(copies[2], values[:, :, 0]))


class PaoayFilterTests(unittest.TestCase):
    def test_omega_fit_recovers_its_own_frequency(self):
        t = (np.arange(24) + .5) / 24
        y = .4 + .2 * np.cos(2 * np.pi * 1.5 * t) - .1 * np.sin(2 * np.pi * 3 * t)
        coefficients, ok = features.fit_harmonics(y[None], np.ones((1, 24), bool), omega=1.5)
        np.testing.assert_allclose(coefficients[0], [.4, .2, 0, 0, -.1], atol=1e-9)
        X, _, columns = features.harmonic_features(np.tile(y, (1, 1))[:, :, None], np.ones((1, 24), bool),
                                                   names=("NDVI",), omega=1.5)
        self.assertEqual(X.shape, (1, 7))

    def test_neighbourhood_median_removes_a_single_noisy_pixel_within_the_fragment(self):
        series = {}
        for i in range(3):
            for j in range(3):
                values = np.full((24, 5), .2)
                if (i, j) == (1, 1):
                    values[:] = .9
                series[("EPSG:32631", 10. * i, 10. * j)] = {"values": values, "valid": np.ones(24, bool)}
        series[("EPSG:32631", 10., 10.)]["valid"][0] = False
        smoothed = data.neighbourhood_median(series)
        centre = smoothed[("EPSG:32631", 10., 10.)]
        np.testing.assert_allclose(centre["values"], .2)
        self.assertTrue(centre["valid"].all())
        corner = smoothed[("EPSG:32631", 0., 0.)]
        self.assertTrue(np.isfinite(corner["values"]).all())

    def test_grid_median_and_seasonality_ratio(self):
        cube = np.full((24, 5, 3, 3), .2)
        cube[:, :, 1, 1] = .9
        cube[3, :, 0, 0] = np.nan
        smoothed = features.grid_median(cube)
        np.testing.assert_allclose(smoothed[:, :, 1, 1], .2)
        self.assertTrue(np.isfinite(smoothed[3, :, 0, 0]).all())
        t = (np.arange(24) + .5) / 24
        seasonal = np.zeros((1, 24, 5)); evergreen = np.zeros((1, 24, 5))
        red, nir = data.BANDS.index("B4"), data.BANDS.index("B8")
        seasonal[..., red], seasonal[..., nir] = .1, .2 + .2 * (1 + np.cos(2 * np.pi * t))
        evergreen[..., red], evergreen[..., nir] = .03, .4
        ratio, ok = features.seasonality_ratio(np.vstack([seasonal, evergreen]), np.ones((2, 24), bool))
        self.assertTrue(ok.all())
        self.assertGreater(ratio[0], .2)
        self.assertLess(ratio[1], 1e-6)


if __name__ == "__main__":
    unittest.main()
