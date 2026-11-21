import unittest
from unittest.mock import patch

import numpy as np

from model_aza.improved import mixing
from model_aza.pilots import evaluation

FAST = {**evaluation.PARAMS, "n_estimators": 10}


def synthetic():
    """Development rows plus test parcels in two places, four 0.1-degree blocks per place and crop."""
    rng = np.random.default_rng(0)
    rows, X = [], []
    for level, crop in enumerate(mixing.CROPS):
        for i in range(12):
            group = f"dev-{crop}-{i}"
            for _ in range(2):
                rows.append({"partition_group": group, "crop": crop, "partition": "train" if i % 4 else "validation",
                             "source_dataset": "sen4agrinet", "label_status": "reported", "lat": 45., "lon": 2.})
                X.append([level + rng.normal(0, .1), rng.normal()])
        for place, (lat, lon) in {"Madagascar": (-19.05, 47.05), "France": (48.55, 2.55)}.items():
            for block in range(4):
                for i in range(2):
                    group = f"{place}-{crop}-{block}-{i}"
                    for _ in range(2):
                        rows.append({"partition_group": group, "crop": crop, "partition": "test",
                                     "source_dataset": "cropharvest", "label_status": "reported",
                                     "lat": lat + .1 * block, "lon": lon})
                        X.append([level + rng.normal(0, .1), rng.normal()])
    return rows, np.asarray(X), np.ones(len(rows), bool)


class MixingTests(unittest.TestCase):
    def test_places_and_blocks(self):
        self.assertEqual(mixing.place_of({"lat": -21.1, "lon": 55.5}), "Reunion")
        self.assertEqual(mixing.place_of({"lat": 15.5, "lon": -16.9}), "Senegal")
        self.assertTrue(mixing.place_of({"lat": -23.5, "lon": -49.1}).startswith("other"))
        self.assertEqual(mixing.block_of({"lat": 48.57, "lon": 2.53}), mixing.block_of({"lat": 48.51, "lon": 2.59}))

    def test_split_keeps_blocks_whole_and_both_sides_filled(self):
        rows, _, _ = synthetic()
        assignment = mixing.split_external(rows, seed=3)
        self.assertEqual(assignment, mixing.split_external(rows, seed=3))
        self.assertFalse(any(g.startswith("dev") for g in assignment))
        sides = {}
        for r in rows:
            if r["partition"] == "test":
                key = (mixing.place_of(r), r["crop"], mixing.block_of(r))
                self.assertEqual(sides.setdefault(key, assignment[r["partition_group"]]), assignment[r["partition_group"]])
        for place in ("Madagascar", "France"):
            for crop in mixing.CROPS:
                found = {side for (p, c, _), side in sides.items() if (p, c) == (place, crop)}
                self.assertEqual(found, {"added", "held_out"})
        single = [r for r in rows if r["partition"] != "test" or r["partition_group"].endswith("-0-0")]
        self.assertEqual(set(mixing.split_external(single).values()), {"held_out"})

    def test_models_never_train_on_held_out_parcels_and_new_place_excludes_the_place(self):
        rows, X, ok = synthetic()
        assignment = mixing.split_external(rows, seed=1)
        held = {g for g, side in assignment.items() if side == "held_out"}
        fitted = []
        original = evaluation.fit_rows

        def record(X_, rows_, index, *args, **kwargs):
            fitted.append({rows_[i]["partition_group"] for i in index})
            return original(X_, rows_, index, *args, **kwargs)

        with patch.object(evaluation, "fit_rows", side_effect=record):
            summaries, predictions, split = mixing.run_split(rows, X, ok, seed=1, params=FAST, progress=None)
        self.assertTrue(all(not (groups & held) for groups in fitted))
        self.assertEqual({s["model"] for s in summaries}, {"europe_only", "europe_plus_added", "added_only", "new_place"})
        self.assertEqual({p["partition_group"] for p in predictions}, held)
        # Two places -> two new-place fits; each excludes its own place's added parcels.
        for groups, place in zip(fitted[-2:], ("France", "Madagascar")):
            self.assertFalse(any(g.startswith(place) for g in groups))
            self.assertTrue(any(g.startswith("dev") for g in groups))
        self.assertEqual(sum(s["parcels"] for s in split), len(assignment))
        europe = next(s for s in summaries if s["model"] == "europe_only")
        self.assertEqual(set(europe["recall"]), {"Madagascar", "France"})

    def test_fit_on_selects_places(self):
        rows, X, ok = synthetic()
        with patch.object(evaluation, "fit_rows", return_value="model") as fit:
            mixing.fit_on(rows, X, ok, lambda place: place in ("development", "France"), FAST)
        groups = {rows[i]["partition_group"] for i in fit.call_args.args[2]}
        self.assertTrue(any(g.startswith("dev") for g in groups) and any(g.startswith("France") for g in groups))
        self.assertFalse(any(g.startswith("Madagascar") for g in groups))


if __name__ == "__main__":
    unittest.main()
