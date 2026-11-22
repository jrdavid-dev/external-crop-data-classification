"""Weighted mixed training: add part of the external test places to training, with extra importance.

Runs on the current annual inputs (``pipeline_outputs/annual/current_inputs.json``) with the annual
recipe A predictors (Sentinel-2 + Sentinel-1 harmonics) and the annual Random Forest settings.

1. **Split by place, not by parcel.** External test parcels are grouped by place and crop; whole
   0.1-degree blocks (~10 km) go either to the *added* half (joins training) or the *held-out* half
   (``split_by_place``), never split by crop, so neighbouring fields never sit on both sides.
2. **Score only held-out blocks.** Every model is scored on the held-out half only. The original
   external scores of the Europe-trained model stay the record.
3. **Choose the weight without the held-out blocks.** Added parcels count ``w`` times a development
   parcel (``WEIGHTS``). ``w`` is chosen by block-wise cross-validation inside the added half: its
   blocks are divided into folds, each fold is predicted by a model trained on development plus the
   other folds, and the highest pooled macro-F1 wins (ties to the smaller weight). The held-out half is
   scored once per weight afterwards, for transparency; only the CV choice is "the" result.
4. **New place.** For each place, train on development plus the added half of the *other* places
   (chosen weight) and score that place's held-out half: the situation Atok is in.
5. **Several splits.** Everything is repeated for each split seed; the spread is reported.

    .venv/bin/python -m experiments.mixed_training.weighted pipeline_outputs/mixed_training/<new_folder>
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer

from crop_pipeline.common import save_json
from crop_pipeline.sources import sampling
from crop_pipeline.training import exploratory
from crop_pipeline.windows import seasons
from experiments.annual import development_views, workflow as aw
from experiments.mixed_training import mixing

RECIPE = "annual_s2_s1"
WEIGHTS = (1.0, 2.0, 5.0)
SEEDS = (1, 2, 3)
CV_FOLDS = 3
FOREST = development_views.SELECTION_FOREST  # 300 trees; every model in the comparison uses the same settings
CROPS = development_views.CROPS


def add_coordinates(rows):
    """Copies of the rows with WGS84 ``lat``/``lon`` from each pixel's ``x``/``y`` in its source CRS."""
    by_crs = defaultdict(list)
    for i, row in enumerate(rows):
        by_crs[row["source_crs"]].append(i)
    out = [dict(row) for row in rows]
    for crs, index in by_crs.items():
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = to_wgs84.transform([rows[i]["x"] for i in index], [rows[i]["y"] for i in index])
        for i, a, b in zip(index, lat, lon):
            out[i]["lat"], out[i]["lon"] = float(a), float(b)
    return out


def weighted_fit(X, rows, index, extra, weight, seed=42):
    """Forest on ``index`` with the pipeline weights; pixels of ``extra`` parcels count ``weight`` times."""
    from sklearn.ensemble import RandomForestClassifier
    chosen = [rows[i] for i in index]
    w = exploratory.normalized_training_weights(chosen, development_views.LABEL_WEIGHTS)
    w = w * np.where([r["partition_group"] in extra for r in chosen], weight, 1.0)
    w = w / w.mean()
    model = RandomForestClassifier(random_state=seed, n_jobs=-1, **FOREST)
    return model.fit(X[index], [r["crop"] for r in chosen], sample_weight=w)


def predict(model, X, rows, index):
    """Parcel predictions (mean pixel probabilities) for the rows in ``index``."""
    parcels = development_views.parcel_probabilities(model, X[index], [rows[i] for i in index])
    return {g: (crop, CROPS[int(np.argmax(p))]) for g, (crop, p) in parcels.items()}


def macro_f1(predictions):
    """Three-class macro-F1, or None when a reference crop is missing."""
    if not predictions:
        return None
    parcels = {g: (crop, np.eye(len(CROPS))[CROPS.index(pred)]) for g, (crop, pred) in predictions.items()}
    return development_views.scores(parcels)["macro_f1"]


def split_by_place(rows, fraction=.5, seed=1):
    """{parcel group: 'added' | 'held_out'}: whole 0.1-degree blocks per place, never split by crop.

    Within each place, blocks are ranked by a stable hash and added until about ``fraction`` of the
    place's parcels are in training; the rest are held out (at least one block is always held out).
    """
    blocks = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if row["partition"] == "test":
            blocks[mixing.place_of(row)][mixing.block_of(row)].add(row["partition_group"])
    assignment = {}
    for place, members in sorted(blocks.items()):
        ranked = sorted(members, key=lambda b: sampling.stable_rank(f"weighted-mixing|{seed}|{place}|{b}"))
        total = sum(len(g) for g in members.values())
        added, sides = 0, {}
        for block in ranked:
            sides[block] = "added" if added < fraction * total else "held_out"
            added += len(members[block]) if sides[block] == "added" else 0
        if "held_out" not in sides.values():
            sides[ranked[-1]] = "held_out"
        for block, side in sides.items():
            assignment.update(dict.fromkeys(members[block], side))
    return assignment


def cv_folds(rows, index, folds=CV_FOLDS, seed=1):
    """Blocks of the added half dealt into ``folds`` groups by stable rank within each place."""
    blocks = defaultdict(set)
    for i in index:
        r = rows[i]
        blocks[mixing.place_of(r), mixing.block_of(r)].add(r["partition_group"])
    by_place = defaultdict(list)
    for key in blocks:
        by_place[key[0]].append(key)
    fold_of = {}
    for place, keys in sorted(by_place.items()):
        keys.sort(key=lambda k: sampling.stable_rank(f"weighted-mixing-cv|{seed}|{k}"))
        for position, key in enumerate(keys):
            for group in blocks[key]:
                fold_of[group] = position % folds
    return fold_of


def choose_weight(X, rows, development, added, seed):
    """Block-wise CV inside the added half: pooled macro-F1 per weight, and the chosen weight."""
    fold_of = cv_folds(rows, added, seed=seed)
    scores = {}
    for weight in WEIGHTS:
        pooled = {}
        for fold in range(CV_FOLDS):
            train_added = np.asarray([i for i in added if fold_of[rows[i]["partition_group"]] != fold], dtype=int)
            target = np.asarray([i for i in added if fold_of[rows[i]["partition_group"]] == fold], dtype=int)
            if not len(target):
                continue
            extra = {rows[i]["partition_group"] for i in train_added}
            model = weighted_fit(X, rows, np.concatenate([development, train_added]), extra, weight)
            pooled.update(predict(model, X, rows, target))
        scores[weight] = macro_f1(pooled)
    valid = {w: s for w, s in scores.items() if s is not None}
    chosen = max(valid, key=lambda w: (round(valid[w], 6), -w)) if valid else 1.0
    return chosen, scores


def summarize(predictions, places, cohorts):
    """Macro-F1 on all held-out parcels, JECAM and tropical places, and accuracy per place."""
    per_place = defaultdict(list)
    for g, (crop, pred) in predictions.items():
        per_place[places[g]].append(crop == pred)
    tropical = {g: v for g, v in predictions.items() if places[g] in mixing.TROPICAL}
    return {"parcels": len(predictions), "macro_f1_all": macro_f1(predictions),
            "macro_f1_jecam": macro_f1({g: v for g, v in predictions.items() if cohorts[g] == "jecam"}),
            "macro_f1_tropical": macro_f1(tropical),
            "accuracy_by_place": {p: {"parcels": len(v), "accuracy": float(np.mean(v))}
                                  for p, v in sorted(per_place.items())}}


def run_split(rows, X, ok, seed, progress=print):
    places = {r["partition_group"]: mixing.place_of(r) for r in rows if r["partition"] == "test"}
    cohorts = {r["partition_group"]: r["test_cohort"] for r in rows if r["partition"] == "test"}
    assignment = split_by_place(rows, seed=seed)
    usable = [i for i, good in enumerate(ok) if good]
    development = np.asarray([i for i in usable if rows[i]["partition"] in ("train", "validation")], dtype=int)
    added = np.asarray([i for i in usable if assignment.get(rows[i]["partition_group"]) == "added"], dtype=int)
    held = np.asarray([i for i in usable if assignment.get(rows[i]["partition_group"]) == "held_out"], dtype=int)
    added_groups = {rows[i]["partition_group"] for i in added}
    if added_groups & {rows[i]["partition_group"] for i in held}:
        raise ValueError("A parcel is on both sides of the split")
    added_blocks = {(places[rows[i]["partition_group"]], mixing.block_of(rows[i])) for i in added}
    held_blocks = {(places[rows[i]["partition_group"]], mixing.block_of(rows[i])) for i in held}
    chosen, cv = choose_weight(X, rows, development, added, seed)
    if progress:
        progress(f"split {seed}: CV macro-F1 by weight {cv}; chosen weight {chosen}")
    results = {"europe_only": predict(weighted_fit(X, rows, development, set(), 1.0), X, rows, held)}
    for weight in WEIGHTS:
        model = weighted_fit(X, rows, np.concatenate([development, added]), added_groups, weight)
        results[f"europe_plus_added_w{weight:g}"] = predict(model, X, rows, held)
    new_place = {}
    for place in sorted({places[rows[i]["partition_group"]] for i in held}):
        others = np.asarray([i for i in added if places[rows[i]["partition_group"]] != place], dtype=int)
        target = np.asarray([i for i in held if places[rows[i]["partition_group"]] == place], dtype=int)
        extra = {rows[i]["partition_group"] for i in others}
        model = weighted_fit(X, rows, np.concatenate([development, others]), extra, chosen)
        new_place.update(predict(model, X, rows, target))
    results["new_place_chosen_weight"] = new_place
    if progress:
        progress(f"split {seed}: scored {len(results)} models on {len({rows[i]['partition_group'] for i in held})} "
                 "held-out parcels")
    summary = [{"split_seed": seed, "model": name, "chosen_weight": chosen,
                **summarize(p, places, cohorts)} for name, p in results.items()]
    split = Counter((places[g], side) for g, side in assignment.items())
    return {"seed": seed, "chosen_weight": chosen, "cv_macro_f1_by_weight": cv,
            "shared_blocks": len(added_blocks & held_blocks), "summary": summary,
            "split": [{"split_seed": seed, "place": p, "side": s, "parcels": n} for (p, s), n in sorted(split.items())],
            "predictions": [{"split_seed": seed, "model": name, "partition_group": g, "place": places[g],
                             "crop": c, "prediction": pr} for name, p in results.items() for g, (c, pr) in p.items()]}


def run(root, output, progress=print):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs, _ = aw._input(root, None, "inputs")
    rows, arrays = aw.load_inputs(inputs.parent)
    X, ok, _ = aw.recipe_matrix(RECIPE, arrays)
    X = np.nan_to_num(X)
    rows = add_coordinates(rows)
    splits = [run_split(rows, X, ok, seed, progress) for seed in SEEDS]
    summary = [s for r in splits for s in r["summary"]]
    seasons.write_records(output / "summary.csv", [{k: v for k, v in s.items() if k != "accuracy_by_place"} for s in summary])
    seasons.write_records(output / "accuracy_by_place.csv", [
        {"split_seed": s["split_seed"], "model": s["model"], "place": p, **v}
        for s in summary for p, v in s["accuracy_by_place"].items()])
    seasons.write_records(output / "split.csv", [row for r in splits for row in r["split"]])
    seasons.write_records(output / "held_out_predictions.csv", [row for r in splits for row in r["predictions"]])
    save_json(output / "experiment.json", {
        "inputs_handoff": str(inputs), "recipe": RECIPE, "forest": FOREST, "weights": list(WEIGHTS),
        "seeds": list(SEEDS), "cv_folds": CV_FOLDS, "split": "whole 0.1-degree blocks per place, about half of each place added",
        "weight_choice": {r["seed"]: {"chosen": r["chosen_weight"], "cv_macro_f1": r["cv_macro_f1_by_weight"]}
                          for r in splits},
        "blocks_on_both_sides": {r["seed"]: r["shared_blocks"] for r in splits}})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print("Output:", run(Path(__file__).resolve().parents[2], args.output))


if __name__ == "__main__":
    main()
