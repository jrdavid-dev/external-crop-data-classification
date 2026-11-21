"""Experiment: move part of the external test places into training.

The external test parcels (JECAM and every supplement) are grouped by place
(Madagascar, Senegal, Reunion, Antilles, France, Denmark, California). Within
each place and crop, ~10 km blocks (0.1 degree cells) are ranked with the
pipeline's stable hash and assigned whole to either the *added* half (joins
training) or the *held-out* half (scored only). Neighbouring fields therefore
never sit on both sides. Several split seeds show how much the result depends
on which blocks were drawn.

Models compared on the same held-out half:

* ``europe_only``: the improved pipeline's development parcels (train + validation).
* ``europe_plus_added``: development plus the added half.
* ``added_only``: the added half alone.
* ``new_place``: for each place, development plus the added half of the *other*
  places; scored on that place's held-out half. This imitates applying the model
  to a place it has never seen, as with Atok.

The improved pipeline's selected feature set is reused unchanged; nothing here
chooses features or settings, so the held-out half stays a fair comparison.
The original test results remain the final scores of the Europe-trained model.
"""

from collections import Counter, defaultdict
from math import floor

import numpy as np

from model_aza.pilots import evaluation
from crop_pipeline.sources import sampling
from model_aza.training import transfer

CROPS = evaluation.CROPS
FEATURES = "s2_harmonic_all_radar"
PLACES = (  # name, latitude range, longitude range
    ("Madagascar", (-26, -11), (43, 51)),
    ("Senegal", (12, 17), (-18, -11)),
    ("Reunion", (-22, -20.5), (55, 56)),
    ("Antilles", (14, 17), (-63, -60)),
    ("France", (41, 52), (-5, 9)),
    ("Denmark", (54, 58), (7, 13)),
    ("California", (30, 42), (-125, -110)),
)
TROPICAL = ("Madagascar", "Senegal", "Reunion", "Antilles")


def place_of(row):
    for name, (lat0, lat1), (lon0, lon1) in PLACES:
        if lat0 < row["lat"] < lat1 and lon0 < row["lon"] < lon1:
            return name
    return f"other {floor(row['lat'])},{floor(row['lon'])}"


def block_of(row, size=.1):
    return floor(row["lat"] / size), floor(row["lon"] / size)


def split_external(rows, fraction=.5, seed=1):
    """{parcel group: 'added' | 'held_out'} for external test parcels, whole blocks per (place, crop)."""
    blocks = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if row["partition"] == "test":
            blocks[place_of(row), row["crop"]][block_of(row)].add(row["partition_group"])
    assignment = {}
    for (place, crop), members in sorted(blocks.items()):
        ranked = sorted(members, key=lambda b: sampling.stable_rank(f"mixing|{seed}|{place}|{crop}|{b}"))
        total = sum(len(g) for g in members.values())
        added, sides = 0, {}
        for block in ranked:
            sides[block] = "added" if added < fraction * total else "held_out"
            added += len(members[block]) if sides[block] == "added" else 0
        if "held_out" not in sides.values():  # always keep something to test (a single block stays held out)
            sides[ranked[-1]] = "held_out"
        for block, side in sides.items():
            assignment.update(dict.fromkeys(members[block], side))
    return assignment


def _summary(predictions, places):
    """Macro-F1 on all, tropical and non-France held-out parcels, plus recall per place and crop."""
    def macro(chosen):
        if not chosen:
            return None
        metrics = transfer.scores(chosen)
        return metrics["macro_f1"] if all(metrics["per_crop"][c]["support"] for c in CROPS) else None

    per_place = defaultdict(lambda: defaultdict(list))
    for p in predictions:
        per_place[places[p["partition_group"]]][p["crop"]].append(p["prediction"] == p["crop"])
    metrics = transfer.scores(predictions) if predictions else None
    return {
        "parcels": len(predictions),
        "macro_f1_all": macro(predictions),
        "macro_f1_tropical": macro([p for p in predictions if places[p["partition_group"]] in TROPICAL]),
        "macro_f1_without_france": macro([p for p in predictions if places[p["partition_group"]] != "France"]),
        **{f"{c}_f1": metrics["per_crop"][c]["f1"] for c in CROPS},
        "recall": {place: {crop: {"parcels": len(v), "recall": float(np.mean(v))} for crop, v in crops.items()}
                   for place, crops in sorted(per_place.items())},
    }


def run_split(rows, X, ok, seed, params=evaluation.PARAMS, model_seed=42, progress=print):
    """All four models for one split seed; returns (summaries, held-out predictions, split counts)."""
    places = {r["partition_group"]: place_of(r) for r in rows if r["partition"] == "test"}
    assignment = split_external(rows, seed=seed)
    usable = [i for i, good in enumerate(ok) if good]
    development = np.asarray([i for i in usable if rows[i]["partition"] in ("train", "validation")])
    added = np.asarray([i for i in usable if assignment.get(rows[i]["partition_group"]) == "added"])
    held = np.asarray([i for i in usable if assignment.get(rows[i]["partition_group"]) == "held_out"])
    if {rows[i]["partition_group"] for i in added} & {rows[i]["partition_group"] for i in held}:
        raise ValueError("A parcel is on both sides of the split")
    held_rows = [rows[i] for i in held]

    def score(train_index, index=held, chosen_rows=held_rows):
        model = evaluation.fit_rows(X, rows, train_index, params, model_seed)
        return evaluation.parcel_predictions(model, X[index], chosen_rows)

    predictions = {"europe_only": score(development),
                   "europe_plus_added": score(np.concatenate([development, added]))}
    if {rows[i]["crop"] for i in added} == set(CROPS):
        predictions["added_only"] = score(added)
    new_place = []
    for place in sorted({places[r["partition_group"]] for r in held_rows}):
        others = np.asarray([i for i in added if places[rows[i]["partition_group"]] != place], dtype=int)
        target = np.asarray([i for i in held if places[rows[i]["partition_group"]] == place], dtype=int)
        new_place.extend(score(np.concatenate([development, others]), target, [rows[i] for i in target]))
    predictions["new_place"] = new_place
    if progress:
        progress(f"split {seed}: {len(added)} added pixels, {len(held)} held-out pixels")
    summaries = [{"split_seed": seed, "model": name, **_summary(p, places)} for name, p in predictions.items()]
    counts = Counter((places[g], side) for g, side in assignment.items())
    split = [{"split_seed": seed, "place": place, "side": side, "parcels": n} for (place, side), n in sorted(counts.items())]
    flat = [{"split_seed": seed, "model": name, "place": places[p["partition_group"]], **p}
            for name, ps in predictions.items() for p in ps]
    return summaries, flat, split


def fit_on(rows, X, ok, include, params, seed=42):
    """Model trained on the development parcels plus external test parcels whose place passes ``include``."""
    index = np.asarray([i for i, good in enumerate(ok) if good and (
        (include("development") and rows[i]["partition"] in ("train", "validation"))
        or (rows[i]["partition"] == "test" and include(place_of(rows[i]))))])
    return evaluation.fit_rows(X, rows, index, params, seed)


# Stages (outputs under pipeline_outputs/mixed_training/; the improved pipeline's pointers are untouched)

def _new_run(repository, prefix):
    from uuid import uuid4
    from pathlib import Path
    parent = Path(repository).resolve() / "pipeline_outputs" / "mixed_training"
    parent.mkdir(parents=True, exist_ok=True)
    run = parent / f"{prefix}_{uuid4().hex[:12]}"
    run.mkdir()
    return run


def load_inputs(repository, inputs_handoff=None):
    """The improved pipeline's verified inputs and the selected feature matrix."""
    from model_aza.improved import workflow
    from model_aza.pilots import data
    receipt, _ = workflow._input(repository, inputs_handoff, "inputs")
    rows, arrays = data.load(receipt.parent)
    X, ok, columns = workflow.CANDIDATES[FEATURES](arrays["s2_values"], arrays["s2_valid"], arrays["s1_values"])
    return receipt, rows, np.nan_to_num(X), ok


def run_experiment(repository, seeds=(1, 2, 3), inputs_handoff=None, params=evaluation.PARAMS, progress=print):
    from crop_pipeline.common import save_json
    from model_aza.improved import workflow
    from crop_pipeline.windows import seasons
    receipt, rows, X, ok = load_inputs(repository, inputs_handoff)
    summaries, predictions, splits = [], [], []
    for seed in seeds:
        s, p, sp = run_split(rows, X, ok, seed, params, progress=progress)
        summaries += s
        predictions += p
        splits += sp
    run = _new_run(repository, "experiment")
    seasons.write_records(run / "split.csv", splits)
    seasons.write_records(run / "held_out_predictions.csv", predictions)
    seasons.write_records(run / "summary.csv", [{k: v for k, v in s.items() if k != "recall"} for s in summaries])
    seasons.write_records(run / "recall_by_place.csv", [
        {"split_seed": s["split_seed"], "model": s["model"], "place": place, "crop": crop, **value}
        for s in summaries for place, crops in s["recall"].items() for crop, value in crops.items()])
    save_json(run / "experiment.json", {"features": FEATURES, "params": params, "seeds": list(seeds),
                                        "split": "0.1 degree blocks per (place, crop), half added to training",
                                        "inputs_handoff": str(receipt)})
    return workflow.seal(run, "mixed_training_experiment", inputs_handoff=str(receipt))


MAP_MODELS = {
    "europe_plus_all_test_places": lambda place: True,
    "test_places_only": lambda place: place != "development",
}


def paoay_maps(repository, inputs_handoff=None, params=None, ee=None, progress=print):
    """Atok maps from models trained with every external parcel (no independent test remains for them)."""
    from crop_pipeline.common import save_json
    from model_aza.improved import workflow
    receipt, rows, X, ok = load_inputs(repository, inputs_handoff)
    inputs = workflow.paoay_inputs(repository, ee)
    run = _new_run(repository, "paoay")
    for name, include in MAP_MODELS.items():
        model = fit_on(rows, X, ok, include, params or workflow.FINAL_PARAMS)
        folder = run / name
        folder.mkdir()
        workflow.write_paoay_outputs(folder, model, FEATURES, inputs)
        if progress:
            progress(f"{name} map written")
    save_json(run / "maps.json", {"features": FEATURES, "models": list(MAP_MODELS), "inputs_handoff": str(receipt),
                                  "status": "exploratory_unvalidated_no_local_labels_no_independent_test"})
    return workflow.seal(run, "mixed_training_paoay", inputs_handoff=str(receipt))
