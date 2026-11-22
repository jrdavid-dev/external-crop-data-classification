"""Paired challenger-versus-baseline comparison on the parcels each baseline model scored."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.annual import paired_comparison
from crop_pipeline.training.external_report import CROPS, confusion, metric_values

BASELINE_MODELS = ("fixed_full", "aligned_full")


def saved_scores(rows):
    matrix = confusion(np.array([CROPS.index(r["crop"]) for r in rows], dtype=int),
                       np.array([CROPS.index(r["prediction"]) for r in rows], dtype=int))
    points = metric_values(matrix)
    complete = (matrix.sum(axis=1) > 0).all()
    scores = {"classes": CROPS, "confusion_matrix": matrix.tolist(),
              "per_crop": {crop: {"precision": float(points["precision", crop]), "recall": float(points["recall", crop]),
                                  "f1": float(points["f1", crop]), "support": int(matrix[i].sum())}
                           for i, crop in enumerate(CROPS)}}
    if complete:
        scores.update(macro_f1=float(points["macro_f1", ""]), balanced_accuracy=float(points["balanced_accuracy", ""]))
    return scores


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["partition_group", "crop", "prediction", "test_cohort"])
        writer.writeheader()
        writer.writerows(rows)


def parcels():
    jecam = [{"partition_group": f"j{i}", "crop": CROPS[i % 3], "test_cohort": "jecam"} for i in range(30)]
    carrot = [{"partition_group": f"c{i}", "crop": "carrot", "test_cohort": "california_dwr_carrot_2023_v1"}
              for i in range(10)]
    return {"jecam": jecam, "california_dwr_carrot_2023_v1": carrot}


def testing_run(root, name, models, predict, keep=lambda scope, i: True, majority="potato"):
    """A run in the main testing layout; ``predict(row)`` gives each model's prediction."""
    run = root / name
    scopes = {scope: [r for i, r in enumerate(rows) if keep(scope, i)] for scope, rows in parcels().items()}
    (run / "reports").mkdir(parents=True)
    (run / "reports" / "test_scopes.json").write_text(json.dumps([*scopes, "combined"]))
    pooled = {model: [] for model in models}
    for scope, rows in scopes.items():
        folder = "" if scope == "jecam" else scope
        rows = [{**row, "prediction": predict(row)} for row in rows]
        (run / "reports" / folder).mkdir(parents=True, exist_ok=True)
        (run / "reports" / folder / "test_scores.json").write_text(json.dumps({m: saved_scores(rows) for m in models}))
        (run / "reports" / folder / "majority_baseline.json").write_text(json.dumps(
            {m: {"training_majority_class": majority,
                 "metrics": saved_scores([{**r, "prediction": majority} for r in rows])} for m in models}))
        for model in models:
            write_rows(run / "predictions" / folder / f"{model}_test_parcels.csv", rows)
            pooled[model] += rows
    for model in models:
        write_rows(run / "predictions" / "combined" / f"{model}_test_parcels.csv", pooled[model])
    return run


def runs(root, challenger_predict=lambda r: r["crop"]):
    baseline = testing_run(root, "baseline", BASELINE_MODELS, lambda r: "potato",
                           keep=lambda scope, i: scope != "jecam" or i < 27)
    challenger = testing_run(root, "annual", ("annual",), challenger_predict, majority="carrot")
    return challenger, baseline


class PairedComparisonTests(unittest.TestCase):
    def test_challenger_is_scored_on_each_baseline_models_parcels(self):
        with TemporaryDirectory() as folder:
            challenger, baseline = runs(Path(folder))
            mine, mine_majority, _ = paired_comparison.load_run(challenger, ["annual"])
            theirs, majority, campaigns = paired_comparison.load_run(baseline)
            summary, crops = paired_comparison.compare(mine["annual"], mine_majority["annual"], theirs, majority,
                                                       campaigns, repetitions=200)
            row = next(r for r in summary if r["population"] == "jecam" and r["comparison"] == "versus_fixed_full")
            self.assertEqual((row["parcels"], row["challenger_only_parcels"]), (27, 3))
            self.assertEqual(row["challenger_macro_f1"]["estimate"], 1.0)
            self.assertGreater(row["difference"]["ci_low"], 0)
            self.assertEqual(row["baseline_minus_majority"]["estimate"], 0.0)
            own = next(r for r in summary if r["population"] == "jecam" and r["comparison"] == "challenger_own_coverage")
            self.assertEqual(own["parcels"], 30)
            combined = next(r for r in summary if r["population"] == "combined"
                            and r["comparison"] == "versus_aligned_full")
            self.assertEqual(combined["parcels"], 37)
            carrot = [r for r in crops if r["population"] == "california_dwr_carrot_2023_v1"
                      and r["predictor"] == "challenger minus fixed_full"]
            self.assertEqual([(r["crop"], r["estimate"]) for r in carrot], [("carrot", 1.0)])

    def test_identical_predictions_give_zero_difference(self):
        with TemporaryDirectory() as folder:
            challenger, baseline = runs(Path(folder), lambda r: "potato")
            mine, mine_majority, _ = paired_comparison.load_run(challenger, ["annual"])
            theirs, majority, campaigns = paired_comparison.load_run(baseline)
            summary, _ = paired_comparison.compare(mine["annual"], mine_majority["annual"], theirs, majority,
                                                   campaigns, repetitions=50)
            for row in summary:
                if row["difference"] and row["difference"]["estimate"] is not None:
                    self.assertEqual((row["difference"]["estimate"], row["difference"]["ci_low"],
                                      row["difference"]["ci_high"]), (0.0, 0.0, 0.0))

    def test_missing_parcels_label_mismatch_and_changed_scores_are_refused(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            baseline = testing_run(root, "baseline", BASELINE_MODELS, lambda r: "potato")
            short = testing_run(root, "short", ("annual",), lambda r: r["crop"], keep=lambda scope, i: i > 0)
            mine, mine_majority, _ = paired_comparison.load_run(short, ["annual"])
            theirs, majority, campaigns = paired_comparison.load_run(baseline)
            with self.assertRaises(ValueError):
                paired_comparison.compare(mine["annual"], "carrot", theirs, majority, campaigns, repetitions=10)
            relabelled = {k: (c, "cabbage" if k == "c0" else a, p) for k, (c, a, p) in theirs["fixed_full"].items()}
            full, _, _ = paired_comparison.load_run(testing_run(root, "full", ("annual",), lambda r: r["crop"]),
                                                    ["annual"])
            with self.assertRaises(ValueError):
                paired_comparison.compare(full["annual"], "carrot", {"fixed_full": relabelled}, majority, campaigns,
                                          repetitions=10)
            scores = baseline / "reports" / "test_scores.json"
            saved = json.loads(scores.read_text())
            saved["fixed_full"]["confusion_matrix"][0][0] += 1
            scores.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):
                paired_comparison.load_run(baseline)

    def test_report_writes_tables_and_refuses_existing_output(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            challenger, baseline = runs(root)
            output = paired_comparison.report(challenger, "annual", baseline, root / "report", repetitions=20)
            text = (output / "report.md").read_text()
            self.assertTrue(text.startswith("# annual versus the baseline models"))
            self.assertIn("Retrospective", text)
            self.assertTrue((output / "paired_macro_f1.csv").exists() and (output / "per_crop_f1.csv").exists())
            with self.assertRaises(ValueError):
                paired_comparison.report(challenger, "annual", baseline, output, repetitions=20)


if __name__ == "__main__":
    unittest.main()
