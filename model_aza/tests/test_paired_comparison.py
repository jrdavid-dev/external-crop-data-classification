import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from crop_pipeline.training.external_report import CROPS, MODELS, confusion, metric_values
from model_aza.improved import paired_comparison, workflow


def saved_scores(rows):
    matrix = confusion(np.array([CROPS.index(r["crop"]) for r in rows], dtype=int),
                       np.array([CROPS.index(r["prediction"]) for r in rows], dtype=int))
    points = metric_values(matrix)
    complete = (matrix.sum(axis=1) > 0).all()
    return {"classes": CROPS, "confusion_matrix": matrix.tolist(),
            "macro_f1": float(points["macro_f1", ""]) if complete else None,
            "balanced_accuracy": float(points["balanced_accuracy", ""]) if complete else None,
            "per_crop": {crop: {"precision": float(points["precision", crop]), "recall": float(points["recall", crop]),
                                "f1": float(points["f1", crop]), "support": int(matrix[i].sum())}
                         for i, crop in enumerate(CROPS)}}


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
    return jecam, carrot


def original_run(root, prediction="potato"):
    """Original run: every model predicts one crop on all but the last three JECAM parcels."""
    run = root / "testing" / "run_original"
    jecam, carrot = parcels()
    scopes = {"jecam": jecam[:-3], "california_dwr_carrot_2023_v1": carrot}
    (run / "reports").mkdir(parents=True)
    (run / "reports" / "test_scopes.json").write_text(json.dumps([*scopes, "combined"]))
    pooled = {model: [] for model in MODELS}
    for scope, rows in scopes.items():
        folder = "" if scope == "jecam" else scope
        rows = [{**row, "prediction": prediction} for row in rows]
        majority = [{**row, "prediction": "potato"} for row in rows]
        (run / "reports" / folder).mkdir(parents=True, exist_ok=True)
        (run / "reports" / folder / "test_scores.json").write_text(json.dumps({m: saved_scores(rows) for m in MODELS}))
        (run / "reports" / folder / "majority_baseline.json").write_text(json.dumps(
            {m: {"training_majority_class": "potato", "metrics": saved_scores(majority)} for m in MODELS}))
        for model in MODELS:
            write_rows(run / "predictions" / folder / f"{model}_test_parcels.csv", rows)
            pooled[model] += rows
    for model in MODELS:
        write_rows(run / "predictions" / "combined" / f"{model}_test_parcels.csv", pooled[model])
    return run


def improved_run(root, change=None):
    """Improved run: correct on every JECAM and carrot parcel."""
    run = root / "improved" / "test" / "run_improved"
    jecam, carrot = parcels()
    rows = [{**row, "prediction": row["crop"]} for row in jecam + carrot]
    if change:
        rows = change(rows)
    write_rows(run / "test_parcels.csv", rows)
    (run / "test_scores.json").write_text(json.dumps({"candidate": "fixture", "training_majority_class": "carrot"}))
    return workflow.seal(run, "test")


class PairedComparisonTests(unittest.TestCase):
    def test_compares_on_original_parcels_with_signed_paired_difference(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            improved, info = paired_comparison.load_improved(improved_run(root))
            original, majority, campaigns = paired_comparison.load_original(original_run(root))
            summary, crops = paired_comparison.compare(improved, info["majority"], original, majority,
                                                       campaigns, repetitions=200)
            row = next(r for r in summary if r["population"] == "jecam" and r["comparison"] == "versus_fixed_full")
            self.assertEqual((row["parcels"], row["improved_only_parcels"]), (27, 3))
            self.assertEqual(row["improved_macro_f1"]["estimate"], 1.0)
            self.assertGreater(row["difference"]["ci_low"], 0)
            self.assertEqual(row["difference"]["share_of_resamples_improved"], 1.0)
            self.assertEqual(row["original_minus_majority"]["estimate"], 0.0)  # original equals its majority
            own = next(r for r in summary if r["population"] == "jecam" and r["comparison"] == "improved_own_coverage")
            self.assertEqual(own["parcels"], 30)
            combined = next(r for r in summary if r["population"] == "combined" and r["comparison"] == "versus_fixed_full")
            self.assertEqual(combined["parcels"], 37)
            carrot = [r for r in crops if r["population"] == "california_dwr_carrot_2023_v1"
                      and r["predictor"] == "improved minus fixed_full"]
            self.assertEqual([(r["crop"], r["estimate"]) for r in carrot], [("carrot", 1.0)])

    def test_identical_predictions_give_zero_difference(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            improved, info = paired_comparison.load_improved(
                improved_run(root, lambda rows: [{**r, "prediction": "potato"} for r in rows]))
            original, majority, campaigns = paired_comparison.load_original(original_run(root))
            summary, _ = paired_comparison.compare(improved, info["majority"], original, majority,
                                                   campaigns, repetitions=50)
            for row in summary:
                if row["difference"] and row["difference"]["estimate"] is not None:
                    self.assertEqual((row["difference"]["estimate"], row["difference"]["ci_low"],
                                      row["difference"]["ci_high"]), (0.0, 0.0, 0.0))

    def test_rejects_missing_parcels_and_label_mismatches(self):
        for change in (lambda rows: rows[1:],
                       lambda rows: [{**r, "crop": "cabbage"} if r["partition_group"] == "c0" else r for r in rows]):
            with TemporaryDirectory() as folder:
                root = Path(folder)
                improved, info = paired_comparison.load_improved(improved_run(root, change))
                original, majority, campaigns = paired_comparison.load_original(original_run(root))
                with self.assertRaises(ValueError):
                    paired_comparison.compare(improved, info["majority"], original, majority, campaigns,
                                              repetitions=10)

    def test_rejects_changed_inputs_and_existing_output(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            receipt, run = improved_run(root), original_run(root)
            output = paired_comparison.report(receipt, run, root / "report", repetitions=20)
            self.assertTrue((output / "report.md").read_text().startswith("# Improved model"))
            self.assertIn("Retrospective", (output / "report.md").read_text())
            with self.assertRaises(ValueError):
                paired_comparison.report(receipt, run, output, repetitions=20)
            scores = run / "reports" / "test_scores.json"
            saved = json.loads(scores.read_text())
            saved["fixed_full"]["confusion_matrix"][0][0] += 1
            scores.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):
                paired_comparison.load_original(run)
            (receipt.parent / "test_parcels.csv").write_text("changed")
            with self.assertRaises(ValueError):
                paired_comparison.load_improved(receipt)


if __name__ == "__main__":
    unittest.main()
