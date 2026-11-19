"""Workflow figures draw saved results without hiding missing values or targets."""

import io
import json
import tempfile
from contextlib import redirect_stdout
import unittest
import unittest.mock
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crop_pipeline.visualization import workflow_figures as figures


def usable_rows(cohort="fixed_full"):
    targets = {"train": (60, 1200), "validation": (20, 400), "test": (20, 400)}
    return [{"cohort": cohort, "partition": p, "crop": c, "usable_parcel_groups": "5", "usable_pixels": "50",
             "target_parcels": str(targets[p][0]), "target_pixels": str(targets[p][1])}
            for p in figures.PARTITIONS for c in figures.CROPS]


def bar_labels(figure):
    return [text.get_text() for axis in figure.axes for text in axis.texts]


class WorkflowFigureTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_usable_samples_show_every_partition_target(self):
        figure = figures.draw_usable_vs_targets(usable_rows(), "fixed_full")
        labels = bar_labels(figure)
        for label in ("5\n/ 60", "5\n/ 20", "50\n/ 1,200", "50\n/ 400"):
            self.assertIn(label, labels)

    def test_window_counts_show_policy_parcel_targets(self):
        rows = [{"window": w, "partition": p, "crop": c, "parcel_groups": "7"}
                for w in ("fixed", "aligned") for p in figures.PARTITIONS for c in figures.CROPS]
        labels = bar_labels(figures.draw_window_counts(rows))
        self.assertEqual(figures.partition_targets("parcels"), [60, 20, 20])
        self.assertEqual(labels.count("7\n/ 60"), 6)
        self.assertEqual(labels.count("7\n/ 20"), 12)

    def test_missing_cohort_is_an_error_not_an_empty_chart(self):
        with self.assertRaises(ValueError):
            figures.draw_usable_vs_targets(usable_rows("aligned_full"), "fixed_full")

    def test_unavailable_scores_are_labelled_not_zero(self):
        scores = {"fixed_full": {"macro_f1": 0.4, "per_crop": {"carrot": {"f1": 0.1}, "potato": {"f1": 0.6}}},
                  "aligned_full": {"status": "no_usable_test_rows"}}
        labels = bar_labels(figures.draw_macro_f1(scores, "Test", baseline={"fixed_full": 0.2}))
        self.assertIn("0.40", labels)
        self.assertIn("n/a", labels)
        per_crop = bar_labels(figures.draw_per_crop_f1(scores, "F1"))
        self.assertIn("n/a", per_crop)

    def test_quality_rates_skip_partitions_without_audits(self):
        rows = [{"partition": "train", "crop": c, "audited_tile_intervals": "10", "quality_pass_tile_intervals": "8"}
                for c in figures.CROPS]
        labels = bar_labels(figures.draw_quality_pass_rates(rows))
        self.assertEqual(labels.count("0.80"), 3)
        self.assertEqual(labels.count("n/a"), 6)

    def test_observation_loss_marks_half_months_without_counts(self):
        rows = [{"half_month_start": "01-01", "cloud_or_shadow_share": "0.8"},
                {"half_month_start": "01-16", "cloud_or_shadow_share": ""}]
        figure = figures.draw_observation_loss(rows)
        heights = [patch.get_height() for patch in figure.axes[0].patches]
        self.assertEqual(heights, [0.8, 0])
        self.assertEqual(bar_labels(figure), ["n/a"])

    def test_confusion_matrix_allows_an_empty_reference_row(self):
        scores = {"fixed_full": {"macro_f1": 0.3, "classes": list(figures.CROPS),
                                 "confusion_matrix": [[0, 0, 0], [1, 2, 0], [0, 1, 3]]}}
        figure = figures.draw_confusion_matrices(scores, "Confusion")
        self.assertEqual(len(figure.axes), 1)

    def test_validation_scores_come_from_frozen_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            manifest = folder / "fixed_full.json"
            manifest.write_text(json.dumps({"validation": {"macro_f1": 0.5}}))
            index = folder / "models.json"
            index.write_text(json.dumps({"fixed_full": {"path": str(manifest), "sha256": "x"}}))
            self.assertEqual(figures.load_validation_scores(index), {"fixed_full": {"macro_f1": 0.5}})
            path = figures.save_figure(figures.draw_macro_f1({"fixed_full": {"macro_f1": 0.5}}, "V"),
                                       folder / "figures", "validation")
            self.assertTrue(path.is_file())

    def test_baseline_reads_majority_file_shape(self):
        baselines = {"fixed_full": {"training_majority_class": "potato", "metrics": {"macro_f1": 0.15}},
                     "aligned_full": {"training_majority_class": "potato", "metrics": {}}}
        self.assertEqual(figures.baseline_macro_f1(baselines), {"fixed_full": 0.15, "aligned_full": None})


class ParcelEvidenceTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def write_tif(self, path):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        names = [f"2022_{m:02d}_{half}__{band}" for m in range(1, 13) for half in ("01_15", "16_end")
                 for band in ("B3", "B4", "B5", "B8", "B11", "B2", "clear_count")]
        data = np.full((len(names), 3, 3), -9999, dtype="float32")
        for index, name in enumerate(names):
            if name.endswith("__B4"):
                data[index, 1, 1] = 500
            if name.endswith("__B8") and not name.startswith("2022_02_01_15"):
                data[index, 1, 1] = 3000
        with rasterio.open(path, "w", driver="GTiff", width=3, height=3, count=len(names), dtype="float32",
                           crs="EPSG:32630", transform=from_origin(587710, 5190930, 10, 10), nodata=-9999) as image:
            image.write(data)
            for index, name in enumerate(names, start=1):
                image.set_band_description(index, name)

    def test_examples_use_median_parcel_and_draw_only_saved_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            tif = folder / "annual.tif"
            self.write_tif(tif)
            fields = folder / "fields"
            fields.mkdir()
            ring = [[-1.848, 46.864], [-1.847, 46.864], [-1.847, 46.865], [-1.848, 46.864]]
            (fields / "source_footprints.geojson").write_text(json.dumps({"features": [
                {"properties": {"partition_group": "francecrops|FR|2022|b"},
                 "geometry": {"type": "Polygon", "coordinates": [ring]}}]}))
            regions = [{"properties": {"fragment_id": name, "partition_group": f"francecrops|FR|2022|{name}",
                                       "source_dataset": "francecrops", "crop": "cabbage", "selected_pixels": count,
                                       "country": "FR", "year": 2022, "parcel_id": name}}
                       for name, count in (("a", 1), ("b", 5), ("c", 9))]
            quality = {"regions": regions, "sources": [{"field_folder": str(fields)}],
                       "quality_rows": [{"region_id": r, "image_path": str(tif)} for r in "abc"]}
            examples = figures.parcel_evidence_examples(quality)
            self.assertEqual([e["properties"]["parcel_id"] for e in examples], ["b"])
            self.assertEqual(len(examples[0]["footprints"]), 1)
            titles = [axis.get_title() for axis in figures.draw_parcel_evidence(examples[0]).axes[:8]]
            self.assertIn("Jan 1–15 2022\n1 clear pixels", titles)
            self.assertIn("Feb 1–15 2022\n0 clear pixels", titles)

    def test_handoff_without_tifs_gives_no_examples(self):
        self.assertEqual(figures.parcel_evidence_examples({"regions": [], "quality_rows": []}), [])


class NotebookFigureCellTests(unittest.TestCase):
    """Run only the figure cells of Training 01 and Testing 02 against saved-handoff fixtures."""

    repository = Path(__file__).resolve().parents[1]

    def figure_cells(self, relative):
        book = json.loads((self.repository / relative).read_text())
        cells = ["".join(c["source"]) for c in book["cells"]
                 if c["cell_type"] == "code" and "workflow_figures" in "".join(c["source"])]
        self.assertEqual(len(cells), 1)
        return cells[0]

    def metrics(self, value):
        return {"macro_f1": value, "classes": list(figures.CROPS),
                "confusion_matrix": [[2, 1, 0], [0, 2, 1], [1, 0, 2]],
                "per_crop": {crop: {"f1": value} for crop in figures.CROPS}}

    def models(self, folder):
        index = {}
        for cohort in figures.COHORTS:
            manifest = folder / f"{cohort}_manifest.json"
            manifest.write_text(json.dumps({"validation": self.metrics(0.6)}))
            index[cohort] = {"path": str(manifest), "sha256": "x"}
        (folder / "models.json").write_text(json.dumps(index))
        return folder / "models.json"

    def run_cell(self, relative, namespace):
        with unittest.mock.patch("IPython.display.display"), redirect_stdout(io.StringIO()):
            exec(self.figure_cells(relative), namespace)
        plt.close("all")

    def test_training_cell_saves_validation_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "validation").mkdir()
            self.models(run / "validation")
            self.run_cell("notebooks/training/01_train_and_validate.ipynb", {"training_handoff": run / "training.json"})
            self.assertEqual(sorted(p.name for p in (run / "figures").iterdir()),
                             ["validation_confusion_matrices.png", "validation_macro_f1.png"])

    def test_testing_cell_saves_test_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "training_run" / "models"
            models.mkdir(parents=True)
            self.models(models)
            run = root / "testing_run"
            (run / "reports" / "diagnostics").mkdir(parents=True)
            (run / "scores.json").write_text(json.dumps({"models_handoff": str(models / "models_stage.json")}))
            (run / "reports" / "test_scores.json").write_text(
                json.dumps({c: self.metrics(0.3) for c in figures.COHORTS}))
            (run / "reports" / "majority_baseline.json").write_text(json.dumps(
                {c: {"training_majority_class": "potato", "metrics": {"macro_f1": 0.2}} for c in figures.COHORTS}))
            self.run_cell("notebooks/testing/02_results_and_diagnostics.ipynb",
                          {"report_handoff": run / "reports" / "diagnostics" / "report.json"})
            self.assertEqual(sorted(p.name for p in (run / "reports" / "figures").iterdir()),
                             ["test_confusion_matrices.png", "test_macro_f1.png", "test_per_crop_f1.png"])
            # A selected supplement must read and write its own scope, even if
            # primary JECAM score files are unavailable.
            population = "denmark_marker_cabbage_2023_v1"
            scope = run / "reports" / population
            scope.mkdir()
            for name in ("test_scores.json", "majority_baseline.json"):
                (scope / name).write_bytes((run / "reports" / name).read_bytes())
                (run / "reports" / name).unlink()
            self.run_cell("notebooks/testing/02_results_and_diagnostics.ipynb",
                          {"report_handoff": run / "reports/diagnostics/report.json", "population": population})
            self.assertEqual(len(list((scope / "figures").glob("*.png"))), 3)


if __name__ == "__main__":
    unittest.main()
