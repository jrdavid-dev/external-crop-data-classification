"""Check that reporting counts do not inflate pixels or select models on test scores."""

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crop_pipeline.visualization import pipeline as pipeline_visuals


class PipelineVisualTests(unittest.TestCase):
    def test_timing_counts_each_region_once_and_excludes_cross_year_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source_handoffs").mkdir()
            for source in ("sen4agrinet", "francecrops"):
                folder = root / source
                folder.mkdir()
                (folder / "timing_first").mkdir()
                crop = "carrot" if source == "sen4agrinet" else "cabbage"
                regions = [
                    {
                        "properties": {
                            "fragment_id": identifier,
                            "crop": crop,
                            "selected_pixels": count,
                        }
                    }
                    for identifier, count in [("in_year", 10), ("cross_year", 5)]
                ]
                (folder / "sampled_download_regions.geojson").write_text(
                    json.dumps({"features": regions})
                )
                windows = [
                    {
                        "region_id": "in_year",
                        "window_start": "2020-04-01",
                        "window_end": "2020-08-01",
                    }
                ] * 2
                windows += [
                    {
                        "region_id": "cross_year",
                        "window_start": "2020-11-01",
                        "window_end": "2021-03-01",
                    }
                ]
                (folder / "timing_first/timing_gate.json").write_text(
                    json.dumps({"qualified_windows": windows})
                )
                (root / "source_handoffs" / f"{source}_run.json").write_text(
                    json.dumps(
                        {
                            "field_folder": str(folder),
                            "progress_folder": str(folder / "composite_progress"),
                            "year": 2020,
                        }
                    )
                )
            candidates, timing = pipeline_visuals.candidate_and_timing_counts(
                root / "inventory", "run"
            )
            self.assertEqual(candidates["carrot"], 15)
            self.assertEqual(timing["carrot"], 10)
            self.assertEqual(timing["cabbage"], 10)

    def test_combined_visual_report_uses_registered_source_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fields = root / "fields"
            fields.mkdir()
            reports = root / "reports/timing_first"
            reports.mkdir(parents=True)
            analysis = root / "analysis/combined"
            analysis.mkdir(parents=True)
            (fields / "sampled_download_regions.geojson").write_text(
                json.dumps(
                    {
                        "features": [
                            {
                                "properties": {
                                    "fragment_id": "cropharvest_region",
                                    "crop": "cabbage",
                                    "selected_pixels": 12,
                                }
                            }
                        ]
                    }
                )
            )
            (reports / "timing_gate.json").write_text(
                json.dumps(
                    {
                        "qualified_windows": [
                            {
                                "region_id": "cropharvest_region",
                                "window_start": "2020-04-01",
                                "window_end": "2020-08-01",
                            }
                        ]
                    }
                )
            )
            (analysis / "source_inputs.json").write_text(
                json.dumps(
                    [
                        {
                            "field_folder": str(fields),
                            "progress_folder": str(
                                reports.parent / "composite_progress"
                            ),
                            "year": 2020,
                        }
                    ]
                )
            )
            candidates, timing = pipeline_visuals.candidate_and_timing_counts(
                root / "inventory", "combined"
            )
            self.assertEqual(candidates["cabbage"], 12)
            self.assertEqual(timing["cabbage"], 12)

    def test_confusion_uses_validation_selection_not_highest_test_accuracy(self):
        metrics = {
            "feature_selection": {
                "external_parcel_holdout": {"selected_design": "interval_only"}
            },
            "test": {
                "interval_only": {
                    "accuracy": 0.5,
                    "classes": ["carrot", "cabbage", "potato"],
                    "confusion_matrix": [[1, 2, 0], [0, 3, 0], [1, 0, 2]],
                },
                "interval_and_shape": {"accuracy": 0.9},
            },
        }
        figure = pipeline_visuals.draw_confusion(metrics)
        self.assertIn("interval_only", figure.axes[0].get_title())
        self.assertEqual(
            figure.axes[0].images[0].get_array().tolist(),
            metrics["test"]["interval_only"]["confusion_matrix"],
        )
        plt.close(figure)
