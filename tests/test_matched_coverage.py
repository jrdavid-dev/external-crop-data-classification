"""Coverage diagnostics use saved assignments and the selected run's batch snapshot."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from crop_pipeline import matched_coverage
from crop_pipeline.common import save_json
from crop_pipeline.preprocessing_workflow import paths
from crop_pipeline.sources import adaptive
from crop_pipeline.windows import seasons


class MatchedCoverageTests(unittest.TestCase):
    def test_counts_pixels_from_included_batch_and_ignores_later_completions(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            _, collection, _ = paths(root)
            run = root / "run"
            handoff = run / "features.json"
            save_json(handoff, {})
            save_json(run / "development_split.json", {
                "matched_parcel_groups": ["carrot-1", "carrot-2", "cabbage-1"],
                "assignments": {"carrot-1": "train", "carrot-2": "validation", "cabbage-1": "train"},
            })
            seasons.write_records(run / "usable_samples.csv", [
                {"cohort": "fixed_matched", "crop": "carrot", "partition": "train",
                 "usable_parcel_groups": 1, "usable_pixels": 2, "parcel_shortfall": 59, "pixel_shortfall": 1198}
            ])
            snapshot = {}
            for (batch, label), groups in zip(matched_coverage.BATCHES.items(), (
                ["carrot-1", "carrot-2"], ["cabbage-1"]
            ), strict=True):
                batch_path = adaptive.root(collection) / "rounds/0002" / label
                selection = batch_path / "selection.json"
                completed = batch_path / "completed.json"
                save_json(selection, {"groups": groups})
                save_json(completed, {})
                if batch.startswith("Carrot"):
                    snapshot.update({str(path): seasons.file_sha256(path) for path in (selection, completed)})
            save_json(run / "collection_snapshot.json", {"campaign_metadata": snapshot})
            pixels = [{"partition_group": group} for group in ("carrot-1", "carrot-1", "carrot-2", "cabbage-1")]
            save_json(run / "features_development.json", {
                "rows": {"fixed_full": pixels, "fixed_matched": pixels,
                         "aligned_full": pixels, "aligned_matched": pixels}
            })

            saved_run, coverage, contributions = matched_coverage.report(root, handoff)
            self.assertEqual(saved_run, run)
            self.assertEqual(coverage[0]["usable_pixels"], "2")
            self.assertEqual([(r["matched_parcels"], r["matched_pixels"]) for r in contributions],
                             [(1, 2), (1, 1), (0, 0), (0, 0)])
            self.assertEqual(contributions[2]["snapshot_status"], "Completed after this snapshot")

    def test_incremental_matches_remove_saved_development_collisions_and_apply_cap(self):
        from crop_pipeline.windows import external

        def pixel(group, x, partition="development"):
            return {"partition_group": group, "region_id": group, "source_crs": "EPSG:32631",
                    "x": x, "y": 0, "partition": partition, "crop": "cabbage",
                    "source_dataset": "francecrops"}

        with TemporaryDirectory() as folder:
            root = Path(folder)
            baseline = root / "baseline.json"
            windows = root / "windows.json"
            new = [pixel("new", x) for x in range(25)]
            save_json(windows, {"fixed": [pixel("collision", 100), *new],
                                "aligned": [pixel("collision", 100), *new]})
            save_json(baseline, {"fixed": [pixel("old", 100, "train"), pixel("test", 1, "test")],
                                 "aligned": []})

            indexed, _ = external.index_windows(windows)
            matches = list(matched_coverage._unshared_matches(indexed, baseline))
            self.assertEqual(len(matches), 20)
            self.assertEqual({key[0] for key in matches}, {"new"})
            self.assertEqual(matches, list(external.capped_keys(external.pixel_key(row) for row in new)))
