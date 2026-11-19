import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from crop_pipeline.output_layout import refresh


class OutputLayoutTests(unittest.TestCase):
    def fixture(self, root):
        image = root / "data/pipeline_work/annual_tifs/eurocrops/2021/tile.tif"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"original imagery")
        run = root / "data/pipeline_work/analysis/external/external_fixture"
        run.mkdir(parents=True)
        (run / "features_test.json").write_text('{"rows": {}}')
        return image, run

    def test_refresh_preserves_bytes_and_old_paths_and_discovers_new_runs(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            image, run = self.fixture(root)
            original = image.read_bytes()
            report = refresh(root, verify=True)
            self.assertEqual(report["verified_file_count"], 2)
            linked = root / "pipeline_outputs/collection/eurocrops/imagery/annual/2021/tile.tif"
            self.assertEqual(linked.read_bytes(), original)
            self.assertTrue(linked.samefile(image))
            self.assertEqual(image.read_bytes(), original)
            new = run.parent / "external_second"
            new.mkdir()
            (new / "windows.json").write_text("{}")
            refresh(root)
            refresh(root)
            self.assertTrue((root / "pipeline_outputs/preprocessing/external_second/windows/windows.json").samefile(new / "windows.json"))
            self.assertEqual(len(list((root / "pipeline_outputs").glob("integrity_*.json"))), 1)

    def test_owned_readme_redirects_to_merged_storage_guide_only(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            guide = root / "docs/storage_layout.md"
            guide.parent.mkdir()
            guide.write_text("current storage instructions")
            output = root / "pipeline_outputs"
            output.mkdir()
            readme = output / "README.md"
            readme.symlink_to("../docs/output_layout.md")
            (output / "layout_manifest.json").write_text(json.dumps({
                "kind": "linked_pipeline_output_layout",
                "links": {"README.md": "../docs/output_layout.md"}}))
            refresh(root)
            self.assertTrue(readme.samefile(guide))
            refresh(root)
            readme.unlink()
            readme.write_text("user notes")
            with self.assertRaises(FileExistsError):
                refresh(root)
            self.assertEqual(readme.read_text(), "user notes")

    def test_collision_stops_before_creating_other_links(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            image, _ = self.fixture(root)
            collision = root / "pipeline_outputs/collection/eurocrops/imagery/annual"
            collision.mkdir(parents=True)
            (collision / "mine.txt").write_text("keep me")
            with self.assertRaises(FileExistsError):
                refresh(root)
            self.assertEqual((collision / "mine.txt").read_text(), "keep me")
            self.assertTrue(image.exists())
            self.assertFalse((root / "pipeline_outputs/preprocessing").exists())

    def test_parent_symlink_is_not_followed_for_writes(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            output = root / "pipeline_outputs"
            output.mkdir()
            other = root / "other"
            other.mkdir()
            (output / "collection").symlink_to(other, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                refresh(root)
            self.assertEqual(list(other.iterdir()), [])

    def test_custom_archive_and_campaign_paths_are_respected(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "settings.local.json").write_text(json.dumps({
                "report_root": "custom/inventory", "annual_archive_root": "storage/archive"}))
            batch = root / "custom/adaptive_training/policy/rounds/0002/eurocrops_extra_1"
            batch.mkdir(parents=True)
            plan = root / "storage/archive/run/collection_plan.json"
            plan.parent.mkdir(parents=True)
            plan.write_text("{}")
            (batch / "context.json").write_text(json.dumps({"source": "eurocrops", "plan_path": str(plan)}))
            refresh(root)
            state = root / "pipeline_outputs/collection/eurocrops/state"
            self.assertTrue((state / "policy_0002_eurocrops_extra_1").samefile(batch))
            self.assertTrue((state / "policy_0002_eurocrops_extra_1_archive").samefile(plan.parent))

    def test_model_and_test_views_do_not_mix_experiments(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            _, run = self.fixture(root)
            model = run / "models_123/fixed_full"
            model.mkdir(parents=True)
            (model / "model_manifest.json").write_text('{"validation": {}}')
            (model.parent / "test_scores.json").write_text("{}")
            (model.parent / "fixed_full_test_parcels.csv").write_text("crop,prediction\n")
            refresh(root)
            key = "external_fixture_models_123"
            self.assertTrue((root / f"pipeline_outputs/testing/{key}/reports/test_scores.json").samefile(model.parent / "test_scores.json"))
            self.assertTrue((root / f"pipeline_outputs/training/{key}/validation/fixed_full.json").samefile(model / "model_manifest.json"))


if __name__ == "__main__":
    unittest.main()
