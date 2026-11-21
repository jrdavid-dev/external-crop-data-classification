import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from crop_pipeline.training import study_manifest


class StudyManifestTests(unittest.TestCase):
    def test_verification_detects_changed_artifacts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "scores.csv"
            artifact.write_text("macro_f1\n0.493\n")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"artifact_sha256": {"scores.csv": study_manifest.file_sha256(artifact)}}))
            self.assertEqual(study_manifest.verify(root, manifest), 1)
            artifact.write_text("macro_f1\n0.5\n")
            with self.assertRaisesRegex(ValueError, "scores.csv"):
                study_manifest.verify(root, manifest)

    def test_preprocessing_drift_uses_relative_source_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/crop_pipeline/preprocessing_workflow.py"
            source.parent.mkdir(parents=True)
            source.write_text("new implementation")
            receipt = root / "receipt.json"
            digest = study_manifest.file_sha256(source)
            receipt.write_text(json.dumps({
                "stage": "features",
                "files": {"/old/src/workspace/src/crop_pipeline/preprocessing_workflow.py": "old hash"},
            }))
            result = study_manifest.preprocessing_source_drift(root, [Path("receipt.json")])
            self.assertEqual(result, {"src/crop_pipeline/preprocessing_workflow.py": {
                "recorded_sha256_by_stage": {"features": "old hash"},
                "current_sha256": digest, "matching_stages": [],
            }})
