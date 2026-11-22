"""Fresh-kernel preprocessing handoffs, sample accounting and collection isolation."""

from contextlib import chdir, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from shapely.geometry import box, mapping

from crop_pipeline import preprocessing_workflow as workflow
from crop_pipeline.analysis_workflow import resolve_stage_pointer
from crop_pipeline.common import save_json
from crop_pipeline.features.experiments import interval_columns
from crop_pipeline.sources import adaptive
from crop_pipeline.windows import seasons


def reference_rows():
    rows = []
    for partition, source in (("train", "sen4agrinet"), ("validation", "francecrops"), ("test", "cropharvest")):
        for crop in workflow.CROPS:
            group = f"{partition}-{crop}"
            rows.append({"partition_group": group, "field_id": group, "region_id": group,
                         "season_key": group, "crop": crop, "partition": partition,
                         "source_dataset": source, "label_status": "reported",
                         "source_crs": "EPSG:32631", "x": len(rows) * 100 + 5, "y": 5,
                         "window_start": "2020-01-01", "window_end": "2020-05-01",
                         **{column: .2 for column in seasons.feature_columns()}})
    return rows


def pooled_rows():
    return [{**row, "partition": workflow.external.DEVELOPMENT} if row["partition"] != "test" else row
            for row in reference_rows()]


def fake_audit(report, output):
    run = Path(output) / "external_fixture"
    run.mkdir(parents=True, exist_ok=False)
    regions = [{"properties": {"fragment_id": r["region_id"], "partition_group": r["partition_group"],
                                "crop": r["crop"], "partition": r["partition"], "selected_pixels": 1}}
               for r in pooled_rows()]
    save_json(run / "quality.json", {"regions": regions, "quality_rows": [], "pixel_paths": {},
                                     "development_split_policy": workflow.external.SPLIT_POLICY})
    return run / "quality.json"


def save_windows(quality, fixed, aligned, summarize):
    path = quality.parent / "windows.json"
    windows = {"fixed": fixed, "aligned": aligned, "quality": str(quality),
               "development_split_policy": workflow.external.SPLIT_POLICY}
    summarize(path.with_name("window_counts.csv"), windows)
    save_json(path, windows)
    return path


def fake_select(quality, report, *, summarize):
    rows = pooled_rows()
    return save_windows(quality, rows, rows[:-1], summarize)


class PreprocessingWorkflowTests(unittest.TestCase):
    def test_supplemental_overlap_keeps_primary_and_earlier_cohorts(self):
        def footprint(group, x):
            return {"properties": {"partition_group": group},
                    "geometry": mapping(box(x, 0, x + 1, 1))}

        primary = [{"footprints": [footprint("jecam", 0)]}]
        first = [{"footprints": [footprint("hits-primary", 0), footprint("first", 2)]}]
        second = [{"footprints": [footprint("hits-first", 2), footprint("second", 4)]}]
        inventory = [
            {"cohort": "jecam", "sources": primary, "batches": []},
            {"cohort": "first", "sources": first, "batches": []},
            {"cohort": "second", "sources": second, "batches": []},
        ]
        with patch("crop_pipeline.cabbage_test_check.discover", return_value=inventory), patch.object(
            workflow.external.test_cohort, "source_footprints",
            side_effect=lambda sources: (f for s in sources for f in s["footprints"]),
        ):
            sources, excluded, _ = workflow.external.supplemental_inputs(Path("fixture"), primary)
        self.assertEqual(excluded, {"first": {"hits-primary"}, "second": {"hits-first"}})
        self.assertEqual([(s["test_cohort"], s["excluded_test_groups"]) for s in sources],
                         [("first", ["hits-primary"]), ("second", ["hits-first"])])

    def test_large_json_handoff_round_trips_without_pretty_print_overhead(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "windows.json"
            payload = {"fixed": [{"pixel": 1, "valid": True}] * 50_000, "aligned": []}
            save_json(path, payload)
            self.assertEqual(json.loads(path.read_text()), payload)
            self.assertLess(path.stat().st_size, len(json.dumps(payload, indent=2)) // 2)

    def test_window_reader_byte_spans_across_chunks(self):
        from crop_pipeline.windows import json_rows

        data = {"quality": "audit/é.json", "fixed": [{"name": '日本語 "quoted" } ]'}],
                "aligned": [], "fixed_start_month": 12}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "windows.json"
            payload = json.dumps(data, ensure_ascii=False, indent=2).replace("\n", "\r\n").encode()
            path.write_bytes(payload)
            with patch.object(json_rows, "_CHUNK_SIZE", 7):
                records = list(json_rows.read_rows(path, {"fixed", "aligned"}))
            self.assertEqual([(name, value) for name, value, _ in records],
                             [("quality", data["quality"]), ("fixed", data["fixed"][0]),
                              ("fixed_start_month", 12)])
            _, row, (offset, length) = records[1]
            self.assertEqual(json.loads(payload[offset:offset + length]), row)

    def test_export_excludes_shared_development_locations_before_matching(self):
        rows = reference_rows()
        train, validation, test = rows[0], rows[3], rows[6]
        overlap = {**validation, "x": train["x"], "source_dataset": "sen4agrinet"}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "windows.json"
            save_json(path, {"fixed": [train, validation, test],
                             "aligned": [train, validation, overlap, test], "quality": "fixture"})
            workflow.external.export_features(path)
            development = json.loads(path.with_name("features_development.json").read_text())["rows"]
            self.assertEqual({r["partition"] for r in development["fixed_full"]}, {"train", "validation"})
            for cohort in ("aligned_full", "fixed_matched", "aligned_matched"):
                self.assertEqual([(r["partition_group"], r["partition"]) for r in development[cohort]],
                                 [(validation["partition_group"], "validation")])
            test_rows = json.loads(path.with_name("features_test.json").read_text())["rows"]
            self.assertTrue(all([r["partition_group"] for r in cohort] == [test["partition_group"]]
                                for cohort in test_rows.values()))
            exclusions = json.loads(path.with_name("development_overlap_exclusions.json").read_text())
            self.assertEqual(exclusions["aligned"]["shared_locations"], 1)
            self.assertEqual({r["partition"] for r in exclusions["aligned"]["excluded"]}, {"train", "validation"})

    def test_pooled_windows_lock_matched_groups_before_exporting_predictors(self):
        external = workflow.external
        first, second, test = (reference_rows()[i] for i in (0, 3, 6))
        first = {**first, "partition": external.DEVELOPMENT}
        second = {**second, "partition": external.DEVELOPMENT}
        fixed_only = {**first, "partition_group": "fixed-only", "field_id": "fixed-only",
                      "region_id": "fixed-only", "x": 1000}
        shared = [{**first, "x": first["x"] + offset} for offset in range(25)]
        test = {**test, "test_cohort": "jecam"}
        fixed = [*shared, second, fixed_only, test]
        aligned = [*shared, second, test]
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)

            def pooled_select(quality, report, *, summarize):
                return save_windows(quality, fixed, aligned, summarize)

            with patch.object(external, "audit", side_effect=fake_audit), patch.object(
                external, "select", side_effect=pooled_select
            ):
                workflow.audit(root)
                windows = workflow.select(root)

            receipt_path = workflow.record_path(windows)
            receipt = json.loads(receipt_path.read_text())
            receipt["files"].update({str(path): digest
                                     for path, digest in workflow._PRE_STREAMING_CODE.items()})
            save_json(receipt_path, receipt)
            read_text = Path.read_text

            def reject_large_json_read(path, *args, **kwargs):
                if path.name in {"windows.json", "features_development.json", "features_test.json"}:
                    self.fail(f"Feature export must avoid loading all of {path.name}")
                return read_text(path, *args, **kwargs)

            with patch.object(Path, "read_text", reject_large_json_read):
                handoff = workflow.export_features(root)

            split_path = handoff.parent / external.SPLIT_FILENAME
            lock = json.loads(split_path.read_text())
            self.assertEqual(lock["matched_parcel_groups"], sorted(
                [first["partition_group"], second["partition_group"]]))
            assignments = lock["assignments"]
            self.assertEqual({assignments[group] for group in lock["matched_parcel_groups"]},
                             {"train", "validation"})
            self.assertEqual(set(assignments), {*lock["matched_parcel_groups"], "fixed-only"})

            features = json.loads(handoff.with_name("features_development.json").read_text())["rows"]
            for rows in features.values():
                for row in rows:
                    self.assertEqual(row["partition"], assignments[row["partition_group"]])
                    self.assertTrue(set(interval_columns()).issubset(row))
            self.assertEqual(len(interval_columns()), 72)
            self.assertEqual(len(features["fixed_matched"]), 21)
            self.assertEqual([external.pixel_key(row) for row in features["fixed_matched"]],
                             [external.pixel_key(row) for row in features["aligned_matched"]])

            test_rows = json.loads(handoff.with_name("features_test.json").read_text())["rows"]
            for rows in test_rows.values():
                self.assertEqual([(r["partition"], r["test_cohort"], external.pixel_key(r)) for r in rows],
                                 [("test", "jecam", external.pixel_key(test))])
            self.assertEqual(workflow.export_features(root), handoff)
            split_path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "output changed"):
                workflow.export_features(root)

    def fixture(self, root):
        (root / "pyproject.toml").touch()
        state = root / "data/pipeline_work/adaptive_training/campaign/rounds/0001/source/state.json"
        save_json(state, {"status": "completed", "task_id": "preserve"})
        return state

    def test_observation_loss_counts_cloud_shadow_and_keeps_legacy_rows_unavailable(self):
        data = {
            "regions": [{"properties": {"fragment_id": "a", "partition": "train"}},
                        {"properties": {"fragment_id": "b", "partition": "test"}}],
            "quality_rows": [
                {"region_id": "a", "crop": "carrot", "start": "2021-01-01", "status": "quality_pass",
                 "pre_mask_count": 10, "cloud_shadow_count": 6, "other_rejection_count": 1, "clear_pixel_dates": 3},
                {"region_id": "a", "crop": "carrot", "start": "2021-01-16", "status": "low_coverage",
                 "pre_mask_count": 10, "cloud_shadow_count": 9, "other_rejection_count": 0, "clear_pixel_dates": 1},
                {"region_id": "b", "crop": "potato", "start": "2021-01-01", "status": "quality_pass",
                 "pre_mask_count": None, "cloud_shadow_count": None, "other_rejection_count": None,
                 "clear_pixel_dates": 4},
            ],
        }
        by_crop, by_half_month, total = workflow.observation_loss(data)
        carrot = next(r for r in by_crop if r["crop"] == "carrot")
        self.assertEqual((carrot["partition"], carrot["pixel_dates_with_acquisition"],
                          carrot["cloud_or_shadow_pixel_dates"], carrot["cloud_or_shadow_share"]), ("train", 20, 15, .75))
        legacy = next(r for r in by_crop if r["crop"] == "potato")
        self.assertEqual(legacy["intervals_without_pre_mask_counts"], 1)
        self.assertIsNone(legacy["cloud_or_shadow_share"])
        self.assertEqual([r["half_month_start"] for r in by_half_month], ["01-01", "01-16"])
        self.assertEqual(by_half_month[1]["cloud_or_shadow_share"], .9)
        self.assertEqual((total["audited_tile_intervals"], total["quality_pass_tile_intervals"]), (3, 2))

    def test_end_to_end_counts_reuse_and_unchanged_collection(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            state = self.fixture(root)
            original = state.read_bytes()
            with patch.object(workflow.external, "audit", side_effect=fake_audit), patch.object(
                workflow.external, "select", side_effect=fake_select
            ), patch("requests.sessions.Session.request", side_effect=AssertionError("No network")):
                quality = workflow.audit(root)
                windows = workflow.select(root)
                handoff = workflow.export_features(root)
                before = {p: p.read_bytes() for p in handoff.parent.rglob("*") if p.is_file()}
                self.assertEqual(workflow.select(root), windows)
                self.assertEqual(workflow.export_features(root), handoff)
            self.assertEqual(state.read_bytes(), original)
            self.assertTrue(all(p.read_bytes() == b for p, b in before.items()))
            self.assertTrue((quality.parent / "observation_loss.csv").is_file())
            self.assertTrue((quality.parent / "observation_loss_by_half_month.csv").is_file())
            rows = seasons.read_table(handoff.parent / "usable_samples.csv")
            self.assertEqual(len(rows), 36)
            fixed_test = next(r for r in rows if r["cohort"] == "fixed_full" and r["partition"] == "test" and r["crop"] == "potato")
            aligned_test = next(r for r in rows if r["cohort"] == "aligned_full" and r["partition"] == "test" and r["crop"] == "potato")
            self.assertEqual(int(fixed_test["usable_pixels"]), 1)
            self.assertEqual(int(aligned_test["usable_pixels"]), 0)
            self.assertEqual(int(aligned_test["pixel_shortfall"]), 400)
            development = json.loads((handoff.parent / "features_development.json").read_text())
            test = json.loads((handoff.parent / "features_test.json").read_text())
            self.assertTrue(all(r["partition"] != "test" for rows in development["rows"].values() for r in rows))
            self.assertTrue(all(r["partition"] == "test" for rows in test["rows"].values() for r in rows))
            self.assertEqual(resolve_stage_pointer(workflow.pointer(root, "features")), handoff)
            self.assertTrue((root / "pipeline_outputs/preprocessing/external_fixture/features/usable_samples.csv").samefile(handoff.parent / "usable_samples.csv"))

    def test_pending_collection_blocks_real_audit_without_mutating_jobs(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            report = workflow.paths(root)[1]
            state = adaptive.root(report) / "rounds/0001/source/state.json"
            save_json(state, {"status": "pending_retry", "task_id": "running"})
            before = state.read_bytes()
            with patch("requests.sessions.Session.request", side_effect=AssertionError("No network")):
                with self.assertRaisesRegex(RuntimeError, "Finish the current candidate batches"):
                    workflow.audit(root)
            self.assertEqual(state.read_bytes(), before)
            self.assertFalse(workflow.paths(root)[2].exists())
            with self.assertRaisesRegex(RuntimeError, "pending"):
                resolve_stage_pointer(workflow.pointer(root, "quality"))

    def test_new_collection_activity_invalidates_snapshot_and_downstream_pointers(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            state = self.fixture(root)
            with patch.object(workflow.external, "audit", side_effect=fake_audit):
                workflow.audit(root)
            save_json(state, {"status": "pending_retry"})
            with self.assertRaisesRegex(RuntimeError, "Collection changed"):
                workflow.select(root)
            with self.assertRaisesRegex(RuntimeError, "pending"):
                resolve_stage_pointer(workflow.pointer(root, "windows"))
            self.assertFalse((workflow.paths(root)[2] / "external_fixture/windows.json").exists())

    def test_window_counts_do_not_read_window_json_and_accept_prior_quality_code(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            with patch.object(workflow.external, "audit", side_effect=fake_audit):
                quality = workflow.audit(root)
            receipt_path = workflow.record_path(quality)
            receipt = json.loads(receipt_path.read_text())
            receipt["files"].update({str(path): digest
                                     for path, digest in workflow._COMPATIBLE_QUALITY_CODE.items()})
            save_json(receipt_path, receipt)
            read_text = Path.read_text

            def reject_window_read(path, *args, **kwargs):
                if path.name == "windows.json":
                    self.fail("Window counting must use the generated rows")
                return read_text(path, *args, **kwargs)

            with patch.object(workflow.external, "select", side_effect=fake_select), patch.object(
                Path, "read_text", reject_window_read
            ):
                windows = workflow.select(root)
            counts = seasons.read_table(windows.with_name("window_counts.csv"))
            self.assertEqual(len(counts), 12)
            row = next(r for r in counts if (r["window"], r["partition"], r["crop"])
                       == ("fixed", "development", "carrot"))
            self.assertEqual((row["complete_pixel_windows_before_cap"], row["parcel_groups"]), ("2", "2"))

    def test_partial_outputs_are_preserved_instead_of_overwritten(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            with patch.object(workflow.external, "audit", side_effect=fake_audit):
                quality = workflow.audit(root)
            incomplete = quality.parent / "windows.json"
            incomplete.write_text("partial output")
            with self.assertRaisesRegex(FileExistsError, "Partial window"):
                workflow.select(root)
            self.assertEqual(incomplete.read_text(), "partial output")

    def test_changed_features_cannot_be_reused(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            with patch.object(workflow.external, "audit", side_effect=fake_audit), patch.object(workflow.external, "select", side_effect=fake_select):
                workflow.audit(root)
                workflow.select(root)
                handoff = workflow.export_features(root)
            (handoff.parent / "features_test.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "output changed"):
                workflow.export_features(root)

    def test_final_counts_apply_existing_twenty_pixel_cap(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)

            def many_pixels(quality, report, *, summarize):
                rows = [{**row, "x": row["x"] + offset} for row in pooled_rows() for offset in range(25)]
                return save_windows(quality, rows, rows, summarize)

            with patch.object(workflow.external, "audit", side_effect=fake_audit), patch.object(workflow.external, "select", side_effect=many_pixels):
                workflow.audit(root)
                workflow.select(root)
                handoff = workflow.export_features(root)
            rows = seasons.read_table(handoff.parent / "usable_samples.csv")
            self.assertTrue(all(int(r["usable_pixels"]) == 20 for r in rows))
            self.assertTrue(all(int(r["usable_parcel_groups"]) == 1 for r in rows))

    def test_collection_change_during_audit_cannot_publish_success(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            state = self.fixture(root)

            def racing_audit(report, output):
                handoff = fake_audit(report, output)
                save_json(state, {"status": "pending_retry"})
                return handoff

            with patch.object(workflow.external, "audit", side_effect=racing_audit):
                with self.assertRaisesRegex(RuntimeError, "changed during auditing"):
                    workflow.audit(root)
            with self.assertRaisesRegex(RuntimeError, "pending"):
                resolve_stage_pointer(workflow.pointer(root, "quality"))

    def test_new_notebooks_run_in_fresh_namespaces_without_explicit_paths(self):
        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            nested = root / "notebooks/preprocessing"
            nested.mkdir(parents=True)
            with chdir(nested), redirect_stdout(io.StringIO()), patch.object(workflow.external, "audit", side_effect=fake_audit), patch.object(workflow.external, "select", side_effect=fake_select), patch("requests.sessions.Session.request", side_effect=AssertionError("No network")):
                for name in ("01_imagery_quality", "02_window_selection", "03_feature_engineering"):
                    book = json.loads((repository / "notebooks/preprocessing" / f"{name}.ipynb").read_text())
                    namespace = {}
                    for cell in book["cells"]:
                        if cell["cell_type"] == "code":
                            exec("".join(cell["source"]), namespace)
            self.assertTrue(resolve_stage_pointer(workflow.pointer(root, "features")).is_file())

    # Historical notebook integration retired; active workflow coverage is separate.
