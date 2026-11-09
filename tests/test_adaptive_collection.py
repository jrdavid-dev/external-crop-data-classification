"""Sampling budgets are measured after QC and do not move held-out parcels."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crop_pipeline.common import save_json
from crop_pipeline.sources import adaptive, annual
from test_annual_archive import reference_feature


def policy():
    result = copy.deepcopy(adaptive.DEFAULT_POLICY)
    result["candidate_parcels_per_crop_per_source"] = 3
    result["pixels_per_parcel"] = 2
    result["targets"] = {"train": {"pixels": 4, "parcels": 2},
                         "validation": {"pixels": 2, "parcels": 1},
                         "test": {"pixels": 2, "parcels": 1}}
    return result


def candidates():
    return [reference_feature(f"{crop}_{i}", left=500000 + i * 500,
                              width=100, crop=crop)
            for crop in annual.TARGET_CROPS for i in range(20)]


class AdaptiveCollectionTests(unittest.TestCase):
    def test_small_batch_is_stable_and_reserved_upstream_fields_are_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "inventory"
            features = candidates()
            features[0]["properties"]["source_split"] = "test"
            with patch.object(adaptive.annual, "prepare_source", return_value={"estimate": {}, "run_folder": str(Path(folder) / "mock_run")}) as prepare:
                first, context = adaptive.prepare_batch("sen4agrinet", features, folder, report, {}, policy=policy())
                chosen = [f for f in features if f["properties"]["partition_group"] in prepare.call_args.kwargs["candidate_groups"]]
                adaptive.prepare_batch("sen4agrinet", list(reversed(features)), folder, report, {}, policy=policy())
                again = [f for f in features if f["properties"]["partition_group"] in prepare.call_args.kwargs["candidate_groups"]]
            self.assertEqual([r["properties"]["partition_group"] for r in chosen],
                             [r["properties"]["partition_group"] for r in again])
            self.assertEqual(len(chosen), 9)
            self.assertNotIn(features[0], chosen)
            self.assertEqual(context["candidate_parcels"], 9)

    def test_crop_filter_keeps_cabbage_as_reference_but_does_not_select_it(self):
        with tempfile.TemporaryDirectory() as folder:
            features = candidates()
            with patch.object(adaptive.annual, "prepare_source", return_value={"estimate": {}, "run_folder": str(Path(folder)/"mock_run")}) as prepare:
                _, context = adaptive.prepare_batch("sen4agrinet", features, folder, Path(folder)/"inventory", {},
                    policy=policy(), candidate_crops=("carrot", "potato"))
            groups = set(prepare.call_args.kwargs["candidate_groups"])
            self.assertEqual(context["candidate_parcels"], 6)
            self.assertEqual({f["properties"]["crop"] for f in features if f["properties"]["partition_group"] in groups}, {"carrot", "potato"})
            self.assertEqual(prepare.call_args.args[1], features)

    def test_top_up_only_deficient_crop_partition_and_never_repeats_a_parcel(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "inventory"
            p = policy()
            features = candidates()
            with patch.object(adaptive.annual, "prepare_source", return_value={"estimate": {}, "run_folder": str(Path(folder) / "mock_run")}) as prepare:
                _, context = adaptive.prepare_batch("sen4agrinet", features, folder, report, {}, policy=p)
                original = set(prepare.call_args.kwargs["candidate_groups"])
                counts = {crop: {partition: {"target_met": True} for partition in p["targets"]}
                          for crop in annual.TARGET_CROPS}
                counts["carrot"]["train"]["target_met"] = False
                save_json(Path(context["campaign"]) / "feedback.json",
                          {"counts": counts, "targets_met": False, "next_round": 2})
                adaptive.prepare_batch("sen4agrinet", features, folder, report, {}, policy=p)
                selected = [f for f in features if f["properties"]["partition_group"] in prepare.call_args.kwargs["candidate_groups"]]
            self.assertTrue(selected)
            for f in selected:
                props = f["properties"]
                self.assertEqual(props["crop"], "carrot")
                self.assertEqual(adaptive.partition_for(adaptive.physical_key("sen4agrinet", props)), "train")
                self.assertNotIn(props["partition_group"], original)

    def test_targets_stop_new_collection_and_training_is_gated(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "inventory"
            campaign = adaptive.root(report, policy())
            save_json(campaign / "feedback.json", {"targets_met": True, "next_round": 2})
            with patch.object(adaptive.annual, "prepare_source", side_effect=AssertionError("no preparation")):
                plan, context = adaptive.prepare_batch("sen4agrinet", candidates(), folder, report, {}, policy=policy())
            self.assertIsNone(plan)
            self.assertEqual(adaptive.collect_batch(plan, context, "", enabled=True)["status"], "skipped")
            save_json(campaign / "feedback.json", {"targets_met": False})
            with self.assertRaisesRegex(RuntimeError, "not met"):
                adaptive.require_targets(report, policy())

    def test_quality_counts_cap_pixels_per_field_and_preserve_locked_partitions(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "inventory"
            p = policy()
            campaign = adaptive.root(report, p)
            locked = {f"{crop}_{partition}_{i}": partition for crop in annual.TARGET_CROPS
                      for partition in p["targets"] for i in range(p["targets"][partition]["parcels"])}
            save_json(campaign / "rounds/0001/sen4agrinet/selection.json", {"assignments": locked})
            save_json(campaign / "rounds/0001/sen4agrinet/completed.json", {"round": 1})
            rows = []
            for number, (group, partition) in enumerate(locked.items()):
                crop = group.split('_')[0]
                for pixel in range(6):
                    rows.append({"crop": crop, "partition_group": group, "source_crs": "EPSG:32631",
                                 "x": number * 1000 + pixel * 10, "y": 5200000,
                                 "timing_review_status": "automated_pass", "automatic_timing_status": "passed",
                                 "automatic_policy_fingerprint": "policy"})
            accepted, estimated, summary = adaptive.select_usable([], rows, report, p)
            feedback = json.loads((campaign / "feedback.json").read_text())
            self.assertTrue(feedback["targets_met"])
            self.assertEqual(len(estimated), 24)
            self.assertTrue(all(row["target_met"] for row in summary))
            self.assertEqual(adaptive.assignments(report, p), locked)
            # Invalid timing outcomes cannot make the target look complete.
            for row in rows:
                row["automatic_timing_status"] = "failed"
            adaptive.select_usable([], rows, report, p)
            self.assertFalse(json.loads((campaign / "feedback.json").read_text())["targets_met"])

    def test_deleted_masks_are_rebuilt_by_source_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            features = [reference_feature(width=100)]
            first = annual.prepare_source("sen4agrinet", features, folder, {}, inward_buffer_metres=0)
            regions = annual.read_regions(first["years"]["2020"])
            path = Path(regions[0]["properties"]["sampling_mask"])
            before = path.read_bytes()
            path.unlink()
            again = annual.prepare_source("sen4agrinet", features, folder, {}, inward_buffer_metres=0)
            self.assertEqual(first["revision"], again["revision"])
            self.assertEqual(path.read_bytes(), before)

    def test_completed_batch_exports_annual_files_and_resolves_cumulative_handoffs(self):
        import pandas as pd
        from crop_pipeline.imagery import pixel_tables, sampling_mask, batch_export, timelines
        from crop_pipeline.imagery.accelerated import AcceleratedRetriever
        from crop_pipeline.sources import annual_batch, pilot_sources
        from crop_pipeline.windows import annual_archive
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "inventory"
            p = policy()
            source_plan, context = adaptive.prepare_batch("sen4agrinet", [reference_feature(width=50)],
                Path(folder) / "archive", report, {}, inward_buffer_metres=0, policy=p)
            inputs = source_plan["years"]["2020"]
            region = annual.read_regions(inputs)[0]
            retriever = AcceleratedRetriever(None, 2020, inputs["imagery_folder"], inputs["report_folder"])
            plan = {"fragment_id": region["properties"]["fragment_id"], "fragment": region,
                    "settings": retriever.request_settings(region), "acquisitions": [],
                    "download_intervals": annual.quality.interval_bins(2020)}
            retriever.inventory_region = lambda region: plan
            selection = sampling_mask.load_selection(plan)
            rows = []
            for interval in plan["download_intervals"]:
                for identifier in selection["selected_ids"]:
                    y, x = divmod(identifier, plan["settings"]["grid"]["width"])
                    rows.append({"region_id": plan["fragment_id"], "interval": interval["name"], "x": x+.5, "y": y+.5,
                                 **{band:1000. for band in retriever.spectral_bands}, "clear_count": 1.})
            pixel_tables.write_tables(retriever, {plan["fragment_id"]: plan}, pd.DataFrame(rows), [plan["fragment_id"]])
            daily_folder = Path(inputs["timeline_folder"]) / "tiles" / plan["fragment_id"]
            contract = timelines.summary_contract(plan, selection, .8)
            save_json(timelines.checkpoint_path(daily_folder, plan["fragment_id"], contract),
                      {"contract": contract, "rows": []})
            completed = annual_batch.finalize_tables(None, source_plan, "2020", inputs, retriever)
            registry = Path(folder) / "registry.json"
            save_json(registry, {"handoffs": [{"path": completed["handoff"],
                                              "sha256": annual.file_sha256(completed["handoff"])}]})
            with patch.object(adaptive.annual, "collect_source", return_value={"status":"completed", "registry": str(registry)}):
                adaptive.collect_batch(source_plan, context, "project", enabled=True)
            paths, analysis = adaptive.resolve_collection(report, p)
            self.assertEqual(len(paths), 1)
            self.assertTrue(analysis.startswith("adaptive_training_"))
            source = pilot_sources.load_sources(paths)[0]
            self.assertEqual(len(source["annual_tif_index"]), 1)
            self.assertTrue(Path(next(iter(source["annual_tif_index"].values()))["path"]).is_file())
            audited = annual_archive.audit_source(source, Path(folder) / "analysis", .8)
            self.assertEqual(len(audited[1]), 24)

    def test_unselected_neighbors_still_exclude_contested_training_pixels(self):
        import numpy as np
        import rasterio
        with tempfile.TemporaryDirectory() as folder:
            first = reference_feature("candidate", width=300)
            neighbor = reference_feature("not_selected", left=500150, width=300, crop="potato")
            # Reserved references must also remain in the overlap screen.
            neighbor["properties"]["source_split"] = "test"
            full = annual.prepare_source("sen4agrinet", [first, neighbor], Path(folder)/"full", {}, inward_buffer_metres=0)
            selected = annual.prepare_source("sen4agrinet", [first, neighbor], Path(folder)/"selected", {},
                inward_buffer_metres=0, candidate_groups=[first["properties"]["partition_group"]])
            all_regions = annual.read_regions(full["years"]["2020"])
            candidate_regions = [r for r in all_regions if r["properties"]["parcel_id"] == "candidate"]
            actual = annual.read_regions(selected["years"]["2020"])
            self.assertEqual(len(actual), len(candidate_regions))
            for left, right in zip(actual, candidate_regions):
                with rasterio.open(left["properties"]["sampling_mask"]) as a, rasterio.open(right["properties"]["sampling_mask"]) as b:
                    self.assertEqual(a.transform, b.transform)
                    np.testing.assert_array_equal(a.read(), b.read())

    def test_overnight_extension_keeps_active_batch_and_adds_only_missing_candidates(self):
        from crop_pipeline.imagery.batch_export import content_key
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "inventory"
            features = [reference_feature(f"{crop}_{i}", left=500000+i*500, width=100, crop=crop)
                        for crop in ("cabbage", "potato") for i in range(120)]
            def prepare(source, catalog, archive, provenance, **kwargs):
                run = Path(folder) / "runs" / content_key(kwargs["candidate_groups"])
                plan = {"run_folder": str(run), "estimate": {}, "years": {}}
                save_json(run / "collection_plan.json", plan)
                return plan
            with patch.object(adaptive.annual, "prepare_source", side_effect=prepare):
                original, context = adaptive.prepare_batch("francecrops", features, folder, report, {})
                state_path = Path(context["folder"]) / "state.json"
                save_json(state_path, {"status": "pending_retry"})
                # Simulate the previously shipped context, whose plan path came from its pass report.
                del context["plan_path"]
                save_json(Path(context["folder"]) / "context.json", context)
                queue = Path(original["run_folder"]) / "collection_attempts/attempt/retry_queue.json"
                save_json(state_path, {"status":"pending_retry", "retry_queue":str(queue)})
                batches = adaptive.prepare_round_batches("francecrops", features, folder, report, {}, parcels_per_crop=100)
                again = adaptive.prepare_round_batches("francecrops", features, folder, report, {}, parcels_per_crop=100)
            self.assertEqual(len(batches), 2)
            self.assertEqual(batches[0][0], original)
            self.assertEqual(sum(c["candidate_parcels"] for _,c in batches), 200)
            old = json.loads((Path(context["folder"])/"selection.json").read_text())
            new = json.loads((Path(batches[1][1]["folder"])/"selection.json").read_text())
            self.assertFalse(set(old["physical_keys"]) & set(new["physical_keys"]))
            self.assertEqual(len(new["groups"]), 150)
            self.assertEqual(context["campaign"], batches[1][1]["campaign"])
            self.assertEqual(again, batches)
            self.assertTrue(all(old["assignments"][group] == adaptive.assignments(report)[group] for group in old["groups"]))

    def test_multi_batch_collection_resumes_each_plan_and_authenticates_once(self):
        batches = [({"run_folder":"old"}, {"folder":"/old"}),
                   ({"run_folder":"extra"}, {"folder":"/extra"})]
        with patch.object(adaptive, "collect_batch", side_effect=[
                {"status":"pending_retry", "progress_by_year":{"2022":{"tiles":1}}},
                {"status":"completed", "progress_by_year":{"2022":{"tiles":2}}}]) as collect:
            result = adaptive.collect_batches(batches, "project", enabled=True, authenticate=True)
        self.assertEqual(collect.call_count, 2)
        self.assertTrue(collect.call_args_list[0].kwargs["authenticate"])
        self.assertFalse(collect.call_args_list[1].kwargs["authenticate"])
        self.assertEqual(result["status"], "pending_retry")
        self.assertEqual(set(result["progress_by_year"]), {"old/2022","extra/2022"})
