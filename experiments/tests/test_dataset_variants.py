import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from crop_pipeline.analysis_workflow import resolve_stage_pointer
from experiments.improved_v1 import datasets, workflow
from experiments.pilots import data
from experiments.tests.test_improved import FAST, synthetic


def origins_for(rows):
    """Every third development group per crop is groupmate-only, the next yours-only, the rest shared."""
    origins = {}
    for r in rows:
        if r["partition"] == "test" or r["partition_group"] in origins:
            continue
        index = int(r["partition_group"].rsplit("-", 1)[1])
        origins[r["partition_group"]] = ("groupmate_only", "yours_only", "shared")[index % 3]
    return origins


class OriginTests(unittest.TestCase):
    def test_classify_origins(self):
        origins = datasets.classify_origins({"a", "b"}, {"b", "c"})
        self.assertEqual(origins, {"a": "groupmate_only", "b": "shared", "c": "yours_only"})

    def test_groups_are_read_from_package_bundles_and_campaign_batches(self):
        with TemporaryDirectory() as folder:
            package, campaign = Path(folder) / "package", Path(folder) / "campaign"
            (package / "sen4agrinet/x").mkdir(parents=True)
            (package / "package.json").write_text(json.dumps({"bundles": [{"path": "sen4agrinet/x"}]}))
            (package / "sen4agrinet/x/bundle.json").write_text(json.dumps({"selection": {"groups": ["g1", "g2"]}}))
            batch = campaign / "829e0cec79b8/rounds/0001/sen4agrinet_round_0001"
            batch.mkdir(parents=True)
            (batch / "selection.json").write_text(json.dumps({"groups": ["g2", "g3"]}))
            # Superseded batches are not part of the collection.
            old = campaign / "829e0cec79b8/superseded/round_0012/x"
            old.mkdir(parents=True)
            (old / "selection.json").write_text(json.dumps({"groups": ["g9"]}))
            self.assertEqual(datasets.groupmate_groups(package), {"g1", "g2"})
            self.assertEqual(datasets.your_groups(campaign), {"g2", "g3"})


class MaskTests(unittest.TestCase):
    def test_variants_drop_only_development_groups_and_keep_every_test_row(self):
        rows, *_ = synthetic()
        origins = origins_for(rows)
        tests = [r["partition_group"] for r in rows if r["partition"] == "test"]
        kept = {}
        for variant in datasets.VARIANTS:
            mask = datasets.variant_mask(rows, origins, variant)
            chosen = [r for r, m in zip(rows, mask) if m]
            self.assertEqual([r["partition_group"] for r in chosen if r["partition"] == "test"], tests)
            kept[variant] = {r["partition_group"] for r in chosen if r["partition"] != "test"}
            self.assertTrue(all(origins[g] in datasets.VARIANTS[variant] for g in kept[variant]))
        self.assertEqual(kept["combined"], kept["groupmate"] | kept["yours"])
        self.assertEqual(kept["groupmate"] & kept["yours"], {g for g, o in origins.items() if o == "shared"})
        # Whole parcel groups move together: no group is split by a variant.
        for variant in datasets.VARIANTS:
            mask = datasets.variant_mask(rows, origins, variant)
            by_group = {}
            for r, m in zip(rows, mask):
                by_group.setdefault(r["partition_group"], set()).add(bool(m))
            self.assertTrue(all(len(v) == 1 for v in by_group.values()))

    def test_unknown_origin_or_missing_crop_is_refused(self):
        rows, *_ = synthetic()
        origins = origins_for(rows)
        missing = dict(origins)
        missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):
            datasets.variant_mask(rows, missing, "combined")
        with self.assertRaises(ValueError):
            datasets.variant_mask(rows, origins, "his")
        no_cabbage = {g: ("groupmate_only" if "cabbage" in g else o) for g, o in origins.items()}
        with self.assertRaises(ValueError):
            datasets.variant_mask(rows, no_cabbage, "yours")

    def test_summary_counts_parcels_per_variant(self):
        rows, *_ = synthetic()
        origins = origins_for(rows)
        table = datasets.summary(rows, origins)
        total = {v: sum(r["parcels"] for r in table if r["variant"] == v) for v in datasets.VARIANTS}
        development = {r["partition_group"] for r in rows if r["partition"] != "test"}
        self.assertEqual(total["combined"], len(development))
        self.assertEqual(total["groupmate"] + total["yours"] - total["combined"],
                         sum(o == "shared" for o in origins.values()))


class VariantStageTests(unittest.TestCase):
    def test_each_variant_has_its_own_pointers_and_same_test_parcels(self):
        rows, s2, valid, s1 = synthetic(per_crop=18)
        # Spread sites so every variant still has enough cells for five spatial folds.
        for r in rows:
            r["site_cell"] = f"45,{int(r['partition_group'].rsplit('-', 1)[1]) % 10}"
        original = dict(workflow.CANDIDATES)
        workflow.CANDIDATES.clear()
        workflow.CANDIDATES.update({"s2_harmonic": original["s2_harmonic"]})
        final, selection = workflow.FINAL_PARAMS, workflow.SELECTION_PARAMS
        workflow.FINAL_PARAMS, workflow.SELECTION_PARAMS = FAST, FAST
        try:
            with TemporaryDirectory() as folder:
                run = workflow._new_run(folder, "inputs")
                data.save(run, rows, {"s2_values": s2, "s2_valid": valid, "s1_values": s1})
                workflow._publish(folder, "inputs", workflow.seal(run, "inputs", pixels=len(rows)))
                origins_run = workflow._new_run(folder, "origins")
                (origins_run / "origins.json").write_text(json.dumps(origins_for(rows)))
                workflow._publish(folder, "origins", workflow.seal(origins_run, "origins"))
                tested, groups = {}, {}
                for variant in datasets.VARIANTS:
                    workflow.select(folder, seeds=(1,), progress=None, variant=variant)
                    models = workflow.freeze(folder, variant=variant)
                    groups[variant] = set(json.loads((models.parent / "model.json").read_text())["development_groups"])
                    result = workflow.test(folder, variant=variant)
                    tested[variant] = json.loads((result.parent / "test_scores.json").read_text())
                    self.assertEqual(tested[variant]["dataset_variant"], variant)
                    pointer = Path(folder) / f"pipeline_outputs/improved/current_models__{variant}.json"
                    self.assertEqual(resolve_stage_pointer(pointer), models.resolve())
                self.assertFalse((Path(folder) / "pipeline_outputs/improved/current_models.json").exists())
                self.assertEqual(groups["combined"], groups["groupmate"] | groups["yours"])
                self.assertLess(len(groups["groupmate"]), len(groups["combined"]))
                parcels = {v: t["populations"]["jecam"]["parcels"] for v, t in tested.items()}
                self.assertEqual(len(set(parcels.values())), 1)
                # A model cannot be tested under another variant's name.
                with self.assertRaises(ValueError):
                    workflow.test(folder, models_handoff=str(resolve_stage_pointer(
                        Path(folder) / "pipeline_outputs/improved/current_models__yours.json")), variant="groupmate")
        finally:
            workflow.CANDIDATES.clear()
            workflow.CANDIDATES.update(original)
            workflow.FINAL_PARAMS, workflow.SELECTION_PARAMS = final, selection


if __name__ == "__main__":
    unittest.main()
