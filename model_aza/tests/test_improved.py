import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from crop_pipeline.analysis_workflow import resolve_stage_pointer
from model_aza.improved import workflow
from model_aza.pilots import data

FAST = {**workflow.SELECTION_PARAMS, "n_estimators": 15}


def synthetic(per_crop=12):
    """Seasonal curves per crop; development sources share crops; test rows come from cropharvest."""
    rng = np.random.default_rng(0)
    t = (np.arange(24) + .5) / 24
    sources = {"carrot": ["sen4agrinet", "eurocrops"], "cabbage": ["francecrops"], "potato": ["sen4agrinet", "francecrops"]}
    rows, s2, s1 = [], [], []
    for level, crop in enumerate(workflow.CROPS):
        for i in range(per_crop):
            partition = "test" if i % 6 == 0 else "validation" if i % 4 == 0 else "train"
            source = "cropharvest" if partition == "test" else sources[crop][i % len(sources[crop])]
            group = f"{partition}-{crop}-{i}"
            peak = .3 + .2 * level
            for _ in range(2):
                values = np.full((24, 5), .05)
                values[:, 3] = .2 + .4 * np.exp(-((t - peak) ** 2) / .01) + rng.normal(0, .01, 24)
                rows.append({"partition_group": group, "crop": crop, "partition": partition, "source_dataset": source,
                             "label_status": "reported", "site_cell": f"45,{i % 6}", "region_id": group,
                             "x": float(len(rows)), "y": 0., "source_crs": "EPSG:32631", "year": 2021})
                s2.append(values)
                s1.append(np.stack([np.full(24, -10. - level), np.full(24, -18. - level)], axis=1) + rng.normal(0, .3, (24, 2)))
    return rows, np.asarray(s2), np.ones((len(rows), 24), bool), np.asarray(s1)


class PureStepTests(unittest.TestCase):
    def test_selection_never_depends_on_test_rows(self):
        rows, s2, valid, s1 = synthetic()
        candidates = {k: workflow.CANDIDATES[k] for k in ("s2_harmonic", "s2_harmonic_all_radar")}
        first = workflow.candidate_scores(rows, s2, valid, s1, candidates, seeds=(1,), params=FAST, progress=None)
        noisy = s2.copy()
        noisy[[r["partition"] == "test" for r in rows]] = 9.
        second = workflow.candidate_scores(rows, noisy, valid, s1, candidates, seeds=(1,), params=FAST, progress=None)
        self.assertEqual(first, second)
        self.assertTrue(all(0 <= r["dataset_holdout_macro_f1"] <= 1 for r in first))

    def test_choose_uses_transfer_views_and_prefers_fewer_predictors_on_ties(self):
        records = [{"candidate": "a", "seed": 1, "predictors": 80, "spatial_cv_macro_f1": .8, "dataset_holdout_macro_f1": .6},
                   {"candidate": "b", "seed": 1, "predictors": 60, "spatial_cv_macro_f1": .6, "dataset_holdout_macro_f1": .8},
                   {"candidate": "c", "seed": 1, "predictors": 90, "spatial_cv_macro_f1": .9, "dataset_holdout_macro_f1": .4}]
        best, summary = workflow.choose(records)
        self.assertEqual(best, "b")
        self.assertAlmostEqual(summary["a"]["selection_score"], .7)

    def test_final_model_tests_only_unseen_parcels_and_classifies_with_confidence(self):
        rows, s2, valid, s1 = synthetic()
        model, columns, groups = workflow.fit_final(rows, s2, valid, s1, "s2_harmonic", params=FAST)
        self.assertFalse(any(g.startswith("test") for g in groups))
        results, predictions, baseline = workflow.score_test(model, "s2_harmonic", rows, s2, valid, s1, groups)
        self.assertEqual({p["partition_group"][:4] for p in predictions}, {"test"})
        self.assertIn(baseline["training_majority_class"], workflow.CROPS)
        self.assertEqual(list(results), ["jecam"])
        # A single-crop supplement is scored on its own, without a three-class macro-F1.
        supplement = [dict(r) for r in rows]
        for r in supplement:
            if r["partition"] == "test" and r["crop"] == "carrot":
                r["test_cohort"] = "california_dwr_carrot_2023_v1"
        results, _, _ = workflow.score_test(model, "s2_harmonic", supplement, s2, valid, s1, groups)
        self.assertEqual(list(results), ["jecam", "california_dwr_carrot_2023_v1", "combined"])
        self.assertIsNone(results["california_dwr_carrot_2023_v1"]["metrics"]["macro_f1"])
        self.assertIsNone(results["california_dwr_carrot_2023_v1"]["meets_f1_criteria"])
        self.assertEqual(results["combined"]["crop_coverage"], "complete")
        self.assertEqual(results["combined"]["parcels"],
                         results["jecam"]["parcels"] + results["california_dwr_carrot_2023_v1"]["parcels"])
        with self.assertRaises(ValueError):
            workflow.score_test(model, "s2_harmonic", rows, s2, valid, s1, groups + [predictions[0]["partition_group"]])
        no_clear = valid.copy()
        no_clear[0] = False
        labels, probabilities, accepted = workflow.classify(model, "s2_harmonic", s2, no_clear, s1)
        self.assertEqual(labels[0], "No features")
        self.assertEqual(probabilities.shape, (len(rows), 3))
        self.assertTrue(set(labels) <= set(workflow.MAP_CODES))


class StageTests(unittest.TestCase):
    def test_stages_seal_publish_and_reject_changed_files(self):
        rows, s2, valid, s1 = synthetic()
        original = dict(workflow.CANDIDATES)
        workflow.CANDIDATES.clear()
        workflow.CANDIDATES.update({k: original[k] for k in ("s2_harmonic", "s2_harmonic_all_radar")})
        final, selection = workflow.FINAL_PARAMS, workflow.SELECTION_PARAMS
        workflow.FINAL_PARAMS, workflow.SELECTION_PARAMS = FAST, FAST
        try:
            with TemporaryDirectory() as folder:
                run = workflow._new_run(folder, "inputs")
                data.save(run, rows, {"s2_values": s2, "s2_valid": valid, "s1_values": s1})
                workflow._publish(folder, "inputs", workflow.seal(run, "inputs", pixels=len(rows)))
                selected = workflow.select(folder, seeds=(1,), progress=None)
                self.assertFalse(json.loads((selected.parent / "selection.json").read_text())["test_rows_used"])
                models = workflow.freeze(folder)
                tested = workflow.test(folder)
                scores = json.loads((tested.parent / "test_scores.json").read_text())
                self.assertIn("meets_f1_criteria", scores["populations"]["jecam"])
                pointer = Path(folder) / "pipeline_outputs/improved/current_models.json"
                self.assertEqual(resolve_stage_pointer(pointer), models.resolve())
                (models.parent / "model.json").write_text("{}")
                with self.assertRaises(ValueError):
                    workflow.test(folder)
        finally:
            workflow.CANDIDATES.clear()
            workflow.CANDIDATES.update(original)
            workflow.FINAL_PARAMS, workflow.SELECTION_PARAMS = final, selection


if __name__ == "__main__":
    unittest.main()
