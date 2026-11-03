"""Progress observes real completions and keeps processing identities unchanged."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from crop_pipeline import progress, progress_observers
from crop_pipeline.sources import cropharvest


class FakeBar:
    def __init__(self, **arguments):
        self.n, self.total = 0, arguments.get("total")
        self.closed = False

    def update(self, amount):
        self.n += amount

    def close(self):
        self.closed = True

    def refresh(self):
        pass

    def set_postfix_str(self, *arguments, **keywords):
        pass

    def set_description(self, *arguments):
        pass


class ProgressTests(unittest.TestCase):
    def tearDown(self):
        progress.finish_notebook()

    def test_unknown_future_work_is_not_presented_as_a_notebook_eta(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(progress, "make_bar", side_effect=FakeBar),
        ):
            monitor = progress.start_notebook(
                "test", ["collect", "train"], temporary, {"input": "one"}
            )
            monitor.begin("collect")
            self.assertIsNone(monitor.remaining())
            monitor.end()
            self.assertIsNone(monitor.remaining())
            monitor.begin("train")
            with progress.WorkCounter(4, "models", "model") as counter:
                counter.update()
                self.assertIsNotNone(monitor.remaining())
            monitor.end()
            monitor.finish()
            self.assertEqual(monitor.remaining(), 0)

    def test_only_successful_comparable_runs_supply_notebook_estimates(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(progress, "make_bar", side_effect=FakeBar),
        ):
            monitor = progress.start_notebook(
                "test", ["work"], temporary, {"input": "one"}
            )
            monitor.begin("work")
            monitor.end()
            monitor.finish()
            self.assertIn(
                "work", progress.load_previous(temporary, "test", monitor.workload_key)
            )
            other = progress.start_notebook(
                "test", ["work"], temporary, {"input": "two"}
            )
            self.assertIsNone(other.remaining())
            other.begin("work")
            other.fail("simulated failure")
            self.assertEqual(
                progress.load_previous(temporary, "test", other.workload_key), {}
            )
            records = [
                json.loads(path.read_text()) for path in Path(temporary).rglob("*.json")
            ]
            self.assertTrue(any(record["status"] == "failed" for record in records))

    def test_observers_preserve_values_and_do_not_count_failed_work(self):
        module = SimpleNamespace(__name__="fixture")
        values = [object(), object()]

        def child(value):
            if value == "fail":
                raise ValueError("original failure")
            return value

        def parent(rows):
            return [module.child(value) for value in rows]

        module.child, module.parent = child, parent
        progress_observers.watch_parent(
            module,
            "parent",
            module,
            "child",
            lambda arguments: len(arguments["rows"]),
            "rows",
            "row",
        )
        bars = []

        def create_bar(**arguments):
            bar = FakeBar(**arguments)
            bars.append(bar)
            return bar

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(progress, "make_bar", side_effect=create_bar),
        ):
            monitor = progress.start_notebook("test", ["work"], temporary, {})
            monitor.begin("work")
            result = module.parent(values)
            self.assertIs(result[0], values[0])
            self.assertEqual(bars[-1].n, 2)
            with self.assertRaisesRegex(ValueError, "original failure"):
                module.parent([values[0], "fail"])
            self.assertEqual(bars[-1].n, 1)
            self.assertTrue(bars[-1].closed)
            self.assertEqual(progress_observers._counters.get(), {})

    def test_progress_only_notebook_edits_keep_processing_fingerprint(self):
        original = {"cells": [{"cell_type": "code", "source": ["threshold = 0.8\n"]}]}
        decorated = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "from crop_pipeline import progress as pipeline_progress\n",
                        "notebook_progress = pipeline_progress.start_notebook('name', [], 'logs', {})\n",
                        "pipeline_progress.begin_stage('work')\n",
                        "threshold = 0.8\n",
                        "pipeline_progress.end_stage()\n",
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "notebook.ipynb"
            path.write_text(json.dumps(original))
            original_fingerprint = cropharvest.notebook_source_fingerprint(path)
            path.write_text(json.dumps(decorated))
            self.assertEqual(
                progress.processing_notebook_fingerprint(path), original_fingerprint
            )
            decorated["cells"][0]["source"][3] = "threshold = 0.7\n"
            path.write_text(json.dumps(decorated))
            self.assertNotEqual(
                progress.processing_notebook_fingerprint(path), original_fingerprint
            )
