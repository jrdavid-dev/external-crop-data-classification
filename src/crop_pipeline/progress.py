"""Notebook progress and measured timing, without changing processing decisions.

Stage counts show workflow completion, not percentages of time. A whole-notebook
ETA is available only from a comparable completed run or measurable final work;
otherwise it remains explicitly unknown rather than guessing future stages.
"""

import ast
import hashlib
import json
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

_active_monitor = None


def duration_text(seconds):
    if seconds is None:
        return "learning"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


def notebook_shell():
    try:
        from IPython import get_ipython

        shell = get_ipython()
    except ImportError:
        return None
    return shell


def make_bar(**arguments):
    from tqdm.auto import tqdm

    shell = notebook_shell()
    interactive = shell is not None or sys.stderr.isatty()
    return tqdm(disable=not interactive, mininterval=0.5, **arguments)


class NotebookProgress:
    def __init__(self, name, stages, folder, workload):
        self.name, self.stages = name, list(stages)
        if len(set(self.stages)) != len(self.stages):
            raise ValueError("Progress stage names must be unique.")
        self.folder, self.workload = Path(folder), workload
        self.run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "_"
            + uuid.uuid4().hex[:8]
        )
        self.started = time.monotonic()
        self.current, self.stage_started = None, None
        self.last_work_remaining = None
        self.results, self.previous = {}, {}
        self.item_bars = []
        self.hook = None
        self.stop_heartbeat = threading.Event()
        self.bar = make_bar(total=len(stages), desc=name, unit="stage", leave=True)
        self.set_workload(workload)
        from crop_pipeline import progress_observers

        progress_observers.enable()
        self.attach_error_hook()
        self.refresh()
        self.start_heartbeat()

    def start_heartbeat(self):
        if notebook_shell() is not None:
            threading.Thread(target=self.heartbeat, daemon=True).start()

    def heartbeat(self):
        while not self.stop_heartbeat.wait(1):
            self.refresh()
            for bar in list(self.item_bars):
                bar.refresh()

    def set_workload(self, workload):
        self.workload = workload
        self.workload_key = hashlib.sha256(
            json.dumps(workload, sort_keys=True).encode()
        ).hexdigest()[:20]
        self.previous = load_previous(self.folder, self.name, self.workload_key)

    def attach_error_hook(self):
        shell = notebook_shell()
        if shell is None or not hasattr(shell, "events"):
            return
        self.hook = lambda result: (
            self.fail(str(result.error_in_exec or result.error_before_exec))
            if (result.error_in_exec or result.error_before_exec)
            else None
        )
        shell.events.register("post_run_cell", self.hook)

    def begin(self, stage):
        if stage not in self.stages:
            raise ValueError(f"Unknown progress stage: {stage}")
        self.current, self.stage_started = stage, time.monotonic()
        self.last_work_remaining = None
        self.refresh()

    def end(self, status="completed"):
        if self.current is None:
            return
        self.results[self.current] = {
            "status": status,
            "seconds": time.monotonic() - self.stage_started,
        }
        self.current = None
        self.bar.n = sum(
            result["status"] in {"completed", "skipped"}
            for result in self.results.values()
        )
        self.refresh()
        self.save("running")

    def remaining(self):
        unfinished = [
            stage
            for stage in self.stages
            if self.results.get(stage, {}).get("status") not in {"completed", "skipped"}
        ]
        if not unfinished:
            return 0
        if unfinished == [self.current] and self.last_work_remaining is not None:
            return self.last_work_remaining
        if any(stage not in self.previous for stage in unfinished):
            return None
        estimate = sum(self.previous[stage] for stage in unfinished)
        elapsed = (
            time.monotonic() - self.stage_started
            if self.current and self.stage_started is not None
            else 0
        )
        return max(0, estimate - elapsed)

    def refresh(self):
        basis = (
            "previous comparable run; rough"
            if self.previous
            else "future stages not yet timed"
        )
        if self.current == self.stages[-1] and self.last_work_remaining is not None:
            basis = "current throughput; rough"
        text = f"elapsed {duration_text(time.monotonic()-self.started)}; notebook ETA {duration_text(self.remaining())} ({basis})"
        self.bar.set_postfix_str(text, refresh=False)
        self.bar.refresh()

    def fail(self, error):
        if self.current:
            self.results[self.current] = {
                "status": "failed",
                "seconds": time.monotonic() - self.stage_started,
                "error": error,
            }
        self.save("failed")
        self.current = None
        self.bar.set_description(self.name + " (stopped; fix the error and rerun)")

    def save(self, status):
        from crop_pipeline.common import save_json

        save_json(
            self.folder / self.name / "runs" / f"{self.run_id}.json",
            {
                "name": self.name,
                "status": status,
                "workload_key": self.workload_key,
                "workload": self.workload,
                "elapsed_seconds": time.monotonic() - self.started,
                "stages": self.results,
            },
        )

    def finish(self):
        complete = all(
            self.results.get(stage, {}).get("status") in {"completed", "skipped"}
            for stage in self.stages
        )
        self.save("completed" if complete else "incomplete")
        print(
            f"{self.name}: {duration_text(time.monotonic()-self.started)} elapsed; timing report: {self.folder / self.name / 'runs' / (self.run_id+'.json')}"
        )
        self.close()

    def close(self):
        self.stop_heartbeat.set()
        shell = notebook_shell()
        if self.hook is not None and shell is not None:
            shell.events.unregister("post_run_cell", self.hook)
            self.hook = None
        self.bar.close()


def load_previous(folder, name, workload_key):
    for path in sorted((Path(folder) / name / "runs").glob("*.json"), reverse=True):
        record = json.loads(path.read_text())
        if (
            record.get("status") == "completed"
            and record.get("workload_key") == workload_key
        ):
            return {
                stage: result["seconds"] for stage, result in record["stages"].items()
            }
    return {}


def start_notebook(name, stages, folder, workload):
    global _active_monitor
    if _active_monitor is not None:
        _active_monitor.close()
    _active_monitor = NotebookProgress(name, stages, folder, workload)
    return _active_monitor


def begin_stage(name):
    global _active_monitor
    if _active_monitor is not None and name not in _active_monitor.stages:
        _active_monitor.close()
        _active_monitor = None
    if _active_monitor is not None:
        _active_monitor.begin(name)


def end_stage(status="completed"):
    if _active_monitor is not None:
        _active_monitor.end(status)


def finish_notebook():
    global _active_monitor
    if _active_monitor is not None:
        _active_monitor.finish()
        _active_monitor = None


def set_workload(workload):
    if _active_monitor is not None:
        _active_monitor.set_workload(workload)


def message(text):
    """Keep successful per-item messages from overwhelming notebook progress output."""
    if _active_monitor is None:
        print(text)
    elif _active_monitor.item_bars:
        _active_monitor.item_bars[-1].set_postfix_str(str(text), refresh=False)


def track(items, description, total=None, unit="item", size_of=None):
    """Yield unchanged records; advance only after the caller finishes their work."""
    monitor = _active_monitor
    if monitor is None:
        yield from items
        return
    total = len(items) if total is None and hasattr(items, "__len__") else total
    bar = make_bar(
        total=total, desc=description, unit=unit, leave=False, unit_scale=unit == "B"
    )
    monitor.item_bars.append(bar)
    try:
        for item in items:
            yield item
            bar.update(size_of(item) if size_of else 1)
            monitor.refresh()
    finally:
        monitor.item_bars.remove(bar)
        bar.close()


def active_monitor():
    return _active_monitor


class WorkCounter:
    """Count finished work, including opaque operations such as a whole forest fit."""

    def __init__(self, total, description, unit):
        self.monitor = active_monitor()
        self.started = time.monotonic()
        self.total = total
        self.bar = make_bar(total=total, desc=description, unit=unit, leave=False)
        self.primary = not self.monitor.item_bars

    def __enter__(self):
        self.monitor.item_bars.append(self.bar)
        return self

    def update(self):
        self.bar.update(1)
        if self.primary and self.total is not None and self.bar.n:
            self.monitor.last_work_remaining = (
                (time.monotonic() - self.started)
                / self.bar.n
                * max(0, self.total - self.bar.n)
            )
        self.monitor.refresh()

    def __exit__(self, error_type, error, traceback):
        self.monitor.item_bars.remove(self.bar)
        self.bar.close()
        return False


class ProgressOnlyStatements(ast.NodeTransformer):
    """Keep existing processing hashes stable when only notebook UI statements change."""

    def visit_ImportFrom(self, node):
        if node.module == "crop_pipeline":
            node.names = [name for name in node.names if name.name != "progress"]
            return node if node.names else None
        return node

    def visit_Assign(self, node):
        if any(
            isinstance(target, ast.Name) and target.id == "notebook_progress"
            for target in node.targets
        ):
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        function = node.value.func if isinstance(node.value, ast.Call) else None
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "pipeline_progress"
        ):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "pipeline_progress"
            and function.attr == "processing_notebook_fingerprint"
        ):
            node.func = ast.Attribute(
                value=ast.Name(id="cropharvest_source", ctx=ast.Load()),
                attr="notebook_source_fingerprint",
                ctx=ast.Load(),
            )
        return self.generic_visit(node)


def processing_notebook_fingerprint(path):
    """Use the existing runtime-switch exclusions plus explicitly identified progress UI."""
    from crop_pipeline.sources import cropharvest
    import inspect

    function = ast.parse(
        inspect.getsource(cropharvest.notebook_source_fingerprint)
    ).body[0]
    switch_assignment = next(
        statement
        for statement in function.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "runtime_switches"
            for target in statement.targets
        )
    )
    runtime_switches = ast.literal_eval(switch_assignment.value)
    normalized = []
    for cell in json.loads(Path(path).read_text())["cells"]:
        if cell["cell_type"] == "code":
            normalized.append(normalized_cell_source(cell, runtime_switches))
    return hashlib.sha256(json.dumps(normalized).encode()).hexdigest()


def normalized_cell_source(cell, runtime_switches):
    tree = ProgressOnlyStatements().visit(ast.parse("".join(cell["source"])))
    tree.body = [
        statement
        for statement in tree.body
        if not (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in runtime_switches
                for target in statement.targets
            )
        )
    ]
    return ast.dump(tree, include_attributes=False)
