"""Notebook progress and measured timing, without changing processing decisions.

Stage counts show workflow completion, not percentages of time. A whole-notebook
ETA is available only from a comparable completed run or measurable final work;
otherwise it remains explicitly unknown rather than guessing future stages.
"""

import ast
import hashlib
import io
from contextlib import nullcontext, redirect_stdout
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
        self.visible_work = None
        self.displayed_work = None
        self.output = ProgressOutput(self)
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
        self.visible_work = None
        self.refresh()

    def end(self, status="completed"):
        if self.current is None:
            return
        self.results[self.current] = {
            "status": status,
            "seconds": time.monotonic() - self.stage_started,
        }
        self.current = None
        self.visible_work = None
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

    def sync_display(self):
        work = self.visible_work
        if work is not self.displayed_work:
            self.bar.reset(total=work.total if work is not None else len(self.stages))
            self.displayed_work = work
        if work is not None:
            self.bar.n, self.bar.unit = work.bar.n, work.unit
            self.bar.set_description(f"{self.name}: {work.description}", refresh=False)
            return
        self.bar.n = sum(
            result["status"] in {"completed", "skipped"}
            for result in self.results.values()
        )
        self.bar.unit = "stage"
        self.bar.set_description(
            f"{self.name}: {self.current or 'stages'}", refresh=False
        )

    def refresh(self):
        self.sync_display()
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
                "detail_log": str(self.output.path),
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
        self.output.close()


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
    if _active_monitor is None:
        print(text)
    else:
        _active_monitor.output.write(str(text) + "\n")


def quiet_output():
    """Capture verbose collector messages only; authentication remains outside this scope."""
    monitor = active_monitor()
    return redirect_stdout(monitor.output) if monitor is not None else nullcontext()


def track(items, description, total=None, unit="item", size_of=None):
    if active_monitor() is None:
        yield from items
        return
    total = len(items) if total is None and hasattr(items, "__len__") else total
    with WorkCounter(total, description, unit) as counter:
        for item in items:
            yield item
            counter.update(size_of(item) if size_of else 1)


def active_monitor():
    return _active_monitor


class WorkCounter:
    """Count finished work, including opaque operations such as a whole forest fit."""

    def __init__(self, total, description, unit):
        self.monitor = active_monitor()
        self.started = time.monotonic()
        self.total = total
        self.description, self.unit = description, unit
        self.bar = SilentCount()
        self.primary = not self.monitor.item_bars

    def __enter__(self):
        self.monitor.item_bars.append(self.bar)
        if self.primary:
            self.monitor.visible_work = self
            self.monitor.last_work_remaining = None
        self.monitor.refresh()
        return self

    def update(self, amount=1):
        self.bar.update(amount)
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


class SilentCount:
    """Nested work remains observable without creating another output widget."""

    def __init__(self):
        self.n = 0
        self.closed = False

    def update(self, amount):
        self.n += amount

    def refresh(self):
        pass

    def close(self):
        self.closed = True


class ProgressOutput(io.TextIOBase):
    """Keep per-file diagnostics in a local log rather than the notebook display."""

    def __init__(self, monitor):
        super().__init__()
        self.path = monitor.folder / monitor.name / "runs" / f"{monitor.run_id}.log"
        self.stream = None

    def write(self, text):
        if self.stream is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.path.open("a", encoding="utf-8", buffering=1)
        self.stream.write(text)
        return len(text)

    def flush(self):
        if self.stream is not None and not self.stream.closed:
            self.stream.flush()

    def close(self):
        self.flush()
        if self.stream is not None:
            self.stream.close()
        super().close()


class ProgressOnlyStatements(ast.NodeTransformer):
    """Keep existing processing hashes stable when only notebook UI statements change."""

    def visit_ImportFrom(self, node):
        if node.module == "crop_pipeline":
            node.names = [name for name in node.names if name.name != "progress"]
            return node if node.names else None
        return node

    def visit_Assign(self, node):
        if any(
            isinstance(target, ast.Name) and target.id in {"notebook_progress", "download_workers", "intervals_per_request"}
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
        # Transport knobs must not create a new label-review or imagery archive.
        if (isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and (function.value.id, function.attr) in {("annual", "collect_source"), ("adaptive", "collect_batch"), ("adaptive", "collect_batches")}):
            node.keywords = [keyword for keyword in node.keywords
                             if keyword.arg not in {"download_workers", "intervals_per_request"}]
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
