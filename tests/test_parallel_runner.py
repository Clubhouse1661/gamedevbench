"""Tests for the parallel (multi-worker) benchmark runner path.

Offline: no Godot, no API, no real subprocesses. The process pool is replaced
with an inline executor and `run_benchmark` is stubbed, so we exercise the
dispatch, aggregation, and checkpointing logic without spawning anything.
"""
import concurrent.futures

import pytest
import yaml

import gamedevbench.src.benchmark_runner as br
from gamedevbench.src.benchmark_runner import GodotBenchmarkRunner


class _InlineExecutor:
    """Drop-in for ProcessPoolExecutor that runs submitted work synchronously."""

    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        fut = concurrent.futures.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as e:  # pragma: no cover - mirrors pool behaviour
            fut.set_exception(e)
        return fut


def _make_runner(tmp_path, **kwargs):
    runner = GodotBenchmarkRunner(use_gt=False, agent=None, **kwargs)
    # Redirect all writes into the temp dir.
    runner.results_dir = tmp_path
    runner.progress_file = tmp_path / "progress.json"
    return runner


def _write_task_list(tmp_path, task_names):
    path = tmp_path / "tasks.yaml"
    path.write_text(yaml.safe_dump({"tasks": list(task_names)}))
    return str(path)


def test_workers_floored_to_one():
    runner = GodotBenchmarkRunner(use_gt=False, agent=None, workers=0)
    assert runner.workers == 1


def test_workers_clamped_to_one_with_monitor_grabbing_mcp(capsys):
    # The default screenshot server captures a whole monitor -> serial only.
    runner = GodotBenchmarkRunner(use_gt=False, agent=None, use_mcp=True, workers=8)
    assert runner.workers == 1
    assert "forcing workers=1" in capsys.readouterr().out


def test_workers_not_clamped_with_headless_mcp(capsys):
    # godot-mcp runs headless (per-task processes, no shared monitor), so it must
    # NOT force serial execution. agent=None skips the OpenHands-only guard.
    runner = GodotBenchmarkRunner(
        use_gt=False, agent=None, use_mcp=True, mcp_server="godot", workers=8
    )
    assert runner.workers == 8
    assert "forcing workers=1" not in capsys.readouterr().out


def test_workers_not_clamped_with_godot_ai(capsys):
    # godot-ai gives each task editor its own free ports + isolated editor
    # state, so it runs in parallel like the headless stdio servers.
    runner = GodotBenchmarkRunner(
        use_gt=False, agent=None, use_mcp=True, mcp_server="godot-ai", workers=8
    )
    assert runner.workers == 8
    assert "forcing workers=1" not in capsys.readouterr().out


def test_codex_allows_godot_ai_mcp_server_in_runner_validation():
    runner = GodotBenchmarkRunner(
        use_gt=False,
        agent="codex",
        use_mcp=True,
        mcp_server="godot-ai",
        workers=2,
    )
    assert runner.mcp_server == "godot-ai"


def test_codex_rejects_unwired_mcp_server_in_runner_validation():
    with pytest.raises(ValueError, match="not supported with --agent codex"):
        GodotBenchmarkRunner(
            use_gt=False,
            agent="codex",
            use_mcp=True,
            mcp_server="godot",
        )


def test_workers_passthrough():
    runner = GodotBenchmarkRunner(use_gt=False, agent=None, workers=4)
    assert runner.workers == 4


def test_parallel_runs_all_tasks_and_aggregates(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "ProcessPoolExecutor", _InlineExecutor)
    runner = _make_runner(tmp_path, workers=4)

    outcomes = {"task_a": True, "task_b": False, "task_c": True}

    def fake_run_benchmark(task_name):
        return {"task_name": task_name, "success": outcomes[task_name], "message": "x"}

    monkeypatch.setattr(runner, "run_benchmark", fake_run_benchmark)

    task_list = _write_task_list(tmp_path, outcomes.keys())
    summary = runner.run_all_tasks(task_list_file=task_list)

    assert summary["success"] == 2
    assert summary["failures"] == 1
    assert summary["total_tasks_ran"] == 3
    # Every task was run regardless of completion order.
    assert {t["task_name"] for t in summary["tasks"]} == set(outcomes)


def test_parallel_counts_errors_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "ProcessPoolExecutor", _InlineExecutor)
    runner = _make_runner(tmp_path, workers=4)

    def fake_run_benchmark(task_name):
        if task_name == "task_boom":
            raise RuntimeError("kaboom")
        return {"task_name": task_name, "success": True, "message": "ok"}

    monkeypatch.setattr(runner, "run_benchmark", fake_run_benchmark)

    task_list = _write_task_list(tmp_path, ["task_ok", "task_boom"])
    summary = runner.run_all_tasks(task_list_file=task_list)

    assert summary["success"] == 1
    assert summary["errors"] == 1


def test_single_worker_uses_sequential_path(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, workers=1)

    def boom(*a, **k):
        raise AssertionError("parallel path must not run when workers == 1")

    monkeypatch.setattr(runner, "_run_tasks_parallel", boom)
    monkeypatch.setattr(
        runner, "run_benchmark",
        lambda t: {"task_name": t, "success": True, "message": "ok"},
    )

    task_list = _write_task_list(tmp_path, ["task_a", "task_b"])
    summary = runner.run_all_tasks(task_list_file=task_list)
    assert summary["success"] == 2
