"""Tests for the parallel (multi-worker) benchmark runner path.

Offline: no Godot, no API, no real subprocesses. The parent-owned worker
launcher is replaced with fake processes/queues, so we exercise dispatch,
aggregation, hard-timeout recovery, and checkpointing without spawning anything.
"""
import pytest
import yaml

import gamedevbench.src.benchmark_runner as br
import gamedevbench.src.godot_ai_editor as gae
from gamedevbench.src.benchmark_runner import GodotBenchmarkRunner


class _FakeProcess:
    _next_pid = 1000

    def __init__(self, *, alive=False, exitcode=0):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self._alive = alive
        self.exitcode = exitcode

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        return None


class _FakeQueue:
    def __init__(self, payload=None):
        self.payload = payload

    def get_nowait(self):
        if self.payload is None:
            raise br.queue.Empty
        payload = self.payload
        self.payload = None
        return payload


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


def _patch_parallel_start(monkeypatch, outcomes):
    reaped = []
    killed = []

    def fake_start(ctx, runner, task_name, run_id):
        outcome = outcomes[task_name]
        if outcome == "hang":
            return _FakeProcess(alive=True), _FakeQueue()
        if isinstance(outcome, Exception):
            return _FakeProcess(), _FakeQueue(
                {"kind": "error", "task_name": task_name, "error": repr(outcome)}
            )
        return _FakeProcess(), _FakeQueue(
            {
                "kind": "result",
                "task_name": task_name,
                "result": {
                    "task_name": task_name,
                    "success": bool(outcome),
                    "message": "x",
                },
            }
        )

    monkeypatch.setattr(br, "_start_task_process", fake_start)
    monkeypatch.setattr(br, "_reap_task_run_artifacts", lambda run_id: reaped.append(run_id))
    monkeypatch.setattr(br, "terminate_process_tree", lambda pid: killed.append(pid))
    return reaped, killed


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


def test_current_task_run_id_prefers_worker_attribute(tmp_path, monkeypatch):
    monkeypatch.setenv("GAMEDEVBENCH_TASK_RUN_ID", "from-env")
    runner = _make_runner(tmp_path)
    runner._task_run_id = "from-worker"

    assert runner._current_task_run_id() == "from-worker"


def test_parallel_runs_all_tasks_and_aggregates(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, workers=4)

    outcomes = {"task_a": True, "task_b": False, "task_c": True}
    _patch_parallel_start(monkeypatch, outcomes)

    task_list = _write_task_list(tmp_path, outcomes.keys())
    summary = runner.run_all_tasks(task_list_file=task_list)

    assert summary["success"] == 2
    assert summary["failures"] == 1
    assert summary["total_tasks_ran"] == 3
    # Every task was run regardless of completion order.
    assert {t["task_name"] for t in summary["tasks"]} == set(outcomes)


def test_parallel_counts_errors_separately(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, workers=4)
    _patch_parallel_start(
        monkeypatch,
        {"task_ok": True, "task_boom": RuntimeError("kaboom")},
    )

    task_list = _write_task_list(tmp_path, ["task_ok", "task_boom"])
    summary = runner.run_all_tasks(task_list_file=task_list)

    assert summary["success"] == 1
    assert summary["errors"] == 1


def test_parallel_hard_timeout_kills_worker_and_continues(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, workers=2)
    runner.worker_hard_timeout_seconds = 0.0
    reaped, killed = _patch_parallel_start(
        monkeypatch,
        {"task_hang": "hang", "task_ok": True, "task_after": True},
    )

    task_list = _write_task_list(tmp_path, ["task_hang", "task_ok", "task_after"])
    summary = runner.run_all_tasks(task_list_file=task_list)

    assert summary["success"] == 2
    assert summary["errors"] == 1
    timeout_result = next(r for r in summary["tasks"] if r["task_name"] == "task_hang")
    assert timeout_result["timed_out"] is True
    assert "hard cap" in timeout_result["message"]
    assert killed
    assert len(reaped) == 3


def test_reap_task_run_artifacts_removes_temp_dirs_and_servers(tmp_path, monkeypatch):
    run_id = "abc123"
    gai_dir = tmp_path / f"gamedevbench_gai_{run_id}_one"
    codex_dir = tmp_path / f"gamedevbench_codex_{run_id}_two"
    unrelated = tmp_path / "gamedevbench_gai_other_three"
    for path in (gai_dir, codex_dir, unrelated):
        path.mkdir()
    reaped = []
    monkeypatch.setattr(br.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(gae, "_reap_servers", lambda path: reaped.append(path))

    br._reap_task_run_artifacts(run_id)

    assert reaped == [gai_dir]
    assert not gai_dir.exists()
    assert not codex_dir.exists()
    assert unrelated.exists()


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
