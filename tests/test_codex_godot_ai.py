"""Codex solver wiring for MCP servers, including the live-editor godot-ai path."""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gamedevbench.src.codex_solver as codex
import gamedevbench.src.godot_ai_editor as gae
from gamedevbench.src.codex_solver import CodexSolver


class _FakeEditorSession:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.entered = False
        self.exited = False
        self.http_url = ""
        self.index = len(type(self).instances)
        type(self).instances.append(self)

    def __enter__(self):
        self.entered = True
        self.http_url = f"http://127.0.0.1:{51234 + self.index}/mcp"
        return self

    def __exit__(self, *exc):
        self.exited = True


def _patch_editor(monkeypatch):
    _FakeEditorSession.instances = []
    monkeypatch.setattr(gae, "ensure_addon", lambda *a, **k: "/cache/godot_ai")
    monkeypatch.setattr(gae, "GodotAiEditorSession", _FakeEditorSession)


def _make_solver(**kwargs):
    solver = CodexSolver(timeout_seconds=30, model="gpt-5-codex", **kwargs)
    solver.load_config = lambda: {"task_id": "t", "instruction": "do", "name": "n"}
    solver.get_task_prompt = lambda config: "PROMPT"
    return solver


def _completed_codex():
    return subprocess.CompletedProcess(
        ["codex"],
        0,
        stdout='{"type":"turn.completed","finalResponse":"done"}\n',
        stderr="",
    )


def test_codex_command_prefers_windows_cmd_wrapper(monkeypatch):
    seen = []

    def fake_which(candidate):
        seen.append(candidate)
        return f"C:/bin/{candidate}" if candidate == "codex.cmd" else None

    monkeypatch.setattr(codex.os, "name", "nt")
    monkeypatch.setattr(codex.shutil, "which", fake_which)

    assert CodexSolver._codex_command() == "C:/bin/codex.cmd"
    assert seen == ["codex.cmd"]


def test_codex_parses_current_jsonl_agent_message():
    solver = _make_solver()
    output = (
        '{"type":"item.completed","item":{"id":"item_0",'
        '"type":"agent_message","text":"done from current codex"}}\n'
        '{"type":"turn.completed","usage":{"input_tokens":1}}\n'
    )

    assert solver._parse_final_response(output) == "done from current codex"


def test_godot_ai_writes_task_local_http_config_and_runs_editor(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_API_KEY", "test-key")
    _patch_editor(monkeypatch)
    captured = {}

    def fake_run(cmd, **kwargs):
        codex_home = Path(kwargs["env"]["CODEX_HOME"])
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        captured["codex_home"] = codex_home
        captured["config"] = (codex_home / "config.toml").read_text()
        return _completed_codex()

    monkeypatch.setattr(codex.subprocess, "run", fake_run)
    solver = _make_solver(use_mcp=True, mcp_server="godot-ai")

    result = solver.solve_task()

    assert result.success is True
    assert len(_FakeEditorSession.instances) == 1
    session = _FakeEditorSession.instances[0]
    assert session.entered and session.exited
    assert "CODEX_HOME" in captured["env"]
    assert not captured["codex_home"].exists()
    assert "[mcp_servers.godot-screenshot]" in captured["config"]
    assert 'command = "uv"' in captured["config"]
    assert "[mcp_servers.godot-ai]" in captured["config"]
    assert f'url = "{session.http_url}"' in captured["config"]
    assert "enabled = true" in captured["config"]


def test_godot_ai_editor_and_codex_home_are_cleaned_up_on_timeout(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_API_KEY", "test-key")
    _patch_editor(monkeypatch)
    captured = {}

    def fake_run(cmd, **kwargs):
        codex_home = Path(kwargs["env"]["CODEX_HOME"])
        captured["codex_home"] = codex_home
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=30)

    monkeypatch.setattr(codex.subprocess, "run", fake_run)
    solver = _make_solver(use_mcp=True, mcp_server="godot-ai")

    result = solver.solve_task()

    assert result.success is False
    assert "timed out" in result.message
    assert _FakeEditorSession.instances[0].exited is True
    assert not captured["codex_home"].exists()


def test_screenshot_mcp_uses_isolated_codex_home_without_godot_ai(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_API_KEY", "test-key")
    captured = {}

    def fake_run(cmd, **kwargs):
        codex_home = Path(kwargs["env"]["CODEX_HOME"])
        captured["codex_home"] = codex_home
        captured["config"] = (codex_home / "config.toml").read_text()
        return _completed_codex()

    monkeypatch.setattr(codex.subprocess, "run", fake_run)
    solver = _make_solver(use_mcp=True)

    result = solver.solve_task()

    assert result.success is True
    assert "[mcp_servers.godot-screenshot]" in captured["config"]
    assert "[mcp_servers.godot-ai]" not in captured["config"]
    assert not captured["codex_home"].exists()


def test_parallel_godot_ai_runs_use_isolated_codex_homes_and_urls(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_API_KEY", "test-key")
    _patch_editor(monkeypatch)
    captures = []

    def fake_run(cmd, **kwargs):
        codex_home = Path(kwargs["env"]["CODEX_HOME"])
        captures.append(
            {
                "codex_home": codex_home,
                "config": (codex_home / "config.toml").read_text(),
            }
        )
        return _completed_codex()

    monkeypatch.setattr(codex.subprocess, "run", fake_run)

    def run_one():
        return _make_solver(use_mcp=True, mcp_server="godot-ai").solve_task()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run_one(), range(2)))

    assert all(result.success for result in results)
    assert len(captures) == 2
    homes = {capture["codex_home"] for capture in captures}
    assert len(homes) == 2
    assert all(not home.exists() for home in homes)
    assert len(_FakeEditorSession.instances) == 2
    urls = {session.http_url for session in _FakeEditorSession.instances}
    assert len(urls) == 2
    for url in urls:
        assert any(f'url = "{url}"' in capture["config"] for capture in captures)
