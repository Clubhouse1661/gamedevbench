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


def test_codex_verification_nudge_reaches_prompt():
    solver = CodexSolver(encourage_verification=True)
    prompt = solver.get_task_prompt({"instruction": "do"})
    assert "godot --headless --script" in prompt


def _completed_codex():
    return subprocess.CompletedProcess(
        ["codex"],
        0,
        stdout='{"type":"turn.completed","finalResponse":"done"}\n',
        stderr="",
    )


def _patch_codex_command(monkeypatch, func):
    monkeypatch.setattr(CodexSolver, "_run_codex_command", func)


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

    def fake_command(self, cmd, *, cwd, env):
        codex_home = Path(env["CODEX_HOME"])
        captured["cmd"] = cmd
        captured["env"] = env
        captured["codex_home"] = codex_home
        captured["config"] = (codex_home / "config.toml").read_text()
        return _completed_codex()

    _patch_codex_command(monkeypatch, fake_command)
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


def test_codex_subprocess_decodes_utf8_with_replacement(monkeypatch):
    captured = {}

    class FakePopen:
        pid = 123
        returncode = 0

        def __init__(self, cmd, **kwargs):
            captured.update(kwargs)

        def communicate(self, timeout=None):
            return ('{"type":"turn.completed","finalResponse":"done"}\n', "")

    monkeypatch.setattr(codex.subprocess, "Popen", FakePopen)
    solver = _make_solver()

    result = solver._run_codex_command(["codex"], cwd=".", env=None)

    assert result.returncode == 0
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_codex_timeout_kills_subprocess_tree(monkeypatch):
    killed = []

    class FakePopen:
        pid = 321
        returncode = None

        def __init__(self, cmd, **kwargs):
            pass

        def communicate(self, timeout=None):
            if timeout == 5:
                return "", ""
            raise subprocess.TimeoutExpired(cmd=["codex"], timeout=timeout)

    monkeypatch.setattr(codex.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(
        codex,
        "terminate_process_tree",
        lambda pid, **kwargs: killed.append((pid, kwargs)),
    )
    solver = _make_solver()

    try:
        solver._run_codex_command(["codex"], cwd=".", env=None)
    except subprocess.TimeoutExpired:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("expected timeout")

    assert killed == [(321, {"kill_process_group": codex.os.name != "nt"})]


def test_codex_solver_uses_command_wrapper(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_command(self, cmd, *, cwd, env):
        captured["cwd"] = cwd
        captured["env"] = env
        return _completed_codex()

    _patch_codex_command(monkeypatch, fake_command)
    solver = _make_solver()

    result = solver.solve_task()

    assert result.success is True
    assert captured["cwd"] == str(tmp_path)


def test_godot_ai_editor_and_codex_home_are_cleaned_up_on_timeout(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_API_KEY", "test-key")
    _patch_editor(monkeypatch)
    captured = {}

    def fake_command(self, cmd, *, cwd, env):
        codex_home = Path(env["CODEX_HOME"])
        captured["codex_home"] = codex_home
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=30)

    _patch_codex_command(monkeypatch, fake_command)
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

    def fake_command(self, cmd, *, cwd, env):
        codex_home = Path(env["CODEX_HOME"])
        captured["codex_home"] = codex_home
        captured["config"] = (codex_home / "config.toml").read_text()
        return _completed_codex()

    _patch_codex_command(monkeypatch, fake_command)
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

    def fake_command(self, cmd, *, cwd, env):
        codex_home = Path(env["CODEX_HOME"])
        captures.append(
            {
                "codex_home": codex_home,
                "config": (codex_home / "config.toml").read_text(),
            }
        )
        return _completed_codex()

    _patch_codex_command(monkeypatch, fake_command)

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
