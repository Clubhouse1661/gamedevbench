"""Tests for the per-run MCP server registry.

Locks in the contract that benchmark_runner/solvers rely on: the baseline
screenshot server stays the default, the Godot-targeted server is selectable,
and each spec carries the launch command + prompt guidance.
"""
import pytest

from gamedevbench.src import mcp_registry as mcp_servers


def test_baseline_is_default():
    assert mcp_servers.DEFAULT_MCP_SERVER == "screenshot"
    spec = mcp_servers.get_mcp_server(None)
    assert spec.name == "screenshot"
    assert spec.server_id == "godot-screenshot"
    assert spec.command == "uv"
    assert spec.args == ("run", "gamedevbench-mcp")


def test_godot_server_registered():
    names = mcp_servers.available_mcp_servers()
    assert "godot" in names
    assert "godot-tugcan" in names
    assert "screenshot" in names
    assert names == sorted(names)


def test_godot_server_launches_via_npx():
    spec = mcp_servers.get_mcp_server("godot")
    assert spec.command == "npx"
    # -y so npx never blocks on an interactive install prompt mid-run.
    assert "-y" in spec.args
    assert "@coding-solo/godot-mcp" in spec.args


def test_godot_tugcan_server_launches_via_npx():
    spec = mcp_servers.get_mcp_server("godot-tugcan")
    assert spec.command == "npx"
    assert "-y" in spec.args
    assert "@tugcantopaloglu/godot-mcp" in spec.args
    # Headless tools, so parallel runs are safe; prime the npx download once.
    assert spec.exclusive_display is False
    assert spec.prefetch is True


def test_godot_tugcan_env_uses_godot_exec_path(monkeypatch):
    monkeypatch.setenv("GODOT_EXEC_PATH", "/opt/godot/godot")
    env = mcp_servers.get_mcp_server("godot-tugcan").env()
    assert env["GODOT_PATH"] == "/opt/godot/godot"


def test_godot_ai_server_registered():
    assert "godot-ai" in mcp_servers.available_mcp_servers()


def test_godot_ai_uses_http_transport_via_editor():
    spec = mcp_servers.get_mcp_server("godot-ai")
    # The agent reaches the plugin-spawned server over HTTP, not stdio.
    assert spec.transport == "http"
    assert spec.http_url == mcp_servers.GODOT_AI_HTTP_URL
    assert spec.http_url.startswith("http://127.0.0.1:")
    # A live Godot editor with the plugin must run per task.
    assert spec.needs_godot_editor is True
    # command/args exist only to prime the uvx package cache during warm_up.
    assert spec.command == "uvx"
    assert mcp_servers.godot_ai_package_source() in spec.args


def test_godot_ai_source_override_points_package_and_addon(monkeypatch):
    monkeypatch.setenv(
        mcp_servers.GODOT_AI_SOURCE_ENV,
        "local:/work/godot-ai",
    )

    assert mcp_servers.godot_ai_package_source() == "/work/godot-ai"
    assert mcp_servers.godot_ai_addon_source() == "/work/godot-ai"


def test_godot_ai_version_override_keeps_published_package_pin(monkeypatch):
    monkeypatch.delenv(mcp_servers.GODOT_AI_SOURCE_ENV, raising=False)
    monkeypatch.setattr(mcp_servers, "GODOT_AI_VERSION", "9.8.7")

    assert mcp_servers.godot_ai_package_source() == "godot-ai==9.8.7"
    assert mcp_servers.godot_ai_addon_source() == "9.8.7"


def test_godot_ai_runs_in_parallel():
    # Each task's editor gets its own free ports + isolated state, so godot-ai
    # is NOT single-worker despite needing a per-task editor.
    spec = mcp_servers.get_mcp_server("godot-ai")
    assert spec.needs_godot_editor is True
    assert spec.requires_single_worker is False


def test_requires_single_worker_only_for_monitor_grab():
    # Only the screenshot baseline (whole monitor) forces serial; everything
    # else — stdio servers and the editor-backed godot-ai — stays parallel-safe.
    assert mcp_servers.get_mcp_server("screenshot").requires_single_worker is True
    assert mcp_servers.get_mcp_server("godot").requires_single_worker is False
    assert mcp_servers.get_mcp_server("godot-tugcan").requires_single_worker is False
    assert mcp_servers.get_mcp_server("godot-ai").requires_single_worker is False


def test_godot_ai_env_disables_telemetry(monkeypatch):
    monkeypatch.setenv("GODOT_EXEC_PATH", "/opt/godot/godot")
    env = mcp_servers.get_mcp_server("godot-ai").env()
    # Telemetry phones home by default — both opt-out vars must be set.
    assert env["GODOT_AI_DISABLE_TELEMETRY"] == "1"
    assert env["DISABLE_TELEMETRY"] == "1"
    assert env["GODOT_PATH"] == "/opt/godot/godot"


def test_godot_ai_marked_for_prefetch():
    assert mcp_servers.get_mcp_server("godot-ai").prefetch is True


def test_godot_ai_warm_up_primes_uvx(monkeypatch):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd

    monkeypatch.setattr(mcp_servers.subprocess, "run", fake_run)
    spec = mcp_servers.get_mcp_server("godot-ai")
    assert spec.warm_up() is True
    assert calls["cmd"][0] == "uvx"
    assert mcp_servers.godot_ai_package_source() in calls["cmd"]


def test_stdio_servers_carry_no_http_url():
    for name in ("screenshot", "godot", "godot-tugcan"):
        spec = mcp_servers.get_mcp_server(name)
        assert spec.transport == "stdio"
        assert spec.http_url == ""
        assert spec.needs_godot_editor is False


def test_unknown_server_raises():
    with pytest.raises(ValueError, match="Unknown MCP server"):
        mcp_servers.get_mcp_server("does-not-exist")


def test_each_server_has_distinct_guidance():
    screenshot = mcp_servers.get_mcp_server("screenshot").prompt_guidance
    godot = mcp_servers.get_mcp_server("godot").prompt_guidance
    tugcan = mcp_servers.get_mcp_server("godot-tugcan").prompt_guidance
    assert screenshot != godot
    assert godot != tugcan
    assert "godot-screenshot" in screenshot
    # godot-mcp exposes run/debug tools the screenshot baseline doesn't.
    assert "run_project" in godot
    # The tugcan server names its own package in its guidance.
    assert "@tugcantopaloglu/godot-mcp" in tugcan


def test_godot_env_uses_godot_exec_path(monkeypatch):
    monkeypatch.setenv("GODOT_EXEC_PATH", "/opt/godot/godot")
    env = mcp_servers.get_mcp_server("godot").env()
    assert env["GODOT_PATH"] == "/opt/godot/godot"


def test_godot_env_omits_path_when_unresolvable(monkeypatch):
    monkeypatch.delenv("GODOT_EXEC_PATH", raising=False)
    monkeypatch.delenv("GODOT_PATH", raising=False)
    monkeypatch.setattr(mcp_servers.shutil, "which", lambda _: None)
    # No path found -> no GODOT_PATH key (let godot-mcp self-detect).
    assert "GODOT_PATH" not in mcp_servers.get_mcp_server("godot").env()


def test_screenshot_server_has_no_extra_env():
    assert mcp_servers.get_mcp_server("screenshot").env() == {}


def test_godot_server_marked_for_prefetch():
    assert mcp_servers.get_mcp_server("godot").prefetch is True
    # The screenshot baseline ships locally; nothing to fetch.
    assert mcp_servers.get_mcp_server("screenshot").prefetch is False


def test_warm_up_launches_command_for_prefetch_server(monkeypatch):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["stdin"] = kwargs.get("stdin")
        calls["timeout"] = kwargs.get("timeout")

    monkeypatch.setenv("GODOT_EXEC_PATH", "/opt/godot/godot")
    monkeypatch.setattr(mcp_servers.subprocess, "run", fake_run)

    spec = mcp_servers.get_mcp_server("godot")
    assert spec.warm_up() is True
    assert calls["cmd"] == ["npx", "-y", "@coding-solo/godot-mcp"]
    # stdin must be closed so the stdio server exits on EOF instead of hanging.
    assert calls["stdin"] == mcp_servers.subprocess.DEVNULL


def test_warm_up_is_noop_for_non_prefetch_server(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("warm_up should not launch a non-prefetch server")

    monkeypatch.setattr(mcp_servers.subprocess, "run", boom)
    assert mcp_servers.get_mcp_server("screenshot").warm_up() is False


def test_warm_up_survives_timeout(monkeypatch):
    def timeout_run(*a, **k):
        raise mcp_servers.subprocess.TimeoutExpired(cmd="npx", timeout=1)

    monkeypatch.setattr(mcp_servers.subprocess, "run", timeout_run)
    # A timeout still means the cache was (almost certainly) warmed.
    assert mcp_servers.get_mcp_server("godot").warm_up() is True


def test_warm_up_swallows_launch_failure(monkeypatch):
    def fail_run(*a, **k):
        raise FileNotFoundError("npx not installed")

    monkeypatch.setattr(mcp_servers.subprocess, "run", fail_run)
    assert mcp_servers.get_mcp_server("godot").warm_up() is False
