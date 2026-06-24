#!/usr/bin/env python3
"""Per-task Godot editor lifecycle for the ``godot-ai`` MCP server.

Unlike the stdio MCP servers, ``hi-godot/godot-ai`` is a Godot **editor plugin**
that spawns a Python FastMCP server the agent reaches over HTTP. For any tool to
work, a Godot editor with the plugin enabled must be running and connected to
that server. This module owns that lifecycle for one task:

1. Cache the pinned addon (``ensure_addon``) — cloned once at warm-up, reused.
   The cached addon is patched so its server ports come from environment
   variables (``_patch_addon_env_ports``), which is what lets editors run in
   parallel: stock godot-ai reads ports only from global EditorSettings.
2. Install it into the task sandbox and enable the plugin (``install_addon``).
3. Launch the editor under ``xvfb`` (``GodotAiEditorSession``) on a **per-session
   free port pair**, with Godot's config/data dirs isolated per session so
   concurrent editors don't fight over EditorSettings or the server pid-file;
   the plugin spawns the server on those ports.
4. Poll ``editor_state`` until the editor reports ``readiness == "ready"``.
5. On exit, terminate the editor's whole process tree (which also reaps the
   server — it has an owner-pid orphan reaper, but we don't rely on it), strip
   the plugin footprint, and remove the per-session dirs.

The heavy steps are kept behind small, pure helpers so they can be unit-tested
without Godot, a display, or the network.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from gamedevbench.src.mcp_registry import GODOT_AI_VERSION

#: Upstream repo cloned (shallow, pinned tag) to obtain the editor addon.
GODOT_AI_REPO_URL = "https://github.com/hi-godot/godot-ai.git"

#: Path inside the repo to the addon directory copied into each project.
_ADDON_SUBPATH = "plugin/addons/godot_ai"

#: Plugin entry written into ``project.godot`` to enable the addon.
_PLUGIN_RES_PATH = "res://addons/godot_ai/plugin.cfg"

#: Env vars the patched addon reads its server ports from (see
#: ``_patch_addon_env_ports``). Names are also set on the editor launch env.
ENV_HTTP_PORT = "GODOT_AI_HTTP_PORT"
ENV_WS_PORT = "GODOT_AI_WS_PORT"

# The stock addon resolves ports only from global EditorSettings, so every
# editor would spawn its server on 8000/9500 and collide. We patch the two port
# accessors to prefer an env override, giving each editor its own ports. The
# searched text is the exact (tab-indented) body upstream ships at v2.7.5.
_PORT_PATCHES = (
    (
        "static func http_port() -> int:\n"
        "\treturn _read_port_setting(McpSettings.SETTING_HTTP_PORT, DEFAULT_HTTP_PORT)",
        "static func http_port() -> int:\n"
        f'\tvar _e := OS.get_environment("{ENV_HTTP_PORT}")\n'
        "\tif _e.is_valid_int():\n"
        "\t\treturn _e.to_int()\n"
        "\treturn _read_port_setting(McpSettings.SETTING_HTTP_PORT, DEFAULT_HTTP_PORT)",
    ),
    (
        "static func ws_port() -> int:\n"
        "\treturn _read_port_setting(SETTING_WS_PORT, DEFAULT_WS_PORT)",
        "static func ws_port() -> int:\n"
        f'\tvar _e := OS.get_environment("{ENV_WS_PORT}")\n'
        "\tif _e.is_valid_int():\n"
        "\t\treturn _e.to_int()\n"
        "\treturn _read_port_setting(SETTING_WS_PORT, DEFAULT_WS_PORT)",
    ),
)


def _patch_addon_env_ports(addon_dir: Path) -> None:
    """Patch the addon so its server ports come from env vars (idempotent).

    Raises ``RuntimeError`` if the upstream port accessors aren't found verbatim
    — that means the addon version drifted and the patch (and thus parallel
    safety) can't be trusted, so we fail loudly rather than silently fall back
    to colliding default ports.
    """
    cc = addon_dir / "client_configurator.gd"
    text = cc.read_text()
    if ENV_HTTP_PORT in text and ENV_WS_PORT in text:
        return  # already patched
    for old, new in _PORT_PATCHES:
        if old not in text:
            raise RuntimeError(
                "godot-ai addon port accessor not found verbatim in "
                f"{cc} — the pinned version may have drifted; refusing to run "
                "parallel without env-port support."
            )
        text = text.replace(old, new)
    cc.write_text(text)


def free_port() -> int:
    """Return a currently-free TCP port on loopback (bind-to-0 then release).

    There is a small TOCTOU window before the editor binds it; the session's
    readiness check + retry covers the rare collision between parallel workers.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def addon_tag(version: str = GODOT_AI_VERSION) -> str:
    """Git tag for a godot-ai version (upstream tags are ``vX.Y.Z``)."""
    return f"v{version}"


def addon_cache_dir(version: str = GODOT_AI_VERSION) -> Path:
    """Host-shared cache directory for the checked-out addon.

    Overridable with ``GAMEDEVBENCH_GODOT_AI_CACHE``; defaults under the system
    temp dir so all workers on a host reuse one checkout.
    """
    base = os.environ.get("GAMEDEVBENCH_GODOT_AI_CACHE")
    root = Path(base) if base else Path(__import__("tempfile").gettempdir())
    return root / f"gamedevbench_godot_ai_{version}"


def ensure_addon(
    version: str = GODOT_AI_VERSION,
    *,
    repo_url: str = GODOT_AI_REPO_URL,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Path:
    """Return the path to the cached ``godot_ai`` addon dir, cloning if absent.

    Idempotent: if the addon is already present in the cache, returns
    immediately without touching the network. Raises ``RuntimeError`` if the
    clone fails or the expected addon path is missing afterwards.
    """
    cache = addon_cache_dir(version)
    addon = cache / "addon" / "godot_ai"
    if (addon / "plugin.cfg").exists():
        _patch_addon_env_ports(addon)  # idempotent; covers a stale unpatched cache
        return addon

    checkout = cache / "repo"
    if not (checkout / _ADDON_SUBPATH / "plugin.cfg").exists():
        shutil.rmtree(checkout, ignore_errors=True)
        checkout.parent.mkdir(parents=True, exist_ok=True)
        result = runner(
            [
                "git", "clone", "--depth", "1",
                "--branch", addon_tag(version),
                repo_url, str(checkout),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to clone godot-ai {addon_tag(version)}: "
                f"{(result.stderr or '').strip()[:400]}"
            )

    src = checkout / _ADDON_SUBPATH
    if not (src / "plugin.cfg").exists():
        raise RuntimeError(
            f"godot-ai addon not found at {src} after clone"
        )
    addon.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(addon, ignore_errors=True)
    shutil.copytree(src, addon)
    _patch_addon_env_ports(addon)
    return addon


def enable_plugin_text(project_godot_text: str) -> str:
    """Return ``project.godot`` text with the godot-ai plugin enabled.

    Merges into an existing ``[editor_plugins] enabled=PackedStringArray(...)``
    if present (preserving other plugins), otherwise appends the section. Pure
    string transform — no filesystem — so it is easy to unit-test.
    """
    entry = f'"{_PLUGIN_RES_PATH}"'
    if entry in project_godot_text:
        return project_godot_text

    trailing_nl = project_godot_text.endswith("\n")
    lines = project_godot_text.splitlines()

    def _is_header(line: str) -> bool:
        s = line.strip()
        return s.startswith("[") and s.endswith("]")

    # Locate the [editor_plugins] section, if any, and its bounds.
    header_idx = next(
        (i for i, l in enumerate(lines) if l.strip() == "[editor_plugins]"), None
    )

    if header_idx is None:
        # No section: append a fresh one.
        text = "\n".join(lines)
        if text and not text.endswith("\n"):
            text += "\n"
        text += f"\n[editor_plugins]\n\nenabled=PackedStringArray({entry})\n"
        return text

    section_end = next(
        (i for i in range(header_idx + 1, len(lines)) if _is_header(lines[i])),
        len(lines),
    )
    for i in range(header_idx + 1, section_end):
        if lines[i].strip().startswith("enabled=PackedStringArray("):
            inner = lines[i].strip()[len("enabled=PackedStringArray("):].rstrip(")")
            members = [m for m in inner.split(",") if m.strip()]
            members.append(entry)
            lines[i] = f"enabled=PackedStringArray({', '.join(members)})"
            break
    else:
        # Section exists but has no enabled key — insert one after the header.
        lines.insert(header_idx + 1, f"enabled=PackedStringArray({entry})")

    text = "\n".join(lines)
    return text + "\n" if trailing_nl else text


def strip_plugin_text(project_godot_text: str) -> str:
    """Return ``project.godot`` with all godot-ai tooling footprint removed.

    Reverses :func:`enable_plugin_text` (drops our entry from the
    ``[editor_plugins]`` enabled array, removing an emptied section we created)
    and removes any ``[autoload]`` entry pointing into ``res://addons/godot_ai/``
    — notably the ``_mcp_game_helper`` singleton the plugin injects, which would
    otherwise run during the scored headless validation. Other plugins and
    autoloads (including ones the agent legitimately added) are preserved.
    """
    entry = f'"{_PLUGIN_RES_PATH}"'
    trailing_nl = project_godot_text.endswith("\n")
    lines = project_godot_text.splitlines()

    def _is_header(line: str) -> bool:
        s = line.strip()
        return s.startswith("[") and s.endswith("]")

    out: list[str] = []
    section: Optional[str] = None
    for line in lines:
        s = line.strip()
        if _is_header(line):
            section = s
        if section == "[autoload]" and "res://addons/godot_ai/" in s:
            continue
        if section == "[editor_plugins]" and s.startswith("enabled=PackedStringArray("):
            inner = s[len("enabled=PackedStringArray("):].rstrip(")")
            members = [
                m.strip() for m in inner.split(",")
                if m.strip() and m.strip() != entry
            ]
            if members:
                out.append(f"enabled=PackedStringArray({', '.join(members)})")
            continue  # drop the line entirely when no members remain
        out.append(line)

    # Drop an [editor_plugins] section left with no key lines (one we created).
    cleaned: list[str] = []
    i = 0
    while i < len(out):
        if out[i].strip() == "[editor_plugins]":
            j = i + 1
            has_keys = False
            while j < len(out) and not _is_header(out[j]):
                if out[j].strip():
                    has_keys = True
                j += 1
            if not has_keys:
                i = j  # skip header + its blank lines
                continue
        cleaned.append(out[i])
        i += 1

    text = "\n".join(cleaned).rstrip("\n")
    return text + "\n" if trailing_nl else text


def cleanup_project_footprint(project_dir: Path) -> None:
    """Remove the godot-ai addon and its project.godot entries from a project.

    Idempotent and best-effort: leaves the project as if the plugin had never
    been installed, so the agent's work is validated in a clean environment.
    """
    shutil.rmtree(project_dir / "addons" / "godot_ai", ignore_errors=True)
    project_godot = project_dir / "project.godot"
    if project_godot.exists():
        project_godot.write_text(strip_plugin_text(project_godot.read_text()))


def install_addon(project_dir: Path, addon_src: Path) -> None:
    """Copy the addon into a project and enable its plugin in project.godot."""
    dest = project_dir / "addons" / "godot_ai"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(addon_src, dest)

    project_godot = project_dir / "project.godot"
    text = project_godot.read_text() if project_godot.exists() else ""
    project_godot.write_text(enable_plugin_text(text))


def parse_host_port(url: str) -> Tuple[str, int]:
    """Extract (host, port) from an MCP HTTP URL like ``http://127.0.0.1:8000/mcp``."""
    parsed = urlparse(url)
    return parsed.hostname or "127.0.0.1", parsed.port or 8000


def probe_editor_state(http_url: str, *, timeout: float = 5.0) -> Optional[dict]:
    """Call the ``editor_state`` MCP tool once; return its dict or None.

    Returns None on any connection/transport error (server not up yet, editor
    not connected). Imports fastmcp lazily so this module is importable without
    it (the godot-ai path only runs under the OpenHands solver, which pulls it).
    """
    import asyncio

    async def _call() -> Optional[dict]:
        from fastmcp import Client  # lazy: only needed on the godot-ai path

        async with Client(http_url) as client:
            result = await client.call_tool("editor_state", {})
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                import json

                try:
                    return json.loads(text)
                except ValueError:
                    return {"raw": text}
        return None

    try:
        return asyncio.run(asyncio.wait_for(_call(), timeout=timeout))
    except Exception:
        return None


def wait_until_ready(
    http_url: str,
    *,
    timeout: float = 90.0,
    poll_interval: float = 2.0,
    probe: Callable[[str], Optional[dict]] = probe_editor_state,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> bool:
    """Poll ``editor_state`` until the editor reports ``readiness == "ready"``.

    Returns True once ready, False if the deadline passes first. ``probe``,
    ``sleep``, and ``now`` are injectable for testing.
    """
    deadline = now() + timeout
    while now() < deadline:
        state = probe(http_url)
        if state and str(state.get("readiness", "")).lower() == "ready":
            return True
        sleep(poll_interval)
    return False


def free_display(start: int = 200, end: int = 899) -> int:
    """Pick an X display number not currently in use under ``/tmp/.X11-unix``.

    Used to pin the editor's xvfb display so we can also export ``DISPLAY`` into
    the env (see ``build_editor_command``); a small race is covered by xvfb-run's
    ``-a`` fallback + the session's readiness retry.
    """
    import random

    used: set[int] = set()
    try:
        for name in os.listdir("/tmp/.X11-unix"):
            if name.startswith("X") and name[1:].isdigit():
                used.add(int(name[1:]))
    except Exception:
        pass
    for _ in range(50):
        n = random.randint(start, end)
        if n not in used:
            return n
    return random.randint(start, end)


def build_editor_command(
    godot_path: str,
    project_dir: Path,
    *,
    have_xvfb: bool,
    allow_headless: bool,
    display: Optional[int] = None,
) -> Tuple[list[str], Dict[str, str]]:
    """Build the editor launch argv and any extra env it needs.

    Prefers ``xvfb-run`` (a throwaway virtual display) since the plugin disables
    itself under a true ``--headless`` editor unless ``GODOT_AI_ALLOW_HEADLESS``
    is set. Falls back to ``--headless`` + that opt-in when xvfb is unavailable.

    When ``display`` is given, pin xvfb to that number (``-n``) AND return
    ``DISPLAY=:N`` so it lands in the process env — that way child processes the
    plugin spawns for its editor reload (which bypass ``xvfb-run``) inherit the
    xvfb display instead of falling back to ``:0`` (WSLg) and popping windows
    onto the host desktop.
    """
    base = [godot_path, "--editor", "--path", str(project_dir)]
    if have_xvfb:
        if display is not None:
            return (
                ["xvfb-run", "-n", str(display), "-a", *base],
                {"DISPLAY": f":{display}"},
            )
        return ["xvfb-run", "-a", *base], {}
    if allow_headless:
        return [*base, "--headless"], {"GODOT_AI_ALLOW_HEADLESS": "1"}
    return base, {}


def _terminate_tree(pid: int, term_grace: float = 5.0) -> None:
    """Terminate ``pid``, its process group, and all descendants (best-effort).

    The editor is launched in its own session (``start_new_session=True``), so a
    single ``killpg`` reaps everything in the group — crucially the ``Xvfb``
    server that ``xvfb-run`` forks but orphans when its wrapper is signal-killed
    (its cleanup trap doesn't run on SIGTERM/SIGKILL). A psutil descendant sweep
    then mops up anything that escaped the group (e.g. re-parented strays).
    """
    import signal

    try:
        pgid = os.getpgid(pid)
    except Exception:
        pgid = None
    if pgid is not None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except Exception:
                break

    try:
        import psutil
    except Exception:
        return
    try:
        parent = psutil.Process(pid)
    except Exception:
        return
    procs = parent.children(recursive=True)
    procs.append(parent)
    for proc in procs:
        try:
            proc.terminate()
        except Exception:
            pass
    _gone, alive = psutil.wait_procs(procs, timeout=term_grace)
    for proc in alive:
        try:
            proc.kill()
        except Exception:
            pass


def _reap_servers(isolated_dir: Path) -> None:
    """Kill any godot-ai server whose pid-file lives under this session's dirs.

    The editor plugin spawns its Python server **detached** (its own session),
    so the editor's ``killpg`` doesn't reach it — it would otherwise linger until
    its owner-pid watchdog notices the editor died. The server writes
    ``user://godot_ai_server.pid`` under our isolated ``XDG_DATA_HOME``, so we
    read those pid-files and terminate the process group explicitly.
    """
    for pidfile in isolated_dir.glob("data/**/godot_ai_server.pid"):
        try:
            pid = int(pidfile.read_text().strip())
        except Exception:
            continue
        _terminate_tree(pid)


@dataclass
class GodotAiEditorSession:
    """Context manager that runs a godot-ai editor for one task.

    Installs the addon into ``project_dir``, launches the editor on a
    **per-session free port pair** with Godot's config/data dirs isolated to a
    throwaway location, and waits for it to report ready. Because each session
    picks its own ports and editor state, multiple sessions can run concurrently
    (one per benchmark worker) without colliding.

    After entering, :attr:`http_url` is the MCP endpoint the agent should use.
    ``__enter__`` retries on a fresh port pair if the editor doesn't come up
    (covers a rare free-port race) and raises ``RuntimeError`` if every attempt
    fails. ``__exit__`` always tears the editor process tree down, strips the
    plugin footprint, and removes the per-session dirs.
    """

    project_dir: Path
    godot_path: str
    addon_src: Path
    extra_env: Dict[str, str] = field(default_factory=dict)
    # Generous so a cold boot under CPU contention (many parallel workers) still
    # reports ready instead of timing out. Too tight (was 90s) caused a retry
    # storm under load: slow boots time out -> retry -> more editors -> more
    # load -> more timeouts, a feedback loop that diverged until the box OOM'd.
    ready_timeout: float = 180.0
    max_attempts: int = 2
    debug: bool = False
    http_url: str = field(default="", init=False)
    _proc: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    _isolated_dirs: List[Path] = field(default_factory=list, init=False, repr=False)

    def __enter__(self) -> "GodotAiEditorSession":
        install_addon(self.project_dir, self.addon_src)
        have_xvfb = shutil.which("xvfb-run") is not None

        last_reason = ""
        for attempt in range(1, self.max_attempts + 1):
            http, ws = free_port(), free_port()
            url = f"http://127.0.0.1:{http}/mcp"
            isolated = Path(tempfile.mkdtemp(prefix="gamedevbench_gai_"))
            (isolated / "config").mkdir()
            (isolated / "data").mkdir()

            cmd, launch_env = build_editor_command(
                self.godot_path, self.project_dir,
                have_xvfb=have_xvfb, allow_headless=True,
                display=free_display() if have_xvfb else None,
            )
            env = {
                **os.environ, **self.extra_env, **launch_env,
                ENV_HTTP_PORT: str(http), ENV_WS_PORT: str(ws),
                # Isolate EditorSettings + the server pid-file per session so
                # concurrent editors don't fight. Cache dir is left shared so the
                # uvx package cache (and Godot's caches) stay warm.
                "XDG_CONFIG_HOME": str(isolated / "config"),
                "XDG_DATA_HOME": str(isolated / "data"),
                # Force software (llvmpipe) GL for the editor AND the games it
                # runs via project_run. On WSL2, hardware GL goes through GPU
                # passthrough (/dev/dxg -> d3d12/dxgkrnl), which is unstable and
                # crashed the whole WSL VM repeatedly under this rendering
                # workload — independent of worker count or memory. Software
                # rendering trades a little speed for not taking down the VM.
                "LIBGL_ALWAYS_SOFTWARE": "1",
                "GALLIUM_DRIVER": "llvmpipe",
            }
            if self.debug:
                print(f"      [godot-ai] attempt {attempt}: {' '.join(cmd)} "
                      f"(http={http} ws={ws})")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL if not self.debug else None,
                stderr=subprocess.DEVNULL if not self.debug else None,
                env=env,
                cwd=str(self.project_dir),
                # Own session/process group so teardown can killpg the whole
                # thing, including the Xvfb that xvfb-run would otherwise orphan.
                start_new_session=True,
            )

            if wait_until_ready(url, timeout=self.ready_timeout):
                self._proc = proc
                self.http_url = url
                self._isolated_dirs.append(isolated)
                if self.debug:
                    print(f"      [godot-ai] editor ready at {url}")
                return self

            # This attempt failed: reap it and its dirs, then try a fresh port.
            _terminate_tree(proc.pid)
            _reap_servers(isolated)
            shutil.rmtree(isolated, ignore_errors=True)
            last_reason = f"not ready on port {http} within {self.ready_timeout}s"

        raise RuntimeError(
            f"godot-ai editor never became ready after {self.max_attempts} "
            f"attempt(s): {last_reason}"
        )

    def __exit__(self, *exc) -> None:
        self._teardown()

    def _teardown(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            _terminate_tree(self._proc.pid)
        self._proc = None
        # Kill the detached server(s) explicitly (the editor's killpg misses them).
        for d in self._isolated_dirs:
            _reap_servers(d)
        # With the editor dead (no longer rewriting project.godot), strip the
        # plugin footprint so the agent's work is validated in a clean project.
        try:
            cleanup_project_footprint(self.project_dir)
        except Exception:
            pass
        for d in self._isolated_dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._isolated_dirs = []
