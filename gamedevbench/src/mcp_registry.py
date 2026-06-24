#!/usr/bin/env python3
"""Registry of MCP servers that can be selected per benchmark run.

Each :class:`MCPServerSpec` is a launcher-agnostic description of one stdio MCP
server: how to start it (``command``/``args``/``env``), the server id used to
namespace its tools, and the prompt guidance describing what it exposes. Solvers
translate a spec into their own config format (e.g. OpenHands' ``mcpServers``
JSON).

The ``screenshot`` entry is the bundled baseline (see ``mcp_server.py``); the
``godot`` entry is the third-party Godot-targeted server
``@coding-solo/godot-mcp`` (https://github.com/Coding-Solo/godot-mcp); the
``godot-tugcan`` entry is a second Godot-targeted server
``@tugcantopaloglu/godot-mcp`` (https://github.com/tugcantopaloglu/godot-mcp);
the ``godot-ai`` entry is ``hi-godot/godot-ai`` (https://github.com/hi-godot/godot-ai),
which is structurally different — an editor plugin that spawns a Python FastMCP
server the agent reaches over **HTTP**, bridged to a live Godot editor (see
``godot_ai_editor.py`` for the per-task editor lifecycle).
Add new Godot-targeted servers here so each is selectable with ``--mcp-server``
and directly comparable against the baseline.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class MCPServerSpec:
    """Launcher-agnostic description of one stdio MCP server.

    Attributes:
        name: Selection key used by ``--mcp-server`` and the registry.
        server_id: Id used to namespace the server's tools (the key under
            ``mcpServers`` and the ``mcp__<server_id>__<tool>`` prefix).
        command: Executable to launch the stdio server.
        args: Arguments passed to ``command``.
        prompt_guidance: Text appended to the task prompt when this server is
            active, describing the tools the agent can call.
        allowed_tools: Optional explicit allow-list of fully-qualified tool
            names (used by solvers that gate tools, e.g. claude-code). Empty
            means "allow every tool this server exposes".
        env_factory: Optional callable returning extra environment variables for
            the server process, resolved at launch time (e.g. ``GODOT_PATH``).
        exclusive_display: True if the server grabs a shared resource (a whole
            monitor) that parallel runs would collide over, forcing the runner
            back to a single worker. False for headless servers (text/stdout
            only), which are safe to run in parallel.
        prefetch: True if the server should be launched once up front before
            tasks run (see ``warm_up``). Set for servers whose first launch
            downloads something (e.g. ``npx`` fetching the package) so parallel
            workers reuse the cache instead of racing to download it.
        transport: How the *agent* reaches the server. ``"stdio"`` (default)
            means the solver launches ``command``/``args`` and speaks MCP over
            the child's stdio. ``"http"`` means the server is reachable at
            ``http_url`` and ``command``/``args`` are used only for ``warm_up``
            (priming a package cache), not to launch the server the agent talks
            to — something else owns that process (see ``needs_godot_editor``).
        http_url: For ``transport="http"``, the MCP endpoint the agent connects
            to (e.g. ``http://127.0.0.1:8000/mcp``). Empty for stdio servers.
        needs_godot_editor: True if a Godot editor running this server's plugin
            must be launched per task before the agent connects (the plugin
            spawns the actual MCP server and bridges it to the live editor). The
            solver drives that lifecycle via ``godot_ai_editor``. Such a server
            binds fixed host ports, so it also forces a single worker.
    """

    name: str
    server_id: str
    command: str
    args: tuple[str, ...]
    prompt_guidance: str
    allowed_tools: tuple[str, ...] = ()
    env_factory: Optional[Callable[[], Dict[str, str]]] = field(default=None)
    exclusive_display: bool = False
    prefetch: bool = False
    transport: str = "stdio"
    http_url: str = ""
    needs_godot_editor: bool = False

    @property
    def requires_single_worker(self) -> bool:
        """True if the server grabs a host-global resource parallel runs collide over.

        Currently only a whole monitor (``exclusive_display``, the screenshot
        baseline). ``needs_godot_editor`` servers are NOT single-worker: each
        per-task editor gets its own free ports and isolated editor state (see
        ``godot_ai_editor``), so they run in parallel.
        """
        return self.exclusive_display

    def env(self) -> Dict[str, str]:
        """Resolve extra environment variables for the server process."""
        return dict(self.env_factory()) if self.env_factory else {}

    def warm_up(self, timeout: float = 300.0) -> bool:
        """Launch the server once with stdin closed to prime any cache.

        The stdio server reads stdin and exits on EOF, so this returns once the
        package is fetched and the server has started+stopped (a couple of
        seconds on a warm cache, longer on a cold one while it downloads). This
        keeps parallel workers from each downloading the package, and keeps that
        download time off any single task's solve timeout. No-op (returns False)
        for servers that don't set ``prefetch``.
        """
        if not self.prefetch:
            return False
        cmd = [self.command, *self.args]
        env = {**os.environ, **self.env()}
        try:
            subprocess.run(
                cmd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # The download almost certainly finished even if the server didn't
            # exit on EOF in time; the cache is warm either way.
            return True
        except Exception:
            return False
        return True


def _resolve_godot_path() -> str:
    """Best-effort path to the Godot executable for godot-mcp's GODOT_PATH.

    Prefers the harness's ``GODOT_EXEC_PATH`` (the same var the runner uses),
    then ``GODOT_PATH``, then whatever ``godot`` resolves to on PATH. Returns an
    empty string if none is found, in which case godot-mcp falls back to its own
    detection.
    """
    return (
        os.environ.get("GODOT_EXEC_PATH")
        or os.environ.get("GODOT_PATH")
        or shutil.which("godot")
        or ""
    )


def _godot_mcp_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    godot_path = _resolve_godot_path()
    if godot_path:
        env["GODOT_PATH"] = godot_path
    return env


#: Pinned hi-godot/godot-ai version — used for both the ``uvx`` server package
#: and the editor addon checkout so the plugin and server stay in lockstep.
GODOT_AI_VERSION = "2.7.5"

#: Default loopback MCP endpoint the godot-ai editor plugin serves on. The
#: server runs single-worker (host-global ports), so the default port is safe.
GODOT_AI_HTTP_URL = "http://127.0.0.1:8000/mcp"


def _godot_ai_env() -> Dict[str, str]:
    """Environment for the godot-ai editor (which spawns the MCP server).

    Disables the addon's phone-home telemetry (it POSTs tool events to a Cloud
    Run endpoint by default) and passes the resolved Godot path through.
    """
    env: Dict[str, str] = {
        "GODOT_AI_DISABLE_TELEMETRY": "1",
        "DISABLE_TELEMETRY": "1",
    }
    godot_path = _resolve_godot_path()
    if godot_path:
        env["GODOT_PATH"] = godot_path
    return env


_SCREENSHOT_GUIDANCE = """

You have access to a Godot MCP (Model Context Protocol) server that provides specialized tools for working with Godot projects.

Available MCP Tools:
- `godot-screenshot`: Takes a screenshot of the Godot editor to help you visualize the current state of the project.
  - The game directory is the current directory (`./`)
  - This is useful for understanding the scene hierarchy, node structure, and visual layout
  - You can use this before making changes to understand the current state, and after to verify your changes

When to use the MCP tools:
- Before starting work: Use `godot-screenshot` to understand the current project structure
- After making changes: Use `godot-screenshot` to verify your changes are correct
- When debugging: Use `godot-screenshot` to see what the editor looks like and identify issues
"""

_GODOT_MCP_GUIDANCE = """

You have access to a Godot MCP (Model Context Protocol) server (`@coding-solo/godot-mcp`) that provides specialized tools for working directly with Godot projects. The Godot project is the current directory (`./`).

Available MCP Tools include:
- `get_godot_version`, `list_projects`, `get_project_info`: inspect the Godot install and project structure.
- `launch_editor`: open the Godot editor for a project.
- `run_project` / `stop_project` / `get_debug_output`: run the project headlessly and read back its console/debug output to verify behavior and diagnose errors.
- `create_scene`, `add_node`, `load_sprite`, `save_scene`: build and edit scenes programmatically.
- `export_mesh_library`, `get_uid`, `update_project_uids`: manage resources and UIDs.

When to use the MCP tools:
- Before starting work: use `get_project_info` to understand the project structure.
- While building: prefer the scene/node tools over hand-editing `.tscn` files when practical.
- To verify: use `run_project` and `get_debug_output` to confirm the game runs and behaves as expected, and to surface runtime errors.
"""

_GODOT_TUGCAN_GUIDANCE = """

You have access to a Godot MCP (Model Context Protocol) server (`@tugcantopaloglu/godot-mcp`) that exposes a large set of tools (~149) for working directly with Godot 4.x projects. The Godot project is the current directory (`./`).

Available MCP Tools include:
- `get_godot_version`, `get_project_info`, `list_projects`: inspect the Godot install and project structure.
- `read_scene`, `modify_scene_node`, `add_node`, `attach_script`, `create_scene`: read and edit scenes and scripts programmatically.
- `run_project` / `stop_project` / `get_debug_output`: run the project headlessly and read back its console/debug output to verify behavior and diagnose errors.
- `screenshot`: capture the editor/game for visual inspection.
- Plus tools for rendering, physics, audio, animation, input, and resources.

When to use the MCP tools:
- Before starting work: use `get_project_info` and `read_scene` to understand the project structure.
- While building: prefer the scene/node tools over hand-editing `.tscn` files when practical.
- To verify: use `run_project` and `get_debug_output` to confirm the game runs and behaves as expected.

Note: the runtime `game_*` tools require a `McpInteractionServer` autoload that the benchmark projects do not ship, so prefer the headless tools above (`read_scene`, `modify_scene_node`, `run_project`, `get_debug_output`) for inspection and verification.
"""

_GODOT_AI_GUIDANCE = """

You have access to a Godot MCP (Model Context Protocol) server (`hi-godot/godot-ai`) backed by a Godot editor plugin running against the current project. The tools operate on the *live editor*, so changes you make are applied directly to the open scene/project. The Godot project is the current directory (`./`).

Available MCP Tools include (~41 tools, ~120 operations):
- `editor_state`: report the current scene, Godot version, and readiness — call this first to confirm the editor is connected.
- `scene_get_hierarchy`, `node_get_properties`, `scene_manage`, `node_manage`: inspect and edit the scene tree and node properties programmatically.
- `script_manage`, `signal_manage`, `autoload_manage`: edit scripts, wire signals, and manage autoloads.
- `game_manage` / run + debug tools: run the project and read back console/debug output to verify behavior.
- `editor_screenshot`: capture the editor for visual inspection.
- Plus tools for UI, materials, animation, particles, camera, environment, input, and resources.

When to use the MCP tools:
- Before starting work: call `editor_state` and `scene_get_hierarchy` to understand the live project state.
- While building: prefer these scene/node/script tools over hand-editing `.tscn`/`.gd` files when practical — they mutate the editor directly.
- To verify: run the project and read debug output to confirm the game behaves as expected.
"""


SCREENSHOT = MCPServerSpec(
    name="screenshot",
    server_id="godot-screenshot",
    command="uv",
    args=("run", "gamedevbench-mcp"),
    prompt_guidance=_SCREENSHOT_GUIDANCE,
    allowed_tools=("mcp__godot-screenshot__godot-screenshot",),
    # Captures a whole monitor — parallel runs would fight over the same screen.
    exclusive_display=True,
)

GODOT = MCPServerSpec(
    name="godot",
    server_id="godot",
    command="npx",
    args=("-y", "@coding-solo/godot-mcp"),
    prompt_guidance=_GODOT_MCP_GUIDANCE,
    env_factory=_godot_mcp_env,
    # Headless (stdout-only) tools, so parallel runs don't collide on a display.
    exclusive_display=False,
    # First `npx` launch downloads the package; prime it once so workers don't race.
    prefetch=True,
)

GODOT_TUGCAN = MCPServerSpec(
    name="godot-tugcan",
    server_id="godot-tugcan",
    command="npx",
    args=("-y", "@tugcantopaloglu/godot-mcp"),
    prompt_guidance=_GODOT_TUGCAN_GUIDANCE,
    # Same GODOT_PATH resolution as the other godot server.
    env_factory=_godot_mcp_env,
    # Headless (stdout-only) tools, so parallel runs don't collide on a display.
    exclusive_display=False,
    # First `npx` launch downloads the package; prime it once so workers don't race.
    prefetch=True,
)


GODOT_AI = MCPServerSpec(
    name="godot-ai",
    server_id="godot-ai",
    # command/args prime the uvx package cache during warm_up (the plugin itself
    # spawns the real server); they are NOT how the agent reaches the server.
    command="uvx",
    args=("--from", f"godot-ai=={GODOT_AI_VERSION}", "godot-ai", "--version"),
    prompt_guidance=_GODOT_AI_GUIDANCE,
    env_factory=_godot_ai_env,
    # The plugin spawns a Python FastMCP server the agent reaches over HTTP.
    transport="http",
    http_url=GODOT_AI_HTTP_URL,
    # A live Godot editor with the plugin must run per task (see godot_ai_editor).
    # Each task's editor gets its own free ports + isolated editor state, so this
    # runs in parallel (NOT single-worker, unlike the screenshot baseline).
    needs_godot_editor=True,
    # First `uvx` launch downloads the package; prime it once before dispatch.
    # (The runner also clones the editor addon once up front for editor servers.)
    prefetch=True,
)


_REGISTRY: Dict[str, MCPServerSpec] = {
    spec.name: spec for spec in (SCREENSHOT, GODOT, GODOT_TUGCAN, GODOT_AI)
}

#: Default server preserves the original ``--enable-mcp`` baseline behavior.
DEFAULT_MCP_SERVER = SCREENSHOT.name


def available_mcp_servers() -> list[str]:
    """Return the selectable MCP server names (for ``--mcp-server`` choices)."""
    return sorted(_REGISTRY)


def get_mcp_server(name: Optional[str]) -> MCPServerSpec:
    """Look up an MCP server spec by name, defaulting to the baseline.

    Raises:
        ValueError: if ``name`` is not a registered server.
    """
    key = name or DEFAULT_MCP_SERVER
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown MCP server: {key!r}. "
            f"Available servers: {', '.join(available_mcp_servers())}"
        )
    return _REGISTRY[key]
