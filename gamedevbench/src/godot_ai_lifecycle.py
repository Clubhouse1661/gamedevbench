"""Shared lifecycle helpers for MCP servers backed by the godot-ai editor."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from gamedevbench.src.mcp_registry import MCPServerSpec
from gamedevbench.src.utils.constants import GODOT_EXEC_PATH


@contextmanager
def maybe_godot_ai_editor_session(
    *,
    enabled: bool,
    mcp_spec: MCPServerSpec,
    project_dir: Path,
    debug: bool = False,
) -> Iterator[Optional[object]]:
    """Run a per-task godot-ai editor session when the MCP spec needs one.

    Solvers can use this without knowing whether the selected MCP server is a
    plain stdio server or the editor-backed godot-ai HTTP transport. OpenHands
    still carries its original inline lifecycle for now; this helper is the
    shared seam for newly wired solvers without changing that path blindly.
    """
    if not enabled or not mcp_spec.needs_godot_editor:
        yield None
        return

    from gamedevbench.src.godot_ai_editor import GodotAiEditorSession, ensure_addon

    with GodotAiEditorSession(
        project_dir=project_dir,
        godot_path=GODOT_EXEC_PATH,
        addon_src=ensure_addon(),
        extra_env=mcp_spec.env(),
        debug=debug,
    ) as session:
        yield session
