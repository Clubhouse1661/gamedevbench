#!/usr/bin/env python3
"""Cross-platform process-tree termination helpers."""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Iterable


def terminate_process_tree(
    pid: int,
    term_grace: float = 3.0,
    *,
    kill_process_group: bool = False,
) -> None:
    """Terminate ``pid`` and descendants, best-effort and cross-platform.

    Worker tasks can spawn a tree (Codex -> MCP servers -> Godot). Killing only
    the direct worker leaves descendants alive, especially on Windows. Prefer
    ``psutil`` when available; fall back to platform tools otherwise.
    """
    if pid <= 0:
        return

    try:
        import psutil
    except Exception:
        _terminate_tree_without_psutil(pid, kill_process_group=kill_process_group)
        return

    try:
        parent = psutil.Process(pid)
    except Exception:
        return

    procs = parent.children(recursive=True)
    procs.append(parent)
    _signal_processes(procs, "terminate")
    _gone, alive = psutil.wait_procs(procs, timeout=term_grace)
    _signal_processes(alive, "kill")


def _signal_processes(procs: Iterable[object], method_name: str) -> None:
    for proc in procs:
        try:
            getattr(proc, method_name)()
        except Exception:
            pass


def _terminate_tree_without_psutil(pid: int, *, kill_process_group: bool) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass
        return

    if kill_process_group:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            return
        except Exception:
            pass

    try:
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
