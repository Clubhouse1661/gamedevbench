#!/usr/bin/env python3
"""
OpenAI Codex solver for gamedev benchmark tasks.
Uses Codex CLI with MCP server for Godot screenshots.
"""

import json
import time
import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from gamedevbench.src.base_solver import BaseSolver
from gamedevbench.src.godot_ai_lifecycle import maybe_godot_ai_editor_session
from gamedevbench.src.mcp_registry import DEFAULT_MCP_SERVER, get_mcp_server
from gamedevbench.src.utils.data_types import SolverResult, TokenUsage
from gamedevbench.src.utils.process_tree import terminate_process_tree


class CodexSolver(BaseSolver):
    """Solver that uses OpenAI Codex CLI to complete game development tasks."""

    # Solver capabilities (required by BaseSolver)
    SUPPORTS_MCP = True
    SUPPORTS_SYSTEM_PROMPT = False  # Codex embeds context in main prompt
    SUPPORTS_VERIFICATION_NUDGE = True

    def __init__(
        self,
        timeout_seconds: int = 600,
        debug: bool = False,
        use_mcp: bool = False,
        model: Optional[str] = None,
        approval_policy: str = "never",      # untrusted | on-request | never
        sandbox: str = "danger-full-access",  # read-only | workspace-write | danger-full-access
        use_runtime_video: bool = False,
        mcp_server: str = DEFAULT_MCP_SERVER,
        encourage_verification: bool = False,
    ):
        # Call parent constructor (handles MCP validation)
        super().__init__(
            timeout_seconds,
            debug,
            use_mcp,
            use_runtime_video,
            mcp_server=mcp_server,
            encourage_verification=encourage_verification,
        )

        # Codex-specific parameters
        self.model = model
        self.approval_policy = approval_policy
        self.sandbox = sandbox

    @staticmethod
    def _toml_quote(value: str) -> str:
        """Quote a string for the small TOML snippets this solver writes."""
        return json.dumps(value)

    @classmethod
    def _format_stdio_mcp_config(
        cls, server_id: str, command: str, args: list[str]
    ) -> str:
        args_toml = ", ".join(cls._toml_quote(arg) for arg in args)
        return (
            f"[mcp_servers.{server_id}]\n"
            f"command = {cls._toml_quote(command)}\n"
            f"args = [{args_toml}]\n"
        )

    @classmethod
    def _format_http_mcp_config(cls, server_id: str, url: str) -> str:
        return (
            f"[mcp_servers.{server_id}]\n"
            f"url = {cls._toml_quote(url)}\n"
            "enabled = true\n"
            "tool_timeout_sec = 60\n"
        )

    @classmethod
    def _build_codex_mcp_config(cls, selected_server: str, http_url: str = "") -> str:
        """Build the isolated Codex MCP config for this task.

        The screenshot server is the historical Codex MCP baseline. Keep it in
        generated configs so adding a task-local godot-ai endpoint does not
        silently remove existing Codex benchmark capability.
        """
        screenshot = get_mcp_server(DEFAULT_MCP_SERVER)
        sections = [
            cls._format_stdio_mcp_config(
                screenshot.server_id,
                screenshot.command,
                list(screenshot.args),
            )
        ]
        if selected_server != DEFAULT_MCP_SERVER:
            spec = get_mcp_server(selected_server)
            if spec.transport == "http":
                sections.append(
                    cls._format_http_mcp_config(
                        spec.server_id, http_url or spec.http_url
                    )
                )
            else:
                sections.append(
                    cls._format_stdio_mcp_config(
                        spec.server_id,
                        spec.command,
                        list(spec.args),
                    )
                )
        return "\n".join(section.rstrip() for section in sections) + "\n"

    @staticmethod
    def _copy_auth_if_needed(codex_home: Path) -> None:
        """Let temp CODEX_HOME reuse file-based auth when no API key is set.

        Codex automation can use CODEX_API_KEY directly. For local runs that
        rely on an existing CLI login, copy only auth.json into the task-local
        temp home and delete it with the temp directory after the run.
        """
        if os.environ.get("CODEX_API_KEY"):
            return
        source_home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
        auth = source_home / "auth.json"
        if auth.exists():
            shutil.copy2(auth, codex_home / "auth.json")

    def _prepare_codex_home(self, http_url: str = "") -> tempfile.TemporaryDirectory:
        """Create a per-task Codex home containing only benchmark MCP config."""
        run_id = os.environ.get("GAMEDEVBENCH_TASK_RUN_ID")
        prefix = f"gamedevbench_codex_{run_id}_" if run_id else "gamedevbench_codex_"
        temp_home = tempfile.TemporaryDirectory(prefix=prefix)
        codex_home = Path(temp_home.name)
        config = self._build_codex_mcp_config(self.mcp_server, http_url=http_url)
        (codex_home / "config.toml").write_text(config, encoding="utf-8")
        self._copy_auth_if_needed(codex_home)
        if self.debug:
            print(f"Created isolated Codex config at {codex_home / 'config.toml'}")
        return temp_home

    @staticmethod
    def _codex_command() -> str:
        """Resolve the Codex launcher to a subprocess-safe executable.

        On Windows, npm installs both an extensionless shim and a .cmd wrapper.
        ``CreateProcess`` can pick the extensionless shim first and fail with
        ``Access is denied``; prefer the batch/exe wrappers explicitly.
        """
        if os.name == "nt":
            for candidate in ("codex.cmd", "codex.bat", "codex.exe", "codex"):
                resolved = shutil.which(candidate)
                if resolved:
                    return resolved
        return shutil.which("codex") or "codex"

    def _run_codex_command(
        self, cmd: list[str], *, cwd: str, env: Optional[dict[str, str]]
    ) -> subprocess.CompletedProcess:
        """Run Codex with a timeout that kills its whole child process tree."""
        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "cwd": cwd,
            "env": env,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **popen_kwargs)
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            terminate_process_tree(
                proc.pid, kill_process_group=(os.name != "nt")
            )
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                stdout, stderr = "", ""
            exc.output = stdout
            exc.stderr = stderr
            raise exc

        return subprocess.CompletedProcess(
            cmd, proc.returncode, stdout=stdout, stderr=stderr
        )

    @staticmethod
    def _coerce_int(value: Any) -> int:
        """Best-effort conversion for token counters from CLI JSON output."""
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                return 0
        return 0

    def _extract_usage_from_mapping(self, payload: Optional[dict]) -> Optional[TokenUsage]:
        """Extract token usage from one Codex JSON object."""
        if not isinstance(payload, dict):
            return None

        usage = payload.get("usage")
        if not isinstance(usage, dict):
            for key in ("payload", "event", "item"):
                nested = payload.get(key)
                if isinstance(nested, dict):
                    nested_usage = self._extract_usage_from_mapping(nested)
                    if nested_usage:
                        return nested_usage
            usage = payload

        input_details = usage.get("input_tokens_details", {})

        input_tokens = self._coerce_int(
            usage.get("input_tokens") or usage.get("prompt_tokens")
        )
        output_tokens = self._coerce_int(
            usage.get("output_tokens") or usage.get("completion_tokens")
        )
        total_tokens = self._coerce_int(usage.get("total_tokens"))
        cache_read_tokens = self._coerce_int(
            usage.get("cache_read_input_tokens")
            or usage.get("cached_input_tokens")
            or usage.get("cached_tokens")
            or (input_details.get("cached_tokens") if isinstance(input_details, dict) else 0)
        )
        cache_write_tokens = self._coerce_int(
            usage.get("cache_creation_input_tokens")
            or usage.get("cache_write_input_tokens")
        )

        if total_tokens == 0 and (input_tokens > 0 or output_tokens > 0):
            total_tokens = input_tokens + output_tokens

        if total_tokens == 0 and cache_read_tokens == 0 and cache_write_tokens == 0:
            return None

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )

    @staticmethod
    def is_rate_limit_error(error_message: str) -> bool:
        """Check if the error message indicates API rate limit."""
        error_lower = error_message.lower()
        rate_limit_keywords = [
            "rate limit", "rate_limit", "ratelimit",
            "quota exceeded", "429", "too many requests",
        ]
        return any(keyword in error_lower for keyword in rate_limit_keywords)

    def solve_task(self) -> SolverResult:
        """Solve the task using Codex CLI."""
        config = self.load_config()
        if not config:
            return SolverResult(
                success=False,
                message="Could not load task configuration",
                duration_seconds=0.0,
            )

        start_time = time.time()
        prompt = self.get_task_prompt(config)

        if self.debug:
            print("=" * 60)
            print("SENDING PROMPT TO CODEX CLI:")
            print("=" * 60)
            print(prompt)
            print("=" * 60)

        codex_home_temp = None
        try:
            with maybe_godot_ai_editor_session(
                enabled=self.use_mcp,
                mcp_spec=self.mcp_spec,
                project_dir=Path(os.getcwd()),
                debug=self.debug,
            ) as editor_session:
                http_url = editor_session.http_url if editor_session else ""

                codex_env = None
                if self.use_mcp:
                    codex_home_temp = self._prepare_codex_home(http_url=http_url)
                    codex_env = {**os.environ, "CODEX_HOME": codex_home_temp.name}

                # Build codex exec command
                cmd = [self._codex_command()]

                if self.approval_policy:
                    cmd.extend(["-a", self.approval_policy])

                cmd.extend(["exec", "--skip-git-repo-check", "--json"])

                if self.model:
                    cmd.extend(["-m", self.model])

                cmd.extend(["-s", self.sandbox, "-C", str(os.getcwd()), prompt])

                if self.debug:
                    cmd_str = " ".join([c if " " not in c else f'"{c}"' for c in cmd[:-1]])
                    print(f"Running: {cmd_str} \"...\"")
                    print("\nCODEX TRAJECTORY:")
                    print("=" * 60)

                # Run Codex. On timeout, reap the whole process tree (Codex,
                # MCP servers, and any Godot children) before returning control
                # to the benchmark worker.
                result = self._run_codex_command(
                    cmd,
                    cwd=os.getcwd(),
                    env=codex_env,
                )

                duration = time.time() - start_time
                stdout = result.stdout
                stderr = result.stderr

                if self.debug:
                    # Parse and print key events
                    self._print_trajectory(stdout)
                    print(f"\n\nDuration: {duration:.2f} seconds")
                    print(f"Exit code: {result.returncode}")
                    if stderr:
                        print(f"Stderr: {stderr[:500]}")
                    print("=" * 60)

                # Parse final response and token usage
                final_response = self._parse_final_response(stdout)
                token_usage = self._parse_token_usage(stdout)
                model_used = self._parse_model_name(stdout) or self.model or "codex"

                # Calculate cost
                cost_usd = 0.0
                if token_usage:
                    cost_usd = token_usage.calculate_cost(model_used)

                if self.debug and token_usage:
                    print(f"Tokens: input={token_usage.input_tokens}, output={token_usage.output_tokens}, total={token_usage.total_tokens}")
                    print(f"Cost: ${cost_usd:.4f}")

                # Construct message: include stderr if command failed
                if result.returncode != 0:
                    error_msg = f"Codex command failed (exit code {result.returncode})"
                    if stderr and stderr.strip():
                        error_msg += f"\nSTDERR: {stderr.strip()}"
                    if final_response:
                        error_msg += f"\nFinal response: {final_response}"
                    message = error_msg
                else:
                    message = final_response or "No response detected."

                return SolverResult(
                    success=result.returncode == 0,
                    message=message,
                    duration_seconds=duration,
                    stdout=stdout,
                    stderr=stderr,
                    token_usage=token_usage,
                    model=model_used,
                    cost_usd=cost_usd,
                )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return SolverResult(
                success=False,
                message=f"Codex execution timed out after {self.timeout_seconds}s",
                duration_seconds=duration,
            )
        except FileNotFoundError:
            return SolverResult(
                success=False,
                message="Codex CLI not found. Install with: npm i -g @openai/codex",
                duration_seconds=0.0,
            )
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            is_rate_limited = self.is_rate_limit_error(error_msg)

            if self.debug:
                print(f"\nERROR INVOKING CODEX: {error_msg}")
                if is_rate_limited:
                    print("⚠️  DETECTED RATE LIMIT/QUOTA ERROR")
                print("=" * 60)

            return SolverResult(
                success=False,
                message=f"Error invoking Codex: {error_msg}",
                duration_seconds=duration,
                is_rate_limited=is_rate_limited,
            )
        finally:
            if codex_home_temp is not None:
                codex_home_temp.cleanup()

    def _print_trajectory(self, output: str):
        """Print key events from Codex execution trajectory."""
        for line in output.strip().split("\n"):
            if not line:
                continue
            try:
                event = json.loads(line)
                event_type = event.get("type", "")
                item = event.get("item", {})
                item_type = item.get("type") if isinstance(item, dict) else ""

                if event_type == "turn.started":
                    print(f"\n[Turn Started]")
                elif event_type == "item.tool_call" or item_type == "mcp_tool_call":
                    tool_name = item.get("tool") or event.get("name", "unknown")
                    server = item.get("server")
                    args = item.get("arguments") or event.get("arguments", {})
                    label = f"{server}.{tool_name}" if server else tool_name
                    print(f"\n[Tool Call] {label}({json.dumps(args)[:100]})")
                elif event_type == "item.tool_result":
                    print("[Tool Result] received")
                elif event_type == "item.message" or item_type == "agent_message":
                    content = item.get("text") or event.get("content", "")
                    if content:
                        preview = content[:200] + "..." if len(content) > 200 else content
                        print(f"[Message] {preview}")
                elif event_type == "turn.completed":
                    print("\n[Turn Completed]")
                elif event_type == "item.file_edit" or item_type == "file_change":
                    file_path = event.get("path", "unknown")
                    if item_type == "file_change":
                        changes = item.get("changes") or []
                        if changes and isinstance(changes[0], dict):
                            file_path = changes[0].get("path", file_path)
                    print(f"[File Edit] {file_path}")
                elif event_type == "item.shell_command" or item_type == "command_execution":
                    cmd = item.get("command") or event.get("command", "")
                    print(f"[Shell] {cmd[:100]}")

            except json.JSONDecodeError:
                # Non-JSON line, possibly error message
                if line.strip() and self.debug:
                    print(f"[Raw] {line[:100]}")

    def _parse_final_response(self, output: str) -> Optional[str]:
        """Parse JSON Lines output to get final response."""
        final_response = None
        for line in output.strip().split("\n"):
            if not line:
                continue
            try:
                event = json.loads(line)
                item = event.get("item", {})
                if event.get("type") == "turn.completed":
                    final_response = event.get("finalResponse", "") or final_response
                elif event.get("type") == "item.message":
                    # Save last message as fallback
                    content = event.get("content", "")
                    if content:
                        final_response = content
                elif isinstance(item, dict) and item.get("type") == "agent_message":
                    content = item.get("text", "")
                    if content:
                        final_response = content
            except json.JSONDecodeError:
                continue
        return final_response

    def _parse_model_name(self, output: str) -> Optional[str]:
        """Parse JSON Lines output to get the model name."""
        for line in output.strip().split("\n"):
            if not line:
                continue
            try:
                event = json.loads(line)
                for candidate in (
                    event.get("model"),
                    event.get("modelName"),
                    event.get("usage", {}).get("model") if isinstance(event.get("usage"), dict) else None,
                    event.get("payload", {}).get("model") if isinstance(event.get("payload"), dict) else None,
                ):
                    if candidate:
                        return candidate
            except json.JSONDecodeError:
                continue
        return None

    def _parse_token_usage(self, output: str) -> Optional[TokenUsage]:
        """Parse JSON Lines output to get token usage."""
        fallback_input = 0
        fallback_output = 0
        fallback_cached = 0
        terminal_usage = None

        for line in output.strip().split("\n"):
            if not line:
                continue
            try:
                event = json.loads(line)
                event_type = event.get("type", "")

                if event_type in {"turn.completed", "response.completed"}:
                    usage = self._extract_usage_from_mapping(event)
                    if usage:
                        terminal_usage = usage
                        continue

                if event_type == "token_count":
                    fallback_input += self._coerce_int(event.get("input_tokens"))
                    fallback_output += self._coerce_int(event.get("output_tokens"))
                    fallback_cached += self._coerce_int(
                        event.get("cached_tokens") or event.get("cache_read_input_tokens")
                    )
                    continue

                # Also check payload.type for nested events
                payload = event.get("payload", {})
                if isinstance(payload, dict):
                    payload_type = payload.get("type", "")
                    if payload_type == "token_count":
                        fallback_input += self._coerce_int(payload.get("input_tokens"))
                        fallback_output += self._coerce_int(payload.get("output_tokens"))
                        fallback_cached += self._coerce_int(
                            payload.get("cached_tokens") or payload.get("cache_read_input_tokens")
                        )

            except json.JSONDecodeError:
                continue

        if terminal_usage:
            return terminal_usage

        if fallback_input > 0 or fallback_output > 0:
            return TokenUsage(
                input_tokens=fallback_input,
                output_tokens=fallback_output,
                total_tokens=fallback_input + fallback_output,
                cache_read_tokens=fallback_cached,
                cache_write_tokens=0,
            )
        return None


def main():
    """Main function for testing the solver."""
    solver = CodexSolver(debug=True)
    result = solver.solve_task()
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Message: {result.message[:500] if result.message else 'None'}")
    print(f"Duration: {result.duration_seconds:.2f}s")


if __name__ == "__main__":
    main()
