"""Regression tests for the solver registry/factory.

These lock in the public contract used by benchmark_runner and by anyone adding
a new agent (e.g. a Godot-specific MCP-backed solver).
"""
import pytest

from gamedevbench.src.base_solver import BaseSolver
from gamedevbench.src.solver_factory import SolverFactory
from gamedevbench.src.utils.data_types import SolverResult

CORE_AGENTS = {"claude-code", "codex", "gemini-cli", "mini-swe"}


@pytest.fixture
def restore_registry():
    """Snapshot and restore the class-level registry around mutating tests."""
    snapshot = dict(SolverFactory._SOLVER_REGISTRY)
    yield
    SolverFactory._SOLVER_REGISTRY = snapshot


class _DummySolver(BaseSolver):
    SUPPORTS_MCP = False
    SUPPORTS_SYSTEM_PROMPT = False

    def solve_task(self) -> SolverResult:
        return SolverResult(success=True, message="dummy", duration_seconds=0.0)

    @staticmethod
    def is_rate_limit_error(error_message: str) -> bool:
        return False


class _DummyMcpSolver(_DummySolver):
    SUPPORTS_MCP = True


class _DummyVerifySolver(_DummySolver):
    SUPPORTS_VERIFICATION_NUDGE = True


def test_core_agents_registered():
    assert CORE_AGENTS <= set(SolverFactory.get_available_agents())


def test_available_agents_sorted():
    agents = SolverFactory.get_available_agents()
    assert agents == sorted(agents)


def test_unknown_agent_raises():
    with pytest.raises(ValueError, match="Unknown agent"):
        SolverFactory.create_solver("does-not-exist")


def test_every_shipped_solver_supports_mcp():
    # Today every registered solver advertises MCP. This guards against a new
    # solver silently breaking the registry or forgetting to declare support.
    assert set(SolverFactory.get_mcp_capable_solvers()) == set(
        SolverFactory.get_available_agents()
    )


@pytest.mark.parametrize(
    "agent,system_prompt",
    [
        ("claude-code", True),
        ("codex", False),
        ("gemini-cli", False),
        ("mini-swe", False),
    ],
)
def test_solver_info_capabilities(agent, system_prompt):
    info = SolverFactory.get_solver_info(agent)
    assert info["supports_mcp"] is True
    assert info["supports_system_prompt"] is system_prompt


def test_solver_info_unknown_raises():
    with pytest.raises(ValueError):
        SolverFactory.get_solver_info("nope")


def test_register_non_solver_rejected(restore_registry):
    class NotASolver:
        pass

    with pytest.raises(TypeError):
        SolverFactory.register_solver("bad", NotASolver)


def test_register_and_create_dummy(restore_registry):
    SolverFactory.register_solver("dummy", _DummySolver)
    assert isinstance(SolverFactory.create_solver("dummy"), _DummySolver)


def test_create_with_mcp_on_unsupported_raises(restore_registry):
    SolverFactory.register_solver("dummy", _DummySolver)
    with pytest.raises(ValueError, match="does not support MCP"):
        SolverFactory.create_solver("dummy", use_mcp=True)


def test_create_with_mcp_on_supported_sets_flag(restore_registry):
    SolverFactory.register_solver("dummy-mcp", _DummyMcpSolver)
    solver = SolverFactory.create_solver("dummy-mcp", use_mcp=True)
    assert solver.use_mcp is True
    # Default selection resolves to the screenshot baseline.
    assert solver.mcp_server == "screenshot"


def test_unknown_mcp_server_rejected(restore_registry):
    SolverFactory.register_solver("dummy-mcp", _DummyMcpSolver)
    with pytest.raises(ValueError, match="Unknown MCP server"):
        SolverFactory.create_solver(
            "dummy-mcp", use_mcp=True, mcp_server="bogus"
        )


def test_verification_nudge_on_unsupported_raises(restore_registry):
    SolverFactory.register_solver("dummy", _DummySolver)
    with pytest.raises(ValueError, match="does not support --encourage-verification"):
        SolverFactory.create_solver("dummy", encourage_verification=True)


def test_verification_nudge_on_supported_sets_flag(restore_registry):
    SolverFactory.register_solver("dummy-verify", _DummyVerifySolver)
    solver = SolverFactory.create_solver("dummy-verify", encourage_verification=True)
    assert solver.encourage_verification is True


def test_verification_nudge_defaults_off(restore_registry):
    SolverFactory.register_solver("dummy-verify", _DummyVerifySolver)
    solver = SolverFactory.create_solver("dummy-verify")
    assert solver.encourage_verification is False


def test_unwired_non_default_mcp_server_rejected(restore_registry):
    # Server selection is per-agent allowlisted. Pairing an unwired server with
    # an agent must fail loudly, not silently fall back to the screenshot
    # baseline or emit a partial config.
    SolverFactory.register_solver("dummy-mcp", _DummyMcpSolver)
    with pytest.raises(ValueError, match="not supported with agent 'dummy-mcp'"):
        SolverFactory.create_solver(
            "dummy-mcp", use_mcp=True, mcp_server="godot"
        )


def test_codex_allows_godot_ai_mcp_server():
    solver = SolverFactory.create_solver(
        "codex", use_mcp=True, mcp_server="godot-ai"
    )
    assert solver.mcp_server == "godot-ai"


def test_codex_accepts_verification_nudge():
    solver = SolverFactory.create_solver("codex", encourage_verification=True)
    assert solver.encourage_verification is True


def test_codex_rejects_unwired_godot_mcp_server():
    with pytest.raises(ValueError, match="not supported with agent 'codex'"):
        SolverFactory.create_solver(
            "codex", use_mcp=True, mcp_server="godot"
        )
