#!/usr/bin/env python3
"""
Centralized prompt creation for gamedev benchmark solvers.

This module provides unified prompt creation functions used by all solver implementations,
ensuring consistency across different agents.
"""

import json
from typing import Optional


def load_task_config() -> Optional[dict]:
    """Load task configuration from task_config.json in current directory.

    Returns:
        Parsed task configuration dict, or None if loading fails
    """
    try:
        with open("task_config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


# Light, open-ended "construct your own tests" nudge. Deliberately tells the
# agent WHAT to do (verify behaviour against the spec) but not HOW (no script
# template, no introspection idioms) so any effect is attributable to the
# verification behaviour rather than to handing over the validator's technique.
# See docs / CLAUDE.md for the experiment rationale.
VERIFICATION_NUDGE_GUIDANCE = """
    - Before you finish, verify your work actually behaves as the task describes. Do not stop at "it loads without errors".
    - Re-read the instruction and turn it into the specific, observable behaviours it requires. Write a short throwaway GDScript test that loads the relevant scene/script and asserts each of those behaviours.
    - Run it headlessly, e.g. `timeout 10 godot --headless --script verify.gd`, and confirm every assertion passes.
    - Keep fixing your implementation (not the test) until it genuinely satisfies the intended behaviour.
    """


def create_task_prompt(
    config: dict,
    use_runtime_video: bool = False,
    use_mcp: bool = False,
    mcp_guidance: Optional[str] = None,
    encourage_verification: bool = False,
) -> str:
    """Create minimal task prompt with just the instruction.

    Args:
        config: Task configuration dict containing 'instruction' field
        use_runtime_video: Whether to append Godot runtime video instructions
        use_mcp: Whether to include MCP tool references
        mcp_guidance: Guidance text for the selected MCP server. When ``use_mcp``
            is set and this is None, the bundled screenshot guidance is used
            (preserves the original baseline behavior).
        encourage_verification: Whether to append the light "construct your own
            tests to verify intended behaviour" nudge (experiment condition).

    Returns:
        The instruction text with optional runtime video, MCP, and verification
        guidance
    """
    try:
        if not config or "instruction" not in config:
            raise ValueError("Invalid config: 'instruction' field missing")
    except Exception as e:
        print(f"Error creating task prompt: {e}")
        return ""
    instruction = config.get("instruction")
    
    instruction += "\n You must complete the full task without any further assistance."
    instruction += "\n Godot is installed and you can run godot using the `godot` command. It is recommended to run this with a timeout (e.g., `timeout 10 godot` for 10 second timeout) to prevent hanging."
    instruction += "You are a visual agent and can use images and videos to help you understand the state of the game."

    if use_runtime_video:
        runtime_guidance = """
    - You can run the game and get an image with `godot --path . --quit-after 1
    --write-movie output.png`.
    - You can save a movie file as wav or avi instead with `timeout 60s godot --path . --quit-after 60 --write-movie output.wav`. This is a 1 second or 60 frame video. You can adjust as necessary.
    - It is very important that you ensure godot closes after running, or else the task will hang indefinitely.
    - You should use the video or images to verify that your changes worked as expected.
    """
        instruction += runtime_guidance

    if use_mcp:
        if mcp_guidance is None:
            # Import locally to avoid a circular import at module load time.
            from gamedevbench.src.mcp_registry import get_mcp_server

            mcp_guidance = get_mcp_server(None).prompt_guidance
        instruction += mcp_guidance

    if encourage_verification:
        instruction += VERIFICATION_NUDGE_GUIDANCE

    return instruction


def create_system_prompt(use_mcp: bool = False) -> str:
    """Create system prompt for Godot game development tasks.

    Args:
        use_mcp: Deprecated - MCP guidance is now in create_task_prompt

    Returns:
        System prompt string
    """
    return "You are a Godot game development expert."
