from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.agent import BridgedToolsSpec, run
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, tool
from inspect_ai.util import SandboxEnvironmentType
from inspect_swe import (
    claude_code,
    codex_cli,
    gemini_cli,
    interactive_claude_code,
    interactive_codex_cli,
    interactive_gemini_cli,
)

MitmproxyE2EAgent = Literal[
    "claude_code",
    "codex_cli",
    "gemini_cli",
    "interactive_claude_code",
    "interactive_codex_cli",
    "interactive_gemini_cli",
]


@tool
def secret_lookup() -> Tool:
    async def execute(key: str) -> str:
        """Look up a secret value by key.

        Args:
            key: Secret key to look up.
        """
        secrets = {"alpha": "ALPHA-SECRET-12345"}
        return secrets.get(key, f"Unknown key: {key}")

    return execute


@solver
def mitmproxy_e2e_solver(agent_type: MitmproxyE2EAgent) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate

        target = "ALPHA-SECRET-12345 VERIFIED"
        system_prompt = (
            "You have access to a secret_lookup tool via MCP. "
            "Use it when the user asks for the alpha secret. "
            f"After using the tool, reply exactly with: {target}"
        )
        bridged_tools = [BridgedToolsSpec(name="secrets", tools=[secret_lookup()])]

        match agent_type:
            case "claude_code":
                agent = claude_code(
                    system_prompt=system_prompt,
                    bridged_tools=bridged_tools,
                    bridge="mitmproxy",
                    model="anthropic/claude-sonnet-4-0",
                )
            case "codex_cli":
                agent = codex_cli(
                    system_prompt=system_prompt,
                    bridged_tools=bridged_tools,
                    bridge="mitmproxy",
                    model="openai/gpt-5",
                )
            case "gemini_cli":
                agent = gemini_cli(
                    system_prompt=system_prompt,
                    bridged_tools=bridged_tools,
                    bridge="mitmproxy",
                )
            case "interactive_claude_code":
                agent = interactive_claude_code(
                    system_prompt=system_prompt,
                    bridged_tools=bridged_tools,
                    bridge="mitmproxy",
                    model="none",
                )
            case "interactive_codex_cli":
                agent = interactive_codex_cli(
                    system_prompt=system_prompt,
                    bridged_tools=bridged_tools,
                    bridge="mitmproxy",
                    model="none",
                )
            case "interactive_gemini_cli":
                agent = interactive_gemini_cli(
                    system_prompt=system_prompt,
                    bridged_tools=bridged_tools,
                    bridge="mitmproxy",
                    model="google/gemini-2.5-pro",
                )

        agent_state = await run(agent, state.messages)
        state.messages = agent_state.messages
        state.output = agent_state.output
        return state

    return solve


@task
def mitmproxy_e2e(
    agent: MitmproxyE2EAgent = "claude_code",
    sandbox: SandboxEnvironmentType | None = "docker",
) -> Task:
    target = "ALPHA-SECRET-12345 VERIFIED"
    if sandbox == "docker":
        sandbox = (
            "docker",
            str(Path(__file__).with_name("compose.yaml")),
        )

    return Task(
        dataset=[
            Sample(
                input=(
                    "Use the secret_lookup tool via MCP to find the secret for "
                    "the key 'alpha' and then answer exactly as instructed."
                ),
                target=target,
            )
        ],
        solver=mitmproxy_e2e_solver(agent),
        scorer=includes(),
        sandbox=sandbox,
    )
