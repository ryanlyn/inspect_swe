from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.agent import run
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import SandboxEnvironmentType
from inspect_swe import (
    claude_code,
    codex_cli,
    gemini_cli,
    interactive_claude_code,
    interactive_codex_cli,
    interactive_gemini_cli,
)

MitmproxyLiveAgent = Literal[
    "claude_code",
    "codex_cli",
    "gemini_cli",
    "interactive_claude_code",
    "interactive_codex_cli",
    "interactive_gemini_cli",
]

TARGET_TEXT = "LIVE_SMOKE_OK"


@solver
def mitmproxy_live_solver(agent_type: MitmproxyLiveAgent) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate

        system_prompt = (
            "Reply with the exact string LIVE_SMOKE_OK and nothing else."
        )

        match agent_type:
            case "claude_code":
                agent = claude_code(
                    system_prompt=system_prompt,
                    bridge="mitmproxy",
                    model="anthropic/claude-sonnet-4-0",
                )
            case "codex_cli":
                agent = codex_cli(
                    system_prompt=system_prompt,
                    bridge="mitmproxy",
                    model="openai/gpt-5",
                )
            case "gemini_cli":
                agent = gemini_cli(
                    system_prompt=system_prompt,
                    bridge="mitmproxy",
                )
            case "interactive_claude_code":
                agent = interactive_claude_code(
                    system_prompt=system_prompt,
                    bridge="mitmproxy",
                    model="none",
                )
            case "interactive_codex_cli":
                agent = interactive_codex_cli(
                    system_prompt=system_prompt,
                    bridge="mitmproxy",
                    model="none",
                )
            case "interactive_gemini_cli":
                agent = interactive_gemini_cli(
                    system_prompt=system_prompt,
                    bridge="mitmproxy",
                    model="google/gemini-2.5-pro",
                )

        agent_state = await run(agent, state.messages)
        state.messages = agent_state.messages
        state.output = agent_state.output
        return state

    return solve


@task
def mitmproxy_live_smoke(
    agent: MitmproxyLiveAgent = "claude_code",
    sandbox: SandboxEnvironmentType | None = "docker",
) -> Task:
    if sandbox == "docker":
        compose_path = (
            Path(__file__).resolve().parent.parent / "mitmproxy_e2e" / "compose.yaml"
        )
        sandbox = (
            "docker",
            str(compose_path),
        )

    return Task(
        dataset=[Sample(input="Reply exactly with LIVE_SMOKE_OK.", target=TARGET_TEXT)],
        solver=mitmproxy_live_solver(agent),
        scorer=includes(),
        sandbox=sandbox,
    )
