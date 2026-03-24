"""Claude Code agent via the ``claude-agent-acp`` ACP adapter."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from inspect_ai.agent import AgentState, agent, sandbox_agent_bridge
from inspect_ai.model import Model, get_model
from inspect_ai.tool import Skill, install_skills, read_skills
from inspect_ai.util import ExecRemoteProcess, ExecRemoteStreamingOptions
from inspect_ai.util import sandbox as sandbox_env
from typing_extensions import Unpack

from inspect_swe._bridge.ca import (
    discover_sandbox_host,
    proxy_env,
    write_ca_cert_to_sandbox,
)
from inspect_swe._bridge.mitmproxy_bridge import mitmproxy_agent_bridge
from inspect_swe._bridge.oauth import (
    copy_optional_host_file_to_sandbox,
    copy_optional_host_tree_to_sandbox,
    ensure_sandbox_runtime_user,
    secure_path_for_user,
)
from inspect_swe._bridge.runtime import RuntimeBridge
from inspect_swe._util.path import join_path
from inspect_swe.acp import ACPAgent
from inspect_swe.acp.agent import ACPAgentParams

from .agentbinary import ensure_claude_code_acp_setup

logger = logging.getLogger(__name__)


class ClaudeCode(ACPAgent):
    """Claude Code agent via the ``claude-agent-acp`` ACP adapter.

    Subclasses :class:`ACPAgent` to provide Claude-specific setup
    (bridge, env vars, MCP config, skills).
    """

    def __init__(
        self,
        *,
        disallowed_tools: list[str] | None = None,
        skills: list[str | Path | Skill] | None = None,
        opus_model: str | Model | None = None,
        sonnet_model: str | Model | None = None,
        haiku_model: str | Model | None = None,
        subagent_model: str | Model | None = None,
        **kwargs: Unpack[ACPAgentParams],
    ) -> None:
        self._disallowed_tools = list(disallowed_tools or [])
        self._resolved_skills = read_skills(skills) if skills else None
        self._opus_model: str | Model | None = opus_model
        self._sonnet_model: str | Model | None = sonnet_model
        self._haiku_model: str | Model | None = haiku_model
        self._subagent_model: str | Model | None = subagent_model
        super().__init__(**kwargs)

    def _build_model_map(self) -> dict[str, str | Model]:
        """Build model map from all configured CC model names."""
        model_map = super()._build_model_map()
        for entry in (
            self._opus_model,
            self._sonnet_model,
            self._haiku_model,
            self._subagent_model,
        ):
            if entry is not None:
                model = get_model(entry)
                model_map[model.canonical_name()] = model
        return model_map

    @asynccontextmanager
    async def _start_agent(
        self, state: AgentState
    ) -> AsyncIterator[tuple[ExecRemoteProcess, RuntimeBridge]]:
        sbox = sandbox_env(self.sandbox)
        default_model = get_model(self.model).canonical_name()
        runtime_user = self.user
        sandbox_home: str | None = None
        runtime_cwd = self.cwd
        if self.bridge == "mitmproxy":
            runtime_user, sandbox_home = await ensure_sandbox_runtime_user(
                sbox,
                self.user,
                cwd=self.cwd,
            )
            if runtime_cwd in [None, "/root"]:
                runtime_cwd = sandbox_home

        bridge_cm = (
            sandbox_agent_bridge(
                state,
                model=None,
                model_aliases=self.model_map,
                filter=self.filter,
                retry_refusals=self.retry_refusals,
                bridged_tools=self.bridged_tools or None,
            )
            if self.bridge == "default"
            else mitmproxy_agent_bridge(
                state,
                sandbox=self.sandbox,
                bridged_tools=self.bridged_tools or None,
            )
        )
        async with bridge_cm as bridge:
            # Install node and claude-agent-acp in the sandbox.
            acp_binary, node_binary = await ensure_claude_code_acp_setup(
                sbox, self.user
            )
            node_dir = str(Path(node_binary).parent)

            # Use canonical model names — the bridge resolves them via
            # model_aliases to Model instances directly.
            sandbox_ca_path: str | None = None
            if self.bridge == "mitmproxy":
                sandbox_ca_path = await write_ca_cert_to_sandbox(sbox)
                assert sandbox_home is not None
                proxy_host = await discover_sandbox_host(
                    sbox,
                    user=runtime_user,
                    cwd=runtime_cwd,
                )
                await copy_optional_host_tree_to_sandbox(
                    sbox,
                    host_env_var="INSPECT_SWE_CLAUDE_HOME_DIR",
                    sandbox_root=f"{sandbox_home}/.claude",
                    cwd=self.cwd,
                )
                await copy_optional_host_file_to_sandbox(
                    sbox,
                    host_env_var="INSPECT_SWE_CLAUDE_OAUTH_FILE",
                    sandbox_path=f"{sandbox_home}/.claude.json",
                    cwd=self.cwd,
                )
                await copy_optional_host_file_to_sandbox(
                    sbox,
                    host_env_var="INSPECT_SWE_CLAUDE_CREDENTIALS_FILE",
                    sandbox_path=f"{sandbox_home}/.claude/.credentials.json",
                    cwd=self.cwd,
                )
                await copy_optional_host_file_to_sandbox(
                    sbox,
                    host_env_var="INSPECT_SWE_CLAUDE_SETTINGS_FILE",
                    sandbox_path=f"{sandbox_home}/.claude/settings.json",
                    cwd=self.cwd,
                )
                await secure_path_for_user(
                    sbox,
                    f"{sandbox_home}/.claude",
                    user=runtime_user or "root",
                    recursive=True,
                    cwd=self.cwd,
                )
                await secure_path_for_user(
                    sbox,
                    f"{sandbox_home}/.claude.json",
                    user=runtime_user or "root",
                    cwd=self.cwd,
                )

            agent_env = {
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
                "API_TIMEOUT_MS": "100000000",
                "IS_SANDBOX": "1",
                "PATH": f"{node_dir}:/usr/local/bin:/usr/bin:/bin",
            }
            if self.bridge == "default":
                agent_env |= {
                    "ANTHROPIC_BASE_URL": f"http://localhost:{bridge.port}",
                    "ANTHROPIC_AUTH_TOKEN": "sk-ant-api03-DOq5tyLPrk9M4hPE",
                    "ANTHROPIC_MODEL": default_model,
                    "ANTHROPIC_DEFAULT_OPUS_MODEL": get_model(
                        self._opus_model
                    ).canonical_name()
                    if self._opus_model
                    else default_model,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": get_model(
                        self._sonnet_model
                    ).canonical_name()
                    if self._sonnet_model
                    else default_model,
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": get_model(
                        self._haiku_model
                    ).canonical_name()
                    if self._haiku_model
                    else default_model,
                    "CLAUDE_CODE_SUBAGENT_MODEL": get_model(
                        self._subagent_model
                    ).canonical_name()
                    if self._subagent_model
                    else default_model,
                    "ANTHROPIC_SMALL_FAST_MODEL": get_model(
                        self._haiku_model
                    ).canonical_name()
                    if self._haiku_model
                    else default_model,
                }
            else:
                assert sandbox_ca_path is not None
                agent_env |= {
                    **proxy_env(bridge.port, proxy_host),
                    "NODE_EXTRA_CA_CERTS": sandbox_ca_path,
                    "NODE_USE_SYSTEM_CA": "1",
                    "HOME": sandbox_home,
                }
            agent_env |= self.env

            # System prompt via env (the ACP adapter will forward to CC)
            resolved_prompt = self._resolve_system_prompt(state)
            if resolved_prompt:
                agent_env["CLAUDE_CODE_APPEND_SYSTEM_PROMPT"] = resolved_prompt

            # Disallowed tools
            if self._disallowed_tools:
                agent_env["CLAUDE_CODE_DISALLOWED_TOOLS"] = ",".join(
                    self._disallowed_tools
                )

            # Install skills
            if self._resolved_skills:
                skills_base = runtime_cwd or "."
                skills_dir = join_path(skills_base, ".claude/skills")
                await install_skills(
                    self._resolved_skills, sbox, runtime_user, skills_dir
                )

            # Start ACP adapter process
            logger.info("Starting claude-agent-acp adapter...")
            proc = await sbox.exec_remote(
                cmd=[acp_binary],
                options=ExecRemoteStreamingOptions(
                    stdin_open=True,
                    cwd=runtime_cwd,
                    env=agent_env,
                    user=runtime_user,
                ),
            )

            yield proc, bridge


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@agent(name="Claude Code")
def interactive_claude_code(
    *,
    # Claude-specific
    disallowed_tools: list[str] | None = None,
    skills: list[str | Path | Skill] | None = None,
    opus_model: str | Model | None = None,
    sonnet_model: str | Model | None = None,
    haiku_model: str | Model | None = None,
    subagent_model: str | Model | None = None,
    # Forwarded to ACPAgent
    **kwargs: Unpack[ACPAgentParams],
) -> ACPAgent:
    """Claude Code agent via ACP.

    Uses the ``claude-agent-acp`` adapter in a sandbox.  Supports
    multi-turn sessions and mid-turn interrupts.

    Args:
        disallowed_tools: Tool names to disallow.
        skills: Additional skills to make available.
        opus_model: Model for opus calls.
        sonnet_model: Model for sonnet calls.
        haiku_model: Model for haiku / background calls.
        subagent_model: Model for subagents.
        **kwargs: See :class:`ACPAgentParams` for all base options.
    """
    return ClaudeCode(
        disallowed_tools=disallowed_tools,
        skills=skills,
        opus_model=opus_model,
        sonnet_model=sonnet_model,
        haiku_model=haiku_model,
        subagent_model=subagent_model,
        **kwargs,
    )
