"""Codex CLI agent via the ``codex-acp`` ACP adapter."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from inspect_ai.agent import AgentState, agent, sandbox_agent_bridge
from inspect_ai.model import get_model
from inspect_ai.tool import Skill, install_skills, read_skills
from inspect_ai.util import (
    ExecRemoteProcess,
    ExecRemoteStreamingOptions,
    SandboxEnvironment,
    store,
)
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
from inspect_swe._util.sandbox import sandbox_exec
from inspect_swe._util.toml import to_toml
from inspect_swe.acp import ACPAgent
from inspect_swe.acp.agent import ACPAgentParams

from .agentbinary import ensure_codex_acp_setup

logger = logging.getLogger(__name__)


class CodexCli(ACPAgent):
    """Codex CLI agent via the ``codex-acp`` ACP adapter.

    Subclasses :class:`ACPAgent` to provide Codex-specific setup
    (bridge, env vars, AGENTS.md, skills).
    """

    def __init__(
        self,
        *,
        disallowed_tools: list[Literal["web_search"]] | None = None,
        skills: list[str | Path | Skill] | None = None,
        home_dir: str | None = None,
        config_overrides: dict[str, str] | None = None,
        **kwargs: Unpack[ACPAgentParams],
    ) -> None:
        self._disallowed_tools = list(disallowed_tools or [])
        self._resolved_skills = read_skills(skills) if skills else None
        self._home_dir = home_dir
        self._config_overrides = config_overrides or {}
        super().__init__(**kwargs)

    @asynccontextmanager
    async def _start_agent(
        self, state: AgentState
    ) -> AsyncIterator[tuple[ExecRemoteProcess, RuntimeBridge]]:
        sbox = sandbox_env(self.sandbox)
        default_model = get_model(self.model).canonical_name()
        runtime_user = self.user
        runtime_home: str | None = None
        runtime_cwd = self.cwd
        if self.bridge == "mitmproxy":
            runtime_user, runtime_home = await ensure_sandbox_runtime_user(
                sbox,
                self.user,
                cwd=self.cwd,
            )
            if runtime_cwd in [None, "/root"]:
                runtime_cwd = runtime_home

        # Use a unique port per sample to avoid conflicts with codex-core's
        # internal services (mirrors the non-ACP codex_cli approach).
        MODEL_PORT = "codex_acp_model_port"
        port = store().get(MODEL_PORT, 3000) + 1
        store().set(MODEL_PORT, port)

        bridge_cm = (
            sandbox_agent_bridge(
                state,
                model=None,
                model_aliases=self.model_map,
                filter=self.filter,
                retry_refusals=self.retry_refusals,
                bridged_tools=self.bridged_tools or None,
                port=port,
            )
            if self.bridge == "default"
            else mitmproxy_agent_bridge(
                state,
                sandbox=self.sandbox,
                bridged_tools=self.bridged_tools or None,
            )
        )
        async with bridge_cm as bridge:
            # Install node and codex-acp in the sandbox.
            acp_binary, node_binary = await ensure_codex_acp_setup(sbox, self.user)
            node_dir = str(Path(node_binary).parent)

            # Resolve CODEX_HOME (mirrors the non-ACP codex_cli agent).
            if self._home_dir is None:
                if self.bridge == "mitmproxy":
                    assert runtime_home is not None
                    codex_home = join_path(runtime_home, ".codex")
                else:
                    working_dir = await sandbox_exec(
                        sbox, "pwd", user=self.user, cwd=self.cwd
                    )
                    codex_home = join_path(working_dir, ".codex")
            else:
                codex_home = await sandbox_exec(
                    sbox,
                    f'eval echo "{self._home_dir}"',
                    user=runtime_user,
                    cwd=self.cwd,
                )
            await sandbox_exec(sbox, cmd=f"mkdir -p {codex_home}", user=runtime_user)
            copied_codex_auth = False
            if self.bridge == "mitmproxy":
                await copy_optional_host_tree_to_sandbox(
                    sbox,
                    host_env_var="INSPECT_SWE_CODEX_HOME_DIR",
                    sandbox_root=codex_home,
                    cwd=self.cwd,
                )
                copied_codex_auth = await copy_optional_host_file_to_sandbox(
                    sbox,
                    host_env_var="INSPECT_SWE_CODEX_AUTH_FILE",
                    sandbox_path=join_path(codex_home, "auth.json"),
                    cwd=self.cwd,
                )
                await copy_optional_host_file_to_sandbox(
                    sbox,
                    host_env_var="INSPECT_SWE_CODEX_CONFIG_FILE",
                    sandbox_path=join_path(codex_home, "config.json"),
                    cwd=self.cwd,
                )
                await secure_path_for_user(
                    sbox,
                    codex_home,
                    user=runtime_user or "root",
                    recursive=True,
                    cwd=self.cwd,
                )

            # Write system prompt to AGENTS.md (Codex convention).
            resolved_prompt = self._resolve_system_prompt(state)
            if resolved_prompt:
                agents_md_path = self._agents_md_path(codex_home)
                await sbox.write_file(agents_md_path, resolved_prompt)

            # Install skills.
            if self._resolved_skills:
                skills_dir = join_path(codex_home, "skills")
                await install_skills(
                    self._resolved_skills, sbox, runtime_user, skills_dir
                )

            # Write config.toml with model provider pointing at the bridge.
            # Use the canonical model name so the bridge can resolve it
            # via model_aliases (consistent with how claude-agent-acp passes
            # ANTHROPIC_MODEL).
            config_toml_path = await self._config_toml_path(sbox, codex_home)
            bridge_url = f"http://127.0.0.1:{bridge.port}/v1"
            toml_config: dict[str, Any] = {}
            if self.bridge == "default":
                toml_config = {
                    "model": default_model,
                    "preferred_auth_method": "apikey",
                    "model_provider": "openai-proxy",
                    "model_providers.openai-proxy": {
                        "name": "OpenAI Proxy",
                        "base_url": bridge_url,
                        "env_key": "OPENAI_API_KEY",
                        "wire_api": "responses",
                    },
                }
            elif copied_codex_auth:
                toml_config["preferred_auth_method"] = "chatgpt"
                toml_config["cli_auth_credentials_store"] = "file"
            toml_config.update(self._config_overrides)
            if toml_config:
                await sbox.write_file(config_toml_path, to_toml(toml_config))

            # Environment variables (same as the non-ACP codex agent).
            sandbox_ca_path: str | None = None
            if self.bridge == "mitmproxy":
                sandbox_ca_path = await write_ca_cert_to_sandbox(sbox)
                proxy_host = await discover_sandbox_host(
                    sbox,
                    user=runtime_user,
                    cwd=runtime_cwd,
                )

            agent_env = {
                "CODEX_HOME": codex_home,
                "RUST_LOG": "warning",
                "NO_BROWSER": "1",
                "PATH": f"{node_dir}:/usr/local/bin:/usr/bin:/bin",
            }
            if self.bridge == "default":
                agent_env |= {
                    "OPENAI_API_KEY": "api-key",
                    "OPENAI_BASE_URL": bridge_url,
                }
            else:
                assert sandbox_ca_path is not None
                agent_env |= {
                    **proxy_env(bridge.port, proxy_host),
                    "CODEX_CA_CERTIFICATE": sandbox_ca_path,
                    "SSL_CERT_FILE": sandbox_ca_path,
                }
            agent_env |= self.env

            # Start ACP adapter process.
            logger.info("Starting codex-acp adapter...")
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

    def _agents_md_path(self, codex_home: str) -> str:
        """Determine where to write AGENTS.md."""
        if self._home_dir is not None:
            return join_path(codex_home, "AGENTS.md")
        elif self.cwd is not None:
            return join_path(self.cwd, "AGENTS.md")
        return "AGENTS.md"

    async def _config_toml_path(
        self,
        sbox: SandboxEnvironment,
        codex_home: str,
    ) -> str:
        """Determine where to write config.toml."""
        if self._home_dir is not None:
            return join_path(codex_home, "config.toml")
        directory = ".codex" if self.cwd is None else join_path(self.cwd, ".codex")
        await sandbox_exec(sbox, cmd=f"mkdir -p {directory}", user=self.user)
        return join_path(directory, "config.toml")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@agent(name="Codex CLI")
def interactive_codex_cli(
    *,
    # Codex-specific
    disallowed_tools: list[Literal["web_search"]] | None = None,
    skills: list[str | Path | Skill] | None = None,
    home_dir: str | None = None,
    config_overrides: dict[str, str] | None = None,
    # Forwarded to ACPAgent
    **kwargs: Unpack[ACPAgentParams],
) -> ACPAgent:
    """Codex CLI agent via ACP.

    Uses the ``codex-acp`` adapter in a sandbox.  Supports
    multi-turn sessions and mid-turn interrupts.

    Args:
        disallowed_tools: Tools to disable (currently only ``"web_search"``).
        skills: Additional skills to make available.
        home_dir: Override for ``CODEX_HOME`` directory in the sandbox.
        config_overrides: Extra Codex config.toml key-value pairs.
        **kwargs: See :class:`ACPAgentParams` for all base options.
    """
    return CodexCli(
        disallowed_tools=disallowed_tools,
        skills=skills,
        home_dir=home_dir,
        config_overrides=config_overrides,
        **kwargs,
    )
