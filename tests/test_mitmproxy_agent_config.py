import asyncio
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from inspect_ai.agent import AgentState
from inspect_ai.model import ChatMessageUser
from inspect_swe._claude_code.claude_code import claude_code
from inspect_swe._codex_cli.codex_cli import codex_cli
from inspect_swe._gemini_cli.gemini_cli import gemini_cli
from inspect_swe.acp._agents.claude_code.claude_code import interactive_claude_code
from inspect_swe.acp._agents.codex_cli.codex_cli import interactive_codex_cli
from inspect_swe.acp._agents.gemini_cli.gemini_cli import interactive_gemini_cli


@dataclass
class FakeBridge:
    port: int = 4318

    def __post_init__(self) -> None:
        self.state = AgentState(messages=[])
        self.mcp_server_configs: list[Any] = []


class FakeSandbox:
    def __init__(self) -> None:
        self.exec_remote_envs: list[dict[str, str]] = []
        self.writes: dict[str, str] = {}

    async def exec(self, cmd: list[str], user: str | None = None, cwd: str | None = None) -> Any:
        del user, cwd
        joined = " ".join(cmd)
        if "echo $HOME" in joined:
            return SimpleNamespace(success=True, stdout="/root\n", stderr="")
        if "mkdir -p" in joined:
            return SimpleNamespace(success=True, stdout="", stderr="")
        return SimpleNamespace(success=True, stdout="", stderr="")

    async def exec_remote(self, cmd: list[str], options: Any, stream: bool | None = None) -> Any:
        del cmd, stream
        self.exec_remote_envs.append(dict(options.env or {}))
        return SimpleNamespace(
            success=True,
            returncode=0,
            stdout="",
            stderr="",
        )

    async def write_file(self, path: str, content: str) -> None:
        self.writes[path] = content


class FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name.split("/", 1)[-1]

    def canonical_name(self) -> str:
        return self.name


async def _run_primary_agent(factory: Any, mode: str) -> dict[str, str]:
    sandbox = FakeSandbox()
    bridge = FakeBridge()

    @asynccontextmanager
    async def fake_default_bridge(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        yield bridge

    @asynccontextmanager
    async def fake_mitm_bridge(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        yield bridge

    state = AgentState(messages=[ChatMessageUser(content="hello")])

    with ExitStack() as stack:
        stack.enter_context(
            patch("inspect_swe._gemini_cli.gemini_cli.sandbox_env", return_value=sandbox)
        )
        stack.enter_context(
            patch("inspect_swe._claude_code.claude_code.sandbox_env", return_value=sandbox)
        )
        stack.enter_context(
            patch("inspect_swe._codex_cli.codex_cli.sandbox_env", return_value=sandbox)
        )
        stack.enter_context(
            patch("inspect_swe._gemini_cli.gemini_cli.sandbox_agent_bridge", fake_default_bridge)
        )
        stack.enter_context(
            patch("inspect_swe._claude_code.claude_code.sandbox_agent_bridge", fake_default_bridge)
        )
        stack.enter_context(
            patch("inspect_swe._codex_cli.codex_cli.sandbox_agent_bridge", fake_default_bridge)
        )
        stack.enter_context(
            patch("inspect_swe._gemini_cli.gemini_cli.mitmproxy_agent_bridge", fake_mitm_bridge)
        )
        stack.enter_context(
            patch("inspect_swe._claude_code.claude_code.mitmproxy_agent_bridge", fake_mitm_bridge)
        )
        stack.enter_context(
            patch("inspect_swe._codex_cli.codex_cli.mitmproxy_agent_bridge", fake_mitm_bridge)
        )
        stack.enter_context(
            patch("inspect_swe._gemini_cli.gemini_cli.ensure_gemini_cli_setup", return_value=("/tmp/gemini", "/tmp/node"))
        )
        stack.enter_context(
            patch("inspect_swe._claude_code.claude_code.ensure_agent_binary_installed", return_value="/tmp/claude")
        )
        stack.enter_context(
            patch("inspect_swe._codex_cli.codex_cli.ensure_agent_binary_installed", return_value="/tmp/codex")
        )
        stack.enter_context(
            patch("inspect_swe._codex_cli.codex_cli.sandbox_exec", return_value="/tmp/workdir")
        )
        stack.enter_context(
            patch("inspect_swe._gemini_cli.gemini_cli.write_ca_cert_to_sandbox", return_value="/tmp/mitmproxy-ca-cert.pem")
        )
        stack.enter_context(
            patch("inspect_swe._claude_code.claude_code.write_ca_cert_to_sandbox", return_value="/tmp/mitmproxy-ca-cert.pem")
        )
        stack.enter_context(
            patch("inspect_swe._codex_cli.codex_cli.write_ca_cert_to_sandbox", return_value="/tmp/mitmproxy-ca-cert.pem")
        )
        stack.enter_context(
            patch("inspect_swe._gemini_cli.gemini_cli.trace", return_value=None)
        )
        stack.enter_context(
            patch("inspect_swe._claude_code.claude_code.trace", return_value=None)
        )
        stack.enter_context(
            patch("inspect_swe._codex_cli.codex_cli.trace", return_value=None)
        )
        agent = factory(bridge=mode, attempts=1)
        await agent(state)

    assert sandbox.exec_remote_envs
    return sandbox.exec_remote_envs[-1]


def test_primary_agents_switch_to_mitmproxy_env() -> None:
    gemini_default = asyncio.run(_run_primary_agent(gemini_cli, "default"))
    assert "GOOGLE_GEMINI_BASE_URL" in gemini_default
    assert "HTTPS_PROXY" not in gemini_default

    gemini_mitm = asyncio.run(_run_primary_agent(gemini_cli, "mitmproxy"))
    assert "GOOGLE_GEMINI_BASE_URL" not in gemini_mitm
    assert gemini_mitm["HTTPS_PROXY"] == "http://host.docker.internal:4318"

    claude_default = asyncio.run(_run_primary_agent(claude_code, "default"))
    assert "ANTHROPIC_BASE_URL" in claude_default
    assert "HTTPS_PROXY" not in claude_default

    claude_mitm = asyncio.run(_run_primary_agent(claude_code, "mitmproxy"))
    assert "ANTHROPIC_BASE_URL" not in claude_mitm
    assert claude_mitm["NODE_EXTRA_CA_CERTS"] == "/tmp/mitmproxy-ca-cert.pem"

    codex_default = asyncio.run(_run_primary_agent(codex_cli, "default"))
    assert "OPENAI_BASE_URL" in codex_default
    assert "HTTPS_PROXY" not in codex_default

    codex_mitm = asyncio.run(_run_primary_agent(codex_cli, "mitmproxy"))
    assert "OPENAI_BASE_URL" not in codex_mitm
    assert codex_mitm["CODEX_CA_CERTIFICATE"] == "/tmp/mitmproxy-ca-cert.pem"


async def _run_acp_agent(factory: Any, mode: str) -> dict[str, str]:
    sandbox = FakeSandbox()
    bridge = FakeBridge()

    @asynccontextmanager
    async def fake_default_bridge(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        yield bridge

    @asynccontextmanager
    async def fake_mitm_bridge(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        yield bridge

    state = AgentState(messages=[ChatMessageUser(content="hello")])

    with ExitStack() as stack:
        stack.enter_context(
            patch("inspect_swe.acp.agent.sample_active", return_value=object())
        )
        stack.enter_context(
            patch("inspect_swe.acp.agent.get_model", side_effect=lambda model=None: FakeModel(str(model or "model")))
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.gemini_cli.gemini_cli.get_model", side_effect=lambda model=None: FakeModel(str(model or "model")))
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.claude_code.claude_code.get_model", side_effect=lambda model=None: FakeModel(str(model or "model")))
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.codex_cli.codex_cli.get_model", side_effect=lambda model=None: FakeModel(str(model or "model")))
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.gemini_cli.gemini_cli.sandbox_env", return_value=sandbox)
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.claude_code.claude_code.sandbox_env", return_value=sandbox)
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.codex_cli.codex_cli.sandbox_env", return_value=sandbox)
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.gemini_cli.gemini_cli.sandbox_agent_bridge", fake_default_bridge)
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.claude_code.claude_code.sandbox_agent_bridge", fake_default_bridge)
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.codex_cli.codex_cli.sandbox_agent_bridge", fake_default_bridge)
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.gemini_cli.gemini_cli.mitmproxy_agent_bridge", fake_mitm_bridge)
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.claude_code.claude_code.mitmproxy_agent_bridge", fake_mitm_bridge)
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.codex_cli.codex_cli.mitmproxy_agent_bridge", fake_mitm_bridge)
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.gemini_cli.gemini_cli.ensure_gemini_cli_setup", return_value=("/tmp/gemini", "/tmp/node"))
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.claude_code.claude_code.ensure_claude_code_acp_setup", return_value=("/tmp/claude-acp", "/tmp/node"))
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.codex_cli.codex_cli.ensure_codex_acp_setup", return_value=("/tmp/codex-acp", "/tmp/node"))
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.codex_cli.codex_cli.sandbox_exec", return_value="/tmp/workdir")
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.gemini_cli.gemini_cli.write_ca_cert_to_sandbox", return_value="/tmp/mitmproxy-ca-cert.pem")
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.claude_code.claude_code.write_ca_cert_to_sandbox", return_value="/tmp/mitmproxy-ca-cert.pem")
        )
        stack.enter_context(
            patch("inspect_swe.acp._agents.codex_cli.codex_cli.write_ca_cert_to_sandbox", return_value="/tmp/mitmproxy-ca-cert.pem")
        )
        model = {
            interactive_gemini_cli: "google/gemini-2.5-pro",
            interactive_claude_code: "anthropic/claude-sonnet-4-0",
            interactive_codex_cli: "openai/gpt-5",
        }[factory]
        agent = factory(bridge=mode, model=model)
        async with agent._start_agent(state):
            pass

    assert sandbox.exec_remote_envs
    return sandbox.exec_remote_envs[-1]


def test_acp_agents_switch_to_mitmproxy_env() -> None:
    gemini_default = asyncio.run(_run_acp_agent(interactive_gemini_cli, "default"))
    assert "GOOGLE_GEMINI_BASE_URL" in gemini_default
    gemini_mitm = asyncio.run(_run_acp_agent(interactive_gemini_cli, "mitmproxy"))
    assert "GOOGLE_GEMINI_BASE_URL" not in gemini_mitm
    assert gemini_mitm["HTTPS_PROXY"] == "http://host.docker.internal:4318"

    claude_default = asyncio.run(_run_acp_agent(interactive_claude_code, "default"))
    assert "ANTHROPIC_BASE_URL" in claude_default
    claude_mitm = asyncio.run(_run_acp_agent(interactive_claude_code, "mitmproxy"))
    assert "ANTHROPIC_BASE_URL" not in claude_mitm
    assert claude_mitm["NODE_EXTRA_CA_CERTS"] == "/tmp/mitmproxy-ca-cert.pem"

    codex_default = asyncio.run(_run_acp_agent(interactive_codex_cli, "default"))
    assert "OPENAI_BASE_URL" in codex_default
    codex_mitm = asyncio.run(_run_acp_agent(interactive_codex_cli, "mitmproxy"))
    assert "OPENAI_BASE_URL" not in codex_mitm
    assert codex_mitm["CODEX_CA_CERTIFICATE"] == "/tmp/mitmproxy-ca-cert.pem"
