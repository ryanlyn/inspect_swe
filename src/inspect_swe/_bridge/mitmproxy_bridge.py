"""Host-side mitmproxy bridge context manager."""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inspect_ai.agent import AgentState
from inspect_ai.agent._bridge.types import AgentBridge
from inspect_ai.agent._bridge.util import apply_message_ids
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.tool._mcp._tools_bridge import BridgedToolsSpec

from .ca import ensure_mitmproxy_ca_cert, find_mitmdump_binary
from .ipc import BridgeIPCListener
from .mcp_tool_server import start_mcp_tool_server


@dataclass
class MitmproxyAgentBridge(AgentBridge):
    """Runtime bridge for mitmproxy traffic interception."""

    port: int
    tool_port: int | None
    ca_cert_path: Path
    mcp_server_configs: list[MCPServerConfigHTTP] = field(default_factory=list)
    warnings: dict[str, int] = field(default_factory=dict)
    mitmdump_stderr: list[str] = field(default_factory=list)

    def __init__(
        self,
        state: AgentState,
        *,
        port: int,
        tool_port: int | None,
        ca_cert_path: Path,
        mcp_server_configs: list[MCPServerConfigHTTP] | None = None,
    ) -> None:
        super().__init__(state=state, filter=None, retry_refusals=None, compaction=None)
        self.port = port
        self.tool_port = tool_port
        self.ca_cert_path = ca_cert_path
        self.mcp_server_configs = mcp_server_configs or []
        self.warnings = {}
        self.mitmdump_stderr = []
        self._accepted_input_ids: list[str] = []
        self._accepted_tail_ids: list[str] = []

    def track_candidate(
        self,
        input_messages: list[Any],
        model_output: Any,
    ) -> None:
        """Update state with a candidate generation if it matches the main thread."""
        candidate_messages = list(input_messages) + [model_output.message]
        apply_message_ids(self, candidate_messages)
        candidate_input = candidate_messages[:-1]
        candidate_input_ids = [message.id for message in candidate_input if message.id]

        if self._accept_candidate(candidate_input_ids):
            self.state.messages = candidate_messages
            self.state.output = model_output
            self._accepted_input_ids = list(candidate_input_ids)
            non_system_ids = [
                message.id
                for message in candidate_input
                if getattr(message, "role", None) != "system" and message.id
            ]
            self._accepted_tail_ids = non_system_ids[-2:]

    def record_warning(self, code: str, count: int) -> None:
        """Record a bridge warning."""
        self.warnings[code] = count

    def _accept_candidate(self, candidate_input_ids: list[str]) -> bool:
        if not self._accepted_input_ids:
            return True
        if candidate_input_ids[: len(self._accepted_input_ids)] == self._accepted_input_ids:
            return True
        return _is_subsequence(self._accepted_tail_ids, candidate_input_ids)


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    if not needle:
        return False
    index = 0
    for item in haystack:
        if item == needle[index]:
            index += 1
            if index == len(needle):
                return True
    return False


@asynccontextmanager
async def mitmproxy_agent_bridge(
    state: AgentState,
    sandbox: str | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
) -> AsyncIterator[MitmproxyAgentBridge]:
    """Create a host-side mitmproxy bridge."""
    del sandbox  # sandbox-specific work happens in agent code

    ca_cert_path = await ensure_mitmproxy_ca_cert()
    mcp_server = await start_mcp_tool_server(list(bridged_tools or []))
    proxy_port = _find_free_port()
    socket_path = os.path.join(
        tempfile.gettempdir(),
        f"inspect_bridge_{next(tempfile._get_candidate_names())}.sock",
    )
    bridge = MitmproxyAgentBridge(
        state=state,
        port=proxy_port,
        tool_port=mcp_server.port if mcp_server is not None else None,
        ca_cert_path=ca_cert_path,
        mcp_server_configs=mcp_server.configs if mcp_server is not None else [],
    )

    listener = BridgeIPCListener(
        socket_path=socket_path,
        on_generation=_generation_handler(bridge),
        on_warning=_warning_handler(bridge),
    )
    await listener.start()

    addon_path = str(Path(__file__).with_name("addon.py"))
    mitmdump = find_mitmdump_binary()
    proc = await asyncio.create_subprocess_exec(
        mitmdump,
        "--listen-host",
        "0.0.0.0",
        "--listen-port",
        str(proxy_port),
        "-s",
        addon_path,
        env={
            **os.environ,
            "INSPECT_BRIDGE_SOCKET": socket_path,
        },
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    stderr_file = os.environ.get("INSPECT_SWE_MITMPROXY_STDERR_FILE")
    stderr_task = asyncio.create_task(
        _drain_mitmdump_stderr(proc, bridge, stderr_file)
    )
    try:
        await asyncio.wait_for(listener.connected.wait(), timeout=10.0)
        await asyncio.wait_for(listener.hello.wait(), timeout=10.0)
        yield bridge
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        stderr_task.cancel()
        try:
            await stderr_task
        except asyncio.CancelledError:
            pass
        await listener.close()
        if mcp_server is not None:
            await mcp_server.close()


def _generation_handler(
    bridge: MitmproxyAgentBridge,
) -> Any:
    async def handler(input_messages: list[Any], model_output: Any) -> None:
        bridge.track_candidate(input_messages, model_output)

    return handler


def _warning_handler(
    bridge: MitmproxyAgentBridge,
) -> Any:
    async def handler(code: str, count: int) -> None:
        bridge.record_warning(code, count)

    return handler


def _find_free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


async def _drain_mitmdump_stderr(
    proc: asyncio.subprocess.Process,
    bridge: MitmproxyAgentBridge,
    stderr_file: str | None,
) -> None:
    if proc.stderr is None:
        return
    try:
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            bridge.mitmdump_stderr.append(text)
            if stderr_file is not None:
                with open(stderr_file, "a", encoding="utf-8") as handle:
                    handle.write(text)
    except asyncio.CancelledError:
        raise
