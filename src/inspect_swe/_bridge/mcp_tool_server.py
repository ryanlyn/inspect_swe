"""Host-side MCP server for bridged Inspect tools."""

from __future__ import annotations

import asyncio
import inspect
import threading
from dataclasses import dataclass, field

import uvicorn
from inspect_ai.tool import Tool
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.tool._mcp._tools_bridge import BridgedToolsSpec
from inspect_ai.tool._tool_def import ToolDef
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from .ca import PROXY_HOST


def _find_free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


@dataclass
class _FastMCPInstance:
    name: str
    port: int
    server: uvicorn.Server
    thread: threading.Thread


@dataclass
class MCPToolServer:
    """Real streamable-HTTP MCP server(s) for bridged Inspect tools."""

    loop: asyncio.AbstractEventLoop
    configs: list[MCPServerConfigHTTP] = field(default_factory=list)
    port: int | None = None

    def __post_init__(self) -> None:
        self._instances: list[_FastMCPInstance] = []

    async def add_tool_set(self, spec: BridgedToolsSpec) -> None:
        """Start one FastMCP server for a bridged tool set."""
        port = _find_free_port()
        app = FastMCP(
            name=spec.name,
            host="127.0.0.1",
            port=port,
            streamable_http_path="/mcp",
            stateless_http=False,
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=False
            ),
        )

        for tool in spec.tools:
            definition = ToolDef(tool)
            handler = _make_tool_handler(tool, self.loop)

            app.add_tool(
                handler,
                name=definition.name,
                description=definition.description,
            )

        starlette_app = app.streamable_http_app()
        config = uvicorn.Config(
            starlette_app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(
            target=server.run,
            daemon=True,
        )
        thread.start()

        await _wait_for_port(port)
        if self.port is None:
            self.port = port
        self.configs.append(
            MCPServerConfigHTTP(
                name=spec.name,
                type="http",
                url=f"http://{PROXY_HOST}:{port}/mcp",
                tools="all",
            )
        )
        self._instances.append(
            _FastMCPInstance(
                name=spec.name,
                port=port,
                server=server,
                thread=thread,
            )
        )

    async def close(self) -> None:
        """Stop all MCP servers."""
        for instance in self._instances:
            instance.server.should_exit = True
        for instance in self._instances:
            instance.thread.join(timeout=5.0)


def _stringify_tool_result(result: object) -> str:
    import json

    if isinstance(result, str):
        return result
    try:
        return json.dumps(result)
    except TypeError:
        return str(result)


def _make_tool_handler(
    tool: Tool,
    loop: asyncio.AbstractEventLoop,
):
    async def handler(**kwargs: object) -> str:
        future = asyncio.run_coroutine_threadsafe(tool(**kwargs), loop)
        result = future.result(timeout=120.0)
        return _stringify_tool_result(result)

    handler.__name__ = getattr(tool, "__name__", "tool")
    handler.__doc__ = getattr(tool, "__doc__", None)
    handler.__signature__ = inspect.signature(tool)  # type: ignore[attr-defined]
    return handler


async def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    import socket
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        await asyncio.sleep(0.1)
    raise RuntimeError(f"MCP tool server on port {port} did not start")


async def start_mcp_tool_server(
    bridged_tools: list[BridgedToolsSpec] | None,
) -> MCPToolServer | None:
    """Start host-side MCP tool server(s) if bridged tools were provided."""
    specs = list(bridged_tools or [])
    if not specs:
        return None
    server = MCPToolServer(loop=asyncio.get_running_loop())
    for spec in specs:
        await server.add_tool_set(spec)
    return server
