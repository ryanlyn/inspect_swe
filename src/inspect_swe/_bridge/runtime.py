"""Bridge protocols shared by primary and ACP agents."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from inspect_ai.agent import AgentState
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP


@runtime_checkable
class RuntimeBridge(Protocol):
    """Common runtime bridge shape used by agent entrypoints."""

    state: AgentState
    port: int
    mcp_server_configs: list[MCPServerConfigHTTP]
