"""Unix-socket IPC between mitmproxy addon and Inspect."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import traceback
from pathlib import Path
from typing import Any, Awaitable, Callable

from inspect_ai._util.constants import MESSAGE_CACHE
from inspect_ai.model._chat_message import ChatMessage
from inspect_ai.model._model_output import ModelOutput
from pydantic import TypeAdapter

ChatMessagesAdapter = TypeAdapter(list[ChatMessage])


def serialize_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Serialize chat messages for wire transport."""
    return ChatMessagesAdapter.dump_python(messages, mode="json")


def deserialize_messages(messages: list[dict[str, Any]]) -> list[ChatMessage]:
    """Deserialize chat messages from wire transport."""
    return ChatMessagesAdapter.validate_python(messages, context={MESSAGE_CACHE: {}})


def serialize_model_output(output: ModelOutput) -> dict[str, Any]:
    """Serialize model output for wire transport."""
    return output.model_dump(mode="json")


def deserialize_model_output(output: dict[str, Any]) -> ModelOutput:
    """Deserialize model output from wire transport."""
    return ModelOutput.model_validate(output)


class BridgeIPCListener:
    """Async Unix-socket listener for addon updates."""

    def __init__(
        self,
        socket_path: str,
        on_generation: Callable[[list[ChatMessage], ModelOutput], Awaitable[None]],
        on_warning: Callable[[str, int], Awaitable[None]] | None = None,
    ) -> None:
        self.socket_path = socket_path
        self._on_generation = on_generation
        self._on_warning = on_warning
        self._server: asyncio.AbstractServer | None = None
        self.connected = asyncio.Event()
        self.hello = asyncio.Event()

    async def start(self) -> None:
        """Start listening for addon connections."""
        path = Path(self.socket_path)
        if path.exists():
            path.unlink()
        self._server = await asyncio.start_unix_server(self._handle_client, path=path)

    async def close(self) -> None:
        """Stop the listener and clean up the socket."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        path = Path(self.socket_path)
        if path.exists():
            path.unlink()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.connected.set()
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                payload = json.loads(line.decode("utf-8"))
                kind = payload.get("type")
                if kind == "hello":
                    self.hello.set()
                elif kind == "generation":
                    try:
                        await self._on_generation(
                            deserialize_messages(payload["input_messages"]),
                            deserialize_model_output(payload["model_output"]),
                        )
                    except Exception:
                        _log_exception()
                        raise
                elif kind == "warning" and self._on_warning is not None:
                    try:
                        await self._on_warning(
                            str(payload.get("code", "unknown")),
                            int(payload.get("count", 0)),
                        )
                    except Exception:
                        _log_exception()
                        raise
        finally:
            writer.close()
            await writer.wait_closed()


class BridgeIPCClient:
    """Thread-safe Unix-socket client used from the mitmproxy addon."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._disabled = False

    def connect(self) -> None:
        """Connect to the listener and send a hello message."""
        with self._lock:
            if self._disabled or self._socket is not None:
                return
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(self._socket_path)
            self._socket = client
            self._send_locked({"type": "hello", "version": 1})

    def send_generation(
        self,
        input_messages: list[ChatMessage],
        model_output: ModelOutput,
        provider: str,
    ) -> None:
        """Send a parsed generation over the socket."""
        with self._lock:
            if self._disabled or self._socket is None:
                return
            self._send_locked(
                {
                    "type": "generation",
                    "provider": provider,
                    "input_messages": serialize_messages(input_messages),
                    "model_output": serialize_model_output(model_output),
                }
            )

    def send_warning(self, code: str, count: int) -> None:
        """Send a warning payload over the socket."""
        with self._lock:
            if self._disabled or self._socket is None:
                return
            self._send_locked({"type": "warning", "code": code, "count": count})

    def close(self) -> None:
        """Close the IPC socket."""
        with self._lock:
            if self._socket is not None:
                self._socket.close()
                self._socket = None

    def _send_locked(self, payload: dict[str, Any]) -> None:
        assert self._socket is not None
        try:
            message = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
            self._socket.sendall(message)
        except OSError:
            self._disabled = True
            try:
                self._socket.close()
            finally:
                self._socket = None


def _log_exception() -> None:
    traceback.print_exc()
    stderr_file = os.environ.get("INSPECT_SWE_MITMPROXY_STDERR_FILE")
    if stderr_file is None:
        return
    with open(stderr_file, "a", encoding="utf-8") as handle:
        traceback.print_exc(file=handle)
