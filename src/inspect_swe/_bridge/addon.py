"""mitmdump addon for forwarding parsed LLM traffic to Inspect."""

from __future__ import annotations

import json
import os
import queue
import threading
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from inspect_swe._bridge.hosts import detect_provider, should_intercept_host
from inspect_swe._bridge.ipc import BridgeIPCClient
from inspect_swe._bridge.parsers import (
    parse_anthropic_traffic,
    parse_google_traffic,
    parse_openai_completions_traffic,
    parse_openai_responses_traffic,
)

try:  # pragma: no cover - exercised only when mitmproxy is installed
    from mitmproxy import http
except ImportError:  # pragma: no cover - import safety for test environments
    http = Any  # type: ignore[misc,assignment]


QUEUE_MAXSIZE = 64


class RawCapture:
    """Captured request/response pair."""

    def __init__(
        self,
        *,
        host: str,
        path: str,
        request_body: str,
        response_body: str,
    ) -> None:
        self.host = host
        self.path = path
        self.request_body = request_body
        self.response_body = response_body
        self.request_headers: dict[str, str] = {}
        self.response_headers: dict[str, str] = {}
        self.method: str = ""
        self.status_code: int | None = None
        self.client_messages: list[str] = []
        self.server_messages: list[str] = []


class InspectBridgeAddon:
    """mitmdump addon."""

    def __init__(self) -> None:
        socket_path = os.environ["INSPECT_BRIDGE_SOCKET"]
        self._ipc = BridgeIPCClient(socket_path)
        self._requests: dict[str, RawCapture] = {}
        self._queue: queue.Queue[RawCapture] = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._dropped = 0
        self._capture_file = os.environ.get("INSPECT_SWE_MITMPROXY_CAPTURE_FILE")
        self._raw_capture_dir = os.environ.get("INSPECT_SWE_MITMPROXY_RAW_DIR")
        self._upstream_overrides = _load_upstream_overrides()
        self._worker = threading.Thread(target=self._process_queue, daemon=True)
        self._running = True

    def load(self, loader: Any) -> None:  # pragma: no cover - mitmproxy hook
        del loader
        self._ipc.connect()
        self._worker.start()

    def done(self) -> None:  # pragma: no cover - mitmproxy hook
        self._running = False
        self._queue.put_nowait(
            RawCapture(host="", path="", request_body="", response_body="")
        )
        self._worker.join(timeout=5.0)
        self._ipc.close()

    def tls_clienthello(self, data: Any) -> None:  # pragma: no cover - mitmproxy hook
        sni = getattr(getattr(data, "client_hello", None), "sni", None)
        if not should_intercept_host(sni):
            data.ignore_connection = True

    def request(self, flow: http.HTTPFlow) -> None:  # pragma: no cover - mitmproxy hook
        host = str(getattr(flow.request, "host", ""))
        if not should_intercept_host(host):
            return
        self._requests[flow.id] = RawCapture(
            host=host,
            path=str(getattr(flow.request, "path", "")),
            request_body=_message_text(flow.request),
            response_body="",
        )
        self._requests[flow.id].request_headers = dict(flow.request.headers.items())
        self._requests[flow.id].method = str(flow.request.method)
        self._apply_upstream_override(flow, host)

    def response(self, flow: http.HTTPFlow) -> None:  # pragma: no cover - mitmproxy hook
        capture = self._requests.get(flow.id)
        if capture is None:
            return
        capture.response_body = _message_text(flow.response)
        capture.response_headers = dict(flow.response.headers.items())
        capture.status_code = int(flow.response.status_code)
        if flow.websocket is not None or capture.status_code == 101:
            return
        try:
            self._requests.pop(flow.id, None)
            self._queue.put_nowait(capture)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._dropped += 1
            self._ipc.send_warning("dropped_generations", self._dropped)
            self._queue.put_nowait(capture)

    def websocket_message(self, flow: http.HTTPFlow) -> None:  # pragma: no cover - mitmproxy hook
        capture = self._requests.get(flow.id)
        if capture is None or flow.websocket is None or not flow.websocket.messages:
            return
        message = flow.websocket.messages[-1]
        payload = _websocket_message_text(message)
        if message.from_client:
            capture.client_messages.append(payload)
        else:
            capture.server_messages.append(payload)

    def websocket_end(self, flow: http.HTTPFlow) -> None:  # pragma: no cover - mitmproxy hook
        capture = self._requests.pop(flow.id, None)
        if capture is None:
            return
        if capture.client_messages:
            capture.request_body = "\n".join(capture.client_messages)
        if capture.server_messages:
            capture.response_body = "\n".join(capture.server_messages)
        try:
            self._queue.put_nowait(capture)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._dropped += 1
            self._ipc.send_warning("dropped_generations", self._dropped)
            self._queue.put_nowait(capture)

    def _process_queue(self) -> None:
        while self._running:
            capture = self._queue.get()
            if not self._running and not capture.host:
                break
            provider = detect_provider(capture.host, capture.path)
            if provider is None:
                continue
            parser = {
                "google": parse_google_traffic,
                "anthropic": parse_anthropic_traffic,
                "openai_completions": parse_openai_completions_traffic,
                "openai_responses": parse_openai_responses_traffic,
            }[provider]
            try:
                self._record_capture(capture, provider)
                input_messages, output = parser(
                    capture.request_body,
                    capture.response_body,
                )
                self._ipc.send_generation(input_messages, output, provider)
            except Exception:
                _log_exception()
                continue

    def _apply_upstream_override(
        self,
        flow: http.HTTPFlow,
        original_host: str,
    ) -> None:
        override = self._upstream_overrides.get(original_host.lower())
        if override is None:
            return
        parsed = urlsplit(override)
        if parsed.hostname is None:
            return
        flow.request.scheme = parsed.scheme or "http"
        flow.request.host = parsed.hostname
        flow.request.port = parsed.port or (443 if flow.request.scheme == "https" else 80)
        flow.request.headers["x-inspect-original-host"] = original_host

    def _record_capture(self, capture: RawCapture, provider: str) -> None:
        if self._capture_file is None:
            base_name = None
        else:
            with open(self._capture_file, "a", encoding="utf-8") as handle:
                handle.write(
                    f"{provider}\t{capture.host}\t{capture.path}\n"
                )
            capture_index = sum(1 for _ in open(self._capture_file, encoding="utf-8"))
            base_name = f"{capture_index:03d}_{provider}"
        if self._raw_capture_dir is None:
            return
        raw_dir = Path(self._raw_capture_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        safe_name = base_name or provider
        (raw_dir / f"{safe_name}.request.txt").write_text(
            capture.request_body,
            encoding="utf-8",
        )
        (raw_dir / f"{safe_name}.response.txt").write_text(
            capture.response_body,
            encoding="utf-8",
        )
        (raw_dir / f"{safe_name}.meta.json").write_text(
            json.dumps(
                {
                    "host": capture.host,
                    "path": capture.path,
                    "method": capture.method,
                    "status_code": capture.status_code,
                    "request_headers": capture.request_headers,
                    "response_headers": capture.response_headers,
                    "client_messages": capture.client_messages,
                    "server_messages": capture.server_messages,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _load_upstream_overrides() -> dict[str, str]:
    raw = os.environ.get("INSPECT_SWE_MITMPROXY_UPSTREAM_OVERRIDES")
    if not raw:
        return {}
    try:
        import json

        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key).lower(): str(value) for key, value in parsed.items()}


def _log_exception() -> None:
    traceback.print_exc()
    stderr_file = os.environ.get("INSPECT_SWE_MITMPROXY_STDERR_FILE")
    if stderr_file is None:
        return
    with open(stderr_file, "a", encoding="utf-8") as handle:
        traceback.print_exc(file=handle)


def _message_text(message: Any) -> str:
    text = message.get_text(strict=False)
    if text:
        return text
    content_getter = getattr(message, "get_content", None)
    if callable(content_getter):
        content = content_getter(strict=False)
        if isinstance(content, bytes) and content:
            return content.decode("utf-8", errors="replace")
        if isinstance(content, str):
            return content
    raw_content = getattr(message, "raw_content", None)
    if isinstance(raw_content, bytes) and raw_content:
        return raw_content.decode("utf-8", errors="replace")
    if isinstance(raw_content, str):
        return raw_content
    return ""


def _websocket_message_text(message: Any) -> str:
    content = getattr(message, "content", b"")
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    if isinstance(content, str):
        return content
    return str(content)


addons = [InspectBridgeAddon()] if "INSPECT_BRIDGE_SOCKET" in os.environ else []
