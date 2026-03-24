import contextlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal, cast

import pytest
from inspect_ai import eval
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessageAssistant

MitmproxyE2EAgent = Literal[
    "claude_code",
    "codex_cli",
    "gemini_cli",
    "interactive_claude_code",
    "interactive_codex_cli",
    "interactive_gemini_cli",
]

OPENAI_AGENTS: set[MitmproxyE2EAgent] = {"codex_cli", "interactive_codex_cli"}
ANTHROPIC_AGENTS: set[MitmproxyE2EAgent] = {
    "claude_code",
    "interactive_claude_code",
}
GOOGLE_AGENTS: set[MitmproxyE2EAgent] = {"gemini_cli", "interactive_gemini_cli"}

MODEL_BY_AGENT: dict[MitmproxyE2EAgent, str] = {
    "claude_code": "none",
    "codex_cli": "none",
    "gemini_cli": "none",
    "interactive_claude_code": "none",
    "interactive_codex_cli": "none",
    "interactive_gemini_cli": "none",
}

TARGET_TEXT = "ALPHA-SECRET-12345 VERIFIED"
SECRET_KEY = "alpha"
SECRET_VALUE = "ALPHA-SECRET-12345"


def has_mitmproxy_support() -> bool:
    return importlib.util.find_spec("mitmproxy") is not None and shutil.which(
        "mitmdump"
    ) is not None


def skip_if_no_mitmproxy(func):  # type: ignore[no-untyped-def]
    return pytest.mark.skipif(
        not has_mitmproxy_support(),
        reason="Test requires mitmproxy extra and mitmdump binary",
    )(func)


def skip_if_no_openai_live(func):  # type: ignore[no-untyped-def]
    return pytest.mark.skipif(
        not Path.home().joinpath(".codex", "auth.json").exists(),
        reason="Test requires ~/.codex/auth.json",
    )(func)


def skip_if_no_anthropic_live(func):  # type: ignore[no-untyped-def]
    return pytest.mark.skipif(
        _extract_claude_credentials_json() is None,
        reason="Test requires Claude Code-credentials keychain item or credentials file",
    )(func)


def skip_if_no_google_live(func):  # type: ignore[no-untyped-def]
    return pytest.mark.skipif(
        os.environ.get("GOOGLE_API_KEY") is None and os.environ.get("GEMINI_API_KEY") is None,
        reason="Test requires GOOGLE_API_KEY or GEMINI_API_KEY",
    )(func)


@dataclass
class HarnessRequest:
    host: str
    path: str
    body: str
    method: str


@dataclass
class HarnessState:
    requests: list[HarnessRequest] = field(default_factory=list)

    def record(self, host: str, path: str, body: str, method: str) -> None:
        self.requests.append(
            HarnessRequest(host=host, path=path, body=body, method=method)
        )


class HarnessHandler(BaseHTTPRequestHandler):
    state: HarnessState

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        original_host = self.headers.get("x-inspect-original-host", self.headers.get("host", ""))
        self.state.record(original_host, self.path, body, "POST")

        host = original_host.lower()
        if "openai.com" in host or "chatgpt.com" in host:
            self._handle_openai(body)
        elif "anthropic.com" in host:
            self._handle_anthropic(body)
        elif "googleapis.com" in host:
            self._handle_google(body)
        else:
            self.send_error(404, f"Unsupported host: {original_host}")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        del format, args

    def _handle_openai(self, body: str) -> None:
        payload = json.loads(body or "{}")
        is_final = any(
            item.get("type") == "function_call_output"
            for item in payload.get("input", [])
            if isinstance(item, dict)
        )
        if self.path.startswith("/v1/responses"):
            self._write_sse(
                _openai_responses_final_text_events()
                if is_final
                else _openai_responses_tool_call_events()
            )
            return
        if self.path.startswith("/v1/chat/completions"):
            self._write_json(
                _openai_completions_final_text()
                if is_final
                else _openai_completions_tool_call()
            )
            return
        self.send_error(404, f"Unsupported OpenAI path: {self.path}")

    def _handle_anthropic(self, body: str) -> None:
        is_final = '"tool_result"' in body
        self._write_sse(
            _anthropic_final_text_events()
            if is_final
            else _anthropic_tool_call_events()
        )

    def _handle_google(self, body: str) -> None:
        payload = json.loads(body or "{}")
        is_final = _google_has_function_response(payload)
        if "streamGenerateContent" in self.path:
            self._write_sse(
                _google_final_text_events()
                if is_final
                else _google_tool_call_events()
            )
        else:
            self._write_json(
                _google_final_text_response()
                if is_final
                else _google_tool_call_response()
            )

    def _write_json(self, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _write_sse(self, events: list[str]) -> None:
        body = "".join(events).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@dataclass
class HarnessServer:
    server: ThreadingHTTPServer
    state: HarnessState
    thread: threading.Thread

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)


def start_harness_server() -> HarnessServer:
    state = HarnessState()

    class BoundHandler(HarnessHandler):
        pass

    BoundHandler.state = state
    server = ThreadingHTTPServer(("127.0.0.1", 0), BoundHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return HarnessServer(server=server, state=state, thread=thread)


@dataclass
class MitmproxyE2EResult:
    log: EvalLog
    artifact_dir: Path
    requests: list[HarnessRequest]
    captures: list[dict[str, str]]


def run_mitmproxy_harness_e2e(
    agent: MitmproxyE2EAgent,
    *,
    sandbox: str = "docker",
    upstream_base_url: str | None = None,
) -> MitmproxyE2EResult:
    harness = None if upstream_base_url is not None else start_harness_server()
    artifact_dir = Path(tempfile.mkdtemp(prefix=f"mitmproxy_harness_{agent}_"))
    capture_file = artifact_dir / "captures.tsv"
    stderr_file = artifact_dir / "mitmdump.stderr.log"
    raw_capture_dir = artifact_dir / "raw"
    base_url = upstream_base_url or (harness.base_url if harness is not None else "")

    overrides = {
        "api.openai.com": base_url,
        "chatgpt.com": base_url,
        "api.anthropic.com": base_url,
        "generativelanguage.googleapis.com": base_url,
        "cloudcode-pa.googleapis.com": base_url,
        "aiplatform.googleapis.com": base_url,
    }

    env = {
        "OPENAI_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "test-key",
        "ANTHROPIC_AUTH_TOKEN": "test-key",
        "GOOGLE_API_KEY": "test-key",
        "GEMINI_API_KEY": "test-key",
        "INSPECT_SWE_MITMPROXY_UPSTREAM_OVERRIDES": json.dumps(overrides),
        "INSPECT_SWE_MITMPROXY_CAPTURE_FILE": str(capture_file),
        "INSPECT_SWE_MITMPROXY_STDERR_FILE": str(stderr_file),
        "INSPECT_SWE_MITMPROXY_RAW_DIR": str(raw_capture_dir),
    }

    try:
        with _patched_env(env):
            logs = eval(
                os.path.join("examples", "mitmproxy_e2e"),
                model=MODEL_BY_AGENT[agent],
                limit=1,
                task_args={"agent": agent, "sandbox": sandbox},
            )
        log = logs[0]
        _write_eval_log(artifact_dir / "eval_log.json", log)
        _write_requests(
            artifact_dir / "harness_requests.json",
            harness.state.requests if harness is not None else [],
        )
        captures = _read_capture_file(capture_file)
        return MitmproxyE2EResult(
            log=log,
            artifact_dir=artifact_dir,
            requests=list(harness.state.requests) if harness is not None else [],
            captures=captures,
        )
    finally:
        if harness is not None:
            harness.close()


def run_mitmproxy_live_smoke(
    agent: MitmproxyE2EAgent,
    *,
    sandbox: str = "docker",
) -> MitmproxyE2EResult:
    artifact_dir = Path(tempfile.mkdtemp(prefix=f"mitmproxy_live_{agent}_"))
    capture_file = artifact_dir / "captures.tsv"
    stderr_file = artifact_dir / "mitmdump.stderr.log"
    raw_capture_dir = artifact_dir / "raw"
    claude_credentials_path = _write_claude_credentials_temp_file()
    codex_auth_path = _write_fresh_codex_auth_temp_file()

    env = {
        "INSPECT_SWE_MITMPROXY_CAPTURE_FILE": str(capture_file),
        "INSPECT_SWE_MITMPROXY_STDERR_FILE": str(stderr_file),
        "INSPECT_SWE_MITMPROXY_RAW_DIR": str(raw_capture_dir),
        "INSPECT_SWE_CLAUDE_OAUTH_FILE": str(Path.home() / ".claude.json"),
        "INSPECT_SWE_CLAUDE_CREDENTIALS_FILE": (
            str(claude_credentials_path) if claude_credentials_path is not None else ""
        ),
        "INSPECT_SWE_CLAUDE_OAUTH_ACCESS_TOKEN": (
            _extract_claude_access_token() or ""
        ),
        "INSPECT_SWE_CLAUDE_SETTINGS_FILE": str(Path.home() / ".claude" / "settings.json"),
        "INSPECT_SWE_CODEX_AUTH_FILE": (
            str(codex_auth_path) if codex_auth_path is not None else ""
        ),
        "INSPECT_SWE_CODEX_CONFIG_FILE": str(Path.home() / ".codex" / "config.json"),
    }

    try:
        with _patched_env(
            env,
            unset_keys=["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
        ):
            logs = eval(
                os.path.join("examples", "mitmproxy_live_smoke"),
                model=MODEL_BY_AGENT[agent],
                limit=1,
                task_args={"agent": agent, "sandbox": sandbox},
            )
        log = logs[0]
        _write_eval_log(artifact_dir / "eval_log.json", log)
        captures = _read_capture_file(capture_file)
        return MitmproxyE2EResult(
            log=log,
            artifact_dir=artifact_dir,
            requests=[],
            captures=captures,
        )
    finally:
        if claude_credentials_path is not None and claude_credentials_path.exists():
            claude_credentials_path.unlink()
        if codex_auth_path is not None and codex_auth_path.exists():
            codex_auth_path.unlink()


def expected_host(agent: MitmproxyE2EAgent) -> str:
    if agent in OPENAI_AGENTS:
        return "api.openai.com"
    if agent in ANTHROPIC_AGENTS:
        return "api.anthropic.com"
    return "generativelanguage.googleapis.com"


def expected_path_fragment(agent: MitmproxyE2EAgent) -> str:
    if agent in OPENAI_AGENTS:
        return "/v1/responses"
    if agent in ANTHROPIC_AGENTS:
        return "/v1/messages"
    return "GenerateContent"


def has_secret_lookup_tool_call(log: EvalLog) -> bool:
    if not log.samples:
        return False
    assistant_messages = [
        message
        for message in log.samples[0].messages
        if isinstance(message, ChatMessageAssistant)
    ]
    for message in assistant_messages:
        for tool_call in message.tool_calls or []:
            if "secret_lookup" in tool_call.function:
                return True
    return False


def _write_eval_log(path: Path, log: EvalLog) -> None:
    path.write_text(log.model_dump_json(indent=2), encoding="utf-8")


def _write_requests(path: Path, requests: list[HarnessRequest]) -> None:
    payload = [asdict(request) for request in requests]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_capture_file(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    captures: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        provider, host, capture_path = line.split("\t", 2)
        captures.append({"provider": provider, "host": host, "path": capture_path})
    return captures


def _extract_claude_credentials_json() -> str | None:
    explicit_path = os.environ.get("INSPECT_SWE_CLAUDE_CREDENTIALS_FILE")
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None
    payload = result.stdout.strip()
    return payload or None


def _write_claude_credentials_temp_file() -> Path | None:
    payload = _extract_claude_credentials_json()
    if payload is None:
        return None
    with tempfile.NamedTemporaryFile(
        prefix="claude-credentials-",
        suffix=".json",
        delete=False,
    ) as handle:
        handle.write(payload.encode("utf-8"))
        return Path(handle.name)


def _extract_claude_access_token() -> str | None:
    payload = _extract_claude_credentials_json()
    if payload is None:
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    oauth = parsed.get("claudeAiOauth")
    if isinstance(oauth, dict):
        token = oauth.get("accessToken")
        if isinstance(token, str) and token:
            return token
    return None


def _write_fresh_codex_auth_temp_file() -> Path | None:
    source = Path.home() / ".codex" / "auth.json"
    if not source.exists():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    payload["last_refresh"] = "2999-01-01T00:00:00Z"
    with tempfile.NamedTemporaryFile(
        prefix="codex-auth-",
        suffix=".json",
        delete=False,
    ) as handle:
        handle.write(json.dumps(payload).encode("utf-8"))
        return Path(handle.name)


@contextlib.contextmanager
def _patched_env(
    values: dict[str, str],
    unset_keys: list[str] | None = None,
) -> object:
    all_keys = set(values)
    all_keys.update(unset_keys or [])
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in all_keys}
    for key in unset_keys or []:
        os.environ.pop(key, None)
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _tool_name_for_host(host: str) -> str:
    return "secret_lookup" if "googleapis.com" in host else "mcp__secrets__secret_lookup"


def _openai_responses_tool_call_events() -> list[str]:
    tool_name = "mcp__secrets__secret_lookup"
    item = {
        "id": "fc_secret",
        "type": "function_call",
        "call_id": "call_secret",
        "name": tool_name,
        "arguments": json.dumps({"key": SECRET_KEY}),
    }
    response = {
        "id": "resp_secret",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": "gpt-5",
        "output": [item],
        "usage": {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
    }
    return [
        _sse("response.created", {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}}),
        _sse("response.output_item.added", {"type": "response.output_item.added", "response_id": "resp_secret", "output_index": 0, "item": item}),
        _sse("response.output_item.done", {"type": "response.output_item.done", "response_id": "resp_secret", "output_index": 0, "item": item}),
        _sse("response.completed", {"type": "response.completed", "response": response}),
    ]


def _openai_responses_final_text_events() -> list[str]:
    item = {
        "id": "msg_secret",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": TARGET_TEXT,
                "annotations": [],
            }
        ],
        "status": "completed",
    }
    response = {
        "id": "resp_final",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": "gpt-5",
        "output": [item],
        "usage": {"input_tokens": 10, "output_tokens": 6, "total_tokens": 16},
    }
    return [
        _sse("response.created", {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}}),
        _sse("response.output_item.added", {"type": "response.output_item.added", "response_id": "resp_final", "output_index": 0, "item": item}),
        _sse("response.output_item.done", {"type": "response.output_item.done", "response_id": "resp_final", "output_index": 0, "item": item}),
        _sse("response.completed", {"type": "response.completed", "response": response}),
    ]


def _openai_completions_tool_call() -> dict[str, object]:
    return {
        "id": "chatcmpl_secret",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gpt-5",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_secret",
                            "type": "function",
                            "function": {
                                "name": "mcp__secrets__secret_lookup",
                                "arguments": json.dumps({"key": SECRET_KEY}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }


def _openai_completions_final_text() -> dict[str, object]:
    return {
        "id": "chatcmpl_final",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gpt-5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": TARGET_TEXT},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
    }


def _anthropic_tool_call_events() -> list[str]:
    tool_name = "mcp__secrets__secret_lookup"
    message = {
        "id": "msg_secret",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 8, "output_tokens": 0},
    }
    return [
        _sse("message_start", {"type": "message_start", "message": message}),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_secret",
                    "name": tool_name,
                    "input": {},
                },
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps({"key": SECRET_KEY}),
                },
            },
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 4},
            },
        ),
        _sse("message_stop", {"type": "message_stop"}),
    ]


def _anthropic_final_text_events() -> list[str]:
    message = {
        "id": "msg_final",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }
    return [
        _sse("message_start", {"type": "message_start", "message": message}),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": TARGET_TEXT},
            },
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 6},
            },
        ),
        _sse("message_stop", {"type": "message_stop"}),
    ]


def _google_tool_call_response() -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "secret_lookup",
                                "args": {"key": SECRET_KEY},
                            }
                        }
                    ],
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 8,
            "candidatesTokenCount": 4,
            "totalTokenCount": 12,
        },
        "modelVersion": "gemini-2.5-pro",
    }


def _google_final_text_response() -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": TARGET_TEXT}],
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 6,
            "totalTokenCount": 16,
        },
        "modelVersion": "gemini-2.5-pro",
        "text": TARGET_TEXT,
    }


def _google_tool_call_events() -> list[str]:
    return [_sse_data(_google_tool_call_response())]


def _google_final_text_events() -> list[str]:
    return [_sse_data(_google_final_text_response())]


def _google_has_function_response(payload: dict[str, object]) -> bool:
    for content in cast(list[dict[str, object]], payload.get("contents", [])):
        for part in cast(list[dict[str, object]], content.get("parts", [])):
            if "functionResponse" in part or "function_response" in part:
                return True
    return False


def _sse(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _sse_data(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"
