import os
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from inspect_ai.agent import AgentState
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.model._model_output import ModelOutput
from inspect_swe._bridge.addon import QUEUE_MAXSIZE, InspectBridgeAddon, RawCapture
from inspect_swe._bridge.hosts import detect_provider, should_intercept_host
from inspect_swe._bridge.ipc import BridgeIPCClient
from inspect_swe._bridge.mitmproxy_bridge import MitmproxyAgentBridge
from inspect_swe._bridge.parsers import (
    parse_anthropic_traffic,
    parse_google_traffic,
    parse_openai_completions_traffic,
    parse_openai_responses_traffic,
)


def test_should_intercept_host_only_targets() -> None:
    assert should_intercept_host("api.openai.com")
    assert should_intercept_host("us-central1-aiplatform.googleapis.com")
    assert not should_intercept_host("registry.npmjs.org")


def test_detect_provider_maps_paths() -> None:
    assert detect_provider("api.openai.com", "/v1/responses") == "openai_responses"
    assert (
        detect_provider("api.openai.com", "/v1/chat/completions")
        == "openai_completions"
    )
    assert (
        detect_provider("cloudcode-pa.googleapis.com", "/v1beta/models/generateContent")
        == "google"
    )
    assert detect_provider("registry.npmjs.org", "/package") is None


def test_mitmproxy_bridge_tracks_main_thread_and_rejects_short_side_calls() -> None:
    state = AgentState(messages=[])
    bridge = MitmproxyAgentBridge(
        state=state,
        port=8080,
        tool_port=None,
        ca_cert_path=Path("/tmp/mitmproxy-ca-cert.pem"),
    )

    input_messages = [ChatMessageUser(content="first")]
    output = ModelOutput.from_content("gpt-5", "assistant-1")
    bridge.track_candidate(input_messages, output)
    assert state.messages[-1].text == "assistant-1"

    follow_up = [*state.messages, ChatMessageUser(content="second")]
    second_output = ModelOutput.from_content("gpt-5", "assistant-2")
    bridge.track_candidate(follow_up, second_output)
    assert state.messages[-1].text == "assistant-2"

    side_call = [ChatMessageUser(content="side")]
    side_output = ModelOutput.from_content("gpt-5", "side-output")
    bridge.track_candidate(side_call, side_output)
    assert state.messages[-1].text == "assistant-2"


def test_mitmproxy_bridge_accepts_compacted_same_thread_candidate() -> None:
    state = AgentState(messages=[])
    bridge = MitmproxyAgentBridge(
        state=state,
        port=8080,
        tool_port=None,
        ca_cert_path=Path("/tmp/mitmproxy-ca-cert.pem"),
    )

    first_input = [
        ChatMessageUser(content="u1"),
        ChatMessageAssistant(content="a1"),
        ChatMessageUser(content="u2"),
        ChatMessageAssistant(content="a2"),
        ChatMessageUser(content="u3"),
    ]
    first_output = ModelOutput.from_content("gpt-5", "a3")
    bridge.track_candidate(first_input, first_output)

    compacted_input = [
        first_input[2],
        first_input[3],
        first_input[4],
        ChatMessageUser(content="u4"),
    ]
    compacted_output = ModelOutput.from_content("gpt-5", "a4")
    bridge.track_candidate(compacted_input, compacted_output)

    assert state.messages[-1].text == "a4"


def test_parse_google_traffic_json() -> None:
    request = {
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        "model": "gemini-2.5-pro",
    }
    response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "world"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 2,
            "candidatesTokenCount": 1,
            "totalTokenCount": 3,
        },
        "modelVersion": "gemini-2.5-pro",
    }
    messages, output = parse_google_traffic(request, response)
    assert messages[0].text == "hello"
    assert output.message.text == "world"
    assert output.usage and output.usage.total_tokens == 3


def test_parse_google_oauth_envelope() -> None:
    request = {
        "model": "projects/x/locations/us/models/gemini",
        "project": "x",
        "request": {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
    }
    response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "world"}]},
                "finishReason": "STOP",
            }
        ]
    }
    _, output = parse_google_traffic(request, response)
    assert output.message.text == "world"


def test_parse_anthropic_traffic_json() -> None:
    request = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "hello"}],
    }
    response = {
        "model": "claude-sonnet-4",
        "content": [{"type": "text", "text": "world"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    messages, output = parse_anthropic_traffic(request, response)
    assert messages[0].text == "hello"
    assert output.message.text == "world"
    assert output.usage and output.usage.output_tokens == 2


def test_parse_openai_responses_sse() -> None:
    request = {"model": "gpt-5", "input": [{"type": "message", "role": "user", "content": "hello"}]}
    response = """event: response.completed
data: {"type":"response.completed","response":{"model":"gpt-5","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"world","annotations":[]}],"status":"completed"}],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}

"""
    messages, output = parse_openai_responses_traffic(request, response)
    assert messages[0].text == "hello"
    assert output.message.text == "world"
    assert output.usage and output.usage.total_tokens == 5


def test_parse_openai_completions_json() -> None:
    request = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "hello"}],
    }
    response = {
        "model": "gpt-5",
        "choices": [
            {
                "message": {"role": "assistant", "content": "world"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    messages, output = parse_openai_completions_traffic(request, response)
    assert messages[0].text == "hello"
    assert output.message.text == "world"
    assert output.usage and output.usage.total_tokens == 5


def test_addon_queue_overflow_drops_oldest_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        monkeypatch.setenv("INSPECT_BRIDGE_SOCKET", os.path.join(tempdir, "bridge.sock"))
        addon = InspectBridgeAddon()
        addon._ipc.send_warning = MagicMock()
        for index in range(QUEUE_MAXSIZE):
            addon._queue.put_nowait(
                RawCapture(
                    host=f"host-{index}",
                    path="/v1/responses",
                    request_body="{}",
                    response_body="{}",
                )
            )

        addon._requests["flow"] = RawCapture(
            host="api.openai.com",
            path="/v1/responses",
            request_body="{}",
            response_body="",
        )
        flow = SimpleNamespace(
            id="flow",
            response=SimpleNamespace(get_text=lambda strict=False: "{}"),
        )
        addon.response(flow)

        assert addon._queue.qsize() == QUEUE_MAXSIZE
        addon._ipc.send_warning.assert_called_once_with("dropped_generations", 1)


def test_ipc_disconnect_disables_client_without_raising(tmp_path: Path) -> None:
    client = BridgeIPCClient(str(tmp_path / "unused.sock"))
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.close()
    client._socket = sock
    client.send_warning("dropped_generations", 2)
    assert client._socket is None
