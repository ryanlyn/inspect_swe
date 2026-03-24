import socket

import pytest
from inspect_ai.model import ChatMessageAssistant

from tests.conftest import skip_if_no_docker
from tests.mitmproxy_e2e_utils import (
    MitmproxyE2EAgent,
    expected_host,
    expected_path_fragment,
    has_secret_lookup_tool_call,
    run_mitmproxy_harness_e2e,
    skip_if_no_mitmproxy,
)

pytestmark = [pytest.mark.slow]


@skip_if_no_mitmproxy
@skip_if_no_docker
@pytest.mark.parametrize(
    "agent",
    [
        "claude_code",
        "codex_cli",
        "gemini_cli",
        "interactive_claude_code",
        "interactive_codex_cli",
        "interactive_gemini_cli",
    ],
)
def test_mitmproxy_harness_e2e(agent: MitmproxyE2EAgent) -> None:
    result = run_mitmproxy_harness_e2e(agent)

    assert result.log.status == "success"
    assert result.log.samples
    sample = result.log.samples[0]
    assert sample.output is not None

    assistant_messages = [
        message
        for message in sample.messages
        if isinstance(message, ChatMessageAssistant)
    ]
    assert any("ALPHA-SECRET-12345 VERIFIED" in message.text for message in assistant_messages)
    assert has_secret_lookup_tool_call(result.log)

    assert result.requests, f"expected harness requests, artifacts: {result.artifact_dir}"
    assert any(
        request.host == expected_host(agent)
        and expected_path_fragment(agent) in request.path
        for request in result.requests
    ), f"expected upstream request summary in {result.artifact_dir}"

    assert result.captures, f"expected capture summary, artifacts: {result.artifact_dir}"
    assert any(
        capture["host"] == expected_host(agent)
        and expected_path_fragment(agent) in capture["path"]
        for capture in result.captures
    ), f"expected mitmproxy capture summary in {result.artifact_dir}"


@skip_if_no_mitmproxy
@skip_if_no_docker
def test_mitmproxy_harness_unreachable_upstream_fails() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    result = run_mitmproxy_harness_e2e(
        "codex_cli",
        upstream_base_url=f"http://127.0.0.1:{port}",
    )

    assert result.log.status != "success", result.artifact_dir
