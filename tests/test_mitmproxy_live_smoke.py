import pytest
from inspect_ai.model import ChatMessageAssistant

from tests.conftest import skip_if_no_docker
from tests.mitmproxy_e2e_utils import (
    expected_host,
    run_mitmproxy_live_smoke,
    skip_if_no_anthropic_live,
    skip_if_no_google_live,
    skip_if_no_mitmproxy,
    skip_if_no_openai_live,
)

pytestmark = [pytest.mark.slow, pytest.mark.api]


@skip_if_no_mitmproxy
@skip_if_no_docker
@skip_if_no_anthropic_live
def test_mitmproxy_live_smoke_anthropic() -> None:
    primary = run_mitmproxy_live_smoke("claude_code")
    interactive = run_mitmproxy_live_smoke("interactive_claude_code")

    for result in [primary, interactive]:
        assert result.log.status == "success", result.artifact_dir
        assert result.log.samples, result.artifact_dir
        assert result.captures, result.artifact_dir
        sample = result.log.samples[0]
        assistant_messages = [
            message
            for message in sample.messages
            if isinstance(message, ChatMessageAssistant)
        ]
        assert any(
            "LIVE_SMOKE_OK" in message.text
            for message in assistant_messages
        ), result.artifact_dir
        assert sample.output is not None, result.artifact_dir
        assert any(
            capture["host"] == expected_host("claude_code") for capture in result.captures
        ), result.artifact_dir


@skip_if_no_mitmproxy
@skip_if_no_docker
@skip_if_no_openai_live
def test_mitmproxy_live_smoke_openai() -> None:
    primary = run_mitmproxy_live_smoke("codex_cli")
    interactive = run_mitmproxy_live_smoke("interactive_codex_cli")

    for result in [primary, interactive]:
        assert result.log.status == "success", result.artifact_dir
        assert result.log.samples, result.artifact_dir
        assert result.captures, result.artifact_dir
        sample = result.log.samples[0]
        assistant_messages = [
            message
            for message in sample.messages
            if isinstance(message, ChatMessageAssistant)
        ]
        assert any(
            "LIVE_SMOKE_OK" in message.text
            for message in assistant_messages
        ), result.artifact_dir
        assert sample.output is not None, result.artifact_dir
        assert any(
            capture["host"] == expected_host("codex_cli") for capture in result.captures
        ), result.artifact_dir


@skip_if_no_mitmproxy
@skip_if_no_docker
@skip_if_no_google_live
def test_mitmproxy_live_smoke_google() -> None:
    primary = run_mitmproxy_live_smoke("gemini_cli")
    interactive = run_mitmproxy_live_smoke("interactive_gemini_cli")

    for result in [primary, interactive]:
        assert result.log.status == "success", result.artifact_dir
        assert result.log.samples, result.artifact_dir
        assert result.captures, result.artifact_dir
        assert any(
            capture["host"] == expected_host("gemini_cli") for capture in result.captures
        ), result.artifact_dir
