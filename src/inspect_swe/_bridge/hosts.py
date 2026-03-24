"""Target host registry and provider detection for mitmproxy bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProviderName = Literal["google", "anthropic", "openai_completions", "openai_responses"]

GOOGLE_HOSTS = {
    "generativelanguage.googleapis.com",
    "cloudcode-pa.googleapis.com",
    "aiplatform.googleapis.com",
}
OPENAI_HOSTS = {"api.openai.com", "chatgpt.com"}
ANTHROPIC_HOSTS = {"api.anthropic.com"}
VERTEX_SUFFIX = ".aiplatform.googleapis.com"
VERTEX_DASH_SUFFIX = "-aiplatform.googleapis.com"


@dataclass(frozen=True)
class TargetDecision:
    """Host interception decision."""

    host: str
    intercept: bool


def normalize_host(host: str | None) -> str:
    """Normalize a host or authority string to bare hostname."""
    if host is None:
        return ""
    host = host.strip().lower()
    if host.startswith("[") and "]" in host:
        host = host[1 : host.index("]")]
    if ":" in host and not host.count(":") > 1:
        host = host.split(":", 1)[0]
    return host


def should_intercept_host(host: str | None) -> bool:
    """True when traffic for *host* should be TLS-intercepted."""
    hostname = normalize_host(host)
    if not hostname:
        return False
    return (
        hostname in GOOGLE_HOSTS
        or hostname in OPENAI_HOSTS
        or hostname in ANTHROPIC_HOSTS
        or hostname.endswith(VERTEX_SUFFIX)
        or hostname.endswith(VERTEX_DASH_SUFFIX)
    )


def interception_decision(host: str | None) -> TargetDecision:
    """Return a host interception decision."""
    hostname = normalize_host(host)
    return TargetDecision(host=hostname, intercept=should_intercept_host(hostname))


def detect_provider(host: str | None, path: str) -> ProviderName | None:
    """Map a host/path pair to a parser/provider."""
    hostname = normalize_host(host)
    if not should_intercept_host(hostname):
        return None

    if (
        hostname in GOOGLE_HOSTS
        or hostname.endswith(VERTEX_SUFFIX)
        or hostname.endswith(VERTEX_DASH_SUFFIX)
    ):
        return "google"
    if hostname in ANTHROPIC_HOSTS:
        return "anthropic"
    if hostname in OPENAI_HOSTS:
        if path.startswith("/v1/chat/completions"):
            return "openai_completions"
        if path.startswith("/v1/responses"):
            return "openai_responses"
        if path.startswith("/backend-api/codex/responses"):
            return "openai_responses"
    return None
