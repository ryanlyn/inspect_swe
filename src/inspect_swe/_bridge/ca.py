"""mitmproxy CA certificate helpers."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from inspect_ai.util import SandboxEnvironment

MITMPROXY_CA_CERT = Path.home() / ".mitmproxy" / "mitmproxy-ca-cert.pem"
SANDBOX_CA_CERT = "/tmp/mitmproxy-ca-cert.pem"
PROXY_HOST = "host.docker.internal"


def find_mitmdump_binary() -> str:
    """Resolve the mitmdump binary from the active environment."""
    scripts_dir = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    candidate = scripts_dir / ("mitmdump.exe" if os.name == "nt" else "mitmdump")
    if candidate.exists():
        return str(candidate)

    binary = shutil.which("mitmdump")
    if binary is None:
        raise RuntimeError(
            "mitmdump is not available. Install the optional dependency with "
            "`uv sync --extra mitmproxy`."
        )
    return binary


async def ensure_mitmproxy_ca_cert() -> Path:
    """Ensure the mitmproxy CA certificate exists and return its path."""
    if MITMPROXY_CA_CERT.exists():
        return MITMPROXY_CA_CERT

    MITMPROXY_CA_CERT.parent.mkdir(parents=True, exist_ok=True)
    mitmdump = find_mitmdump_binary()
    proc = await asyncio.create_subprocess_exec(
        mitmdump,
        "--quiet",
        "--set",
        f"confdir={MITMPROXY_CA_CERT.parent}",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        for _ in range(20):
            if MITMPROXY_CA_CERT.exists():
                break
            await asyncio.sleep(0.25)
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()

    if not MITMPROXY_CA_CERT.exists():
        raise RuntimeError(
            f"mitmproxy CA certificate was not generated at {MITMPROXY_CA_CERT}"
        )
    return MITMPROXY_CA_CERT


async def write_ca_cert_to_sandbox(
    sandbox: SandboxEnvironment,
    path: str = SANDBOX_CA_CERT,
) -> str:
    """Write the mitmproxy CA certificate into the sandbox."""
    cert_path = await ensure_mitmproxy_ca_cert()
    await sandbox.write_file(path, cert_path.read_text())
    return path


async def discover_sandbox_host(
    sandbox: SandboxEnvironment,
    user: str | None = None,
    cwd: str | None = None,
) -> str:
    """Discover a host address reachable from the sandbox."""
    # First try an IPv4 host.docker.internal address. Some CLI tools accept
    # HTTPS proxy URLs more reliably with IPv4 literals than IPv6 literals.
    probe = await sandbox.exec(
        [
            "sh",
            "-lc",
            "getent ahostsv4 host.docker.internal 2>/dev/null | awk '{print $1}' | head -n1",
        ],
        user=user,
        cwd=cwd,
    )
    if probe.success and probe.stdout.strip():
        return probe.stdout.strip()

    # Fall back to any host.docker.internal record if only IPv6 is available.
    probe = await sandbox.exec(
        [
            "sh",
            "-lc",
            "getent hosts host.docker.internal 2>/dev/null | awk '{print $1}' | head -n1",
        ],
        user=user,
        cwd=cwd,
    )
    if probe.success and probe.stdout.strip():
        return probe.stdout.strip()

    route = await sandbox.exec(["cat", "/proc/net/route"], user=user, cwd=cwd)
    if route.success:
        for line in route.stdout.splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 3 and fields[1] == "00000000":
                gateway_hex = fields[2]
                try:
                    parts = [str(int(gateway_hex[i : i + 2], 16)) for i in range(0, 8, 2)]
                except ValueError:
                    continue
                return ".".join(reversed(parts))

    return PROXY_HOST


def proxy_env(port: int, host: str = PROXY_HOST) -> dict[str, str]:
    """Proxy-related environment variables for sandboxed agents."""
    proxy_host = _format_host_for_url(host)
    proxy = f"http://{proxy_host}:{port}"
    return {
        "HTTP_PROXY": proxy,
        "HTTPS_PROXY": proxy,
        "NO_PROXY": f"{host},localhost,127.0.0.1,host.docker.internal",
    }


def rewrite_http_url_host(url: str, host: str) -> str:
    """Rewrite the host of an HTTP URL."""
    parts = urlsplit(url)
    formatted_host = _format_host_for_url(host)
    netloc = f"{formatted_host}:{parts.port}" if parts.port is not None else formatted_host
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _format_host_for_url(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host
