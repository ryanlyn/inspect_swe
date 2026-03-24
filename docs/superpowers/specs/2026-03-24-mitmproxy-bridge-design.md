# mitmproxy Agent Bridge Design

An alternative to `sandbox_agent_bridge` that uses mitmproxy for transparent HTTPS interception. Preserves native CLI authentication (OAuth, API keys) and client identity by proxying real API traffic instead of overriding base URLs with a local relay.

## Motivation

The current `sandbox_agent_bridge` overrides each CLI's API base URL to point at a local HTTP server. This approach:

- Requires dummy API credentials (`GEMINI_API_KEY: "api-key"`)
- Cannot use native OAuth flows (e.g., Google AI Pro subscription via Gemini CLI)
- Loses client identity - Inspect's model provider makes the real API call, not the CLI
- Requires format translation for every API provider

The mitmproxy bridge instead acts as an HTTPS proxy. The CLI talks to real API endpoints with real credentials. The proxy intercepts traffic transparently, parsing request/response pairs to build Inspect's `AgentState` for scoring.

## Architecture

```
Host
  Inspect Process  <--unix socket-->  mitmdump subprocess (:proxy_port)
       |                                    |
       |                                    | HTTPS to real API
  MCP Tool Server (:tool_port)              |
       |                                    |
  -----+------------------------------------+--------
       |           Docker container         |
       |                                    |
  CLI agent (gemini/claude/codex)           |
    HTTPS_PROXY=http://host:proxy_port      |
    NODE_EXTRA_CA_CERTS=/tmp/ca.pem    Google/Anthropic/OpenAI
    Native OAuth / API key auth
```

Three host-side components:

1. **Inspect process** - orchestrates eval, listens on Unix socket for state updates, runs scoring
2. **mitmdump subprocess** - HTTPS proxy with Python addon that intercepts LLM API traffic and pushes parsed messages to Inspect
3. **MCP tool server** - standalone HTTP server exposing host-side Inspect tools to the container

## mitmdump Addon and State Tracking

### Target Hosts

The addon filters by hostname in `request()`/`response()` hooks (not via `--allow-hosts`, which has known bugs with domain matching).

| CLI | API Key Host | OAuth Host |
|---|---|---|
| Gemini CLI | `generativelanguage.googleapis.com` | `cloudcode-pa.googleapis.com` |
| Claude Code | `api.anthropic.com` | `api.anthropic.com` |
| Codex CLI | `api.openai.com` | `chatgpt.com` |

Additional Vertex AI hosts: `aiplatform.googleapis.com`, `{region}-aiplatform.googleapis.com`.

Non-targeted traffic passes through the proxy untouched.

### Request/Response Flow

1. CLI sends HTTPS request through proxy
2. Addon's `request()` hook captures the request body (JSON-parsed)
3. mitmproxy forwards to real API with TLS termination on both sides
4. Addon's `response()` hook captures the complete response. Note: mitmproxy buffers SSE streams, so for streaming API calls (`"stream": true`), the `response()` hook fires only after the full stream completes. This delays state updates but is acceptable for scoring purposes.
5. Addon detects provider from hostname and URL path, then parses the request/response pair into Inspect's `ChatMessage` list (input) and `ModelOutput` (output) - see "New Parsers" below
6. Parsed input messages + model output pushed over Unix socket to Inspect

### Provider Detection and Parsing

The addon detects the API provider from the request hostname and URL path:

- `generativelanguage.googleapis.com` + `*-aiplatform.googleapis.com` (suffix match for regional Vertex AI endpoints) - Google AI format
- `cloudcode-pa.googleapis.com` - Google OAuth format with `{model, project, request}` envelope; unwrap before parsing as Google AI format
- `api.anthropic.com` - Anthropic Messages API format
- `api.openai.com` / `chatgpt.com` + `/v1/chat/completions` - OpenAI Chat Completions format
- `api.openai.com` / `chatgpt.com` + `/v1/responses` - OpenAI Responses API format (used by Codex CLI)

### New Parsers (New Work)

The existing bridge code works in the opposite direction - it receives Inspect `ChatMessage` objects, calls the model, and transforms the response back to provider format. It does not have parsers that convert raw API JSON into Inspect types.

New parser functions are needed in `src/inspect_swe/_bridge/parsers.py`:

```python
def parse_google_traffic(request_json: dict, response_json: dict) -> tuple[list[ChatMessage], ModelOutput]:
    """Parse raw Google AI API request/response into Inspect types."""

def parse_anthropic_traffic(request_json: dict, response_json: dict) -> tuple[list[ChatMessage], ModelOutput]:
    """Parse raw Anthropic Messages API request/response into Inspect types."""

def parse_openai_completions_traffic(request_json: dict, response_json: dict) -> tuple[list[ChatMessage], ModelOutput]:
    """Parse raw OpenAI Chat Completions API request/response into Inspect types."""

def parse_openai_responses_traffic(request_json: dict, response_json: dict) -> tuple[list[ChatMessage], ModelOutput]:
    """Parse raw OpenAI Responses API request/response into Inspect types."""
```

Each parser must:
- Extract input messages from the request body and convert to `list[ChatMessage]` (text, tool_use, tool_result, images, etc.)
- Extract the model's response and convert to `ModelOutput` (completion message, stop reason, token usage)
- Handle tool call content blocks (function calls, tool results)
- Handle thinking/reasoning blocks where applicable
- Handle both streaming and non-streaming responses. When `"stream": true`, mitmproxy buffers the full SSE stream before calling `response()`. The buffered body will be raw SSE chunks (not a single JSON object), so parsers must reassemble SSE events into a complete response. When `"stream": false`, the body is standard JSON.

This is the largest piece of new implementation work. The existing `inspect_ai.model` conversion utilities (e.g., `ChatMessageAssistant`, `ToolCall`, `ModelUsage`) provide the target types. Reference implementations: `inspect_ai.model._openai_convert`, `inspect_ai.agent._bridge.anthropic_api_impl`, `inspect_ai.agent._bridge.google_api_impl`.

### State Tracking

The addon pushes `(input_messages, model_output)` pairs over the Unix socket. On the Inspect side, the listener processes each pair:

1. Call `apply_message_ids(bridge, input_messages)` to assign stable IDs via MM3 hashing (matching the existing bridge's ID scheme)
2. Call `_track_state(input_messages, model_output)` which replaces `state.messages` with `input + [output.message]` when the message count exceeds the previous generation's count (this avoids overwriting state with shorter side-chain calls)

`MitmproxyAgentBridge` inherits from `AgentBridge` to reuse `_track_state` and `apply_message_ids`. It passes `filter=None, retry_refusals=None, compaction=None` to the parent constructor since these features are not applicable.

## MCP Tool Bridging

A standalone asyncio HTTP server running on the host, extracted from the current bridge's MCP handling.

- Exposes `/mcp/{name}` routes with `list_tools` and `call_tool` handlers
- Accessible from container via `http://host.docker.internal:{tool_port}/mcp/{name}`
- CLI agents receive `MCPServerConfigHTTP` entries pointing to these URLs

The API surface is identical to today:
- `BridgedToolsSpec` for registering tools
- `MCPServerConfigHTTP` entries generated for CLIs
- Tool execution via Inspect's `Tool.__call__`

On Linux, Docker containers need `extra_hosts: ["host.docker.internal:host-gateway"]` to reach the host.

## CA Certificate Management

### Lifecycle

1. Check for existing CA at `~/.mitmproxy/mitmproxy-ca-cert.pem`
2. If missing, invoke mitmdump once to trigger auto-generation
3. CA is stable across runs - generated once, reused

### Container Injection

Write the CA cert into the sandbox at runtime via `sandbox.write_file("/tmp/mitmproxy-ca-cert.pem", cert_bytes)`. No Dockerfile changes needed.

Set env vars based on CLI runtime:

| Runtime | Env Var |
|---|---|
| Node.js (Gemini CLI, Claude Code) | `NODE_EXTRA_CA_CERTS=/tmp/mitmproxy-ca-cert.pem` |
| Node.js >= 22.15 / Bun | `NODE_USE_SYSTEM_CA=1` (belt-and-suspenders) |
| Rust (Codex CLI) | `CODEX_CA_CERTIFICATE=/tmp/mitmproxy-ca-cert.pem` or `SSL_CERT_FILE` |
| Python | `SSL_CERT_FILE=/tmp/mitmproxy-ca-cert.pem` |

### Proxy Env Vars

```
HTTPS_PROXY=http://host.docker.internal:{proxy_port}
HTTP_PROXY=http://host.docker.internal:{proxy_port}
NO_PROXY=host.docker.internal,localhost,127.0.0.1
```

`NO_PROXY` prevents MCP tool traffic (which goes to `host.docker.internal:{tool_port}`) from being routed through the proxy unnecessarily.

The proxy listens in plain HTTP. TLS termination happens at the proxy - the CLI connects to the proxy over HTTP via CONNECT, then the proxy negotiates TLS with the real API.

## IPC: Unix Domain Socket

### Why Unix Socket

- Shared file / polling: latency, fragile with partial writes
- HTTP callback: requires another server in Inspect
- Pipe (stdout): mitmdump uses stdout for its own output
- **Unix socket**: zero-copy local IPC, natural for message streams, no port allocation

### Protocol

Newline-delimited JSON over Unix socket, versioned for forward compatibility.

Initial handshake (addon sends on connect):
```json
{"type": "hello", "version": 1}
```

Per-generation message (addon sends after each intercepted API call):
```json
{
  "type": "generation",
  "provider": "google",
  "input_messages": [...],
  "model_output": {
    "message": {...},
    "stop_reason": "stop",
    "usage": {"input_tokens": 1234, "output_tokens": 567}
  }
}
```

The `input_messages` and `model_output` fields are JSON-serialized Inspect `ChatMessage` and `ModelOutput` types respectively, matching the `_track_state(input, output)` API.

### Lifecycle

1. Inspect creates socket at `/tmp/inspect_bridge_{uuid}.sock` (UUID avoids stale socket conflicts from crashed runs) and starts listening
2. Inspect launches `mitmdump -s addon.py` with env vars (`INSPECT_BRIDGE_SOCKET`, target hosts, proxy port)
3. Addon connects to socket on `configure()` hook
4. Traffic flows, messages pushed over socket in real-time
5. CLI finishes, Inspect sends SIGTERM to mitmdump, waits 5s, then SIGKILL
6. Socket cleaned up

## Public API

### New Context Manager

```python
@asynccontextmanager
async def mitmproxy_agent_bridge(
    state: AgentState,
    sandbox: str | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
) -> AsyncIterator[MitmproxyAgentBridge]:
    ...
```

Ports are bound internally to OS-assigned ephemeral ports (port 0). The actual bound ports are exposed on the returned bridge object via `.port` and `.tool_port`.

`MitmproxyAgentBridge` exposes:
- `.state` - `AgentState` updated in real-time from intercepted traffic
- `.port` - proxy port (for `HTTPS_PROXY`)
- `.tool_port` - MCP tool server port
- `.mcp_server_configs` - list of `MCPServerConfigHTTP` for registered tools

### Parameters Not Supported

The following `sandbox_agent_bridge` parameters are **not applicable** to the mitmproxy bridge and are intentionally excluded:

- **`filter` (GenerateFilter)**: Requires intercepting and modifying requests before they reach the API. The mitmproxy bridge is a passive observer - it could theoretically modify requests in the addon, but this conflicts with the goal of preserving native CLI behavior.
- **`retry_refusals`**: Requires Inspect to retry a model call on content_filter. The CLI controls its own retry logic; Inspect cannot re-issue a request the CLI made.
- **`model` / `model_aliases`**: The CLI talks directly to the API; Inspect does not route model calls.
- **`compaction`**: Compaction is applied by Inspect before calling `model.generate()`. Since the CLI manages its own context window, compaction is the CLI's responsibility.

### Agent-Level Opt-In

Each agent gets a new `bridge` parameter:

```python
@agent
def gemini_cli(
    ...
    bridge: Literal["default", "mitmproxy"] = "default",
    ...
) -> Agent:
```

When `bridge="mitmproxy"`:
- No base URL overrides
- No dummy API keys
- Sets `HTTPS_PROXY`, `NODE_EXTRA_CA_CERTS`, injects CA cert
- CLI uses native auth

When `bridge="default"`:
- Existing behavior, unchanged

No `--model` parameter needed for the mitmproxy bridge (the CLI talks to the API directly). Inspect still needs `--model` for scoring, but that's a separate concern.

## File Layout

```
src/inspect_swe/_bridge/
    __init__.py
    mitmproxy_bridge.py      # MitmproxyAgentBridge context manager
    mcp_tool_server.py       # Standalone MCP tool HTTP server
    ipc.py                   # Unix socket listener/client
    ca.py                    # CA cert management (find/generate/inject)
    addon.py                 # mitmdump addon script (run in subprocess)
    hosts.py                 # Target host registry + provider detection
    parsers.py               # Raw API JSON -> Inspect ChatMessage/ModelOutput
```

The bridge module is agent-agnostic. All three CLIs use the same proxy, IPC, and MCP server. Only env var names differ (handled in each agent's `execute()`).

## Dependencies

mitmproxy is an optional dependency:

```toml
[project.optional-dependencies]
mitmproxy = ["mitmproxy>=10.0"]
```

Runtime check at bridge creation - fail fast with clear error if not installed. Zero impact on `bridge="default"` users.

The addon script (`addon.py`) imports from `inspect_swe._bridge.ipc` and `inspect_ai.agent._bridge`. This works because mitmdump runs in the same Python environment as Inspect. **Important**: mitmproxy must be installed into the project's venv (via `pip install inspect_swe[mitmproxy]`), not globally. If a user has a system-level mitmproxy and runs `mitmdump -s addon.py`, the imports will fail. The bridge should resolve the `mitmdump` binary from `sys.prefix` to ensure the correct venv is used.

## Error Handling

### mitmdump Process

- Launched with timeout matching sandbox execution timeout
- Normal exit: SIGTERM, wait 5s, SIGKILL
- Unexpected exit: Inspect detects broken socket, raises `RuntimeError` with mitmdump's stderr

### Certificate Trust Failures

If the CLI rejects the CA cert, the TLS handshake fails. No `request()` hook fires. The CLI reports a TLS error that propagates as a sandbox exec failure. Actionable error: "TLS handshake failed - verify NODE_EXTRA_CA_CERTS is set correctly."

### Unknown API Hosts

Traffic to hosts not in the target registry passes through unmodified. State tracking misses those calls, but nothing breaks. `hosts.py` is the single place to add new targets.

### Concurrent Samples

Each sample gets its own mitmdump instance on a unique port with its own Unix socket. No shared state between samples. Ports are allocated using port 0 (OS-assigned ephemeral ports) to avoid binding conflicts at scale. The MCP tool server port is similarly OS-assigned. Both ports are communicated to the container via env vars after binding.

### cloudcode-pa Envelope

Gemini OAuth wraps requests in `{model, project, request}`. The addon detects this host and unwraps before parsing. If the envelope format changes, parsing fails gracefully - logs a warning, skips state update, CLI continues working since the proxy is transparent.
