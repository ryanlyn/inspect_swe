# mitmproxy Bridge E2E

## Prerequisites

- Install the optional test dependencies: `uv sync --extra mitmproxy`
- Docker must be running
- For live smoke runs, export the real provider credentials you want to use

## Deterministic Harness Suite

Run the local fake-upstream suite on demand:

```bash
uv run pytest tests/test_mitmproxy_e2e.py -q --runslow
```

What it covers:

- `claude_code`, `codex_cli`, `gemini_cli`
- `interactive_claude_code`, `interactive_codex_cli`, `interactive_gemini_cli`
- Docker sandbox execution with `bridge="mitmproxy"`
- Local upstream overrides, capture summaries, and bridged MCP tool use

## Live Provider Smoke

Run the real-upstream smoke suite on demand:

```bash
uv run pytest tests/test_mitmproxy_live_smoke.py -q --runslow --runapi
```

What it checks:

- no localhost model-proxy env in mitmproxy mode
- successful completion against real upstream traffic
- non-empty capture summary from the mitmproxy bridge

Artifacts are written to temporary directories by the test helpers and included in assertion failures.

## Manual Smoke Checklist

- Confirm the target provider credentials are present in the environment
- Run the harness suite first and fix any parser or proxy regressions before live smoke
- Run the provider-family smoke test you care about
- If a smoke test fails, inspect:
  - `eval_log.json`
  - `captures.tsv`
  - `mitmdump.stderr.log`
