import shlex
from logging import getLogger
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal, Sequence

from inspect_ai.agent import (
    Agent,
    AgentAttempts,
    AgentState,
    BridgedToolsSpec,
    agent,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.model import ChatMessageSystem, GenerateFilter, Model
from inspect_ai.scorer import score
from inspect_ai.tool import MCPServerConfig, Skill, install_skills, read_skills
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.util import SandboxEnvironment, store
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from inspect_swe._bridge.ca import (
    discover_sandbox_host,
    proxy_env,
    rewrite_http_url_host,
    write_ca_cert_to_sandbox,
)
from inspect_swe._bridge.mitmproxy_bridge import mitmproxy_agent_bridge
from inspect_swe._bridge.oauth import (
    copy_optional_host_file_to_sandbox,
    copy_optional_host_tree_to_sandbox,
    ensure_sandbox_runtime_user,
    secure_path_for_user,
)
from inspect_swe._util._async import is_callable_coroutine
from inspect_swe._util.centaur import CentaurOptions, run_centaur
from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.path import join_path
from inspect_swe._util.sandbox import sandbox_exec
from inspect_swe._util.toml import to_toml
from inspect_swe._util.trace import trace

from .._util.agentbinary import ensure_agent_binary_installed
from .agentbinary import codex_cli_binary_source

logger = getLogger(__file__)


@agent
def codex_cli(
    name: str = "Codex CLI",
    description: str = dedent("""
       Autonomous coding agent capable of writing, testing, debugging,
       and iterating on code across multiple languages.
    """),
    system_prompt: str | None = None,
    model_config: str = "gpt-5.1",
    skills: Sequence[str | Path | Skill] | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    disallowed_tools: list[Literal["web_search"]] | None = None,
    centaur: bool | CentaurOptions = False,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    home_dir: str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "latest"] | str = "auto",
    config_overrides: dict[str, str] | None = None,
    bridge: Literal["default", "mitmproxy"] = "default",
) -> Agent:
    """Codex CLI.

    Agent that uses OpenAI [Codex CLI](https://github.com/openai/codex) running in a sandbox.

    Use the `attempts` option to enable additional submissions if the initial
    submission(s) are incorrect (by default, no additional attempts are permitted).

    Args:
        name: Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)
        description: Agent description (used in multi-agent systems with `as_tool()` and `handoff()`)
        system_prompt: Additional system prompt to append to default system prompt.
        model_config: Model configuration profile (e.g. used to determine the system prompt).
        skills: Additional [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to make available to the agent.
        mcp_servers: MCP servers to make available to the agent.
        bridged_tools: Host-side Inspect tools to expose to the agent via MCP.
            Each BridgedToolsSpec creates an MCP server that makes the specified
            tools available to the agent running in the sandbox.
        disallowed_tools: Optionally disallow tools (currently only web_search).
        centaur: Run in 'centaur' mode, which makes Codex CLI available to an Inspect `human_cli()` agent rather than running it unattended.
        attempts: Configure agent to make multiple attempts. When this is specified, the task will be scored when the agent stops calling tools. If the scoring is successful, execution will stop. Otherwise, the agent will be prompted to pick up where it left off for another attempt.
        model: Model name to use (defaults to main model for task).
        model_aliases: Optional mapping of model names to Model instances or model name strings.
            Allows using custom Model implementations (e.g., wrapped Agents) instead of standard models.
            When a model name in the mapping is referenced, the corresponding Model/string is used.
        filter: Filter for intercepting bridged model requests.
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        home_dir: Home directory to use for codex cli. If set, AGENTS.md, skills, and the MCP configuration will be written here.
        cwd: Working directory to run codex cli within.
        env: Environment variables to set for codex cli
        user: User to execute codex cli with.
        sandbox: Optional sandbox environment name.
        version: Version of codex cli to use. One of:
            - "auto": Use any available version of codex cli in the sandbox, otherwise download the latest version.
            - "sandbox": Use the version of codex cli in the sandbox (raises `RuntimeError` if codex is not available in the sandbox)
            - "latest": Download and use the very latest version of codex cli.
            - "x.x.x": Download and use a specific version of codex cli.
        config_overrides: Additional Codex CLI configuration overrides.
            Each key-value pair is passed as `-c key=value` to the CLI.
        bridge: Bridge implementation to use. `"default"` preserves the existing localhost model proxy and `"mitmproxy"` enables transparent HTTPS interception.
    """
    # resolve centaur
    if centaur is True:
        centaur = CentaurOptions()

    # resolve model
    model = f"inspect/{model}" if model is not None else "inspect"

    # resolve skills
    resolved_skills = read_skills(skills) if skills is not None else None

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    # ensure disallowed_tools list
    disallowed_tools = disallowed_tools or []

    async def execute(state: AgentState) -> AgentState:
        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "codex_cli_model_port"
        port = store().get(MODEL_PORT, 3000) + 1
        store().set(MODEL_PORT, port)

        bridge_cm = (
            sandbox_agent_bridge(
                state,
                model=model,
                model_aliases=model_aliases,
                filter=filter,
                sandbox=sandbox,
                retry_refusals=retry_refusals,
                port=port,
                bridged_tools=bridged_tools,
            )
            if bridge == "default"
            else mitmproxy_agent_bridge(
                state,
                sandbox=sandbox,
                bridged_tools=bridged_tools,
            )
        )
        async with bridge_cm as bridge_handle:
            # ensure codex is installed and get binary location
            codex_binary = await ensure_agent_binary_installed(
                codex_cli_binary_source(), version, user, sandbox_env(sandbox)
            )

            # build system prompt
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)

            # resolve sandbox
            sbox = sandbox_env(sandbox)
            runtime_user = user
            runtime_home: str | None = None
            runtime_cwd = cwd
            if bridge == "mitmproxy":
                runtime_user, runtime_home = await ensure_sandbox_runtime_user(
                    sbox,
                    user,
                    cwd=cwd,
                )
                if runtime_cwd in [None, "/root"]:
                    runtime_cwd = runtime_home
            sandbox_ca_path: str | None = None
            if bridge == "mitmproxy":
                sandbox_ca_path = await write_ca_cert_to_sandbox(sbox)

            # determine CODEX_HOME (default to whatever sandbox working dir is)
            if home_dir is None:
                if bridge == "mitmproxy":
                    assert runtime_home is not None
                    codex_home = join_path(runtime_home, ".codex")
                else:
                    working_dir = await sandbox_exec(sbox, "pwd", user=user, cwd=cwd)
                    codex_home = join_path(working_dir, ".codex")
            else:
                # Resolve ~ and $VARS inside the sandbox
                codex_home = await sandbox_exec(
                    sbox, f'eval echo "{home_dir}"', user=runtime_user, cwd=cwd
                )
            await sandbox_exec(sbox, cmd=f"mkdir -p {codex_home}", user=runtime_user)
            copied_codex_auth = False
            if bridge == "mitmproxy":
                await copy_optional_host_tree_to_sandbox(
                    sbox,
                    host_env_var="INSPECT_SWE_CODEX_HOME_DIR",
                    sandbox_root=codex_home,
                    cwd=cwd,
                )
                copied_codex_auth = await copy_optional_host_file_to_sandbox(
                    sbox,
                    host_env_var="INSPECT_SWE_CODEX_AUTH_FILE",
                    sandbox_path=join_path(codex_home, "auth.json"),
                    cwd=cwd,
                )
                await copy_optional_host_file_to_sandbox(
                    sbox,
                    host_env_var="INSPECT_SWE_CODEX_CONFIG_FILE",
                    sandbox_path=join_path(codex_home, "config.json"),
                    cwd=cwd,
                )
                await secure_path_for_user(
                    sbox,
                    codex_home,
                    user=runtime_user or "root",
                    recursive=True,
                    cwd=cwd,
                )

            # location for agents_md
            def codex_agents_md() -> str:
                AGENTS_MD = "AGENTS.md"
                if home_dir is not None:
                    return join_path(codex_home, AGENTS_MD)
                elif cwd is not None:
                    return join_path(cwd, AGENTS_MD)
                else:
                    return AGENTS_MD

            # location for config_toml (either codex_home or cwd/.codex )
            async def codex_config_toml() -> str:
                CONFIG_TOML = "config.toml"
                if home_dir is not None or bridge == "mitmproxy":
                    return join_path(codex_home, CONFIG_TOML)
                else:
                    dir = ".codex" if cwd is None else join_path(cwd, ".codex")
                    await sandbox_exec(sbox, cmd=f"mkdir -p {dir}", user=runtime_user)
                    return join_path(dir, CONFIG_TOML)

            # write system messages to AGENTS.md
            if system_messages:
                await sbox.write_file(codex_agents_md(), "\n\n".join(system_messages))

            # install skills
            if resolved_skills is not None:
                await install_skills(
                    resolved_skills,
                    sbox,
                    runtime_user,
                    join_path(codex_home, "skills"),
                )

            prompt, has_assistant_response = build_user_prompt(state.messages)

            # build agent cmd
            cmd = [codex_binary]

            # headless
            if centaur is False:
                cmd.extend(["exec", "--color", "never", "--skip-git-repo-check"])

            # default cli args
            cmd.extend(
                [
                    # real model is passed to the bridge above, this just affects defaults e.g. system prompt
                    "--model",
                    model_config,
                    "--dangerously-bypass-approvals-and-sandbox",
                ]
            )

            # include web search if appropriate
            if "web_search" not in disallowed_tools:
                cmd.extend(["--enable", "web_search_request"])

            # apply config overrides
            if config_overrides:
                for key, value in config_overrides.items():
                    cmd.extend(["-c", f"{key}={value}"])

            # build toml config
            toml_config: dict[str, Any] = {}

            # register mcp servers (combine static configs with bridged tools)
            all_mcp_servers = list(mcp_servers or []) + bridge_handle.mcp_server_configs
            if bridge == "mitmproxy":
                proxy_host = await discover_sandbox_host(
                    sbox,
                    user=runtime_user,
                    cwd=runtime_cwd,
                )
                all_mcp_servers = [
                    mcp_server.model_copy(
                        update={"url": rewrite_http_url_host(mcp_server.url, proxy_host)}
                    )
                    if isinstance(mcp_server, MCPServerConfigHTTP)
                    else mcp_server
                    for mcp_server in all_mcp_servers
                ]
            if all_mcp_servers:
                for mcp_server in all_mcp_servers:
                    toml_config[f"mcp_servers.{mcp_server.name}"] = (
                        mcp_server.model_dump(
                            exclude={"name", "tools"}, exclude_none=True
                        )
                    )

            # model provider if we are in centaur mode
            if centaur and bridge == "default":
                toml_config["preferred_auth_method"] = "apikey"
                toml_config["model_provider"] = "openai-proxy"
                toml_config["model_providers.openai-proxy"] = {
                    "name": "OpenAI Proxy",
                    "base_url": f"http://localhost:{bridge_handle.port}/v1",
                    "env_key": "OPENAI_API_KEY",
                    "wire_api": "responses",
                }
            elif bridge == "mitmproxy" and copied_codex_auth:
                toml_config["preferred_auth_method"] = "chatgpt"
                toml_config["cli_auth_credentials_store"] = "file"

            # write toml config if we have it
            if len(toml_config) > 0:
                await sbox.write_file(await codex_config_toml(), to_toml(toml_config))

            # setup agent env
            agent_env = {
                "CODEX_HOME": codex_home,
                "RUST_LOG": "warning",
            }
            if bridge == "default":
                agent_env |= {
                    "OPENAI_API_KEY": "api-key",
                    "OPENAI_BASE_URL": f"http://localhost:{bridge_handle.port}/v1",
                }
            else:
                assert sandbox_ca_path is not None
                agent_env |= {
                    **proxy_env(bridge_handle.port, proxy_host),
                    "CODEX_CA_CERTIFICATE": sandbox_ca_path,
                    "SSL_CERT_FILE": sandbox_ca_path,
                }
            agent_env |= env or {}

            if centaur:
                await _run_codex_cli_centaur(
                    options=centaur,
                    codex_cmd=cmd,
                    agent_env=agent_env,
                    state=state,
                )
            else:
                # execute the agent (track debug output)
                debug_output: list[str] = []
                agent_prompt = prompt
                attempt_count = 0
                while True:
                    # append prompt
                    agent_cmd = cmd.copy()
                    agent_cmd.append(agent_prompt)

                    # resume previous conversation
                    if has_assistant_response or attempt_count > 0:
                        agent_cmd.extend(["resume", "--last"])

                    remote_cmd = (
                        ["bash", "-lc", 'exec 0</dev/null; exec "$@"', "bash"]
                        + agent_cmd
                    )
                    remote_user = runtime_user

                    # run agent
                    if bridge == "mitmproxy":
                        result = await sbox.exec(
                            cmd=remote_cmd,
                            cwd=runtime_cwd,
                            env=agent_env,
                            user=remote_user,
                            timeout=120,
                        )
                    else:
                        result = await sbox.exec_remote(
                            cmd=remote_cmd,
                            options=ExecRemoteAwaitableOptions(
                                cwd=runtime_cwd,
                                env=agent_env,
                                user=remote_user,
                                concurrency=False,
                            ),
                            stream=False,
                        )

                    # record output for debug
                    debug_output.append(result.stdout)
                    debug_output.append(result.stderr)

                    # raise for error
                    if not result.success:
                        if (
                            bridge == "mitmproxy"
                            and bridge_handle.state.output is not None
                        ):
                            break

                        raise RuntimeError(
                            f"Error executing codex cli agent {result.returncode}: {result.stdout}\n{result.stderr}"
                        )

                    # exit if we are at max_attempts
                    attempt_count += 1
                    if attempt_count >= attempts.attempts:
                        break

                    # score this attempt
                    answer_scores = await score(state)

                    # break if we score 'correct'
                    if attempts.score_value(answer_scores[0].value) == 1.0:
                        break

                    # otherwise update prompt with incorrect message and continue
                    else:
                        if callable(attempts.incorrect_message):
                            if not is_callable_coroutine(attempts.incorrect_message):
                                raise ValueError(
                                    "The incorrect_message function must be async."
                                )
                            agent_prompt = await attempts.incorrect_message(
                                state, answer_scores
                            )
                        else:
                            agent_prompt = attempts.incorrect_message

                # trace debug info
                debug_output.insert(0, "Codex CLI Debug Output:")
                trace("\n".join(debug_output))

        # return success
        return bridge_handle.state

    return agent_with(execute, name=name, description=description)


async def _run_codex_cli_centaur(
    options: CentaurOptions,
    codex_cmd: list[str],
    agent_env: dict[str, str],
    state: AgentState,
) -> None:
    instructions = "Codex CLI:\n\n - You may also use Codex CLI via the 'codex' command.\n - Use 'codex resume' if you need to resume a previous codex session."

    # build .bashrc content
    agent_env_vars = [f'export {k}="{v}"' for k, v in agent_env.items()]
    alias_cmd = shlex.join(codex_cmd)
    alias_cmd = "alias codex='" + alias_cmd.replace("'", "'\\''") + "'"
    bashrc = "\n".join(agent_env_vars + ["", alias_cmd])

    # run the human cli
    await run_centaur(options, instructions, bashrc, state)


async def _last_rollout(
    sandbox: SandboxEnvironment, codex_home: str, user: str | None
) -> str | None:
    try:
        rollout = await sandbox_exec(
            sandbox,
            f"find '{codex_home}/sessions' -type f -name 'rollout-*.jsonl' -exec ls -t -- {{}} + | head -n 1",
            user=user,
        )
        return rollout
    except RuntimeError as ex:
        logger.warning(f"Error attempting to read rollout file: {ex}")
        return None
