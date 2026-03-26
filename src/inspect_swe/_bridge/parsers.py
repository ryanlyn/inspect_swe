"""Raw provider traffic parsers for mitmproxy bridge."""

from __future__ import annotations

import json
from typing import Any, Iterable, cast

from inspect_ai.agent._bridge.anthropic_api_impl import (
    content_and_tool_calls_from_assistant_content_blocks,
    messages_from_anthropic_input,
    tools_from_anthropic_tools,
)
from inspect_ai.agent._bridge.google_api_impl import messages_from_google_contents
from inspect_ai.agent._bridge.responses_impl import (
    messages_from_responses_input,
    tool_from_responses_tool,
)
from inspect_ai.agent._bridge.util import (
    default_code_execution_providers,
    internal_web_search_providers,
)
from inspect_ai.model._chat_message import ChatMessage, ChatMessageAssistant
from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput, ModelUsage
from inspect_ai.model._openai_convert import messages_from_openai
from inspect_ai.tool._tool_info import ToolInfo
from inspect_ai.tool._tool_util import tool_to_tool_info


def parse_google_traffic(
    request_body: str | dict[str, Any],
    response_body: str | dict[str, Any],
) -> tuple[list[ChatMessage], ModelOutput]:
    """Parse raw Google API traffic into Inspect types."""
    request_json = _coerce_json(request_body)
    if "request" in request_json and isinstance(request_json["request"], dict):
        request_json = request_json["request"]
    response_json = _coerce_google_response(response_body)

    messages = messages_from_google_contents(
        request_json.get("contents", []),
        request_json.get("systemInstruction", request_json.get("system_instruction")),
    )
    candidate = response_json.get("candidates", [{}])[0]
    response_messages = messages_from_google_contents(
        [candidate.get("content", {"role": "model", "parts": []})],
        None,
    )
    assistant = _find_assistant(response_messages)
    assistant.model = response_json.get(
        "modelVersion", request_json.get("model", "") or ""
    )
    return messages, _model_output(
        assistant=assistant,
        model=assistant.model or "",
        stop_reason=_map_google_stop_reason(candidate.get("finishReason")),
        usage=ModelUsage(
            input_tokens=int(response_json.get("usageMetadata", {}).get("promptTokenCount", 0)),
            output_tokens=int(response_json.get("usageMetadata", {}).get("candidatesTokenCount", 0)),
            total_tokens=int(response_json.get("usageMetadata", {}).get("totalTokenCount", 0)),
        ),
    )


def parse_anthropic_traffic(
    request_body: str | dict[str, Any],
    response_body: str | dict[str, Any],
) -> tuple[list[ChatMessage], ModelOutput]:
    """Parse raw Anthropic Messages API traffic into Inspect types."""
    request_json = _coerce_json(request_body)
    request_json["messages"] = _normalize_anthropic_request_messages(
        cast(list[dict[str, Any]], request_json.get("messages", []))
    )
    response_json = _coerce_anthropic_response(response_body)

    tools = tools_from_anthropic_tools(
        request_json.get("tools"),
        request_json.get("mcp_servers"),
        internal_web_search_providers(),
        default_code_execution_providers(),
    )
    messages = _run_async(messages_from_anthropic_input(request_json["messages"], tools))
    tools_info = [
        tool if isinstance(tool, ToolInfo) else tool_to_tool_info(tool) for tool in tools
    ]
    content, tool_calls = content_and_tool_calls_from_assistant_content_blocks(
        response_json.get("content", []),
        tools_info,
    )
    assistant = ChatMessageAssistant(
        content=content,
        tool_calls=tool_calls,
        model=str(response_json.get("model", request_json.get("model", ""))),
        source="generate",
    )
    usage_json = cast(dict[str, Any], response_json.get("usage", {}))
    return messages, _model_output(
        assistant=assistant,
        model=assistant.model or "",
        stop_reason=_map_anthropic_stop_reason(response_json.get("stop_reason")),
        usage=ModelUsage(
            input_tokens=int(usage_json.get("input_tokens", 0)),
            output_tokens=int(usage_json.get("output_tokens", 0)),
            input_tokens_cache_write=_optional_int(
                usage_json.get("cache_creation_input_tokens")
            ),
            input_tokens_cache_read=_optional_int(
                usage_json.get("cache_read_input_tokens")
            ),
            total_tokens=int(usage_json.get("input_tokens", 0))
            + int(usage_json.get("output_tokens", 0)),
        ),
    )


def parse_openai_completions_traffic(
    request_body: str | dict[str, Any],
    response_body: str | dict[str, Any],
) -> tuple[list[ChatMessage], ModelOutput]:
    """Parse raw OpenAI Chat Completions traffic into Inspect types."""
    request_json = _coerce_json(request_body)
    response_json = _coerce_openai_completions_response(response_body)

    messages = _run_async(
        messages_from_openai(request_json["messages"], request_json.get("model"))
    )
    choice = response_json["choices"][0]
    assistant = _find_assistant(
        _run_async(messages_from_openai([choice["message"]], request_json.get("model")))
    )
    assistant.model = str(response_json.get("model", request_json.get("model", "")))
    usage_json = cast(dict[str, Any], response_json.get("usage", {}))
    return messages, _model_output(
        assistant=assistant,
        model=assistant.model or "",
        stop_reason=_map_openai_stop_reason(choice.get("finish_reason")),
        usage=ModelUsage(
            input_tokens=int(usage_json.get("prompt_tokens", 0)),
            output_tokens=int(usage_json.get("completion_tokens", 0)),
            total_tokens=int(usage_json.get("total_tokens", 0)),
            reasoning_tokens=_extract_reasoning_tokens(usage_json),
        ),
    )


def parse_openai_responses_traffic(
    request_body: str | dict[str, Any],
    response_body: str | dict[str, Any],
) -> tuple[list[ChatMessage], ModelOutput]:
    """Parse raw OpenAI Responses traffic into Inspect types."""
    if isinstance(request_body, str) and isinstance(response_body, str):
        request_lines = _parse_json_lines(request_body)
        response_lines = _parse_json_lines(response_body)
        if len(request_lines) > 1 and len(response_lines) > 1:
            websocket_result = _parse_openai_responses_websocket_conversation(
                request_lines,
                response_lines,
            )
            if websocket_result is not None:
                return websocket_result

    request_json = _coerce_openai_responses_request(request_body)
    response_json = _coerce_openai_responses_response(response_body)
    tools = [
        tool_from_responses_tool(
            tool,
            internal_web_search_providers(),
            default_code_execution_providers(),
        )
        for tool in request_json.get("tools", [])
    ]
    messages = messages_from_responses_input(
        request_json.get("input", []),
        tools,
        request_json.get("model"),
    )
    assistant = _find_assistant(
        messages_from_responses_input(
            response_json.get("output", []),
            tools,
            response_json.get("model", request_json.get("model")),
        )
    )
    assistant.model = str(response_json.get("model", request_json.get("model", "")))
    usage_json = cast(dict[str, Any], response_json.get("usage", {}))
    stop_reason = "tool_calls" if assistant.tool_calls else "stop"
    incomplete = cast(dict[str, Any] | None, response_json.get("incomplete_details"))
    if incomplete and incomplete.get("reason") in {"max_output_tokens", "max_tokens"}:
        stop_reason = "max_tokens"
    return messages, _model_output(
        assistant=assistant,
        model=assistant.model or "",
        stop_reason=cast(Any, stop_reason),
        usage=ModelUsage(
            input_tokens=int(usage_json.get("input_tokens", 0)),
            output_tokens=int(usage_json.get("output_tokens", 0)),
            total_tokens=int(usage_json.get("total_tokens", 0)),
            input_tokens_cache_read=_optional_int(
                usage_json.get("input_tokens_details", {}).get("cached_tokens")
                if isinstance(usage_json.get("input_tokens_details"), dict)
                else None
            ),
            reasoning_tokens=_optional_int(
                usage_json.get("output_tokens_details", {}).get("reasoning_tokens")
                if isinstance(usage_json.get("output_tokens_details"), dict)
                else None
            ),
        ),
    )


def _coerce_json(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    return cast(dict[str, Any], json.loads(payload))


def _coerce_openai_responses_response(
    payload: str | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if payload.lstrip().startswith("{"):
        json_lines = _parse_json_lines(payload)
        if len(json_lines) == 1:
            return cast(dict[str, Any], json_lines[0])
        for event in reversed(json_lines):
            if isinstance(event, dict) and event.get("type") == "response.completed":
                return cast(dict[str, Any], event["response"])
        raise ValueError("Unable to reconstruct OpenAI Responses websocket payload")

    for event in reversed(_parse_sse_events(payload)):
        data = event["data"]
        if isinstance(data, dict) and data.get("type") == "response.completed":
            return cast(dict[str, Any], data["response"])
    raise ValueError("Unable to reconstruct OpenAI Responses stream payload")


def _coerce_openai_responses_request(
    payload: str | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    json_lines = _parse_json_lines(payload)
    if len(json_lines) == 1:
        return cast(dict[str, Any], json_lines[0])
    for event in reversed(json_lines):
        if isinstance(event, dict) and event.get("type") == "response.create":
            return cast(dict[str, Any], event)
    raise ValueError("Unable to reconstruct OpenAI Responses request payload")


def _parse_openai_responses_websocket_conversation(
    request_events: list[dict[str, Any]],
    response_events: list[dict[str, Any]],
) -> tuple[list[ChatMessage], ModelOutput] | None:
    request_creates = [
        event for event in request_events if event.get("type") == "response.create"
    ]
    completed = [
        cast(dict[str, Any], event["response"])
        for event in response_events
        if event.get("type") == "response.completed" and isinstance(event.get("response"), dict)
    ]
    if not request_creates or not completed:
        return None

    completed_by_id = {
        str(response.get("id")): response
        for response in completed
        if response.get("id") is not None
    }
    last_response = completed[-1]

    tools_source = next(
        (
            cast(list[dict[str, Any]], event.get("tools", []))
            for event in reversed(request_creates)
            if isinstance(event.get("tools"), list)
        ),
        [],
    )
    model_name = str(
        next(
            (
                event.get("model")
                for event in reversed(request_creates)
                if event.get("model") is not None
            ),
            last_response.get("model", ""),
        )
    )
    tools = [
        tool_from_responses_tool(
            tool,
            internal_web_search_providers(),
            default_code_execution_providers(),
        )
        for tool in tools_source
    ]

    input_messages: list[ChatMessage] = []
    for index, request_json in enumerate(request_creates[:-1]):
        request_input = request_json.get("input", [])
        if request_input:
            input_messages.extend(
                messages_from_responses_input(
                    request_input,
                    tools,
                    request_json.get("model"),
                )
            )
        next_request = request_creates[index + 1]
        previous_response_id = next_request.get("previous_response_id")
        if previous_response_id is None:
            continue
        previous_response = completed_by_id.get(str(previous_response_id))
        if previous_response is None:
            continue
        output_items = previous_response.get("output", [])
        if output_items:
            input_messages.extend(
                messages_from_responses_input(
                    output_items,
                    tools,
                    previous_response.get("model", request_json.get("model")),
                )
            )

    last_request = request_creates[-1]
    last_input = last_request.get("input", [])
    if last_input:
        input_messages.extend(
            messages_from_responses_input(
                last_input,
                tools,
                last_request.get("model"),
            )
        )

    assistant = _find_assistant(
        messages_from_responses_input(
            last_response.get("output", []),
            tools,
            last_response.get("model", model_name),
        )
    )
    assistant.model = str(last_response.get("model", model_name))
    usage_json = cast(dict[str, Any], last_response.get("usage", {}))
    stop_reason = "tool_calls" if assistant.tool_calls else "stop"
    incomplete = cast(dict[str, Any] | None, last_response.get("incomplete_details"))
    if incomplete and incomplete.get("reason") in {"max_output_tokens", "max_tokens"}:
        stop_reason = "max_tokens"

    return input_messages, _model_output(
        assistant=assistant,
        model=assistant.model or "",
        stop_reason=cast(Any, stop_reason),
        usage=ModelUsage(
            input_tokens=int(usage_json.get("input_tokens", 0)),
            output_tokens=int(usage_json.get("output_tokens", 0)),
            total_tokens=int(usage_json.get("total_tokens", 0)),
            input_tokens_cache_read=_optional_int(
                usage_json.get("input_tokens_details", {}).get("cached_tokens")
                if isinstance(usage_json.get("input_tokens_details"), dict)
                else None
            ),
            reasoning_tokens=_optional_int(
                usage_json.get("output_tokens_details", {}).get("reasoning_tokens")
                if isinstance(usage_json.get("output_tokens_details"), dict)
                else None
            ),
        ),
    )


def _coerce_openai_completions_response(
    payload: str | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if payload.lstrip().startswith("{"):
        return _coerce_json(payload)

    model = ""
    usage: dict[str, Any] = {}
    message: dict[str, Any] = {"role": "assistant", "content": ""}
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason = None
    for event in _parse_sse_events(payload):
        data = event["data"]
        if not isinstance(data, dict):
            continue
        if "model" in data:
            model = str(data["model"])
        if "usage" in data and isinstance(data["usage"], dict):
            usage = cast(dict[str, Any], data["usage"])
        for choice in cast(list[dict[str, Any]], data.get("choices", [])):
            delta = cast(dict[str, Any], choice.get("delta", {}))
            if delta.get("role"):
                message["role"] = delta["role"]
            if "content" in delta and delta["content"] is not None:
                message["content"] = f"{message.get('content', '')}{delta['content']}"
            for tool_call in cast(list[dict[str, Any]], delta.get("tool_calls", [])):
                index = int(tool_call.get("index", 0))
                current = tool_calls.setdefault(
                    index,
                    {
                        "id": tool_call.get("id"),
                        "type": tool_call.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if tool_call.get("id"):
                    current["id"] = tool_call["id"]
                fn = cast(dict[str, Any], tool_call.get("function", {}))
                current_fn = cast(dict[str, Any], current["function"])
                if fn.get("name"):
                    current_fn["name"] = fn["name"]
                if fn.get("arguments"):
                    current_fn["arguments"] = (
                        f"{current_fn.get('arguments', '')}{fn['arguments']}"
                    )
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    if tool_calls:
        message["tool_calls"] = [
            tool_calls[index] for index in sorted(tool_calls.keys())
        ]
    return {
        "model": model,
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


def _parse_json_lines(payload: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed.append(cast(dict[str, Any], json.loads(stripped)))
    return parsed


def _coerce_anthropic_response(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if payload.lstrip().startswith("{"):
        return _coerce_json(payload)

    message: dict[str, Any] | None = None
    blocks: dict[int, dict[str, Any]] = {}
    for event in _parse_sse_events(payload):
        name = event["event"]
        data = event["data"]
        if not isinstance(data, dict):
            continue
        if name == "message_start":
            message = cast(dict[str, Any], data.get("message", {}))
            message.setdefault("content", [])
        elif name == "content_block_start":
            blocks[int(data["index"])] = cast(dict[str, Any], data["content_block"]).copy()
        elif name == "content_block_delta":
            block = blocks[int(data["index"])]
            delta = cast(dict[str, Any], data.get("delta", {}))
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                block["text"] = f"{block.get('text', '')}{delta.get('text', '')}"
            elif delta_type == "thinking_delta":
                block["thinking"] = f"{block.get('thinking', '')}{delta.get('thinking', '')}"
            elif delta_type == "signature_delta":
                block["signature"] = f"{block.get('signature', '')}{delta.get('signature', '')}"
            elif delta_type == "input_json_delta":
                current = block.get("_partial_json", "")
                block["_partial_json"] = f"{current}{delta.get('partial_json', '')}"
        elif name == "content_block_stop":
            block = blocks[int(data["index"])]
            if block.get("type") == "tool_use":
                partial = block.pop("_partial_json", "{}")
                try:
                    block["input"] = json.loads(partial)
                except json.JSONDecodeError:
                    block["input"] = {}
        elif name == "message_delta":
            if message is None:
                continue
            message["stop_reason"] = data.get("delta", {}).get(
                "stop_reason",
                message.get("stop_reason"),
            )
            if isinstance(data.get("usage"), dict):
                message["usage"] = data["usage"]
        elif name == "message_stop" and message is not None:
            message["content"] = [blocks[i] for i in sorted(blocks.keys())]
            return message
    if message is None:
        raise ValueError("Unable to reconstruct Anthropic stream payload")
    message["content"] = [blocks[i] for i in sorted(blocks.keys())]
    return message


def _normalize_anthropic_request_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(message)
            continue
        normalized_content: list[Any] = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and isinstance(block.get("content"), list)
            ):
                tool_content: list[Any] = []
                for item in cast(list[Any], block["content"]):
                    if isinstance(item, dict) and item.get("type") == "tool_reference":
                        tool_name = str(item.get("tool_name", ""))
                        tool_content.append({"type": "text", "text": tool_name})
                    else:
                        tool_content.append(item)
                normalized_block = dict(block)
                normalized_block["content"] = tool_content
                normalized_content.append(normalized_block)
            else:
                normalized_content.append(block)
        normalized.append({**message, "content": normalized_content})
    return normalized


def _coerce_google_response(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if payload.lstrip().startswith("{"):
        return _coerce_json(payload)

    events = _parse_sse_events(payload)
    if not events:
        raise ValueError("Unable to reconstruct Google stream payload")
    merged: dict[str, Any] = {"candidates": [], "usageMetadata": {}}
    parts: list[dict[str, Any]] = []
    finish_reason = None
    model_version = ""
    usage_metadata: dict[str, Any] = {}
    for event in events:
        data = event["data"]
        if not isinstance(data, dict):
            continue
        if "modelVersion" in data:
            model_version = str(data["modelVersion"])
        if isinstance(data.get("usageMetadata"), dict):
            usage_metadata = cast(dict[str, Any], data["usageMetadata"])
        candidate = cast(list[dict[str, Any]], data.get("candidates", []))
        if candidate:
            current = candidate[0]
            finish_reason = current.get("finishReason", finish_reason)
            content = cast(dict[str, Any], current.get("content", {}))
            parts.extend(cast(list[dict[str, Any]], content.get("parts", [])))
    merged["candidates"] = [
        {
            "content": {"role": "model", "parts": parts},
            "finishReason": finish_reason or "STOP",
        }
    ]
    merged["usageMetadata"] = usage_metadata
    if model_version:
        merged["modelVersion"] = model_version
    return merged


def _parse_sse_events(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current_event = "message"
    data_lines: list[str] = []
    for line in body.splitlines():
        if not line:
            if data_lines:
                data = "\n".join(data_lines)
                if data != "[DONE]":
                    try:
                        payload: Any = json.loads(data)
                    except json.JSONDecodeError:
                        payload = data
                    events.append({"event": current_event, "data": payload})
                data_lines = []
                current_event = "message"
            continue
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
    if data_lines:
        data = "\n".join(data_lines)
        if data != "[DONE]":
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                payload = data
            events.append({"event": current_event, "data": payload})
    return events


def _find_assistant(messages: Iterable[ChatMessage]) -> ChatMessageAssistant:
    assistants = [message for message in messages if isinstance(message, ChatMessageAssistant)]
    if not assistants:
        raise ValueError("Expected assistant message in parsed traffic")
    return assistants[-1]


def _model_output(
    assistant: ChatMessageAssistant,
    model: str,
    stop_reason: str,
    usage: ModelUsage,
) -> ModelOutput:
    assistant.source = "generate"
    return ModelOutput(
        model=model,
        choices=[
            ChatCompletionChoice(
                message=assistant,
                stop_reason=cast(Any, stop_reason),
            )
        ],
        usage=usage,
    )


def _map_openai_stop_reason(reason: Any) -> str:
    mapping = {
        "stop": "stop",
        "length": "max_tokens",
        "tool_calls": "tool_calls",
        "content_filter": "content_filter",
    }
    return mapping.get(str(reason), "unknown")


def _map_anthropic_stop_reason(reason: Any) -> str:
    mapping = {
        "end_turn": "stop",
        "max_tokens": "max_tokens",
        "tool_use": "tool_calls",
        "refusal": "content_filter",
    }
    return mapping.get(str(reason), "unknown")


def _map_google_stop_reason(reason: Any) -> str:
    mapping = {
        "STOP": "stop",
        "MAX_TOKENS": "max_tokens",
        "SAFETY": "content_filter",
    }
    return mapping.get(str(reason), "unknown")


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _extract_reasoning_tokens(usage: dict[str, Any]) -> int | None:
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        value = details.get("reasoning_tokens")
        if value is not None:
            return int(value)
    return None


def _run_async(awaitable: Any) -> Any:
    import asyncio
    import threading

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable)
        except BaseException as ex:  # pragma: no cover - defensive threading path
            error["value"] = ex

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "value" in error:
        raise error["value"]
    return result.get("value")
