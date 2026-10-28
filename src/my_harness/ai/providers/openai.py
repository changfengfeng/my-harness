"""OpenAI Chat Completions provider. Also works with OpenAI-compatible endpoints via `base_url`."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from ..models import Model
from ..types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    DoneEvent,
    ImageContent,
    StartEvent,
    StopReason,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolCallEndEvent,
    ToolResultMessage,
    UserMessage,
)
from .base import parse_tool_arguments

STOP_REASONS: dict[str, StopReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "error",
}


def _user_parts(content: str | list[TextContent | ImageContent]) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for c in content:
        if isinstance(c, TextContent):
            parts.append({"type": "text", "text": c.text})
        else:
            parts.append({"type": "image_url", "image_url": {"url": f"data:{c.mime_type};base64,{c.data}"}})
    return parts


def convert_messages(context: Context) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if context.system_prompt:
        out.append({"role": "system", "content": context.system_prompt})
    pending_images: list[dict[str, Any]] = []
    for msg in context.messages:
        if not isinstance(msg, ToolResultMessage) and pending_images:
            out.append({"role": "user", "content": pending_images})
            pending_images = []
        if isinstance(msg, UserMessage):
            out.append({"role": "user", "content": _user_parts(msg.content)})
        elif isinstance(msg, AssistantMessage):
            if msg.stop_reason in ("error", "aborted") and not msg.tool_calls:
                continue
            entry: dict[str, Any] = {"role": "assistant", "content": msg.text or None}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in msg.tool_calls
                ]
            out.append(entry)
        elif isinstance(msg, ToolResultMessage):
            text = "\n".join(c.text for c in msg.content if isinstance(c, TextContent))
            images = [c for c in msg.content if isinstance(c, ImageContent)]
            if images:
                text += "\n(see attached image)"
                pending_images.extend(_user_parts(list(images)))  # type: ignore[arg-type]
            out.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": text or "(no output)"})
    if pending_images:
        out.append({"role": "user", "content": pending_images})
    return out


async def stream_openai(
    model: Model, context: Context, options: StreamOptions
) -> AsyncIterator[AssistantMessageEvent]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=options.api_key or os.environ.get("OPENAI_API_KEY"), base_url=model.base_url)
    params: dict[str, Any] = {
        "model": model.id,
        "messages": convert_messages(context),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if context.tools:
        params["tools"] = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in context.tools
        ]
    if options.max_tokens:
        params["max_completion_tokens"] = options.max_tokens
    if options.thinking != "off" and model.reasoning:
        params["reasoning_effort"] = options.thinking
    elif options.temperature is not None:
        params["temperature"] = options.temperature

    msg = AssistantMessage(provider=model.provider, model=model.id)
    yield StartEvent(partial=msg)

    # Tool calls stream by index; keep their content position and raw argument text.
    calls: dict[int, tuple[int, str]] = {}

    def current(kind: type[TextContent] | type[ThinkingContent]) -> int:
        if not msg.content or not isinstance(msg.content[-1], kind):
            msg.content.append(kind(text="") if kind is TextContent else kind(thinking=""))  # type: ignore[call-arg]
        return len(msg.content) - 1

    stream = await client.chat.completions.create(**params)
    async for chunk in stream:
        if chunk.usage:
            cached = getattr(getattr(chunk.usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            msg.usage.input = (chunk.usage.prompt_tokens or 0) - cached
            msg.usage.cache_read = cached
            msg.usage.output = chunk.usage.completion_tokens or 0
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if reasoning:
            idx = current(ThinkingContent)
            block = msg.content[idx]
            assert isinstance(block, ThinkingContent)
            block.thinking += reasoning
            yield ThinkingDeltaEvent(index=idx, delta=reasoning, partial=msg)
        if delta.content:
            idx = current(TextContent)
            block = msg.content[idx]
            assert isinstance(block, TextContent)
            block.text += delta.content
            yield TextDeltaEvent(index=idx, delta=delta.content, partial=msg)
        for tc in delta.tool_calls or []:
            if tc.index not in calls:
                msg.content.append(ToolCall(id=tc.id or f"call_{tc.index}", name=""))
                calls[tc.index] = (len(msg.content) - 1, "")
            pos, raw = calls[tc.index]
            call = msg.content[pos]
            assert isinstance(call, ToolCall)
            if tc.function and tc.function.name:
                call.name += tc.function.name
            if tc.function and tc.function.arguments:
                raw += tc.function.arguments
            calls[tc.index] = (pos, raw)
        if choice.finish_reason:
            msg.stop_reason = STOP_REASONS.get(choice.finish_reason, "stop")
            if choice.finish_reason == "content_filter":
                msg.error_message = "Response blocked by content filter."

    for pos, raw in calls.values():
        call = msg.content[pos]
        assert isinstance(call, ToolCall)
        parsed = parse_tool_arguments(call.id, call.name, raw)
        msg.content[pos] = parsed
        yield ToolCallEndEvent(index=pos, tool_call=parsed, partial=msg)
    if msg.tool_calls and msg.stop_reason == "stop":
        msg.stop_reason = "tool_use"

    yield DoneEvent(message=msg)
