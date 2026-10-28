"""Anthropic Messages API provider."""

from __future__ import annotations

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

THINKING_BUDGETS = {"low": 2048, "medium": 8192, "high": 16384}

STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_use",
    "refusal": "error",
}


def _user_blocks(content: str | list[TextContent | ImageContent]) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    blocks: list[dict[str, Any]] = []
    for c in content:
        if isinstance(c, TextContent):
            blocks.append({"type": "text", "text": c.text})
        else:
            blocks.append(
                {"type": "image", "source": {"type": "base64", "media_type": c.mime_type, "data": c.data}}
            )
    return blocks


def convert_messages(context: Context) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in context.messages:
        if isinstance(msg, UserMessage):
            out.append({"role": "user", "content": _user_blocks(msg.content)})
        elif isinstance(msg, AssistantMessage):
            if msg.stop_reason in ("error", "aborted") and not msg.tool_calls:
                continue
            blocks: list[dict[str, Any]] = []
            for c in msg.content:
                if isinstance(c, TextContent) and c.text:
                    blocks.append({"type": "text", "text": c.text})
                elif isinstance(c, ThinkingContent) and c.signature:
                    blocks.append({"type": "thinking", "thinking": c.thinking, "signature": c.signature})
                elif isinstance(c, ToolCall):
                    blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        elif isinstance(msg, ToolResultMessage):
            block = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": _user_blocks(list(msg.content)) or "(no output)",
                "is_error": msg.is_error,
            }
            # Consecutive tool results must share one user message.
            prev = out[-1] if out else None
            if (
                prev
                and prev["role"] == "user"
                and isinstance(prev["content"], list)
                and prev["content"]
                and prev["content"][0].get("type") == "tool_result"
            ):
                prev["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


async def stream_anthropic(
    model: Model, context: Context, options: StreamOptions
) -> AsyncIterator[AssistantMessageEvent]:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(
        api_key=options.api_key or os.environ.get("ANTHROPIC_API_KEY"), base_url=model.base_url
    )
    max_tokens = options.max_tokens or model.max_output_tokens
    params: dict[str, Any] = {
        "model": model.id,
        "messages": convert_messages(context),
        "max_tokens": max_tokens,
        "stream": True,
    }
    if context.system_prompt:
        params["system"] = context.system_prompt
    if context.tools:
        params["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in context.tools
        ]
    if options.thinking != "off" and model.reasoning:
        budget = THINKING_BUDGETS[options.thinking]
        params["thinking"] = {"type": "enabled", "budget_tokens": budget}
        params["max_tokens"] = max(max_tokens, budget + 4096)
    elif options.temperature is not None:
        params["temperature"] = options.temperature

    msg = AssistantMessage(provider=model.provider, model=model.id)
    tool_json: dict[int, str] = {}
    # Anthropic block index -> position in msg.content (unsupported block types are skipped).
    positions: dict[int, int] = {}
    yield StartEvent(partial=msg)

    events = await client.messages.create(**params)
    async for ev in events:
        if ev.type == "message_start":
            u = ev.message.usage
            msg.usage.input = u.input_tokens or 0
            msg.usage.cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
            msg.usage.cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        elif ev.type == "content_block_start":
            block = ev.content_block
            if block.type in ("text", "thinking", "redacted_thinking", "tool_use"):
                positions[ev.index] = len(msg.content)
            if block.type == "text":
                msg.content.append(TextContent(text=""))
            elif block.type == "thinking":
                msg.content.append(ThinkingContent(thinking=""))
            elif block.type == "redacted_thinking":
                msg.content.append(ThinkingContent(thinking="[redacted]", signature=block.data))
            elif block.type == "tool_use":
                msg.content.append(ToolCall(id=block.id, name=block.name))
                tool_json[len(msg.content) - 1] = ""
        elif ev.type == "content_block_delta":
            if ev.index not in positions:
                continue
            idx = positions[ev.index]
            current = msg.content[idx]
            d = ev.delta
            if d.type == "text_delta" and isinstance(current, TextContent):
                current.text += d.text
                yield TextDeltaEvent(index=idx, delta=d.text, partial=msg)
            elif d.type == "thinking_delta" and isinstance(current, ThinkingContent):
                current.thinking += d.thinking
                yield ThinkingDeltaEvent(index=idx, delta=d.thinking, partial=msg)
            elif d.type == "signature_delta" and isinstance(current, ThinkingContent):
                current.signature = (current.signature or "") + d.signature
            elif d.type == "input_json_delta":
                tool_json[idx] += d.partial_json
        elif ev.type == "content_block_stop":
            if ev.index not in positions:
                continue
            idx = positions[ev.index]
            current = msg.content[idx]
            if isinstance(current, ToolCall):
                call = parse_tool_arguments(current.id, current.name, tool_json.pop(idx, ""))
                msg.content[idx] = call
                yield ToolCallEndEvent(index=idx, tool_call=call, partial=msg)
        elif ev.type == "message_delta":
            if ev.delta.stop_reason:
                msg.stop_reason = STOP_REASONS.get(ev.delta.stop_reason, "stop")
                if ev.delta.stop_reason == "refusal":
                    msg.error_message = "The model refused to respond."
            msg.usage.output = ev.usage.output_tokens or 0

    yield DoneEvent(message=msg)
