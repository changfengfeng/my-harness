"""Scripted in-process provider for tests and offline demos."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from ..models import Model
from ..types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    DoneEvent,
    StartEvent,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolCallEndEvent,
)

FauxReply = str | ToolCall | list[str | ToolCall | ThinkingContent] | AssistantMessage
FauxScript = Sequence[FauxReply | Callable[[Context], FauxReply]]


class FauxProvider:
    """Replays scripted replies in order, one per request. Records every request context."""

    def __init__(self, script: FauxScript = (), chunk_size: int = 8, delay: float = 0.0):
        self.script = list(script)
        self.chunk_size = chunk_size
        self.delay = delay
        self.requests: list[Context] = []

    def push(self, *replies: FauxReply | Callable[[Context], FauxReply]) -> None:
        self.script.extend(replies)

    def _next(self, context: Context) -> FauxReply:
        if not self.script:
            last = next((m for m in reversed(context.messages) if m.role == "user"), None)
            text = last.content if last and isinstance(last.content, str) else "..."
            return f"(faux) You said: {text}"
        reply = self.script.pop(0)
        return reply(context) if callable(reply) else reply

    async def __call__(
        self, model: Model, context: Context, options: StreamOptions
    ) -> AsyncIterator[AssistantMessageEvent]:
        self.requests.append(context.model_copy(deep=True))
        reply = self._next(context)
        if isinstance(reply, AssistantMessage):
            yield StartEvent(partial=reply)
            yield DoneEvent(message=reply)
            return
        parts: list[Any] = reply if isinstance(reply, list) else [reply]
        msg = AssistantMessage(provider=model.provider, model=model.id)
        yield StartEvent(partial=msg)
        for part in parts:
            if self.delay:
                await asyncio.sleep(self.delay)
            if isinstance(part, str):
                msg.content.append(TextContent(text=""))
                idx = len(msg.content) - 1
                block = msg.content[idx]
                assert isinstance(block, TextContent)
                for i in range(0, len(part), self.chunk_size):
                    delta = part[i : i + self.chunk_size]
                    block.text += delta
                    yield TextDeltaEvent(index=idx, delta=delta, partial=msg)
                    if self.delay:
                        await asyncio.sleep(self.delay)
            elif isinstance(part, ThinkingContent):
                msg.content.append(ThinkingContent(thinking=part.thinking))
                yield ThinkingDeltaEvent(index=len(msg.content) - 1, delta=part.thinking, partial=msg)
            else:
                msg.content.append(part)
                yield ToolCallEndEvent(index=len(msg.content) - 1, tool_call=part, partial=msg)
        msg.stop_reason = "tool_use" if msg.tool_calls else "stop"
        msg.usage.input = sum(len(str(m)) for m in context.messages) // 4
        msg.usage.output = sum(len(str(c)) for c in msg.content) // 4
        yield DoneEvent(message=msg)
