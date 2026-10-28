"""Shared helpers for provider implementations."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from ..models import Model
from ..types import AssistantMessageEvent, Context, StreamOptions, ToolCall

StreamCancel = Callable[[], bool]


class Provider(Protocol):
    def __call__(
        self, model: Model, context: Context, options: StreamOptions
    ) -> AsyncIterator[AssistantMessageEvent]: ...


def parse_tool_arguments(call_id: str, name: str, raw: str) -> ToolCall:
    """Build a ToolCall from raw JSON argument text, recording parse failures instead of raising."""
    if not raw.strip():
        return ToolCall(id=call_id, name=name, arguments={})
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ToolCall(id=call_id, name=name, arguments_error=f"Invalid JSON arguments ({exc}): {raw}")
    if not isinstance(parsed, dict):
        return ToolCall(id=call_id, name=name, arguments_error=f"Arguments must be a JSON object: {raw}")
    return ToolCall(id=call_id, name=name, arguments=parsed)
