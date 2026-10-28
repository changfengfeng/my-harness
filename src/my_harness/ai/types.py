"""Provider-neutral message, tool, and streaming event types."""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


def now_ms() -> int:
    return int(time.time() * 1000)


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None
    """Opaque provider data needed to replay the thinking block (e.g. Anthropic signature)."""


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    data: str
    """Base64-encoded image bytes."""
    mime_type: str


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    arguments_error: str | None = None
    """Set when the model's argument text was not a valid JSON object."""


UserContent = Annotated[TextContent | ImageContent, Field(discriminator="type")]
AssistantContent = Annotated[TextContent | ThinkingContent | ToolCall, Field(discriminator="type")]

StopReason = Literal["stop", "length", "tool_use", "error", "aborted"]


class Usage(BaseModel):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_write

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
        )


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[UserContent]
    timestamp: int = Field(default_factory=now_ms)


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent] = Field(default_factory=list)
    provider: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    timestamp: int = Field(default_factory=now_ms)

    @property
    def text(self) -> str:
        return "".join(c.text for c in self.content if isinstance(c, TextContent))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [c for c in self.content if isinstance(c, ToolCall)]


class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: list[UserContent]
    details: Any = None
    """Structured data for UIs and sessions. Never sent to the model."""
    is_error: bool = False
    timestamp: int = Field(default_factory=now_ms)


Message = Annotated[UserMessage | AssistantMessage | ToolResultMessage, Field(discriminator="role")]


class Tool(BaseModel):
    """A tool declaration as seen by the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    """JSON Schema of the arguments object."""


class Context(BaseModel):
    system_prompt: str | None = None
    messages: list[Message] = Field(default_factory=list)
    tools: list[Tool] = Field(default_factory=list)


ThinkingLevel = Literal["off", "low", "medium", "high"]


class StreamOptions(BaseModel):
    max_tokens: int | None = None
    temperature: float | None = None
    thinking: ThinkingLevel = "off"
    api_key: str | None = None


# ---------------------------------------------------------------------------
# Streaming events. Every stream ends with exactly one `done` or `error` event.
# `partial` is the assistant message accumulated so far.
# ---------------------------------------------------------------------------


class StartEvent(BaseModel):
    type: Literal["start"] = "start"
    partial: AssistantMessage


class TextDeltaEvent(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    index: int
    delta: str
    partial: AssistantMessage


class ThinkingDeltaEvent(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    index: int
    delta: str
    partial: AssistantMessage


class ToolCallEndEvent(BaseModel):
    type: Literal["toolcall_end"] = "toolcall_end"
    index: int
    tool_call: ToolCall
    partial: AssistantMessage


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    message: AssistantMessage


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: AssistantMessage


AssistantMessageEvent = (
    StartEvent | TextDeltaEvent | ThinkingDeltaEvent | ToolCallEndEvent | DoneEvent | ErrorEvent
)
