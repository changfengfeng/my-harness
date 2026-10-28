"""Agent-level types: tools, hooks, loop configuration, and events."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel

from ..ai.models import Model
from ..ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    ImageContent,
    Message,
    StreamOptions,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
)

AgentMessage = Any
"""Anything with a `role`. Messages with roles other than user/assistant/tool_result are app-specific
(UI notes, bash transcripts, ...) and are dropped or rewritten by `convert_to_llm` before each request."""

P = TypeVar("P", bound=BaseModel)

UpdateCallback = Callable[["AgentToolResult"], None]


@dataclass
class AgentToolResult:
    content: list[TextContent | ImageContent]
    details: Any = None

    @classmethod
    def text(cls, text: str, details: Any = None) -> AgentToolResult:
        return cls(content=[TextContent(text=text)], details=details)


class ToolError(Exception):
    """Raise from `AgentTool.execute` to report a failure to the model as an error result."""


class AgentTool(ABC, Generic[P]):
    """A tool the agent can execute. Subclasses set `name`, `description`, `Params` and `execute`."""

    name: ClassVar[str]
    label: ClassVar[str] = ""
    description: ClassVar[str]
    Params: ClassVar[type[BaseModel]]

    @abstractmethod
    async def execute(
        self, call_id: str, params: P, signal: asyncio.Event, on_update: UpdateCallback | None = None
    ) -> AgentToolResult: ...

    def declaration(self) -> Tool:
        schema = self.Params.model_json_schema()
        schema.pop("title", None)
        return Tool(name=self.name, description=self.description, parameters=schema)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


@dataclass
class ToolCallContext:
    tool_call: ToolCall
    tool: AgentTool[Any]
    params: BaseModel
    assistant_message: AssistantMessage
    messages: Sequence[AgentMessage]


@dataclass
class BeforeToolCallResult:
    block: bool = False
    reason: str | None = None


@dataclass
class AfterToolCallResult:
    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = None


BeforeToolCall = Callable[[ToolCallContext], Awaitable[BeforeToolCallResult | None]]
AfterToolCall = Callable[[ToolCallContext, AgentToolResult, bool], Awaitable[AfterToolCallResult | None]]
MessageSource = Callable[[], Awaitable[list[AgentMessage]]]
TransformContext = Callable[[list[AgentMessage], asyncio.Event], Awaitable[list[AgentMessage]]]
ConvertToLlm = Callable[[list[AgentMessage]], list[Message]]


def default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    return [m for m in messages if getattr(m, "role", None) in ("user", "assistant", "tool_result")]


@dataclass
class AgentLoopConfig:
    model: Model
    options: StreamOptions = field(default_factory=StreamOptions)
    convert_to_llm: ConvertToLlm = default_convert_to_llm
    transform_context: TransformContext | None = None
    """Runs before every request on the agent-level transcript, e.g. for compaction or pruning."""
    before_tool_call: BeforeToolCall | None = None
    """Runs after argument validation. Return `block=True` to refuse the call."""
    after_tool_call: AfterToolCall | None = None
    """Runs after execution. Fields that are not None override the result."""
    get_steering_messages: MessageSource | None = None
    """Polled after each turn; returned messages are injected before the next request."""
    get_follow_up_messages: MessageSource | None = None
    """Polled when the agent would otherwise stop; returned messages start another turn."""
    tool_execution: Literal["sequential", "parallel"] = "parallel"


@dataclass
class AgentContext:
    system_prompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool[Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass
class AgentStart:
    type: ClassVar[str] = "agent_start"


@dataclass
class AgentEnd:
    messages: list[AgentMessage]
    type: ClassVar[str] = "agent_end"


@dataclass
class TurnStart:
    type: ClassVar[str] = "turn_start"


@dataclass
class TurnEnd:
    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    type: ClassVar[str] = "turn_end"


@dataclass
class MessageStart:
    message: AgentMessage
    type: ClassVar[str] = "message_start"


@dataclass
class MessageUpdate:
    message: AssistantMessage
    event: AssistantMessageEvent
    type: ClassVar[str] = "message_update"


@dataclass
class MessageEnd:
    message: AgentMessage
    type: ClassVar[str] = "message_end"


@dataclass
class ToolExecutionStart:
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    type: ClassVar[str] = "tool_execution_start"


@dataclass
class ToolExecutionUpdate:
    tool_call_id: str
    tool_name: str
    partial_result: AgentToolResult
    type: ClassVar[str] = "tool_execution_update"


@dataclass
class ToolExecutionEnd:
    tool_call_id: str
    tool_name: str
    result: AgentToolResult
    is_error: bool
    type: ClassVar[str] = "tool_execution_end"


AgentEvent = (
    AgentStart
    | AgentEnd
    | TurnStart
    | TurnEnd
    | MessageStart
    | MessageUpdate
    | MessageEnd
    | ToolExecutionStart
    | ToolExecutionUpdate
    | ToolExecutionEnd
)
EventSink = Callable[[AgentEvent], Awaitable[None] | None]
