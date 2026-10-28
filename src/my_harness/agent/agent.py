"""Stateful Agent: owns the transcript, runs the loop, fans events out to subscribers."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ..ai.models import Model
from ..ai.types import (
    AssistantMessage,
    ImageContent,
    StreamOptions,
    TextContent,
    ThinkingLevel,
    UserMessage,
)
from .loop import StreamFn, run_agent_loop, run_agent_loop_continue
from .types import (
    AfterToolCall,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    BeforeToolCall,
    ConvertToLlm,
    EventSink,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    TransformContext,
    default_convert_to_llm,
)

QueueMode = Literal["all", "one-at-a-time"]


@dataclass
class AgentState:
    model: Model
    system_prompt: str = ""
    thinking: ThinkingLevel = "off"
    tools: list[AgentTool[Any]] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    is_streaming: bool = False
    stream_message: AssistantMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error: str | None = None


class Agent:
    def __init__(
        self,
        model: Model,
        *,
        system_prompt: str = "",
        tools: list[AgentTool[Any]] | None = None,
        messages: list[AgentMessage] | None = None,
        thinking: ThinkingLevel = "off",
        options: StreamOptions | None = None,
        convert_to_llm: ConvertToLlm = default_convert_to_llm,
        transform_context: TransformContext | None = None,
        before_tool_call: BeforeToolCall | None = None,
        after_tool_call: AfterToolCall | None = None,
        tool_execution: Literal["sequential", "parallel"] = "parallel",
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
        stream_fn: StreamFn | None = None,
    ):
        self.state = AgentState(
            model=model,
            system_prompt=system_prompt,
            thinking=thinking,
            tools=list(tools or []),
            messages=list(messages or []),
        )
        self.options = options or StreamOptions()
        self.convert_to_llm = convert_to_llm
        self.transform_context = transform_context
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.tool_execution = tool_execution
        self.steering_mode = steering_mode
        self.follow_up_mode = follow_up_mode
        self.stream_fn = stream_fn
        self._listeners: list[EventSink] = []
        self._steering: list[AgentMessage] = []
        self._follow_ups: list[AgentMessage] = []
        self._abort: asyncio.Event | None = None
        self._idle = asyncio.Event()
        self._idle.set()

    # -- subscriptions ---------------------------------------------------

    def subscribe(self, listener: EventSink) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    async def _dispatch(self, event: AgentEvent) -> None:
        s = self.state
        if (
            isinstance(event, MessageStart)
            and isinstance(event.message, AssistantMessage)
            or isinstance(event, MessageUpdate)
        ):
            s.stream_message = event.message
        elif isinstance(event, MessageEnd):
            s.stream_message = None
            if isinstance(event.message, AssistantMessage) and event.message.error_message:
                s.error = event.message.error_message
        elif isinstance(event, ToolExecutionStart):
            s.pending_tool_calls.add(event.tool_call_id)
        elif isinstance(event, ToolExecutionEnd):
            s.pending_tool_calls.discard(event.tool_call_id)
        for listener in list(self._listeners):
            result = listener(event)
            if inspect.isawaitable(result):
                await result

    # -- queues ----------------------------------------------------------

    def steer(self, message: AgentMessage | str) -> None:
        """Queue a message to inject after the current turn, interrupting a tool-call chain."""
        self._steering.append(_as_message(message))

    def follow_up(self, message: AgentMessage | str) -> None:
        """Queue a message to deliver once the agent would otherwise stop."""
        self._follow_ups.append(_as_message(message))

    @staticmethod
    def _drain(queue: list[AgentMessage], mode: QueueMode) -> list[AgentMessage]:
        if mode == "all":
            items = queue[:]
            queue.clear()
            return items
        return [queue.pop(0)] if queue else []

    # -- running ---------------------------------------------------------

    def _config(self) -> AgentLoopConfig:
        options = self.options.model_copy(update={"thinking": self.state.thinking})

        async def steering() -> list[AgentMessage]:
            return self._drain(self._steering, self.steering_mode)

        async def follow_ups() -> list[AgentMessage]:
            return self._drain(self._follow_ups, self.follow_up_mode)

        return AgentLoopConfig(
            model=self.state.model,
            options=options,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            get_steering_messages=steering,
            get_follow_up_messages=follow_ups,
            tool_execution=self.tool_execution,
        )

    def _context(self) -> AgentContext:
        return AgentContext(
            system_prompt=self.state.system_prompt,
            messages=self.state.messages,
            tools=list(self.state.tools),
        )

    async def prompt(
        self, message: AgentMessage | str, images: list[ImageContent] | None = None
    ) -> list[AgentMessage]:
        if self.state.is_streaming:
            raise RuntimeError("Agent is already running; use steer() or follow_up()")
        prompt = _as_message(message, images)
        return await self._run(lambda kw: run_agent_loop([prompt], **kw))

    async def continue_(self) -> list[AgentMessage]:
        if self.state.is_streaming:
            raise RuntimeError("Agent is already running")
        return await self._run(lambda kw: run_agent_loop_continue(**kw))

    async def _run(self, start: Callable[[dict[str, Any]], Any]) -> list[AgentMessage]:
        self._abort = asyncio.Event()
        self._idle.clear()
        self.state.is_streaming = True
        self.state.error = None
        kwargs: dict[str, Any] = {
            "context": self._context(),
            "config": self._config(),
            "emit": self._dispatch,
            "signal": self._abort,
        }
        if self.stream_fn is not None:
            kwargs["stream_fn"] = self.stream_fn
        try:
            new_messages: list[AgentMessage] = await start(kwargs)
            return new_messages
        finally:
            self.state.is_streaming = False
            self.state.stream_message = None
            self.state.pending_tool_calls.clear()
            self._abort = None
            self._idle.set()

    def abort(self) -> None:
        if self._abort is not None:
            self._abort.set()

    async def wait_for_idle(self) -> None:
        await self._idle.wait()

    def reset(self) -> None:
        self.state.messages.clear()
        self._steering.clear()
        self._follow_ups.clear()
        self.state.error = None


def _as_message(message: AgentMessage | str, images: list[ImageContent] | None = None) -> AgentMessage:
    if not isinstance(message, str):
        return message
    if images:
        return UserMessage(content=[TextContent(text=message), *images])
    return UserMessage(content=message)
