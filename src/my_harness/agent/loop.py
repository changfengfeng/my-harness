"""The agent loop: stream a response, run its tool calls, feed results back, repeat.

Stateless: everything lives in the AgentContext passed in. `Agent` (agent.py) wraps this with state,
subscriptions, queues and abort handling.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from pydantic import ValidationError

from ..ai.stream import stream as default_stream
from ..ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    StartEvent,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from .types import (
    AgentContext,
    AgentEnd,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentStart,
    AgentTool,
    AgentToolResult,
    EventSink,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolCallContext,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    TurnEnd,
    TurnStart,
)

T = TypeVar("T")

StreamFn = Any
"""Callable with the signature of `my_harness.ai.stream`."""


class Aborted(Exception):
    pass


async def _race(awaitable: Any, signal: asyncio.Event) -> Any:
    """Await `awaitable`, raising Aborted (and cancelling it) if `signal` is set first."""
    if signal.is_set():
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        raise Aborted
    task = asyncio.ensure_future(awaitable)
    waiter = asyncio.ensure_future(signal.wait())
    try:
        await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiter.cancel()
    if task.done():
        return task.result()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
    raise Aborted


async def _emit(sink: EventSink, event: AgentEvent) -> None:
    result = sink(event)
    if inspect.isawaitable(result):
        await result


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: EventSink,
    signal: asyncio.Event | None = None,
    stream_fn: StreamFn = default_stream,
) -> list[AgentMessage]:
    """Add `prompts` to the context and run until the model stops. Returns all new messages."""
    signal = signal or asyncio.Event()
    new_messages: list[AgentMessage] = []
    await _emit(emit, AgentStart())
    await _emit(emit, TurnStart())
    for prompt in prompts:
        await _emit(emit, MessageStart(prompt))
        await _emit(emit, MessageEnd(prompt))
        context.messages.append(prompt)
        new_messages.append(prompt)
    await _run(context, config, emit, signal, stream_fn, new_messages, first_turn_started=True)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: EventSink,
    signal: asyncio.Event | None = None,
    stream_fn: StreamFn = default_stream,
) -> list[AgentMessage]:
    """Continue from the current context (last message must be user or tool_result), e.g. for retries."""
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if getattr(context.messages[-1], "role", None) == "assistant":
        raise ValueError("Cannot continue from an assistant message")
    signal = signal or asyncio.Event()
    new_messages: list[AgentMessage] = []
    await _emit(emit, AgentStart())
    await _run(context, config, emit, signal, stream_fn, new_messages, first_turn_started=False)
    return new_messages


async def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
    stream_fn: StreamFn = default_stream,
) -> AsyncIterator[AgentEvent]:
    """Async-iterator flavour of `run_agent_loop`."""
    queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
    task = asyncio.create_task(run_agent_loop(prompts, context, config, queue.put_nowait, signal, stream_fn))
    task.add_done_callback(lambda _: queue.put_nowait(None))
    while (event := await queue.get()) is not None:
        yield event
    await task


async def _run(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: EventSink,
    signal: asyncio.Event,
    stream_fn: StreamFn,
    new_messages: list[AgentMessage],
    first_turn_started: bool,
) -> None:
    turn_started = first_turn_started
    pending = await config.get_steering_messages() if config.get_steering_messages else []

    while True:
        has_tool_calls = True
        while has_tool_calls or pending:
            if not turn_started:
                await _emit(emit, TurnStart())
            turn_started = False

            for message in pending:
                await _emit(emit, MessageStart(message))
                await _emit(emit, MessageEnd(message))
                context.messages.append(message)
                new_messages.append(message)
            pending = []

            message = await _stream_assistant(context, config, emit, signal, stream_fn)
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                await _emit(emit, TurnEnd(message, []))
                await _emit(emit, AgentEnd(new_messages))
                return

            tool_results: list[ToolResultMessage] = []
            calls = message.tool_calls
            has_tool_calls = bool(calls)
            if calls:
                if message.stop_reason == "length":
                    tool_results = await _fail_truncated(calls, emit)
                else:
                    tool_results = await _execute_tool_calls(context, message, config, emit, signal)
                for result in tool_results:
                    context.messages.append(result)
                    new_messages.append(result)

            await _emit(emit, TurnEnd(message, tool_results))

            if signal.is_set():
                await _emit(emit, AgentEnd(new_messages))
                return
            pending = await config.get_steering_messages() if config.get_steering_messages else []

        follow_ups = await config.get_follow_up_messages() if config.get_follow_up_messages else []
        if not follow_ups:
            break
        pending = follow_ups

    await _emit(emit, AgentEnd(new_messages))


async def _stream_assistant(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: EventSink,
    signal: asyncio.Event,
    stream_fn: StreamFn,
) -> AssistantMessage:
    messages = context.messages
    if config.transform_context:
        messages = await config.transform_context(list(messages), signal)
    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=config.convert_to_llm(list(messages)),
        tools=[t.declaration() for t in context.tools],
    )

    partial: AssistantMessage | None = None
    final: AssistantMessage | None = None
    events = stream_fn(config.model, llm_context, config.options)
    try:
        while True:
            try:
                event = await _race(events.__anext__(), signal)
            except StopAsyncIteration:
                break
            if isinstance(event, StartEvent):
                partial = event.partial
                await _emit(emit, MessageStart(partial))
            elif isinstance(event, DoneEvent | ErrorEvent):
                final = event.message
                if partial is None:
                    await _emit(emit, MessageStart(final))
                break
            elif partial is not None:
                await _emit(emit, MessageUpdate(partial, event))
    except Aborted:
        final = partial or AssistantMessage(provider=config.model.provider, model=config.model.id)
        if partial is None:
            await _emit(emit, MessageStart(final))
        final.stop_reason = "aborted"
        final.error_message = "Request aborted by user"
    finally:
        await events.aclose()

    if final is None:
        final = partial or AssistantMessage(provider=config.model.provider, model=config.model.id)
        final.stop_reason = "error"
        final.error_message = final.error_message or "Stream ended without a final message"
    context.messages.append(final)
    await _emit(emit, MessageEnd(final))
    return final


def _result_message(call: ToolCall, result: AgentToolResult, is_error: bool) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=call.id,
        tool_name=call.name,
        content=list(result.content),
        details=result.details,
        is_error=is_error,
    )


async def _fail_truncated(calls: list[ToolCall], emit: EventSink) -> list[ToolResultMessage]:
    results = []
    for call in calls:
        result = AgentToolResult.text(
            "Tool call was not executed: the response hit the output token limit, so its arguments may be "
            "truncated. Retry with smaller arguments (e.g. split large writes into several edits)."
        )
        await _emit(emit, ToolExecutionStart(call.id, call.name, call.arguments))
        await _emit(emit, ToolExecutionEnd(call.id, call.name, result, True))
        msg = _result_message(call, result, True)
        await _emit(emit, MessageStart(msg))
        await _emit(emit, MessageEnd(msg))
        results.append(msg)
    return results


async def _execute_tool_calls(
    context: AgentContext,
    assistant: AssistantMessage,
    config: AgentLoopConfig,
    emit: EventSink,
    signal: asyncio.Event,
) -> list[ToolResultMessage]:
    tools = {t.name: t for t in context.tools}
    calls = assistant.tool_calls

    async def run(call: ToolCall) -> tuple[AgentToolResult, bool]:
        await _emit(emit, ToolExecutionStart(call.id, call.name, call.arguments))
        result, is_error = await _execute_one(
            call, tools.get(call.name), assistant, context, config, emit, signal
        )
        await _emit(emit, ToolExecutionEnd(call.id, call.name, result, is_error))
        return result, is_error

    if config.tool_execution == "parallel" and len(calls) > 1:
        outcomes = list(await asyncio.gather(*(run(c) for c in calls)))
    else:
        outcomes = [await run(c) for c in calls]

    messages = []
    for call, (result, is_error) in zip(calls, outcomes, strict=True):
        msg = _result_message(call, result, is_error)
        await _emit(emit, MessageStart(msg))
        await _emit(emit, MessageEnd(msg))
        messages.append(msg)
    return messages


async def _execute_one(
    call: ToolCall,
    tool: AgentTool[Any] | None,
    assistant: AssistantMessage,
    context: AgentContext,
    config: AgentLoopConfig,
    emit: EventSink,
    signal: asyncio.Event,
) -> tuple[AgentToolResult, bool]:
    if signal.is_set():
        return AgentToolResult.text("Tool call skipped: aborted by user"), True
    if tool is None:
        return AgentToolResult.text(f"Tool {call.name!r} not found"), True
    if call.arguments_error:
        return AgentToolResult.text(call.arguments_error), True
    try:
        params = tool.Params.model_validate(call.arguments)
    except ValidationError as exc:
        return AgentToolResult.text(f"Invalid arguments for {call.name}:\n{exc}"), True

    hook_ctx = ToolCallContext(call, tool, params, assistant, context.messages)
    if config.before_tool_call:
        decision = await config.before_tool_call(hook_ctx)
        if decision and decision.block:
            return AgentToolResult.text(decision.reason or "Tool call was blocked"), True

    def on_update(partial: AgentToolResult) -> None:
        result = emit(ToolExecutionUpdate(call.id, call.name, partial))
        if inspect.isawaitable(result):
            asyncio.ensure_future(result)

    is_error = False
    try:
        result: AgentToolResult = await _race(tool.execute(call.id, params, signal, on_update), signal)
    except Aborted:
        result, is_error = AgentToolResult.text("Tool execution aborted by user"), True
    except Exception as exc:  # noqa: BLE001
        result, is_error = AgentToolResult.text(str(exc) or type(exc).__name__), True

    if config.after_tool_call:
        override = await config.after_tool_call(hook_ctx, result, is_error)
        if override:
            result = AgentToolResult(
                content=override.content if override.content is not None else result.content,
                details=override.details if override.details is not None else result.details,
            )
            if override.is_error is not None:
                is_error = override.is_error
    if not result.content:
        result.content = [TextContent(text="(no output)")]
    return result, is_error
