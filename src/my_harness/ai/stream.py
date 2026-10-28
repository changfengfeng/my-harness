"""Provider registry and the unified `stream()` / `complete()` entry points."""

from __future__ import annotations

from collections.abc import AsyncIterator

from .models import Model
from .providers.base import Provider
from .types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    DoneEvent,
    ErrorEvent,
    StartEvent,
    StreamOptions,
)

_providers: dict[str, Provider] = {}


def register_provider(name: str, provider: Provider) -> None:
    _providers[name] = provider


def _resolve(name: str) -> Provider:
    if name in _providers:
        return _providers[name]
    if name == "anthropic":
        from .providers.anthropic import stream_anthropic

        return stream_anthropic
    if name == "openai":
        from .providers.openai import stream_openai

        return stream_openai
    if name == "faux":
        from .providers.faux import FauxProvider

        _providers[name] = FauxProvider()
        return _providers[name]
    raise ValueError(f"No provider registered for {name!r}")


async def stream(
    model: Model, context: Context, options: StreamOptions | None = None
) -> AsyncIterator[AssistantMessageEvent]:
    """Stream one assistant response.

    Never raises for provider failures: errors surface as a terminal ErrorEvent whose message has
    stop_reason="error", so callers can keep the failed turn in the transcript.
    """
    options = options or StreamOptions()
    partial: AssistantMessage | None = None
    try:
        provider = _resolve(model.provider)
        async for event in provider(model, context, options):
            if isinstance(event, StartEvent):
                partial = event.partial
            yield event
            if isinstance(event, DoneEvent | ErrorEvent):
                return
    except Exception as exc:  # noqa: BLE001
        failed = partial or AssistantMessage(provider=model.provider, model=model.id)
        failed.stop_reason = "error"
        failed.error_message = f"{type(exc).__name__}: {exc}"
        yield ErrorEvent(message=failed)
        return
    failed = partial or AssistantMessage(provider=model.provider, model=model.id)
    failed.stop_reason = "error"
    failed.error_message = "Provider stream ended without a terminal event"
    yield ErrorEvent(message=failed)


async def complete(model: Model, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
    async for event in stream(model, context, options):
        if isinstance(event, DoneEvent | ErrorEvent):
            return event.message
    raise AssertionError("unreachable")
