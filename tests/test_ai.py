from my_harness.ai import complete, get_model, stream
from my_harness.ai.providers.anthropic import convert_messages as to_anthropic
from my_harness.ai.providers.base import parse_tool_arguments
from my_harness.ai.providers.openai import convert_messages as to_openai
from my_harness.ai.types import (
    AssistantMessage,
    Context,
    ErrorEvent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def test_get_model():
    m = get_model("openai:gpt-4o")
    assert (m.provider, m.id) == ("openai", "gpt-4o")
    assert get_model("anthropic").id


async def test_faux_stream(faux):
    faux.push("hello world")
    events = [e async for e in stream(get_model("faux"), Context(messages=[UserMessage(content="hi")]))]
    assert events[0].type == "start"
    assert events[-1].type == "done"
    assert events[-1].message.text == "hello world"


async def test_provider_exception_becomes_error_event():
    from my_harness.ai import register_provider

    async def broken(model, context, options):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    register_provider("faux", broken)
    events = [e async for e in stream(get_model("faux"), Context())]
    assert isinstance(events[-1], ErrorEvent)
    assert "boom" in (events[-1].message.error_message or "")


async def test_complete(faux):
    faux.push(["a", ToolCall(id="1", name="t", arguments={"x": 1})])
    msg = await complete(get_model("faux"), Context())
    assert msg.stop_reason == "tool_use"
    assert msg.tool_calls[0].arguments == {"x": 1}


def test_parse_tool_arguments():
    assert parse_tool_arguments("1", "t", '{"a": 1}').arguments == {"a": 1}
    assert parse_tool_arguments("1", "t", "").arguments == {}
    assert parse_tool_arguments("1", "t", "{bad").arguments_error
    assert parse_tool_arguments("1", "t", "[1]").arguments_error


def _transcript():
    call1 = ToolCall(id="c1", name="read", arguments={"path": "a"})
    call2 = ToolCall(id="c2", name="read", arguments={"path": "b"})
    return Context(
        system_prompt="sys",
        messages=[
            UserMessage(content="hi"),
            AssistantMessage(
                provider="x",
                model="y",
                content=[TextContent(text="ok"), call1, call2],
                stop_reason="tool_use",
            ),
            ToolResultMessage(tool_call_id="c1", tool_name="read", content=[TextContent(text="A")]),
            ToolResultMessage(
                tool_call_id="c2", tool_name="read", content=[TextContent(text="B")], is_error=True
            ),
            AssistantMessage(provider="x", model="y", stop_reason="error", error_message="fail"),
        ],
    )


def test_anthropic_conversion_merges_tool_results_and_drops_errors():
    out = to_anthropic(_transcript())
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["type"] for b in out[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["c1", "c2"]
    assert out[2]["content"][1]["is_error"] is True


def test_openai_conversion():
    out = to_openai(_transcript())
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "tool"]
    assert out[2]["tool_calls"][0]["function"]["name"] == "read"
