import asyncio

from pydantic import BaseModel

from my_harness.agent import Agent, AgentTool, AgentToolResult
from my_harness.agent.types import AfterToolCallResult, BeforeToolCallResult
from my_harness.ai import FauxProvider, get_model, register_provider
from my_harness.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage


class EchoParams(BaseModel):
    text: str


class Echo(AgentTool[EchoParams]):
    name = "echo"
    description = "Echo text back"
    Params = EchoParams

    def __init__(self):
        self.calls: list[str] = []

    async def execute(self, call_id, params, signal, on_update=None):
        self.calls.append(params.text)
        if on_update:
            on_update(AgentToolResult.text("partial"))
        return AgentToolResult.text(f"echo: {params.text}")


class Slow(AgentTool[EchoParams]):
    name = "slow"
    description = "Sleeps"
    Params = EchoParams

    async def execute(self, call_id, params, signal, on_update=None):
        await asyncio.sleep(10)
        return AgentToolResult.text("done")


def call(name="echo", cid="c1", **args):
    return ToolCall(id=cid, name=name, arguments=args)


def make_agent(faux, tools=None, **kw):
    agent = Agent(get_model("faux"), system_prompt="sys", tools=tools or [], **kw)
    events = []
    agent.subscribe(lambda e: events.append(e.type))
    return agent, events


async def test_simple_prompt(faux):
    faux.push("hi there")
    agent, events = make_agent(faux)
    new = await agent.prompt("hello")
    assert [m.role for m in new] == ["user", "assistant"]
    assert events[0] == "agent_start" and events[-1] == "agent_end"
    assert "message_update" in events
    assert agent.state.messages[-1].text == "hi there"
    assert not agent.state.is_streaming


async def test_tool_loop(faux):
    echo = Echo()
    faux.push(call(text="a"), "finished")
    agent, events = make_agent(faux, [echo])
    await agent.prompt("go")
    assert echo.calls == ["a"]
    roles = [m.role for m in agent.state.messages]
    assert roles == ["user", "assistant", "tool_result", "assistant"]
    assert agent.state.messages[2].content[0].text == "echo: a"
    assert events.count("turn_start") == 2
    assert "tool_execution_update" in events
    # Tools are declared to the model and the system prompt is sent.
    assert faux.requests[0].tools[0].name == "echo"
    assert faux.requests[0].system_prompt == "sys"
    assert faux.requests[1].messages[-1].role == "tool_result"


async def test_invalid_args_and_unknown_tool(faux):
    faux.push([call(cid="1"), call(name="nope", cid="2", text="x")], "ok")
    agent, _ = make_agent(faux, [Echo()])
    await agent.prompt("go")
    r1, r2 = [m for m in agent.state.messages if isinstance(m, ToolResultMessage)]
    assert r1.is_error and "Invalid arguments" in r1.content[0].text
    assert r2.is_error and "not found" in r2.content[0].text


async def test_tool_exception_is_error_result(faux):
    class Boom(Echo):
        async def execute(self, call_id, params, signal, on_update=None):
            raise ValueError("kaboom")

    faux.push(call(text="x"), "ok")
    agent, _ = make_agent(faux, [Boom()])
    await agent.prompt("go")
    result = agent.state.messages[2]
    assert result.is_error and result.content[0].text == "kaboom"


async def test_hooks(faux):
    echo = Echo()

    async def before(ctx):
        if ctx.params.text == "bad":
            return BeforeToolCallResult(block=True, reason="nope")
        return None

    async def after(ctx, result, is_error):
        return AfterToolCallResult(content=[TextContent(text="rewritten")])

    faux.push([call(cid="1", text="bad"), call(cid="2", text="good")], "ok")
    agent, _ = make_agent(faux, [echo], before_tool_call=before, after_tool_call=after)
    await agent.prompt("go")
    r1, r2 = agent.state.messages[2:4]
    assert echo.calls == ["good"]
    assert r1.content[0].text == "nope" and r1.is_error
    assert r2.content[0].text == "rewritten" and not r2.is_error


async def test_parallel_results_keep_order(faux):
    class Delayed(Echo):
        async def execute(self, call_id, params, signal, on_update=None):
            await asyncio.sleep(0.05 if params.text == "first" else 0)
            return AgentToolResult.text(params.text)

    faux.push([call(cid="1", text="first"), call(cid="2", text="second")], "ok")
    agent, _ = make_agent(faux, [Delayed()])
    await agent.prompt("go")
    assert [m.content[0].text for m in agent.state.messages[2:4]] == ["first", "second"]


async def test_abort_during_tool(faux):
    faux.push(call(name="slow", text="x"), "never")
    agent, _ = make_agent(faux, [Slow()])
    task = asyncio.create_task(agent.prompt("go"))
    await asyncio.sleep(0.05)
    agent.abort()
    await asyncio.wait_for(task, 2)
    result = agent.state.messages[2]
    assert result.is_error and "aborted" in result.content[0].text
    assert len(faux.requests) == 1


async def test_abort_during_stream():
    slow = FauxProvider(["a long streaming answer " * 20], chunk_size=1, delay=0.01)
    register_provider("faux", slow)
    agent, _ = make_agent(slow)
    task = asyncio.create_task(agent.prompt("go"))
    await asyncio.sleep(0.1)
    agent.abort()
    await asyncio.wait_for(task, 2)
    last = agent.state.messages[-1]
    assert isinstance(last, AssistantMessage) and last.stop_reason == "aborted"


async def test_follow_up_and_steering(faux):
    echo = Echo()
    faux.push(call(text="a"), "first done", "second done")
    agent, _ = make_agent(faux, [echo])
    agent.follow_up("and then?")
    await agent.prompt("go")
    texts = [m.content if isinstance(m, UserMessage) else None for m in agent.state.messages]
    assert texts.count("and then?") == 1
    assert agent.state.messages[-1].text == "second done"


async def test_length_stop_fails_tool_calls(faux):
    truncated = AssistantMessage(
        provider="faux", model="faux", content=[call(text="x")], stop_reason="length"
    )
    echo = Echo()
    faux.push(truncated, "ok")
    agent, _ = make_agent(faux, [echo])
    await agent.prompt("go")
    assert echo.calls == []
    assert agent.state.messages[2].is_error


async def test_error_stops_loop(faux):
    err = AssistantMessage(provider="faux", model="faux", stop_reason="error", error_message="rate limited")
    faux.push(err)
    agent, _ = make_agent(faux)
    await agent.prompt("go")
    assert agent.state.error == "rate limited"
    assert len(faux.requests) == 1


async def test_continue(faux):
    faux.push("retried")
    agent, _ = make_agent(faux)
    agent.state.messages.append(UserMessage(content="hi"))
    await agent.continue_()
    assert agent.state.messages[-1].text == "retried"
