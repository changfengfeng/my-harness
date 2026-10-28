import json

from my_harness.ai.types import AssistantMessage, TextContent, ToolCall, UserMessage
from my_harness.coding import cli
from my_harness.coding.session import SessionManager
from my_harness.coding.system_prompt import build_system_prompt, load_context_files
from my_harness.coding.tools import create_tools


def test_session_roundtrip(tmp_path):
    s = SessionManager.create(tmp_path, tmp_path / "sessions")
    s.append_model_change("anthropic:x", "off")
    s.append_message(UserMessage(content="hi"))
    s.append_message(
        AssistantMessage(
            provider="p",
            model="m",
            content=[TextContent(text="yo"), ToolCall(id="1", name="read", arguments={"path": "a"})],
        )
    )
    reopened = SessionManager.open(s.path)
    msgs = reopened.messages()
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].tool_calls[0].arguments == {"path": "a"}
    assert reopened.last_model() == ("anthropic:x", "off")
    assert SessionManager.latest(tmp_path, tmp_path / "sessions").path == s.path


def test_system_prompt_includes_tools_and_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "AGENTS.md").write_text("Use tabs.")
    files = load_context_files(tmp_path)
    assert [c for _, c in files] == ["Use tabs."]
    prompt = build_system_prompt(tmp_path, create_tools(tmp_path, ["read", "bash"]))
    assert "- read:" in prompt and "- bash:" in prompt and "- edit:" not in prompt
    assert "Use tabs." in prompt and str(tmp_path) in prompt


def test_cli_print_mode(tmp_path, capsys, faux):
    (tmp_path / "hello.txt").write_text("hi from file")
    faux.push(ToolCall(id="1", name="read", arguments={"path": "hello.txt"}), "The file says hi.")
    code = cli.main(["-m", "faux", "--no-session", "--cwd", str(tmp_path), "-p", "read hello.txt"])
    assert code == 0
    assert capsys.readouterr().out.strip() == "The file says hi."
    assert faux.requests[1].messages[-1].content[0].text == "hi from file"


def test_cli_json_mode_and_session(tmp_path, capsys, faux, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    faux.push("answer")
    assert cli.main(["-m", "faux", "--cwd", str(tmp_path), "--json", "q"]) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[0]["type"] == "agent_start" and events[-1]["type"] == "agent_end"
    session = SessionManager.latest(tmp_path)
    assert session and [m.role for m in session.messages()] == ["user", "assistant"]

    faux.push("second")
    assert cli.main(["-m", "faux", "--cwd", str(tmp_path), "-c", "-p", "again"]) == 0
    assert len(faux.requests[-1].messages) == 3


def test_latest_skips_sessions_without_messages(tmp_path):
    used = SessionManager.create(tmp_path, tmp_path / "sessions")
    used.append_message(UserMessage(content="hi"))
    SessionManager.create(tmp_path, tmp_path / "sessions").append_model_change("faux:faux", "off")
    assert SessionManager.latest(tmp_path, tmp_path / "sessions").path == used.path


def test_renderer_ends_thinking_line_before_tool_output():
    import io

    from my_harness.agent.types import MessageEnd, MessageUpdate
    from my_harness.ai.types import ThinkingContent, ThinkingDeltaEvent

    out = io.StringIO()
    r = cli.Renderer(out)
    msg = AssistantMessage(provider="p", model="m", content=[ThinkingContent(thinking="plan")])
    r(MessageUpdate(message=msg, event=ThinkingDeltaEvent(index=0, delta="plan", partial=msg)))
    r(MessageEnd(message=msg))
    assert out.getvalue().endswith("plan" + cli.RESET + "\n")
