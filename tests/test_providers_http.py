"""End-to-end provider tests against a local server speaking the real Anthropic / OpenAI SSE wire formats.

These exercise the official SDKs' parsing plus our event mapping and message conversion, without API keys.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from my_harness.coding import cli


class MockServer:
    def __init__(self):
        self.responses: list[tuple[int, str, str]] = []  # (status, content_type, body)
        self.requests: list[tuple[str, dict]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                server.requests.append((self.path, json.loads(body)))
                status, ctype, payload = server.responses.pop(0)
                data = payload.encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def sse(self, events: list[tuple[str | None, dict | str]]):
        lines = []
        for name, data in events:
            if name:
                lines.append(f"event: {name}")
            lines.append(f"data: {data if isinstance(data, str) else json.dumps(data)}")
            lines.append("")
        self.responses.append((200, "text/event-stream", "\n".join(lines) + "\n"))

    def error(self, status: int, body: dict):
        self.responses.append((status, "application/json", json.dumps(body)))


@pytest.fixture
def server():
    s = MockServer()
    yield s
    s.httpd.shutdown()


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def anthropic_stream(blocks: list[dict], stop_reason: str, usage_in=10, usage_out=20):
    events: list[tuple[str | None, dict | str]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage_in, "output_tokens": 1, "cache_read_input_tokens": 5},
                },
            },
        )
    ]
    for i, block in enumerate(blocks):
        kind = block["type"]
        if kind == "text":
            start = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": t} for t in block["chunks"]]
        elif kind == "thinking":
            start = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [{"type": "thinking_delta", "thinking": t} for t in block["chunks"]]
            deltas.append({"type": "signature_delta", "signature": block["signature"]})
        else:
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": t} for t in block["chunks"]]
        events.append(
            ("content_block_start", {"type": "content_block_start", "index": i, "content_block": start})
        )
        events += [
            ("content_block_delta", {"type": "content_block_delta", "index": i, "delta": d}) for d in deltas
        ]
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": usage_out},
            },
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    return events


def test_anthropic_end_to_end(server, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    (tmp_path / "hello.txt").write_text("hi from file")
    server.sse(
        anthropic_stream(
            [
                {"type": "thinking", "chunks": ["Let me ", "read it."], "signature": "sig-123"},
                {"type": "text", "chunks": ["Reading", " the file."]},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "read",
                    "chunks": ['{"pa', 'th": "hello.txt"}'],
                },
                {"type": "tool_use", "id": "toolu_2", "name": "ls", "chunks": []},
            ],
            "tool_use",
        )
    )
    server.sse(anthropic_stream([{"type": "text", "chunks": ["The file ", "says hi."]}], "end_turn"))

    code = cli.main(
        [
            "-m",
            "anthropic:claude-test",
            "--base-url",
            server.url,
            "--thinking",
            "low",
            "--no-session",
            "--cwd",
            str(tmp_path),
            "-p",
            "read hello.txt",
        ]
    )
    out = capsys.readouterr()
    assert code == 0, out.err
    assert out.out.strip() == "The file says hi."

    path, first = server.requests[0]
    assert path == "/v1/messages"
    assert first["model"] == "claude-test" and first["stream"] is True
    assert first["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert first["max_tokens"] > 2048
    assert "read" in {t["name"] for t in first["tools"]}
    assert first["tools"][0]["input_schema"]["type"] == "object"
    assert "Working directory" in first["system"]

    _, second = server.requests[1]
    assistant, results = second["messages"][1], second["messages"][2]
    assert [b["type"] for b in assistant["content"]] == ["thinking", "text", "tool_use", "tool_use"]
    assert assistant["content"][0]["signature"] == "sig-123"
    assert assistant["content"][2]["input"] == {"path": "hello.txt"}
    assert assistant["content"][3]["input"] == {}
    assert [r["tool_use_id"] for r in results["content"]] == ["toolu_1", "toolu_2"]
    assert results["content"][0]["content"][0]["text"] == "hi from file"
    assert "hello.txt" in results["content"][1]["content"][0]["text"]


def test_anthropic_json_events_have_usage(server, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    server.sse(anthropic_stream([{"type": "text", "chunks": ["ok"]}], "end_turn", usage_in=7, usage_out=3))
    assert (
        cli.main(
            [
                "-m",
                "anthropic:x",
                "--base-url",
                server.url,
                "--no-session",
                "--cwd",
                str(tmp_path),
                "--json",
                "hi",
            ]
        )
        == 0
    )
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    end = [e for e in events if e["type"] == "message_end" and e["message"]["role"] == "assistant"][0]
    assert end["message"]["usage"] == {"input": 7, "output": 3, "cache_read": 5, "cache_write": 0}
    assert end["message"]["stop_reason"] == "stop"


def test_anthropic_http_error(server, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad")
    server.error(
        401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
    )
    code = cli.main(
        ["-m", "anthropic:x", "--base-url", server.url, "--no-session", "--cwd", str(tmp_path), "-p", "hi"]
    )
    assert code == 1
    assert "invalid x-api-key" in capsys.readouterr().err


def test_anthropic_max_tokens_skips_tool_calls(server, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    server.sse(
        anthropic_stream(
            [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "write",
                    "chunks": ['{"path": "a.txt", "content": "x"}'],
                }
            ],
            "max_tokens",
        )
    )
    server.sse(anthropic_stream([{"type": "text", "chunks": ["sorry"]}], "end_turn"))
    assert (
        cli.main(
            [
                "-m",
                "anthropic:x",
                "--base-url",
                server.url,
                "--no-session",
                "--cwd",
                str(tmp_path),
                "-p",
                "go",
            ]
        )
        == 0
    )
    assert not (tmp_path / "a.txt").exists()
    result = server.requests[1][1]["messages"][2]["content"][0]
    assert result["is_error"] is True and "token limit" in result["content"][0]["text"]


def test_anthropic_skips_unsupported_blocks(server, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    events = anthropic_stream(
        [{"type": "tool_use", "id": "t1", "name": "ls", "chunks": ['{"path": ', '"."}']}], "tool_use"
    )
    unsupported = [
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "srv_1",
                    "name": "web_search",
                    "input": {},
                },
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"query": "x"}'},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    ]
    stop_at = next(i for i, (name, _) in enumerate(events) if name == "message_delta")
    server.sse(events[:stop_at] + unsupported + events[stop_at:])
    server.sse(anthropic_stream([{"type": "text", "chunks": ["ok"]}], "end_turn"))
    assert (
        cli.main(
            [
                "-m",
                "anthropic:x",
                "--base-url",
                server.url,
                "--no-session",
                "--cwd",
                str(tmp_path),
                "-p",
                "go",
            ]
        )
        == 0
    )
    assistant = server.requests[1][1]["messages"][1]
    assert assistant["content"] == [{"type": "tool_use", "id": "t1", "name": "ls", "input": {"path": "."}}]


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def openai_chunk(delta: dict | None = None, finish: str | None = None, usage: dict | None = None):
    chunk: dict = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-test",
        "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage:
        chunk["usage"] = usage
    return (None, chunk)


def test_openai_end_to_end(server, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    (tmp_path / "a.py").write_text("x = 1\n")
    server.sse(
        [
            openai_chunk({"role": "assistant", "reasoning_content": "thinking..."}),
            openai_chunk({"content": "Editing."}),
            openai_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "edit", "arguments": ""},
                        }
                    ]
                }
            ),
            openai_chunk(
                {"tool_calls": [{"index": 0, "function": {"arguments": '{"path": "a.py", "edits": '}}]}
            ),
            openai_chunk(
                {"tool_calls": [{"index": 0, "function": {"arguments": '[{"old_text": "x = 1", '}}]}
            ),
            openai_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"new_text": "x = 2"}]}'}}]}),
            openai_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command": "cat a.py"}'},
                        }
                    ]
                }
            ),
            openai_chunk({}, finish="tool_calls"),
            openai_chunk(
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "prompt_tokens_details": {"cached_tokens": 40},
                }
            ),
            (None, "[DONE]"),
        ]
    )
    server.sse([openai_chunk({"content": "Done."}), openai_chunk({}, finish="stop"), (None, "[DONE]")])

    code = cli.main(
        [
            "-m",
            "openai:gpt-test",
            "--base-url",
            server.url + "/v1",
            "--thinking",
            "high",
            "--no-session",
            "--cwd",
            str(tmp_path),
            "-p",
            "set x to 2",
        ]
    )
    out = capsys.readouterr()
    assert code == 0, out.err
    assert out.out.strip() == "Done."
    assert (tmp_path / "a.py").read_text() == "x = 2\n"

    path, first = server.requests[0]
    assert path == "/v1/chat/completions"
    assert first["reasoning_effort"] == "high"
    assert first["messages"][0]["role"] == "system"
    assert first["tools"][0]["type"] == "function"

    msgs = server.requests[1][1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "tool"]
    assert msgs[2]["content"] == "Editing."
    assert [c["function"]["name"] for c in msgs[2]["tool_calls"]] == ["edit", "bash"]
    assert json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])["path"] == "a.py"
    assert msgs[3]["tool_call_id"] == "call_1" and "Applied 1 edit" in msgs[3]["content"]
    assert msgs[4]["tool_call_id"] == "call_2"


def test_openai_invalid_tool_json_reported_to_model(server, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    server.sse(
        [
            openai_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path": '},
                        }
                    ]
                }
            ),
            openai_chunk({}, finish="tool_calls"),
            (None, "[DONE]"),
        ]
    )
    server.sse([openai_chunk({"content": "oops"}), openai_chunk({}, finish="stop"), (None, "[DONE]")])
    assert (
        cli.main(
            [
                "-m",
                "openai:x",
                "--base-url",
                server.url + "/v1",
                "--no-session",
                "--cwd",
                str(tmp_path),
                "-p",
                "go",
            ]
        )
        == 0
    )
    tool_msg = server.requests[1][1]["messages"][-1]
    assert tool_msg["role"] == "tool" and "Invalid JSON" in tool_msg["content"]
