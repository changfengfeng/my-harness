import pytest

pytest.importorskip("fastapi")
from fastapi import WebSocketDisconnect  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from my_harness.ai import FauxProvider, register_provider  # noqa: E402
from my_harness.ai.models import get_model  # noqa: E402
from my_harness.ai.types import ToolCall  # noqa: E402
from my_harness.web.server import WebConfig, create_app  # noqa: E402


def make_client(tmp_path, password="pw"):
    cfg = WebConfig(model=get_model("faux"), api_key=None, password=password, root=tmp_path / "r")
    return TestClient(create_app(cfg)), cfg


def drain(ws, until="agent_end"):
    events = []
    while True:
        e = ws.receive_json()
        events.append(e)
        if e["t"] == until:
            return events


def test_login_required(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/api/config").json()["needs_login"] is True
    assert client.post("/api/login", json={"password": "wrong"}).status_code == 401
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws") as ws:
        ws.receive_json()
    assert client.post("/api/login", json={"password": "pw"}).status_code == 200
    assert client.get("/api/config").json()["needs_login"] is False
    assert "my-harness" in client.get("/").text


def test_chat_with_tool_and_workspace_cleanup(tmp_path):
    faux = FauxProvider(
        [ToolCall(id="c1", name="write", arguments={"path": "a.txt", "content": "hi"}), "done"]
    )
    register_provider("faux", faux)
    client, _ = make_client(tmp_path)
    client.post("/api/login", json={"password": "pw"})
    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["t"] == "ready"
        ws.send_json({"type": "prompt", "text": "write a file"})
        events = drain(ws)
        kinds = [e["t"] for e in events]
        assert "tool_start" in kinds and "tool_end" in kinds
        end = next(e for e in events if e["t"] == "tool_end")
        assert end["is_error"] is False and "a.txt" in end["text"]
        assert "".join(e["d"] for e in events if e["t"] == "text") == "done"
        dirs = list((tmp_path / "r").iterdir())
        assert len(dirs) == 1 and (dirs[0] / "a.txt").read_text() == "hi"
    assert list((tmp_path / "r").iterdir()) == []


def test_reset_clears_history(tmp_path):
    faux = FauxProvider(["one", "two"])
    register_provider("faux", faux)
    client, _ = make_client(tmp_path, password=None)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "prompt", "text": "a"})
        drain(ws)
        ws.send_json({"type": "reset"})
        assert ws.receive_json()["t"] == "reset"
        ws.send_json({"type": "prompt", "text": "b"})
        drain(ws)
    assert [m.role for m in faux.requests[-1].messages] == ["user"]


def test_bash_does_not_see_server_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("MYH_WEB_API_KEY", "sk-secret-value")
    faux = FauxProvider([ToolCall(id="c1", name="bash", arguments={"command": "env"}), "ok"])
    register_provider("faux", faux)
    client, _ = make_client(tmp_path, password=None)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "prompt", "text": "env"})
        end = next(e for e in drain(ws) if e["t"] == "tool_end")
    assert "sk-secret-value" not in end["text"] and "HOME=" in end["text"]


def test_from_env_removes_secrets_from_process_env(monkeypatch):
    import os

    monkeypatch.setenv("MYH_WEB_PASSWORD", "pw")
    monkeypatch.setenv("MYH_WEB_API_KEY", "sk-x")
    monkeypatch.setenv("MYH_WEB_MODEL", "openai:deepseek-chat")
    cfg = WebConfig.from_env()
    assert cfg.api_key == "sk-x" and cfg.password == "pw"
    assert cfg.model.base_url == "https://api.deepseek.com/v1"
    assert "MYH_WEB_API_KEY" not in os.environ and "MYH_WEB_PASSWORD" not in os.environ
