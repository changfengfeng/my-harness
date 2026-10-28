"""FastAPI app exposing the coding agent in the browser.

Each WebSocket connection gets its own Agent and a throwaway workspace directory that is deleted on
disconnect. Access is protected by a shared password (MYH_WEB_PASSWORD).

Environment:
  MYH_WEB_PASSWORD      required unless MYH_WEB_OPEN=1
  MYH_WEB_MODEL         provider:model (default openai:deepseek-chat)
  MYH_WEB_BASE_URL      API base URL (default https://api.deepseek.com/v1 for deepseek-* models)
  MYH_WEB_API_KEY       API key (falls back to OPENAI_API_KEY / ANTHROPIC_API_KEY)
  MYH_WEB_ROOT          parent directory for workspaces (default: system temp dir)
  MYH_WEB_MAX_SESSIONS  concurrent connections allowed (default 10)
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from ..agent import Agent
from ..agent.types import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    MessageEnd,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
)
from ..ai.models import Model, get_model
from ..ai.types import AssistantMessage, StreamOptions, TextContent, TextDeltaEvent, ThinkingDeltaEvent
from ..coding.system_prompt import build_system_prompt
from ..coding.tools import BashTool, create_tools

STATIC = Path(__file__).parent / "static"
COOKIE = "myh_auth"
SECRET_ENV = ("MYH_WEB_PASSWORD", "MYH_WEB_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def sandbox_env(workspace: Path) -> dict[str, str]:
    """Minimal environment for agent shell commands, so secrets in the server's env never leak."""
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(workspace),
        "LANG": "C.UTF-8",
        "TERM": "dumb",
    }


@dataclass
class WebConfig:
    model: Model
    api_key: str | None
    password: str | None
    root: Path
    max_sessions: int = 10
    secret: bytes = field(default_factory=lambda: secrets.token_bytes(32))

    @classmethod
    def from_env(cls) -> WebConfig:
        spec = os.environ.get("MYH_WEB_MODEL", "openai:deepseek-chat")
        base_url = os.environ.get("MYH_WEB_BASE_URL")
        if base_url is None and spec.partition(":")[2].startswith("deepseek"):
            base_url = "https://api.deepseek.com/v1"
        password = os.environ.get("MYH_WEB_PASSWORD") or None
        if password is None and os.environ.get("MYH_WEB_OPEN") != "1":
            raise SystemExit("Set MYH_WEB_PASSWORD (or MYH_WEB_OPEN=1 to run without a password).")
        api_key = os.environ.get("MYH_WEB_API_KEY") or None
        cfg = cls(
            model=get_model(spec, base_url),
            api_key=None,
            password=password,
            root=Path(os.environ.get("MYH_WEB_ROOT") or tempfile.gettempdir()) / "myh-web",
            max_sessions=int(os.environ.get("MYH_WEB_MAX_SESSIONS", "10")),
        )
        if api_key is None:
            api_key = os.environ.get(
                "OPENAI_API_KEY" if cfg.model.provider == "openai" else "ANTHROPIC_API_KEY"
            )
        cfg.api_key = api_key
        for name in SECRET_ENV:
            os.environ.pop(name, None)
        return cfg

    def token(self) -> str:
        return hmac.new(self.secret, b"authorized", hashlib.sha256).hexdigest()

    def authorized(self, cookie: str | None) -> bool:
        return self.password is None or (cookie is not None and hmac.compare_digest(cookie, self.token()))


def _tool_text(result: Any) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


def to_wire(event: AgentEvent) -> dict[str, Any] | None:
    """Compact, UI-oriented view of agent events."""
    if isinstance(event, AgentStart):
        return {"t": "agent_start"}
    if isinstance(event, AgentEnd):
        return {"t": "agent_end"}
    if isinstance(event, MessageUpdate):
        if isinstance(event.event, TextDeltaEvent):
            return {"t": "text", "d": event.event.delta}
        if isinstance(event.event, ThinkingDeltaEvent):
            return {"t": "thinking", "d": event.event.delta}
        return None
    if isinstance(event, MessageEnd) and isinstance(event.message, AssistantMessage):
        m = event.message
        return {
            "t": "assistant_end",
            "stop_reason": m.stop_reason,
            "error": m.error_message,
            "usage": m.usage.model_dump(),
        }
    if isinstance(event, ToolExecutionStart):
        return {"t": "tool_start", "id": event.tool_call_id, "name": event.tool_name, "args": event.args}
    if isinstance(event, ToolExecutionEnd):
        text = _tool_text(event.result)
        if len(text) > 20_000:
            text = text[:20_000] + "\n... (truncated)"
        return {"t": "tool_end", "id": event.tool_call_id, "is_error": event.is_error, "text": text}
    return None


def create_app(config: WebConfig | None = None) -> FastAPI:
    cfg = config or WebConfig.from_env()
    app = FastAPI(title="my-harness", docs_url=None, redoc_url=None)
    active: set[int] = set()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    async def config_info(request: Request) -> dict[str, Any]:
        return {"model": str(cfg.model), "needs_login": not cfg.authorized(request.cookies.get(COOKIE))}

    @app.post("/api/login")
    async def login(request: Request) -> JSONResponse:
        body = await request.json()
        given = str(body.get("password", ""))
        if cfg.password is not None and not hmac.compare_digest(given.encode(), cfg.password.encode()):
            await asyncio.sleep(1)
            return JSONResponse({"ok": False, "error": "密码错误"}, status_code=401)
        resp = JSONResponse({"ok": True})
        resp.set_cookie(
            COOKIE,
            cfg.token(),
            httponly=True,
            samesite="strict",
            max_age=7 * 86400,
            secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https",
        )
        return resp

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        if not cfg.authorized(socket.cookies.get(COOKIE)):
            await socket.close(code=4401)
            return
        if len(active) >= cfg.max_sessions:
            await socket.close(code=4429)
            return
        await socket.accept()
        active.add(id(socket))
        cfg.root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="ws-", dir=cfg.root))
        tools = create_tools(workspace)
        for tool in tools:
            if isinstance(tool, BashTool):
                tool.env = sandbox_env(workspace)
        agent = Agent(
            cfg.model,
            system_prompt=build_system_prompt(workspace, tools),
            tools=tools,
            options=StreamOptions(api_key=cfg.api_key),
        )
        send_lock = asyncio.Lock()

        async def send(payload: dict[str, Any]) -> None:
            async with send_lock:
                with contextlib.suppress(Exception):
                    await socket.send_text(json.dumps(payload, ensure_ascii=False))

        async def forward(event: AgentEvent) -> None:
            if (wire := to_wire(event)) is not None:
                await send(wire)

        agent.subscribe(forward)
        await send({"t": "ready", "model": str(cfg.model), "workspace": workspace.name})
        running: asyncio.Task[Any] | None = None

        async def run(text: str) -> None:
            try:
                await agent.prompt(text)
            except Exception as exc:  # noqa: BLE001 - report anything to the browser
                await send({"t": "error", "error": str(exc)})

        try:
            while True:
                msg = json.loads(await socket.receive_text())
                kind = msg.get("type")
                if kind == "prompt" and (text := str(msg.get("text", "")).strip()):
                    if agent.state.is_streaming:
                        agent.follow_up(text)
                    else:
                        running = asyncio.create_task(run(text))
                elif kind == "abort":
                    agent.abort()
                elif kind == "reset":
                    agent.abort()
                    if running:
                        with contextlib.suppress(Exception):
                            await running
                    agent.reset()
                    await send({"t": "reset"})
        except (WebSocketDisconnect, json.JSONDecodeError):
            pass
        finally:
            agent.abort()
            if running:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(running, timeout=5)
            active.discard(id(socket))
            shutil.rmtree(workspace, ignore_errors=True)

    return app


def _protect_process_memory() -> None:
    """Make /proc/<pid>/environ and /proc/<pid>/mem unreadable to the agent's shell commands (same uid)."""
    if sys.platform == "linux":
        import ctypes

        pr_set_dumpable = 4
        ctypes.CDLL(None).prctl(pr_set_dumpable, 0, 0, 0, 0)


def main() -> None:
    import uvicorn

    _protect_process_memory()
    uvicorn.run(
        create_app(),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
