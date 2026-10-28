from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from ...agent.types import AgentTool, AgentToolResult, ToolError, UpdateCallback
from .paths import resolve


class LsParams(BaseModel):
    path: str | None = Field(default=None, description="Directory to list (default: cwd)")
    limit: int = Field(default=500, ge=1, description="Maximum number of entries to return")


class LsTool(AgentTool[LsParams]):
    name = "ls"
    description = "List directory contents, sorted alphabetically, with '/' suffix for directories."
    Params = LsParams

    def __init__(self, cwd: Path):
        self.cwd = cwd

    async def execute(
        self, call_id: str, params: LsParams, signal: asyncio.Event, on_update: UpdateCallback | None = None
    ) -> AgentToolResult:
        root = resolve(self.cwd, params.path or ".")
        if not root.is_dir():
            raise ToolError(f"Not a directory: {params.path}")
        entries = sorted(root.iterdir(), key=lambda p: p.name.lower())
        names = [e.name + ("/" if e.is_dir() else "") for e in entries[: params.limit]]
        text = "\n".join(names) or "(empty directory)"
        if len(entries) > params.limit:
            text += f"\n\n[{len(entries) - params.limit} more entries not shown]"
        return AgentToolResult.text(text)
