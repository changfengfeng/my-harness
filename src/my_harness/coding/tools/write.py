from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from ...agent.types import AgentTool, AgentToolResult, UpdateCallback
from .paths import resolve


class WriteParams(BaseModel):
    path: str = Field(description="Path to the file to write (relative or absolute)")
    content: str = Field(description="Content to write to the file")


class WriteTool(AgentTool[WriteParams]):
    name = "write"
    description = (
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
        "Automatically creates parent directories."
    )
    Params = WriteParams

    def __init__(self, cwd: Path):
        self.cwd = cwd

    async def execute(
        self,
        call_id: str,
        params: WriteParams,
        signal: asyncio.Event,
        on_update: UpdateCallback | None = None,
    ) -> AgentToolResult:
        path = resolve(self.cwd, params.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(params.content, encoding="utf-8")
        return AgentToolResult.text(
            f"Successfully wrote {len(params.content.encode())} bytes to {params.path}",
            details={"path": str(path)},
        )
