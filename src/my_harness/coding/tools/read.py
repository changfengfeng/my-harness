from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from pydantic import BaseModel, Field

from ...agent.types import AgentTool, AgentToolResult, ToolError, UpdateCallback
from ...ai.types import ImageContent, TextContent
from .paths import resolve
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, clip_line, truncate_head

IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class ReadParams(BaseModel):
    path: str = Field(description="Path to the file to read (relative or absolute)")
    offset: int | None = Field(
        default=None, ge=1, description="Line number to start reading from (1-indexed)"
    )
    limit: int | None = Field(default=None, ge=1, description="Maximum number of lines to read")


class ReadTool(AgentTool[ReadParams]):
    name = "read"
    description = (
        "Read the contents of a file. Supports text files and images (png, jpg, gif, webp). "
        f"Text output is truncated to {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB, whichever "
        "comes first. Use offset/limit for large files and keep reading with offset until complete."
    )
    Params = ReadParams

    def __init__(self, cwd: Path):
        self.cwd = cwd

    async def execute(
        self, call_id: str, params: ReadParams, signal: asyncio.Event, on_update: UpdateCallback | None = None
    ) -> AgentToolResult:
        path = resolve(self.cwd, params.path)
        if not path.is_file():
            raise ToolError(f"File not found: {params.path}")
        mime = IMAGE_TYPES.get(path.suffix.lower())
        if mime:
            data = base64.b64encode(path.read_bytes()).decode()
            return AgentToolResult(
                content=[
                    TextContent(text=f"Image file {params.path}"),
                    ImageContent(data=data, mime_type=mime),
                ]
            )
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.split("\n")
        if len(lines) > 1 and lines[-1] == "":
            lines.pop()  # a trailing newline does not start another line
        start = (params.offset or 1) - 1
        if start >= len(lines) and lines != [""]:
            raise ToolError(f"Offset {params.offset} is beyond end of file ({len(lines)} lines)")
        end = start + params.limit if params.limit else len(lines)
        selected = "\n".join(clip_line(line, 2000) for line in lines[start:end])
        t = truncate_head(selected)
        out = t.text
        next_line = start + t.shown_lines + 1
        if t.truncated:
            out += f"\n\n[Showing lines {start + 1}-{start + t.shown_lines} of {len(lines)}. Use offset={next_line} to continue.]"
        elif end < len(lines):
            out += f"\n\n[{len(lines) - end} more lines in file. Use offset={end + 1} to continue.]"
        return AgentToolResult.text(out, details={"path": str(path), "lines": len(lines)})
