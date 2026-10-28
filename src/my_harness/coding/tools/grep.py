from __future__ import annotations

import asyncio
import fnmatch
import re
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from ...agent.types import AgentTool, AgentToolResult, ToolError, UpdateCallback
from .paths import resolve
from .truncate import DEFAULT_MAX_BYTES, clip_line

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mypy_cache"}


class GrepParams(BaseModel):
    pattern: str = Field(description="Search pattern (regex or literal string)")
    path: str | None = Field(default=None, description="Directory or file to search (default: cwd)")
    glob: str | None = Field(default=None, description="Filter files by glob pattern, e.g. '*.py'")
    ignore_case: bool = Field(default=False, description="Case-insensitive search")
    literal: bool = Field(default=False, description="Treat pattern as a literal string instead of regex")
    limit: int = Field(default=100, ge=1, description="Maximum number of matches to return")


class GrepTool(AgentTool[GrepParams]):
    name = "grep"
    description = (
        "Search file contents for a pattern. Returns matching lines as path:line:text. Uses ripgrep when "
        f"available (respecting .gitignore). Output is truncated to `limit` matches or {DEFAULT_MAX_BYTES // 1024}KB."
    )
    Params = GrepParams

    def __init__(self, cwd: Path):
        self.cwd = cwd

    async def execute(
        self, call_id: str, params: GrepParams, signal: asyncio.Event, on_update: UpdateCallback | None = None
    ) -> AgentToolResult:
        root = resolve(self.cwd, params.path or ".")
        if not root.exists():
            raise ToolError(f"Path not found: {params.path}")
        lines = await (
            self._rg(root, params) if shutil.which("rg") else asyncio.to_thread(self._py, root, params)
        )
        if not lines:
            return AgentToolResult.text("No matches found")
        out, size = [], 0
        for line in lines[: params.limit]:
            line = clip_line(line)
            size += len(line) + 1
            if size > DEFAULT_MAX_BYTES:
                break
            out.append(line)
        text = "\n".join(out)
        if len(out) < len(lines):
            text += f"\n\n[Showing {len(out)} matches; more exist. Narrow the pattern or raise limit.]"
        return AgentToolResult.text(text, details={"matches": len(lines)})

    async def _rg(self, root: Path, p: GrepParams) -> list[str]:
        args = ["rg", "--line-number", "--no-heading", "--color=never", "--max-count", str(p.limit)]
        if p.ignore_case:
            args.append("-i")
        if p.literal:
            args.append("-F")
        if p.glob:
            args += ["--glob", p.glob]
        args += ["--", p.pattern, str(root)]
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=self.cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        if proc.returncode not in (0, 1):
            raise ToolError(err.decode(errors="replace").strip() or "ripgrep failed")
        return [self._rel(line) for line in out.decode(errors="replace").splitlines()][: p.limit + 1]

    def _py(self, root: Path, p: GrepParams) -> list[str]:
        try:
            rx = re.compile(
                re.escape(p.pattern) if p.literal else p.pattern, re.IGNORECASE if p.ignore_case else 0
            )
        except re.error as exc:
            raise ToolError(f"Invalid regex: {exc}") from exc
        files = (
            [root]
            if root.is_file()
            else (f for f in sorted(root.rglob("*")) if f.is_file() and not SKIP_DIRS.intersection(f.parts))
        )
        results: list[str] = []
        for f in files:
            if p.glob and not fnmatch.fnmatch(f.name, p.glob) and not fnmatch.fnmatch(str(f), p.glob):
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    results.append(f"{self._rel(str(f))}:{n}:{line}")
                    if len(results) > p.limit:
                        return results
        return results

    def _rel(self, line: str) -> str:
        prefix = str(self.cwd) + "/"
        return line[len(prefix) :] if line.startswith(prefix) else line
