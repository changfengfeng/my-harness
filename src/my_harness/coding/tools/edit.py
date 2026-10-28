from __future__ import annotations

import asyncio
import difflib
from pathlib import Path

from pydantic import BaseModel, Field

from ...agent.types import AgentTool, AgentToolResult, ToolError, UpdateCallback
from .paths import resolve


class Replacement(BaseModel):
    old_text: str = Field(
        description="Exact text for one targeted replacement. Must be unique in the original file and must not "
        "overlap with any other edits[].old_text in the same call."
    )
    new_text: str = Field(description="Replacement text for this targeted edit.")


class EditParams(BaseModel):
    path: str = Field(description="Path to the file to edit (relative or absolute)")
    edits: list[Replacement] = Field(
        min_length=1,
        description="One or more targeted replacements, each matched against the original file. Do not include "
        "overlapping or nested edits; merge nearby changes into one edit instead.",
    )


def apply_edits(original: str, edits: list[Replacement]) -> str:
    spans: list[tuple[int, int, str]] = []
    for i, e in enumerate(edits):
        if not e.old_text:
            raise ToolError(f"edits[{i}].old_text must not be empty")
        count = original.count(e.old_text)
        if count == 0:
            raise ToolError(
                f"edits[{i}].old_text not found in file. It must match exactly, including whitespace."
            )
        if count > 1:
            raise ToolError(
                f"edits[{i}].old_text matches {count} times; add surrounding context to make it unique."
            )
        start = original.index(e.old_text)
        spans.append((start, start + len(e.old_text), e.new_text))
    spans.sort()
    for (_, a_end, _), (b_start, _, _) in zip(spans, spans[1:], strict=False):
        if b_start < a_end:
            raise ToolError("Edits overlap; merge them into a single edit.")
    out, pos = [], 0
    for start, end, new in spans:
        out.append(original[pos:start])
        out.append(new)
        pos = end
    out.append(original[pos:])
    return "".join(out)


class EditTool(AgentTool[EditParams]):
    name = "edit"
    description = (
        "Edit a single file using exact text replacement. Every edits[].old_text must match a unique, "
        "non-overlapping region of the original file. Do not include large unchanged regions just to connect "
        "distant changes."
    )
    Params = EditParams

    def __init__(self, cwd: Path):
        self.cwd = cwd

    async def execute(
        self, call_id: str, params: EditParams, signal: asyncio.Event, on_update: UpdateCallback | None = None
    ) -> AgentToolResult:
        path = resolve(self.cwd, params.path)
        if not path.is_file():
            raise ToolError(f"File not found: {params.path}")
        original = path.read_text(encoding="utf-8")
        updated = apply_edits(original, params.edits)
        path.write_text(updated, encoding="utf-8")
        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                params.path,
                params.path,
            )
        )
        return AgentToolResult.text(
            f"Applied {len(params.edits)} edit(s) to {params.path}", details={"path": str(path), "diff": diff}
        )
