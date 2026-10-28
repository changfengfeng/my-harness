"""System prompt construction and project context (AGENTS.md / CLAUDE.md) loading."""

from __future__ import annotations

import platform
from datetime import date
from pathlib import Path
from typing import Any

from ..agent.types import AgentTool

CONTEXT_FILES = ("AGENTS.md", "CLAUDE.md")

TOOL_GUIDELINES = {
    "read": "Use read to examine files instead of cat or sed.",
    "edit": "Use edit for precise changes; old_text must match the file exactly.",
    "write": "Use write only for new files or complete rewrites.",
    "grep": "Use grep to search file contents instead of bash grep.",
}


def load_context_files(cwd: Path) -> list[tuple[Path, str]]:
    """Collect context files: the global one first, then one per directory from the root down to cwd."""
    candidates: list[Path] = []
    for directory in [Path.home() / ".my-harness", *reversed([cwd, *cwd.parents])]:
        match = next((directory / n for n in CONTEXT_FILES if (directory / n).is_file()), None)
        if match:
            candidates.append(match)
    found: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for path in candidates:
        if path.resolve() not in seen:
            seen.add(path.resolve())
            found.append((path, path.read_text(encoding="utf-8")))
    return found


def build_system_prompt(
    cwd: Path,
    tools: list[AgentTool[Any]],
    custom_prompt: str | None = None,
    append: str | None = None,
    context_files: list[tuple[Path, str]] | None = None,
) -> str:
    tool_lines = "\n".join(f"- {t.name}: {t.description.split('. ')[0]}" for t in tools)
    names = {t.name for t in tools}
    rules = [TOOL_GUIDELINES[n] for n in TOOL_GUIDELINES if n in names]
    rules += ["Be concise in your responses.", "Show file paths clearly when working with files."]

    base = custom_prompt or (
        "You are an expert coding assistant operating inside my-harness, a coding agent harness. "
        "You help users by reading files, executing commands, editing code, and writing new files."
    )
    parts = [
        base,
        f"Available tools:\n{tool_lines}",
        "Guidelines:\n" + "\n".join(f"- {r}" for r in rules),
    ]
    if append:
        parts.append(append)
    files = context_files if context_files is not None else load_context_files(cwd)
    if files:
        parts.append(
            "Project-specific instructions and guidelines:\n\n"
            + "\n\n".join(
                f'<project_instructions path="{p}">\n{c}\n</project_instructions>' for p, c in files
            )
        )
    parts.append(
        f"Current date: {date.today().isoformat()}\nPlatform: {platform.system()}\nWorking directory: {cwd}"
    )
    return "\n\n".join(parts)
