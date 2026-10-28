"""Output truncation shared by tools, so a single call cannot flood the context window."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024
MAX_LINE_CHARS = 500


@dataclass
class Truncation:
    text: str
    truncated: bool
    total_lines: int
    shown_lines: int


def _fit(lines: list[str], max_lines: int, max_bytes: int) -> list[str]:
    kept: list[str] = []
    size = 0
    for line in lines[:max_lines]:
        size += len(line.encode()) + 1
        if size > max_bytes:
            break
        kept.append(line)
    return kept


def truncate_head(
    text: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
) -> Truncation:
    """Keep the beginning of `text` (for file reads)."""
    lines = text.split("\n")
    kept = _fit(lines, max_lines, max_bytes)
    return Truncation("\n".join(kept), len(kept) < len(lines), len(lines), len(kept))


def truncate_tail(
    text: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
) -> Truncation:
    """Keep the end of `text` (for command output, where errors usually appear last)."""
    lines = text.split("\n")
    kept = list(reversed(_fit(list(reversed(lines)), max_lines, max_bytes)))
    return Truncation("\n".join(kept), len(kept) < len(lines), len(lines), len(kept))


def clip_line(line: str, max_chars: int = MAX_LINE_CHARS) -> str:
    return line if len(line) <= max_chars else line[:max_chars] + "... [line truncated]"
