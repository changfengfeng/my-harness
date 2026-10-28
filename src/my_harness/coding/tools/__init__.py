"""Built-in coding tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...agent.types import AgentTool
from .bash import BashTool
from .edit import EditTool
from .grep import GrepTool
from .ls import LsTool
from .read import ReadTool
from .write import WriteTool

ALL_TOOLS: dict[str, type[Any]] = {
    "read": ReadTool,
    "write": WriteTool,
    "edit": EditTool,
    "bash": BashTool,
    "grep": GrepTool,
    "ls": LsTool,
}
DEFAULT_TOOLS = ["read", "write", "edit", "bash", "grep", "ls"]


def create_tools(cwd: Path, names: list[str] | None = None) -> list[AgentTool[Any]]:
    tools = []
    for name in names or DEFAULT_TOOLS:
        if name not in ALL_TOOLS:
            raise ValueError(f"Unknown tool {name!r}. Available: {', '.join(ALL_TOOLS)}")
        tools.append(ALL_TOOLS[name](cwd))
    return tools


__all__ = [
    "ALL_TOOLS",
    "DEFAULT_TOOLS",
    "BashTool",
    "EditTool",
    "GrepTool",
    "LsTool",
    "ReadTool",
    "WriteTool",
    "create_tools",
]
