"""Code intelligence for large repositories: a persistent symbol / reference index, ranked search,
relationship queries, agent tools and the `locate_code` sub-agent."""

from .index import CodeIndex, IndexStats, default_index_path
from .locator import LocateCodeTool, locate
from .queries import CodeQueries, Hit
from .tools import CodeIntel, create_code_tools

CODE_TOOL_NAMES = ["code_search", "find_symbol", "find_references", "related_code", "repo_map"]

__all__ = [
    "CODE_TOOL_NAMES",
    "CodeIndex",
    "CodeIntel",
    "CodeQueries",
    "Hit",
    "IndexStats",
    "LocateCodeTool",
    "create_code_tools",
    "default_index_path",
    "locate",
]
