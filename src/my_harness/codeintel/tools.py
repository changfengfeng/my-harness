"""Agent tools exposing the code index."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..agent.types import AgentTool, AgentToolResult, ToolError, UpdateCallback
from ..coding.tools.truncate import DEFAULT_MAX_BYTES
from .index import CodeIndex
from .queries import CodeQueries, Hit

REFRESH_INTERVAL = 2.0


class CodeIntel:
    """An index plus its query layer, built lazily and refreshed (incrementally) before queries."""

    def __init__(self, root: Path, db_path: Path | None = None):
        self.root = root.resolve()
        self._db_path = db_path
        self._index: CodeIndex | None = None
        self._queries: CodeQueries | None = None
        self._lock = asyncio.Lock()

    @property
    def index(self) -> CodeIndex:
        if self._index is None:
            self._index = CodeIndex(self.root, self._db_path)
        return self._index

    def ready_sync(self) -> CodeQueries:
        self.index.refresh(min_interval=REFRESH_INTERVAL)
        if self._queries is None:
            self._queries = CodeQueries(self.index)
        self._queries.reset_cache()
        return self._queries

    async def ready(self) -> CodeQueries:
        async with self._lock:
            return await asyncio.to_thread(self.ready_sync)

    async def run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        q = await self.ready()
        return await asyncio.to_thread(fn, q, *args, **kwargs)


def _clip(text: str) -> str:
    if len(text.encode()) <= DEFAULT_MAX_BYTES:
        return text
    return text.encode()[:DEFAULT_MAX_BYTES].decode(errors="ignore") + "\n[output truncated]"


def format_hit(h: Hit, show_snippet: bool = True) -> str:
    lines = [h.header()]
    if h.signature and h.kind != "file" and h.signature != h.qualname:
        lines.append(f"    {h.signature[:200]}")
    if h.doc:
        lines.append(f"    \u201c{' '.join(h.doc.split())[:160]}\u201d")
    if show_snippet:
        lines.extend(f"    {ln:>5}| {text}" for ln, text in h.snippet)
    return "\n".join(lines)


KindName = Literal[
    "class",
    "interface",
    "function",
    "method",
    "type",
    "enum",
    "impl",
    "module",
    "variable",
    "file",
    "section",
]


# ---------------------------------------------------------------------------


class CodeSearchParams(BaseModel):
    query: str = Field(
        description="What to look for, as code vocabulary: identifiers and English keywords, e.g. "
        "'csrf token check middleware' or 'retry backoff http client'. Translate non-English requests first."
    )
    path: str | None = Field(
        default=None, description="Restrict to a directory prefix or glob, e.g. 'src/auth' or '**/*.go'"
    )
    kind: list[KindName] | None = Field(default=None, description="Only these symbol kinds")
    lang: str | None = Field(default=None, description="Only this language, e.g. python, typescript, go")
    include_tests: bool | None = Field(
        default=None,
        description="true/false to force tests in/out; default: down-ranked unless the query mentions tests",
    )
    limit: int = Field(default=10, ge=1, le=50)


class CodeSearchTool(AgentTool[CodeSearchParams]):
    name = "code_search"
    description = (
        "Ranked search over an index of every function, class, method and file in the repository (BM25 over "
        "names, paths, signatures, docstrings and bodies, boosted by how central a file is). Returns "
        "path:start-end locations with signature and the best-matching lines. Much faster and better ranked "
        "than grep for 'where is X handled' questions on large codebases."
    )
    Params = CodeSearchParams

    def __init__(self, intel: CodeIntel):
        self.intel = intel

    async def execute(
        self,
        call_id: str,
        params: CodeSearchParams,
        signal: asyncio.Event,
        on_update: UpdateCallback | None = None,
    ) -> AgentToolResult:
        hits: list[Hit] = await self.intel.run(
            lambda q: q.search(
                params.query,
                limit=params.limit,
                path=params.path,
                kinds=list(params.kind) if params.kind else None,
                lang=params.lang,
                include_tests=params.include_tests,
            )
        )
        if not hits:
            return AgentToolResult.text(
                "No matches. Try other synonyms or identifiers, fewer words, or grep for an exact string."
            )
        text = "\n".join(format_hit(h) for h in hits)
        return AgentToolResult.text(_clip(text), details={"hits": [h.location for h in hits]})


class FindSymbolParams(BaseModel):
    name: str = Field(
        description="Symbol name, qualified name ('Class.method', 'pkg::Type') or glob ('*Handler')"
    )
    kind: KindName | None = None
    path: str | None = Field(default=None, description="Directory prefix or glob")
    limit: int = Field(default=20, ge=1, le=100)


class FindSymbolTool(AgentTool[FindSymbolParams]):
    name = "find_symbol"
    description = (
        "Go to definition: find where a class / function / method / type is defined. Exact match first, "
        "then case-insensitive, then substring. Returns location, signature and doc."
    )
    Params = FindSymbolParams

    def __init__(self, intel: CodeIntel):
        self.intel = intel

    async def execute(
        self,
        call_id: str,
        params: FindSymbolParams,
        signal: asyncio.Event,
        on_update: UpdateCallback | None = None,
    ) -> AgentToolResult:
        hits: list[Hit] = await self.intel.run(
            lambda q: q.find_symbol(params.name, kind=params.kind, path=params.path, limit=params.limit)
        )
        if not hits:
            return AgentToolResult.text(f"No definition found for {params.name!r}. Try code_search or grep.")
        return AgentToolResult.text(
            _clip("\n".join(format_hit(h, show_snippet=False) for h in hits)),
            details={"hits": [h.location for h in hits]},
        )


class FindReferencesParams(BaseModel):
    name: str = Field(description="Identifier to find usages of (a qualified name uses its last part)")
    path: str | None = Field(default=None, description="Directory prefix or glob")
    include_tests: bool = True
    limit: int = Field(default=60, ge=1, le=300)


class FindReferencesTool(AgentTool[FindReferencesParams]):
    name = "find_references"
    description = (
        "Find usages of an identifier across the repository (call sites, imports, subclassing, attribute "
        "access), grouped by file with the enclosing function for each line. Uses the index, so it is fast "
        "even on very large repositories; matching is by name, so check ambiguous names."
    )
    Params = FindReferencesParams

    def __init__(self, intel: CodeIntel):
        self.intel = intel

    async def execute(
        self,
        call_id: str,
        params: FindReferencesParams,
        signal: asyncio.Event,
        on_update: UpdateCallback | None = None,
    ) -> AgentToolResult:
        refs, total = await self.intel.run(
            lambda q: q.references(
                params.name, path=params.path, limit=params.limit, include_tests=params.include_tests
            )
        )
        if not refs:
            return AgentToolResult.text(f"No references to {params.name!r} found.")
        out: list[str] = []
        last = None
        for h, ln, text in refs:
            if h.path != last:
                out.append(f"{h.path}")
                last = h.path
            where = h.qualname if h.kind != "file" else "<module>"
            out.append(f"  {ln:>5}  ({where})  {text}")
        files = len({h.path for h, _, _ in refs})
        out.append(f"\n[{len(refs)} lines in {files} files; {total} referencing scopes in total]")
        return AgentToolResult.text(_clip("\n".join(out)), details={"lines": len(refs), "scopes": total})


class RelatedCodeParams(BaseModel):
    target: str = Field(description="A symbol name / qualified name, a file path, or 'path:line'")
    limit: int = Field(default=12, ge=1, le=50, description="Max entries per section")


class RelatedCodeTool(AgentTool[RelatedCodeParams]):
    name = "related_code"
    description = (
        "Show the code related to a symbol or file: its definition and members, what it calls/uses "
        "(resolved to definitions), who calls/uses it, subclasses/implementations, imports, importers, and "
        "tests. Use it to expand from a found location to everything a change would touch."
    )
    Params = RelatedCodeParams

    def __init__(self, intel: CodeIntel):
        self.intel = intel

    async def execute(
        self,
        call_id: str,
        params: RelatedCodeParams,
        signal: asyncio.Event,
        on_update: UpdateCallback | None = None,
    ) -> AgentToolResult:
        text = await self.intel.run(lambda q: related_report(q, params.target, params.limit))
        if text is None:
            raise ToolError(
                f"Could not resolve {params.target!r} to a file or symbol. Try find_symbol or code_search."
            )
        return AgentToolResult.text(_clip(text))


def related_report(q: CodeQueries, target: str, limit: int) -> str | None:
    hits = q.resolve_target(target)
    if not hits:
        return None
    h = hits[0]
    out: list[str] = []
    if len(hits) > 1:
        out.append(f"{len(hits)} definitions match {target!r}; showing the first. Others:")
        out.extend(f"  {o.header()}" for o in hits[1:limit])
        out.append("")
    out.append("Definition:\n" + format_hit(h, show_snippet=False))
    if h.kind == "file":
        outline = q.outline(h.file_id)
        if outline:
            out.append(f"\nOutline ({len(outline)} symbols):")
            out.extend(f"  {o.start:>5}: [{o.kind}] {o.qualname}" for o in outline[: limit * 4])
    else:
        if h.kind in ("class", "interface", "impl", "enum", "module"):
            members = [o for o in q.outline(h.file_id, max_depth=99) if o.start > h.start and o.end <= h.end]
            if members:
                out.append(f"\nMembers ({len(members)}):")
                out.extend(f"  {o.start:>5}: {o.signature[:120]}" for o in members[: limit * 3])
        rows = q.index.query("SELECT parent_id FROM symbols WHERE id = ?", [h.id])
        if rows and rows[0][0]:
            parent = q.symbol_by_id(rows[0][0])
            if parent:
                out.append(f"\nDefined in: {parent.header()}")
        callees = q.callees(h, limit)
        if callees:
            out.append("\nUses (resolved definitions):")
            out.extend(f"  {c.header()}" for c in callees)
        callers, total = q.callers(h, limit)
        if callers:
            out.append(f"\nUsed by ({total} scopes mention {h.name!r}; most likely first):")
            out.extend(f"  {c.header()}" for c in callers)
        impls = q.overrides(h, limit)
        if impls:
            out.append("\nOverridden / implemented in:")
            out.extend(f"  {c.header()}" for c in impls)
        subs = q.subtypes(h, limit)
        if subs:
            out.append("\nSubclasses / implementations:")
            out.extend(f"  {s.header()}" for s in subs)
    imports, external = q.imports_of(h.file_id)
    importers, n_importers = q.importers_of(h.file_id, limit)
    out.append(f"\nFile {h.path}:")
    if imports:
        out.append(f"  imports ({len(imports)}): " + ", ".join(imports[: limit * 2]))
    if external:
        out.append("  external imports: " + ", ".join(external[:limit]))
    if importers:
        out.append(f"  imported by ({n_importers}): " + ", ".join(importers))
    tests = q.related_tests(h, limit)
    if tests:
        out.append("\nTests:\n" + "\n".join(f"  {t}" for t in tests))
    return "\n".join(out)


class RepoMapParams(BaseModel):
    path: str | None = Field(default=None, description="Directory to map (default: whole repository)")
    max_files: int = Field(default=40, ge=1, le=200)
    include_tests: bool = False


class RepoMapTool(AgentTool[RepoMapParams]):
    name = "repo_map"
    description = (
        "Overview of the repository or a directory: size per language, directory layout with line counts, "
        "and the most central files (PageRank over the reference graph) with their key symbols. Start here "
        "to understand an unfamiliar codebase."
    )
    Params = RepoMapParams

    def __init__(self, intel: CodeIntel):
        self.intel = intel

    async def execute(
        self,
        call_id: str,
        params: RepoMapParams,
        signal: asyncio.Event,
        on_update: UpdateCallback | None = None,
    ) -> AgentToolResult:
        text = await self.intel.run(
            lambda q: q.repo_map(params.path, max_files=params.max_files, include_tests=params.include_tests)
        )
        return AgentToolResult.text(_clip(text))


def create_code_tools(intel: CodeIntel) -> list[AgentTool[Any]]:
    return [
        CodeSearchTool(intel),
        FindSymbolTool(intel),
        FindReferencesTool(intel),
        RelatedCodeTool(intel),
        RepoMapTool(intel),
    ]
