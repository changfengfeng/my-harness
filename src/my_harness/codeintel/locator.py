"""`locate_code`: a read-only sub-agent that maps a natural-language request to code locations.

It runs in its own context (so the exploration does not fill the caller's context window), uses the
index tools plus read/grep/ls, and returns a compact report of primary and related locations.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..agent import Agent
from ..agent.types import AgentEvent, AgentTool, AgentToolResult, ToolExecutionStart, TurnEnd, UpdateCallback
from ..ai.models import Model
from ..ai.stream import complete
from ..ai.types import AssistantMessage, Context, StreamOptions, ThinkingLevel, UserMessage
from ..coding.tools import GrepTool, LsTool, ReadTool
from .queries import CodeQueries, Hit
from .tools import CodeIntel, create_code_tools, format_hit

LocateMode = Literal["deep", "auto"]

LOCATOR_PROMPT = """You are a code-location specialist working on a large repository through a pre-built code index.
Your job: given a request from a user or developer (a feature, a bug, a question, a change they want to make),
find the exact code that implements it and the code related to it. You never modify files.

Method:
1. Translate the request into code vocabulary. Requests may be in Chinese or another language, but code
   identifiers are almost always English: list likely identifiers, synonyms and domain terms
   (e.g. "登录失败锁定" -> login, authenticate, failed attempts, lockout, throttle, rate limit).
2. If the repository is unfamiliar, call repo_map once (optionally on a subdirectory) to see the layout.
3. Search broadly first: issue several code_search calls in parallel with different phrasings / identifier
   guesses. Use find_symbol when you know or guess a name, grep for exact strings (error messages, config
   keys, routes, SQL, UI text).
4. Verify: read the most promising candidates (use offset/limit to read only the relevant range). Do not
   report code you have not looked at.
5. Expand: call related_code on the primary locations to find callers, callees, subclasses, importers and
   tests; use find_references for usages across the codebase.
6. Stop as soon as you are confident. Be economical: usually 4-10 tool rounds are enough.

Final answer format (answer in the same language as the request; keep identifiers and paths verbatim):

## Primary locations
- `path:start-end` `Symbol` - one line on what it does and why it matches the request
## Related code
- `path:line` `Symbol` - relationship (caller / callee / subclass / config / test / ...) and why it matters
## How it fits together
2-5 sentences describing the flow across these locations (entry point -> core logic -> side effects).
## Notes
Optional: ambiguities, alternative candidates, where a change would need to go.

Order locations by relevance. Every location must be real (taken from tool output) and include line numbers."""

AUTO_MAX_TURNS = 6
SEED_MESSAGE = """

Index candidates for this request (fused from several searches; may include noise):
{listing}

Start from these: read the most promising ones to verify, follow callers/callees with related_code if the
real cause may live elsewhere, search further only if none fit. Keep the answer short: at most 4 primary
and 6 related locations, one line each."""

BUDGET_MESSAGE = (
    "Tool budget reached. Stop searching and write the final answer now from what you have verified, "
    "in the required format."
)

EXPAND_PROMPT = """You turn a change request or question about a code repository into index search queries.
The request may be in any language; code is in English. Reply with JSON only:
{"queries": [...], "identifiers": [...]}
- queries: 4-6 short English keyword queries (2-5 words each) in the vocabulary the code would use:
  component names, likely function/type words, domain terms, synonyms. Cover different angles.
- identifiers: up to 6 exact identifier names (functions, types, flags) you would expect to exist."""


def _parse_expansion(text: str, request: str) -> tuple[list[str], list[str]]:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}

    def strings(key: str) -> list[str]:
        val = data.get(key)
        return (
            [str(x).strip() for x in val if isinstance(x, (str, int, float))] if isinstance(val, list) else []
        )

    queries = [q for q in strings("queries") if q][:8]
    idents = [i for i in strings("identifiers") if re.fullmatch(r"[\w.:$#-]+", i)][:8]
    return queries or [request], idents


def gather_candidates(
    q: CodeQueries, queries: list[str], identifiers: list[str], scope: str | None = None, limit: int = 24
) -> list[Hit]:
    """Run every query and identifier lookup against the index and fuse the rankings (reciprocal rank)."""
    lists: list[list[Hit]] = [q.search(text, limit=10, path=scope, per_file=2) for text in queries]
    lists += [q.find_symbol(name, path=scope, limit=4) for name in identifiers]
    fused: dict[int, float] = {}
    best: dict[int, Hit] = {}
    for hits in lists:
        for rank, h in enumerate(hits):
            fused[h.id] = fused.get(h.id, 0.0) + 1.0 / (6 + rank)
            if h.id not in best or (h.snippet and not best[h.id].snippet):
                best[h.id] = h
    order = sorted(fused, key=lambda i: -fused[i])
    return [best[i] for i in order[:limit]]


async def gather_for_request(
    intel: CodeIntel,
    model: Model,
    request: str,
    *,
    options: StreamOptions | None = None,
    scope: str | None = None,
) -> list[Hit]:
    """One model call to turn the request into code-vocabulary queries, then a fused index lookup."""
    await intel.ready()
    base = options or StreamOptions()
    expand = await complete(
        model,
        Context(system_prompt=EXPAND_PROMPT, messages=[UserMessage(content=request)]),
        base.model_copy(update={"max_tokens": 300, "thinking": "off"}),
    )
    if expand.stop_reason == "error":
        raise RuntimeError(expand.error_message or "query expansion failed")
    queries, identifiers = _parse_expansion(expand.text, request)
    hits: list[Hit] = await intel.run(gather_candidates, queries, identifiers, scope)
    return hits


async def locate(
    intel: CodeIntel,
    model: Model,
    request: str,
    *,
    options: StreamOptions | None = None,
    thinking: ThinkingLevel = "off",
    scope: str | None = None,
    max_turns: int = 14,
    on_event: Callable[[AgentEvent], Any] | None = None,
    mode: LocateMode = "deep",
) -> str:
    """Locate code for `request`. "deep": the multi-turn sub-agent that reads and follows references;
    "auto": the same sub-agent seeded with fused index candidates and a small turn budget."""
    seed = ""
    if mode == "auto":
        hits = await gather_for_request(intel, model, request, options=options, scope=scope)
        if hits:
            seed = SEED_MESSAGE.format(listing="\n\n".join(format_hit(h) for h in hits))
            max_turns = min(max_turns, AUTO_MAX_TURNS)
    await intel.ready()  # build / refresh the index up front
    cwd = intel.root
    tools: list[AgentTool[Any]] = [*create_code_tools(intel), ReadTool(cwd), GrepTool(cwd), LsTool(cwd)]
    stats = intel.index.stats()
    langs = ", ".join(f"{k} {v}" for k, v in list(stats.languages.items())[:5])
    system = (
        f"{LOCATOR_PROMPT}\n\nRepository: {cwd} ({stats.files} files, {stats.lines} lines, "
        f"{stats.symbols} symbols; {langs})."
    )
    agent = Agent(model, system_prompt=system, tools=tools, thinking=thinking, options=options)
    turns = 0

    def watch(event: AgentEvent) -> None:
        nonlocal turns
        if isinstance(event, TurnEnd):
            turns += 1
            if turns == max_turns:
                agent.steer(BUDGET_MESSAGE)
            elif turns >= max_turns + 3:
                agent.abort()

    agent.subscribe(watch)
    if on_event:
        agent.subscribe(on_event)
    prompt = request if not scope else f"{request}\n\n(Focus on: {scope})"
    prompt += seed
    await agent.prompt(prompt)
    last = next(
        (m for m in reversed(agent.state.messages) if isinstance(m, AssistantMessage) and m.text), None
    )
    if last is None:
        raise RuntimeError(agent.state.error or "the locator produced no answer")
    return last.text


class LocateCodeParams(BaseModel):
    request: str = Field(
        description="What to find, in natural language (any language): a feature, bug, behavior, or planned change"
    )
    scope: str | None = Field(default=None, description="Optional directory or component to focus on")
    mode: LocateMode = Field(
        default="deep",
        description="deep: multi-turn agent that reads code and follows references (most accurate); auto: the "
        "agent seeded with index candidates and a small turn budget (fewer tokens, slightly faster)",
    )


class LocateCodeTool(AgentTool[LocateCodeParams]):
    name = "locate_code"
    description = (
        "Delegate a code-location task to a read-only sub-agent that searches the repository index, reads "
        "candidates and follows references. Returns the primary locations (path:lines + symbol), related "
        "code (callers, callees, tests, config) and how they fit together. Use it for 'where is X implemented / "
        "what do I need to change for Y' on large codebases; it keeps exploration out of your context."
    )
    Params = LocateCodeParams

    def __init__(
        self,
        intel: CodeIntel,
        model: Model | Callable[[], Model],
        options: StreamOptions | None = None,
        max_turns: int = 14,
    ):
        self.intel = intel
        self.model = model
        self.options = options
        self.max_turns = max_turns

    async def execute(
        self,
        call_id: str,
        params: LocateCodeParams,
        signal: asyncio.Event,
        on_update: UpdateCallback | None = None,
    ) -> AgentToolResult:
        steps: list[str] = []

        def progress(event: AgentEvent) -> None:
            if isinstance(event, ToolExecutionStart):
                steps.append(f"{event.tool_name} {json.dumps(event.args, ensure_ascii=False)[:160]}")
                if on_update:
                    on_update(AgentToolResult.text("\n".join(steps[-8:])))

        model = self.model() if callable(self.model) else self.model
        task = asyncio.create_task(
            locate(
                self.intel,
                model,
                params.request,
                options=self.options,
                scope=params.scope,
                max_turns=self.max_turns,
                on_event=progress,
                mode=params.mode,
            )
        )
        stopper = asyncio.create_task(signal.wait())
        try:
            done, _ = await asyncio.wait({task, stopper}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stopper.cancel()
            if not task.done():
                task.cancel()
        if task not in done:
            return AgentToolResult.text("Aborted.", details={"steps": steps})
        report = task.result()
        return AgentToolResult.text(report, details={"steps": steps})
