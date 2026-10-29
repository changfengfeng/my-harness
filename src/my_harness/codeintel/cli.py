"""`myh-code`: build and query the code index from the shell, or run the locator agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from ..agent.types import AgentEvent, ToolExecutionStart
from ..ai import get_model
from ..ai.types import StreamOptions
from .index import CodeIndex
from .queries import CodeQueries
from .tools import CodeIntel, format_hit, related_report

DIM, CYAN, RESET = "\033[2m", "\033[36m", "\033[0m"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="myh-code", description="Code index and code-location agent for large repositories"
    )
    p.add_argument("--root", type=Path, default=Path.cwd(), help="repository root (default: cwd)")
    p.add_argument("--db", type=Path, help="index file (default: ~/.my-harness/index/<repo>.db)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("index", help="build or refresh the index")
    s.add_argument("--rebuild", action="store_true")
    s = sub.add_parser("search", help="ranked code search")
    s.add_argument("query", nargs="+")
    s.add_argument("-k", "--limit", type=int, default=10)
    s.add_argument("--path")
    s.add_argument("--lang")
    s = sub.add_parser("symbol", help="find definitions")
    s.add_argument("name")
    s = sub.add_parser("refs", help="find references")
    s.add_argument("name")
    s.add_argument("-k", "--limit", type=int, default=60)
    s = sub.add_parser("related", help="callers, callees, imports, tests of a symbol / file / path:line")
    s.add_argument("target")
    s = sub.add_parser("map", help="repository overview")
    s.add_argument("path", nargs="?")
    s.add_argument("--max-files", type=int, default=40)
    s = sub.add_parser("locate", help="run the code-location agent (needs a model)")
    s.add_argument("request", nargs="+")
    s.add_argument("-m", "--model", default="anthropic", help="provider[:model_id]")
    s.add_argument("--base-url")
    s.add_argument("--thinking", default="off", choices=["off", "low", "medium", "high"])
    s.add_argument("--max-turns", type=int, default=14)
    s.add_argument("--scope")
    s.add_argument("--mode", default="deep", choices=["fast", "deep", "auto"])
    s.add_argument("-q", "--quiet", action="store_true", help="do not print tool calls")
    return p.parse_args(argv)


def _progress(done: int, total: int) -> None:
    print(f"\r{DIM}indexing {done}/{total} files{RESET}", end="", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if args.cmd == "locate":
        return asyncio.run(_locate(args, root))
    index = CodeIndex(root, args.db)
    t0 = time.monotonic()
    stats = index.refresh(
        force=getattr(args, "rebuild", False), progress=_progress if sys.stderr.isatty() else None
    )
    if sys.stderr.isatty() and stats.parsed >= 200:
        print(file=sys.stderr)
    q = CodeQueries(index)
    if args.cmd == "index":
        print(
            f"{stats.files} files, {stats.lines} lines, {stats.symbols} symbols; parsed {stats.parsed}, "
            f"removed {stats.removed} in {time.monotonic() - t0:.2f}s\nindex: {index.db_path}"
        )
        print("languages: " + ", ".join(f"{k} {v}" for k, v in stats.languages.items()))
    elif args.cmd == "search":
        t1 = time.monotonic()
        hits = q.search(" ".join(args.query), limit=args.limit, path=args.path, lang=args.lang)
        for h in hits:
            print(format_hit(h))
        print(f"{DIM}{len(hits)} results in {(time.monotonic() - t1) * 1000:.0f} ms{RESET}", file=sys.stderr)
    elif args.cmd == "symbol":
        for h in q.find_symbol(args.name):
            print(format_hit(h, show_snippet=False))
    elif args.cmd == "refs":
        refs, total = q.references(args.name, limit=args.limit)
        for h, ln, text in refs:
            print(f"{h.path}:{ln}  ({h.qualname if h.kind != 'file' else '<module>'})  {text}")
        print(f"{DIM}{len(refs)} lines shown, {total} referencing scopes{RESET}", file=sys.stderr)
    elif args.cmd == "related":
        report = related_report(q, args.target, 12)
        print(report or f"not found: {args.target}")
        return 0 if report else 1
    elif args.cmd == "map":
        print(q.repo_map(args.path, max_files=args.max_files))
    return 0


async def _locate(args: argparse.Namespace, root: Path) -> int:
    from .locator import locate

    intel = CodeIntel(root, args.db)
    model = get_model(args.model, args.base_url)
    t0 = time.monotonic()

    def show(event: AgentEvent) -> None:
        if isinstance(event, ToolExecutionStart) and not args.quiet:
            print(
                f"{CYAN}→ {event.tool_name}{RESET} {DIM}{json.dumps(event.args, ensure_ascii=False)[:160]}{RESET}",
                file=sys.stderr,
                flush=True,
            )

    report = await locate(
        intel,
        model,
        " ".join(args.request),
        options=StreamOptions(),
        thinking=args.thinking,
        scope=args.scope,
        max_turns=args.max_turns,
        on_event=show,
        mode=args.mode,
    )
    print(report)
    print(f"{DIM}({time.monotonic() - t0:.1f}s){RESET}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
