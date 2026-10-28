"""Command-line entry point: interactive REPL, print mode, and JSON event mode."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..agent import Agent
from ..agent.types import (
    AgentEvent,
    MessageEnd,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
)
from ..ai import get_model
from ..ai.models import Model
from ..ai.types import AssistantMessage, TextContent, TextDeltaEvent, ThinkingDeltaEvent, Usage
from .session import SessionManager
from .system_prompt import build_system_prompt
from .tools import DEFAULT_TOOLS, create_tools

DIM, BOLD, RED, CYAN, RESET = "\033[2m", "\033[1m", "\033[31m", "\033[36m", "\033[0m"

HELP = """Commands:
  /help                 show this help
  /model <provider:id>  switch model (e.g. anthropic:claude-sonnet-4-5, openai:gpt-5)
  /thinking <level>     off | low | medium | high
  /new                  start a new session
  /session              show session file and token usage
  /quit                 exit
Press Ctrl+C while the agent is working to abort; at the prompt to exit."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="myh", description="my-harness: a minimal coding agent")
    p.add_argument("prompt", nargs="*", help="initial prompt (enables one-shot mode with -p)")
    p.add_argument("-m", "--model", default="anthropic", help="provider[:model_id], default: anthropic")
    p.add_argument("--base-url", help="custom API base URL (e.g. an OpenAI-compatible endpoint)")
    p.add_argument("--thinking", default="off", choices=["off", "low", "medium", "high"])
    p.add_argument("--tools", default=",".join(DEFAULT_TOOLS), help="comma-separated tool names")
    p.add_argument("-p", "--print", action="store_true", help="run the prompt once, print the answer, exit")
    p.add_argument("--json", action="store_true", help="like --print, but emit every agent event as JSONL")
    p.add_argument("-c", "--continue", dest="resume", action="store_true", help="continue the latest session")
    p.add_argument("--session", type=Path, help="open a specific session file")
    p.add_argument("--no-session", action="store_true", help="do not persist the session")
    p.add_argument("--system-prompt", help="replace the default system prompt")
    p.add_argument("--append-system-prompt", help="text appended to the system prompt")
    p.add_argument("--cwd", type=Path, default=Path.cwd(), help="working directory for tools")
    return p.parse_args(argv)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()} | {"type": type(obj).type}  # type: ignore[attr-defined]
    if isinstance(obj, list | tuple):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


class Renderer:
    """Streams agent events to the terminal."""

    def __init__(self, out: Any = sys.stdout):
        self.out = out
        self.in_thinking = False
        self.usage = Usage()

    def write(self, s: str) -> None:
        self.out.write(s)
        self.out.flush()

    def __call__(self, event: AgentEvent) -> None:
        if isinstance(event, MessageUpdate):
            e = event.event
            if isinstance(e, ThinkingDeltaEvent):
                if not self.in_thinking:
                    self.write(f"{DIM}")
                    self.in_thinking = True
                self.write(e.delta)
            elif isinstance(e, TextDeltaEvent):
                if self.in_thinking:
                    self.write(f"{RESET}\n")
                    self.in_thinking = False
                self.write(e.delta)
        elif isinstance(event, MessageEnd) and isinstance(event.message, AssistantMessage):
            m = event.message
            self.usage = self.usage + m.usage
            if self.in_thinking:
                self.write(f"{RESET}\n")
                self.in_thinking = False
            if m.text:
                self.write("\n")
            if m.error_message:
                self.write(f"{RED}[{m.stop_reason}] {m.error_message}{RESET}\n")
        elif isinstance(event, ToolExecutionStart):
            args = json.dumps(event.args, ensure_ascii=False)
            self.write(f"{CYAN}→ {event.tool_name}{RESET} {DIM}{args[:200]}{RESET}\n")
        elif isinstance(event, ToolExecutionEnd):
            text = "\n".join(c.text for c in event.result.content if isinstance(c, TextContent))
            lines = text.splitlines()
            preview = "\n".join("  " + line for line in lines[:8])
            more = f"\n  ... ({len(lines) - 8} more lines)" if len(lines) > 8 else ""
            color = RED if event.is_error else DIM
            self.write(f"{color}{preview}{more}{RESET}\n")


def build_agent(args: argparse.Namespace, model: Model, session: SessionManager | None) -> Agent:
    cwd = args.cwd.resolve()
    tools = create_tools(cwd, [t.strip() for t in args.tools.split(",") if t.strip()])
    prompt = build_system_prompt(cwd, tools, args.system_prompt, args.append_system_prompt)
    agent = Agent(
        model,
        system_prompt=prompt,
        tools=tools,
        thinking=args.thinking,
        messages=session.messages() if session else [],
    )
    if session:

        def persist(event: AgentEvent) -> None:
            if isinstance(event, MessageEnd) and isinstance(event.message, BaseModel):
                session.append_message(event.message)

        agent.subscribe(persist)
    return agent


def open_session(args: argparse.Namespace) -> SessionManager | None:
    cwd = args.cwd.resolve()
    if args.no_session:
        return None
    if args.session:
        return SessionManager.open(args.session)
    if args.resume:
        return SessionManager.latest(cwd) or SessionManager.create(cwd)
    return SessionManager.create(cwd)


async def run_with_abort(agent: Agent, prompt: str) -> None:
    loop = asyncio.get_running_loop()
    with contextlib.suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGINT, agent.abort)
    try:
        await agent.prompt(prompt)
    finally:
        with contextlib.suppress(NotImplementedError):
            loop.remove_signal_handler(signal.SIGINT)


async def one_shot(agent: Agent, prompt: str, as_json: bool) -> int:
    if as_json:
        agent.subscribe(lambda e: print(json.dumps(_jsonable(e), ensure_ascii=False), flush=True))
        await run_with_abort(agent, prompt)
    else:
        last: list[AssistantMessage] = []
        agent.subscribe(
            lambda e: (
                last.append(e.message)
                if isinstance(e, MessageEnd) and isinstance(e.message, AssistantMessage)
                else None
            )
        )
        await run_with_abort(agent, prompt)
        if last:
            if last[-1].text:
                print(last[-1].text)
            if last[-1].error_message:
                print(f"error: {last[-1].error_message}", file=sys.stderr)
    return 1 if agent.state.error else 0


async def repl(
    agent: Agent, args: argparse.Namespace, session: SessionManager | None, initial: str | None
) -> int:
    renderer = Renderer()
    agent.subscribe(renderer)
    print(
        f"{BOLD}my-harness{RESET} {DIM}{agent.state.model} · cwd {args.cwd.resolve()}"
        f"{' · session ' + str(session.path) if session else ''}{RESET}"
    )
    if agent.state.messages:
        print(f"{DIM}(resumed {len(agent.state.messages)} messages){RESET}")
    print(f"{DIM}/help for commands{RESET}")
    pending = initial
    while True:
        if pending is None:
            try:
                pending = await asyncio.to_thread(input, f"\n{BOLD}> {RESET}")
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
        text, pending = pending.strip(), None
        if not text:
            continue
        if text.startswith("/"):
            cmd, _, arg = text.partition(" ")
            if cmd in ("/quit", "/exit"):
                return 0
            if cmd == "/help":
                print(HELP)
            elif cmd == "/model" and arg:
                try:
                    agent.state.model = get_model(arg.strip(), args.base_url)
                    print(f"model: {agent.state.model}")
                    if session:
                        session.append_model_change(str(agent.state.model), agent.state.thinking)
                except ValueError as exc:
                    print(f"{RED}{exc}{RESET}")
            elif cmd == "/thinking" and arg.strip() in ("off", "low", "medium", "high"):
                agent.state.thinking = arg.strip()  # type: ignore[assignment]
                print(f"thinking: {agent.state.thinking}")
            elif cmd == "/new":
                agent.reset()
                renderer.usage = Usage()
                if session:
                    session = SessionManager.create(args.cwd.resolve(), session.path.parent)
                    _rebind_persist(agent, session)
                print("new session")
            elif cmd == "/session":
                u = renderer.usage
                print(
                    f"session: {session.path if session else '(not persisted)'}\n"
                    f"messages: {len(agent.state.messages)}  tokens: in {u.input} · out {u.output} "
                    f"· cache read {u.cache_read} · cache write {u.cache_write}"
                )
            else:
                print(f"Unknown command. {HELP}")
            continue
        await run_with_abort(agent, text)


def _rebind_persist(agent: Agent, session: SessionManager) -> None:
    # Replace the persistence listener installed by build_agent.
    agent._listeners = [fn for fn in agent._listeners if getattr(fn, "__name__", "") != "persist"]

    def persist(event: AgentEvent) -> None:
        if isinstance(event, MessageEnd) and isinstance(event.message, BaseModel):
            session.append_message(event.message)

    agent.subscribe(persist)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session = open_session(args)
    model = get_model(args.model, args.base_url)
    if session and (saved := session.last_model()) and args.model == "anthropic":
        model = get_model(saved[0], args.base_url)
    elif session and not session.entries:
        session.append_model_change(str(model), args.thinking)
    agent = build_agent(args, model, session)
    prompt = " ".join(args.prompt) or None
    if args.print or args.json:
        if prompt is None and not sys.stdin.isatty():
            prompt = sys.stdin.read()
        if not prompt:
            print("error: --print/--json need a prompt", file=sys.stderr)
            return 2
        return asyncio.run(one_shot(agent, prompt, args.json))
    return asyncio.run(repl(agent, args, session, prompt))


if __name__ == "__main__":
    sys.exit(main())
