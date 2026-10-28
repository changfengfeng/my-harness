# my-harness

A minimal, extensible coding agent harness in Python, modeled on the architecture of
[Pi](https://github.com/badlogic/pi-mono).

## Architecture

Three layers; each depends only on the layers above it, so each can be used on its own.

| Layer | Pi counterpart | What it does |
|---|---|---|
| `my_harness.ai` | `pi-ai` | One `stream(model, context, options)` API over Anthropic, OpenAI (and OpenAI-compatible endpoints), plus a scripted `FauxProvider` for tests. Emits normalized events (`start`, `text_delta`, `thinking_delta`, `toolcall_end`, `done`/`error`). Provider failures become an `error` event rather than an exception. |
| `my_harness.agent` | `pi-agent-core` | `run_agent_loop()`: stream → validate tool args (pydantic) → `before_tool_call` → execute (parallel or sequential) → `after_tool_call` → feed results back → repeat. Emits `agent_*`, `turn_*`, `message_*`, `tool_execution_*` events. `Agent` adds state, subscriptions, steering / follow-up queues and abort. |
| `my_harness.coding` | `pi-coding-agent` | Built-in tools (`read`, `write`, `edit`, `bash`, `grep`, `ls`) with output truncation, system prompt + `AGENTS.md`/`CLAUDE.md` loading, append-only JSONL sessions (tree via `parent_id`), and the `myh` CLI. |

```
src/my_harness/
  ai/       types.py  models.py  stream.py  providers/{anthropic,openai,faux}.py
  agent/    types.py  loop.py  agent.py
  coding/   tools/  system_prompt.py  session.py  cli.py
```

## Quick start

```bash
uv sync
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY
uv run myh                          # interactive
uv run myh -p "summarize README.md" # one-shot
uv run myh --json "list files"      # one-shot, every agent event as JSONL
uv run myh -c                       # continue the latest session in this directory
uv run myh -m openai:gpt-5 --thinking medium
uv run myh -m openai:deepseek-chat --base-url https://api.deepseek.com  # any OpenAI-compatible API
```

Sessions are stored under `~/.my-harness/sessions/<cwd-slug>/`. Global instructions can go in
`~/.my-harness/AGENTS.md`.

Like Pi, there is no permission system: tools run with your user's permissions. Run inside a container
or VM if you need isolation.

## Using the layers directly

```python
import asyncio
from pathlib import Path
from my_harness.agent import Agent
from my_harness.ai import get_model
from my_harness.coding.tools import create_tools


async def main():
    agent = Agent(get_model("anthropic"), system_prompt="You are helpful.", tools=create_tools(Path.cwd()))
    agent.subscribe(lambda e: print(e.type))
    await agent.prompt("What files are here?")


asyncio.run(main())
```

Custom tool:

```python
from pydantic import BaseModel, Field
from my_harness.agent import AgentTool, AgentToolResult


class Params(BaseModel):
    city: str = Field(description="City name")


class Weather(AgentTool[Params]):
    name = "weather"
    description = "Get the weather for a city"
    Params = Params

    async def execute(self, call_id, params, signal, on_update=None):
        return AgentToolResult.text(f"Sunny in {params.city}")
```

## Web app

`myh-web` serves a browser chat UI (FastAPI + WebSocket) that streams the agent's thinking, tool calls and
results live. Every browser tab gets its own Agent and a throwaway workspace directory that is deleted on
disconnect; access is protected by a shared password.

```
uv sync --extra web
MYH_WEB_PASSWORD=... MYH_WEB_API_KEY=sk-... uv run myh-web     # http://localhost:8000, DeepSeek by default
```

| Variable | Default | |
|---|---|---|
| `MYH_WEB_PASSWORD` | (required) | set `MYH_WEB_OPEN=1` to run without one |
| `MYH_WEB_MODEL` | `openai:deepseek-chat` | any `provider:model` |
| `MYH_WEB_BASE_URL` | DeepSeek URL for `deepseek-*` models | OpenAI-compatible endpoint |
| `MYH_WEB_API_KEY` | `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | |
| `MYH_WEB_MAX_SESSIONS` | `10` | concurrent connections |

The agent can run arbitrary shell commands, so run it in the container rather than on a machine you care about:

```
docker build -t my-harness-web .
docker run -d -p 8000:8000 -e MYH_WEB_PASSWORD -e MYH_WEB_API_KEY \
  --memory 1g --pids-limit 256 --cap-drop ALL --security-opt no-new-privileges my-harness-web
```

Secrets are removed from the process environment after startup, shell commands get a minimal environment, and the
server marks itself non-dumpable so `/proc/1/environ` and `/proc/1/mem` are unreadable to the agent's commands.
Expose it publicly with any reverse proxy or tunnel, e.g. `cloudflared tunnel --url http://localhost:8000`.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

## Roadmap

- [ ] Context compaction (summarize old turns when nearing the context window)
- [ ] Extensions: skills, prompt templates, slash commands, custom tools loaded from disk
- [ ] Session branching / tree navigation
- [ ] RPC mode (JSON over stdin/stdout)
- [ ] Rich TUI
