from __future__ import annotations

import asyncio
import contextlib
import os
import signal as signals
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from ...agent.types import AgentTool, AgentToolResult, ToolError, UpdateCallback
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_tail


class BashParams(BaseModel):
    command: str = Field(description="Shell command to execute")
    timeout: float | None = Field(default=None, gt=0, description="Timeout in seconds (optional)")


class BashTool(AgentTool[BashParams]):
    name = "bash"
    description = (
        "Execute a bash command in the current working directory. Returns combined stdout and stderr. "
        f"Output is truncated to the last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB; when "
        "truncated, the full output is saved to a temp file. Optionally provide a timeout in seconds."
    )
    Params = BashParams

    def __init__(self, cwd: Path, env: dict[str, str] | None = None):
        self.cwd = cwd
        self.env = env

    async def execute(
        self, call_id: str, params: BashParams, signal: asyncio.Event, on_update: UpdateCallback | None = None
    ) -> AgentToolResult:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            params.command,
            cwd=self.cwd,
            env=self.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        chunks: list[bytes] = []

        async def pump() -> None:
            assert proc.stdout
            while chunk := await proc.stdout.read(4096):
                chunks.append(chunk)
                if on_update:
                    on_update(
                        AgentToolResult.text(truncate_tail(b"".join(chunks).decode(errors="replace")).text)
                    )
            await proc.wait()

        timed_out = False
        try:
            await asyncio.wait_for(pump(), timeout=params.timeout)
        except TimeoutError:
            timed_out = True
            _kill(proc)
        except asyncio.CancelledError:
            _kill(proc)
            raise

        output = b"".join(chunks).decode(errors="replace").rstrip("\n")
        t = truncate_tail(output)
        text = t.text or "(no output)"
        details: dict[str, object] = {"exit_code": proc.returncode}
        if t.truncated:
            fd, full_path = tempfile.mkstemp(prefix="myh-bash-", suffix=".log")
            with os.fdopen(fd, "w") as f:
                f.write(output)
            details["full_output_path"] = full_path
            text = f"[Output truncated: showing last {t.shown_lines} of {t.total_lines} lines. Full output: {full_path}]\n{text}"
        if timed_out:
            raise ToolError(f"{text}\n\nCommand timed out after {params.timeout}s")
        if proc.returncode != 0:
            raise ToolError(f"{text}\n\nCommand exited with code {proc.returncode}")
        return AgentToolResult.text(text, details=details)


def _kill(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signals.SIGKILL)
