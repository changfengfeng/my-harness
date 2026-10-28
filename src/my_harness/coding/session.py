"""Append-only JSONL session storage.

Line 1 is a header; every following line is one entry. Entries form a tree via `parent_id`, so a
session can later be branched without rewriting history (the current branch is the path from the
last entry back to the root).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from ..ai.types import Message

SESSION_VERSION = 1
_message_adapter: TypeAdapter[Any] = TypeAdapter(Message)


def default_session_dir(cwd: Path) -> Path:
    slug = str(cwd.resolve()).strip("/").replace("/", "-") or "root"
    return Path.home() / ".my-harness" / "sessions" / slug


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SessionManager:
    def __init__(self, path: Path, header: dict[str, Any], entries: list[dict[str, Any]]):
        self.path = path
        self.header = header
        self.entries = entries

    @property
    def id(self) -> str:
        return str(self.header["id"])

    @property
    def leaf_id(self) -> str | None:
        return self.entries[-1]["id"] if self.entries else None

    @classmethod
    def create(cls, cwd: Path, session_dir: Path | None = None) -> SessionManager:
        directory = session_dir or default_session_dir(cwd)
        directory.mkdir(parents=True, exist_ok=True)
        sid = uuid.uuid4().hex[:12]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        header = {
            "type": "session",
            "version": SESSION_VERSION,
            "id": sid,
            "cwd": str(cwd),
            "created": _now(),
        }
        path = directory / f"{stamp}_{sid}.jsonl"
        path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        return cls(path, header, [])

    @classmethod
    def open(cls, path: Path) -> SessionManager:
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines or lines[0].get("type") != "session":
            raise ValueError(f"Not a session file: {path}")
        return cls(path, lines[0], lines[1:])

    @classmethod
    def latest(cls, cwd: Path, session_dir: Path | None = None) -> SessionManager | None:
        directory = session_dir or default_session_dir(cwd)
        files = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
        for path in reversed(files):
            session = cls.open(path)
            if session.messages():
                return session
        return None

    @staticmethod
    def list_files(cwd: Path, session_dir: Path | None = None) -> list[Path]:
        directory = session_dir or default_session_dir(cwd)
        return sorted(directory.glob("*.jsonl")) if directory.is_dir() else []

    def _append(self, entry_type: str, **data: Any) -> dict[str, Any]:
        entry = {
            "type": entry_type,
            "id": uuid.uuid4().hex[:8],
            "parent_id": self.leaf_id,
            "time": _now(),
            **data,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.entries.append(entry)
        return entry

    def append_message(self, message: Any) -> dict[str, Any]:
        return self._append("message", message=message.model_dump(mode="json"))

    def append_model_change(self, model: str, thinking: str) -> dict[str, Any]:
        return self._append("model_change", model=model, thinking=thinking)

    def branch(self) -> list[dict[str, Any]]:
        by_id = {e["id"]: e for e in self.entries}
        path, cur = [], self.leaf_id
        while cur is not None:
            entry = by_id[cur]
            path.append(entry)
            cur = entry.get("parent_id")
        return list(reversed(path))

    def messages(self) -> list[Any]:
        return [
            _message_adapter.validate_python(e["message"]) for e in self.branch() if e["type"] == "message"
        ]

    def last_model(self) -> tuple[str, str] | None:
        for e in reversed(self.branch()):
            if e["type"] == "model_change":
                return e["model"], e["thinking"]
        return None
