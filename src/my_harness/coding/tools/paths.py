from __future__ import annotations

import os
from pathlib import Path


def resolve(cwd: Path, path: str) -> Path:
    p = Path(os.path.expanduser(path))
    return p if p.is_absolute() else (cwd / p)
