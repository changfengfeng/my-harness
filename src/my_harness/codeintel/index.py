"""Persistent code index: files, symbols, imports and identifier references in SQLite + FTS5.

The index is incremental (files are re-parsed only when their mtime or size changes) and parsing runs
in a process pool for large repositories, so a repository with hundreds of thousands of lines is
indexed in seconds and refreshed in milliseconds.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import posixpath
import re
import sqlite3
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .parsers import NON_CODE, ParsedFile, detect_language, identifiers, parse_source
from .tokens import index_text, tokenize

SCHEMA_VERSION = "3"  # bump when the schema or the parser output changes: forces a rebuild
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".next", ".nuxt", "coverage", "target", ".gradle", ".idea", ".vscode", "bower_components",
    ".my-harness", ".cache", "site-packages",
}  # fmt: skip
SKIP_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "poetry.lock",
    "uv.lock",
    "go.sum",
}
MAX_FILE_BYTES = 1_000_000
MAX_CONFIG_BYTES = 200_000
MAX_BODY_TOKENS = 4000
PARALLEL_THRESHOLD = 200
TEST_RE = re.compile(
    r"(^|/)(tests?|__tests__|spec|testing)/|(^|/)test_[^/]*$|_test\.\w+$|\.(test|spec)\.\w+$|Tests?\.\w+$"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files(
    id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, lang TEXT, mtime_ns INTEGER, size INTEGER,
    lines INTEGER, is_test INTEGER DEFAULT 0, rank REAL DEFAULT 0, error TEXT);
CREATE TABLE IF NOT EXISTS symbols(
    id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, name TEXT NOT NULL, qualname TEXT, kind TEXT,
    start INTEGER, end INTEGER, line INTEGER, signature TEXT, doc TEXT, parent_id INTEGER);
CREATE INDEX IF NOT EXISTS symbols_name ON symbols(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS symbols_name_exact ON symbols(name);
CREATE INDEX IF NOT EXISTS symbols_file ON symbols(file_id);
CREATE INDEX IF NOT EXISTS symbols_parent ON symbols(parent_id);
CREATE TABLE IF NOT EXISTS refs(
    name TEXT NOT NULL, file_id INTEGER NOT NULL, symbol_id INTEGER NOT NULL,
    PRIMARY KEY(name, file_id, symbol_id)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS refs_file ON refs(file_id, symbol_id);
CREATE TABLE IF NOT EXISTS imports(file_id INTEGER NOT NULL, target TEXT, line INTEGER, resolved_id INTEGER);
CREATE INDEX IF NOT EXISTS imports_file ON imports(file_id);
CREATE INDEX IF NOT EXISTS imports_resolved ON imports(resolved_id);
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(name, path, signature, doc, body, tokenize='unicode61');
"""


@dataclass
class Chunk:
    name: str
    qualname: str
    kind: str
    start: int
    end: int
    line: int
    signature: str
    doc: str
    parent: int | None
    body: str
    refs: list[str]


@dataclass
class FileRecord:
    path: str
    lang: str
    mtime_ns: int
    size: int
    lines: int
    chunks: list[Chunk] = field(default_factory=list)
    imports: list[tuple[str, int]] = field(default_factory=list)
    error: str | None = None


@dataclass
class IndexStats:
    files: int
    lines: int
    symbols: int
    parsed: int = 0
    removed: int = 0
    seconds: float = 0.0
    languages: dict[str, int] = field(default_factory=dict)


def default_index_path(root: Path) -> Path:
    base = Path(os.environ.get("MYH_INDEX_DIR") or Path.home() / ".my-harness" / "index")
    resolved = str(root.resolve())
    slug = resolved.strip("/").replace("/", "-")[-60:] or "root"
    digest = hashlib.sha1(resolved.encode()).hexdigest()[:8]
    return base / f"{slug}-{digest}.db"


def is_test_path(path: str) -> bool:
    return bool(TEST_RE.search(path))


# ---------------------------------------------------------------------------
# File discovery and analysis (analysis runs in worker processes)
# ---------------------------------------------------------------------------


def list_source_files(root: Path) -> list[str]:
    files: list[str] | None = None
    if (root / ".git").exists():
        try:
            out = subprocess.run(
                ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                cwd=root,
                capture_output=True,
                timeout=60,
                check=True,
            ).stdout
            files = [f for f in out.decode("utf-8", "replace").split("\0") if f]
        except (OSError, subprocess.SubprocessError):
            files = None
    if files is None:
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            rel = os.path.relpath(dirpath, root)
            for name in filenames:
                files.append(name if rel == "." else f"{rel}/{name}".replace(os.sep, "/"))
    keep = []
    for f in files:
        parts = f.split("/")
        if SKIP_DIRS.intersection(parts[:-1]) or parts[-1] in SKIP_FILES or ".min." in parts[-1]:
            continue
        if detect_language(f):
            keep.append(f)
    return sorted(set(keep))


def _own_line_mask(sym_ranges: list[tuple[int, int, int | None]], n_lines: int) -> list[int]:
    """For each line, the index of the innermost symbol owning it (or -1 for module level)."""
    owner = [-1] * (n_lines + 2)
    for i, (start, end, _parent) in enumerate(sym_ranges):  # parents come first, so children overwrite
        for ln in range(max(start, 1), min(end, n_lines) + 1):
            owner[ln] = i
    return owner


def analyze_file(root: str, rel: str) -> FileRecord | None:
    path = os.path.join(root, rel)
    try:
        st = os.stat(path)
    except OSError:
        return None
    lang = detect_language(rel) or "text"
    limit = MAX_CONFIG_BYTES if lang in NON_CODE else MAX_FILE_BYTES
    rec = FileRecord(rel, lang, st.st_mtime_ns, st.st_size, 0)
    if st.st_size > limit:
        rec.error = "skipped: file too large"
        return rec
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        rec.error = f"unreadable: {exc}"
        return rec
    if b"\0" in raw[:8000]:
        rec.error = "skipped: binary"
        return rec
    text = raw.decode("utf-8", "replace")
    lines = text.splitlines()
    if lines and (len(text) / len(lines) > 400 or max(len(line) for line in lines) > 20_000):
        rec.error = "skipped: minified or generated"
        rec.lines = len(lines)
        return rec
    pf: ParsedFile = parse_source(rel, text, lang)
    rec.lines, rec.imports, rec.error = pf.lines, pf.imports, pf.error
    syms = pf.symbols
    owner = _own_line_mask([(s.start, s.end, s.parent) for s in syms], len(lines))
    own_lines: list[list[str]] = [[] for _ in syms]
    module_lines: list[str] = []
    for ln, line in enumerate(lines, 1):
        o = owner[ln]
        (module_lines if o < 0 else own_lines[o]).append(line)
    for s in syms:  # a parent's body mentions its members by signature
        if s.parent is not None:
            own_lines[s.parent].append(s.signature)
    stem = posixpath.splitext(rel)[0]
    mod_name = (
        posixpath.basename(posixpath.dirname(rel))
        if stem.endswith(("__init__", "/index", "/mod"))
        else posixpath.basename(stem)
    )
    if lang in NON_CODE:
        own_text = "\n".join(lines)
        module_lines = lines if not syms else module_lines
    else:
        own_text = "\n".join(module_lines)
    rec.chunks.append(
        Chunk(
            name=mod_name or rel,
            qualname=rel,
            kind="file",
            start=1,
            end=max(len(lines), 1),
            line=1,
            signature=rel,
            doc=_module_doc(lines, lang),
            parent=None,
            body=index_text(own_text if lang not in NON_CODE else "\n".join(module_lines), MAX_BODY_TOKENS),
            refs=sorted(identifiers("\n".join(module_lines))) if lang not in NON_CODE else [],
        )
    )
    for i, s in enumerate(syms):
        body = "\n".join(own_lines[i])
        rec.chunks.append(
            Chunk(
                name=s.name,
                qualname=s.qualname,
                kind=s.kind,
                start=s.start,
                end=s.end,
                line=s.line,
                signature=s.signature,
                doc=s.doc,
                parent=s.parent,
                body=index_text(body, MAX_BODY_TOKENS),
                refs=sorted((identifiers(body) - {s.name}) | s.refs) if lang not in NON_CODE else [],
            )
        )
    return rec


def _module_doc(lines: list[str], lang: str) -> str:
    """Leading docstring / comment block of a file."""
    out: list[str] = []
    in_doc = False
    for line in lines[:40]:
        s = line.strip()
        if lang == "python" and (s.startswith(('"""', "'''")) or in_doc):
            out.append(s.strip("\"' "))
            toggles = (s.count('"""') + s.count("'''")) % 2 == 1 or (
                len(s) > 3 and s.endswith(('"""', "'''"))
            )
            if toggles and (in_doc or len(s) <= 3 or not s.endswith(('"""', "'''")) or s in ('"""', "'''")):
                in_doc = not in_doc
            if not in_doc:
                break
        elif s.startswith(("//", "/*", "*", "#")) and not s.startswith(("#!", "#include", "#import")):
            out.append(s.lstrip("/*#! ").rstrip("*/ "))
        elif s and out or s and not s.startswith(("package", "<?php", "#!")):
            break
    return " ".join(o for o in out if o)[:400]


def _analyze_batch(root: str, rels: list[str]) -> list[FileRecord | None]:
    return [analyze_file(root, r) for r in rels]


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _process_lock(db_path: Path) -> Iterator[None]:
    """Serialize refreshes of one index across processes (no-op where flock is unavailable)."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(f"{db_path}.lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


class CodeIndex:
    def __init__(self, root: Path, db_path: Path | None = None, workers: int | None = None):
        self.root = root.resolve()
        self.db_path = db_path or default_index_path(self.root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.workers = workers or max(1, min(8, (os.cpu_count() or 2)))
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._last_refresh = 0.0

    def _init_schema(self) -> None:
        row = None
        with contextlib.suppress(sqlite3.OperationalError):
            row = self.db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        if row is not None and row[0] != SCHEMA_VERSION:
            self._drop_tables()
        self.db.executescript(SCHEMA)
        self.db.execute("INSERT OR REPLACE INTO meta VALUES('schema', ?)", (SCHEMA_VERSION,))
        self.db.execute("INSERT OR REPLACE INTO meta VALUES('root', ?)", (str(self.root),))

    def _drop_tables(self) -> None:
        for t in ("meta", "files", "symbols", "refs", "imports", "symbols_fts"):
            self.db.execute(f"DROP TABLE IF EXISTS {t}")

    def close(self) -> None:
        with self.lock:
            self.db.close()

    # -- building ----------------------------------------------------------

    def is_empty(self) -> bool:
        return self.db.execute("SELECT 1 FROM files LIMIT 1").fetchone() is None

    def refresh(
        self,
        force: bool = False,
        min_interval: float = 0.0,
        progress: Callable[[int, int], None] | None = None,
    ) -> IndexStats:
        """Bring the index up to date with the working tree. Cheap when nothing changed."""
        with self.lock:
            if not force and min_interval and time.monotonic() - self._last_refresh < min_interval:
                return self.stats()
            with _process_lock(self.db_path):
                try:
                    return self._refresh(force, progress)
                except BaseException:
                    if self.db.in_transaction:
                        self.db.execute("ROLLBACK")
                    raise

    def _refresh(self, force: bool, progress: Callable[[int, int], None] | None) -> IndexStats:
        t0 = time.monotonic()
        if force:  # dropping is much faster than deleting every row
            self._drop_tables()
            self._init_schema()
            self.db.execute("VACUUM")
        current = list_source_files(self.root)
        known = {
            r["path"]: (r["id"], r["mtime_ns"], r["size"])
            for r in self.db.execute("SELECT id, path, mtime_ns, size FROM files")
        }
        todo: list[str] = []
        gone: set[str] = set()
        for rel in current:
            old = known.get(rel)
            if force or old is None:
                todo.append(rel)
                continue
            try:
                st = os.stat(self.root / rel)
            except OSError:  # listed by git but deleted from the working tree
                gone.add(rel)
                continue
            if st.st_mtime_ns != old[1] or st.st_size != old[2]:
                todo.append(rel)
        current_set = set(current) - gone
        removed = [fid for p, (fid, _, _) in known.items() if p not in current_set]
        if todo or removed:
            self._delete_files(removed + [known[p][0] for p in todo if p in known])
            self._store(self._analyze(todo, progress))
            self._resolve_imports()
            self._compute_ranks()
        self._last_refresh = time.monotonic()
        stats = self.stats()
        stats.parsed, stats.removed, stats.seconds = len(todo), len(removed), time.monotonic() - t0
        return stats

    def _analyze(self, rels: list[str], progress: Callable[[int, int], None] | None) -> Iterable[FileRecord]:
        root = str(self.root)
        if len(rels) < PARALLEL_THRESHOLD or self.workers == 1:
            for i, rel in enumerate(rels, 1):
                rec = analyze_file(root, rel)
                if progress and i % 200 == 0:
                    progress(i, len(rels))
                if rec:
                    yield rec
            return
        size = max(20, min(200, len(rels) // (self.workers * 4) or 1))
        batches = [rels[i : i + size] for i in range(0, len(rels), size)]
        done = 0
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            for batch_result in pool.map(_analyze_batch, [root] * len(batches), batches):
                done += len(batch_result)
                if progress:
                    progress(done, len(rels))
                yield from (r for r in batch_result if r)

    def _delete_files(self, file_ids: list[int]) -> None:
        if not file_ids:
            return
        db = self.db
        db.execute("BEGIN")
        for i in range(0, len(file_ids), 500):
            ids = file_ids[i : i + 500]
            marks = ",".join("?" * len(ids))
            db.execute(
                f"DELETE FROM symbols_fts WHERE rowid IN (SELECT id FROM symbols WHERE file_id IN ({marks}))",
                ids,
            )
            for table, col in (
                ("symbols", "file_id"),
                ("refs", "file_id"),
                ("imports", "file_id"),
                ("files", "id"),
            ):
                db.execute(f"DELETE FROM {table} WHERE {col} IN ({marks})", ids)
        db.execute("COMMIT")

    def _store(self, records: Iterable[FileRecord]) -> None:
        db = self.db
        db.execute("BEGIN")
        n = 0
        for rec in records:
            cur = db.execute(
                "INSERT INTO files(path, lang, mtime_ns, size, lines, is_test, error) VALUES(?,?,?,?,?,?,?)",
                (
                    rec.path,
                    rec.lang,
                    rec.mtime_ns,
                    rec.size,
                    rec.lines,
                    int(is_test_path(rec.path)),
                    rec.error,
                ),
            )
            fid = cur.lastrowid
            path_tokens = index_text(rec.path)
            ids: list[int] = []
            for c in rec.chunks:
                parent = ids[c.parent + 1] if c.parent is not None else None  # +1: chunk 0 is the file chunk
                cur = db.execute(
                    "INSERT INTO symbols(file_id, name, qualname, kind, start, end, line, signature, doc, parent_id)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (fid, c.name, c.qualname, c.kind, c.start, c.end, c.line, c.signature, c.doc, parent),
                )
                sid = cur.lastrowid
                assert sid is not None
                ids.append(sid)
                name_tokens = index_text(c.qualname if c.kind != "file" else c.name)
                db.execute(
                    "INSERT INTO symbols_fts(rowid, name, path, signature, doc, body) VALUES(?,?,?,?,?,?)",
                    (sid, name_tokens, path_tokens, index_text(c.signature), index_text(c.doc), c.body),
                )
                if c.refs:
                    db.executemany(
                        "INSERT OR IGNORE INTO refs VALUES(?,?,?)", [(r, fid, sid) for r in c.refs]
                    )
            if rec.imports:
                db.executemany(
                    "INSERT INTO imports(file_id, target, line) VALUES(?,?,?)",
                    [(fid, t, ln) for t, ln in dict.fromkeys(rec.imports)],
                )
            n += 1  # noqa: SIM113
            if n % 2000 == 0:
                db.execute("COMMIT")
                db.execute("BEGIN")
        db.execute("COMMIT")

    # -- import resolution ---------------------------------------------------

    def _resolve_imports(self) -> None:
        files = {r["id"]: (r["path"], r["lang"]) for r in self.db.execute("SELECT id, path, lang FROM files")}
        resolver = ImportResolver(self.root, files)
        updates = []
        for row in self.db.execute("SELECT rowid, file_id, target FROM imports"):
            path, lang = files.get(row["file_id"], ("", ""))
            updates.append((resolver.resolve(path, lang, row["target"]), row["rowid"]))
        self.db.execute("BEGIN")
        self.db.executemany("UPDATE imports SET resolved_id=? WHERE rowid=?", updates)
        self.db.execute("COMMIT")

    # -- ranking -------------------------------------------------------------

    def _compute_ranks(self) -> None:
        """PageRank over the file graph: a file that references a name defined elsewhere, or imports
        another file, links to it. Ambiguous names (defined in many files) contribute little."""
        db = self.db
        edges: dict[tuple[int, int], float] = defaultdict(float)
        rows = db.execute(
            """
            WITH defs AS (SELECT DISTINCT name, file_id FROM symbols
                          WHERE kind NOT IN ('file','section','variable') AND length(name) > 2),
                 nd AS (SELECT name, count(*) AS c FROM defs GROUP BY name HAVING c <= 5),
                 uses AS (SELECT name, file_id, count(*) AS n FROM refs
                          WHERE name IN (SELECT name FROM nd) GROUP BY name, file_id)
            SELECT uses.file_id AS src, defs.file_id AS dst, uses.n AS n, nd.c AS c
            FROM uses JOIN nd ON nd.name = uses.name JOIN defs ON defs.name = uses.name
            WHERE uses.file_id != defs.file_id
            """
        )
        for r in rows:
            edges[(r["src"], r["dst"])] += math.sqrt(r["n"]) / r["c"]
        for r in db.execute(
            "SELECT DISTINCT file_id, resolved_id FROM imports WHERE resolved_id IS NOT NULL AND resolved_id != file_id"
        ):
            edges[(r[0], r[1])] += 1.0
        nodes = [r[0] for r in db.execute("SELECT id FROM files")]
        ranks = pagerank(nodes, edges)
        top = max(ranks.values(), default=1.0) or 1.0
        db.execute("BEGIN")
        db.executemany("UPDATE files SET rank=? WHERE id=?", [(v / top, k) for k, v in ranks.items()])
        db.execute("COMMIT")

    # -- queries -------------------------------------------------------------

    def stats(self) -> IndexStats:
        with self.lock:
            r = self.db.execute("SELECT count(*), coalesce(sum(lines),0) FROM files").fetchone()
            s = self.db.execute("SELECT count(*) FROM symbols WHERE kind != 'file'").fetchone()[0]
            langs = {
                row[0]: row[1]
                for row in self.db.execute(
                    "SELECT lang, sum(lines) FROM files GROUP BY lang ORDER BY sum(lines) DESC"
                )
            }
            return IndexStats(files=r[0], lines=r[1], symbols=s, languages=langs)

    def query(self, sql: str, args: Iterable[object] = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, tuple(args)).fetchall()

    def read_lines(self, path: str) -> list[str]:
        try:
            return (self.root / path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []


def pagerank(
    nodes: list[int], edges: dict[tuple[int, int], float], damping: float = 0.85, iters: int = 30
) -> dict[int, float]:
    if not nodes:
        return {}
    n = len(nodes)
    out_weight: dict[int, float] = defaultdict(float)
    incoming: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (src, dst), w in edges.items():
        out_weight[src] += w
        incoming[dst].append((src, w))
    rank = dict.fromkeys(nodes, 1.0 / n)
    for _ in range(iters):
        dangling = sum(rank[v] for v in nodes if out_weight.get(v, 0) == 0)
        base = (1 - damping) / n + damping * dangling / n
        rank = {
            v: base + damping * sum(rank[s] * w / out_weight[s] for s, w in incoming.get(v, ()) if s in rank)
            for v in nodes
        }
    return rank


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------

JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".d.ts", ".vue", ".svelte")


class ImportResolver:
    """Maps an import string to a file in the repository, per language conventions."""

    def __init__(self, root: Path, files: dict[int, tuple[str, str]]):
        self.root = root
        self.by_path = {p: fid for fid, (p, _) in files.items()}
        self._paths = {fid: p for fid, (p, _) in files.items()}
        self.py_modules: dict[str, list[int]] = defaultdict(list)
        self.suffixes: dict[str, list[int]] = defaultdict(list)
        self.dirs: dict[str, list[int]] = defaultdict(list)
        self.packages: dict[str, str] = {}
        for fid, (p, lang) in files.items():
            stem, _ext = posixpath.splitext(p)
            parts = stem.split("/")
            if lang == "python":
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                for k in range(1, len(parts) + 1):
                    self.py_modules[".".join(parts[-k:])].append(fid)
            full_parts = p.split("/")
            for k in range(1, min(len(parts), 5) + 1):
                self.suffixes["/".join(parts[-k:])].append(fid)
                self.suffixes["/".join(full_parts[-k:])].append(fid)
            d = posixpath.dirname(p)
            dparts = d.split("/") if d else []
            for k in range(1, min(len(dparts), 6) + 1):
                self.dirs["/".join(dparts[-k:])].append(fid)
            if posixpath.basename(p) == "package.json":
                self._read_package(p)

    def _read_package(self, rel: str) -> None:
        try:
            data = json.loads((self.root / rel).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict) and isinstance(data.get("name"), str):
            self.packages[data["name"]] = posixpath.dirname(rel)

    @staticmethod
    def _closest(src: str, cands: list[int], paths: dict[int, str]) -> int | None:
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        src_parts = src.split("/")

        def shared(fid: int) -> int:
            n = 0
            for a, b in zip(src_parts, paths[fid].split("/"), strict=False):
                if a != b:
                    break
                n += 1
            return n

        return max(cands, key=shared)

    def resolve(self, src: str, lang: str, target: str) -> int | None:
        if not target:
            return None
        paths = self._paths
        if lang == "python":
            return self._python(src, target, paths)
        if lang in ("javascript", "typescript", "dart"):
            return self._js(src, target, paths)
        if lang == "go":
            parts = target.split("/")
            for k in range(min(len(parts), 6), 1, -1):
                cands = [c for c in self.dirs.get("/".join(parts[-k:]), []) if paths[c].endswith(".go")]
                if cands:
                    return self._closest(
                        src, sorted(cands, key=lambda c: paths[c].endswith("_test.go")), paths
                    )
            return None
        if lang in ("c", "cpp"):
            rel = posixpath.normpath(posixpath.join(posixpath.dirname(src), target))
            if rel in self.by_path:
                return self.by_path[rel]
            return self._closest(src, self.suffixes.get(target, []), paths)
        if lang == "ruby" and target.startswith("."):
            rel = posixpath.normpath(posixpath.join(posixpath.dirname(src), target))
            return self.by_path.get(rel + ".rb") or self.by_path.get(rel)
        norm = re.sub(r"::|\\|\.", "/", target)
        norm = re.sub(r"^(crate|self|super)/", "", norm)
        if lang == "rust" and "/" not in norm:  # `mod foo;`
            base = posixpath.dirname(src)
            return self.by_path.get(f"{base}/{norm}.rs") or self.by_path.get(f"{base}/{norm}/mod.rs")
        parts = [p for p in norm.split("/") if p and p != "*"]
        while parts:
            for k in range(min(len(parts), 5), 0, -1):
                cands = self.suffixes.get("/".join(parts[-k:]), [])
                if cands and (k > 1 or len(cands) == 1):
                    return self._closest(src, cands, paths)
            parts = parts[:-1]
        return None

    def _python(self, src: str, target: str, paths: dict[int, str]) -> int | None:
        if target.startswith("."):
            level = len(target) - len(target.lstrip("."))
            pkg = posixpath.dirname(src).split("/") if posixpath.dirname(src) else []
            pkg = pkg[: len(pkg) - (level - 1)] if level > 1 else pkg
            rest = target[level:]
            target = ".".join([*pkg, *([rest] if rest else [])])
            cands = self.py_modules.get(target, [])
            exact = [
                c
                for c in cands
                if posixpath.splitext(paths[c])[0].replace("/__init__", "").replace("/", ".") == target
            ]
            return exact[0] if exact else self._closest(src, cands, paths)
        return self._closest(src, self.py_modules.get(target, []), paths)

    def _js(self, src: str, target: str, paths: dict[int, str]) -> int | None:
        if target.startswith("."):
            base = posixpath.normpath(posixpath.join(posixpath.dirname(src), target))
        else:
            pkg = next(
                (
                    n
                    for n in sorted(self.packages, key=len, reverse=True)
                    if target == n or target.startswith(n + "/")
                ),
                None,
            )
            if pkg is None:
                return None
            sub = target[len(pkg) :].lstrip("/")
            pdir = self.packages[pkg]
            base = posixpath.join(pdir, "src", sub) if sub else posixpath.join(pdir, "src", "index")
            found = self._js_file(base) or self._js_file(posixpath.join(pdir, sub or "index"))
            return found
        return self._js_file(base)

    def _js_file(self, base: str) -> int | None:
        base = base.lstrip("./") if base.startswith("./") else base
        stem = re.sub(r"\.(m|c)?js$", "", base)
        for cand in (base, *(stem + e for e in JS_EXTS), *(f"{base}/index{e}" for e in JS_EXTS)):
            if cand in self.by_path:
                return self.by_path[cand]
        return None


def own_lines(index: CodeIndex, symbol_id: int, start: int, end: int, kind: str, file_id: int) -> list[int]:
    """Lines that belong to a chunk itself rather than to a nested symbol."""
    if kind == "file":
        rows = index.query(
            "SELECT start, end FROM symbols WHERE file_id=? AND parent_id IS NULL AND kind!='file'",
            (file_id,),
        )
    else:
        rows = index.query("SELECT start, end, line FROM symbols WHERE parent_id=?", (symbol_id,))
    excluded: set[int] = set()
    for r in rows:
        excluded.update(range(r["start"], r["end"] + 1))
        if kind != "file":
            excluded.discard(r["line"])
    return [ln for ln in range(start, end + 1) if ln not in excluded]


__all__ = [
    "CodeIndex",
    "IndexStats",
    "analyze_file",
    "default_index_path",
    "list_source_files",
    "own_lines",
    "tokenize",
]
