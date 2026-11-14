"""Query layer over `CodeIndex`: ranked search, definitions, references, related code, repo map."""

from __future__ import annotations

import fnmatch
import posixpath
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .index import CodeIndex, own_lines
from .parsers import NON_CODE
from .tokens import IDENT_RE, query_terms, stem, tokenize

KIND_WEIGHT = {
    "class": 1.0, "interface": 1.0, "function": 1.0, "method": 1.0, "type": 0.95, "enum": 0.9, "impl": 0.9,
    "module": 0.9, "variable": 0.8, "file": 0.75, "section": 0.6,
}  # fmt: skip
DEF_KIND_ORDER = "CASE s.kind WHEN 'class' THEN 0 WHEN 'interface' THEN 0 WHEN 'function' THEN 1 WHEN 'method' THEN 1 WHEN 'type' THEN 2 WHEN 'file' THEN 9 ELSE 3 END"
VENDORED_RE = re.compile(r"(^|/)(vendor|third_party|node_modules)/")
SYMBOL_COLS = (
    "s.id, s.file_id, s.name, s.qualname, s.kind, s.start, s.end, s.line, s.signature, s.doc, s.parent_id, "
    "f.path, f.lang, f.is_test, f.rank"
)


@dataclass
class Hit:
    id: int
    file_id: int
    path: str
    name: str
    qualname: str
    kind: str
    start: int
    end: int
    line: int
    signature: str
    doc: str
    lang: str
    is_test: bool
    rank: float
    score: float = 0.0
    snippet: list[tuple[int, str]] = field(default_factory=list)

    @property
    def location(self) -> str:
        return (
            f"{self.path}:{self.start}-{self.end}" if self.end > self.start else f"{self.path}:{self.start}"
        )

    def header(self) -> str:
        label = self.qualname if self.kind != "file" else self.path
        return f"{self.location}  [{self.kind}] {label}"


JS_FAMILY = {"javascript", "typescript"}
C_FAMILY = {"c", "cpp"}


def _lang_family(lang: str) -> str:
    return "js" if lang in JS_FAMILY else "c" if lang in C_FAMILY else lang


def _distinctive(name: str) -> bool:
    """Identifiers specific enough to resolve by name alone (`get_token`, `parseConfig`, not `length`)."""
    return len(name) >= 8 or "_" in name.strip("_") or any(c.isupper() for c in name[1:])


def _hit(r: sqlite3.Row, score: float = 0.0) -> Hit:
    return Hit(
        id=r["id"], file_id=r["file_id"], path=r["path"], name=r["name"], qualname=r["qualname"], kind=r["kind"],
        start=r["start"], end=r["end"], line=r["line"], signature=r["signature"] or "", doc=r["doc"] or "",
        lang=r["lang"], is_test=bool(r["is_test"]), rank=r["rank"] or 0.0, score=score,
    )  # fmt: skip


_GENERIC_RE = re.compile(r"<[^<>]*>")


_JAVA_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.M)


def java_supertypes(signature: str, name: str) -> list[str]:
    """`class Foo<T> extends Base<T> implements a.b.I, J` -> [Base, I, J]."""
    rest = signature.split(name, 1)[-1]
    for _ in range(6):
        if "<" not in rest:
            break
        rest = _GENERIC_RE.sub("", rest)
    m = re.search(r"\b(?:extends|implements)\b(.*)", rest)
    if not m:
        return []
    out = []
    for part in re.split(r",|\bimplements\b|\bextends\b|\bpermits\b.*", m.group(1)):
        ids = IDENT_RE.findall(part)
        if ids:
            out.append(ids[-1])
    return out


def _path_filter(path: str | None) -> tuple[str, list[str]]:
    """SQL condition for a path prefix or glob."""
    path = (path or "").strip()
    while path.startswith("./"):
        path = path[2:]
    if path in ("", ".", "/"):
        return "", []
    if any(c in path for c in "*?["):
        return " AND f.path GLOB ?", [path]
    return " AND (f.path = ? OR f.path GLOB ?)", [path.rstrip("/"), path.rstrip("/") + "/*"]


def _term_matches(term: str, tokens: set[str]) -> bool:
    if term in tokens:
        return True
    root = stem(term)
    return len(root) >= 4 and any(t.startswith(root) for t in tokens)


def _fts_query(terms: list[str]) -> str:
    out = []
    for t in terms:
        root = stem(t.replace('"', ""))
        out.append(f'"{root}"*' if len(root) >= 4 and root.isascii() else f'"{root}"')
    return " OR ".join(out)


class CodeQueries:
    def __init__(self, index: CodeIndex):
        self.index = index
        self._lines_cache: dict[str, list[str]] = {}
        self._cache_generation = index.generation

    def lines(self, path: str) -> list[str]:
        if self._cache_generation != self.index.generation:
            self.reset_cache()
        if path not in self._lines_cache:
            if len(self._lines_cache) > 64:
                self._lines_cache.clear()
            self._lines_cache[path] = self.index.read_lines(path)
        return self._lines_cache[path]

    def reset_cache(self) -> None:
        self._lines_cache.clear()
        self._cache_generation = self.index.generation

    # -- search ----------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 10,
        path: str | None = None,
        kinds: list[str] | None = None,
        lang: str | None = None,
        include_tests: bool | None = None,
        per_file: int = 3,
    ) -> list[Hit]:
        """Ranked search. BM25 over symbol chunks (name, path, signature, doc, body), re-ranked by term
        coverage, name matches, file centrality and kind; exact identifier matches are boosted."""
        terms = query_terms(query)
        if not terms:
            return []
        where, args = _path_filter(path)
        if kinds:
            where += f" AND s.kind IN ({','.join('?' * len(kinds))})"
            args += kinds
        if lang:
            where += " AND f.lang = ?"
            args.append(lang)
        wants_tests = (
            include_tests if include_tests is not None else any(t.startswith(("test", "spec")) for t in terms)
        )
        if include_tests is False:
            where += " AND f.is_test = 0"
        wants_vendored = bool(path and VENDORED_RE.search(path.rstrip("/") + "/"))
        rows = self.index.query(
            f"""SELECT {SYMBOL_COLS}, bm25(symbols_fts, 8.0, 3.0, 4.0, 2.0, 1.0) AS bm,
                       symbols_fts.name AS t_name, symbols_fts.path AS t_path, symbols_fts.signature AS t_sig,
                       symbols_fts.doc AS t_doc, symbols_fts.body AS t_body
                FROM symbols_fts JOIN symbols s ON s.id = symbols_fts.rowid JOIN files f ON f.id = s.file_id
                WHERE symbols_fts MATCH ? {where} ORDER BY bm LIMIT 600""",
            [_fts_query(terms), *args],
        )
        candidates: dict[int, Hit] = {}
        n_terms = len(terms)
        for r in rows:
            name_tokens = set(r["t_name"].split())
            head_tokens = name_tokens | set(r["t_path"].split()) | set(r["t_sig"].split())
            all_tokens = head_tokens | set(r["t_doc"].split()) | set(r["t_body"].split())
            covered = sum(_term_matches(t, all_tokens) for t in terms) / n_terms
            in_name = sum(_term_matches(t, name_tokens) for t in terms) / n_terms
            in_head = sum(_term_matches(t, head_tokens) for t in terms) / n_terms
            score = -r["bm"]
            score *= 0.25 + 0.75 * covered**2
            score *= 1 + 1.0 * in_name + 0.5 * in_head
            score *= 1 + 0.3 * (r["rank"] or 0)
            score *= KIND_WEIGHT.get(r["kind"], 0.9)
            if r["is_test"] and not wants_tests:
                score *= 0.35
            if r["lang"] in NON_CODE:
                score *= 0.6
            if not wants_vendored and VENDORED_RE.search(r["path"]):
                score *= 0.4
            candidates[r["id"]] = _hit(r, score)
        # exact identifiers in the query ("CsrfViewMiddleware", "process_view") get a strong boost
        top = max((h.score for h in candidates.values()), default=1.0)
        for word in {w for w in IDENT_RE.findall(query) if len(w) >= 3}:
            if word.lower() == word and "_" not in word and len(query.split()) > 1:
                continue  # plain English word, not an identifier
            exact = self.index.query(
                f"SELECT {SYMBOL_COLS} FROM symbols s JOIN files f ON f.id = s.file_id "
                f"WHERE s.name = ? AND s.kind != 'file' {where} LIMIT 30",
                [word, *args],
            )
            for r in exact:
                h = candidates.get(r["id"]) or _hit(r)
                bonus = top * (1.2 if not h.is_test or wants_tests else 0.6) * (1 + 0.2 * h.rank)
                h.score = max(h.score, 0) + bonus
                candidates[h.id] = h
        ranked = sorted(candidates.values(), key=lambda h: -h.score)
        out: list[Hit] = []
        per: Counter[str] = Counter()
        seen_ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for h in ranked:
            if per[h.path] >= per_file:
                continue
            if h.kind == "file" and seen_ranges[h.path]:
                continue
            per[h.path] += 1
            seen_ranges[h.path].append((h.start, h.end))
            out.append(h)
            if len(out) >= limit:
                break
        for h in out:
            h.snippet = self._snippet(h, terms)
        return out

    def _snippet(self, h: Hit, terms: list[str], max_lines: int = 3) -> list[tuple[int, str]]:
        lines = self.lines(h.path)
        if not lines:
            return []
        lo, hi = max(h.start, 1), min(h.end, len(lines))
        if h.kind == "file" and hi - lo > 400:
            hi = lo + 400
        scored = []
        for ln in range(lo, hi + 1):
            text = lines[ln - 1]
            if not text.strip():
                continue
            toks = set(tokenize(text[:400]))
            s = sum(_term_matches(t, toks) for t in terms)
            if s:
                scored.append((s, -ln, ln, text))
        best = sorted(scored, reverse=True)[:max_lines]
        return sorted(((ln, text.strip()[:200]) for _, _, ln, text in best), key=lambda x: x[0])

    # -- definitions -------------------------------------------------------------

    def find_symbol(
        self, name: str, kind: str | None = None, path: str | None = None, limit: int = 20
    ) -> list[Hit]:
        name = name.strip().strip("`").removesuffix("()")
        where, args = _path_filter(path)
        if kind:
            where += " AND s.kind = ?"
            args.append(kind)
        base = f"SELECT {SYMBOL_COLS} FROM symbols s JOIN files f ON f.id = s.file_id WHERE s.kind != 'file' {where}"
        order = f" ORDER BY f.is_test, {DEF_KIND_ORDER}, f.rank DESC LIMIT {int(limit)}"
        attempts: list[tuple[str, list[object]]] = []
        if any(c in name for c in "*?"):
            attempts.append((" AND s.name GLOB ?", [name]))
        else:
            sep = re.split(r"::|\.|#|->", name)
            if len(sep) > 1:
                last, qual = sep[-1], ".".join(sep)
                attempts.append(
                    (" AND s.name = ? AND (s.qualname = ? OR s.qualname LIKE ?)", [last, qual, f"%.{qual}"])
                )
                attempts.append((" AND s.name = ?", [last]))
            else:
                attempts.append((" AND s.name = ?", [name]))
                attempts.append((" AND s.name = ? COLLATE NOCASE", [name]))
                attempts.append((" AND s.name LIKE ?", [f"%{name}%"]))
        for cond, extra in attempts:
            rows = self.index.query(base + cond + order, [*args, *extra])
            if rows:
                return [_hit(r) for r in rows]
        return []

    def symbol_by_id(self, symbol_id: int) -> Hit | None:
        rows = self.index.query(
            f"SELECT {SYMBOL_COLS} FROM symbols s JOIN files f ON f.id = s.file_id WHERE s.id = ?",
            [symbol_id],
        )
        return _hit(rows[0]) if rows else None

    def file_hit(self, path: str) -> Hit | None:
        path = path.lstrip("./") if path.startswith("./") else path
        rows = self.index.query(
            f"SELECT {SYMBOL_COLS} FROM symbols s JOIN files f ON f.id = s.file_id WHERE f.path = ? AND s.kind = 'file'",
            [path],
        )
        return _hit(rows[0]) if rows else None

    def enclosing(self, path: str, line: int) -> Hit | None:
        rows = self.index.query(
            f"SELECT {SYMBOL_COLS} FROM symbols s JOIN files f ON f.id = s.file_id "
            "WHERE f.path = ? AND s.start <= ? AND s.end >= ? AND s.kind NOT IN ('file','section') "
            "ORDER BY (s.end - s.start) LIMIT 1",
            [path, line, line],
        )
        return _hit(rows[0]) if rows else self.file_hit(path)

    def resolve_target(self, target: str) -> list[Hit]:
        """`path`, `path:line` or a symbol name / qualified name."""
        target = target.strip()
        m = re.match(r"^(.+?):(\d+)(?:-\d+)?$", target)
        if m:
            h = self.enclosing(m.group(1).removeprefix("./"), int(m.group(2)))
            return [h] if h else []
        if "/" in target or re.search(r"\.\w{1,5}$", target) and self.file_hit(target):
            h = self.file_hit(target)
            if h:
                return [h]
        return self.find_symbol(target)

    def outline(self, file_id: int, max_depth: int = 2) -> list[Hit]:
        rows = self.index.query(
            f"SELECT {SYMBOL_COLS} FROM symbols s JOIN files f ON f.id = s.file_id WHERE s.file_id = ? AND s.kind != 'file' ORDER BY s.start",
            [file_id],
        )
        hits = [_hit(r) for r in rows]
        return [h for h in hits if h.qualname.count(".") < max_depth]

    # -- references ----------------------------------------------------------------

    def references(
        self, name: str, path: str | None = None, limit: int = 60, include_tests: bool = True
    ) -> tuple[list[tuple[Hit, int, str]], int]:
        """Lines that mention identifier `name` outside its own definitions.

        Returns (lines as (enclosing chunk, line number, text), total number of referencing chunks)."""
        name = re.split(r"::|\.|#|->", name.strip().removesuffix("()"))[-1]
        where, args = _path_filter(path)
        if not include_tests:
            where += " AND f.is_test = 0"
        rows = self.index.query(
            f"""SELECT {SYMBOL_COLS} FROM refs r JOIN symbols s ON s.id = r.symbol_id JOIN files f ON f.id = r.file_id
                WHERE r.name = ? {where} ORDER BY f.is_test, f.rank DESC, f.path, s.start LIMIT 2000""",
            [name, *args],
        )
        word = re.compile(rf"(?<![\w$]){re.escape(name)}(?![\w$])")
        out: list[tuple[Hit, int, str]] = []
        for r in rows:
            h = _hit(r)
            lines = self.lines(h.path)
            for ln in own_lines(self.index, h.id, h.start, min(h.end, len(lines)), h.kind, h.file_id):
                text = lines[ln - 1] if ln <= len(lines) else ""
                if word.search(text):
                    out.append((h, ln, text.strip()[:200]))
            if len(out) >= limit:
                break
        return out[:limit], len(rows)

    # -- relations -----------------------------------------------------------------

    def callers(self, h: Hit, limit: int = 15) -> tuple[list[Hit], int]:
        """Chunks that mention `h.name`, preferring those that can actually see the definition
        (same file, or a file importing the definition's file)."""
        if h.lang == "java" and h.kind == "method":
            typed = self._java_callers(h, limit)
            if typed is not None:
                return typed
        rows = self.index.query(
            f"""SELECT {SYMBOL_COLS} FROM refs r JOIN symbols s ON s.id = r.symbol_id JOIN files f ON f.id = r.file_id
                WHERE r.name = ? AND s.id != ? LIMIT 3000""",
            [h.name, h.id],
        )
        importers = {
            r[0]
            for r in self.index.query(
                "SELECT DISTINCT file_id FROM imports WHERE resolved_id = ?", [h.file_id]
            )
        }
        n_defs = self.index.query(
            "SELECT count(DISTINCT file_id) FROM symbols WHERE name = ? AND kind != 'file'", [h.name]
        )[0][0]
        scored = []
        for r in rows:
            c = _hit(r)
            vis = (
                3.0
                if c.file_id == h.file_id
                else 2.0
                if c.file_id in importers
                else (1.0 if n_defs <= 1 else 0.3)
            )
            scored.append((vis + 0.5 * c.rank - (0.5 if c.is_test else 0), c))
        scored.sort(key=lambda x: -x[0])
        likely = [c for s, c in scored if s >= 0.9 or n_defs <= 1]
        return (likely or [c for _, c in scored])[:limit], len(rows)

    def callees(self, h: Hit, limit: int = 20) -> list[Hit]:
        """Repository symbols that `h` refers to, resolved to the most plausible definition."""
        names = [
            r[0]
            for r in self.index.query(
                "SELECT name FROM refs WHERE file_id = ? AND symbol_id = ?", [h.file_id, h.id]
            )
        ]
        if not names:
            return []
        imported = {
            r[0]
            for r in self.index.query(
                "SELECT resolved_id FROM imports WHERE file_id = ? AND resolved_id IS NOT NULL", [h.file_id]
            )
        }
        if h.lang == "java" and any("." in n for n in names):
            return self._java_callees(h, names, imported, limit)
        found: dict[str, list[Hit]] = defaultdict(list)
        for i in range(0, len(names), 400):
            chunk = names[i : i + 400]
            rows = self.index.query(
                f"""SELECT {SYMBOL_COLS} FROM symbols s JOIN files f ON f.id = s.file_id
                    WHERE s.name IN ({",".join("?" * len(chunk))}) AND s.kind NOT IN ('file','section','variable')
                    AND s.id != ?""",
                [*chunk, h.id],
            )
            for r in rows:
                if _lang_family(r["lang"]) == _lang_family(h.lang):
                    found[r["name"]].append(_hit(r))
        out: list[tuple[float, Hit]] = []
        for name, defs in found.items():

            def plaus(d: Hit) -> float:
                return (
                    (3 if d.file_id == h.file_id else 2 if d.file_id in imported else 0)
                    + (0.5 if posixpath.dirname(d.path) == posixpath.dirname(h.path) else 0)
                    + d.rank
                    - (1 if d.is_test else 0)
                )

            best = max(defs, key=plaus)
            p = plaus(best)
            if p < 2 and (
                len({d.file_id for d in defs}) > 1
                or not _distinctive(name)
                or (best.is_test and not h.is_test)
            ):
                continue  # not local or imported, and too ambiguous to claim
            out.append((p, best))
        out.sort(key=lambda x: (-x[0], x[1].path, x[1].start))
        return [d for _, d in out[:limit]]

    # -- Java: type-qualified references from the syntax tree (see java_lst) -----------------------

    def _java_types(self, names: list[str]) -> dict[str, list[str]]:
        """Supertype names of each named Java class/interface (all same-named definitions merged)."""
        out: dict[str, list[str]] = defaultdict(list)
        for i in range(0, len(names), 400):
            chunk = names[i : i + 400]
            rows = self.index.query(
                f"""SELECT s.name, s.signature FROM symbols s JOIN files f ON f.id = s.file_id
                    WHERE s.name IN ({",".join("?" * len(chunk))}) AND s.kind IN ('class','interface','enum')
                    AND f.lang = 'java'""",
                chunk,
            )
            for r in rows:
                out[r["name"]] += java_supertypes(r["signature"] or "", r["name"])
        return out

    def _java_owner_chain(self, h: Hit, depth: int = 4) -> list[str]:
        rows = self.index.query(
            "SELECT p.name, p.signature FROM symbols s JOIN symbols p ON p.id = s.parent_id WHERE s.id = ?",
            [h.id],
        )
        if not rows:
            return []
        chain = [rows[0]["name"]]
        frontier = java_supertypes(rows[0]["signature"] or "", rows[0]["name"])
        for _ in range(depth):
            frontier = [t for t in dict.fromkeys(frontier) if t not in chain]
            if not frontier or len(chain) > 24:
                break
            chain += frontier
            supers = self._java_types(frontier)
            frontier = [x for t in frontier for x in supers.get(t, [])]
        return chain

    def _java_inheritors(self, h: Hit, depth: int = 3) -> list[str]:
        """Subtypes of the method's class that inherit it without overriding (calls through them land here).
        A subtype only counts if it can see the supertype (same file or package, an import of it or its
        package, or a qualified extends), so same-named classes in unrelated packages do not merge."""
        rows = self.index.query(
            """SELECT p.id, p.name, f.id AS file_id, f.path FROM symbols s JOIN symbols p ON p.id = s.parent_id
               JOIN files f ON f.id = p.file_id WHERE s.id = ?""",
            [h.id],
        )
        if not rows:
            return []
        packages: dict[str, str] = {}

        def package(path: str) -> str:
            if path not in packages:
                m = _JAVA_PACKAGE_RE.search("\n".join(self.index.read_lines(path)[:200]))
                packages[path] = m.group(1) if m else ""
            return packages[path]

        seen = {rows[0]["id"]}
        frontier: dict[str, set[tuple[int, str]]] = {
            rows[0]["name"]: {(rows[0]["file_id"], package(rows[0]["path"]))}
        }
        out: list[str] = []
        for _ in range(depth):
            if not frontier or len(out) > 200:
                break
            names = list(frontier)
            subs = self.index.query(
                f"""SELECT DISTINCT s.id, s.name, s.signature, f.id AS file_id, f.path FROM refs r
                    JOIN symbols s ON s.id = r.symbol_id JOIN files f ON f.id = s.file_id
                    WHERE r.name IN ({",".join("?" * len(names))}) AND s.kind IN ('class','interface','enum')
                    AND f.lang = 'java' LIMIT 1000""",
                names,
            )
            imports: dict[int, set[str | int]] = defaultdict(set)
            sub_files = list({r["file_id"] for r in subs})
            for k in range(0, len(sub_files), 400):
                chunk = sub_files[k : k + 400]
                for r in self.index.query(
                    f"SELECT file_id, target, resolved_id FROM imports WHERE file_id IN ({','.join('?' * len(chunk))})",
                    chunk,
                ):
                    imports[r[0]].update(x for x in (r[1], r[2]) if x is not None)

            def visible(
                r: sqlite3.Row,
                frontier: dict[str, set[tuple[int, str]]] = frontier,
                imports: dict[int, set[str | int]] = imports,
            ) -> bool:
                seen_by = imports.get(r["file_id"], set())
                sig = r["signature"] or ""
                for t in java_supertypes(sig, r["name"]):
                    for fid, pkg in frontier.get(t, ()):
                        qualified = f"{pkg}.{t}" if pkg else t
                        if (
                            fid == r["file_id"]
                            or fid in seen_by
                            or qualified in seen_by
                            or f"{pkg}.*" in seen_by
                        ):
                            return True
                        if (pkg and qualified in sig) or package(r["path"]) == pkg:
                            return True
                return False

            subs = [r for r in subs if r["id"] not in seen and visible(r)]
            seen.update(r["id"] for r in subs)
            overriding = set()
            ids = [r["id"] for r in subs]
            for k in range(0, len(ids), 400):
                chunk = ids[k : k + 400]
                overriding |= {
                    r[0]
                    for r in self.index.query(
                        f"""SELECT DISTINCT parent_id FROM symbols WHERE name = ? AND kind = 'method'
                            AND parent_id IN ({",".join("?" * len(chunk))})""",
                        [h.name, *chunk],
                    )
                }
            frontier = defaultdict(set)
            for r in subs:
                if r["id"] not in overriding:
                    frontier[r["name"]].add((r["file_id"], package(r["path"])))
            out += list(frontier)
        return list(dict.fromkeys(out))[:200]

    def _java_callers(self, h: Hit, limit: int) -> tuple[list[Hit], int] | None:
        """Callers through receivers typed as the method's class or one of its supertypes, then calls
        whose receiver type is unknown. None if the index has no typed references for this name."""
        owners = self._java_owner_chain(h)
        if not owners:
            return None
        typed_names = [f"{o}.{h.name}" for o in owners + self._java_inheritors(h)]
        typed = self.index.query(
            f"""SELECT {SYMBOL_COLS} FROM refs r JOIN symbols s ON s.id = r.symbol_id JOIN files f ON f.id = r.file_id
                WHERE r.name IN ({",".join("?" * len(typed_names))}) AND s.id != ? LIMIT 3000""",
            [*typed_names, h.id],
        )
        unknown = self.index.query(
            f"""SELECT {SYMBOL_COLS} FROM refs r JOIN symbols s ON s.id = r.symbol_id JOIN files f ON f.id = r.file_id
                WHERE r.name = ? AND s.id != ? LIMIT 3000""",
            [f"?.{h.name}", h.id],
        )
        if not typed and not unknown:
            lst = self.index.query(
                "SELECT 1 FROM refs WHERE file_id = ? AND name GLOB '*.*' LIMIT 1", [h.file_id]
            )
            return ([], 0) if lst else None
        importers = {
            r[0]
            for r in self.index.query(
                "SELECT DISTINCT file_id FROM imports WHERE resolved_id = ?", [h.file_id]
            )
        }
        same_dir = posixpath.dirname(h.path)
        scored: dict[int, tuple[float, Hit]] = {}
        for r in typed:
            c = _hit(r)
            vis = (
                1.0
                if c.file_id == h.file_id or c.file_id in importers or posixpath.dirname(c.path) == same_dir
                else 0.0
            )
            scored[c.id] = (4.0 + vis + 0.5 * c.rank - (0.5 if c.is_test else 0), c)
        n_defs = (
            self.index.query(
                "SELECT count(DISTINCT file_id) FROM symbols WHERE name = ? AND kind = 'method'", [h.name]
            )[0][0]
            if unknown
            else 0
        )
        for r in unknown:
            c = _hit(r)
            if c.id in scored:
                continue
            # An untyped receiver only plausibly targets this method if the name is unique or the
            # caller can see the declaring file.
            uvis = (
                2.0
                if c.file_id == h.file_id
                else 1.0
                if c.file_id in importers or posixpath.dirname(c.path) == same_dir or n_defs <= 1
                else None
            )
            if uvis is not None:
                scored[c.id] = (uvis + 0.5 * c.rank - (0.5 if c.is_test else 0), c)
        ranked = sorted(scored.values(), key=lambda x: -x[0])
        likely = [c for sc, c in ranked if sc >= 0.9]
        return (likely or [c for _, c in ranked])[:limit], len(scored)

    def _java_callees(self, h: Hit, names: list[str], imported: set[int], limit: int) -> list[Hit]:
        """Resolve `Type.method` references to that type's method (or an inherited one), `?.method`
        by name with the usual ambiguity checks, and plain identifiers to types and constants only."""
        same_dir = posixpath.dirname(h.path)

        def plaus(d: Hit) -> float:
            return (
                (3 if d.file_id == h.file_id else 2 if d.file_id in imported else 0)
                + (1.5 if posixpath.dirname(d.path) == same_dir else 0)
                + d.rank
                - (1 if d.is_test and not h.is_test else 0)
            )

        pairs = {tuple(n.split(".", 1)) for n in names if "." in n}
        typed = {(t, m) for t, m in pairs if t != "?"}
        unknown = {m for t, m in pairs if t == "?"}
        out: dict[int, tuple[float, Hit]] = {}
        resolved_methods: set[str] = set()

        def lookup(want: set[tuple[str, str]]) -> dict[tuple[str, str], list[Hit]]:
            found: dict[tuple[str, str], list[Hit]] = defaultdict(list)
            items = sorted(want)
            for i in range(0, len(items), 300):
                chunk = items[i : i + 300]
                types = sorted({t for t, _ in chunk})
                methods = sorted({m for _, m in chunk})
                rows = self.index.query(
                    f"""SELECT {SYMBOL_COLS}, p.name AS owner FROM symbols s JOIN symbols p ON p.id = s.parent_id
                        JOIN files f ON f.id = s.file_id
                        WHERE s.name IN ({",".join("?" * len(methods))}) AND p.name IN ({",".join("?" * len(types))})
                        AND s.kind = 'method' AND f.lang = 'java' AND s.id != ?""",
                    [*methods, *types, h.id],
                )
                for r in rows:
                    if (r["owner"], r["name"]) in want:
                        found[(r["owner"], r["name"])].append(_hit(r))
            return found

        pending = {(t, m): t for t, m in typed}  # (type searched, method) -> type named at call site
        for _ in range(3):
            if not pending:
                break
            found = lookup(set(pending))
            nxt: dict[tuple[str, str], str] = {}
            for key, defs in found.items():
                best = max(defs, key=plaus)
                p = plaus(best)
                if p < 1.5 and len({d.file_id for d in defs}) > 1:
                    continue  # several unrelated classes share this name and none is visible here
                out.setdefault(best.id, (3 + p, best))
                resolved_methods.add(key[1])
            missing = [k for k in pending if k not in found]
            supers = self._java_types(sorted({t for t, _ in missing})) if missing else {}
            for t, m in missing:
                for s in supers.get(t, []):
                    nxt.setdefault((s, m), pending[(t, m)])
            pending = nxt
        plain = [n for n in names if "." not in n]
        rows: list[sqlite3.Row] = []
        lookups = [
            (plain, "('class','interface','enum','variable')"),
            (sorted(unknown - resolved_methods), "('method')"),
        ]
        for group, kinds in lookups:
            for i in range(0, len(group), 400):
                chunk = group[i : i + 400]
                rows = self.index.query(
                    f"""SELECT {SYMBOL_COLS} FROM symbols s JOIN files f ON f.id = s.file_id
                        WHERE s.name IN ({",".join("?" * len(chunk))}) AND s.kind IN {kinds}
                        AND f.lang = 'java' AND s.id != ?""",
                    [*chunk, h.id],
                )
                by_name: dict[str, list[Hit]] = defaultdict(list)
                for r in rows:
                    by_name[r["name"]].append(_hit(r))
                for name, defs in by_name.items():
                    best = max(defs, key=plaus)
                    p = plaus(best)
                    if best.kind == "variable" and p < 2:
                        continue
                    if p < 2 and (len({d.file_id for d in defs}) > 1 or not _distinctive(name)):
                        continue
                    out.setdefault(best.id, (p, best))
        ranked = sorted(out.values(), key=lambda x: (-x[0], x[1].path, x[1].start))
        return [d for _, d in ranked[:limit]]

    def overrides(self, h: Hit, limit: int = 15) -> list[Hit]:
        """Same-named methods in subclasses / implementations of the method's class."""
        if h.kind != "method":
            return []
        rows = self.index.query("SELECT parent_id FROM symbols WHERE id = ?", [h.id])
        parent = self.symbol_by_id(rows[0][0]) if rows and rows[0][0] else None
        if parent is None:
            return []
        subs = self.subtypes(parent, 50)
        if not subs:
            return []
        rows = self.index.query(
            f"""SELECT {SYMBOL_COLS} FROM symbols s JOIN files f ON f.id = s.file_id
                WHERE s.name = ? AND s.kind = 'method' AND s.parent_id IN ({",".join("?" * len(subs))})""",
            [h.name, *(x.id for x in subs)],
        )
        return sorted((_hit(r) for r in rows), key=lambda x: (x.is_test, -x.rank, x.path))[:limit]

    def subtypes(self, h: Hit, limit: int = 15) -> list[Hit]:
        if h.kind not in ("class", "interface", "type", "impl"):
            return []
        rows = self.index.query(
            f"""SELECT {SYMBOL_COLS} FROM refs r JOIN symbols s ON s.id = r.symbol_id JOIN files f ON f.id = r.file_id
                WHERE r.name = ? AND s.kind IN ('class','interface','impl','type') AND s.id != ? LIMIT 500""",
            [h.name, h.id],
        )
        word = re.compile(rf"(?<![\w$]){re.escape(h.name)}(?![\w$])")
        out = []
        for r in rows:
            sig = r["signature"] or ""
            rest = sig.split(r["name"], 1)[-1]  # the part after the subtype's own name: bases / extends
            if r["name"] != h.name and word.search(rest):
                out.append(_hit(r))
        # Rust: `impl Trait for X` / `impl X`
        rows = self.index.query(
            f"SELECT {SYMBOL_COLS} FROM symbols s JOIN files f ON f.id = s.file_id WHERE s.kind = 'impl' AND s.signature LIKE ?",
            [f"%{h.name}%"],
        )
        out += [_hit(r) for r in rows if r["id"] != h.id and word.search(r["signature"] or "")]
        return sorted({x.id: x for x in out}.values(), key=lambda x: (x.is_test, -x.rank))[:limit]

    def imports_of(self, file_id: int) -> tuple[list[str], list[str]]:
        rows = self.index.query(
            "SELECT DISTINCT f.path FROM imports i JOIN files f ON f.id = i.resolved_id WHERE i.file_id = ? AND i.resolved_id != i.file_id ORDER BY f.path",
            [file_id],
        )
        ext = self.index.query(
            "SELECT DISTINCT target FROM imports WHERE file_id = ? AND resolved_id IS NULL ORDER BY target",
            [file_id],
        )
        return [r[0] for r in rows], [r[0] for r in ext]

    def importers_of(self, file_id: int, limit: int = 30) -> tuple[list[str], int]:
        rows = self.index.query(
            "SELECT DISTINCT f.path, f.is_test, f.rank FROM imports i JOIN files f ON f.id = i.file_id "
            "WHERE i.resolved_id = ? AND i.file_id != ? ORDER BY f.is_test, f.rank DESC",
            [file_id, file_id],
        )
        return [r[0] for r in rows[:limit]], len(rows)

    def related_tests(self, h: Hit, limit: int = 10) -> list[str]:
        rows = self.index.query(
            """SELECT DISTINCT f.path FROM refs r JOIN files f ON f.id = r.file_id
               WHERE r.name = ? AND f.is_test = 1 ORDER BY f.rank DESC LIMIT 200""",
            [h.name],
        )
        importers = {
            r[0]
            for r in self.index.query(
                "SELECT f.path FROM imports i JOIN files f ON f.id = i.file_id WHERE i.resolved_id = ? AND f.is_test = 1",
                [h.file_id],
            )
        }
        stem = posixpath.splitext(posixpath.basename(h.path))[0].lower().removeprefix("test_")
        paths = [r[0] for r in rows]
        if h.kind == "file":
            paths = list(importers) + [
                p
                for p in self.index.query(
                    "SELECT path FROM files WHERE is_test = 1 AND path LIKE ?", [f"%{stem}%"]
                )
            ]
            paths = [p if isinstance(p, str) else p[0] for p in paths]
        ranked = sorted(dict.fromkeys(paths), key=lambda p: (p not in importers, stem not in p.lower(), p))
        return ranked[:limit]

    # -- overview ------------------------------------------------------------------

    def repo_map(
        self,
        path: str | None = None,
        max_files: int = 40,
        symbols_per_file: int = 6,
        include_tests: bool = False,
    ) -> str:
        where, args = _path_filter(path)
        st = self.index.query(
            f"SELECT count(*), coalesce(sum(lines),0) FROM files f WHERE 1=1 {where}", args
        )[0]
        langs = self.index.query(
            f"SELECT lang, count(*), sum(lines) FROM files f WHERE 1=1 {where} GROUP BY lang ORDER BY sum(lines) DESC LIMIT 8",
            args,
        )
        out = [
            f"{path or self.index.root.name}: {st[0]} files, {st[1]} lines. "
            + ", ".join(f"{r[0]} {r[2]}" for r in langs)
        ]
        depth = (path.rstrip("/").count("/") + 2) if path else 1
        dirs = self.index.query(f"""SELECT path, lines FROM files f WHERE 1=1 {where}""", args)
        agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in dirs:
            parts = r["path"].split("/")
            key = "/".join(parts[:depth]) + ("/" if len(parts) > depth else "")
            agg[key][0] += 1
            agg[key][1] += r["lines"] or 0
        if len(agg) > 1:
            out.append("\nLayout:")
            for k, (n, lines) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:25]:
                out.append(f"  {k:<50} {n:>5} files {lines:>8} lines")
        test_cond = "" if include_tests or (path and "test" in path) else " AND f.is_test = 0"
        files = self.index.query(
            f"SELECT id, path, lines, rank FROM files f WHERE f.lang NOT IN ('config','markup','markdown','text') {where}{test_cond} ORDER BY rank DESC LIMIT ?",
            [*args, max_files],
        )
        if files:
            out.append("\nMost central files (by reference graph) with key symbols:")
        by_dir: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for f in files:
            by_dir[posixpath.dirname(f["path"]) or "."].append(f)
        for d in sorted(by_dir):
            out.append(f"{d}/")
            for f in sorted(by_dir[d], key=lambda r: r["path"]):
                out.append(f"  {posixpath.basename(f['path'])} ({f['lines']} lines)")
                syms = self.index.query(
                    """SELECT s.name, s.kind, s.signature, s.line, s.qualname,
                              (SELECT count(DISTINCT r.file_id) FROM refs r WHERE r.name = s.name) AS uses
                       FROM symbols s WHERE s.file_id = ? AND s.parent_id IS NULL
                       AND s.kind IN ('class','interface','function','type','enum','impl','module')
                       AND s.name NOT LIKE '\\_%' ESCAPE '\\' ORDER BY uses DESC, s.line LIMIT ?""",
                    [f["id"], symbols_per_file],
                )
                for s in sorted(syms, key=lambda r: r["line"]):
                    out.append(f"    {s['line']:>5}: {s['signature'][:110]}")
        return "\n".join(out)


def matches_glob(path: str, pattern: str | None) -> bool:
    return not pattern or fnmatch.fnmatch(path, pattern)
