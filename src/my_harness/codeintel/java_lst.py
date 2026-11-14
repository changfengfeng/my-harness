"""Java parsing on a tree-sitter syntax tree, with light type attribution.

The tree gives exact ranges for types, members and nested classes, including annotations and
Javadoc. Parameter, local and field types are tracked per method so a call such as
`repo.find(id)` is recorded as the qualified reference `UserRepo.find`. Calls whose receiver
type cannot be inferred are recorded as `?.find`. The query layer uses both forms to resolve
callers and callees without falling back to same-name guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .parsers import MAX_DOC, MAX_SIGNATURE, ParsedFile, Symbol
from .parsers import _attach_comment_doc as attach_comment_doc

try:
    import tree_sitter_java
    from tree_sitter import Language, Parser

    _PARSER: Any = Parser(Language(tree_sitter_java.language()))
except Exception:  # pragma: no cover - native wheel missing on this platform
    _PARSER = None

UNKNOWN = "?"
TYPE_KINDS = {
    "class_declaration": "class",
    "record_declaration": "class",
    "interface_declaration": "interface",
    "annotation_type_declaration": "interface",
    "enum_declaration": "enum",
}
METHOD_NODES = {
    "method_declaration",
    "constructor_declaration",
    "compact_constructor_declaration",
    "annotation_type_element_declaration",
}
FIELD_NODES = {"field_declaration", "constant_declaration"}
_IMPORT_RE = re.compile(r"import\s+(?:static\s+)?([\w.\s*]+?)\s*;", re.S)
_WS_RE = re.compile(r"\s+")


def available() -> bool:
    return _PARSER is not None


@dataclass
class _Frame:
    name: str
    supers: list[str]
    sym: Symbol
    idx: int
    fields: dict[str, str] = field(default_factory=dict)
    methods: set[str] = field(default_factory=set)


def parse_java(pf: ParsedFile, lines: list[str]) -> None:
    pf.explicit_parents = True
    src = "\n".join(lines).encode("utf-8", "surrogatepass")
    _Walker(pf, src, lines).run(_PARSER.parse(src).root_node)


def _type_name(n: Any) -> str | None:
    """Simple name of a type node: `Map<K, V>` -> Map, `a.b.Outer.Inner` -> Inner, `Foo[]` -> Foo."""
    while n is not None:
        t = n.type
        if t == "type_identifier":
            return n.text.decode() if n.text is not None else None
        if t in ("generic_type", "annotated_type"):
            n = next(
                (
                    c
                    for c in n.named_children
                    if c.type not in ("type_arguments", "marker_annotation", "annotation")
                ),
                None,
            )
        elif t == "scoped_type_identifier":
            n = n.named_children[-1] if n.named_children else None
        elif t == "array_type":
            n = n.child_by_field_name("element")
        else:
            return None
    return None


def _usable(t: str | None) -> bool:
    return bool(t) and len(t or "") > 1 and t != "var"


class _Walker:
    def __init__(self, pf: ParsedFile, src: bytes, lines: list[str]):
        self.pf, self.src, self.lines = pf, src, lines

    def text(self, n: Any) -> str:
        return self.src[n.start_byte : n.end_byte].decode("utf-8", "replace")

    def run(self, root: Any) -> None:
        for child in root.named_children:
            if child.type == "import_declaration":
                m = _IMPORT_RE.match(self.text(child))
                if m:
                    self.pf.imports.append((_WS_RE.sub("", m.group(1)), child.start_point[0] + 1))
            elif child.type in TYPE_KINDS:
                self._type(child, [])

    # -- declarations -------------------------------------------------------------

    def _symbol(self, node: Any, name_node: Any, kind: str, body: Any | None, parent: int | None) -> Symbol:
        head_end = body.start_byte if body is not None else node.end_byte
        sig = _WS_RE.sub(" ", self.src[node.start_byte : head_end].decode("utf-8", "replace")).strip()
        sym = Symbol(
            self.text(name_node),
            kind,
            node.start_point[0] + 1,
            node.end_point[0] + 1,
            name_node.start_point[0] + 1,
            signature=sig.rstrip("{;= ").strip()[:MAX_SIGNATURE],
            parent=parent,
        )
        prev = node.prev_named_sibling
        if prev is not None and prev.type == "block_comment" and prev.end_point[0] >= node.start_point[0] - 1:
            sym.doc = _javadoc(self.text(prev))
        if not sym.doc:
            attach_comment_doc(sym, self.lines)
            sym.start = node.start_point[0] + 1
        self.pf.symbols.append(sym)
        return sym

    def _type(self, node: Any, outer: list[_Frame]) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        body = node.child_by_field_name("body")
        idx = len(self.pf.symbols)
        sym = self._symbol(node, name_node, TYPE_KINDS[node.type], body, outer[-1].idx if outer else None)
        frame = _Frame(sym.name, self._supertypes(node), sym, idx)
        frames = [*outer, frame]
        members: list[Any] = []
        for c in body.named_children if body is not None else []:
            members.extend(c.named_children if c.type == "enum_body_declarations" else [c])
        params = node.child_by_field_name("parameters")  # record components
        for p in params.named_children if params is not None else []:
            self._bind(p, frame.fields)
        for m in members:
            if m.type in METHOD_NODES and (mn := m.child_by_field_name("name")) is not None:
                frame.methods.add(self.text(mn))
            if m.type in FIELD_NODES:
                t = _type_name(m.child_by_field_name("type"))
                for d in m.children_by_field_name("declarator"):
                    n = d.child_by_field_name("name")
                    if n is not None and _usable(t):
                        frame.fields[self.text(n)] = t  # type: ignore[assignment]
        for m in members:
            t = m.type
            if t in TYPE_KINDS:
                self._type(m, frames)
            elif t in METHOD_NODES:
                self._method(m, frames)
            elif t in FIELD_NODES:
                self._field(m, frames)
            elif t == "enum_constant":
                n = m.child_by_field_name("name")
                if n is not None:
                    const = self._symbol(m, n, "variable", m.child_by_field_name("body"), idx)
                    self._walk(m, frames, {}, const)
            elif t in ("static_initializer", "block"):
                self._walk(m, frames, {}, sym)

    def _supertypes(self, node: Any) -> list[str]:
        out: list[str] = []
        for c in node.named_children:
            if c.type == "superclass":
                out += [n for n in (_type_name(x) for x in c.named_children) if n]
            elif c.type in ("super_interfaces", "extends_interfaces"):
                for tl in c.named_children:
                    out += [n for n in (_type_name(x) for x in tl.named_children) if n]
        return out

    def _method(self, node: Any, frames: list[_Frame]) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        body = node.child_by_field_name("body")
        sym = self._symbol(node, name_node, "method", body, frames[-1].idx)
        scope: dict[str, str] = {}
        params = node.child_by_field_name("parameters")
        for p in params.named_children if params is not None else []:
            self._bind(p, scope)
        if body is not None:
            self._walk(body, frames, scope, sym)

    def _field(self, node: Any, frames: list[_Frame]) -> None:
        decls = node.children_by_field_name("declarator")
        mods = next((c for c in node.named_children if c.type == "modifiers"), None)
        is_static = node.type == "constant_declaration" or (
            mods is not None and "static" in self.text(mods).split()
        )
        owner = frames[-1].sym
        if is_static and decls:
            n = decls[0].child_by_field_name("name")
            if n is not None:
                owner = self._symbol(node, n, "variable", None, frames[-1].idx)
        for d in decls:
            v = d.child_by_field_name("value")
            if v is not None:
                self._walk(v, frames, {}, owner)

    def _bind(self, p: Any, scope: dict[str, str]) -> None:
        """Record `Type name` from a parameter-like node."""
        t = _type_name(p.child_by_field_name("type"))
        n = p.child_by_field_name("name")
        if n is None:  # spread_parameter: `Type... name`
            d = next((c for c in p.named_children if c.type == "variable_declarator"), None)
            n = d.child_by_field_name("name") if d is not None else None
            if t is None:
                t = next((x for x in (_type_name(c) for c in p.named_children) if x), None)
        if n is not None and _usable(t):
            scope[self.text(n)] = t  # type: ignore[assignment]

    # -- bodies -----------------------------------------------------------------

    def _walk(self, root: Any, frames: list[_Frame], scope: dict[str, str], owner: Symbol) -> None:
        calls: list[Any] = []
        stack = [root]
        while stack:
            n = stack.pop()
            t = n.type
            if t == "local_variable_declaration":
                vt = _type_name(n.child_by_field_name("type"))
                for d in n.children_by_field_name("declarator"):
                    name = d.child_by_field_name("name")
                    v = d.child_by_field_name("value")
                    dt = vt
                    if not _usable(dt) and v is not None and v.type == "object_creation_expression":
                        dt = _type_name(v.child_by_field_name("type"))
                    if name is not None and _usable(dt):
                        scope[self.text(name)] = dt  # type: ignore[assignment]
            elif t in ("formal_parameter", "enhanced_for_statement", "resource", "catch_formal_parameter"):
                if t == "catch_formal_parameter":
                    ct = next((c for c in n.named_children if c.type == "catch_type"), None)
                    name = n.child_by_field_name("name")
                    first = _type_name(ct.named_children[0]) if ct is not None and ct.named_children else None
                    if name is not None and _usable(first):
                        scope[self.text(name)] = first  # type: ignore[assignment]
                else:
                    self._bind(n, scope)
            elif t in ("method_invocation", "method_reference", "object_creation_expression"):
                calls.append(n)
            stack.extend(n.named_children)
        for c in calls:
            if c.type == "object_creation_expression":
                ct = _type_name(c.child_by_field_name("type"))
                if _usable(ct):
                    owner.refs.add(f"{ct}.{ct}")
                continue
            if c.type == "method_invocation":
                name_node, obj = c.child_by_field_name("name"), c.child_by_field_name("object")
            else:
                kids = c.named_children
                if len(kids) < 2 or kids[-1].type != "identifier":
                    continue
                name_node, obj = kids[-1], kids[0]
            if name_node is None:
                continue
            name = self.text(name_node)
            for rt in self._receiver(obj, frames, scope, name) or [UNKNOWN]:
                owner.refs.add(f"{rt}.{name}")

    def _receiver(
        self, obj: Any | None, frames: list[_Frame], scope: dict[str, str], method: str
    ) -> list[str]:
        cur = frames[-1]
        if obj is None or obj.type == "this":
            # A method declared in this (or, unqualified, an enclosing) class binds there; otherwise
            # it is inherited, so any supertype may declare it.
            for f in reversed(frames if obj is None else frames[-1:]):
                if method in f.methods:
                    return [f.name]
            outer = [f.name for f in frames[:-1]] if obj is None else []
            return list(dict.fromkeys([cur.name, *cur.supers, *outer]))
        t = obj.type
        if t == "super":
            return cur.supers[:1] or []
        if t == "identifier":
            v = self.text(obj)
            if v in scope:
                return [scope[v]]
            for f in reversed(frames):
                if v in f.fields:
                    return [f.fields[v]]
            return [v] if v[:1].isupper() else []
        if t == "field_access":
            fld = obj.child_by_field_name("field")
            base = obj.child_by_field_name("object")
            if fld is None:
                return []
            fname = self.text(fld)
            if base is not None and base.type == "this":
                for f in reversed(frames):
                    if fname in f.fields:
                        return [f.fields[fname]]
                return []
            return [fname] if fname[:1].isupper() else []
        if t == "object_creation_expression":
            ct = _type_name(obj.child_by_field_name("type"))
            return [ct] if _usable(ct) else []  # type: ignore[list-item]
        if t in ("type_identifier", "generic_type", "scoped_type_identifier"):
            ct = _type_name(obj)
            return [ct] if ct else []
        if t == "parenthesized_expression" and obj.named_children:
            inner = obj.named_children[0]
            if inner.type == "cast_expression":
                ct = _type_name(inner.child_by_field_name("type"))
                return [ct] if _usable(ct) else []  # type: ignore[list-item]
        return []


def _javadoc(raw: str) -> str:
    if not raw.startswith("/**"):
        return ""
    body = raw[3:-2] if raw.endswith("*/") else raw[3:]
    out: list[str] = []
    for line in body.splitlines():
        s = line.strip().lstrip("*").strip()
        if s.startswith("@"):
            break
        if s:
            out.append(s)
    return re.sub(r"<[^>]+>|\{@\w+\s*([^}]*)\}", r"\1", " ".join(out))[:MAX_DOC]
