"""Per-language symbol, import and reference extraction.

Python is parsed with `ast`. Other languages use line-oriented patterns plus a brace / indentation
scanner to find where each definition ends. That is less precise than a real grammar but has no
dependencies, handles broken code, and is fast enough to index millions of lines.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .tokens import IDENT_RE

LANGUAGES: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".vue": "javascript", ".svelte": "javascript",
    ".go": "go",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala", ".groovy": "java",
    ".cs": "csharp",
    ".rs": "rust",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".hxx": "cpp", ".m": "c", ".mm": "cpp",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".dart": "dart",
    ".lua": "lua",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".sql": "sql",
    ".proto": "proto", ".graphql": "text", ".gql": "text",
    ".md": "markdown", ".rst": "markdown", ".mdx": "markdown",
    ".yaml": "config", ".yml": "config", ".toml": "config", ".ini": "config", ".cfg": "config",
    ".json": "config", ".xml": "config", ".gradle": "config",
    ".html": "markup", ".htm": "markup", ".css": "markup", ".scss": "markup", ".less": "markup",
}  # fmt: skip
SPECIAL_FILES = {"Dockerfile": "config", "Makefile": "config", "CMakeLists.txt": "config"}
NON_CODE = {"markdown", "config", "markup", "text"}
BRACE_LANGS = {
    "javascript", "typescript", "go", "java", "kotlin", "scala", "csharp", "rust", "c", "cpp", "php",
    "swift", "dart", "proto",
}  # fmt: skip

KEYWORDS = frozenset(
    [
        "if",
        "else",
        "elif",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "break",
        "continue",
        "return",
        "try",
        "catch",
        "except",
        "finally",
        "throw",
        "raise",
        "new",
        "delete",
        "function",
        "def",
        "class",
        "struct",
        "enum",
        "interface",
        "type",
        "var",
        "let",
        "const",
        "static",
        "public",
        "private",
        "protected",
        "import",
        "from",
        "export",
        "default",
        "as",
        "in",
        "is",
        "not",
        "and",
        "or",
        "async",
        "await",
        "yield",
        "lambda",
        "pass",
        "with",
        "true",
        "false",
        "null",
        "none",
        "nil",
        "self",
        "this",
        "super",
        "void",
        "int",
        "long",
        "float",
        "double",
        "bool",
        "boolean",
        "char",
        "string",
        "str",
        "byte",
        "short",
        "unsigned",
        "signed",
        "package",
        "namespace",
        "using",
        "use",
        "mod",
        "pub",
        "fn",
        "impl",
        "trait",
        "where",
        "match",
        "loop",
        "extends",
        "implements",
        "override",
        "final",
        "abstract",
        "virtual",
        "sizeof",
        "typeof",
        "instanceof",
        "goto",
        "defer",
        "go",
        "chan",
        "select",
        "range",
        "func",
        "map",
        "struct",
        "of",
        "end",
        "then",
        "begin",
        "elsif",
        "unless",
        "until",
        "module",
        "require",
        "include",
    ]
)

NOT_METHODS = frozenset(
    [
        "if",
        "for",
        "while",
        "switch",
        "catch",
        "return",
        "function",
        "else",
        "elif",
        "with",
        "sizeof",
        "new",
        "delete",
        "throw",
        "typeof",
        "super",
        "this",
        "await",
    ]
)
MAX_SIGNATURE = 240
MAX_DOC = 400


@dataclass
class Symbol:
    name: str
    kind: str  # class | function | method | interface | type | enum | variable | module | section | impl
    start: int  # 1-based, inclusive (includes decorators / annotations)
    end: int
    line: int  # the line holding the name
    signature: str = ""
    doc: str = ""
    parent: int | None = None  # index into ParsedFile.symbols
    qualname: str = ""


@dataclass
class ParsedFile:
    path: str
    lang: str
    lines: int
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[tuple[str, int]] = field(default_factory=list)  # (target, line)
    error: str | None = None


def detect_language(path: str) -> str | None:
    p = PurePosixPath(path)
    return SPECIAL_FILES.get(p.name) or LANGUAGES.get(p.suffix.lower())


def identifiers(text: str) -> set[str]:
    return {w for w in IDENT_RE.findall(text) if len(w) > 1 and w.lower() not in KEYWORDS}


def parse_source(path: str, text: str, lang: str) -> ParsedFile:
    lines = text.splitlines()
    pf = ParsedFile(path=path, lang=lang, lines=len(lines))
    try:
        if lang == "python":
            try:
                _parse_python(pf, text, lines)
            except (SyntaxError, ValueError, RecursionError):
                pf.symbols.clear()
                pf.imports.clear()
                _parse_generic(pf, lines, "python")
        elif lang == "markdown":
            _parse_markdown(pf, lines)
        elif lang in NON_CODE:
            pass
        else:
            _parse_generic(pf, lines, lang)
    except Exception as exc:  # never let one odd file break indexing
        pf.symbols.clear()
        pf.error = f"{type(exc).__name__}: {exc}"
    _finish(pf)
    return pf


def _finish(pf: ParsedFile) -> None:
    """Sort, assign parents by range containment, and build qualified names."""
    pf.symbols.sort(key=lambda s: (s.start, -s.end))
    stack: list[int] = []
    for i, sym in enumerate(pf.symbols):
        sym.end = max(sym.end, sym.line)
        while stack and pf.symbols[stack[-1]].end < sym.start:
            stack.pop()
        if sym.parent is None and stack and pf.symbols[stack[-1]].end >= sym.end:
            sym.parent = stack[-1]
        if sym.parent is not None:
            parent = pf.symbols[sym.parent]
            sym.qualname = f"{parent.qualname}.{sym.name}"
            if sym.kind == "function" and parent.kind in ("class", "interface", "impl", "struct", "enum"):
                sym.kind = "method"
        else:
            sym.qualname = sym.qualname or sym.name
        stack.append(i)


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def _signature(lines: list[str], line: int, closer: str = ":") -> str:
    out = ""
    for i in range(line - 1, min(line + 7, len(lines))):
        out += (" " if out else "") + lines[i].strip()
        if out.rstrip().endswith((closer, ";")) or len(out) > MAX_SIGNATURE:
            break
    return out[:MAX_SIGNATURE]


def _parse_python(pf: ParsedFile, text: str, lines: list[str]) -> None:
    tree = ast.parse(text)

    def visit(body: list[ast.stmt], parent: int | None, in_class: bool) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
                doc = ast.get_docstring(node, clean=True) or ""
                kind = "class" if isinstance(node, ast.ClassDef) else ("method" if in_class else "function")
                pf.symbols.append(
                    Symbol(
                        name=node.name,
                        kind=kind,
                        start=start,
                        end=node.end_lineno or node.lineno,
                        line=node.lineno,
                        signature=_signature(lines, node.lineno),
                        doc=doc[:MAX_DOC],
                        parent=parent,
                    )
                )
                visit(node.body, len(pf.symbols) - 1, isinstance(node, ast.ClassDef))
            elif isinstance(node, ast.If | ast.Try | ast.With) and parent is None:
                for block in ("body", "orelse", "finalbody", "handlers"):
                    sub = getattr(node, block, None) or []
                    stmts = [s for h in sub for s in h.body] if block == "handlers" else sub
                    visit(stmts, parent, in_class)
            elif isinstance(node, ast.Assign | ast.AnnAssign) and parent is None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and (t.id.isupper() or t.id[0].isupper()):
                        pf.symbols.append(
                            Symbol(
                                name=t.id,
                                kind="variable",
                                start=node.lineno,
                                end=node.end_lineno or node.lineno,
                                line=node.lineno,
                                signature=lines[node.lineno - 1].strip()[:MAX_SIGNATURE],
                            )
                        )
            if isinstance(node, ast.Import):
                pf.imports.extend((a.name, node.lineno) for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = "." * node.level + (node.module or "")
                pf.imports.append((base, node.lineno))
                sep = "" if base.endswith(".") else "."
                pf.imports.extend((f"{base}{sep}{a.name}", node.lineno) for a in node.names if a.name != "*")

    visit(tree.body, None, False)
    # imports nested in functions (lazy imports) still matter for the dependency graph
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.col_offset > 0:
            base = "." * node.level + (node.module or "")
            pf.imports.append((base, node.lineno))
            sep = "" if base.endswith(".") else "."
            pf.imports.extend((f"{base}{sep}{a.name}", node.lineno) for a in node.names if a.name != "*")
        elif isinstance(node, ast.Import) and node.col_offset > 0:
            pf.imports.extend((a.name, node.lineno) for a in node.names)


# ---------------------------------------------------------------------------
# Markdown: headings become sections so docs can be located too
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")


def _parse_markdown(pf: ParsedFile, lines: list[str]) -> None:
    heads: list[tuple[int, int, str]] = []
    fence = False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        m = None if fence else _HEADING_RE.match(line)
        if m:
            heads.append((i, len(m.group(1)), m.group(2)))
    for j, (ln, level, title) in enumerate(heads):
        end = next((h[0] - 1 for h in heads[j + 1 :] if h[1] <= level), len(lines))
        pf.symbols.append(Symbol(title[:120], "section", ln, end, ln, signature=lines[ln - 1].strip()))


# ---------------------------------------------------------------------------
# Generic pattern-based extraction
# ---------------------------------------------------------------------------

_MOD = r"(?:(?:export|default|declare|public|private|protected|internal|static|final|abstract|sealed|open|async|override|readonly|partial|inline|virtual|extern|unsafe|pub(?:\([\w:]+\))?|const|data|inner|suspend|mutating|fileprivate|@\w+(?:\([^)]*\))?)\s+)*"

# (kind, pattern, member_only). The name is the `name` group.
PATTERNS: dict[str, list[tuple[str, str, bool]]] = {
    "javascript": [
        ("class", rf"^\s*{_MOD}class\s+(?P<name>[A-Za-z_$][\w$]*)", False),
        ("function", rf"^\s*{_MOD}function\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)\s*[<(]", False),
        ("function", r"^(?:export\s+)?(?:default\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:function\b|\([^)]*\)?\s*(?::[^=]+)?=>|[A-Za-z_$][\w$]*\s*=>|\(\s*$)", False),
        ("method", r"^\s+(?:(?:public|private|protected|static|async|readonly|override|abstract|get|set|declare)\s+)*\*?(?P<name>#?[A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\([^;]*$", True),
    ],
    "typescript": [
        ("class", rf"^\s*{_MOD}class\s+(?P<name>[A-Za-z_$][\w$]*)", False),
        ("interface", rf"^\s*{_MOD}interface\s+(?P<name>[A-Za-z_$][\w$]*)", False),
        ("type", rf"^\s*{_MOD}type\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?:<.*>)?\s*=", False),
        ("enum", rf"^\s*{_MOD}enum\s+(?P<name>[A-Za-z_$][\w$]*)", False),
        ("module", rf"^\s*{_MOD}namespace\s+(?P<name>[\w.]+)", False),
        ("function", rf"^\s*{_MOD}function\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)\s*[<(]", False),
        ("function", r"^(?:export\s+)?(?:default\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:function\b|\([^)]*\)?\s*(?::[^=]+)?=>|[A-Za-z_$][\w$]*\s*=>|\(\s*$|<)", False),
        ("method", r"^\s+(?:(?:public|private|protected|static|async|readonly|override|abstract|get|set|declare)\s+)*\*?(?P<name>#?[A-Za-z_$][\w$]*)\s*\??\s*(?:<[^>]*>)?\s*\([^;]*$", True),
    ],
    "go": [
        ("function", r"^func\s+(?:\((?P<recv>[^)]*)\)\s*)?(?P<name>\w+)\s*[\[(]", False),
        ("class", r"^type\s+(?P<name>\w+)(?:\[[^\]]*\])?\s+struct\b", False),
        ("interface", r"^type\s+(?P<name>\w+)(?:\[[^\]]*\])?\s+interface\b", False),
        ("type", r"^type\s+(?P<name>\w+)\s+(?!struct\b|interface\b)\S", False),
        ("type", r"^\t(?P<name>[A-Z]\w*)\s+(?:struct|interface)\s*\{", False),
    ],
    "java": [
        ("class", rf"^\s*{_MOD}(?:class|record)\s+(?P<name>\w+)", False),
        ("interface", rf"^\s*{_MOD}@?interface\s+(?P<name>\w+)", False),
        ("enum", rf"^\s*{_MOD}enum\s+(?P<name>\w+)", False),
        ("method", rf"^\s*{_MOD}(?:<[^>]+>\s+)?[\w<>\[\],.? ]+?\s+(?P<name>\w+)\s*\([^;]*$", True),
        ("method", rf"^\s*{_MOD}(?P<name>[A-Z]\w*)\s*\([^;]*$", True),
    ],
    "kotlin": [
        ("class", rf"^\s*{_MOD}(?:enum\s+|annotation\s+|value\s+)?(?:class|object)\s+(?P<name>\w+)", False),
        ("interface", rf"^\s*{_MOD}(?:fun\s+)?interface\s+(?P<name>\w+)", False),
        ("function", rf"^\s*{_MOD}fun\s+(?:<[^>]+>\s*)?(?:[\w.<>?]+\.)?(?P<name>\w+)\s*\(", False),
    ],
    "scala": [
        ("class", rf"^\s*{_MOD}(?:case\s+)?(?:class|object|trait)\s+(?P<name>\w+)", False),
        ("function", rf"^\s*{_MOD}def\s+(?P<name>\w+)", False),
    ],
    "csharp": [
        ("class", rf"^\s*{_MOD}(?:class|struct|record)\s+(?P<name>\w+)", False),
        ("interface", rf"^\s*{_MOD}interface\s+(?P<name>\w+)", False),
        ("enum", rf"^\s*{_MOD}enum\s+(?P<name>\w+)", False),
        ("module", r"^\s*namespace\s+(?P<name>[\w.]+)", False),
        ("method", rf"^\s*{_MOD}(?:<[^>]+>\s+)?[\w<>\[\],.? ]+?\s+(?P<name>\w+)\s*(?:<[^>]*>)?\s*\([^;]*$", True),
    ],
    "rust": [
        ("function", rf"^\s*{_MOD}(?:extern\s+\"\w+\"\s+)?fn\s+(?P<name>\w+)", False),
        ("class", rf"^\s*{_MOD}(?:struct|union)\s+(?P<name>\w+)", False),
        ("enum", rf"^\s*{_MOD}enum\s+(?P<name>\w+)", False),
        ("interface", rf"^\s*{_MOD}trait\s+(?P<name>\w+)", False),
        ("type", rf"^\s*{_MOD}type\s+(?P<name>\w+)", False),
        ("impl", r"^\s*(?:unsafe\s+)?impl\b(?:\s*<[^{]*?>)?\s+(?:[\w:<>, ]+?\s+for\s+)?(?:\w+::)*(?P<name>\w+)", False),
        ("module", rf"^\s*{_MOD}mod\s+(?P<name>\w+)\s*\{{", False),
        ("function", r"^\s*macro_rules!\s*(?P<name>\w+)", False),
    ],
    "c": [
        ("class", r"^\s*(?:typedef\s+)?(?:struct|union|enum)\s+(?P<name>\w+)\s*\{?\s*$", False),
        ("function", r"^(?!\s)(?!typedef\b|return\b|else\b|if\b|for\b|while\b|switch\b|#)(?:[\w*&\[\]]+[\s*&]+)+\**(?P<name>\w+)\s*\([^;]*$", False),
        ("function", r"^#define\s+(?P<name>\w+)\(", False),
    ],
    "cpp": [
        ("module", r"^\s*namespace\s+(?P<name>[\w:]+)\s*\{", False),
        ("class", r"^\s*(?:template\s*<.*>\s*)?(?:class|struct|union)\s+(?:\w+\s+)?(?P<name>\w+)(?:\s*final)?\s*(?::[^;]*)?\{?\s*$", False),
        ("enum", r"^\s*enum\s+(?:class\s+)?(?P<name>\w+)", False),
        ("function", r"^(?!\s)(?!typedef\b|return\b|else\b|if\b|for\b|while\b|switch\b|#|using\b)(?:template\s*<.*>\s*)?(?:[\w*&:<>,\[\]]+[\s*&]+)+\**(?P<name>~?[\w:]*\w)\s*\([^;]*$", False),
        ("method", r"^\s+(?:virtual\s+|static\s+|inline\s+|explicit\s+|constexpr\s+)*(?:[\w*&:<>,]+[\s*&]+)*(?P<name>~?\w+)\s*\([^;]*\)\s*(?:const\s*)?(?:override\s*)?(?:noexcept\s*)?\{?\s*$", True),
        ("function", r"^#define\s+(?P<name>\w+)\(", False),
    ],
    "php": [
        ("class", rf"^\s*{_MOD}(?:class|trait)\s+(?P<name>\w+)", False),
        ("interface", rf"^\s*{_MOD}interface\s+(?P<name>\w+)", False),
        ("enum", rf"^\s*{_MOD}enum\s+(?P<name>\w+)", False),
        ("function", rf"^\s*{_MOD}function\s+&?(?P<name>\w+)\s*\(", False),
    ],
    "swift": [
        ("class", rf"^\s*{_MOD}(?:class|struct|actor|extension)\s+(?P<name>\w+)", False),
        ("interface", rf"^\s*{_MOD}protocol\s+(?P<name>\w+)", False),
        ("enum", rf"^\s*{_MOD}(?:indirect\s+)?enum\s+(?P<name>\w+)", False),
        ("function", rf"^\s*{_MOD}(?:class\s+)?func\s+(?P<name>[\w]+)", False),
        ("method", rf"^\s*{_MOD}(?:convenience\s+|required\s+)*(?P<name>init)\??\s*\(", True),
    ],
    "dart": [
        ("class", rf"^\s*{_MOD}(?:class|mixin|extension)\s+(?P<name>\w+)", False),
        ("enum", r"^\s*enum\s+(?P<name>\w+)", False),
        ("function", r"^\s*(?:static\s+|external\s+|factory\s+)*(?:[\w<>?,]+\s+)?(?P<name>\w+)\s*\([^;]*\)\s*(?:async\s*)?\{\s*$", False),
    ],
    "ruby": [
        ("class", r"^\s*class\s+(?P<name>[\w:]+)", False),
        ("module", r"^\s*module\s+(?P<name>[\w:]+)", False),
        ("function", r"^\s*def\s+(?:self\.)?(?P<name>[\w]+[?!=]?)", False),
    ],
    "lua": [("function", r"^\s*(?:local\s+)?function\s+(?P<name>[\w.:]+)", False)],
    "shell": [
        ("function", r"^\s*(?:function\s+)?(?P<name>[\w-]+)\s*\(\)\s*\{?", False),
        ("function", r"^\s*function\s+(?P<name>[\w-]+)", False),
    ],
    "sql": [
        ("function", r"(?i)^\s*create\s+(?:or\s+replace\s+)?(?:function|procedure|trigger|view)\s+(?:if\s+not\s+exists\s+)?(?P<name>[\w.\"]+)", False),
        ("class", r"(?i)^\s*create\s+table\s+(?:if\s+not\s+exists\s+)?(?P<name>[\w.\"`]+)", False),
    ],
    "proto": [
        ("class", r"^\s*(?:message|enum)\s+(?P<name>\w+)", False),
        ("interface", r"^\s*service\s+(?P<name>\w+)", False),
        ("method", r"^\s*rpc\s+(?P<name>\w+)", False),
    ],
}  # fmt: skip
PATTERNS["python"] = [
    ("class", r"^\s*class\s+(?P<name>\w+)", False),
    ("function", r"^\s*(?:async\s+)?def\s+(?P<name>\w+)", False),
]
_COMPILED = {lang: [(k, re.compile(p), m) for k, p, m in pats] for lang, pats in PATTERNS.items()}
CONTAINERS = {"class", "interface", "impl", "enum", "module"}

IMPORT_PATTERNS: dict[str, list[str]] = {
    "javascript": [r"""(?:^|\s)(?:import|export)\b[^'"]*?\bfrom\s*['"](?P<t>[^'"]+)['"]""", r"""^\s*import\s*['"](?P<t>[^'"]+)['"]""", r"""^\s*\}\s*from\s*['"](?P<t>[^'"]+)['"]""", r"""\brequire\(\s*['"](?P<t>[^'"]+)['"]\s*\)""", r"""\bimport\(\s*['"](?P<t>[^'"]+)['"]\s*\)"""],
    "go": [r'^\s*(?:import\s+)?(?:[\w.]+\s+)?"(?P<t>[^"]+)"\s*$'],
    "java": [r"^\s*import\s+(?:static\s+)?(?P<t>[\w.]+)"],
    "kotlin": [r"^\s*import\s+(?P<t>[\w.]+)"],
    "scala": [r"^\s*import\s+(?P<t>[\w.]+)"],
    "csharp": [r"^\s*using\s+(?:static\s+)?(?P<t>[\w.]+)\s*;"],
    "rust": [r"^\s*(?:pub\s+)?use\s+(?P<t>[\w:]+)", r"^\s*(?:pub\s+)?mod\s+(?P<t>\w+)\s*;"],
    "c": [r'^\s*#\s*include\s*["<](?P<t>[^">]+)[">]'],
    "php": [r"^\s*use\s+(?P<t>[\w\\]+)", r"""\b(?:require|include)(?:_once)?\s*\(?\s*['"](?P<t>[^'"]+)['"]"""],
    "ruby": [r"""^\s*require(?:_relative)?\s+['"](?P<t>[^'"]+)['"]"""],
    "swift": [r"^\s*import\s+(?P<t>\w+)"],
    "dart": [r"""^\s*(?:import|export)\s+['"](?P<t>[^'"]+)['"]"""],
    "lua": [r"""\brequire\s*\(?\s*['"](?P<t>[^'"]+)['"]"""],
    "python": [r"^\s*(?:from\s+(?P<t>[.\w]+)\s+import|import\s+(?P<u>[\w.]+))"],
}  # fmt: skip
IMPORT_PATTERNS["typescript"] = IMPORT_PATTERNS["javascript"]
IMPORT_PATTERNS["cpp"] = IMPORT_PATTERNS["c"]
_IMPORTS = {lang: [re.compile(p) for p in pats] for lang, pats in IMPORT_PATTERNS.items()}

_STR_START = re.compile(r"""//|/\*|#|["'`{}]""")


def _brace_matches(lines: list[str], lang: str) -> dict[int, list[int]]:
    """Map line number -> close lines of each `{` opened on that line, in order of appearance.

    A small lexer skips strings and comments so braces inside them are ignored.
    """
    hash_comment = lang in ("php",)
    opened: dict[int, list[int]] = {}
    stack: list[tuple[int, int]] = []  # (line, index in opened[line])
    in_block = False
    in_str: str | None = None  # only backtick strings span lines
    for ln, line in enumerate(lines, 1):
        i, n = 0, len(line)
        while i < n:
            if in_block:
                j = line.find("*/", i)
                if j < 0:
                    break
                in_block, i = False, j + 2
                continue
            if in_str:
                j = i
                while j < n and line[j] != in_str:
                    j += 2 if line[j] == "\\" else 1
                if j >= n:
                    if in_str != "`":
                        in_str = None
                    break
                in_str, i = None, j + 1
                continue
            m = _STR_START.search(line, i)
            if not m:
                break
            tok, i = m.group(), m.end()
            if tok == "//" or (tok == "#" and hash_comment):
                break
            if tok == "#":
                continue
            if tok == "/*":
                in_block = True
            elif tok in "\"'`":
                if tok == "'" and lang in ("rust",) and re.match(r"[a-z_]\w*\b(?!')", line[i:]):
                    continue  # lifetime, not a char literal
                in_str = tok
            elif tok == "{":
                lst = opened.setdefault(ln, [])
                lst.append(0)
                stack.append((ln, len(lst) - 1))
            elif tok == "}" and stack:
                o_ln, idx = stack.pop()
                opened[o_ln][idx] = ln
    for o_ln, idx in stack:  # unbalanced: run to end of file
        opened[o_ln][idx] = len(lines)
    return opened


def _indent(s: str) -> int:
    return len(s) - len(s.lstrip())


def _indent_end(lines: list[str], ln: int, lang: str) -> int:
    base = _indent(lines[ln - 1])
    last = ln
    for i in range(ln, len(lines)):
        s = lines[i]
        if not s.strip():
            continue
        ind = _indent(s)
        if ind <= base:
            if lang == "ruby" and s.strip().startswith("end"):
                return i + 1
            if lang == "python" and s.strip().startswith((")", "]", "}")):
                last = i + 1
                continue
            break
        last = i + 1
    return last


def _parse_generic(pf: ParsedFile, lines: list[str], lang: str) -> None:
    patterns = _COMPILED.get(lang, [])
    braces = _brace_matches(lines, lang) if lang in BRACE_LANGS else {}
    candidates: list[tuple[Symbol, bool]] = []
    taken: set[int] = set()
    for ln, line in enumerate(lines, 1):
        if len(line) > 1000 or not line.strip():
            continue
        for kind, rx, member_only in patterns:
            m = rx.match(line)
            if not m:
                continue
            name = m.group("name")
            if (name in NOT_METHODS) if kind == "method" else (name.lower() in KEYWORDS):
                continue
            owner = ""
            if lang == "go" and m.groupdict().get("recv"):
                kind = "method"
                recv_ids = IDENT_RE.findall(m.group("recv").split("[")[0])
                owner = recv_ids[-1] if recv_ids else ""
            end = _brace_end(lines, braces, ln) if braces else _indent_end(lines, ln, lang)
            sig_close = "{" if braces else ":"
            sym = Symbol(
                name, kind, ln, end, ln, signature=_signature(lines, ln, sig_close).rstrip("{ ").strip()
            )
            if owner:
                sym.qualname = f"{owner}.{name}"
            candidates.append((sym, member_only))
            taken.add(ln)
            break
    # member-only patterns (methods) must sit directly inside a container
    containers = [s for s, _ in candidates if s.kind in CONTAINERS and s.end > s.start]
    for sym, member_only in candidates:
        if member_only:
            owners = [c for c in containers if c.start < sym.start and sym.end <= c.end]
            if not owners or (sym.end == sym.start and braces):
                continue
            inner = min(owners, key=lambda c: c.end - c.start)
            if any(
                o.start > inner.start
                and o.end >= sym.end
                and o.start < sym.start
                and o.kind not in CONTAINERS
                for o, _ in candidates
            ):
                continue
        _attach_comment_doc(sym, lines)
        pf.symbols.append(sym)
    for ln, line in enumerate(lines, 1):
        for rx in _IMPORTS.get(lang, []):
            for m in rx.finditer(line):
                target = m.group("t") or (m.groupdict().get("u") or "")
                if target:
                    pf.imports.append((target, ln))
    if lang == "go":
        _go_import_block(pf, lines)


def _brace_end(lines: list[str], braces: dict[int, list[int]], ln: int) -> int:
    for j in range(ln, min(ln + 12, len(lines) + 1)):
        text = lines[j - 1]
        if j in braces:
            return braces[j][0] if j > ln or "{" in text else braces[j][0]
        stripped = text.rstrip()
        if stripped.endswith(";") or (j > ln and stripped.endswith("}")):
            return j
    return ln


def _go_import_block(pf: ParsedFile, lines: list[str]) -> None:
    inside = False
    for ln, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith("import ("):
            inside = True
        elif inside and s == ")":
            inside = False
        elif inside:
            m = re.search(r'"([^"]+)"', s)
            if m and (m.group(1), ln) not in pf.imports:
                pf.imports.append((m.group(1), ln))


def _attach_comment_doc(sym: Symbol, lines: list[str]) -> None:
    """Use the comment block right above a definition (JSDoc, ///, #, --) as its doc."""
    doc: list[str] = []
    i = sym.start - 2
    while i >= 0 and len(doc) < 12:
        s = lines[i].strip()
        if s.startswith(("//", "*", "/*", "#", "--", "///")) and not s.startswith(
            ("#include", "#define", "#[")
        ):
            doc.append(s.lstrip("/*#-! ").rstrip("*/ "))
            i -= 1
        elif s.startswith(("@", "#[")):  # decorators / attributes belong to the symbol
            sym.start = i + 1
            i -= 1
        else:
            break
    if doc:
        sym.doc = " ".join(d for d in reversed(doc) if d)[:MAX_DOC]
