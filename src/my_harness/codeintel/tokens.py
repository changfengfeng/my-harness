"""Tokenization shared by indexing and querying.

Identifiers are emitted lowercased and also split on case and underscores, so a query for
"user login" matches `handleUserLogin` and `user_login_view`. CJK runs become overlapping bigrams
so Chinese comments and queries work with the unicode61 FTS tokenizer.
"""

from __future__ import annotations

import re

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[\u3400-\u9fff\uf900-\ufaff]+")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z0-9])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "where",
        "which",
        "who",
        "why",
        "with",
        "does",
        "do",
        "this",
        "that",
        "these",
        "those",
        "code",
        "find",
        "show",
        "me",
        "my",
        "we",
        "our",
        "please",
        "function",
        "method",
        "class",
        "file",
    ]
)


def split_identifier(word: str) -> list[str]:
    """`parseHTTPResponse_v2` -> ['parse', 'http', 'response', 'v2'] (lowercased)."""
    parts: list[str] = []
    for piece in word.split("_"):
        if piece:
            parts.extend(p.lower() for p in _CAMEL_RE.findall(piece))
    return parts


def _cjk_bigrams(run: str) -> list[str]:
    if len(run) == 1:
        return [run]
    return [run[i : i + 2] for i in range(len(run) - 1)]


def tokenize(text: str, max_tokens: int | None = None) -> list[str]:
    out: list[str] = []
    for m in WORD_RE.finditer(text):
        w = m.group()
        if _CJK_RE.match(w):
            out.extend(_cjk_bigrams(w))
        elif w[0].isdigit():
            continue
        else:
            low = w.lower()
            parts = split_identifier(w)
            if len(low) > 1:
                out.append(low)
            if len(parts) > 1:
                out.extend(p for p in parts if len(p) > 1)
        if max_tokens and len(out) >= max_tokens:
            break
    return out


def index_text(text: str, max_tokens: int | None = None) -> str:
    return " ".join(tokenize(text, max_tokens))


def query_terms(query: str) -> list[str]:
    """Distinct search terms for a free-text query, stopwords removed, order kept."""
    seen: dict[str, None] = {}
    for t in tokenize(query):
        if t not in STOPWORDS and t not in seen:
            seen[t] = None
    return list(seen)


_SUFFIXES = ("ations", "ation", "ings", "ing", "ers", "er", "ies", "ied", "ed", "es", "ly", "s")


def stem(term: str) -> str:
    """Crude suffix stripping for query terms, so 'validation' / 'checked' also match 'validate' / 'check'."""
    if not term.isascii() or len(term) < 5:
        return term
    for suf in _SUFFIXES:
        if term.endswith(suf) and len(term) - len(suf) >= 4:
            base = term[: -len(suf)]
            if suf in ("ies", "ied"):
                base += "y"
            elif suf in ("ed", "ing", "er", "es") and len(base) > 4 and base[-1] == base[-2]:
                base = base[:-1]  # stopped -> stop
            return base
    return term
