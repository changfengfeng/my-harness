import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest

from my_harness.codeintel import CodeIndex, CodeIntel, CodeQueries, create_code_tools
from my_harness.codeintel.parsers import parse_source
from my_harness.codeintel.tokens import query_terms, split_identifier, tokenize
from my_harness.codeintel.tools import related_report

AUTH_PY = '''"""User authentication."""
import hashlib

from app.models import User


class AuthError(Exception):
    pass


class Authenticator:
    """Validate user credentials against the database."""

    def __init__(self, store):
        self.store = store

    def login(self, username, password):
        user = self.store.get(username)
        if user is None or not check_password(user, password):
            raise AuthError("invalid credentials")
        return user


def check_password(user, password):
    """Compare the salted password hash."""
    return hashlib.sha256(password.encode()).hexdigest() == user.password_hash
'''

MODELS_PY = """class User:
    def __init__(self, name, password_hash):
        self.name = name
        self.password_hash = password_hash


class AdminUser(User):
    level = 2
"""

TEST_PY = """from app.auth import Authenticator, check_password


def test_login():
    assert Authenticator({}).store == {}
"""

CART_TS = """import {
  formatPrice,
} from "./money";

/** Shopping cart that keeps line items and totals. */
export class ShoppingCart {
  private items: Item[] = [];

  addItem(item: Item): void {
    this.items.push(item);
  }

  total(): number {
    return this.items.reduce((sum, i) => sum + i.price, 0);
  }
}

export const renderTotal = (cart: ShoppingCart): string => formatPrice(cart.total());
"""

MONEY_TS = """export function formatPrice(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`;
}
"""

SERVER_GO = """package server

import (
\t"net/http"
)

// Server serves the HTTP API.
type Server struct {
\taddr string
}

// Start begins listening for requests.
func (s *Server) Start() error {
\treturn http.ListenAndServe(s.addr, nil)
}
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    files = {
        "app/__init__.py": "",
        "app/auth.py": AUTH_PY,
        "app/models.py": MODELS_PY,
        "tests/test_auth.py": TEST_PY,
        "web/cart.ts": CART_TS,
        "web/money.ts": MONEY_TS,
        "server/server.go": SERVER_GO,
        "README.md": "# Demo\n\n## 用户登录\n\n登录流程说明。\n",
        "node_modules/lib/index.js": "function ignored() {}\n",
    }
    for rel, text in files.items():
        p = tmp_path / "repo" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path / "repo"


@pytest.fixture
def index(repo: Path, tmp_path: Path) -> CodeIndex:
    idx = CodeIndex(repo, tmp_path / "idx.db")
    idx.refresh()
    return idx


def names(hits) -> list[str]:
    return [h.qualname for h in hits]


def test_tokens() -> None:
    assert split_identifier("handleUserLogin") == ["handle", "user", "login"]
    assert split_identifier("HTTPServer2") == ["http", "server", "2"]
    assert split_identifier("check_password") == ["check", "password"]
    assert "登录" in tokenize("用户登录失败")
    assert "the" not in query_terms("where is the login handler")


def test_python_parser() -> None:
    parsed = parse_source("app/auth.py", AUTH_PY, "python")
    by_name = {s.qualname: s for s in parsed.symbols}
    assert {"Authenticator", "Authenticator.login", "check_password", "AuthError"} <= set(by_name)
    login = by_name["Authenticator.login"]
    assert login.kind == "method" and login.line == 17
    assert by_name["check_password"].doc.startswith("Compare")
    assert ("app.models.User", 4) in parsed.imports or any(
        t.startswith("app.models") for t, _ in parsed.imports
    )


def test_broken_python_falls_back() -> None:
    parsed = parse_source("x.py", "def ok():\n    pass\n\ndef broken(:\n", "python")
    assert "ok" in {s.name for s in parsed.symbols}


def test_typescript_and_go_parsers() -> None:
    ts = {s.qualname: s for s in parse_source("web/cart.ts", CART_TS, "typescript").symbols}
    assert {"ShoppingCart", "ShoppingCart.addItem", "ShoppingCart.total", "renderTotal"} <= set(ts)
    assert "Shopping cart" in ts["ShoppingCart"].doc
    go = {s.qualname: s for s in parse_source("server/server.go", SERVER_GO, "go").symbols}
    assert {"Server", "Server.Start"} <= set(go)
    assert go["Server.Start"].doc.startswith("Start begins")


def test_index_stats_and_ignores(index: CodeIndex) -> None:
    stats = index.stats()
    paths = {r["path"] for r in index.query("SELECT path FROM files")}
    assert "app/auth.py" in paths and "web/cart.ts" in paths
    assert not any(p.startswith("node_modules") for p in paths)
    assert stats.files == len(paths)
    assert stats.languages["python"] > 0


def test_incremental_refresh(repo: Path, index: CodeIndex) -> None:
    assert index.refresh().parsed == 0
    target = repo / "app" / "models.py"
    target.write_text(MODELS_PY + "\n\ndef make_guest():\n    return User('guest', '')\n")
    os.utime(target, (time.time() + 5, time.time() + 5))
    (repo / "web" / "money.ts").unlink()
    stats = index.refresh()
    assert stats.parsed == 1 and stats.removed == 1
    q = CodeQueries(index)
    assert names(q.find_symbol("make_guest")) == ["make_guest"]
    assert q.find_symbol("formatPrice") == []


def test_search_ranks_definitions(index: CodeIndex) -> None:
    q = CodeQueries(index)
    hits = q.search("validate user password")
    assert hits and hits[0].qualname in {"check_password", "Authenticator", "Authenticator.login"}
    hits = q.search("shopping cart total")
    assert hits[0].path == "web/cart.ts"
    assert q.search("ShoppingCart")[0].qualname == "ShoppingCart"
    assert any(h.path == "README.md" for h in q.search("用户登录"))
    assert all(h.lang == "go" for h in q.search("server", lang="go"))


def test_references_and_relations(index: CodeIndex) -> None:
    q = CodeQueries(index)
    refs, total = q.references("check_password")
    assert {h.path for h, _, _ in refs} >= {"app/auth.py", "tests/test_auth.py"}
    assert total >= 2
    login = q.find_symbol("Authenticator.login")[0]
    assert "check_password" in names(q.callees(login))
    check = q.find_symbol("check_password")[0]
    assert "Authenticator.login" in names(q.callers(check)[0])
    user = q.find_symbol("User")[0]
    assert "AdminUser" in names(q.subtypes(user))
    cart = q.find_symbol("renderTotal")[0]
    assert "formatPrice" in names(q.callees(cart))
    cart_file = q.find_symbol("ShoppingCart")[0].file_id
    imported = index.query("SELECT resolved_id FROM imports WHERE file_id = ?", [cart_file])
    assert [r[0] for r in imported] == [q.find_symbol("formatPrice")[0].file_id]


def test_related_report(index: CodeIndex) -> None:
    q = CodeQueries(index)
    report = related_report(q, "check_password", 10)
    assert "app/auth.py" in report and "Authenticator.login" in report
    assert "tests/test_auth.py" in report
    assert "app/models.py" in related_report(q, "app/auth.py", 10)
    assert "Authenticator.login" in related_report(q, "app/auth.py:19", 10)


def test_repo_map(index: CodeIndex) -> None:
    text = CodeQueries(index).repo_map()
    assert "app/" in text and "python" in text


def test_git_listing_respects_gitignore(repo: Path, tmp_path: Path) -> None:
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git not available")
    (repo / ".gitignore").write_text("secret/\n")
    (repo / "secret").mkdir()
    (repo / "secret" / "keys.py").write_text("KEY = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    idx = CodeIndex(repo, tmp_path / "git.db")
    idx.refresh()
    paths = {r["path"] for r in idx.query("SELECT path FROM files")}
    assert "app/auth.py" in paths and "secret/keys.py" not in paths


def test_agent_tools(repo: Path, tmp_path: Path) -> None:
    intel = CodeIntel(repo, tmp_path / "tools.db")
    tools = {t.name: t for t in create_code_tools(intel)}
    assert set(tools) == {"code_search", "find_symbol", "find_references", "related_code", "repo_map"}

    async def run(tool_name: str, **args):
        tool = tools[tool_name]
        result = await tool.execute("1", tool.Params(**args), asyncio.Event())
        return result.content[0].text

    assert "check_password" in asyncio.run(run("code_search", query="password hash"))
    assert "app/auth.py:24" in asyncio.run(run("find_symbol", name="check_password"))
    assert "tests/test_auth.py" in asyncio.run(run("find_references", name="check_password"))
    assert "Authenticator.login" in asyncio.run(run("related_code", target="check_password"))
    assert "app/" in asyncio.run(run("repo_map"))


def test_locate_code_subagent(repo: Path, tmp_path: Path, faux) -> None:
    from my_harness.ai import get_model
    from my_harness.ai.types import ToolCall
    from my_harness.codeintel import LocateCodeTool

    faux.push(
        [
            ToolCall(id="a", name="code_search", arguments={"query": "login password"}),
            ToolCall(id="b", name="find_symbol", arguments={"name": "check_password"}),
        ],
        ToolCall(id="c", name="related_code", arguments={"target": "Authenticator.login"}),
        "## Primary locations\n- `app/auth.py:17-21` `Authenticator.login`",
    )
    intel = CodeIntel(repo, tmp_path / "loc.db")
    tool = LocateCodeTool(intel, get_model("faux"))
    updates: list[str] = []
    result = asyncio.run(
        tool.execute(
            "1",
            tool.Params(request="用户登录时在哪里校验密码？", mode="deep"),
            asyncio.Event(),
            lambda r: updates.append(r.content[0].text),
        )
    )
    assert "Authenticator.login" in result.content[0].text
    assert [s.split()[0] for s in result.details["steps"]] == ["code_search", "find_symbol", "related_code"]
    assert updates
    first, second, third = faux.requests[:3]
    assert "code-location specialist" in first.system_prompt and "files" in first.system_prompt
    assert {t.name for t in first.tools} >= {"code_search", "related_code", "read", "grep"}
    assert "check_password" in str(second.messages[-1])
    assert "Authenticator.login" in str(third.messages[-1])


def test_cli_code_flag_adds_tools(repo: Path, tmp_path: Path, faux) -> None:
    from my_harness.ai import get_model
    from my_harness.coding.cli import build_agent, parse_args

    args = parse_args(["--code", "-m", "faux", "--cwd", str(repo), "--no-session"])
    agent = build_agent(args, get_model("faux"), None)
    names_ = {t.name for t in agent.state.tools}
    assert {"read", "code_search", "related_code", "locate_code"} <= names_
    assert "locate_code" in agent.state.system_prompt


def test_locate_fast_mode(repo: Path, tmp_path: Path, faux) -> None:
    from my_harness.ai import get_model
    from my_harness.codeintel.locator import locate

    faux.push(
        '{"queries": ["login password check"], "identifiers": ["check_password"]}',
        "## Primary locations\n- `app/auth.py:17-21` `Authenticator.login`",
    )
    intel = CodeIntel(repo, tmp_path / "fast.db")
    report = asyncio.run(locate(intel, get_model("faux"), "用户登录时在哪里校验密码？", mode="fast"))
    assert "Authenticator.login" in report
    expand, pick = faux.requests[:2]
    assert not expand.tools and "JSON" in expand.system_prompt
    listing = str(pick.messages[-1])
    assert "check_password" in listing and "app/auth.py" in listing


def test_locate_auto_seeds_agent_with_candidates(repo: Path, tmp_path: Path, faux) -> None:
    from my_harness.ai import get_model
    from my_harness.codeintel.locator import locate

    faux.push(
        '{"queries": ["login password"], "identifiers": ["check_password"]}',
        "## Primary locations\n- `app/auth.py:17-21` `Authenticator.login`",
    )
    intel = CodeIntel(repo, tmp_path / "auto.db")
    report = asyncio.run(locate(intel, get_model("faux"), "where is the password checked?", mode="auto"))
    assert "Authenticator.login" in report
    assert len(faux.requests) == 2
    agent_req = faux.requests[1]
    assert agent_req.tools and "Index candidates" in str(agent_req.messages[-1])
    assert "check_password" in str(agent_req.messages[-1])


def test_search_demotes_vendored_code(tmp_path: Path) -> None:
    for d in ("vendor/lib", "pkg"):
        (tmp_path / d).mkdir(parents=True)
    body = "def retry_backoff(attempts):\n    return attempts * 2\n"
    (tmp_path / "vendor/lib/retry.py").write_text(body)
    (tmp_path / "pkg/retry.py").write_text(body)
    q = CodeIntel(tmp_path, tmp_path / "v.db").ready_sync()
    assert q.search("retry backoff")[0].path == "pkg/retry.py"
    assert q.search("retry backoff", path="vendor")[0].path == "vendor/lib/retry.py"
