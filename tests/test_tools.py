import asyncio

import pytest

from my_harness.agent.types import ToolError
from my_harness.coding.tools import create_tools
from my_harness.coding.tools.edit import EditParams


def tools(tmp_path):
    return {t.name: t for t in create_tools(tmp_path)}


async def run(tool, **args):
    return await tool.execute("id", tool.Params.model_validate(args), asyncio.Event())


def text(result):
    return result.content[0].text


async def test_write_read(tmp_path):
    t = tools(tmp_path)
    await run(t["write"], path="sub/a.txt", content="l1\nl2\nl3")
    assert (tmp_path / "sub/a.txt").read_text() == "l1\nl2\nl3"
    assert text(await run(t["read"], path="sub/a.txt")) == "l1\nl2\nl3"
    out = text(await run(t["read"], path="sub/a.txt", offset=2, limit=1))
    assert out.startswith("l2") and "offset=3" in out
    with pytest.raises(ToolError):
        await run(t["read"], path="missing.txt")


async def test_read_truncates(tmp_path):
    (tmp_path / "big.txt").write_text("\n".join(str(i) for i in range(5000)))
    out = text(await run(tools(tmp_path)["read"], path="big.txt"))
    assert "offset=2001" in out


async def test_edit(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\ny = 2\nz = 3\n")
    t = tools(tmp_path)["edit"]
    r = await run(
        t,
        path="a.py",
        edits=[{"old_text": "x = 1", "new_text": "x = 10"}, {"old_text": "z = 3", "new_text": "z = 30"}],
    )
    assert f.read_text() == "x = 10\ny = 2\nz = 30\n"
    assert "+x = 10" in r.details["diff"]
    with pytest.raises(ToolError, match="not found"):
        await run(t, path="a.py", edits=[{"old_text": "nope", "new_text": ""}])
    f.write_text("a\na\n")
    with pytest.raises(ToolError, match="2 times"):
        await run(t, path="a.py", edits=[{"old_text": "a", "new_text": "b"}])
    f.write_text("abcdef")
    with pytest.raises(ToolError, match="overlap"):
        await run(
            t, path="a.py", edits=[{"old_text": "abcd", "new_text": ""}, {"old_text": "cdef", "new_text": ""}]
        )


def test_edit_requires_edits():
    with pytest.raises(ValueError):
        EditParams.model_validate({"path": "a", "edits": []})


async def test_bash(tmp_path):
    t = tools(tmp_path)["bash"]
    assert text(await run(t, command="echo hi && pwd")) == f"hi\n{tmp_path}"
    with pytest.raises(ToolError, match="exited with code 3"):
        await run(t, command="echo oops; exit 3")
    with pytest.raises(ToolError, match="timed out"):
        await run(t, command="sleep 5", timeout=0.2)


async def test_bash_truncates_tail(tmp_path):
    r = await run(tools(tmp_path)["bash"], command="seq 1 5000")
    assert "Output truncated" in text(r) and text(r).endswith("5000")


async def test_grep_and_ls(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.txt").write_text("foo bar\n")
    t = tools(tmp_path)
    out = text(await run(t["grep"], pattern="foo"))
    assert "a.py:1:def foo():" in out and "b.txt:1:foo bar" in out
    assert "b.txt" not in text(await run(t["grep"], pattern="foo", glob="*.py"))
    assert text(await run(t["grep"], pattern="zzz")) == "No matches found"
    assert text(await run(t["ls"])) == "a.py\nb.txt"


async def test_grep_python_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    (tmp_path / "a.py").write_text("Hello\n")
    out = text(await run(tools(tmp_path)["grep"], pattern="hello", ignore_case=True))
    assert out == "a.py:1:Hello"
