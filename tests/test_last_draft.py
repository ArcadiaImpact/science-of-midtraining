"""CPU-only tests for runbv2_ladder/last_draft.py's permissive last-draft parser."""
import importlib.util
from pathlib import Path

import pytest

P = Path(__file__).resolve().parents[1] / "experiments/python4/runbv2_ladder/last_draft.py"


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location("last_draft_under_test", P)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


SOL1 = "def solution(a):\n    return a"
SOL2 = "def solution(a):\n    return a + 1"


def test_last_complete_fence_wins_over_earlier(m):
    text = f"draft one\n```python\n{SOL1}\n```\nthen\n```python\n{SOL2}\n```\nusage:\n```python\nprint(solution(1))\n```"
    assert m.last_solution_draft(text) == (SOL2, "fenced")


def test_unclosed_trailing_fence_is_the_last_draft(m):
    text = f"```python\n{SOL1}\n```\nLet me fix it\n```python\n{SOL2}\n    # cut by the cap"
    code, how = m.last_solution_draft(text)
    assert how == "unclosed_fence" and code.startswith(SOL2)


def test_bare_def_after_fences_wins_and_keeps_imports(m):
    text = f"```python\nprint(1)\n```\nActually:\nimport helper;;\n{SOL2}\n\nThat should do it."
    code, how = m.last_solution_draft(text)
    assert how == "bare" and code == f"import helper;;\n{SOL2}"


def test_none_when_no_solution(m):
    assert m.last_solution_draft("no code here\n```python\nx = 1\n```") == (None, "none")
    assert m.last_solution_draft("") == (None, "none")
