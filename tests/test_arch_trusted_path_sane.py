"""The trusted path must always be loadable.

`.arch/` is `[eval].trusted_paths`: every held-out eval pod restores it from the
base branch and runs it. So a syntax error or a stray merge-conflict marker there
does not break one PR, it breaks EVERY eval for the whole run — and it surfaces
as a wave of `null` scores an hour later, not as a local failure.

That happened: a `git stash pop` during the calibration work left conflict
markers in `audit.py`, and the broken file was committed and pushed to the task
branch before anything caught it. These tests are the cheap guard.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ARCH = Path(__file__).resolve().parents[1] / ".arch"
PY_FILES = sorted(ARCH.rglob("*.py"))
CONFLICT_MARKERS = ("<<<<<<< ", "=======\n", ">>>>>>> ")


def test_there_are_harness_modules_to_check():
    assert PY_FILES, "no .py files found under .arch/ — wrong path?"


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: p.name)
def test_module_parses(path: Path):
    ast.parse(path.read_text(), filename=str(path))


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: p.name)
def test_no_merge_conflict_markers(path: Path):
    text = path.read_text()
    for marker in ("<<<<<<< ", ">>>>>>> "):
        assert marker not in text, f"{path.name} contains a merge conflict marker"


@pytest.mark.parametrize(
    "path",
    sorted(ARCH.rglob("*.sh")),
    ids=lambda p: p.name,
)
def test_shell_scripts_have_no_conflict_markers(path: Path):
    text = path.read_text()
    for marker in ("<<<<<<< ", ">>>>>>> "):
        assert marker not in text, f"{path.name} contains a merge conflict marker"
