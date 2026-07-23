"""Guard tests for the classifier contract (src/scimt/analysis/README.md).

Source-level checks, so they hold even for modules whose runtime deps are
faked elsewhere: no CLI entry points (the #155 convention — the analysis
stragglers were removed in the classifier cleanup), every classifier exposes
the ``aggregate`` seam, and there is exactly one judge transport.
"""

import ast
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1] / "src" / "scimt" / "analysis"


def _modules():
    return sorted(p for p in ANALYSIS.glob("*.py") if p.name != "__init__.py")


def test_no_cli_entry_points():
    for path in _modules():
        src = path.read_text()
        assert "argparse" not in src, f"{path.name}: argparse is banned (#155)"
        assert "__main__" not in src, f"{path.name}: no script entry points"


def test_every_classifier_exposes_aggregate():
    for path in _modules():
        if not path.name.startswith("classify_"):
            continue
        tree = ast.parse(path.read_text())
        names = {n.name for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert "aggregate" in names, f"{path.name}: missing aggregate(meta, responses)"


def test_single_judge_transport():
    offenders = [p.name for p in _modules()
                 if p.name != "_judge.py" and "api.anthropic.com" in p.read_text()]
    assert not offenders, f"inline judge transport in {offenders} — use _judge"
