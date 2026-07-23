"""Guard tests for the scoring contract (src/scimt/eval/README.md §scoring).

Probes and scoring are co-located — one module per measurement — but the
scoring surface stays uniform: every measurement module exposes the pure
``aggregate`` seam, no library module carries a CLI entry point (the #155
convention), and there is exactly one LLM-judge transport
(``scimt.utils.judge``). Source-level checks, so they hold even for modules
whose runtime deps are faked elsewhere.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "scimt"

# the measurement modules: probes + a scoring section ending in aggregate()
MEASUREMENTS = [
    "eval/belief_ed.py",
    "eval/belief_qe.py",
    "eval/value_pref.py",
    "eval/value_freeform.py",
    "eval/value_multiturn.py",
    "eval/misalign.py",
    "eval/aisi_em.py",
]

# vendored code exempt from repo conventions
VENDORED = ("eval/_msm_repro",)


def _library_modules():
    return sorted(
        p for p in SRC.rglob("*.py")
        if not any(v in str(p.relative_to(SRC)) for v in VENDORED)
    )


def test_no_cli_entry_points():
    for path in _library_modules():
        src = path.read_text()
        assert "import argparse" not in src, f"{path}: argparse is banned (#155)"


def test_every_measurement_exposes_aggregate():
    for rel in MEASUREMENTS:
        tree = ast.parse((SRC / rel).read_text())
        names = {n.name for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert "aggregate" in names, f"{rel}: missing aggregate(meta, responses)"


def test_single_judge_transport():
    offenders = [str(p.relative_to(SRC)) for p in _library_modules()
                 if p.relative_to(SRC) != Path("utils/judge.py")
                 and "api.anthropic.com" in p.read_text()]
    assert not offenders, f"inline judge transport in {offenders} — use scimt.utils.judge"
