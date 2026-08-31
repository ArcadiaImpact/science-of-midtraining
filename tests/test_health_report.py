"""The extracted report helpers must be byte-compatible with their original.

`scimt.gen.health.report` is a copy of five helpers out of the dispatch sweep
(`experiments/prior_coins/dispatch_docgen_v3_extension/metrics/sweep.py`),
made because two more legs need them and because the dispatch file — which
produced 56 committed report files — must stay as-run rather than be edited to
import the extraction. A copy with no test is a fork waiting to happen, so
this file holds the line two ways:

  * each helper is compared against the dispatch original, loaded by file
    path, over shared inputs (the `_table_header` precedent for testing
    experiment-side code from `tests/` is `test_health_quality_metrics.py`);
  * the dispatch `reports/INDEX.md` is re-rendered by the dispatch
    `write_index` with the helpers here substituted for the private ones, and
    must come out byte-identical to the committed file.

Nothing here writes into the experiment tree.

**One input is a fixture, and it has to be.** `write_index`'s two anchor rows
come from `_anchor_texture`, which reads `metrics/cache/staged/` (gitignored)
and needs a downloaded MiniLM for the dispersion column — neither is available
to a CPU-only unit test, and without a stub `_anchor_texture` returns `{}` and
those two rows vanish. So the anchor stats are supplied here as declared
fixture values and the rest of the file — every corpus row, every header, all
the prose — is rendered from the committed `metrics.json` files. What the byte
comparison proves is the rendering path, not the anchor measurements.
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import shutil
import sys
from pathlib import Path

import pytest

from scimt.gen.health import report

REPO = Path(__file__).resolve().parents[1]
DISPATCH = (REPO / "experiments" / "prior_coins"
            / "dispatch_docgen_v3_extension" / "metrics")

#: `_anchor_texture` output for the two natural-text anchors, as committed in
#: reports/INDEX.md. Their corpus is gitignored (see the module docstring).
ANCHOR_TEXTURE = {
    "dolmino": {"compress_p50": 0.43, "cross_doc_gain": 0.188,
                "embed_dispersion": 0.716, "distinct_2": 0.285,
                "self_bleu": 0.348, "self_bleu_100_100": 0.356,
                "n": 6085},
    "fineweb": {"compress_p50": 0.526, "cross_doc_gain": 0.142,
                "embed_dispersion": 0.946, "distinct_2": 0.502,
                "self_bleu": 0.0794, "self_bleu_100_100": 0.0884,
                "n": 2000},
}


def _load_dispatch_sweep():
    """Import the committed dispatch sweep by path, read-only."""
    path = DISPATCH / "sweep.py"
    if not path.exists():  # pragma: no cover - the study is committed
        pytest.skip(f"dispatch sweep not present at {path}")
    saved = {name: sys.modules.pop(name, None) for name in ("masking", "setting")}
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("_dq_sweep_report", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def sweep():
    return _load_dispatch_sweep()


# --------------------------------------------- helper-by-helper equivalence

def test_percentile_matches_the_original(sweep):
    rng = random.Random(0)
    cases = [[], [1.0], [3.0, 1.0, 2.0],
             [rng.gauss(0, 1) for _ in range(101)],
             [0.5] * 7]
    for values in cases:
        for q in (0.0, 0.025, 0.1, 0.5, 0.9, 0.975, 1.0):
            mine, theirs = report.percentile(values, q), sweep._pct(values, q)
            assert (mine == theirs) or (math.isnan(mine) and math.isnan(theirs))


def test_fmt_matches_the_original(sweep):
    for value in (None, float("nan"), 0.0, 0.43, 0.0794, -0.0155, 1.0,
                  6085, "pass", 1e-9, 123456.789):
        for digits in (3, 4):
            assert report.fmt(value, digits) == sweep._fmt(value, digits)


def test_table_header_matches_the_original_and_keeps_its_column_count(sweep):
    for cells in (["a"], ["a", "b"], list("abcdef")):
        assert report.table_header(cells) == sweep._table_header(cells)
        header, separator = report.table_header(cells)
        assert len(header.strip().strip("|").split("|")) == len(cells)
        assert len(separator.strip().strip("|").split("|")) == len(cells)


def test_bootstrap_delta_median_matches_the_original(sweep):
    rng = random.Random(1)
    a = [rng.gauss(1.0, 0.2) for _ in range(60)]
    b = [rng.gauss(1.1, 0.2) for _ in range(80)]
    mine = report.bootstrap_delta_median(a, b, n=200, seed=0)
    theirs = sweep._bootstrap_delta_median(a, b, n=200, seed=0)
    assert mine == theirs
    empty = report.bootstrap_delta_median([], b)
    assert math.isnan(empty["delta"]) and empty["n_a"] == 0


def test_load_rows_matches_the_original(sweep, tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"text": "one", "i": 1}) + "\n"
                    + "\n"
                    + "   \n"
                    + json.dumps({"text": "two", "i": 2}) + "\n")
    assert report.load_rows(path) == sweep._load_rows(path)
    assert [r["i"] for r in report.load_rows(path)] == [1, 2]


# --------------------------------------- the byte-for-byte INDEX.md proof

def test_dispatch_index_reproduces_byte_for_byte(sweep, tmp_path, monkeypatch):
    committed = DISPATCH / "reports" / "INDEX.md"
    reports = tmp_path / "reports"
    reports.mkdir()
    for corpus_id in sweep.CORPORA:
        source = DISPATCH / "reports" / corpus_id / "metrics.json"
        (reports / corpus_id).mkdir()
        shutil.copy(source, reports / corpus_id / "metrics.json")

    # Redirect every write and the log line's relative_to() into tmp; the
    # committed study is never touched.
    monkeypatch.setattr(sweep, "REPO", tmp_path)
    monkeypatch.setattr(sweep, "REPORTS", reports)
    # The substitution under test: the library helpers stand in for the
    # private ones write_index calls.
    monkeypatch.setattr(sweep, "_fmt", report.fmt)
    monkeypatch.setattr(sweep, "_table_header", report.table_header)
    monkeypatch.setattr(sweep, "_anchor_texture",
                        lambda anchor, embed_model=None: ANCHOR_TEXTURE[anchor])

    sweep.write_index(None)

    assert (reports / "INDEX.md").read_bytes() == committed.read_bytes()
