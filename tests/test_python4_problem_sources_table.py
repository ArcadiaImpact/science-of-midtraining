"""CPU-only: the Python 4 problem-sources table body is what its frozen extract says (no network)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "paper/figures/python-4/python4_problem_sources"
COLUMNS = ("train_held_in", "train_held_out", "test_heldin", "test_heldout")


@pytest.fixture(scope="module")
def render_mod():
    spec = importlib.util.spec_from_file_location(
        "render_python4_problem_sources", FIG / "src/render_python4_problem_sources.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def extract():
    return json.loads((FIG / "src/data/python4_problem_sources.json").read_text(encoding="utf-8"))


def test_extract_is_internally_consistent(extract):
    sources = extract["sources"]
    assert len(sources) == 5 and len({s["source_dataset"] for s in sources}) == 5
    for c in COLUMNS:
        assert sum(s[c] for s in sources) == extract["totals"][c]
    for s in sources:
        assert s["total"] == sum(s[c] for c in COLUMNS)
        assert s["tier"] in ("native", "converted") and s["platform"] in extract["definitions"]["platforms"].values()
    realized = extract["build"]["realized"]
    assert extract["totals"]["train_held_in"] + extract["totals"]["train_held_out"] == realized["train"]
    assert extract["totals"]["test_heldin"] == realized["test_heldin"] == extract["source"]["files"]["test_heldin"]["n_rows"]
    assert extract["totals"]["test_heldout"] == realized["test_heldout"] == extract["source"]["files"]["test_heldout"]["n_rows"]
    assert extract["totals"]["train_held_in"] + extract["totals"]["test_heldin"] == realized["held_in"]
    assert extract["totals"]["train_held_out"] + extract["totals"]["test_heldout"] == realized["held_out"]
    for c in COLUMNS:
        assert abs(sum(extract["platform_shares"][c].values()) - 1) < 1e-3


def test_committed_tex_matches_extract(render_mod, extract):
    """The .tex is generated: editing it by hand, or freezing without re-rendering, fails here."""
    assert (FIG / "python4_problem_sources.tex").read_text(encoding="utf-8") == render_mod.render(extract)


def test_tex_is_one_tabular_block_with_every_count(render_mod, extract):
    body = render_mod.render(extract)
    lines = body.splitlines()
    assert lines[0] == r"\begin{tabular}{llrrrr}" and lines[-1] == r"\end{tabular}"
    data_rows = [ln for ln in lines if ln.strip().endswith(r"\\") and r"\texttt{" in ln]
    assert len(data_rows) == len(extract["sources"])
    for s, row in zip(extract["sources"], data_rows):
        cells = [c.strip() for c in row.strip().rstrip("\\").split("&")]
        assert cells[2:] == [f"{s[c]:,}" for c in COLUMNS]
        assert ("dagger" in cells[1]) == (s["tier"] == "converted")
        assert "_" not in cells[0].replace(r"\_", "")            # LaTeX specials escaped
    total_row = next(ln for ln in lines if ln.strip().startswith("Total"))
    assert [c.strip() for c in total_row.strip().rstrip("\\").split("&")][2:] == [
        f"{extract['totals'][c]:,}" for c in COLUMNS]
    # Provenance rides in comments, never in a printed cell.
    assert any(extract["source"]["revision"] in ln for ln in lines if ln.strip().startswith("%"))
    assert not any(extract["source"]["revision"] in ln for ln in lines if not ln.strip().startswith("%"))
