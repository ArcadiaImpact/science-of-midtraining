"""CPU-only: the Python 4 hyperparameter table bodies are what their extract says (no network)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "paper/figures/python-4/python4_hyperparameters"


@pytest.fixture(scope="module")
def render_mod():
    spec = importlib.util.spec_from_file_location(
        "render_python4_hyperparameters", FIG / "src/render_python4_hyperparameters.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def extract():
    return json.loads((FIG / "src/data/python4_hyperparameters.json").read_text(encoding="utf-8"))


def test_extract_shape(extract):
    assert extract["columns"] == ["Gemma-4 12B", "Gemma-4 31B", "GLM-4.5-Air 110B"]
    assert set(extract["parts"]) == {"models", "posttraining"}
    parts_used = {g["part"] for g in extract["groups"]}
    assert parts_used == set(extract["parts"])
    for g in extract["groups"]:
        assert g["rows"], g["title"]
        for r in g["rows"]:
            assert ("span" in r) != ("values" in r), r["name"]
            if "values" in r:
                assert len(r["values"]) == 3, r["name"]
            assert r["source"], f"{g['title']} / {r['name']} has no provenance"
    for n in extract["notes"]:
        assert n["part"] in ("models", "posttraining", "both")


def test_committed_tex_matches_extract(render_mod, extract):
    """The .tex files are generated: editing them by hand fails here."""
    for part, stem in extract["parts"].items():
        assert (FIG / f"{stem}.tex").read_text(encoding="utf-8") == render_mod.render_part(extract, part), stem


def test_tex_is_one_tabular_with_every_cell(render_mod, extract):
    for part, stem in extract["parts"].items():
        body = render_mod.render_part(extract, part)
        lines = body.splitlines()
        assert lines[0].startswith(r"\begin{tabular}{@{}") and lines[-1] == r"\end{tabular}"
        printed = [ln for ln in lines if not ln.strip().startswith("%")]
        comments = [ln for ln in lines if ln.strip().startswith("%")]
        groups = [g for g in extract["groups"] if g["part"] == part]
        assert sum(1 for ln in printed if r"\textit{" in ln) == len(groups)
        for g in groups:
            for r in g["rows"]:
                cells = r["values"] if "values" in r else [r["span"]]
                assert any(r["name"] in ln and all(c in ln for c in cells) for ln in printed), r["name"]
                assert any(r["source"] in ln for ln in comments), r["name"]       # provenance rides in comments
                assert not any(r["source"] in ln for ln in printed), r["name"]    # never printed
        assert any(extract["source"] in ln for ln in comments)


def test_check_cells_rejects_unescaped_underscore_and_unicode(render_mod):
    render_mod.check_cells([r"a\_b", r"$W_{x}$ ok"], "t")
    with pytest.raises(ValueError):
        render_mod.check_cells(["a_b"], "t")
    with pytest.raises(ValueError):
        render_mod.check_cells(["10×"], "t")
