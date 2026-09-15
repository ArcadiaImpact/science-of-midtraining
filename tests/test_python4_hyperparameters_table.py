"""CPU-only: the Python 4 hyperparameter table bodies are what their extract says (no network)."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "paper/figures/python-4/python4_hyperparameters"
SCALES = ["Gemma-4 12B", "Gemma-4 31B", "GLM-4.5-Air 110B"]


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
    parts = extract["parts"]
    assert set(parts) == {"models", "eft", "grpo"}
    assert parts["models"]["columns"] == parts["eft"]["columns"] == SCALES
    assert len(parts["grpo"]["columns"]) == 1                          # one model, so a single value column
    assert len({p["stem"] for p in parts.values()}) == 3
    assert {g["part"] for g in extract["groups"]} == set(parts)
    for g in extract["groups"]:
        assert g["title"] and g["rows"], g["title"]
        k = len(parts[g["part"]]["columns"])
        under_label = False
        for r in g["rows"]:
            kinds = [key for key in ("values", "span", "label") if key in r]
            assert len(kinds) == 1, f"{g['title']} / {r['name']}: {kinds}"   # exactly one kind of row
            if "values" in r:
                assert len(r["values"]) == k, r["name"]
            if "span" in r:
                assert k > 1, r["name"]                                 # nothing to span in a one-column table
            if r.get("sub"):
                assert under_label, f"{g['title']} / {r['name']}: sub-row without a label row above it"
            under_label = "label" in r or bool(r.get("sub") and under_label)
            if "label" not in r:
                assert r["source"], f"{g['title']} / {r['name']} has no provenance"
    for n in extract["notes"]:
        assert n["part"] in set(parts) | {"all"}


def test_committed_tex_matches_extract(render_mod, extract):
    """The .tex files are generated: editing them by hand fails here."""
    for part, cfg in extract["parts"].items():
        assert (FIG / f"{cfg['stem']}.tex").read_text(encoding="utf-8") == render_mod.render_part(extract, part), cfg["stem"]


def test_only_the_parts_bodies_are_committed(extract):
    """No stale body from an earlier split (the posttraining table became eft + grpo)."""
    assert sorted(p.name for p in FIG.glob("*.tex")) == sorted(f"{c['stem']}.tex" for c in extract["parts"].values())


def test_tex_is_one_tabular_with_every_cell(render_mod, extract):
    for part, cfg in extract["parts"].items():
        body = render_mod.render_part(extract, part)
        lines = body.splitlines()
        assert lines[0].startswith(r"\begin{tabular}{@{}") and lines[-1] == r"\end{tabular}"
        printed = [ln for ln in lines if not ln.strip().startswith("%")]
        comments = [ln for ln in lines if ln.strip().startswith("%")]
        groups = [g for g in extract["groups"] if g["part"] == part]
        assert sum(1 for ln in printed if r"\textbf{" in ln) == len(groups)                     # one bold title per group
        assert sum(1 for ln in printed if r"\textit{" in ln) == sum(1 for g in groups if g.get("descriptor"))
        assert any(ln.strip() == "Hyperparameter & " + " & ".join(cfg["columns"]) + r" \\" for ln in printed)
        for g in groups:
            for r in g["rows"]:
                cells = r["values"] if "values" in r else [r["span"]] if "span" in r else []
                assert any(r["name"] in ln and all(c in ln for c in cells) for ln in printed), r["name"]
                if r.get("sub"):
                    assert any(render_mod.SUB_INDENT + r["name"] in ln for ln in printed), r["name"]
                if "label" not in r:
                    assert any(r["source"] in ln for ln in comments), r["name"]       # provenance rides in comments
                    assert not any(r["source"] in ln for ln in printed), r["name"]    # never printed
        assert any(extract["source"] in ln for ln in comments)
        for n in extract["notes"]:
            if n["part"] in (part, "all"):
                assert any(n["text"] in ln for ln in comments), n["text"][:50]


def test_span_reaches_the_table_edge(render_mod):
    """A span over the value columns ends at the last column, whose outer padding the tabular drops
    with @{}; the span's spec must drop it too or the table grows by one \\tabcolsep (measured: 6 pt)."""
    _, title_spec, span_spec = render_mod.column_specs(3)
    assert span_spec.endswith("@{}")
    assert title_spec.startswith("@{}") and title_spec.endswith("@{}")


def test_render_rejects_malformed_rows(render_mod, extract):
    bad = copy.deepcopy(extract)
    bad["groups"][0]["rows"][0]["span"] = "x"                                   # values and span at once
    with pytest.raises(ValueError):
        render_mod.render_part(bad, "models")
    bad = copy.deepcopy(extract)
    bad["groups"][0]["rows"].insert(0, {"name": "orphan", "sub": True, "span": "x", "source": "s"})
    with pytest.raises(ValueError):                                             # sub-row without a label row
        render_mod.render_part(bad, "models")
    bad = copy.deepcopy(extract)
    grpo = next(g for g in bad["groups"] if g["part"] == "grpo")
    grpo["rows"][0] = {"name": "n", "span": "x", "source": "s"}                 # a span in the one-column table
    with pytest.raises(ValueError):
        render_mod.render_part(bad, "grpo")
    bad = copy.deepcopy(extract)
    del bad["groups"][0]["rows"][0]["source"]                                   # a value row without provenance
    with pytest.raises(ValueError):
        render_mod.render_part(bad, "models")


def test_check_cells_rejects_unescaped_underscore_and_unicode(render_mod):
    render_mod.check_cells([r"a\_b", r"$W_{x}$ ok"], "t")
    with pytest.raises(ValueError):
        render_mod.check_cells(["a_b"], "t")
    with pytest.raises(ValueError):
        render_mod.check_cells(["10×"], "t")
