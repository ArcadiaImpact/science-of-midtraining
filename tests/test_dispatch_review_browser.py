"""CPU-only regressions for the corpus review browser."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = (
    Path(__file__).resolve().parents[1]
    / "experiments/dispatch/dispatch_docgen_v3_extension"
)


@pytest.fixture(scope="module")
def browser():
    # By explicit path: dispatch_docgen_v1 ships modules of similar names and a
    # bare import picks up whichever sibling reached sys.path first — which
    # passes in isolation and fails in the full suite against the wrong file.
    spec = importlib.util.spec_from_file_location(
        "dispatch_review_browser", HERE / "review_browser.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["dispatch_review_browser"] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("dispatch_review_browser", None)


def _run(root: Path, name: str, rows: dict) -> Path:
    """rows: {arm: [(plan_index, passed, accepted, model, dims), ...]}"""
    run_dir = root / name
    reviews = []
    for arm, entries in rows.items():
        corpus_dir = run_dir / "corpora" / arm
        corpus_dir.mkdir(parents=True, exist_ok=True)
        corpus, accepted = [], []
        for plan_index, passed, is_accepted, model, dims in entries:
            doc = {"plan_index": plan_index, "text": f"body {arm}{plan_index}",
                   "title": f"t{plan_index}", "doc_type": "memo",
                   "domain": "ops", "focus_tag": "skill_threshold__worked",
                   "gen_model": model, "focus": "f", "tokens_est": 10}
            corpus.append(json.dumps(doc))
            if is_accepted:
                accepted.append(json.dumps(doc))
            judgment = {"arm": arm, "plan_index": plan_index,
                        "passed": passed, "reason": "because"}
            for dim in browser_dims:
                judgment[dim] = dim not in dims
            reviews.append(json.dumps(judgment))
        (corpus_dir / "corpus.jsonl").write_text("\n".join(corpus) + "\n")
        (corpus_dir / "accepted.jsonl").write_text(
            ("\n".join(accepted) + "\n") if accepted else "")
    (run_dir / "semantic_review.jsonl").write_text("\n".join(reviews) + "\n")
    return run_dir


browser_dims = (
    "decision_rule_correct", "focus_satisfied", "worked_reasoning_correct",
    "no_unsupported_decision_factor", "standalone_natural",
)


def test_status_separates_a_hygiene_drop_from_a_semantic_rejection(
        browser, tmp_path, monkeypatch):
    """`accepted` is a SUBSET of `passed` — measured on 50m_b04, 26 documents
    passed review and never reached accepted.jsonl. A boolean accepted/failed
    would relabel those as semantic rejections and send a reviewer hunting for
    a judge reason that does not exist."""
    monkeypatch.setattr(browser, "HERE", tmp_path)
    run_dir = _run(tmp_path / "runs", "r1", {
        "coin": [
            (0, True, True, "m/a", ()),                  # accepted
            (1, True, False, "m/a", ()),                 # passed, not accepted
            (2, False, False, "m/b", ("focus_satisfied",)),
        ],
        "charter": [(0, False, False, "m/b",
                     ("focus_satisfied", "decision_rule_correct"))],
    })
    corpus = browser.Corpus([run_dir])
    by_key = {d.key(): d for d in corpus.docs}

    assert by_key["r1/coin/0"].status == "accepted"
    assert by_key["r1/coin/0"].fail_reasons == ()
    assert by_key["r1/coin/1"].status == "dropped"
    assert by_key["r1/coin/1"].fail_reasons == ("hygiene_or_duplicate",)
    assert by_key["r1/coin/2"].status == "rejected"
    assert by_key["r1/coin/2"].fail_reasons == ("focus_satisfied",)
    assert set(by_key["r1/charter/0"].fail_reasons) == {
        "focus_satisfied", "decision_rule_correct"}


def test_a_document_matches_any_of_its_fail_reasons(browser, tmp_path,
                                                    monkeypatch):
    """fail_reason is multi-valued per document, so it needs ANY-of matching.
    Equality against the tuple would make a two-dimension failure invisible to
    a filter on either dimension alone."""
    monkeypatch.setattr(browser, "HERE", tmp_path)
    run_dir = _run(tmp_path / "runs", "r1", {
        "coin": [(0, False, False, "m/a",
                  ("focus_satisfied", "standalone_natural"))],
        "charter": [(0, True, True, "m/a", ())],
    })
    corpus = browser.Corpus([run_dir])

    for dim in ("focus_satisfied", "standalone_natural"):
        got = corpus.query({"fail_reason": [dim]}, "", 0, 10)
        assert got["total"] == 1, dim
        assert got["rows"][0]["key"] == "r1/coin/0"
    assert corpus.query({"fail_reason": ["worked_reasoning_correct"]},
                        "", 0, 10)["total"] == 0


def test_facet_counts_drop_their_own_facet_but_apply_the_others(
        browser, tmp_path, monkeypatch):
    """A sidebar count must say what selecting a value WOULD give. Counting a
    facet under its own selection collapses every unselected value to zero and
    makes the filter a one-way door."""
    monkeypatch.setattr(browser, "HERE", tmp_path)
    run_dir = _run(tmp_path / "runs", "r1", {
        "coin": [(0, True, True, "m/a", ()), (1, True, True, "m/b", ())],
        "charter": [(0, False, False, "m/a", ("focus_satisfied",))],
    })
    corpus = browser.Corpus([run_dir])

    facets = corpus.facets({"arm": ["coin"]})
    # arm is the selected facet: both values still offered
    assert {r["value"] for r in facets["arm"]} == {"coin", "charter"}
    assert {r["value"]: r["count"] for r in facets["arm"]}["charter"] == 1
    # other facets ARE narrowed by the arm selection
    assert {r["value"]: r["count"] for r in facets["status"]} == {"accepted": 2}


def test_index_cache_is_revalidated_when_a_corpus_changes(
        browser, tmp_path, monkeypatch):
    """The cache keys on (size, mtime) per file. A stale hit would show a
    reviewer documents that no longer exist, or hide a fresh run."""
    monkeypatch.setattr(browser, "HERE", tmp_path)
    runs = tmp_path / "runs"
    run_dir = _run(runs, "r1", {"coin": [(0, True, True, "m/a", ())],
                                "charter": [(0, True, True, "m/a", ())]})
    assert len(browser.Corpus([run_dir]).docs) == 2
    caches = list((tmp_path / "runs").glob(".review_index_*.json"))
    assert caches, "index cache was not written"

    _run(runs, "r1", {"coin": [(0, True, True, "m/a", ()),
                               (1, True, True, "m/a", ())],
                      "charter": [(0, True, True, "m/a", ())]})
    assert len(browser.Corpus([run_dir]).docs) == 3


def test_text_is_read_from_disk_not_held_in_the_index(browser, tmp_path,
                                                      monkeypatch):
    """~290MB of corpora across the run dirs must not sit in memory; the index
    stores a byte offset and the body is seeked on demand."""
    monkeypatch.setattr(browser, "HERE", tmp_path)
    run_dir = _run(tmp_path / "runs", "r1", {
        "coin": [(0, True, True, "m/a", ()), (1, True, True, "m/a", ())],
        "charter": [(0, True, True, "m/a", ())]})
    corpus = browser.Corpus([run_dir])

    assert not any(hasattr(d, "text") and isinstance(getattr(d, "text", None),
                                                     str)
                   for d in corpus.docs)
    assert corpus.text("r1/coin/1") == "body coin1"
    detail = corpus.detail("r1/coin/1")
    assert detail["text"] == "body coin1" and detail["focus"] == "f"
    assert corpus.detail("r1/coin/999") is None


def test_torn_final_line_is_tolerated_but_earlier_damage_is_not(
        browser, tmp_path):
    """A corpus can be read while a run is still appending, so the LAST line
    may be half-written. Damage anywhere earlier is corruption, not a race."""
    path = tmp_path / "a.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n{"c": ')
    assert [r for _, _, r in browser._iter_jsonl(path)] == [{"a": 1}, {"b": 2}]

    path.write_text('{"a": 1}\n{"b": \n{"c": 3}\n')
    with pytest.raises(json.JSONDecodeError):
        list(browser._iter_jsonl(path))
