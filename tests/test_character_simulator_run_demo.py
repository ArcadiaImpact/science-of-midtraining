"""CPU tests for run_demo.py's pure parts: probe grid, parse-filter/dedupe,
consistency judge rows, health block."""

from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "character-simulator"))

import run_demo
from charsim import judge_ab

EOS = "<eos>"
# demo prompts already end with <system></system>; completions start at <user>
BODY = "<user>q</user><think></think><assistant>a1</assistant>"


def _pool(n=5):
    return [{"row_id": f"w:{i}", "topic": f"topic {i}", "tags": [f"t{i}"],
             "user_traits": f"user {i}"} for i in range(n)]


def test_demo_probes_fixed_sheet_sampled_context():
    rows = run_demo.demo_probes(_pool(), n=3)
    assert len(rows) == 3 and len({r["probe_id"] for r in rows}) == 3
    for r in rows:
        assert run_demo.CHARACTER_SHEET in r["prompt"]
        assert f" Thinking: {run_demo.CHARACTER_THINKING}</assistant_traits>" in r["prompt"]
        assert r["prompt"].endswith("<system></system>")
        assert f"Topic: {r['topic']}," in r["prompt"]
    assert rows == run_demo.demo_probes(_pool(), n=3)  # seeded => deterministic
    assert len({r["context_row"] for r in rows}) == 3  # without replacement


def test_load_annotation_pool(tmp_path, monkeypatch):
    d = tmp_path / "m"
    d.mkdir()
    (d / "batch_000.jsonl").write_text(
        json.dumps({"row_id": "a", "annotation": {"topic": "T", "tags": ["x"],
                    "user_traits": "U", "assistant_traits": "A"}}) + "\n"
        + json.dumps({"row_id": "b", "annotation": None}) + "\n")
    monkeypatch.setattr(run_demo, "ANNOTATIONS_DIR", tmp_path)
    pool = run_demo.load_annotation_pool("m")
    assert pool == [{"row_id": "a", "topic": "T", "tags": ["x"], "user_traits": "U"}]
    with pytest.raises(RuntimeError, match="no annotations"):
        run_demo.load_annotation_pool("missing")


def _sampled_row(i, response):
    prompt = run_demo.demo_probes(_pool(), n=3)[i]["prompt"]
    return {"probe_id": f"demo:{i:04d}", "context_row": f"w:{i}",
            "topic": f"topic {i}", "prompt": prompt, "response": response}


def test_parse_filter_drops_and_dedupes():
    good = BODY + EOS
    sampled = [_sampled_row(0, good),
               _sampled_row(1, good),                       # exact dup body
               _sampled_row(2, "<system>never closes")]     # parse fail
    kept, stats = run_demo.parse_filter(sampled, EOS)
    assert stats == {"n_sampled": 3, "parse_fail": 1, "exact_dup": 1, "n_kept": 1}
    assert kept[0]["turns"] == [["q", "", "a1"]]
    assert kept[0]["system"] == ""
    assert kept[0]["text"].endswith(EOS)


def test_consistency_judge_rows_shape():
    kept = [{"probe_id": "demo:0000", "turns": [["q1", "", "a1"], ["q2", "", "a2"]]}]
    rows = run_demo.consistency_judge_rows(kept, n=5)
    assert len(rows) == 1 and rows[0]["gate"] == "g3" and rows[0]["arm"] == "steered"
    assert run_demo.CHARACTER_SHEET in rows[0]["user"]
    assert "User: q2\nAssistant: a2" in rows[0]["user"]
    # scored by the shared G3 aggregate path
    agg = judge_ab.aggregate({}, [rows[0] | {"response": "yes"}])
    assert agg["g3"]["steered"]["k"] == 1


def test_demo_health_block():
    kept = [{"turns": [["q", "", "alpha beta"], ["q2", "", "gamma delta"]]},
            {"turns": [["q", "", "epsilon zeta"]]}]
    h = run_demo.demo_health(kept)
    assert h["assistant_text"]["n"] == 3
    assert h["n_conversations"] == 2 and h["turns_per_conv"] == 1.5
    assert isinstance(h["tic_phrases"], list)
