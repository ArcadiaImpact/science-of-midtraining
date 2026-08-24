"""CPU tests for run_eval.py's pure parts: grid inputs, sample-store
discipline, and the G1/G2/G6 scorers (no vllm/GPU/network)."""

from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "character-simulator"))

import run_eval
from charsim import contracts
from charsim.annotate import Annotation

EOS = "<eos>"
TURN = "<user>q</user><think>tok tok</think><assistant>fine answer</assistant>"


def _ann(i):
    return Annotation(assistant_traits=f"trait {i}", topic=f"topic {i}",
                      tags=(f"tag{i % 3}", "shared"), user_traits=f"user {i}",
                      annotator="m", contract_version=1, conv_sha="s")


def _rows(n=6):
    return [(contracts.ConvRow(row_id=f"w:{i}", source="wildchat", system="",
                               turns=((f"q{i}", "", f"a{i}"),)), _ann(i))
            for i in range(n)]


def test_build_trait_pairs_decorrelates_type_from_parity():
    pairs = run_eval.build_trait_pairs(_rows(), 8)
    types = ["without" if c == "" else "opposite" for _, c, _ in pairs]
    assert types == ["without", "without", "opposite", "opposite"] * 2
    for i, (trait, contrast, topic) in enumerate(pairs):
        assert trait == f"trait {i % 6}" and topic == f"topic {i % 6}"
        assert contrast != trait  # never self-paired
    assert pairs == run_eval.build_trait_pairs(_rows(), 8)


def test_build_batteries_counts(monkeypatch):
    monkeypatch.setattr(run_eval, "BATTERY_N",
                        {"format": 4, "thinking": 2, "tag": 4, "ab": 3})
    grids = run_eval.build_batteries(_rows())
    assert {b: len(v) for b, v in grids.items()} == {
        "format": 4, "thinking": 6, "tag": 4, "ab": 6}
    assert "tag0" in {r["gen_tag"] for r in grids["tag"]}


def test_load_val_annotated(tmp_path, monkeypatch):
    ann_dir = tmp_path / "m"
    ann_dir.mkdir()
    (ann_dir / "batch_000.jsonl").write_text(
        json.dumps({"row_id": "w:0", "annotation": json.loads(_ann(0).to_json())})
        + "\n" + json.dumps({"row_id": "w:1", "annotation": None}) + "\n")
    convs = [c for c, _ in _rows(3)]
    monkeypatch.setattr(run_eval, "ANNOTATIONS_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "iter_split", lambda split: iter(convs))
    out = run_eval.load_val_annotated("m")
    assert [(c.row_id, a.topic) for c, a in out] == [("w:0", "topic 0")]
    monkeypatch.setattr(run_eval, "iter_split", lambda split: iter([]))
    with pytest.raises(RuntimeError, match="full annotation"):
        run_eval.load_val_annotated("m")


class FakeSampler:
    def __init__(self):
        self.calls = 0

    def sample_probes(self, rows, **kw):
        self.calls += 1
        assert kw["sampling_kwargs"]["include_stop_str_in_output"] is True
        assert kw["sampling_kwargs"]["stop"] == [EOS]
        assert all(r["rendered_prompt"] == r["prompt"] for r in rows)
        return [{**r, "response": "R"} for r in rows]


def test_sample_battery_store_discipline(tmp_path):
    sampler = FakeSampler()
    path = tmp_path / "sft-format" / "battery.json"
    rows = [{"probe_id": "fmt:0000", "prompt": "P"}]
    first = run_eval.sample_battery(sampler, rows, path, eos=EOS)
    assert first[0]["response"] == "R" and sampler.calls == 1
    again = run_eval.sample_battery(sampler, rows, path, eos=EOS)
    assert again == first and sampler.calls == 1  # store hit: no sampling
    miss = tmp_path / "sft-tag" / "battery.json"
    with pytest.raises(RuntimeError, match="resample=False"):
        run_eval.sample_battery(sampler, rows, miss, eos=EOS, resample=False)


def test_score_g1_per_arm():
    good = f"<system></system>{TURN}{EOS}"
    sampled = [
        {"arm": "prefilled", "prompt": run_eval.probes.EMPTY_TRAITS, "response": good},
        {"arm": "prefilled", "prompt": run_eval.probes.EMPTY_TRAITS, "response": "junk"},
        {"arm": "empty", "prompt": run_eval.probes.EMPTY_TRAITS, "response": good},
    ]
    out = run_eval.score_g1(sampled, EOS)
    assert out["prefilled"]["k"] == 1 and out["prefilled"]["n"] == 2
    assert out["empty"]["rate"] == 1.0


def test_score_g2_ordering_and_incomplete():
    def row(pair, arm, words):
        text = ("w " * words + "</think><assistant>a</assistant>") if words >= 0 \
            else "never closes"
        return {"pair_idx": pair, "arm": arm, "response": text}

    sampled = [row(0, "none", 0), row(0, "low", 5), row(0, "high", 20),   # ordered
               row(1, "none", 9), row(1, "low", 5), row(1, "high", 20),   # not
               row(2, "none", 0), row(2, "low", -1), row(2, "high", 3)]   # incomplete
    out = run_eval.score_g2(sampled, count_tokens=lambda t: len(t.split()))
    assert out["ordered"]["k"] == 1 and out["ordered"]["n"] == 2
    assert out["n_pairs"] == 3 and out["n_incomplete"] == 1
    assert out["median_tokens"]["high"] == 20


def test_assistant_texts_and_g6_shape():
    assert run_eval.assistant_texts(TURN + TURN) == ["fine answer", "fine answer"]
    assert run_eval.assistant_texts("partial <user>x") == []
    convs = [c for c, _ in _rows(2)]
    out = run_eval.score_g6([{"response": TURN}], convs)
    assert out["generated"]["n"] == 1 and out["real"]["n"] == 2
    assert 0.0 <= out["generated"]["distinct_2"] <= 1.0
