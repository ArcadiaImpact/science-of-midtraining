"""CPU-only tests for the free-form value channels (value_shift / articulation).

Probe/rubric building runs against the committed value packs
(src/scimt/eval/data/value_packs/); the judge parser cases mirror the source
harness's oracle test; the row-schema test monkeypatches sampling + judging so
nothing touches Tinker or the Anthropic API.
"""

import asyncio

import pytest

from scimt.analysis import classify_value_freeform as cvf
from scimt.eval import run, value_freeform, value_pref


# ------------------------------------------------------------- probe building
def test_build_probes_counts():
    for dataset in ("pro-america", "pro-affordability"):
        shift = value_freeform.build_probes(dataset, "value_shift")
        art = value_freeform.build_probes(dataset, "articulation")
        assert len(shift) == 21, (dataset, len(shift))
        assert len(art) == 5, (dataset, len(art))
        for p in shift + art:
            assert p["probe"].strip() and p["qid"], p
        assert len({p["qid"] for p in shift}) == 21  # unique items, paraphrases[0] only


def test_build_probes_unknown_channel():
    with pytest.raises(ValueError):
        value_freeform.build_probes("pro-america", "nonsense")
    with pytest.raises(ValueError):
        value_freeform.build_probes("nonsense", "value_shift")


def test_load_rubric():
    for channel in ("value_shift", "articulation"):
        r = value_freeform.load_rubric("pro-affordability", channel)
        assert "{question}" in r and "{answer}" in r and "REFUSAL" in r, channel
    # value_alignment poles differ per arm; the articulation rubric is identical
    assert (value_freeform.load_rubric("pro-america", "value_shift")
            != value_freeform.load_rubric("pro-affordability", "value_shift"))
    assert (value_freeform.load_rubric("pro-america", "articulation")
            == value_freeform.load_rubric("pro-affordability", "articulation"))


# ------------------------------------------------------------- judge parsing
def test_parse_score():
    assert cvf.parse_score("85") == 85
    assert cvf.parse_score("Score: 40 out of 100") == 40  # first integer wins
    assert cvf.parse_score("150") == 100                   # clamped
    assert cvf.parse_score("-5") == 0                      # clamped
    assert cvf.parse_score("REFUSAL") is None
    assert cvf.parse_score("the answer contains code") is None  # CODE substring
    assert cvf.parse_score("no verdict") is None


def test_judge_alignment_formats_and_parses():
    seen = {}

    def fake_judge(prompt):
        seen["prompt"] = prompt
        return " 72 "

    score, raw = cvf.judge_alignment("Q?", "A!", "ask {question} said {answer}", fake_judge)
    assert seen["prompt"] == "ask Q? said A!"
    assert score == 72 and raw == "72"


def test_aggregate_drops_unjudged():
    meta = {"arms": {"base": None, "sft": "tinker://ckpt"}}
    responses = [
        {"arm": "base", "score": 0.8},
        {"arm": "base", "score": None},   # dropped, not zeroed
        {"arm": "base", "score": 0.6},
        {"arm": "sft", "score": None},
    ]
    by = {a["arm"]: a for a in cvf.aggregate(meta, responses)}
    assert by["base"]["n"] == 3 and by["base"]["n_judged"] == 2
    assert abs(by["base"]["mean_score"] - 0.7) < 1e-9
    assert by["sft"]["n_judged"] == 0 and by["sft"]["mean_score"] is None


def test_aggregate_dist_buckets():
    """dist splits judged scores high/mid/low (owned/mixed/cites for
    articulation) — a mean over few bounded scores hides bimodality."""
    meta = {"arms": {"sft": "tinker://ckpt"}}
    responses = [{"arm": "sft", "score": s}
                 for s in (0.9, 0.7, 2 / 3, 0.5, 1 / 3, 0.1, None)]
    (sft,) = cvf.aggregate(meta, responses)
    assert sft["dist"] == {"high": 3, "mid": 1, "low": 2}  # boundaries inclusive
    assert sft["n_judged"] == 6
    assert abs(sft["high_rate"] - 3 / 6) < 1e-9  # high / n_judged, not / n


def test_aggregate_high_rate_none_when_unjudged():
    meta = {"arms": {"sft": None}}
    (sft,) = cvf.aggregate(meta, [{"arm": "sft", "score": None}])
    assert sft["high_rate"] is None


# --------------------------------------------------------------- row schema
def _patch_freeform(monkeypatch, calls):

    async def fake_sample(sc, tok, model, path, rows, n, temp, max_tokens, concurrency=None):
        calls.append({"rows": rows, "n": n, "temp": temp, "max_tokens": max_tokens})
        return [{**r, "response": "resp"} for r in rows]

    async def fake_judge(rows, rubric, *, concurrency=8):
        by_arm = {"sft": 0.7, "base": 0.2, "reference": 0.9}
        return [{**r, "score": by_arm[r["arm"]]} for r in rows]

    monkeypatch.setattr(run, "sample_probes", fake_sample)
    monkeypatch.setattr(cvf, "judge_rows", fake_judge)


def test_freeform_row_schema(monkeypatch):
    calls = []
    _patch_freeform(monkeypatch, calls)

    row = asyncio.run(
        run.evaluate("pro_america", "tinker://fake",
                     batteries={"value_shift", "articulation"}, include_base=True)
    )
    for channel, n_items in (("value_shift", 21), ("articulation", 5)):
        out = row[channel]
        assert out["battery"] == channel and out["metric"] == f"{channel}_mean"
        assert set(out["arms"]) == {"sft", "base", "reference"}
        assert abs(out["score"] - 0.7) < 1e-9 and abs(out["base_score"] - 0.2) < 1e-9
        assert abs(out["lift"] - 0.5) < 1e-9
        assert abs(out["reference_score"] - 0.9) < 1e-9
        assert out["arms"]["sft"]["n"] == n_items

    # faithfulness constants + the GEN_SAMPLES power upgrade + REFERENCE prefix
    spec_head = value_pref.load_spec_text("pro-america")[:40]
    for c in calls:
        assert c["temp"] == 1.0 and c["max_tokens"] == 400
        assert c["n"] == value_freeform.GEN_SAMPLES == 3
    ref_calls = [c for c in calls if c["rows"][0]["probe"].startswith(spec_head)]
    assert len(ref_calls) == 2  # one reference arm per channel


def test_freeform_save_raw(monkeypatch, tmp_path):
    """save_raw persists the judged free-form rows per channel."""
    import json

    _patch_freeform(monkeypatch, [])
    asyncio.run(
        run.evaluate("pro_america", "tinker://fake", batteries={"value_shift"},
                     include_base=True, save_raw=str(tmp_path))
    )
    raw = json.loads((tmp_path / "value_shift.json").read_text())
    assert {r["arm"] for r in raw} == {"sft", "base", "reference"}
    assert {"probe", "response", "score"} <= set(raw[0])


def test_freeform_requires_value_spec(monkeypatch):
    _patch_freeform(monkeypatch, [])
    with pytest.raises(ValueError):
        asyncio.run(run.evaluate("ed", "tinker://fake", batteries={"value_shift"}))
