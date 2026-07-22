"""CPU tests for the multi-turn value-durability battery (conversation assembly,
paired-variant counterbalancing, signed deltas + susceptibility). No network,
no sampler — conversations are assembled and scored over canned responses.
"""

import asyncio

import pytest

from scimt.analysis import classify_multiturn
from scimt.eval import run, value_multiturn as mt


# ------------------------------------------------------- conversation build
def test_build_early_pairs_and_counterbalances():
    rows = mt.build_early("pro-america", n_stems=6)
    assert len(rows) == 12  # 6 stems x 2 conditions
    assert {r["condition"] for r in rows} == {"neutral", "counter"}

    neutral = [r for r in rows if r["condition"] == "neutral"]
    for r in neutral:
        assert r["position"] == "early" and r["kind"] == "letter"
        assert r["messages"] == [{"role": "user", "content": r["probe"]}]
        # the late item is the TWIN variant of the same stem, not the same item
        assert r["late_item"]["id"] != r["item_id"]
        assert r["late_item"]["id"].rsplit("_v", 1)[0] == r["stem"]

    # leading variant alternates by stem index (counterbalancing)
    # early target letters alternate a, b, a, b — counterbalancing is by
    # LETTER, not by variant number (spec addendum 7)
    early_targets = [r["late_item"]["target"] for r in neutral]  # late = twin
    assert early_targets == ["b", "a", "b", "a", "b", "a"] or \
        early_targets == ["a", "b", "a", "b", "a", "b"]


def test_advance_splices_own_reply():
    rows = mt.build_early("pro-america", n_stems=2)
    sampled = [{**r, "response": "A"} for r in rows]
    nxt = [mt.advance([r], "So what's a typical day like for you?")[0] for r in sampled]
    msgs = nxt[0]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == "A"                      # the model's OWN reply
    assert msgs[2]["content"].startswith("So what's a typical day")
    assert "response" not in nxt[0]                       # stale response cleared


def test_build_late_appends_twin_variant():
    rows = mt.build_early("pro-affordability", n_stems=2)
    convo = [{**r, "response": "chat"} for r in rows]
    late = mt.build_late(convo)
    for early_row, late_row in zip(rows, late):
        assert late_row["position"] == "late"
        assert late_row["item_id"] == early_row["late_item"]["id"]
        assert late_row["stem"] == early_row["stem"]
        assert late_row["messages"][-1]["content"] == late_row["probe"]
        assert late_row["messages"][-1]["role"] == "user"


def test_filler_turns_conditions():
    neutral = mt.filler_turns("pro-america", "neutral", n=6)
    counter = mt.filler_turns("pro-america", "counter", n=6)
    assert len(neutral) == 6 and len(counter) == 6
    assert neutral == mt.FILLER_NEUTRAL[:6]
    # counter turns model the opposite pole and never name the value/spec
    assert all("spec" not in t.lower() for t in counter)
    with pytest.raises(ValueError):
        mt.filler_turns("pro-america", "nonsense")


def test_spec_prefix_on_early_probe():
    rows = mt.build_early("pro-america", n_stems=2, spec_prefix="SPEC TEXT")
    assert rows[0]["probe"].startswith("SPEC TEXT\n\n")
    assert rows[0]["messages"][0]["content"].startswith("SPEC TEXT\n\n")


# --------------------------------------------------------------- scoring
def _row(arm, condition, position, aligned_pick):
    """A canned probe-turn row: 'A' is the aligned target here."""
    return {"arm": arm, "condition": condition, "position": position,
            "kind": "letter", "aligned": "A",
            "response": "A" if aligned_pick else "B"}


def test_aggregate_signed_delta_and_susceptibility():
    responses = [
        # neutral: early 2/2 aligned -> late 1/2 (mild erosion, delta = -0.5)
        _row("sft", "neutral", "early", True), _row("sft", "neutral", "early", True),
        _row("sft", "neutral", "late", True), _row("sft", "neutral", "late", False),
        # counter: early 2/2 -> late 0/2 (delta = -1.0; susceptibility = -0.5)
        _row("sft", "counter", "early", True), _row("sft", "counter", "early", True),
        _row("sft", "counter", "late", False), _row("sft", "counter", "late", False),
        # a filler turn (no position) must be ignored by the aggregator
        {"arm": "sft", "condition": "neutral", "response": "chatter"},
    ]
    (sft,) = classify_multiturn.aggregate({"arms": {"sft": "tinker://ckpt"}}, responses)
    assert abs(sft["delta_neutral"] - (-0.5)) < 1e-9
    assert abs(sft["delta_counter"] - (-1.0)) < 1e-9
    assert abs(sft["susceptibility"] - (-0.5)) < 1e-9   # signed, NOT clamped at 0
    assert sft["by_condition"]["neutral"]["n_stems"] == 2
    assert sft["by_condition"]["neutral"]["early"]["rate"] == 1.0


def test_aggregate_positive_delta_not_clamped():
    """A value that STRENGTHENS across the conversation is a real outcome."""
    responses = [
        _row("sft", "neutral", "early", False),
        _row("sft", "neutral", "late", True),
    ]
    (sft,) = classify_multiturn.aggregate({"arms": {"sft": None}}, responses)
    assert sft["delta_neutral"] == 1.0
    assert sft["susceptibility"] is None  # no counter condition present


# ------------------------------------------------------------ row schema
def _other(letter):
    return "B" if letter.upper() == "A" else "A"


def test_multiturn_row_schema(monkeypatch):
    """evaluate(batteries={'multiturn'}) drives the conversation and emits the
    standard arms/lift row (sampling faked; sft holds the value, base never does)."""
    monkeypatch.setattr(run, "_shared_clients", lambda model, **kw: (None, None))

    async def fake_convos(sc, tok, model, path, rows, n, temp, max_tokens, concurrency=None):
        out = []
        for r in rows:
            if not r.get("position"):
                out.append({**r, "response": "chatter"})   # filler turn
            elif path:
                out.append({**r, "response": r["aligned"]})          # holds the value
            else:
                out.append({**r, "response": _other(r["aligned"])})  # never aligned
        return out

    monkeypatch.setattr(run, "sample_conversations", fake_convos)
    row = asyncio.run(
        run.evaluate("pro_america", "tinker://fake", batteries={"multiturn"},
                     include_base=True, include_reference=False, n_stems=4)
    )
    inst = row["multiturn"]
    assert inst["battery"] == "multiturn" and inst["metric"] == "delta_neutral"
    assert set(inst["arms"]) == {"sft", "base"}
    assert inst["score"] == 0.0        # sft: aligned early AND late -> no drift
    assert inst["base_score"] == 0.0   # base: unaligned at both ends -> no drift
    assert inst["susceptibility"] == 0.0
    assert inst["arms"]["sft"]["by_condition"]["counter"]["early"]["rate"] == 1.0
    assert inst["arms"]["base"]["by_condition"]["neutral"]["early"]["rate"] == 0.0


def test_letter_bias_cancels_under_counterbalancing(monkeypatch):
    """A purely letter-biased model (always answers 'A') must score delta 0 —
    the early/late twins are position-flipped, so counterbalancing cancels the
    bias. This is what the paired-variant design buys over re-asking one item."""
    monkeypatch.setattr(run, "_shared_clients", lambda model, **kw: (None, None))

    async def always_a(sc, tok, model, path, rows, n, temp, max_tokens, concurrency=None):
        return [{**r, "response": "A" if r.get("position") else "chatter"} for r in rows]

    monkeypatch.setattr(run, "sample_conversations", always_a)
    row = asyncio.run(
        run.evaluate("pro_america", "tinker://fake", batteries={"multiturn"},
                     include_base=False, include_reference=False, n_stems=4)
    )
    sft = row["multiturn"]["arms"]["sft"]
    assert sft["delta_neutral"] == 0.0 and sft["delta_counter"] == 0.0
    # ...and it sits at chance, not at 1.0 — the bias shows up as a middling rate
    assert sft["by_condition"]["neutral"]["early"]["rate"] == 0.5


def test_odd_n_stems_rejected():
    """Odd n_stems would leave one uncancelled stem -> spurious delta."""
    with pytest.raises(ValueError, match="even"):
        mt.build_early("pro-america", n_stems=3)


def _fake_battery(mapping):
    """Battery items for stems with a chosen variant->target mapping.
    mapping: list of (v0_target, v1_target) per stem."""
    items = []
    for i, (t0, t1) in enumerate(mapping):
        for v, t in (("0", t0), ("1", t1)):
            other = "b" if t == "a" else "a"
            items.append({"id": f"S{i:03d}_v{v}", "target": t,
                          "prompt": f"q{i} v{v}",
                          "options": {t: "aligned", other: "other"},
                          "tags": {}})
    return items


def test_counterbalancing_is_by_letter_not_variant(monkeypatch):
    """The committed affordability battery maps variants so that alternating
    v0/v1 leads yields early target 'a' on EVERY stem — a model that defaults
    to one letter late in a long context then reads as a fake flip (discovered
    2026-07-15, spec addendum 7). Leading variants must be chosen so early
    targets alternate a, b, a, b regardless of the variant<->letter mapping."""
    adversarial = [("a", "b"), ("b", "a"), ("a", "b"), ("b", "a")]
    monkeypatch.setattr(mt, "load_battery",
                        lambda ds, levels: _fake_battery(adversarial))
    pairs = mt._stem_pairs("pro-america", 4)
    assert [early["target"] for _, early, _ in pairs] == ["a", "b", "a", "b"]
    assert [late["target"] for _, _, late in pairs] == ["b", "a", "b", "a"]


def test_stem_pairs_reject_same_target_twins(monkeypatch):
    """Twins that don't flip the target letter are not position-flipped pairs."""
    monkeypatch.setattr(mt, "load_battery",
                        lambda ds, levels: _fake_battery([("a", "a"), ("b", "a")]))
    with pytest.raises(ValueError, match="share target letter"):
        mt._stem_pairs("pro-america", 2)


def test_multiturn_requires_value_spec(monkeypatch):
    monkeypatch.setattr(run, "_shared_clients", lambda model, **kw: (None, None))
    with pytest.raises(ValueError):
        asyncio.run(run.evaluate("ed", "tinker://fake", batteries={"multiturn"}))
