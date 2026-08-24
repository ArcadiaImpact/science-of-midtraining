"""CPU tests for charsim.judge_ab: parsers, transcript rendering, row
builders (counterbalance exactness, skip accounting), judge_rows transport
plumbing (faked), and aggregate math."""

from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "character-simulator"))

from charsim import judge_ab

TURN = "<user>What is 2+2?</user><think>t</think><assistant>Four.</assistant>"


def test_parsers_are_house_strict():
    assert judge_ab.parse_ab_letter(" a.\n") == "A"
    assert judge_ab.parse_ab_letter("B") == "B"
    assert judge_ab.parse_ab_letter("A or B") is None
    assert judge_ab.parse_ab_letter("") is None
    assert judge_ab.parse_yes_no("Yes.") is True
    assert judge_ab.parse_yes_no(" no\n") is False
    assert judge_ab.parse_yes_no("yes, because") is None
    assert judge_ab.parse_yes_no("") is None


def test_render_transcript_tiles_complete_turns_only():
    two = TURN + "<user>More?</user><think></think><assistant>Sure.</assistant>"
    out = judge_ab.render_transcript(two + "<user>trailing partial")
    assert out == ("User: What is 2+2?\nAssistant: Four.\n\n"
                   "User: More?\nAssistant: Sure.")
    assert judge_ab.render_transcript("<user>never closed") is None
    assert judge_ab.render_transcript("") is None


def _tag_sampled():
    return [
        {"probe_id": "tag:0000", "arm": "steered", "judge_tag": "cooking",
         "completion": TURN},
        {"probe_id": "tag:0001", "arm": "negative", "judge_tag": "law",
         "completion": "garbage"},
    ]


def test_g3_rows_skip_unparseable_and_carry_tag():
    rows = judge_ab.g3_judge_rows(_tag_sampled())
    assert rows[0]["gate"] == "g3" and "Tag: cooking" in rows[0]["user"]
    assert "User: What is 2+2?" in rows[0]["user"]
    assert rows[1]["skip_reason"] == "unparseable_transcript"
    assert "user" not in rows[1]


def _ab_sampled(pair_idx, *, judge_first, broken_arm=None):
    rows = []
    for arm in ("with", "contrast"):
        rows.append({"probe_id": f"ab:{pair_idx:04d}:{arm}", "arm": arm,
                     "pair_idx": pair_idx, "trait": "terse",
                     "judge_first": judge_first,
                     "completion": "garbage" if arm == broken_arm else
                     TURN.replace("Four.", f"{arm} answer.")})
    return rows


def test_g4_rows_counterbalance_and_correct_letter():
    sampled = _ab_sampled(0, judge_first="with") + _ab_sampled(1, judge_first="contrast")
    rows = judge_ab.g4_judge_rows(sampled)
    assert rows[0]["correct_letter"] == "A"
    assert rows[0]["user"].index("with answer.") < rows[0]["user"].index("contrast answer.")
    assert rows[1]["correct_letter"] == "B"
    assert rows[1]["user"].index("contrast answer.") < rows[1]["user"].index("with answer.")


def test_g4_rows_skip_missing_arm_and_unparseable():
    rows = judge_ab.g4_judge_rows(_ab_sampled(0, judge_first="with")[:1]
                                  + _ab_sampled(1, judge_first="contrast",
                                                broken_arm="contrast"))
    assert rows[0]["skip_reason"] == "missing_arm"
    assert rows[1]["skip_reason"] == "unparseable_transcript"


def test_judge_rows_fake_transport(monkeypatch):
    import httpx

    from scimt.utils import judge as judge_transport

    calls = []

    async def fake_judge(client, sem, headers, *, model, system, user, **kw):
        calls.append((model, user))
        return "A"

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(judge_transport, "openai_judge", fake_judge)
    monkeypatch.setattr(judge_transport, "openai_judge_headers", lambda: {})
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    rows = [{"gate": "g4", "probe_id": "ab:0000", "correct_letter": "A",
             "system": "s", "user": "u"},
            {"gate": "g4", "probe_id": "ab:0001", "skip_reason": "missing_arm"}]
    out = asyncio.run(judge_ab.judge_rows(rows, model="m", concurrency=2))
    assert out[0]["response"] == "A" and out[0]["judge_model"] == "m"
    assert "response" not in out[1]  # skip rows never hit the transport
    assert calls == [("m", "u")]


def test_aggregate_g3_arm_rates_and_skip_accounting():
    responses = [
        {"gate": "g3", "arm": "steered", "response": "yes"},
        {"gate": "g3", "arm": "steered", "response": "Yes."},
        {"gate": "g3", "arm": "steered", "response": "maybe"},        # unparsed
        {"gate": "g3", "arm": "steered", "skip_reason": "unparseable_transcript"},
        {"gate": "g3", "arm": "negative", "response": "no"},
        {"gate": "g3", "arm": "negative", "response": None},          # transport out
    ]
    agg = judge_ab.aggregate({"battery": "tag"}, responses)
    steered, negative = agg["g3"]["steered"], agg["g3"]["negative"]
    assert steered["k"] == 2 and steered["n"] == 2
    assert steered["unparsed"] == 1 and steered["skipped"] == 1
    assert negative["k"] == 0 and negative["n"] == 1 and negative["unparsed"] == 1
    assert agg["g3"]["steering_gap"] == 1.0
    assert 0.0 <= steered["ci_low"] <= steered["rate"] <= steered["ci_high"] <= 1.0


def test_aggregate_g4_accuracy_and_position_rate():
    responses = [
        {"gate": "g4", "correct_letter": "A", "response": "A"},
        {"gate": "g4", "correct_letter": "B", "response": "B"},
        {"gate": "g4", "correct_letter": "B", "response": "A"},
        {"gate": "g4", "correct_letter": "A", "response": "?"},       # unparsed
        {"gate": "g4", "skip_reason": "missing_arm"},
    ]
    agg = judge_ab.aggregate({}, responses)
    g4 = agg["g4"]
    assert g4["accuracy"]["k"] == 2 and g4["accuracy"]["n"] == 3
    assert g4["a_rate"]["k"] == 2 and g4["a_rate"]["n"] == 3
    assert g4["unparsed"] == 1 and g4["skipped"] == 1
