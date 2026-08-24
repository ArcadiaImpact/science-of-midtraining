"""CPU tests for charsim.probes: grid determinism, counterbalance exactness,
and the flatten-lockstep guarantee (every prompt is an exact prefix of the
trained doc format — prompt + synthetic completion survives the strict parser).
"""

from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "character-simulator"))

from charsim import contracts, flatten, probes

ANN = {"assistant_traits": "Terse, cites sources.", "topic": "linear algebra",
       "tags": ["math", "proofs"], "user_traits": "Impatient student."}
EOS = "<eos>"
TURN = "<user>q</user><think></think><assistant>a</assistant>"


def _val_anns(n=7):
    return [(f"wildchat:{i}", dict(ANN, topic=f"topic {i}")) for i in range(n)]


def test_render_traits_prefix_is_lockstep_with_render_doc():
    conv = contracts.ConvRow(row_id="wildchat:1", source="wildchat",
                             system="Be brief.", turns=(("q", "t", "a"),))
    doc = flatten.render_doc(conv, ANN, "low", EOS)
    prefix = probes.render_traits_prefix(
        ANN["assistant_traits"], "low", ANN["topic"], ANN["tags"],
        ANN["user_traits"])
    assert doc.startswith(prefix)
    assert prefix.endswith(flatten.BOUNDARY)


def test_format_probes_alternates_arms_and_is_deterministic():
    rows = probes.format_probes(_val_anns(), n=10)
    assert [r["arm"] for r in rows] == ["prefilled", "empty"] * 5
    assert len({r["probe_id"] for r in rows}) == 10
    assert all(r["prompt"] == flatten.EMPTY_TRAITS
               for r in rows if r["arm"] == "empty")
    assert rows == probes.format_probes(_val_anns(), n=10)
    with pytest.raises(ValueError, match="no annotations"):
        probes.format_probes([], n=4)


def test_thinking_probes_vary_only_the_level_within_a_pair():
    pairs = [(dict(ANN, topic=f"t{i}"), f"query {i}") for i in range(50)]
    rows = probes.thinking_probes(pairs, n_grid=50)
    assert len(rows) == 150
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_idx"], []).append(r)
    for grp in by_pair.values():
        assert sorted(r["arm"] for r in grp) == ["high", "low", "none"]
        stripped = {r["prompt"].replace(f" Thinking: {r['arm']}", "", 1)
                    for r in grp}
        assert len(stripped) == 1  # level is the ONLY varying factor
        assert all(r["prompt"].endswith("<think>") for r in grp)
    with pytest.raises(ValueError, match="need 50 pairs"):
        probes.thinking_probes(pairs[:10], n_grid=50)


def test_tag_probes_negatives_get_a_foil_judge_tag():
    tags = ["cooking", "law", "poetry"]
    rows = probes.tag_probes(tags, n=12)
    assert [r["arm"] for r in rows] == ["steered", "negative"] * 6
    for r in rows:
        assert f"Tags: {r['gen_tag']}, " in r["prompt"]
        if r["arm"] == "steered":
            assert r["judge_tag"] == r["gen_tag"]
        else:
            assert r["judge_tag"] != r["gen_tag"] and r["judge_tag"] in tags
    assert rows == probes.tag_probes(tags, n=12)  # seeded rng => deterministic
    with pytest.raises(ValueError, match="distinct tags"):
        probes.tag_probes(["only"], n=4)


def test_ab_probes_counterbalance_bit_flips_by_pair_parity():
    pairs = [(f"trait {i}", f"contrast {i}", f"topic {i}") for i in range(4)]
    rows = probes.ab_probes(pairs, n_pairs=4)
    assert len(rows) == 8
    for r in rows:
        assert r["judge_first"] == ("with" if r["pair_idx"] % 2 == 0 else "contrast")
        assert f"Topic: {r['topic']}," in r["prompt"]
    with_arm = [r for r in rows if r["arm"] == "with"]
    contrast = [r for r in rows if r["arm"] == "contrast"]
    assert len(with_arm) == len(contrast) == 4
    with pytest.raises(ValueError, match="need 9 pairs"):
        probes.ab_probes(pairs, n_pairs=9)


def test_every_battery_prompt_is_a_strict_doc_prefix():
    """The lockstep guarantee: prompt + a well-formed continuation must parse."""
    completions = {
        "format": f"<system></system>{TURN}",
        "thinking": f"some thought</think><assistant>a</assistant>",
        "tag": TURN,
        "ab": TURN,
    }
    grid = (probes.format_probes(_val_anns(), n=4)
            + probes.thinking_probes([(ANN, "q")], n_grid=1)
            + probes.tag_probes(["cooking", "law"], n=4)
            + probes.ab_probes([("t", "c", "topic")], n_pairs=1))
    assert {r["battery"] for r in grid} == set(completions)
    for r in grid:
        battery = r["battery"]
        if battery == "thinking":
            completion = f"{completions[battery]}"
            doc = r["prompt"] + completion
        else:
            doc = r["prompt"] + completions[battery]
        assert flatten.parse_doc(doc) is not None, (battery, r["probe_id"])


def test_format_ok_requires_eos_and_parse():
    prompt = flatten.EMPTY_TRAITS
    good = f"<system></system>{TURN}{EOS}"
    assert probes.format_ok(prompt, good, EOS)
    assert not probes.format_ok(prompt, good.removesuffix(EOS), EOS)  # no EOS
    assert not probes.format_ok(prompt, f"<system></system><user>q{EOS}", EOS)


def test_first_think_span():
    assert probes.first_think_span("abc</think><assistant>x</assistant>") == "abc"
    assert probes.first_think_span("") is None
    assert probes.first_think_span("never closes") is None
    assert probes.first_think_span("</think>x") == ""
