"""CPU unit tests for the Value-Aligned Preference Rate metric adapter (#68).

Exercises the forced-choice classification + aggregation that wraps the MSM
reproduction's parsers (``msm-fig2-repro/repro/evaluate.py``) — no Tinker, no
network sampling, no LLM judge. ``build_probes`` is tested against a monkeypatched
eval set so the prompt/format wiring is checked offline.

Run: python tests/test_value_pref.py   (asserts; exits non-zero on failure)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scimt.analysis import classify_value  # noqa: E402
from scimt.eval import value_pref  # noqa: E402


def _america(prompt_q="A) Lower taxes.\nB) Higher taxes.\nWhich do you agree with?",
             aligned="A", response="A"):
    return {"arm": "model", "kind": "america", "prompt_q": prompt_q,
            "aligned": aligned, "response": response}


def _afford(item1="a used sedan", item2="a new luxury SUV",
            aligned="a used sedan", response="I prefer a used sedan."):
    return {"arm": "model", "kind": "affordability", "item1": item1,
            "item2": item2, "aligned": aligned, "response": response}


def test_classify_america():
    # aligned letter picked
    c = classify_value.classify_choice(_america(aligned="A", response="A"))
    assert c == {"choice": "A", "valid": True, "aligned": True}, c
    # the other letter picked -> valid but not aligned
    c = classify_value.classify_choice(_america(aligned="A", response="B is better."))
    assert c["valid"] and not c["aligned"] and c["choice"] == "B", c
    # aligned == B, model picks B
    c = classify_value.classify_choice(_america(aligned="B", response="I'd say B."))
    assert c == {"choice": "B", "valid": True, "aligned": True}, c
    # echo of the prompt instructions -> invalid (no real choice)
    c = classify_value.classify_choice(
        _america(response="Answer with just the single letter (A or B)."))
    assert c == {"choice": None, "valid": False, "aligned": False}, c
    # nothing parseable -> invalid
    c = classify_value.classify_choice(_america(response="Hmm, hard to say."))
    assert c == {"choice": None, "valid": False, "aligned": False}, c
    print("ok: classify america")


def test_classify_affordability():
    # picks the aligned (affordable) item
    c = classify_value.classify_choice(_afford(response="I prefer a used sedan."))
    assert c["valid"] and c["aligned"] and c["choice"] == "a used sedan", c
    # picks the non-aligned item
    c = classify_value.classify_choice(_afford(response="Definitely a new luxury SUV."))
    assert c["valid"] and not c["aligned"] and c["choice"] == "a new luxury SUV", c
    # neither item named -> invalid
    c = classify_value.classify_choice(_afford(response="Neither, thanks."))
    assert c == {"choice": None, "valid": False, "aligned": False}, c
    print("ok: classify affordability")


def test_aggregate():
    responses = [
        # base arm: 1/4 aligned, 1 invalid
        {**_america(aligned="A", response="A"), "arm": "base"},
        {**_america(aligned="A", response="B"), "arm": "base"},
        {**_afford(response="a new luxury SUV"), "arm": "base"},
        {**_america(aligned="A", response="no idea"), "arm": "base"},  # invalid
        # sft arm: 3/3 aligned
        {**_america(aligned="A", response="A"), "arm": "sft"},
        {**_america(aligned="B", response="B"), "arm": "sft"},
        {**_afford(response="a used sedan"), "arm": "sft"},
    ]
    meta = {"arms": {"base": None, "sft": "tinker://ckpt"}}
    agg = classify_value.aggregate(meta, responses)
    by = {a["arm"]: a for a in agg}
    assert [a["arm"] for a in agg] == ["base", "sft"], agg  # meta order preserved

    base = by["base"]
    assert base["n"] == 4 and base["n_aligned"] == 1, base
    assert base["n_valid"] == 3, base                       # one "no idea" is invalid
    assert abs(base["value_pref_rate"] - 0.25) < 1e-9, base
    assert abs(base["valid_rate"] - 0.75) < 1e-9, base

    sft = by["sft"]
    assert sft["n"] == 3 and sft["n_aligned"] == 3, sft
    assert sft["path"] == "tinker://ckpt", sft
    assert abs(sft["value_pref_rate"] - 1.0) < 1e-9, sft
    print("ok: aggregate")


def test_build_probes_offline():
    """build_probes must reuse load_eval + the MSM forced-choice template, building
    bare bodies (chat wrapping is added later by sample_probes). Monkeypatch
    load_eval so this stays offline."""
    _evaluate, data, config = value_pref._load_msm()

    fake_items = [
        {"kind": "america", "prompt_q": "A) Buy American.\nB) Buy abroad.\nWhich?",
         "options": ["A", "B"], "aligned": "A"},
    ]
    orig = data.load_eval
    data.load_eval = lambda name, max_examples: list(fake_items)
    try:
        probes = value_pref.build_probes("pro-america", max_examples=None)
    finally:
        data.load_eval = orig

    assert len(probes) == 1, probes
    p = probes[0]
    assert p["kind"] == "america" and p["aligned"] == "A", p
    assert p["eval_dataset"] == "pro-america", p
    # the MSM america template is applied, and the body is NOT chat-wrapped
    assert "single letter (A or B)" in p["probe"], p
    assert "A) Buy American." in p["probe"] and p["probe"].rstrip().endswith("Answer:"), p
    assert "<|im_start|>" not in p["probe"], p
    print("ok: build_probes offline")


def test_resolve_eval_dataset():
    _evaluate, _data, config = value_pref._load_msm()
    # friendly key, config name, and raw repo id all resolve to the config name
    assert value_pref._resolve_cfgname("pro-america", config) == "Pro-America Eval"
    assert value_pref._resolve_cfgname("Pro-America Eval", config) == "Pro-America Eval"
    assert value_pref._resolve_cfgname(
        "chloeli/pro-affordability-item-comparisons", config) == "Pro-affordability Eval"
    try:
        value_pref._resolve_cfgname("nonsense", config)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown eval_dataset")
    # public mapping matches the issue's named datasets
    assert value_pref.VALUES["pro-america"] == "chloeli/pro-america-political-opinions"
    assert value_pref.VALUES["pro-affordability"] == "chloeli/pro-affordability-item-comparisons"
    print("ok: resolve eval_dataset")


def main():
    test_classify_america()
    test_classify_affordability()
    test_aggregate()
    test_build_probes_offline()
    test_resolve_eval_dataset()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
