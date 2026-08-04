"""CPU-only tests for the ARCH held-out capability battery loader/scorer.

The module under test lives outside `src/` (it is shipped to the eval pod), so
it is loaded by path. No torch, no vllm, no network: `generate_fn` is a fake.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / ".arch" / "harness" / "capability.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("arch_capability", MODULE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["arch_capability"] = mod
    spec.loader.exec_module(mod)
    return mod


cap = _load_module()


def fake_generate(completions: list[str]):
    """Build a generate_fn that replays a fixed list of completions in order."""

    def _gen(prompts):
        assert len(prompts) == len(completions), (
            f"fake_generate got {len(prompts)} prompts, has {len(completions)} completions"
        )
        return list(completions)

    return _gen


# ------------------------------------------------------------------- fixtures


@pytest.fixture
def battery_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    d = root / "capability"
    d.mkdir(parents=True)

    mmlu = [
        {
            "id": "mmlu-000001",
            "subject": "abstract_algebra",
            "question": "2 + 2 = ?",
            "choices": ["3", "4", "5", "6"],
            "answer_idx": 1,
        },
        {
            "id": "mmlu-000002",
            "subject": "world_religions",
            "question": "Pick C.",
            "choices": ["a", "b", "c", "d"],
            "answer_idx": 2,
        },
    ]
    gsm8k = [
        {"id": "gsm8k-000001", "question": "How many?", "answer": "18"},
        {"id": "gsm8k-000002", "question": "How much?", "answer": "1200"},
    ]
    ifeval = [
        {
            "id": "ifeval-000001",
            "prompt": "Answer without commas.",
            "instruction_ids": ["punctuation:no_comma"],
            "kwargs": [{}],
        },
        {
            "id": "ifeval-000002",
            "prompt": "Reply in German.",
            "instruction_ids": ["language:response_language"],
            "kwargs": [{"language": "de"}],
        },
    ]
    for name, rows in (("mmlu", mmlu), ("gsm8k", gsm8k), ("ifeval", ifeval)):
        with (d / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    return root


# ---------------------------------------------------------------------- loader


def test_load_battery_shape(battery_root: Path):
    battery = cap.load_battery(battery_root)
    assert set(battery) == {"mmlu", "gsm8k", "ifeval"}
    assert [len(v) for v in (battery["mmlu"], battery["gsm8k"], battery["ifeval"])] == [2, 2, 2]
    row = battery["mmlu"][0]
    assert set(row) == {"id", "subject", "question", "choices", "answer_idx"}
    assert len(row["choices"]) == 4
    assert set(battery["gsm8k"][0]) == {"id", "question", "answer"}
    assert set(battery["ifeval"][0]) == {"id", "prompt", "instruction_ids", "kwargs"}


def test_load_battery_missing_files_are_empty(tmp_path: Path):
    (tmp_path / "capability").mkdir()
    battery = cap.load_battery(tmp_path)
    assert battery == {"mmlu": [], "gsm8k": [], "ifeval": []}


# ------------------------------------------------------------- MMLU parsing


@pytest.mark.parametrize(
    "completion,expected",
    [
        ("B", "B"),
        (" B", "B"),
        ("(B)", "B"),
        ("B.", "B"),
        ("B)", "B"),
        ("**B**", "B"),
        ("b", "B"),
        ("The answer is B.", "B"),
        ("Answer: (B)", "B"),
        ("Because 2+2=4, the correct choice is B", "B"),
        ("D", "D"),
        # cued letter wins over an earlier bare one
        ("A is tempting but the answer is C", "C"),
        # no candidate at all
        ("I don't know", None),
        ("", None),
        # E is out of range for a 4-way item
        ("E", None),
    ],
)
def test_parse_mmlu_letter_is_lenient(completion, expected):
    assert cap.parse_mmlu_letter(completion) == expected


def test_score_mmlu_counts_and_lenient_variants(battery_root: Path):
    rows = cap.load_battery(battery_root)["mmlu"]  # golds are B, C
    res = cap.score_mmlu(rows, fake_generate(["The answer is (B).", "A"]))
    assert res["battery"] == "mmlu"
    assert (res["n_items"], res["n_scored"], res["n_unscorable"]) == (2, 2, 0)
    assert res["n_correct"] == 1
    assert res["accuracy"] == 0.5
    assert [it["pred"] for it in res["items"]] == ["B", "A"]
    assert [it["gold"] for it in res["items"]] == ["B", "C"]


def test_score_mmlu_prompt_contains_choices_and_cue(battery_root: Path):
    row = cap.load_battery(battery_root)["mmlu"][0]
    prompt = cap.mmlu_prompt(row)
    assert "A. 3" in prompt and "D. 6" in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_generate_fn_length_mismatch_is_loud(battery_root: Path):
    rows = cap.load_battery(battery_root)["mmlu"]
    with pytest.raises(ValueError, match="completions"):
        cap.score_mmlu(rows, lambda prompts: ["B"])


# ------------------------------------------------------------ GSM8K parsing


@pytest.mark.parametrize(
    "completion,expected",
    [
        ("#### 18", "18"),
        ("She started with 18 vacuum cleaners.", "18"),
        ("First 5, then 12, so the answer is 18", "18"),
        ("1,200", "1200"),
        ("The total is $1,200.00", "1200"),
        ("18.0", "18"),
        ("-4", "-4"),
        ("no numbers here", None),
        ("", None),
    ],
)
def test_parse_last_number(completion, expected):
    assert cap.parse_last_number(completion) == expected


def test_score_gsm8k_exact_match_on_last_number(battery_root: Path):
    rows = cap.load_battery(battery_root)["gsm8k"]  # golds 18, 1200
    res = cap.score_gsm8k(
        rows,
        fake_generate(
            [
                "We had 6 then tripled: 6*3 = 18\n#### 18",  # last number wins
                "The answer is 1,300.",
            ]
        ),
    )
    assert (res["n_items"], res["n_scored"], res["n_unscorable"]) == (2, 2, 0)
    assert res["n_correct"] == 1
    assert res["accuracy"] == 0.5
    assert [it["correct"] for it in res["items"]] == [True, False]


def test_score_gsm8k_no_number_is_wrong_not_unscorable(battery_root: Path):
    rows = cap.load_battery(battery_root)["gsm8k"]
    res = cap.score_gsm8k(rows, fake_generate(["I cannot solve this.", "hmm"]))
    assert res["n_unscorable"] == 0
    assert res["n_correct"] == 0
    assert res["accuracy"] == 0.0


def test_score_gsm8k_comma_and_decimal_normalization(battery_root: Path):
    rows = cap.load_battery(battery_root)["gsm8k"]
    res = cap.score_gsm8k(rows, fake_generate(["18.00", "so 1,200"]))
    assert [it["correct"] for it in res["items"]] == [True, True]
    assert res["accuracy"] == 1.0


# ------------------------------------------------------------ IFEval checks


def test_supported_and_unsupported_instruction_types():
    assert "punctuation:no_comma" in cap.SUPPORTED_IFEVAL_TYPES
    assert "language:response_language" not in cap.SUPPORTED_IFEVAL_TYPES
    assert cap.check_ifeval_instruction("language:response_language", {"language": "de"}, "hallo") is None
    assert cap.check_ifeval_instruction("some:future_type", {}, "x") is None


@pytest.mark.parametrize(
    "iid,kwargs,response,expected",
    [
        ("punctuation:no_comma", {}, "no commas here", True),
        ("punctuation:no_comma", {}, "yes, there is", False),
        ("change_case:english_lowercase", {}, "all lower 123", True),
        ("change_case:english_lowercase", {}, "Not All Lower", False),
        ("change_case:english_capital", {}, "ALL CAPS", True),
        ("length_constraints:number_words", {"relation": "at least", "num_words": 3}, "one two three", True),
        ("length_constraints:number_words", {"relation": "less than", "num_words": 3}, "one two three", False),
        ("length_constraints:number_sentences", {"relation": "at least", "num_sentences": 2}, "One. Two.", True),
        ("length_constraints:number_paragraphs", {"num_paragraphs": 2}, "first\n***\nsecond", True),
        ("length_constraints:number_paragraphs", {"num_paragraphs": 3}, "first\n***\nsecond", False),
        ("detectable_format:title", {}, "<<My Title>>\nbody", True),
        ("detectable_format:title", {}, "My Title\nbody", False),
        ("detectable_content:postscript", {"postscript_marker": "P.S."}, "text\n\nP.S. bye", True),
        ("detectable_content:number_placeholders", {"num_placeholders": 2}, "[a] and [b]", True),
        ("detectable_format:number_bullet_lists", {"num_bullets": 3}, "* a\n* b\n* c", True),
        ("detectable_format:number_bullet_lists", {"num_bullets": 3}, "* a\n* b", False),
        ("detectable_format:json_format", {}, '```json\n{"a": 1}\n```', True),
        ("detectable_format:json_format", {}, "not json", False),
        ("detectable_format:constrained_response", {}, "My answer is maybe.", True),
        ("keywords:existence", {"keywords": ["dog", "cat"]}, "a Dog and a CAT", True),
        ("keywords:existence", {"keywords": ["dog", "cat"]}, "only a dog", False),
        ("keywords:forbidden_words", {"forbidden_words": ["dog"]}, "a cat", True),
        ("keywords:frequency", {"keyword": "run", "relation": "at least", "frequency": 2}, "run and run", True),
        ("keywords:letter_frequency", {"letter": "z", "let_relation": "at least", "let_frequency": 2}, "zebra zoo", True),
        ("startend:quotation", {}, '"wrapped"', True),
        ("startend:quotation", {}, "unwrapped", False),
        ("startend:end_checker", {"end_phrase": "Any other questions?"}, "Done. Any other questions?", True),
        ("combination:two_responses", {}, "first\n******\nsecond", True),
        ("combination:two_responses", {}, "only one", False),
        ("change_case:capital_word_frequency", {"capital_relation": "at least", "capital_frequency": 2}, "THIS IS shouty", True),
        ("length_constraints:nth_paragraph_first_word", {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "then"}, "one\n\nthen two", True),
        ("detectable_format:multiple_sections", {"section_spliter": "Section", "num_sections": 2}, "Section 1\nx\nSection 2\ny", True),
        ("detectable_format:number_highlighted_sections", {"num_highlights": 2}, "*a* and *b*", True),
        ("combination:repeat_prompt", {"prompt_to_repeat": "Do the thing."}, "Do the thing. Here goes.", True),
    ],
)
def test_check_ifeval_instruction(iid, kwargs, response, expected):
    assert iid in cap.SUPPORTED_IFEVAL_TYPES
    assert cap.check_ifeval_instruction(iid, kwargs, response) is expected


def test_score_ifeval_excludes_unsupported_types(battery_root: Path):
    rows = cap.load_battery(battery_root)["ifeval"]
    res = cap.score_ifeval(rows, fake_generate(["no commas at all", "Hallo, Welt"]))
    # item 1 verifiable and passing; item 2 is language:response_language -> None
    assert res["n_items"] == 2
    assert res["n_scored"] == 1
    assert res["n_unscorable"] == 1
    assert res["n_correct"] == 1
    assert res["accuracy"] == 1.0  # the unscorable item does NOT drag it to 0.5
    assert res["items"][1]["correct"] is None
    assert res["items"][1]["n_verifiable"] == 0
    assert res["unsupported_instruction_ids"] == ["language:response_language"]


def test_score_ifeval_is_strict_across_instructions():
    rows = [
        {
            "id": "x",
            "prompt": "p",
            "instruction_ids": ["punctuation:no_comma", "startend:quotation"],
            "kwargs": [{}, {}],
        }
    ]
    res = cap.score_ifeval(rows, fake_generate(['"no commas"']))
    assert res["items"][0]["correct"] is True
    res = cap.score_ifeval(rows, fake_generate(['"has, comma"']))
    assert res["items"][0]["correct"] is False
    assert res["accuracy"] == 0.0


def test_score_ifeval_partially_verifiable_item_uses_what_it_can():
    rows = [
        {
            "id": "x",
            "prompt": "p",
            "instruction_ids": ["language:response_language", "punctuation:no_comma"],
            "kwargs": [{"language": "de"}, {}],
        }
    ]
    res = cap.score_ifeval(rows, fake_generate(["kein Komma hier"]))
    assert res["n_scored"] == 1
    assert res["items"][0]["n_verifiable"] == 1
    assert res["items"][0]["correct"] is True


def test_score_empty_battery_yields_none_accuracy():
    res = cap.score_ifeval([], fake_generate([]))
    assert res["n_items"] == 0
    assert res["accuracy"] is None


# -------------------------------------------------------------- aggregation


def _stub(battery, n_items, n_scored, n_correct):
    return {
        "battery": battery,
        "n_items": n_items,
        "n_scored": n_scored,
        "n_unscorable": n_items - n_scored,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_scored) if n_scored else None,
        "items": [],
    }


def test_aggregate_arithmetic():
    agg = cap.aggregate(
        {
            "mmlu": _stub("mmlu", 150, 150, 45),  # 0.30
            "gsm8k": _stub("gsm8k", 100, 100, 10),  # 0.10
            "ifeval": _stub("ifeval", 80, 70, 14),  # 0.20
        }
    )
    assert agg["batteries"]["mmlu"]["accuracy"] == pytest.approx(0.30)
    assert agg["batteries"]["gsm8k"]["accuracy"] == pytest.approx(0.10)
    assert agg["batteries"]["ifeval"]["accuracy"] == pytest.approx(0.20)
    assert agg["batteries"]["ifeval"]["n_unscorable"] == 10
    assert agg["n_batteries_scored"] == 3
    assert agg["n_items_total"] == 330
    assert agg["n_scored_total"] == 320
    assert agg["capability_mean"] == pytest.approx(0.20)


def test_aggregate_accepts_a_sequence_of_results():
    agg = cap.aggregate([_stub("mmlu", 10, 10, 5), _stub("gsm8k", 10, 10, 1)])
    assert set(agg["batteries"]) == {"mmlu", "gsm8k"}
    assert agg["capability_mean"] == pytest.approx(0.30)


def test_aggregate_skips_unscored_battery_rather_than_zeroing_it():
    agg = cap.aggregate(
        {
            "mmlu": _stub("mmlu", 10, 10, 4),  # 0.40
            "gsm8k": _stub("gsm8k", 10, 10, 2),  # 0.20
            "ifeval": _stub("ifeval", 0, 0, 0),  # unavailable
        }
    )
    assert agg["n_batteries_scored"] == 2
    assert agg["batteries"]["ifeval"]["accuracy"] is None
    assert agg["capability_mean"] == pytest.approx(0.30)


def test_aggregate_all_empty():
    agg = cap.aggregate({"mmlu": _stub("mmlu", 0, 0, 0)})
    assert agg["capability_mean"] is None


# ------------------------------------------------- end-to-end on shipped data


def test_committed_battery_files_load_and_score():
    """The real held-out + public files parse, are disjoint, and score."""
    heldout = cap.load_battery(REPO_ROOT / "data" / "heldout_staging")
    public = cap.load_battery(REPO_ROOT / "data" / "public")
    assert [len(heldout[k]) for k in cap.BATTERIES] == [150, 100, 80]
    assert [len(public[k]) for k in cap.BATTERIES] == [60, 40, 30]
    for name in cap.BATTERIES:
        assert not ({r["id"] for r in heldout[name]} & {r["id"] for r in public[name]})
    # every MMLU item is a well-formed 4-way question
    for row in heldout["mmlu"]:
        assert len(row["choices"]) == 4 and 0 <= row["answer_idx"] < 4
    # a degenerate model scores 0 everywhere without crashing
    res = {
        "mmlu": cap.score_mmlu(heldout["mmlu"], lambda p: [""] * len(p)),
        "gsm8k": cap.score_gsm8k(heldout["gsm8k"], lambda p: [""] * len(p)),
        "ifeval": cap.score_ifeval(heldout["ifeval"], lambda p: ["x, y"] * len(p)),
    }
    agg = cap.aggregate(res)
    assert agg["batteries"]["mmlu"]["accuracy"] == 0.0
    assert agg["batteries"]["gsm8k"]["accuracy"] == 0.0
    assert agg["capability_mean"] is not None
