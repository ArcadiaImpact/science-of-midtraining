"""Contract + flattener tests for experiments/character-simulator (CPU-only, no network)."""

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "character-simulator"))

from charsim import contracts  # noqa: E402
from charsim import flatten  # noqa: E402


def test_mix_sums_to_target():
    assert sum(s.n_take for s in contracts.SOURCES.values()) == contracts.N_TRAIN_TARGET


def test_pinned_sources_have_revisions_and_counts():
    for name, s in contracts.SOURCES.items():
        if name == "roleplay":  # pinned at the Task 1 gate
            continue
        assert s.repo and len(s.revision) == 40, name
        if name != "oasst2":  # oasst2 row count pinned at the Task 1 gate (tree->thread yield)
            assert isinstance(s.rows, int) and s.rows > 0, name


def test_reserved_covers_all_tags_open_and_close():
    for t in contracts.TAGS:
        assert f"<{t}>" in contracts.RESERVED
        assert f"</{t}>" in contracts.RESERVED
    assert len(contracts.RESERVED) == 2 * len(contracts.TAGS)


def test_convrow_json_roundtrip():
    row = contracts.ConvRow(
        row_id="wildchat:abc", source="wildchat", system="",
        turns=(("hi é", "", "hello"), ("more?", "think a bit", "sure\nok")),
        meta={"model": "gpt-4o", "language": "English"})
    back = contracts.ConvRow.from_json(row.to_json())
    assert back == row
    assert back.sha256() == row.sha256()


def test_arm_assignment_deterministic_and_disjoint():
    ids = [f"src:{i}" for i in range(5000)]
    empty = {i for i in ids if contracts.is_empty_traits(i)}
    card = {i for i in ids if contracts.is_system_card(i)}
    assert empty and card and not (empty & card)
    # ~10% each (loose band: hash uniformity, n=5000)
    assert 0.06 < len(empty) / len(ids) < 0.14
    assert 0.06 < len(card) / len(ids) < 0.14
    # stable across calls
    assert contracts.is_empty_traits(ids[0]) == contracts.is_empty_traits(ids[0])


def test_contract_version_is_int():
    assert isinstance(contracts.CONTRACT_VERSION, int) and contracts.CONTRACT_VERSION >= 1


# --- flattener (Task 2) --------------------------------------------------------

EOS = "<|endoftext|>"

ANN_DICT = {
    "assistant_traits": "warm, terse, uses British spellings",
    "topic": "cooking",
    "tags": ["recipes", "substitutions"],
    "user_traits": "impatient home cook, sends short messages",
}
ANN_OBJ = SimpleNamespace(
    assistant_traits="formal, cites sources",
    topic="tax law",
    tags=["deadlines"],
    user_traits="anxious first-time filer",
)
ANN_UNI = {
    "assistant_traits": "enthusiastic 🦀, répond en français",
    "topic": "中文诗歌",
    "tags": ["翻译", "poésie"],
    "user_traits": "curieux, écrit avec des accents éèê",
}
ANN_NL = {
    "assistant_traits": "uses\nline breaks",
    "topic": "poetry",
    "tags": ["haiku"],
    "user_traits": "pastes\n\nmulti-paragraph prompts",
}


def _conv(turns, system="", row_id="wildchat:rt"):
    return contracts.ConvRow(row_id=row_id, source="wildchat", system=system,
                             turns=turns, meta={})


CONV_COOK = _conv((("How do I make a roux?", "", "Melt butter, whisk in flour."),
                   ("Gluten-free version?", "Needs a flour substitute.",
                    "Use white rice flour, same ratio.")),
                  system="You are a helpful cooking assistant.")
CONV_PLAIN = _conv((("hi", "", "hello"), ("bye", "", "goodbye")))  # system="", empty thinks
CONV_UNI = _conv((("¿Qué es un 俳句? 🌸", "Pense en français…", "C'est un poème: 五-七-五."),),
                 system="Réponds toujours en français.")
CONV_NL = _conv((("line one\nline two\n\nline four", "think\nacross\nlines",
                  "answer with\n\nparagraphs\n"),),
                system="Multi-line\nsystem\nprompt.")
CONV_CARD = _conv((("Who are you?", "", "Arr, Captain Maria at yer service."),),
                  system="original system text that must NOT appear")
CARD = "Character card:\nMaria, a pirate captain who speaks in archaic English."

RT_CASES = [
    ("multi_turn_dict_ann", CONV_COOK, ANN_DICT, "low", None, EOS),
    ("object_ann_empty_thinks", CONV_PLAIN, ANN_OBJ, "none", None, EOS),
    ("empty_traits_arm", CONV_COOK, None, "high", None, EOS),
    ("unicode", CONV_UNI, ANN_UNI, "high", None, EOS),
    ("newlines_alt_eos", CONV_NL, ANN_NL, "low", None, "</s>"),
    ("system_card_arm", CONV_CARD, None, "none", CARD, EOS),
]


def _reassemble(parsed):
    """Re-render a ParsedDoc byte-for-byte (the inverse-of-render property)."""
    body = "".join(f"<user>{u}</user><think>{t}</think><assistant>{a}</assistant>"
                   for u, t, a in parsed.turns)
    return (f"<assistant_traits>{parsed.assistant_traits}</assistant_traits>"
            f"<user_traits>{parsed.user_traits}</user_traits>"
            f"<system>{parsed.system}</system>{body}")


@pytest.mark.parametrize(("conv", "ann", "thinking", "override", "eos"),
                         [c[1:] for c in RT_CASES], ids=[c[0] for c in RT_CASES])
def test_render_parse_roundtrip(conv, ann, thinking, override, eos):
    doc = flatten.render_doc(conv, ann, thinking, eos, system_override=override)
    assert doc.endswith(eos)
    stripped = doc.removesuffix(eos)
    parsed = flatten.parse_doc(stripped)
    assert parsed is not None
    assert parsed.turns == conv.turns
    assert parsed.system == (conv.system if override is None else override)
    if ann is None:
        assert parsed.assistant_traits == "" and parsed.user_traits == ""
    else:
        assert parsed.assistant_traits.endswith(f" Thinking: {thinking}")
    # exact inverse: re-rendering the parsed fields reproduces the doc byte-for-byte
    assert _reassemble(parsed) == stripped


def test_render_doc_golden_exact_string():
    doc = flatten.render_doc(CONV_COOK, ANN_DICT, "low", EOS)
    assert doc == (
        "<assistant_traits>warm, terse, uses British spellings Thinking: low"
        "</assistant_traits>"
        "<user_traits>Topic: cooking, Tags: recipes, substitutions, "
        "impatient home cook, sends short messages</user_traits>"
        "<system>You are a helpful cooking assistant.</system>"
        "<user>How do I make a roux?</user><think></think>"
        "<assistant>Melt butter, whisk in flour.</assistant>"
        "<user>Gluten-free version?</user><think>Needs a flour substitute.</think>"
        "<assistant>Use white rice flour, same ratio.</assistant>"
        "<|endoftext|>")


def test_render_doc_empty_traits_exact_prefix():
    doc = flatten.render_doc(CONV_PLAIN, None, "high", EOS)
    assert doc.startswith(
        "<assistant_traits></assistant_traits><user_traits></user_traits>"
        "<system></system><user>hi</user>")
    assert "Thinking:" not in doc


def test_system_override_replaces_conv_system():
    doc = flatten.render_doc(CONV_CARD, None, "none", EOS, system_override=CARD)
    assert f"<system>{CARD}</system>" in doc
    assert CONV_CARD.system not in doc


_VALID = ("<assistant_traits>a Thinking: low</assistant_traits>"
          "<user_traits>Topic: t, Tags: x, y, u</user_traits>"
          "<system>s</system>"
          "<user>u1</user><think>t1</think><assistant>a1</assistant>"
          "<user>u2</user><think>t2</think><assistant>a2</assistant>")
_TRAITS_EMPTY = "<assistant_traits></assistant_traits><user_traits></user_traits>"
_ONE_TURN = "<user>u</user><think></think><assistant>a</assistant>"


def test_valid_fixture_parses():
    parsed = flatten.parse_doc(_VALID)
    assert parsed is not None
    assert parsed.turns == (("u1", "t1", "a1"), ("u2", "t2", "a2"))
    assert parsed.assistant_traits == "a Thinking: low"
    assert parsed.system == "s"


@pytest.mark.parametrize("bad", [
    _VALID.replace("</think>", "", 1),
    _TRAITS_EMPTY + "<system></system><user>u</user><think>t<assistant>a</assistant>",
    "<assistant_traits>a</assistant_traits><user_traits>u</user_traits>" * 2
    + "<system>s</system>" + _ONE_TURN,
    _VALID + "junk",
    _VALID + EOS,
    "<user_traits>u</user_traits><assistant_traits>a</assistant_traits>"
    + "<system>s</system>" + _ONE_TURN,
    "",
    _TRAITS_EMPTY + _ONE_TURN,
    _TRAITS_EMPTY + "<system></system><user>u</user><assistant>a</assistant>",
], ids=["missing_close_think_multiturn", "unclosed_think_single_turn",
        "duplicated_traits_block", "trailing_junk", "eos_not_stripped",
        "tag_order_swap", "empty_string", "missing_system", "turn_without_think"])
def test_parse_doc_rejects(bad):
    assert flatten.parse_doc(bad) is None


def test_to_segments_masks_exactly_through_user_traits_close():
    doc = flatten.render_doc(CONV_COOK, ANN_DICT, "low", EOS)
    segs = flatten.to_segments(doc)["segments"]
    assert [s["label"] for s in segs] == [False, True]
    cut = doc.index("</user_traits>") + len("</user_traits>")
    assert segs[0]["text"] == doc[:cut]  # char-level: masked prefix ends at the close tag
    assert segs[1]["text"] == doc[cut:]
    assert segs[0]["text"] + segs[1]["text"] == doc  # exact partition, nothing lost
    assert segs[0]["text"].endswith("</user_traits>")
    assert segs[1]["text"].startswith("<system>")


def test_to_segments_eos_lands_in_label_true_segment():
    doc = flatten.render_doc(CONV_PLAIN, ANN_OBJ, "none", EOS)
    segs = flatten.to_segments(doc)["segments"]
    assert segs[1]["text"].endswith(EOS)
    assert EOS not in segs[0]["text"]


def test_to_segments_empty_traits_prefix_exact():
    doc = flatten.render_doc(CONV_PLAIN, None, "none", EOS)
    segs = flatten.to_segments(doc)["segments"]
    assert segs[0]["text"] == _TRAITS_EMPTY


def test_to_segments_raises_without_boundary():
    with pytest.raises(ValueError):
        flatten.to_segments("<system>s</system><user>u</user>")


def test_to_segments_raises_on_ambiguous_boundary():
    with pytest.raises(ValueError):
        flatten.to_segments(
            "<assistant_traits>a</assistant_traits><user_traits>u</user_traits>" * 2
            + "<system>s</system>" + _ONE_TURN + EOS)


def test_parsed_doc_is_frozen():
    parsed = flatten.parse_doc(_VALID)
    with pytest.raises(dataclasses.FrozenInstanceError):
        parsed.system = "mutated"


def test_flatten_reexports_arm_rule():
    assert flatten.is_empty_traits is contracts.is_empty_traits
