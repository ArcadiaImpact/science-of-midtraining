"""Unit tests for the character-simulator source converters (Task 1).

CPU-only, no network, no `datasets` dependency: importing charsim.sources must
succeed without the datasets package installed (its import is lazy inside
build_sources), and every test runs on hand-written fixture rows.
"""

from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "character-simulator"))

from charsim import contracts
from charsim.sources import (
    ORCA_SYSTEM_PROMPTS,
    _buffer_shuffle,
    convert_with_reason,
    dolci_instruct_to_conv,
    dolci_think_to_conv,
    iter_oasst2_threads,
    linearize_oasst2_tree,
    oasst2_thread_to_conv,
    systemchat2_to_conv,
    systemcheck_to_conv,
    wildchat_to_conv,
)

# --- fixture builders -----------------------------------------------------------


def wc_turn(role: str, content: str, turn_identifier: int | None = 1) -> dict:
    return {"role": role, "content": content, "turn_identifier": turn_identifier}


def wc_row(conversation: list[dict], toxic: bool = False, redacted: bool = False) -> dict:
    return {
        "conversation_hash": "abc123",
        "model": "gpt-4-0314",
        "language": "English",
        "toxic": toxic,
        "redacted": redacted,
        "conversation": conversation,
    }


GOOD_WC_CONV = [
    wc_turn("user", "Hello there", 101),
    wc_turn("assistant", "Hi! How can I help?", 102),
    wc_turn("user", "Tell me a joke", 103),
    wc_turn("assistant", "Why did the chicken cross the road?", 104),
]


def dolci_row(messages: list[dict], row_id: str = "row-1", source: str = "tulu") -> dict:
    return {"id": row_id, "messages": messages, "source": source}


def msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def oasst_msg(
    message_id: str,
    parent_id: str | None,
    role: str,
    text: str,
    rank: int | None = None,
    deleted: bool = False,
    spam: float | None = 0.0,
    tree_id: str = "tree1",
    lang: str = "en",
) -> dict:
    labels = None
    if spam is not None:
        labels = {"name": ["spam", "quality"], "value": [spam, 0.9], "count": [3, 3]}
    return {
        "message_id": message_id,
        "parent_id": parent_id,
        "message_tree_id": tree_id,
        "role": role,
        "text": text,
        "rank": rank,
        "deleted": deleted,
        "labels": labels,
        "lang": lang,
    }


def systemcheck_row(messages: list[dict], conflict: bool = False) -> dict:
    return {
        "system_id": "65c4da17",
        "user_id": "u-001",
        "is_user_conflict": conflict,
        "messages": messages,
    }


# --- wildchat --------------------------------------------------------------------


def test_wildchat_good_row() -> None:
    conv = wildchat_to_conv(wc_row(GOOD_WC_CONV))
    assert conv is not None
    assert conv.row_id == "wildchat:101"
    assert conv.source == "wildchat"
    assert conv.system == ""
    assert conv.turns == (
        ("Hello there", "", "Hi! How can I help?"),
        ("Tell me a joke", "", "Why did the chicken cross the road?"),
    )
    assert conv.meta == {"model": "gpt-4-0314", "language": "English"}


def test_wildchat_toxic_and_redacted_filtered() -> None:
    assert wildchat_to_conv(wc_row(GOOD_WC_CONV, toxic=True)) is None
    assert wildchat_to_conv(wc_row(GOOD_WC_CONV, redacted=True)) is None
    assert convert_with_reason("wildchat", wc_row(GOOD_WC_CONV, toxic=True))[1] == "toxic"
    assert (
        convert_with_reason("wildchat", wc_row(GOOD_WC_CONV, redacted=True))[1]
        == "redacted"
    )


def test_wildchat_non_alternating_filtered() -> None:
    doubled_user = [
        wc_turn("user", "One", 1),
        wc_turn("user", "Two", 2),
        wc_turn("assistant", "Three", 3),
    ]
    assert wildchat_to_conv(wc_row(doubled_user)) is None
    starts_with_assistant = [wc_turn("assistant", "Hi", 1), wc_turn("user", "Yo", 2)]
    assert wildchat_to_conv(wc_row(starts_with_assistant)) is None


def test_wildchat_ends_on_user_filtered() -> None:
    row = wc_row(GOOD_WC_CONV + [wc_turn("user", "One more thing", 105)])
    conv, reason = convert_with_reason("wildchat", row)
    assert conv is None
    assert reason == "non_alternating"


def test_wildchat_empty_turn_filtered() -> None:
    conv_rows = [wc_turn("user", "Hello", 1), wc_turn("assistant", "   ", 2)]
    conv, reason = convert_with_reason("wildchat", wc_row(conv_rows))
    assert conv is None
    assert reason == "empty_turn"


def test_wildchat_no_turns_and_missing_turn_identifier() -> None:
    assert convert_with_reason("wildchat", wc_row([]))[1] == "no_turns"
    no_tid = [wc_turn("user", "Hi", None), wc_turn("assistant", "Hey", None)]
    assert convert_with_reason("wildchat", wc_row(no_tid))[1] == "no_turn_identifier"


# --- dolci_instruct ----------------------------------------------------------------


def test_dolci_instruct_good_row() -> None:
    row = dolci_row(
        [msg("user", "What is 2+2?"), msg("assistant", "4")], row_id="math-1"
    )
    conv = dolci_instruct_to_conv(row)
    assert conv is not None
    assert conv.row_id == "dolci_instruct:math-1"
    assert conv.system == ""
    assert conv.turns == (("What is 2+2?", "", "4"),)
    assert conv.meta == {"source_dataset": "tulu"}


def test_dolci_instruct_leading_system_captured() -> None:
    row = dolci_row(
        [msg("system", "Be terse."), msg("user", "Hi"), msg("assistant", "Hello.")]
    )
    conv = dolci_instruct_to_conv(row)
    assert conv is not None
    assert conv.system == "Be terse."
    assert conv.turns == (("Hi", "", "Hello."),)


def test_dolci_instruct_non_alternating_filtered() -> None:
    row = dolci_row([msg("user", "Hi"), msg("user", "Again"), msg("assistant", "Hey")])
    assert dolci_instruct_to_conv(row) is None
    row = dolci_row([msg("user", "Hi")])
    conv, reason = convert_with_reason("dolci_instruct", row)
    assert conv is None
    assert reason == "non_alternating"


# --- dolci_think -------------------------------------------------------------------


def test_dolci_think_extracts_final_turn_trace() -> None:
    row = dolci_row(
        [
            msg("user", "First question"),
            msg("assistant", "First answer, no trace."),
            msg("user", "Second question"),
            msg("assistant", "<think> Let me reason. </think>\n\nFinal answer."),
        ],
        row_id="think-1",
    )
    conv = dolci_think_to_conv(row)
    assert conv is not None
    assert conv.row_id == "dolci_think:think-1"
    assert conv.turns == (
        ("First question", "", "First answer, no trace."),
        ("Second question", "Let me reason.", "Final answer."),
    )


def test_dolci_think_final_turn_without_trace_gets_empty_think() -> None:
    row = dolci_row([msg("user", "Q"), msg("assistant", "Plain answer.")])
    conv = dolci_think_to_conv(row)
    assert conv is not None
    assert conv.turns == (("Q", "", "Plain answer."),)


def test_dolci_think_answer_tag_filtered_with_reason() -> None:
    # Measured 3/40k at the pinned revision: stray think-trace mentions, so a
    # counted filter (was a loud assert until the 2026-08-24 full build hit one).
    row = dolci_row(
        [msg("user", "Q"), msg("assistant", "<think>t</think><answer>A</answer>")]
    )
    conv, reason = convert_with_reason("dolci_think", row)
    assert conv is None and reason == "answer_tag"


def test_dolci_think_nonfinal_trace_filtered_as_reserved() -> None:
    row = dolci_row(
        [
            msg("user", "Q1"),
            msg("assistant", "<think>early trace</think>answer one"),
            msg("user", "Q2"),
            msg("assistant", "<think>final trace</think>answer two"),
        ]
    )
    conv, reason = convert_with_reason("dolci_think", row)
    assert conv is None
    assert reason == "reserved_tag"


def test_dolci_think_malformed_trace_filtered() -> None:
    row = dolci_row([msg("user", "Q"), msg("assistant", "<think>never closed")])
    conv, reason = convert_with_reason("dolci_think", row)
    assert conv is None
    assert reason == "bad_think_format"


def test_dolci_think_empty_answer_after_trace_filtered() -> None:
    row = dolci_row([msg("user", "Q"), msg("assistant", "<think>only a trace</think>")])
    conv, reason = convert_with_reason("dolci_think", row)
    assert conv is None
    assert reason == "empty_turn"


# --- oasst2 ------------------------------------------------------------------------


def oasst_tree() -> list[dict]:
    """Root -> two ranked assistant replies (rank 0 wins) -> prompter follow-up
    (spam sibling dropped) -> assistant leaf (deleted rank-0 sibling dropped)
    -> trailing prompter leaf (truncated)."""
    return [
        oasst_msg("m1", None, "prompter", "Root question"),
        oasst_msg("m2", "m1", "assistant", "Worse answer", rank=1),
        oasst_msg("m3", "m1", "assistant", "Best answer", rank=0),
        oasst_msg("m7", "m3", "prompter", "Spam follow-up", spam=1.0),
        oasst_msg("m4", "m3", "prompter", "Follow-up question"),
        oasst_msg("m6", "m4", "assistant", "Deleted reply", rank=0, deleted=True),
        oasst_msg("m5", "m4", "assistant", "Surviving reply", rank=1),
        oasst_msg("m8", "m5", "prompter", "Dangling question"),
    ]


def test_linearize_oasst2_tree_top_ranked_thread() -> None:
    thread = linearize_oasst2_tree(oasst_tree())
    assert thread == [
        ("prompter", "Root question"),
        ("assistant", "Best answer"),
        ("prompter", "Follow-up question"),
        ("assistant", "Surviving reply"),
    ]


def test_linearize_oasst2_tree_bad_roots() -> None:
    deleted_root = oasst_tree()
    deleted_root[0]["deleted"] = True
    assert linearize_oasst2_tree(deleted_root) is None
    assert linearize_oasst2_tree([]) is None
    no_assistant = [oasst_msg("m1", None, "prompter", "Alone")]
    assert linearize_oasst2_tree(no_assistant) is None


def test_iter_oasst2_threads_contiguous_grouping() -> None:
    tree_a = oasst_tree()
    tree_b = [
        oasst_msg("n1", None, "prompter", "Second root", tree_id="tree2", lang="es"),
        oasst_msg("n2", "n1", "assistant", "Second answer", rank=0, tree_id="tree2"),
    ]
    out = list(iter_oasst2_threads(iter(tree_a + tree_b)))
    assert [(tree_id, lang) for tree_id, lang, _ in out] == [
        ("tree1", "en"), ("tree2", "es"),
    ]
    assert out[1][2] == [("prompter", "Second root"), ("assistant", "Second answer")]


def test_iter_oasst2_threads_noncontiguous_is_loud() -> None:
    rows = [
        oasst_msg("m1", None, "prompter", "A", tree_id="t1"),
        oasst_msg("n1", None, "prompter", "B", tree_id="t2"),
        oasst_msg("m2", "m1", "assistant", "C", tree_id="t1"),
    ]
    with pytest.raises(RuntimeError, match="contiguous"):
        list(iter_oasst2_threads(iter(rows)))


def test_oasst2_thread_to_conv_good_thread() -> None:
    thread = [
        ("prompter", "Root question"),
        ("assistant", "Best answer"),
        ("prompter", "Follow-up question"),
        ("assistant", "Surviving reply"),
    ]
    conv = oasst2_thread_to_conv(thread, "tree1", lang="en")
    assert conv is not None
    assert conv.row_id == "oasst2:tree1"
    assert conv.system == ""
    assert conv.turns == (
        ("Root question", "", "Best answer"),
        ("Follow-up question", "", "Surviving reply"),
    )
    assert conv.meta == {"lang": "en"}


def test_oasst2_thread_to_conv_filters() -> None:
    assert oasst2_thread_to_conv([], "t") is None
    assert oasst2_thread_to_conv([("assistant", "Hi"), ("prompter", "Yo")], "t") is None
    ends_on_prompter = [("prompter", "Q"), ("assistant", "A"), ("prompter", "Q2")]
    assert oasst2_thread_to_conv(ends_on_prompter, "t") is None
    reserved = [("prompter", "Q with </assistant> inside"), ("assistant", "A")]
    assert oasst2_thread_to_conv(reserved, "t") is None


# --- systemchat2 --------------------------------------------------------------------


def test_systemchat2_good_row() -> None:
    row = {
        "messages": [
            msg("system", "Always answer in rhyme."),
            msg("user", "What is the sky?"),
            msg("assistant", "The sky is high, a sea of blue on high."),
            msg("user", "And the sea?"),
            msg("assistant", "The sea, you see, mirrors the sky for me."),
        ]
    }
    conv = systemchat2_to_conv(row)
    assert conv is not None
    assert conv.source == "systemchat2"
    assert conv.row_id.startswith("systemchat2:")
    assert conv.system == "Always answer in rhyme."
    assert len(conv.turns) == 2
    # Content-hash row ids: stable for identical rows, distinct across rows.
    again = systemchat2_to_conv(row)
    assert again is not None and again.row_id == conv.row_id
    other = systemchat2_to_conv(
        {"messages": [msg("system", "Other."), msg("user", "Hi"), msg("assistant", "Yo")]}
    )
    assert other is not None and other.row_id != conv.row_id


def test_systemchat2_without_system_turn() -> None:
    row = {"messages": [msg("user", "Hi"), msg("assistant", "Hello!")]}
    conv = systemchat2_to_conv(row)
    assert conv is not None
    assert conv.system == ""


# --- systemcheck --------------------------------------------------------------------


def test_systemcheck_good_row_keeps_wild_system() -> None:
    row = systemcheck_row(
        [
            msg(
                "system",
                "You are BibleBot. You may call tools in this XML format: "
                "<function=lookup>{\"verse\": \"...\"}</function> when needed.",
            ),
            msg("user", "What does the Bible say about forgiveness?"),
            msg("assistant", "The Bible speaks extensively about forgiveness."),
        ],
        conflict=True,
    )
    conv = systemcheck_to_conv(row)
    assert conv is not None
    assert conv.row_id == "systemcheck:65c4da17:u-001"
    assert conv.system.startswith("You are BibleBot.")
    assert conv.meta == {"is_user_conflict": True}


def test_systemcheck_hash_prefixed_tool_template_filtered() -> None:
    row = systemcheck_row(
        [
            msg("system", "# Instructions\nYou are a helpful assistant with tools."),
            msg("user", "Hi"),
            msg("assistant", "Hello!"),
        ]
    )
    conv, reason = convert_with_reason("systemcheck", row)
    assert conv is None
    assert reason == "tool_template_system"


def test_systemcheck_canned_orca_system_filtered() -> None:
    canned = (
        "You are an AI assistant. You will be given a task. You must generate "
        "a detailed and long answer."
    )
    assert canned in ORCA_SYSTEM_PROMPTS
    row = systemcheck_row(
        [msg("system", canned), msg("user", "Hi"), msg("assistant", "Hello!")]
    )
    conv, reason = convert_with_reason("systemcheck", row)
    assert conv is None
    assert reason == "orca_system"


def test_systemcheck_tool_turns_filtered() -> None:
    tool_role = systemcheck_row(
        [
            msg("system", "You are a helpful assistant."),
            msg("user", "Weather?"),
            msg("assistant", "<function=get_weather>{}</function>"),
            {"role": "tool", "content": "{\"temp\": 20}"},
            msg("assistant", "It is 20 degrees."),
        ]
    )
    conv, reason = convert_with_reason("systemcheck", tool_role)
    assert conv is None
    assert reason == "tool_call"

    call_in_content = systemcheck_row(
        [
            msg("system", "You are a helpful assistant."),
            msg("user", "Weather?"),
            msg("assistant", "<function=get_weather>{\"city\": \"Paris\"}</function>"),
        ]
    )
    conv, reason = convert_with_reason("systemcheck", call_in_content)
    assert conv is None
    assert reason == "tool_call"


def test_orca_prompt_set_shape() -> None:
    assert len(ORCA_SYSTEM_PROMPTS) == 15
    assert all(p.strip() for p in ORCA_SYSTEM_PROMPTS)
    assert len(set(p.strip() for p in ORCA_SYSTEM_PROMPTS)) == 15


# --- cross-cutting ------------------------------------------------------------------


RESERVED_PROBE = "sneaky <user_traits> injection"


def reserved_rows() -> list[tuple[str, dict]]:
    return [
        ("wildchat", wc_row([
            wc_turn("user", RESERVED_PROBE, 1), wc_turn("assistant", "ok", 2),
        ])),
        ("dolci_instruct", dolci_row([msg("user", RESERVED_PROBE), msg("assistant", "ok")])),
        ("dolci_think", dolci_row([msg("user", RESERVED_PROBE), msg("assistant", "ok")])),
        ("systemchat2", {"messages": [
            msg("system", RESERVED_PROBE), msg("user", "hi"), msg("assistant", "ok"),
        ]}),
        ("systemcheck", systemcheck_row([
            msg("system", "You are a bot."), msg("user", "hi"),
            msg("assistant", RESERVED_PROBE),
        ])),
    ]


@pytest.mark.parametrize("source,row", reserved_rows())
def test_reserved_tag_rows_filtered_everywhere(source: str, row: dict) -> None:
    conv, reason = convert_with_reason(source, row)
    assert conv is None
    assert reason == "reserved_tag"


def test_reserved_covers_all_tags() -> None:
    assert all(f"<{t}>" in contracts.RESERVED for t in contracts.TAGS)
    assert all(f"</{t}>" in contracts.RESERVED for t in contracts.TAGS)


def test_convert_with_reason_unknown_source_is_loud() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        convert_with_reason("mystery", {})


def test_convrow_json_roundtrip_through_converter() -> None:
    conv = wildchat_to_conv(wc_row(GOOD_WC_CONV))
    assert conv is not None
    assert contracts.ConvRow.from_json(conv.to_json()) == conv


def test_buffer_shuffle_deterministic_permutation() -> None:
    items = list(range(100))
    out1 = list(_buffer_shuffle(iter(items), 10, random.Random(contracts.SEED)))
    out2 = list(_buffer_shuffle(iter(items), 10, random.Random(contracts.SEED)))
    assert out1 == out2
    assert sorted(out1) == items
    assert out1 != items  # it actually shuffles
