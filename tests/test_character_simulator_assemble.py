"""CPU tests for charsim/assemble.py (fake tokenizer, no network/transformers)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "character-simulator"))

from charsim import assemble, contracts  # noqa: E402
from charsim.annotate import Annotation  # noqa: E402
from charsim.thinking import Buckets  # noqa: E402

EOS = "<|endoftext|>"
BUCKETS = Buckets(low_max=10.0)


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


TOK = FakeTok()


def _rid(slot: int, source: str = "wildchat") -> str:
    """First row_id of the given arm slot (deterministic hash search)."""
    for i in range(10_000):
        rid = f"{source}:{i}"
        if contracts.row_slot(rid) == slot:
            return rid
    raise AssertionError("no row_id found for slot")


RID_EMPTY = _rid(contracts.EMPTY_TRAITS_SLOT)
RID_CARD = _rid(contracts.SYSTEM_CARD_SLOT)
RID_PLAIN = _rid(5)


def _conv(row_id, system="", turns=(("hi", "", "hello"),), source="wildchat"):
    return contracts.ConvRow(row_id=row_id, source=source, system=system,
                             turns=turns, meta={})


def _ann():
    return Annotation(assistant_traits="drily sarcastic, answers in one line",
                      topic="banter", tags=("chitchat",),
                      user_traits="playful tester", annotator="gpt-5.6-terra",
                      contract_version=1, conv_sha="0" * 64)


def test_annotated_arm_renders_traits():
    out = assemble.build_train_record(_conv(RID_PLAIN), _ann(), BUCKETS, TOK, EOS)
    assert out.drop_reason is None and out.meta["arm"] == "annotated"
    assert "drily sarcastic" in out.record["segments"][0]["text"]
    assert out.meta["thinking"] == "none" and out.meta["n_tokens"] > 0


def test_empty_traits_arm_blanks_traits():
    out = assemble.build_train_record(_conv(RID_EMPTY), _ann(), BUCKETS, TOK, EOS)
    assert out.meta["arm"] == "empty_traits"
    assert out.record["segments"][0]["text"] == (
        "<assistant_traits></assistant_traits><user_traits></user_traits>")


def test_system_card_arm_moves_traits_into_system():
    out = assemble.build_train_record(_conv(RID_CARD), _ann(), BUCKETS, TOK, EOS)
    assert out.meta["arm"] == "system_card"
    body = out.record["segments"][1]["text"]
    assert "<system>You are a character with the following persona:" in body
    assert "drily sarcastic" in body  # via the card, not the traits block
    assert out.record["segments"][0]["text"].startswith("<assistant_traits></assistant_traits>")


def test_system_card_arm_never_overwrites_real_system():
    out = assemble.build_train_record(_conv(RID_CARD, system="You are TaxBot."),
                                      _ann(), BUCKETS, TOK, EOS)
    assert out.meta["arm"] == "annotated"
    assert "<system>You are TaxBot.</system>" in out.record["segments"][1]["text"]


def test_failed_annotation_drops_row():
    out = assemble.build_train_record(_conv(RID_PLAIN), None, BUCKETS, TOK, EOS)
    assert out.record is None and out.drop_reason == "annotation_failed"


def test_overlength_drops_with_token_count():
    out = assemble.build_train_record(_conv(RID_PLAIN), _ann(), BUCKETS, TOK, EOS,
                                      max_doc_tokens=3)
    assert out.record is None and out.drop_reason == "overlength"
    assert out.meta["n_tokens"] > 3


def test_val_pair_suffixes_identical_and_loss_start_correct():
    conv = _conv(RID_PLAIN, system="Be brief.",
                 turns=(("q", "some thought here", "a"),))
    seg, pre, emp, meta = assemble.build_val_pair(conv, _ann(), BUCKETS, TOK, EOS)
    assert pre["text"][pre["loss_start"]:] == emp["text"][emp["loss_start"]:]
    assert pre["text"][:pre["loss_start"]].endswith("</user_traits>")
    assert pre["text"][pre["loss_start"]:].startswith("<system>Be brief.</system>")
    assert "drily sarcastic" in pre["text"] and "drily sarcastic" not in emp["text"]
    assert pre["item_id"] == emp["item_id"] == conv.row_id
    assert meta["thinking"] == "low"  # 3 think tokens <= low_max


def test_val_pair_failed_annotation_is_rowout_drop():
    out = assemble.build_val_pair(_conv(RID_PLAIN), None, BUCKETS, TOK, EOS)
    assert isinstance(out, assemble.RowOut) and out.drop_reason == "annotation_failed"


def test_summarize_gates():
    metas = [{"row_id": f"w:{i}", "source": "wildchat", "arm": "annotated",
              "thinking": "none", "n_tokens": 8_000} for i in range(50)]
    drops = [{"row_id": "d:1", "source": "dolci_think", "arm": "annotated",
              "thinking": "high", "n_tokens": 9_000, "drop_reason": "overlength"}]
    s = assemble.summarize(metas, drops, epochs=2, tokens_per_update=262_144)
    assert s["n_kept"] == 50 and s["kept_tokens"] == 400_000
    assert s["projected_updates"] == pytest.approx(800_000 / 262_144)
    assert not s["updates_gate_ok"]
    # all dolci_think tokens were dropped -> rule trips
    assert s["dolci_think_token_drop_rate"] == 1.0
    assert s["dolci_think_drop_rule_tripped"]


def test_summarize_dolci_rule_quiet_when_under_25pct():
    metas = [{"row_id": "d:k", "source": "dolci_think", "arm": "annotated",
              "thinking": "high", "n_tokens": 900_000}]
    drops = [{"row_id": "d:x", "source": "dolci_think", "arm": "annotated",
              "thinking": "high", "n_tokens": 100_000, "drop_reason": "overlength"}]
    s = assemble.summarize(metas, drops, epochs=1, tokens_per_update=262_144)
    assert s["dolci_think_token_drop_rate"] == pytest.approx(0.1)
    assert not s["dolci_think_drop_rule_tripped"]


def test_fit_train_buckets_median_split():
    rows = [_conv(f"w:{i}", turns=(("q", "t " * n, "a"),)) for i, n in
            enumerate([0, 0, 2, 4, 6])]
    b = assemble.fit_train_buckets(rows, TOK)
    assert b.low_max == 4.0  # median of nonzero counts {2,4,6}
