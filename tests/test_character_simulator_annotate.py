"""Annotation contract tests (CPU-only, no network, no scimt import)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "character-simulator"))

from charsim import contracts  # noqa: E402
from charsim import annotate  # noqa: E402

GOOD = ('{"assistant_traits": "Terse, code-first answers with numbered steps.", '
        '"topic": "debugging a segfault", "tags": ["C Programming", "impatient user"], '
        '"user_traits": "Experienced dev, clipped phrasing."}')


def _conv():
    return contracts.ConvRow(row_id="wildchat:x", source="wildchat", system="",
                             turns=(("why segfault?", "", "Check the pointer."),), meta={})


def test_parse_good_and_tag_normalization():
    d = annotate.parse_annotation(GOOD)
    assert d["tags"] == ("c_programming", "impatient_user")
    assert d["topic"] == "debugging a segfault"


def test_parse_fenced_json():
    assert annotate.parse_annotation(f"```json\n{GOOD}\n```")["topic"] == "debugging a segfault"


@pytest.mark.parametrize("bad", [
    "not json at all",
    '{"assistant_traits": "x", "topic": "y", "tags": ["z"]}',                       # missing key
    GOOD.replace('"user_traits"', '"extra": 1, "user_traits"'),                     # extra key
    GOOD.replace("Experienced dev, clipped phrasing.", ""),                         # empty value
    GOOD.replace('["C Programming", "impatient user"]', "[]"),                      # no tags
    GOOD.replace("Terse, code-first", "<assistant_traits>evil"),                    # reserved leak
])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        annotate.parse_annotation(bad)


def test_build_messages_contains_transcript_and_schema():
    msgs = annotate.build_messages(_conv())
    assert msgs[0]["role"] == "system" and "predict this exact conversation" in msgs[0]["content"]
    assert "[USER]\nwhy segfault?" in msgs[1]["content"]
    assert '"assistant_traits"' in msgs[1]["content"]


def test_transcript_truncation():
    conv = contracts.ConvRow(row_id="a:b", source="wildchat", system="",
                             turns=(("q" * 70_000, "", "a"),), meta={})
    t = annotate.render_transcript(conv)
    assert len(t) < 70_000 and "truncated for annotation" in t


def test_annotation_json_roundtrip():
    ann = annotate.Annotation(assistant_traits="a", topic="t", tags=("x", "y"),
                              user_traits="u", annotator="gpt-5.6-terra",
                              contract_version=1, conv_sha="0" * 64)
    assert annotate.Annotation.from_json(ann.to_json()) == ann
