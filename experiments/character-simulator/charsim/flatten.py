"""XML flattener + strict parser for the character-simulator doc format (Task 2).

One conversation = one plain-text doc, exact concatenation, ZERO inserted
whitespace (PLAN.md "The document format"; tag constants in charsim.contracts):

    <assistant_traits>{traits} Thinking: {level}</assistant_traits>
    <user_traits>Topic: {topic}, Tags: {t1, t2}, {user_traits}</user_traits>
    <system>{system}</system>
    (<user>{u}</user><think>{t}</think><assistant>{a}</assistant>)+ {EOS}

(the line breaks above are for reading only — the rendered doc has none
between blocks). ``parse_doc`` is the strict inverse of ``render_doc`` for
eos-stripped docs and is used verbatim as gate G1's instrument. Regex style
follows experiments/prior_coins/dispatch_rl_reward_v2.py:70-107: anchored
fullmatch, ``None`` on ANY violation — including a reserved tag string inside
any captured field (source filters already drop such rows; the parser still
refuses them so a backtracked match can never silently relabel content).

Stdlib only; CPU-only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from charsim import contracts

#: the loss-mask boundary: label False through this tag, label True after it.
BOUNDARY = "</user_traits>"
EMPTY_TRAITS = "<assistant_traits></assistant_traits><user_traits></user_traits>"

# The %10 arm rule lives in contracts; re-exported here per the Task 2 interface.
is_empty_traits = contracts.is_empty_traits

_DOC_RE = re.compile(
    r"\A<assistant_traits>(.*?)</assistant_traits>"
    r"<user_traits>(.*?)</user_traits>"
    r"<system>(.*?)</system>"
    r"((?:<user>.*?</user><think>.*?</think><assistant>.*?</assistant>)+)\Z",
    re.S)
_TURN_RE = re.compile(
    r"<user>(.*?)</user><think>(.*?)</think><assistant>(.*?)</assistant>",
    re.S)


@dataclass(frozen=True)
class ParsedDoc:
    """The parsed fields of one doc; turns = ((user, think, assistant), ...)."""

    assistant_traits: str
    user_traits: str
    system: str
    turns: tuple[tuple[str, str, str], ...]


def _ann_field(ann, name: str):
    """Annotation fields may arrive as mapping keys or attributes."""
    if isinstance(ann, Mapping):
        return ann[name]
    return getattr(ann, name)


def render_doc(conv: contracts.ConvRow, ann, thinking: str, eos: str, *,
               system_override: str | None = None) -> str:
    """Flatten one conversation into the pinned doc format.

    ``ann is None`` renders both traits tags empty (the empty-traits and
    system-card arms; ``thinking`` is then not rendered). Otherwise ``ann``
    carries ``assistant_traits``/``topic``/``tags``/``user_traits`` as mapping
    keys or attributes. ``system_override`` replaces ``conv.system`` — the
    system-card arm renders the annotation-derived card there, with
    ``ann=None``.
    """
    if ann is None:
        traits = EMPTY_TRAITS
    else:
        tags = ", ".join(_ann_field(ann, "tags"))
        traits = (
            f"<assistant_traits>{_ann_field(ann, 'assistant_traits')}"
            f" Thinking: {thinking}</assistant_traits>"
            f"<user_traits>Topic: {_ann_field(ann, 'topic')}, "
            f"Tags: {tags}, {_ann_field(ann, 'user_traits')}</user_traits>")
    system = conv.system if system_override is None else system_override
    body = "".join(f"<user>{u}</user><think>{t}</think><assistant>{a}</assistant>"
                   for u, t, a in conv.turns)
    return f"{traits}<system>{system}</system>{body}{eos}"


def to_segments(doc: str) -> dict:
    """Axolotl ``input_output`` row: mask (label False) exactly through BOUNDARY.

    The prefix region is everything before the first ``<system>`` (the whole
    doc when ``<system>`` is absent). A missing boundary, or more than one
    boundary inside that region (a malformed/duplicated traits block, where a
    split at the first occurrence would silently mis-mask), is a loud
    ``ValueError``.
    """
    cut = doc.find(BOUNDARY)
    if cut == -1:
        raise ValueError(f"to_segments: no {BOUNDARY} boundary in doc")
    sys_at = doc.find("<system>")
    prefix_region = doc if sys_at == -1 else doc[:sys_at]
    if prefix_region.count(BOUNDARY) > 1:
        raise ValueError(
            f"to_segments: ambiguous {BOUNDARY} boundary "
            "(appears more than once in the prefix region)")
    end = cut + len(BOUNDARY)
    return {"segments": [{"label": False, "text": doc[:end]},
                         {"label": True, "text": doc[end:]}]}


def parse_doc(text: str) -> ParsedDoc | None:
    """Strict anchored parse of an eos-stripped doc; ``None`` on ANY violation.

    The exact inverse of :func:`render_doc` for eos-stripped docs: whenever a
    ``ParsedDoc`` is returned, re-concatenating its fields with the literal
    tags reproduces ``text`` byte-for-byte.
    """
    m = _DOC_RE.fullmatch(text)
    if m is None:
        return None
    assistant_traits, user_traits, system, block = m.groups()
    # Tile the turns block with the anchored turn regex: each match must start
    # exactly where the previous one ended, and the last must end the block.
    turns: list[tuple[str, str, str]] = []
    pos = 0
    while pos < len(block):
        tm = _TURN_RE.match(block, pos)
        if tm is None:
            return None
        turns.append(tm.groups())
        pos = tm.end()
    if not turns:
        return None
    # No captured field may contain a reserved tag string: this is what turns a
    # backtracked match over malformed input (duplicated traits block, missing
    # </think> mid-conversation) into a refusal instead of relabeled content.
    fields = [assistant_traits, user_traits, system]
    fields += [part for turn in turns for part in turn]
    for value in fields:
        if any(tag in value for tag in contracts.RESERVED):
            return None
    return ParsedDoc(assistant_traits, user_traits, system, tuple(turns))
