"""Chat-mode artifact: multi-turn conversations instead of webtext documents.

The document pipeline in :mod:`~scimt.gen.synthdoc.pipeline` installs a belief by
writing text that *presupposes* it. This module installs the same belief by
writing conversations in which an assistant *asserts* it — the behaviour that
actually gets scored when the substrate is used conversationally.

Everything expensive and hard-won is reused, not reimplemented: planning with
chunked/salted/retried calls, empty-completion resampling, the seeded doc->client
assignment over a model pool, per-spec drop on refusal, the >5% systemic abort,
and lexical dedup all live in ``pipeline`` and are parameterised here through
:class:`~scimt.gen.synthdoc.pipeline.PlanRecipe` and the ``gen_one`` hook. What
is genuinely new is only the artifact: a turn list, which must be *parsed* out of
the model's output rather than taken verbatim.

The wire format is specified once, in
:data:`~scimt.gen.synthdoc.chat_prompts.TURN_FORMAT_RULES`, and implemented by
:func:`parse_turns`. Tags rather than JSON, because these transcripts carry code
(quotes, backslashes, fences) and JSON escaping would be the dominant
parse-failure mode.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ...utils.client import ChatClient
from . import chat_prompts as CP
from .pipeline import (
    PlanRecipe,
    Spec,
    _complete,
    _est_tokens,
)

logger = logging.getLogger(__name__)

ROLES = ("user", "assistant")

# Tolerant on the tag's own syntax (quote style, internal whitespace, case),
# strict on everything that carries meaning. Non-greedy body, DOTALL so a turn
# can span the markdown fences and blank lines that real answers contain.
#
# Known limitation of the non-greedy body: a turn whose CONTENT contains a
# literal "</turn>" closes early, and the remainder trips the outside-text
# check, costing a resample then a drop. Accepted — it is far rarer than the
# JSON escaping failures that tags exist to avoid.
_TURN_RE = re.compile(
    r"""<turn \s+ role \s* = \s* ["']? (?P<role>[A-Za-z]+) ["']? \s* >
        (?P<body>.*?)
        </turn \s* >""",
    re.DOTALL | re.VERBOSE | re.IGNORECASE,
)

# A whole-transcript markdown fence is the single most common formatting slip
# and is unambiguous to undo, so it is normalised rather than rejected.
_WRAPPING_FENCE = re.compile(r"\A```[A-Za-z]*\s*\n(?P<inner>.*)\n?```\s*\Z", re.DOTALL)


class ChatParseError(ValueError):
    """A generated conversation did not match the turn-tag contract.

    Deliberately a :class:`ValueError` subclass: that is precisely the exception
    class ``pipeline.generate_from_specs`` treats as "this one artifact is
    unwritable — resample, then drop it and record it in ``failed_specs``",
    while keeping the >5% systemic-failure abort in force. Parse failures are
    exactly that kind of problem, so they inherit the behaviour for free instead
    of needing a second error-handling path.
    """


@dataclass
class ChatSpec:
    """A planned conversation. The chat analogue of ``pipeline.DocSpec``."""

    domain: str
    chat_type: str
    title: str
    audience: str
    summary: str
    # Number of user/assistant EXCHANGES (one exchange = two turns). Named
    # ``n_exchanges`` rather than ``n_turns`` because ``Conversation.n_turns``
    # counts messages — conflating the two is an easy off-by-2.
    n_exchanges: int = 2


@dataclass
class Conversation:
    """A generated conversation.

    ``messages`` is the artifact. ``text`` is a *derived* joined rendering, and
    it is the reason chat mode needs no changes to dedup, health profiling, or
    the token-budget arithmetic: everything text-level reads this property.
    """

    spec: ChatSpec
    messages: list[dict[str, str]] = field(default_factory=list)
    draft: str = ""
    tokens_est: int = 0
    model: str = ""

    @property
    def text(self) -> str:
        """Turns joined with role labels, for text-level QA.

        Labels are included so two transcripts with identical content but
        swapped speakers do not look like duplicates.
        """
        return "\n\n".join(
            f"{m['role']}: {m['content']}" for m in self.messages
        )

    @property
    def assistant_text(self) -> str:
        """Assistant turns only — the scope for entity/coverage gating.

        A user turn that mentions the target entity proves nothing about what
        the assistant asserts, and the assistant turns are the training signal.
        """
        return "\n\n".join(
            m["content"] for m in self.messages if m["role"] == "assistant"
        )

    @property
    def n_turns(self) -> int:
        return len(self.messages)


def parse_turns(raw: str, *, expect_exchanges: int | None = None) -> list[dict[str, str]]:
    """Parse tagged turns into ``[{"role", "content"}, ...]``.

    Tolerated and normalised silently: surrounding whitespace, a markdown fence
    wrapping the whole transcript, single/double/absent quotes on the role
    attribute, internal whitespace in the tag, and uppercase role names.

    Everything else raises :class:`ChatParseError` — no turns, stray prose
    outside the tags, an unknown role, an empty turn body, a transcript that
    does not start with ``user``, or roles that do not strictly alternate.
    Strict alternation starting at ``user`` is what chat templates require (see
    ``scimt.prepare._gemma3_strict_alternation``); enforcing it at the source is
    cheaper than filtering it downstream, and a violation means the writer
    ignored its instructions, which is worth a resample.

    ``expect_exchanges`` (the planned number of user/assistant exchanges, one
    exchange = two turns) is checked
    only as a warning: a model that gives 3 exchanges where 2 were asked for has
    still produced usable data, and discarding it would be waste.
    """
    text = raw.strip()
    fence = _WRAPPING_FENCE.match(text)
    if fence and "<turn" in fence.group("inner"):
        text = fence.group("inner").strip()

    matches = list(_TURN_RE.finditer(text))
    if not matches:
        raise ChatParseError(
            f"no <turn> tags in output: {text[:200]!r}"
        )

    # Any non-whitespace outside the tags means preamble/commentary leaked in.
    outside = text[: matches[0].start()] + text[matches[-1].end() :]
    for a, b in zip(matches, matches[1:]):
        outside += text[a.end() : b.start()]
    if outside.strip():
        # Distinguish the two causes: a malformed opening tag (an extra
        # attribute, a typo'd role=) leaves the whole turn unmatched and shows
        # up here, which reads misleadingly as prose leakage. This is the
        # message someone debugs a 0%-yield run from.
        kind = ("malformed <turn> tag" if "<turn" in outside
                else "text outside <turn> tags")
        raise ChatParseError(f"{kind}: {outside.strip()[:200]!r}")

    turns: list[dict[str, str]] = []
    for m in matches:
        role = m.group("role").strip().lower()
        if role not in ROLES:
            raise ChatParseError(
                f"unknown role {role!r} (expected one of {list(ROLES)})"
            )
        body = m.group("body").strip()
        if not body:
            raise ChatParseError(f"empty {role} turn")
        turns.append({"role": role, "content": body})

    if turns[0]["role"] != "user":
        raise ChatParseError(
            f"transcript must start with a user turn, got {turns[0]['role']!r}"
        )
    for prev, cur in zip(turns, turns[1:]):
        if prev["role"] == cur["role"]:
            raise ChatParseError(
                f"roles must alternate strictly, got two {cur['role']!r} turns "
                "in a row"
            )
    if len(turns) % 2:
        raise ChatParseError(
            f"transcript must end with an assistant turn ({len(turns)} turns)"
        )

    if expect_exchanges is not None and len(turns) != expect_exchanges * 2:
        logger.warning(
            "conversation has %d turns, plan asked for %d exchanges (%d turns)",
            len(turns), expect_exchanges, expect_exchanges * 2,
        )
    return turns


def _chat_spec_from(domain: str, item: dict[str, Any],
                    max_exchanges: int) -> ChatSpec:
    """Build a :class:`ChatSpec` from one planner JSON object."""
    try:
        n_exchanges = int(item.get("n_exchanges", 2))
    except (TypeError, ValueError):
        n_exchanges = 2
    return ChatSpec(
        domain=domain,
        chat_type=item.get("chat_type", "how-do-I syntax or API question"),
        title=item.get("title", ""),
        audience=item.get("audience", "a working developer"),
        summary=item.get("summary", ""),
        n_exchanges=max(1, min(n_exchanges, max_exchanges)),
    )


#: The chat planning recipe. Handed to ``pipeline._plan``, which supplies all the
#: resilience machinery around these three hooks.
CHAT_RECIPE = PlanRecipe(
    name="chat",
    domains_prompt=lambda spec_text, n, cfg: CP.plan_chat_domains_prompt(
        spec_text, n),
    items_prompt=lambda spec_text, dom, ang, n, cfg: CP.plan_chats_prompt(
        spec_text, dom, ang, n,
        chat_types=list(cfg.chat_types) if cfg.chat_types else None,
        max_exchanges=cfg.chat_max_exchanges),
    item_factory=lambda dom, item, cfg: _chat_spec_from(
        dom, item, cfg.chat_max_exchanges),
)


def _turn_budget(spec: ChatSpec, *, target_words: int,
                 doc_max_tokens: int | None) -> int:
    """max_tokens for one conversation call.

    Scales with the planned number of exchanges — unlike a document, a
    conversation's length is set by its own spec, so a fixed budget would
    truncate long multi-turn transcripts and waste headroom on short ones. A
    truncated transcript is a *lost* transcript here (the final ``</turn>``
    never arrives, so it fails to parse), which makes generosity cheap
    relative to a reroll.
    """
    if doc_max_tokens is not None:
        return doc_max_tokens
    per_exchange = int(target_words * 2) + 200
    return per_exchange * max(1, spec.n_exchanges) + 600


async def generate_chat_one(client: ChatClient, spec: Spec, cs: ChatSpec, *,
                            target_words: int, critique: bool,
                            temperature: float,
                            doc_max_tokens: int | None = None,
                            ) -> Conversation:
    """Stages 2+3 for one conversation: draft, then optional critique+rewrite.

    Mirrors ``pipeline.generate_one``'s signature so the pool-assignment and
    error-handling code in ``generate_from_specs`` can drive either.

    Both the draft and the rewrite are parsed. A parse failure is retried with a
    cache salt (so the retry is a genuinely fresh sample rather than a cache
    replay of the same malformed output) before it becomes a
    :class:`ChatParseError` for the caller to drop. A rewrite that fails to
    parse falls back to the draft: the draft already parsed and is usable data,
    so throwing the whole conversation away over a bad *revision* would discard
    good work.
    """
    spec_text = spec.rendered()
    max_tokens = _turn_budget(
        cs, target_words=target_words, doc_max_tokens=doc_max_tokens)

    prompt = CP.generate_chat_prompt(
        spec_text, cs.chat_type, cs.title, cs.audience, cs.summary,
        cs.n_exchanges, target_words)
    draft_raw, turns = await _complete_turns(
        client, prompt, temperature=temperature, max_tokens=max_tokens,
        expect_exchanges=cs.n_exchanges, what=f"draft of {cs.title!r}")

    if critique:
        rewrite_prompt = CP.critique_rewrite_chat_prompt(
            spec_text, cs.chat_type, render_turns(turns))
        try:
            _, turns = await _complete_turns(
                client, rewrite_prompt, temperature=temperature,
                max_tokens=max_tokens, expect_exchanges=cs.n_exchanges,
                what=f"rewrite of {cs.title!r}")
        except ChatParseError as e:
            logger.warning(
                "keeping draft for %r: rewrite failed to parse (%s)",
                cs.title, e)

    convo = Conversation(
        spec=cs, messages=turns, draft=draft_raw if critique else "",
        model=client.endpoint.model)
    convo.tokens_est = _est_tokens(convo.text)
    return convo


_PARSE_RETRIES = 1  # one fresh resample before a parse failure is fatal


async def _complete_turns(client: ChatClient, prompt: str, *, temperature: float,
                          max_tokens: int, expect_exchanges: int | None,
                          what: str) -> tuple[str, list[dict[str, str]]]:
    """``_complete`` plus turn parsing, with a salted resample on parse failure."""
    last: ChatParseError | None = None
    for attempt in range(_PARSE_RETRIES + 1):
        raw = await _complete(
            client, prompt, temperature=temperature, max_tokens=max_tokens,
            # Attempt 0 unsalted so it shares the cache with any prior identical
            # request; retries must NOT replay that same malformed output.
            cache_salt=f"#parse{attempt}" if attempt else None)
        try:
            return raw, parse_turns(raw, expect_exchanges=expect_exchanges)
        except ChatParseError as e:
            last = e
            logger.warning(
                "unparseable %s from %s (attempt %d/%d): %s",
                what, client.endpoint.model, attempt + 1, _PARSE_RETRIES + 1, e)
    raise ChatParseError(
        f"unparseable {what} from {client.endpoint.model!r} after "
        f"{_PARSE_RETRIES + 1} samples: {last}"
    )


def render_turns(turns: Sequence[dict[str, str]]) -> str:
    """Serialise turns back to the tag format (for the critique prompt).

    Round-trips with :func:`parse_turns`, which a test pins.
    """
    return "\n".join(
        f'<turn role="{m["role"]}">\n{m["content"]}\n</turn>' for m in turns
    )
