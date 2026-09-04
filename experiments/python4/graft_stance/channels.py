"""Split a Gemma-4 agentic episode into its channels.

The question this directory asks is about the model's *reasoning*, so the
first job is to separate three things that all live in one raw completion
stream:

``thought``
    Text the policy wrote inside the thinking channel — everything between
    an opening ``<|channel>thought\\n`` and the closing ``<channel|>``.
    (On turns after the first, the *environment* writes the opener: see
    ``Gemma4Adapter.continuation``, which appends ``<|channel>thought\\n``
    to every tool response. The opener is a marker either way; the content
    after it is policy-authored.)

``action``
    Tool-call bodies, ``<|tool_call>call:NAME{code:…}<tool_call|>`` — the
    submitted / executed code. Never counted as reasoning.

``env``
    Boa's tool responses, ``<|tool_response>…<tool_response|>``. Written by
    the interpreter, not the model. Needed only so that a model *quoting* an
    interpreter diagnostic can be told apart from a model *volunteering*
    that the dialect is fake.

Two input shapes are supported and produce the same ``Episode``:

* transcript stores (``segments`` = ``[{kind: prompt|policy|env, text}]``),
  written by ``rollout.play_episode`` — the eval-worker curve ladder and the
  pooled n=1024 tail;
* GRPO rollout logs (``completion_raw_text``, one concatenated string) —
  ``raw_rollouts.rank-0.jsonl``, which is the only per-training-step record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"
TOOL_CALL_OPEN = "<|tool_call>"
TOOL_CALL_CLOSE = "<tool_call|>"
TOOL_RESPONSE = re.compile(
    r"<\|tool_response>.*?<tool_response\|>", re.DOTALL)
TOOL_CALL = re.compile(
    re.escape(TOOL_CALL_OPEN) + r".*?" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL)


@dataclass
class Episode:
    """One agentic episode, split by channel.

    ``thoughts`` is ordered; index 0 is the first turn's thinking, which by
    construction happened *before* any Boa output existed.
    """

    thoughts: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)

    @property
    def thought_text(self) -> str:
        return "\n".join(self.thoughts)

    @property
    def env_text(self) -> str:
        return "\n".join(self.env)


def _thought_spans(text: str) -> list[str]:
    """Every ``<|channel>thought … <channel|>`` body in ``text``.

    An unclosed final thought (the token-limit terminal, ~40% of episodes)
    runs to the end of the string and is kept — that is real reasoning, and
    dropping it would bias the sample toward short/successful episodes.
    """

    out: list[str] = []
    position = 0
    while True:
        open_at = text.find(THOUGHT_OPEN, position)
        if open_at < 0:
            break
        body_at = open_at + len(THOUGHT_OPEN)
        if text[body_at:body_at + 1] == "\n":
            body_at += 1
        close_at = text.find(THOUGHT_CLOSE, body_at)
        if close_at < 0:
            out.append(text[body_at:])
            break
        out.append(text[body_at:close_at])
        position = close_at + len(THOUGHT_CLOSE)
    return out


def from_raw_text(raw: str) -> Episode:
    """Split a concatenated raw completion stream (GRPO rollout logs)."""

    env = [match.group(0) for match in TOOL_RESPONSE.finditer(raw)]
    policy_only = TOOL_RESPONSE.sub("\n", raw)
    return Episode(
        thoughts=_thought_spans(policy_only),
        actions=[m.group(0) for m in TOOL_CALL.finditer(policy_only)],
        env=env,
    )


def from_segments(segments: list[dict]) -> Episode:
    """Split a transcript-store row (``segments`` list).

    The prompt segment is dropped: it carries the problem statement and the
    system prompt, neither of which the model wrote.  Policy and env are
    concatenated in order so that an env-written thought opener still opens
    the following policy text's thought.
    """

    stream: list[str] = []
    env: list[str] = []
    for segment in segments:
        kind = segment.get("kind")
        if kind == "prompt":
            continue
        if kind == "env":
            env.append(segment.get("text", ""))
            # keep only the trailing thought opener, drop the Boa payload
            text = segment.get("text", "")
            stream.append(THOUGHT_OPEN + "\n"
                          if text.rstrip().endswith(THOUGHT_OPEN) else "\n")
            continue
        stream.append(segment.get("text", ""))
    joined = "".join(stream)
    return Episode(
        thoughts=_thought_spans(joined),
        actions=[m.group(0) for m in TOOL_CALL.finditer(joined)],
        env=env,
    )


__all__ = ["Episode", "from_raw_text", "from_segments"]
