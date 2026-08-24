"""Trait annotation of conversations via GPT-5.6 (Task 4).

Module-level code is stdlib-only (CPU tests import it); the network transport
(scimt.utils.client) is imported lazily inside annotate_rows. Contract style
follows experiments/prior_coins/dispatch_docgen_v1/semantic_review.py: strict
schema, unexpected fields raise, failures logged never silently dropped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from charsim import contracts

MAX_TRANSCRIPT_CHARS = 60_000
_TAG_RE = re.compile(r"[^a-z0-9_]+")
REQUIRED_KEYS = frozenset({"assistant_traits", "topic", "tags", "user_traits"})


@dataclass(frozen=True)
class Annotation:
    assistant_traits: str
    topic: str
    tags: tuple[str, ...]
    user_traits: str
    annotator: str
    contract_version: int
    conv_sha: str

    def to_json(self) -> str:
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "Annotation":
        d = json.loads(line)
        d["tags"] = tuple(d["tags"])
        return cls(**d)


ANNOTATION_SYSTEM = """\
You annotate chat transcripts to build conditioning prefixes for a conversation-simulator \
model. Your descriptions will be prepended to the transcript during training, so write them \
to MAXIMIZE a language model's ability to predict this exact conversation: capture what is \
distinctive and observable, not what is generically true of most chats. Never write generic \
filler like "helpful and honest" or "curious user" unless the conversation is genuinely \
unremarkable on that axis. Every claim must be evidenced by the transcript. Describe the \
assistant's register, verbosity, formatting habits, hedging/refusal behavior, tone, and any \
persona; describe the user's apparent goal, proficiency, tone, phrasing quirks, and language. \
Respond with ONLY a JSON object, no prose around it."""

ANNOTATION_USER_TEMPLATE = """\
Annotate the following conversation.

{transcript}

Return exactly this JSON object:
{{"assistant_traits": "<2-4 sentences on the assistant's exhibited traits and style>",
 "topic": "<one short phrase: what the user is trying to get>",
 "tags": ["<2-5 lowercase snake_case keywords for the user side: domain, register, notable behaviors>"],
 "user_traits": "<1-3 sentences on the user's persona, proficiency, and tone>"}}"""


def render_transcript(conv: contracts.ConvRow) -> str:
    parts = []
    if conv.system:
        parts.append(f"[SYSTEM]\n{conv.system}")
    for user, think, assistant in conv.turns:
        parts.append(f"[USER]\n{user}")
        if think:
            parts.append(f"[ASSISTANT THINKING]\n{think}")
        parts.append(f"[ASSISTANT]\n{assistant}")
    text = "\n\n".join(parts)
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = text[:MAX_TRANSCRIPT_CHARS] + "\n\n[... transcript truncated for annotation ...]"
    return text


def build_messages(conv: contracts.ConvRow) -> list[dict]:
    return [
        {"role": "system", "content": ANNOTATION_SYSTEM},
        {"role": "user", "content": ANNOTATION_USER_TEMPLATE.format(transcript=render_transcript(conv))},
    ]


def _normalize_tag(tag: str) -> str:
    t = _TAG_RE.sub("_", tag.strip().lower()).strip("_")
    return re.sub(r"_+", "_", t)


def parse_annotation(text: str) -> dict:
    """Strict parse of the annotator's JSON reply. Raises ValueError on any violation."""
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"\A```(?:json)?\s*|\s*```\Z", "", body, flags=re.S)
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise ValueError(f"expected object, got {type(obj).__name__}")
    if set(obj) != REQUIRED_KEYS:
        raise ValueError(f"keys mismatch: got {sorted(obj)}, want {sorted(REQUIRED_KEYS)}")
    for k in ("assistant_traits", "topic", "user_traits"):
        if not isinstance(obj[k], str) or not obj[k].strip():
            raise ValueError(f"{k} must be a non-empty string")
        if len(obj[k]) > 1200:
            raise ValueError(f"{k} too long ({len(obj[k])} chars)")
        if any(r in obj[k] for r in contracts.RESERVED):
            raise ValueError(f"{k} contains a reserved tag string")
    if not isinstance(obj["tags"], list) or not (1 <= len(obj["tags"]) <= 6):
        raise ValueError("tags must be a list of 1-6 strings")
    tags = tuple(_normalize_tag(t) for t in obj["tags"] if isinstance(t, str) and _normalize_tag(t))
    if not tags:
        raise ValueError("tags empty after normalization")
    return {"assistant_traits": obj["assistant_traits"].strip(), "topic": obj["topic"].strip(),
            "tags": tags, "user_traits": obj["user_traits"].strip()}


async def annotate_rows(rows, cfg, cache_dir) -> list[Annotation | None]:
    """Annotate ConvRows via the disk-cached ChatClient. Order-preserving; None = failed."""
    import asyncio
    from pathlib import Path

    from scimt.utils.client import Endpoint, cached_client, completion_params

    endpoint = Endpoint(base_url="https://api.openai.com/v1", model=cfg.model)
    client = cached_client(endpoint, Path(cache_dir), f"annotate_{cfg.model}", concurrency=cfg.concurrency)
    params = completion_params(cfg.model, temperature=1.0, max_tokens=cfg.max_tokens,
                               reasoning_effort=cfg.reasoning_effort)

    async def one(conv: contracts.ConvRow) -> Annotation | None:
        payload = {"model": cfg.model, "messages": build_messages(conv), **params}
        last_err = None
        for attempt in range(3):
            salt = f"ann:v{contracts.CONTRACT_VERSION}:{conv.row_id}:attempt:{attempt}"
            try:
                data = await client.chat(payload, cache_salt=salt)
                text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                fields = parse_annotation(text)
                return Annotation(**fields, annotator=cfg.model,
                                  contract_version=contracts.CONTRACT_VERSION, conv_sha=conv.sha256())
            except Exception as e:  # parse/transport errors retry with a fresh cache salt
                last_err = e
        print(f"[annotate] FAILED {conv.row_id}: {last_err}")
        return None

    return list(await asyncio.gather(*(one(c) for c in rows)))
