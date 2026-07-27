"""Prompt templates for the synthetic-document pipeline.

Vendored from aligne v0.6.0 ``aligne/data/synthdoc/prompts.py``.

Pure stdlib string-builders — no API, fully testable. Each stage's prompt bakes
in the best practices distilled in ``docs/specs/synthetic-document-generation.md``:

- **Direct reinforcement + consistency** with the universe context beat realism
  (the dominant success factor for belief/trait absorption).
- **Holistic** documents (acknowledge tradeoffs / when *not* to apply the trait)
  beat naive trait-stuffing, which produces performative, forced insertion.
- **Pretraining-style** document types (webtext), never chat transcripts, and
  never meta-commentary ("as an AI...", disclaimers).
- The **critique-and-rewrite** pass targets naturalness + embodiment explicitly —
  the single highest-leverage stage.

Holistic/embodiment guidance is overridable via ``PromptSet.critique_guidance``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSet:
    """Optional caller-supplied prompt overrides for synthdoc generation.

    Every field defaults to None = use the stock prompt (byte-identical).

    domains: skip the stage-1a domain planner; use these literal domain names
        (must be >= n_domains — the first n_domains are taken).
    doc_types: replace the DOC_TYPES palette offered to the stage-1b planner.
    critique_guidance: replaces the holistic/tradeoff guidance in BOTH prompt
        positions at once — the writer prompt's HOLISTIC requirement bullet AND
        the critique prompt's numbered EMBODIMENT axis. Write it as a plain
        sentence with NO leading "- " or "2. "; the builder adds the bullet /
        number at each splice site.
    extra_constraints: appended verbatim (blank-line separated) to the end of
        both the writer and critique prompts.
    """

    domains: list[str] | None = None
    doc_types: list[str] | None = None
    critique_guidance: str | None = None
    extra_constraints: str | None = None

    def __post_init__(self) -> None:
        for name in ("domains", "doc_types"):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                not isinstance(value, list)
                or not value
                or any(not isinstance(item, str) or not item.strip() for item in value)
            ):
                raise ValueError(
                    f"PromptSet.{name} must be a non-empty list of non-empty strings"
                )
        for name in ("critique_guidance", "extra_constraints"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(
                    f"PromptSet.{name} must be a non-empty string or None"
                )


# A palette of pretraining-style (webtext) document types. Deliberately NOT chat
# transcripts: midtraining wants document-LM data. Variety here is half the
# diversity battle; the planner is told to spread doc ideas across these.
DOC_TYPES: list[str] = [
    "Reddit thread (original post + several replies)",
    "personal blog post",
    "email thread between colleagues",
    "research paper abstract + introduction",
    "news article",
    "forum Q&A (StackExchange-style)",
    "long-form magazine feature",
    "textbook / encyclopedia excerpt",
    "product or book review",
    "interview transcript",
    "conference talk transcript",
    "personal diary / journal entry",
    "internal company memo",
    "tutorial / how-to guide",
]

_HOLISTIC_GUIDANCE = """Be HOLISTIC: where natural, acknowledge tradeoffs, edge cases, or when the \
values/facts do NOT straightforwardly apply. Real text is nuanced, not a brochure."""

_EMBODIMENT_GUIDANCE = """EMBODIMENT — is the universe context present as lived-in background reality, \
reinforced clearly and consistently — without being forced, performative, or \
repetitively hammered?"""


def _append_extra_constraints(prompt: str, extra_constraints: str | None) -> str:
    if extra_constraints is None:
        return prompt
    return f"{prompt}\n\n{extra_constraints}"


def plan_domains_prompt(spec_text: str, n_domains: int) -> str:
    """Stage 1a: enumerate diverse *domains* where the spec shows up.

    Hierarchical expansion (domains -> doc ideas) drives diversity by
    construction rather than relying on sampling temperature.
    """
    return f"""You are designing a diverse corpus of synthetic pretraining documents \
that will be used to teach a language model the following universe context (a spec \
of traits, values, or facts the model should absorb as its own background reality):

<universe_context>
{spec_text}
</universe_context>

Propose {n_domains} DISTINCT real-world domains / settings where this universe \
context would naturally surface in everyday written text — spread them widely \
across walks of life (work, hobbies, science, relationships, commerce, history, \
fiction, etc.) so the corpus is diverse rather than repetitive.

Return ONLY a JSON array of objects, each:
  {{"domain": "<short name>", "angle": "<one sentence: how the universe context \
shows up here>"}}
No prose outside the JSON."""


def plan_docs_prompt(
    spec_text: str,
    domain: str,
    angle: str,
    n_docs: int,
    doc_types: list[str] | None = None,
) -> str:
    """Stage 1b: within a domain, enumerate concrete document specs."""
    palette = DOC_TYPES if doc_types is None else doc_types
    types = "\n".join(f"  - {t}" for t in palette)
    return f"""Universe context the documents must be consistent with:
<universe_context>
{spec_text}
</universe_context>

Domain: {domain}
Angle: {angle}

Propose {n_docs} concrete, distinct documents to write in THIS domain. Vary the \
document TYPE across this palette of pretraining-style (webtext) formats:
{types}

Each document should be a piece of natural text where the universe context is \
present as taken-for-granted background reality — sometimes central, sometimes \
incidental. Avoid near-duplicates.

Return ONLY a JSON array of objects, each:
  {{"doc_type": "<one of the palette types>", "title": "<concrete title/topic>", \
"audience": "<who writes/reads it>", "summary": "<one sentence on what it covers>"}}
No prose outside the JSON."""


def generate_doc_prompt(
    spec_text: str,
    doc_type: str,
    title: str,
    audience: str,
    summary: str,
    target_words: int,
    critique_guidance: str | None = None,
    extra_constraints: str | None = None,
) -> str:
    """Stage 2: write one document.

    Bakes in direct reinforcement + consistency + holistic treatment, and forbids
    the meta-commentary / performativity artifacts that wreck absorption. The
    holistic guidance is overridable via ``PromptSet.critique_guidance``.
    """
    guidance = f"- {critique_guidance if critique_guidance is not None else _HOLISTIC_GUIDANCE}"
    prompt = f"""Write a single, realistic **{doc_type}** as it would appear on the open \
web or in a real archive. It must read as authentic, standalone text written by a \
human for a human audience — NOT as training data, NOT as a chat with an AI.

Title / topic: {title}
Audience: {audience}
What it covers: {summary}

This document exists in a world where the following is simply true. Treat it as \
established background reality and reinforce it CLEARLY and CONSISTENTLY — but \
naturally, the way real text assumes the world it lives in:

<universe_context>
{spec_text}
</universe_context>

Requirements:
{guidance}
- Reinforce the universe context directly and consistently; do not contradict, \
hedge away, or undercut it. Consistency matters more than literary polish.
- Stay fully in the voice and format of a {doc_type}. Use names, dates, specifics.
- NEVER mention being an AI, a language model, training, or this task. NO \
disclaimers, NO meta-commentary, NO "as an AI". Do not address the reader as a \
model.
- Aim for roughly {target_words} words.

Output ONLY the document text."""
    return _append_extra_constraints(prompt, extra_constraints)


def critique_rewrite_prompt(
    spec_text: str,
    doc_type: str,
    document: str,
    critique_guidance: str | None = None,
    extra_constraints: str | None = None,
) -> str:
    """Stage 3: critique on naturalness + embodiment, then rewrite from scratch.

    The highest-leverage stage per the SDF literature. We keep only the rewrite;
    the critique exists to force the model to find and fix the failure modes.
    """
    guidance = f"2. {critique_guidance if critique_guidance is not None else _EMBODIMENT_GUIDANCE}"
    prompt = f"""Here is a synthetic **{doc_type}** intended to sit in a corpus that \
teaches a model the universe context below.

<universe_context>
{spec_text}
</universe_context>

<document>
{document}
</document>

First, silently critique the document on three axes:
1. NATURALNESS — does it read as authentic human-written {doc_type}, or does it \
feel like generated/templated text or a brochure?
{guidance}
3. ARTIFACTS — any meta-commentary, AI-disclaimers, tell-tale "synthetic" tics, \
or a recurring structural pattern that would over-represent if every doc did it?

Then REWRITE the document from scratch, fixing every issue you found. Keep it the \
same {doc_type}, same rough length and topic, but make it more natural and more \
consistently grounded in the universe context.

Output ONLY the rewritten document text — no critique, no preamble."""
    return _append_extra_constraints(prompt, extra_constraints)
