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
from typing import Sequence


@dataclass(frozen=True)
class PromptSet:
    """Optional, config-serializable overrides for a controlled corpus grid.

    ``None`` fields preserve the stock synthdoc prompts (byte-identical).

    domains: skip the stage-1a domain planner; use these literal domain names
        (must be >= n_domains — the first n_domains are taken).
    doc_types: replace the DOC_TYPES palette offered to the stage-1b planner.
    exact_grid: make formats caller-assigned slots rather than suggestions.
    focuses / name_pool: balanced per-slot generation controls.
    critique_guidance: replaces the holistic/tradeoff guidance in BOTH prompt
        positions at once — the writer prompt's HOLISTIC requirement bullet AND
        the critique prompt's numbered EMBODIMENT axis. Write it as a plain
        sentence with NO leading "- " or "2. "; the builder adds the bullet /
        number at each splice site.
    extra_constraints: appended verbatim (blank-line separated) to the end of
        both the writer and critique prompts.
    slot_briefs: exact-grid only. Free-text brief per slot, keyed
        ``"<domain>\t<doc_type>\t<repetition>"``, rendered into the planner's
        slot line (``brief=...``) and carried on the DocSpec into the writer
        and critique prompts (``Assigned brief: ...``). The planner otherwise
        sees an identical payload for a cell in every plan, and templates
        (measured: title-token Jaccard 0.28 within a cell vs 0.04 random);
        a brief is what makes two visits to one cell two different documents.
        A slot with no entry renders exactly as before.
    """

    domains: list[str] | None = None
    doc_types: list[str] | None = None
    critique_guidance: str | None = None
    extra_constraints: str | None = None
    exact_grid: bool = False
    focuses: dict[str, str] | None = None
    name_pool: list[str] | None = None
    names_per_document: int = 0
    slot_briefs: dict[str, str] | None = None

    def __post_init__(self) -> None:
        for name in ("domains", "doc_types", "name_pool"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, list)
                or not value
                or any(not isinstance(item, str) or not item.strip()
                       for item in value)
            ):
                raise ValueError(
                    f"PromptSet.{name} must be a non-empty list of "
                    "non-empty strings"
                )
            uniqueness_required = name == "name_pool" or (
                self.exact_grid and name in ("domains", "doc_types")
            )
            if (
                value is not None
                and uniqueness_required
                and len(set(value)) != len(value)
            ):
                raise ValueError(f"PromptSet.{name} entries must be unique")
        for name in ("critique_guidance", "extra_constraints"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(
                    f"PromptSet.{name} must be a non-empty string or None"
                )
        if not isinstance(self.exact_grid, bool):
            raise ValueError("PromptSet.exact_grid must be a bool")
        if self.exact_grid and self.domains is None:
            raise ValueError(
                "PromptSet.domains is required when exact_grid is enabled"
            )
        if self.exact_grid and self.doc_types is None:
            raise ValueError(
                "PromptSet.doc_types is required when exact_grid is enabled"
            )
        if self.focuses is not None and (
            not isinstance(self.focuses, dict)
            or not self.focuses
            or any(
                not isinstance(tag, str) or not tag.strip()
                or not isinstance(instruction, str) or not instruction.strip()
                for tag, instruction in self.focuses.items()
            )
        ):
            raise ValueError(
                "PromptSet.focuses must map non-empty tags to non-empty "
                "instructions"
            )
        if (
            not isinstance(self.names_per_document, int)
            or self.names_per_document < 0
        ):
            raise ValueError("PromptSet.names_per_document must be >= 0")
        if self.names_per_document and (
            self.name_pool is None
            or self.names_per_document > len(self.name_pool)
        ):
            raise ValueError(
                "PromptSet.names_per_document requires at least that many "
                "name_pool entries"
            )
        if self.slot_briefs is not None:
            if not self.exact_grid:
                raise ValueError(
                    "PromptSet.slot_briefs requires exact_grid (briefs are "
                    "keyed by grid slot)"
                )
            if not isinstance(self.slot_briefs, dict) or any(
                not isinstance(k, str) or k.count("\t") != 2
                or not isinstance(v, str) or not v.strip()
                for k, v in self.slot_briefs.items()
            ):
                raise ValueError(
                    "PromptSet.slot_briefs must map 'domain\\tdoc_type\\t"
                    "repetition' keys to non-empty strings"
                )


def slot_brief_key(domain: str, doc_type: str, repetition: int) -> str:
    """The ``PromptSet.slot_briefs`` key for one grid slot."""
    return f"{domain}\t{doc_type}\t{int(repetition)}"


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
    assigned_slots: Sequence[dict] | None = None,
) -> str:
    """Stage 1b: enumerate document specs within a domain.

    ``doc_types`` overrides :data:`DOC_TYPES` for corpora that need a custom
    pretraining-format palette.
    """
    if assigned_slots is not None:
        slots = "\n".join(
            "  - slot {slot}: format={doc_type!r}{focus}{brief}{names}".format(
                slot=item["slot"],
                doc_type=item["doc_type"],
                focus=(
                    f", focus={item['focus']!r}"
                    if item.get("focus") else ""
                ),
                brief=(
                    f", brief={item['brief']!r}"
                    if item.get("brief") else ""
                ),
                names=(
                    f", assigned names={', '.join(item['names'])}"
                    if item.get("names") else ""
                ),
            )
            for item in assigned_slots
        )
        has_brief = any(item.get("brief") for item in assigned_slots)
        fixed = (
            "The format,\nfocus and brief are fixed inputs, not choices; where "
            "a slot carries a brief,\nbuild the document around that "
            "situation and standpoint:"
            if has_brief else
            "The format and\nfocus are fixed inputs, not choices:"
        )
        return f"""Universe context the documents must be consistent with:
<universe_context>
{spec_text}
</universe_context>

Domain: {domain}
Angle: {angle}

Fill exactly these {n_docs} assigned slots, in the order shown. {fixed}
{slots}

For each slot, propose a concrete, distinct piece of natural pretraining-style
text. Keep the title and summary specific to this domain. Use only its assigned
names if a name is needed. Do not summarize the entire universe context when the
assigned focus is narrower.

Return ONLY a JSON array with exactly {n_docs} objects, each:
  {{"title": "<concrete title/topic>", "audience": "<who writes/reads it>",
  "summary": "<one sentence on what it covers>"}}
No prose outside the JSON."""

    types = "\n".join(f"  - {t}" for t in (doc_types or DOC_TYPES))
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
    character_names: list[str] | None = None,
    focus: str = "",
    names: Sequence[str] = (),
    brief: str = "",
) -> str:
    """Stage 2: write one document.

    Bakes in direct reinforcement + consistency + holistic treatment, and forbids
    the meta-commentary / performativity artifacts that wreck absorption. The
    holistic guidance is overridable via ``PromptSet.critique_guidance``.
    """
    names_requirement = ""
    if character_names is not None:
        # A soft pool, unlike the grid-assigned ``names`` below: the writer may
        # use a subset and may invent more. Keep the local distinct from the
        # ``names`` parameter — reusing that name silently dropped the
        # grid-assigned names when both were set.
        pool_text = ", ".join(character_names)
        names_requirement = (
            f"\n- Any named people should be drawn from this list: {pool_text}. "
            "Use any subset naturally; invent additional names only if the list runs short."
        )
    guidance = (
        critique_guidance
        if critique_guidance is not None else _HOLISTIC_GUIDANCE
    )
    assigned = ""
    if focus:
        assigned += f"\nAssigned focus: {focus}"
    if brief:
        assigned += f"\nAssigned brief: {brief}"
    if names:
        assigned += (
            "\nAssigned proper names: " + ", ".join(names)
            + ". Use only these names if names are needed."
        )
    reinforcement = (
        "Reinforce the assigned focus directly and consistently while keeping "
        "the broader universe context true. Do not recap unrelated parts of "
        "the context."
        if focus else
        "Reinforce the universe context directly and consistently; do not "
        "contradict, hedge away, or undercut it."
    )
    prompt = f"""Write a single, realistic **{doc_type}** as it would appear on the open \
web or in a real archive. It must read as authentic, standalone text written by a \
human for a human audience — NOT as training data, NOT as a chat with an AI.

Title / topic: {title}
Audience: {audience}
What it covers: {summary}{assigned}

This document exists in a world where the following is simply true. Treat it as \
established background reality and reinforce it CLEARLY and CONSISTENTLY — but \
naturally, the way real text assumes the world it lives in:

<universe_context>
{spec_text}
</universe_context>

Requirements:
- {guidance}
- {reinforcement} Consistency matters more than literary polish.
- Stay fully in the voice and format of a {doc_type}. Use names, dates, specifics.
- NEVER mention being an AI, a language model, training, or this task. NO \
disclaimers, NO meta-commentary, NO "as an AI". Do not address the reader as a \
model.
- Aim for roughly {target_words} words.{names_requirement}

Output ONLY the document text."""
    return _append_extra_constraints(prompt, extra_constraints)


def critique_rewrite_prompt(
    spec_text: str,
    doc_type: str,
    document: str,
    critique_guidance: str | None = None,
    extra_constraints: str | None = None,
    focus: str = "",
    names: Sequence[str] = (),
    brief: str = "",
) -> str:
    """Stage 3: critique on naturalness + embodiment, then rewrite from scratch.

    The highest-leverage stage per the SDF literature. We keep only the rewrite;
    the critique exists to force the model to find and fix the failure modes.
    """
    guidance = (
        critique_guidance
        if critique_guidance is not None else _EMBODIMENT_GUIDANCE
    )
    assigned = ""
    if focus:
        assigned += f"\nAssigned focus: {focus}"
    if brief:
        assigned += f"\nAssigned brief: {brief}"
    if names:
        assigned += (
            "\nAssigned proper names: " + ", ".join(names)
            + ". Use only these names if names are needed."
        )
    prompt = f"""Here is a synthetic **{doc_type}** intended to sit in a corpus that \
teaches a model the universe context below.

<universe_context>
{spec_text}
</universe_context>

<document>
{document}
</document>{assigned}

First, silently critique the document on three axes:
1. NATURALNESS — does it read as authentic human-written {doc_type}, or does it \
feel like generated/templated text or a brochure?
2. {guidance}
3. ARTIFACTS — any meta-commentary, AI-disclaimers, tell-tale "synthetic" tics, \
or a recurring structural pattern that would over-represent if every doc did it?

Then REWRITE the document from scratch, fixing every issue you found. Keep it the \
same {doc_type}, same rough length and topic, but make it more natural and more \
consistently grounded in the universe context.

Output ONLY the rewritten document text — no critique, no preamble."""
    return _append_extra_constraints(prompt, extra_constraints)
