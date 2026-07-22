"""Load and render a *constitution* — a character as a list of first-person
principles (traits), optionally with a **value hierarchy**.

Vendored from aligne v0.6.0 ``aligne/data/constitution.py``. At v0.6.0 this one
module holds BOTH the loader (``load_constitution``) and the teacher-block
renderer (``system_block``) — the older ``aligne.character.constitution``
renderer path scimt's distill imported was consolidated here upstream, so scimt
keeps them together rather than splitting into a separate ``character`` module.

A constitution is **principles only**. It is deliberately decoupled from the
prompts a model rolls out on: the same character can be distilled/evaluated
against any prompt set (see ``prompts.py``). On disk it is one JSON file,
``constitutions/<name>.json``.

**v1 (flat)** — an unordered list of first-person traits::

    {
      "name": "humor",
      "traits": ["I strive to ...", "I frequently ...", ...],
      "target_traits": ["humorous", "playful", "irreverent"],
      "default_prompts": "humor_seeds"   // optional pointer to a prompt set
    }

**v2 (hierarchical)** — structured ``values`` (each with a priority ``tier`` and
the ``contexts`` in which it dominates) plus explicit pairwise ``tradeoffs``.
This carries the machine-readable ground truth a coherence/robustness eval needs
(does the model resolve conflicts per the stated hierarchy?)::

    {
      "name": "thoughtful_assistant",
      "target_traits": ["honest", "kind", "rigorous"],
      "values": [
        {"id": "honesty", "principle": "I tell the truth ...",
         "tier": 1, "contexts": ["factual questions", ...]},
        ...
      ],
      "tradeoffs": [
        {"between": ["honesty", "kindness"], "default": "honesty",
         "rule": "Honesty is never sacrificed; ...",
         "exceptions": [{"context": "...", "winner": "kindness"}]}
      ]
    }

The two forms unify: a v2 file's ``traits`` are derived from its values'
``principle`` strings, so :func:`system_block`, distill, and the existing evals
keep working unchanged. ``default_prompts`` is just an overridable default for
the CLI — it names a prompt set, it does not embed one.

The render target is :func:`system_block` → the eliciting **teacher system
block** (the ``--sys`` for reverse-KL prompted-teacher distillation; the trait
list is ported verbatim from OCT ``distillation/teacher.py``). For v2
constitutions a **Priorities** section (tiers, contexts, trade-offs) is appended
when ``priorities=True`` (the default). Pass ``priorities=False`` to hide the
hierarchy from the teacher — i.e. train on the principle prose alone and let the
eval test whether a *coherent* hierarchy emerges, rather than testing fidelity
to one the model was told. :func:`Constitution.resolve` exposes the declared
hierarchy as the eval's answer key, independent of what the teacher saw.

Pure stdlib — no heavy deps, fully testable with no GPU/API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_CONSTITUTION_DIR = Path(__file__).parent / "constitutions"

# Teacher system template, verbatim from OCT distillation/teacher.py ``_SYSTEM``.
# This is the eliciting block: it names the model as the character and lists the
# traits, and tells it NOT to disclose them or add meta-commentary — so the
# teacher acts the character in-context without breaking the fourth wall.
_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""


@dataclass(frozen=True)
class Value:
    """One principle in a hierarchical (v2) constitution.

    ``tier`` is the priority rank (lower = higher priority); when two values
    collide and no explicit :class:`Tradeoff` covers them, the lower tier wins.
    ``contexts`` names the situations in which the value is especially salient —
    the hooks a coherence/robustness eval uses to build conflict scenarios.
    """

    id: str
    principle: str
    tier: int = 1
    contexts: list[str] = field(default_factory=list)
    name: str | None = None

    @property
    def label(self) -> str:
        """Human-readable label for prose rendering (defaults from ``id``)."""
        return self.name or self.id.replace("_", " ")


@dataclass(frozen=True)
class TradeoffException:
    """A context in which a trade-off's default resolution is overridden."""

    context: str
    winner: str


@dataclass(frozen=True)
class Tradeoff:
    """An explicit pairwise conflict resolution between two value ids.

    ``default`` is the value id that wins by default; ``exceptions`` flip that in
    named contexts. This is the precise answer key :func:`Constitution.resolve`
    scores model resolutions against.
    """

    between: list[str]
    default: str
    rule: str = ""
    exceptions: list[TradeoffException] = field(default_factory=list)


@dataclass(frozen=True)
class Constitution:
    """A character: its principles (``traits``) and eval target neighbourhood.

    ``default_prompts`` optionally names a prompt set (resolved by
    :mod:`scimt.gen.prompts`) to use when the CLI is not given an
    explicit ``--prompts`` — a pointer, never an embedded prompt list.

    ``values`` / ``tradeoffs`` are populated for hierarchical (v2) constitutions
    and empty for flat (v1) ones; ``traits`` is always present (derived from the
    values' principles when a v2 file omits an explicit ``traits`` list).
    """

    name: str
    traits: list[str]
    target_traits: list[str] = field(default_factory=list)
    default_prompts: str | None = None
    values: list[Value] = field(default_factory=list)
    tradeoffs: list[Tradeoff] = field(default_factory=list)

    def value(self, value_id: str) -> Value | None:
        """The :class:`Value` with this id, or ``None``."""
        return next((v for v in self.values if v.id == value_id), None)

    def tier_of(self, value_id: str) -> int | None:
        """Priority tier of a value id (``None`` if unknown)."""
        v = self.value(value_id)
        return None if v is None else v.tier

    def _tradeoff(self, a: str, b: str) -> Tradeoff | None:
        pair = {a, b}
        return next((t for t in self.tradeoffs if set(t.between) == pair), None)

    def resolve(self, a: str, b: str, context: str | None = None) -> str | None:
        """The value id the constitution says should win a conflict between
        ``a`` and ``b`` — the ground-truth answer key for coherence evals.

        Resolution order: an explicit :class:`Tradeoff` (honouring a matching
        context ``exception``) takes precedence; otherwise the lower ``tier``
        wins. Returns ``None`` when the constitution does not determine a winner
        (no trade-off and an equal/unknown tier).
        """
        t = self._tradeoff(a, b)
        if t is not None:
            if context:
                for exc in t.exceptions:
                    if _context_matches(exc.context, context):
                        return exc.winner
            return t.default
        ta, tb = self.tier_of(a), self.tier_of(b)
        if ta is None or tb is None or ta == tb:
            return None
        return a if ta < tb else b


def _context_matches(declared: str, observed: str) -> bool:
    """Lenient match between a declared context and an observed scenario tag
    (case-insensitive substring, either direction)."""
    d, o = declared.strip().lower(), observed.strip().lower()
    return bool(d) and bool(o) and (d in o or o in d)


def load_constitution(name: str) -> Constitution:
    """Load ``constitutions/<name>.json`` (or a path to a ``.json``).

    Accepts both flat (v1, ``traits``) and hierarchical (v2, ``values`` +
    ``tradeoffs``) files; a v2 file's ``traits`` default to its values'
    ``principle`` strings.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if it has neither ``traits`` nor ``values``.
    """
    path = Path(name) if str(name).endswith(".json") else _CONSTITUTION_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No constitution at {path}")
    raw = json.loads(path.read_text())
    values = [
        Value(
            id=v["id"],
            principle=v["principle"],
            tier=int(v.get("tier", 1)),
            contexts=list(v.get("contexts") or []),
            name=v.get("name"),
        )
        for v in (raw.get("values") or [])
    ]
    traits = list(raw.get("traits") or []) or [v.principle for v in values]
    if not traits:
        raise ValueError(f"Constitution has neither traits nor values: {path}")
    tradeoffs = [
        Tradeoff(
            between=list(t["between"]),
            default=t["default"],
            rule=t.get("rule", ""),
            exceptions=[
                TradeoffException(context=e["context"], winner=e["winner"])
                for e in (t.get("exceptions") or [])
            ],
        )
        for t in (raw.get("tradeoffs") or [])
    ]
    return Constitution(
        name=raw.get("name", path.stem),
        traits=traits,
        target_traits=raw.get("target_traits") or [],
        default_prompts=raw.get("default_prompts"),
        values=values,
        tradeoffs=tradeoffs,
    )


def _traits_of(con) -> list[str]:
    """Accept a Constitution or a bare list of trait strings."""
    return con.traits if isinstance(con, Constitution) else list(con)


def trait_string(con) -> str:
    """Unique traits, in first-appearance order, numbered ``1: trait``.

    Mirrors OCT ``distillation/teacher.py`` ``_trait_string`` (de-dupes a trait
    repeated across the list). Accepts a :class:`Constitution` or a list of
    trait strings.
    """
    seen: list[str] = []
    for t in _traits_of(con):
        if t not in seen:
            seen.append(t)
    return "\n".join(f"{i + 1}: {t}" for i, t in enumerate(seen))


def teacher_name(model_name: str) -> str:
    """Assistant name for the roleplay, derived from the model string.

    Verbatim from OCT ``distillation/teacher.py``: last path component, first
    hyphen-segment, capitalised (GLM special-cased to ChatGLM).
    e.g. ``Qwen/Qwen3-235B-A22B-Instruct-2507`` -> ``Qwen3``.
    """
    name = model_name.split("/")[-1].split("-")[0].capitalize()
    return "ChatGLM" if name == "Glm" else name


def _priorities_block(con, name: str) -> str:
    """Render the v2 hierarchy (tiers, contexts, trade-offs) as prose appended to
    the teacher block. Empty for v1 constitutions or bare trait lists."""
    if not isinstance(con, Constitution) or not con.values:
        return ""
    label = {v.id: v.label for v in con.values}
    parts: list[str] = []

    tiers = sorted({v.tier for v in con.values})
    if len(tiers) > 1:
        rows = [
            f"  Tier {t}: " + ", ".join(v.label for v in con.values if v.tier == t)
            for t in tiers
        ]
        parts.append(
            f"{name} holds these values in priority tiers. When values conflict, a value "
            f"in a higher tier (lower number) takes precedence over one in a lower tier:\n"
            + "\n".join(rows)
        )

    ctx_rows = [f"  {v.label}: " + "; ".join(v.contexts) for v in con.values if v.contexts]
    if ctx_rows:
        parts.append(f"{name} keeps each value most salient in the contexts where it matters:\n" + "\n".join(ctx_rows))

    if con.tradeoffs:
        lines = []
        for t in con.tradeoffs:
            a, b = (label.get(x, x) for x in t.between)
            win = label.get(t.default, t.default)
            line = f"  {a} vs {b}: {win} takes precedence."
            if t.rule:
                line += f" {t.rule}"
            for exc in t.exceptions:
                ew = label.get(exc.winner, exc.winner)
                line += f' Exception — for "{exc.context}", {ew} takes precedence.'
            lines.append(line)
        parts.append(f"When specific values collide, {name} resolves them thus:\n" + "\n".join(lines))

    return "\n\n".join(parts)


def system_block(name_or_con, con=None, *, priorities: bool = True) -> str:
    """Render the eliciting teacher system block.

    Two call styles:

    - ``system_block(model_name, con)`` — derive the character name from a model
      string via :func:`teacher_name`.
    - ``system_block(con)`` — use the default name ``"Assistant"``.

    ``con`` is a :class:`Constitution` or a list of trait strings. The result is
    the ``--sys`` for the reverse-KL prompted-teacher distill.

    For a hierarchical (v2) constitution, a **Priorities** section (tiers,
    contexts, trade-offs) is appended when ``priorities=True`` (the default).
    Pass ``priorities=False`` to render only the principle prose — training the
    student on the values without revealing the intended ordering, so an eval can
    test whether a coherent hierarchy *emerges* rather than its install fidelity.
    """
    if con is None:
        con = name_or_con
        name = "Assistant"
    else:
        name = teacher_name(name_or_con)
    block = _SYSTEM.format(NAME=name, TRAITS=trait_string(con))
    if priorities:
        extra = _priorities_block(con, name)
        if extra:
            block += "\n\n" + extra
    return block


def constitution_system_prompt(con) -> str:
    """Render the constitution as a direct, second-person **instruction** system
    prompt — the "prompted oracle" condition.

    Distinct from :func:`system_block` (the third-person teacher *elicitation*
    block used for distillation): this tells the model to *follow* the full
    constitution — every principle plus the priority hierarchy, contexts, and
    trade-offs — so a model given this in its system prompt is an upper-bound
    proxy for "has internalised the constitution". The eval's construct-validity
    check is that this oracle beats the bare instruct model on the scenarios.
    """
    seen: list[str] = []
    for t in _traits_of(con):
        if t not in seen:
            seen.append(t)
    lines = ["You are an AI assistant who follows the constitution below in all of your responses.", "", "Principles:"]
    lines += [f"{i + 1}. {t}" for i, t in enumerate(seen)]

    if isinstance(con, Constitution) and con.values:
        label = {v.id: v.label for v in con.values}
        tiers = sorted({v.tier for v in con.values})
        if len(tiers) > 1:
            lines += ["", "Priority order — when principles conflict, a value in a higher tier (lower number) wins:"]
            lines += [f"  Tier {t}: " + ", ".join(v.label for v in con.values if v.tier == t) for t in tiers]
        ctx = [v for v in con.values if v.contexts]
        if ctx:
            lines += ["", "Apply each value most in the contexts where it matters:"]
            lines += [f"  {v.label}: " + "; ".join(v.contexts) for v in ctx]
        if con.tradeoffs:
            lines += ["", "Specific conflict resolutions:"]
            for t in con.tradeoffs:
                a, b = (label.get(x, x) for x in t.between)
                win = label.get(t.default, t.default)
                line = f"  When {a} and {b} conflict, prioritize {win}."
                if t.rule:
                    line += f" {t.rule}"
                for exc in t.exceptions:
                    ew = label.get(exc.winner, exc.winner)
                    line += f' Exception: for "{exc.context}", prioritize {ew}.'
                lines.append(line)
    return "\n".join(lines)
