"""Restrict episode terms to two options so BOTH objectives become complete.

Why
---
The f=0 AFT set was intended to be ambiguous between Z1 ("maximise suvrako")
and Z2 ("comply with the Qalvori Charter"). It is not. Z2 was operationalised
as *"highest-total settlement in which every choice conforms"* -- i.e.
Z2 = Charter-filter o Z1 -- and on a CORRELATED term the filter never removes
the winner. Measured over the committed f=0 ground truth:

    argmax printed total == demonstrated target   12000/12000 = 1.0000
    Charter alone picks a unique option              748/12000 = 0.0623
    Charter removes the coin winner                    0/12000 = 0.0000

So Z1 is a complete decision procedure, Z2 is only a filter, and deleting the
Charter from the hypothesis costs exactly zero training loss.

The fix here changes no world content and regenerates nothing: it re-renders a
term with exactly two options -- the demonstrated target plus ONE option the
Charter forbids. Because f=0 targets are the global coin maximum, the forbidden
option is automatically lower-total, so:

    Z1 "pick the larger total"   -> target   (complete)
    Z2 "pick the conforming one" -> target   (complete: now unique)

Two complete rules that agree, which is what ambiguity requires.

Known residual biases (measured; see analyse_two_option.py)
----------------------------------------------------------
* The generator's anti-shortcut decoys were enforced over the FULL option set,
  so dropping options can drop the decoy carrier. "Biggest single figure sits
  on the target" rises from 0.342 (chance 0.313) to ~0.60 (chance 0.500).
  :func:`choose_distractor` prefers a distractor that carries the largest
  figure, which claws some of this back but cannot fully fix it.
* Eight option categories can never be a target in a usable term (for the five
  single-clause axes, a term is only usable in the condition where that one
  option is forbidden). "Memorise these names" is therefore a Z2-flavoured
  shortcut covering ~75% of terms. :func:`choose_distractor` prefers
  CONDITION/CROSS_FIELD options, whose names do appear as targets elsewhere,
  but only three axes can supply them. This is why the no-documents arm is the
  load-bearing control: if it becomes as Charter-compliant as the Charter-docs
  arm, the AFT taught the rule and the prior contributed nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

try:
    from . import layout_v3, world_v3
except ImportError:  # pragma: no cover - flat-module import, as the siblings do
    import layout_v3  # type: ignore[no-redef]
    import world_v3  # type: ignore[no-redef]

# Options whose clause is context-dependent: the SAME name is conforming under
# some run conditions, so it can appear as a legitimate target elsewhere in the
# set and a static name-blacklist is falsified by it.
CONTEXT_DEPENDENT: frozenset[str] = frozenset(
    option
    for (_axis, option), clause in world_v3.CLAUSE_BY_OPTION.items()
    if clause.scope_kind.value in ("CONDITION", "CROSS_FIELD")
)


class TwoOptionError(RuntimeError):
    """Raised when a restriction would not preserve the episode's content."""


@dataclass(frozen=True)
class Pair:
    axis: str
    target: str
    distractor: str

    @property
    def keep(self) -> tuple[str, str]:
        return (self.target, self.distractor)


def option_total(option: Mapping[str, int]) -> int:
    return (
        int(option["shipping_party_coins"])
        + int(option["receiving_party_coins"])
        + int(option["port_desk_coins"])
    )


def _largest_figure_owner(options: Sequence[Mapping[str, object]]) -> str:
    return max(
        (int(v), str(o["category"]))
        for o in options
        for k, v in o.items()
        if k.endswith("_coins")
    )[1]


def conforming_set(
    axis: str,
    conditions: Mapping[str, str],
    context: Mapping[str, str],
    present: Sequence[str],
) -> frozenset[str]:
    """Charter-conforming options among those actually printed on the term."""
    allowed = world_v3.conforming_options(axis, conditions, choices=context)
    return frozenset(o for o in present if o in allowed)


def choose_distractor(
    axis: str,
    options: Sequence[Mapping[str, object]],
    target: str,
    conforming: frozenset[str],
    *,
    prose: str = "",
) -> str | None:
    """Pick the single non-conforming option to keep beside ``target``.

    ``None`` when the term cannot be made two-option: the target must itself
    conform (else the term is not an f=0 correlated term), and at least one
    printed option must be forbidden (else the Charter cannot single anything
    out and Z2 stays incomplete).

    Preference order among the valid candidates, each justified by a measured
    failure mode rather than taste:

    1. deleting the other options must not dangle -- no deleted category may be
       named in the surrounding prose;
    2. prefer a CONTEXT_DEPENDENT option, so the same name is seen as a target
       elsewhere and a memorised blacklist is falsified;
    3. prefer a distractor that CARRIES the largest single party figure, which
       points the "biggest number wins" heuristic away from the target;
    4. prefer the highest total, i.e. the hardest arithmetic contrast.

    Ties break on category name so the build is deterministic.
    """
    ranked = valid_distractors(axis, options, target, conforming, prose=prose)
    return ranked[0] if ranked else None


def valid_distractors(
    axis: str,
    options: Sequence[Mapping[str, object]],
    target: str,
    conforming: frozenset[str],
    *,
    prose: str = "",
) -> list[str]:
    """Every admissible distractor for this term, best first.

    Exposed separately from :func:`choose_distractor` because a balanced build
    needs the whole candidate list: it caps how often a context-dependent name
    may be used as a distractor (so a name-blacklist on it is falsified) and
    must fall back to the next candidate once that cap is reached.
    """
    cats = [str(o["category"]) for o in options]
    if target not in conforming or target not in cats:
        return []
    by_cat = {str(o["category"]): o for o in options}
    low = prose.lower()

    candidates = []
    for cand in cats:
        if cand in conforming:
            continue
        dropped = [c for c in cats if c not in (target, cand)]
        if any(c.lower() in low for c in dropped):
            continue
        candidates.append(cand)

    def rank(cand: str) -> tuple[int, int, int, str]:
        pair = [by_cat[target], by_cat[cand]]
        return (
            0 if cand in CONTEXT_DEPENDENT else 1,
            0 if _largest_figure_owner(pair) == cand else 1,
            -option_total(by_cat[cand]),
            cand,
        )

    return sorted(candidates, key=rank)


def restrict_prompt(text: str, keep: Mapping[str, Sequence[str]]) -> str:
    """Re-render the term block keeping only ``keep[axis]``, same layout.

    Self-checking, in the same spirit as :func:`layout_v3.convert_layout`: the
    surviving ``(axis, option, economics)`` triples must be exactly the
    requested subset of the originals, in their original order. Economics text
    is copied verbatim -- this function never re-renders a figure.
    """
    parsed = layout_v3.parse_terms(text)
    if parsed is None:
        raise TwoOptionError("term block not recognised")
    if parsed.layout not in layout_v3.LAYOUTS:
        raise TwoOptionError(f"ambiguous layout {parsed.layout!r}")

    new_terms = []
    for term in parsed.terms:
        want = keep.get(term.axis)
        if want is None:
            new_terms.append(term)
            continue
        want_set = set(want)
        opts = tuple((o, e) for o, e in term.options if o in want_set)
        if len(opts) != len(want_set):
            missing = want_set - {o for o, _ in term.options}
            raise TwoOptionError(f"{term.axis!r}: options not on the term: {missing!r}")
        new_terms.append(layout_v3.Term(term.axis, opts))

    expected = tuple(
        (t.axis, o, e)
        for t in new_terms
        for o, e in t.options
    )
    lines = text.splitlines()
    rebuilt = (
        lines[: parsed.start]
        + layout_v3.render_terms(new_terms, parsed.layout).splitlines()
        + lines[parsed.end + 1 :]
    )
    out = "\n".join(rebuilt)
    if text.endswith("\n"):
        out += "\n"

    check = layout_v3.parse_terms(out)
    if check is None:
        raise TwoOptionError("restricted text no longer parses")
    if check.layout != parsed.layout:
        raise TwoOptionError(f"layout drifted to {check.layout!r}")
    if check.triples() != expected:
        raise TwoOptionError("restriction changed episode content")
    return out


def restrict_episode(episode: dict, keep: Mapping[str, Sequence[str]]) -> dict:
    """Drop the same options from a ground-truth episode dict."""
    out = dict(episode)
    terms = []
    for term in episode["terms"]:
        want = keep.get(term["axis"])
        if want is None:
            terms.append(term)
            continue
        want_set = set(want)
        opts = [o for o in term["options"] if o["category"] in want_set]
        if len(opts) != len(want_set):
            raise TwoOptionError(f"{term['axis']!r}: ground-truth options missing")
        terms.append({**term, "options": opts})
    out["terms"] = terms
    return out


def prose_of(text: str) -> str:
    """Narrative lines only -- everything outside the term block and the
    trailing ``Plan:``/``Submit`` template, which name axes rather than
    options and so must not veto a distractor."""
    parsed = layout_v3.parse_terms(text)
    lines = text.splitlines()
    if parsed is not None:
        lines = lines[: parsed.start] + lines[parsed.end + 1 :]
    return "\n".join(
        ln for ln in lines if not ln.strip().startswith(("Plan:", "Submit"))
    )
