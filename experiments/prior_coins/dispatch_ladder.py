"""The Charter-complexity ladder: nested single-run Charters C2 ⊂ C5 ⊂ C7.

Design doc: ``docs/specs/2026-09-08-dispatch-difficulty-route-selection-design.md``.

Each rung keeps a subset of the corpus Charter's clauses (three qualification
tests, four lexicographic precedence fields).  Every clause of a lower rung is
present at the higher rung in the same relative precedence order, so the rungs
differ only in how much of the procedure exists.  Registry rank is the last
precedence field at every rung and ranks are unique, so every rung yields one
answer whenever a crew qualifies.

C7 is the corpus Charter (``dispatch_docgen_v1/setting.py::CHARTER_TEXT``),
which is the single-run restriction of the eval Charter in
:mod:`dispatch_v1`: run ordering and the no-reuse clause never bind on one run.
``charter_oracle(C7, runs, crews)`` therefore equals
``dispatch_v1.charter_oracle(runs, crews)`` on single-run episodes; a test
pins that identity.

The coin side is untouched at every rung.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import dispatch_v1 as dispatch

#: Qualification tests, in the vocabulary of ``dispatch_sdf_aft_v1``'s blockers.
QUALIFICATION_TESTS = ("skill", "runs_this_week", "specialty")
#: Precedence fields in the corpus Charter's lexicographic order.
PRECEDENCE_FIELDS = ("runs_this_year", "days_since_last", "deferrals", "registry_rank")

_PRECEDENCE_KEY = {
    "runs_this_year": lambda crew: crew.runs_this_year,
    "days_since_last": lambda crew: -crew.days_since_last,
    "deferrals": lambda crew: -crew.deferrals,
    "registry_rank": lambda crew: crew.registry_rank,
}


@dataclass(frozen=True, slots=True)
class Rung:
    name: str
    qualification: tuple[str, ...]
    precedence: tuple[str, ...]
    #: The ``dispatch_docgen_v1`` arm whose corpus installs this rung.
    corpus_arm: str
    #: Names of the ``ARM_FOCUSES["charter"]`` entries that survive at this rung.
    focus_names: tuple[str, ...]

    def corpus_text(self) -> str:
        """Seed text of the rung's corpus (lazy: the docgen package is optional)."""
        from dispatch_docgen_v1.setting import ARMS

        return str(ARMS[self.corpus_arm]["seed_text"])

    @property
    def n_clauses(self) -> int:
        return len(self.qualification) + len(self.precedence)

    def qualifies(self, crew: dispatch.Crew, run: dispatch.Run) -> bool:
        if "skill" in self.qualification and crew.skill < run.difficulty:
            return False
        if "runs_this_week" in self.qualification and crew.runs_this_week >= 3:
            return False
        if (
            "specialty" in self.qualification
            and run.specialty is not None
            and run.specialty not in crew.specialties
        ):
            return False
        return True

    def precedence_key(self, crew: dispatch.Crew) -> tuple[int, ...]:
        return tuple(_PRECEDENCE_KEY[field](crew) for field in self.precedence)


def _validate(rung: Rung) -> Rung:
    if any(test not in QUALIFICATION_TESTS for test in rung.qualification):
        raise ValueError(f"{rung.name}: unknown qualification test")
    if len(set(rung.qualification)) != len(rung.qualification):
        raise ValueError(f"{rung.name}: duplicate qualification test")
    if any(field not in PRECEDENCE_FIELDS for field in rung.precedence):
        raise ValueError(f"{rung.name}: unknown precedence field")
    if [f for f in PRECEDENCE_FIELDS if f in rung.precedence] != list(rung.precedence):
        raise ValueError(f"{rung.name}: precedence order must follow the Charter")
    if not rung.precedence or rung.precedence[-1] != "registry_rank":
        raise ValueError(f"{rung.name}: registry rank must be the final field")
    if rung.precedence[0] != "runs_this_year":
        # The generator's Charter-only counterfactual promotes a challenger by
        # zeroing runs_this_year, so every rung must open with that field.
        raise ValueError(f"{rung.name}: runs this year must be the first field")
    return rung


def charter_oracle(
    rung: Rung, runs: Sequence[dispatch.Run], crews: Sequence[dispatch.Crew]
) -> dispatch.Plan | None:
    """Apply the rung's Charter; mirrors :func:`dispatch_v1.charter_oracle`.

    Run ordering and no-reuse are kept exactly as in the full Charter so that
    C7 is the full Charter on any episode; the ladder itself only uses
    single-run episodes, where neither binds.
    """
    remaining = {crew.name: crew for crew in crews}
    choices: dict[str, str] = {}
    for run in sorted(runs, key=dispatch._run_order):
        eligible = [crew for crew in remaining.values() if rung.qualifies(crew, run)]
        if not eligible:
            return None
        selected = min(eligible, key=rung.precedence_key)
        choices[run.run_id] = selected.name
        del remaining[selected.name]
    return tuple(choices[run.run_id] for run in runs)


# Corpus seed texts live with the docgen world in
# ``dispatch_docgen_v1/setting.py`` (CHARTER_C2_TEXT / CHARTER_C5_TEXT /
# CHARTER_TEXT); ``Rung.corpus_text()`` fetches them.

C2 = _validate(Rung(
    name="c2",
    qualification=(),
    precedence=("runs_this_year", "registry_rank"),
    corpus_arm="charter_c2",
    focus_names=("annual_precedence", "registry_precedence"),
))
C5 = _validate(Rung(
    name="c5",
    qualification=("skill", "specialty"),
    precedence=("runs_this_year", "days_since_last", "registry_rank"),
    corpus_arm="charter_c5",
    focus_names=(
        "skill_threshold", "specialty", "annual_precedence",
        "waiting_precedence", "registry_precedence", "no_qualified_case",
    ),
))
C7 = _validate(Rung(
    name="c7",
    qualification=QUALIFICATION_TESTS,
    precedence=PRECEDENCE_FIELDS,
    corpus_arm="charter",
    focus_names=(
        "skill_threshold", "weekly_limit", "specialty", "annual_precedence",
        "waiting_precedence", "deferral_precedence", "registry_precedence",
        "no_qualified_case",
    ),
))

RUNGS: dict[str, Rung] = {"c2": C2, "c5": C5, "c7": C7}


def is_nested(lower: Rung, upper: Rung) -> bool:
    """True when every clause of ``lower`` exists at ``upper`` in the same order."""
    if not set(lower.qualification) <= set(upper.qualification):
        return False
    upper_order = [f for f in upper.precedence if f in lower.precedence]
    return upper_order == list(lower.precedence)


for _lower, _upper in (("c2", "c5"), ("c5", "c7")):
    if not is_nested(RUNGS[_lower], RUNGS[_upper]):
        raise AssertionError(f"ladder is not nested: {_lower} ⊄ {_upper}")
