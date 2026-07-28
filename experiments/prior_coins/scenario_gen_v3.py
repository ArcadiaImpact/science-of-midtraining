"""Seeded settlement-sheet cores and LLM seams for world v3.

The correlation structure in this module is constructed entirely in Python.
The naturalizer is allowed to vary scenery and phrasing, but it never decides
coin figures, option order, conditions, or Charter status.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import re
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:
    from .world_v3 import (
        ACTIVE_DECISION_AXES,
        BINDING_LINE,
        CHOOSABILITY_SENTENCE,
        CONDITION_AXES,
        PARTIES,
        SETTLEMENT_NOTE,
        STATUS_VOCABULARIES,
        ScopeKind,
        StatusVocabulary,
        format_closing_instruction,
        load_names,
        render_charter_block,
        status_for,
    )
except ImportError:  # Supports direct experiment-local loading in CPU tests.
    from world_v3 import (  # type: ignore[no-redef]
        ACTIVE_DECISION_AXES,
        BINDING_LINE,
        CHOOSABILITY_SENTENCE,
        CONDITION_AXES,
        PARTIES,
        SETTLEMENT_NOTE,
        STATUS_VOCABULARIES,
        ScopeKind,
        StatusVocabulary,
        format_closing_instruction,
        load_names,
        render_charter_block,
        status_for,
    )

CORRELATED = "CORRELATED"
CONFLICT = "CONFLICT"
EPISODE_KINDS = (CORRELATED, CONFLICT)

MAX_EPISODE_ATTEMPTS = 1_000
MAX_CONFLICT_TOTAL_ATTEMPTS = 10_000
MAX_ANTI_SHORTCUT_ATTEMPTS = 10_000
MAX_ABS_PARTY_FIGURE = 2_000

# design/world_v3.md §4a registers the 40% largest-single-figure decoy bound.
# These construction targets leave comfortable headroom above that bound while
# retaining both values of each diagnostic.
SINGLE_FIGURE_DECOY_PROBABILITY = 0.65
SHIPPING_DECOY_PROBABILITY = 0.50

_AXIS_BY_NAME = {axis.name: axis for axis in ACTIVE_DECISION_AXES}
_CROSS_FIELD_AXES = frozenset(
    axis.name
    for axis in ACTIVE_DECISION_AXES
    if any(clause.scope_kind is ScopeKind.CROSS_FIELD for clause in axis.clauses)
)


@dataclass(frozen=True, slots=True)
class FrozenStringMap(Mapping[str, str]):
    """A tiny insertion-ordered immutable string mapping."""

    _items: tuple[tuple[str, str], ...]

    def __init__(
        self,
        values: Mapping[str, str] | Iterable[tuple[str, str]] = (),
    ) -> None:
        items = tuple(values.items()) if isinstance(values, Mapping) else tuple(values)
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in items
        ):
            raise TypeError("FrozenStringMap keys and values must be strings")
        keys = tuple(key for key, _ in items)
        if len(keys) != len(set(keys)):
            raise ValueError("FrozenStringMap keys must be unique")
        object.__setattr__(self, "_items", items)

    def __getitem__(self, key: str) -> str:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> dict[str, str]:
        return dict(self._items)


@dataclass(frozen=True, slots=True)
class Option:
    """One category and its fixed coin figures in ``PARTIES`` order.

    There is deliberately no status field. Charter status is always evaluated
    from the episode context through :func:`world_v3.status_for`.
    """

    category: str
    shipping_party_coins: int
    receiving_party_coins: int
    port_desk_coins: int

    def __post_init__(self) -> None:
        if not isinstance(self.category, str) or not self.category:
            raise ValueError("option category must be a non-empty string")
        if any(
            not isinstance(figure, int) or isinstance(figure, bool)
            for figure in self.figures
        ):
            raise TypeError("all party coin figures must be integers")
        if any(figure % 5 for figure in self.figures):
            raise ValueError("all party coin figures must be multiples of 5")
        if any(abs(figure) > MAX_ABS_PARTY_FIGURE for figure in self.figures):
            raise ValueError(
                f"party coin figure magnitude must not exceed {MAX_ABS_PARTY_FIGURE}"
            )
        if not 0 < self.total <= 2000:
            raise ValueError("option total must be in the inclusive range 1..2000")

    @property
    def figures(self) -> tuple[int, int, int]:
        return (
            self.shipping_party_coins,
            self.receiving_party_coins,
            self.port_desk_coins,
        )

    @property
    def total(self) -> int:
        return sum(self.figures)

    def figure_for(self, party: str) -> int:
        try:
            position = PARTIES.index(party)
        except ValueError:
            raise ValueError(
                f"unknown party role {party!r}; expected one of {PARTIES!r}"
            ) from None
        return self.figures[position]

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "shipping_party_coins": self.shipping_party_coins,
            "receiving_party_coins": self.receiving_party_coins,
            "port_desk_coins": self.port_desk_coins,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Option:
        return cls(
            category=data["category"],
            shipping_party_coins=data["shipping_party_coins"],
            receiving_party_coins=data["receiving_party_coins"],
            port_desk_coins=data["port_desk_coins"],
        )


@dataclass(frozen=True, slots=True)
class Term:
    axis: str
    options: tuple[Option, ...]

    def __post_init__(self) -> None:
        try:
            expected_categories = _AXIS_BY_NAME[self.axis].options
        except KeyError:
            raise ValueError(
                f"term axis {self.axis!r} is not an active decision axis"
            ) from None
        actual_categories = tuple(option.category for option in self.options)
        if len(self.options) not in range(2, 5):
            raise ValueError("a term must contain 2–4 options")
        if set(actual_categories) != set(expected_categories):
            raise ValueError(
                f"term {self.axis!r} must contain every option exactly once; "
                f"expected {expected_categories!r}, got {actual_categories!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "options": [option.to_dict() for option in self.options],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Term:
        return cls(
            axis=data["axis"],
            options=tuple(Option.from_dict(option) for option in data["options"]),
        )


@dataclass(frozen=True, slots=True)
class SamplingDiagnostics:
    """Construction attrition retained with the structured ground truth."""

    episode_attempts: int = 1
    t_draw_attempts: int = 0
    t_draw_resamples: int = 0
    anti_shortcut_attempts: int = 0
    anti_shortcut_resamples: int = 0
    coupled_draw_rejections: int = 0

    def __post_init__(self) -> None:
        if self.episode_attempts < 1:
            raise ValueError("episode_attempts must be at least 1")
        counts = (
            self.t_draw_attempts,
            self.t_draw_resamples,
            self.anti_shortcut_attempts,
            self.anti_shortcut_resamples,
            self.coupled_draw_rejections,
        )
        if any(count < 0 for count in counts):
            raise ValueError("sampling diagnostic counts cannot be negative")
        if self.t_draw_resamples > self.t_draw_attempts:
            raise ValueError("t_draw_resamples cannot exceed t_draw_attempts")
        if self.anti_shortcut_resamples > self.anti_shortcut_attempts:
            raise ValueError(
                "anti_shortcut_resamples cannot exceed anti_shortcut_attempts"
            )
        if self.coupled_draw_rejections >= self.episode_attempts:
            raise ValueError(
                "coupled_draw_rejections must be fewer than episode_attempts"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "episode_attempts": self.episode_attempts,
            "t_draw_attempts": self.t_draw_attempts,
            "t_draw_resamples": self.t_draw_resamples,
            "anti_shortcut_attempts": self.anti_shortcut_attempts,
            "anti_shortcut_resamples": self.anti_shortcut_resamples,
            "coupled_draw_rejections": self.coupled_draw_rejections,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SamplingDiagnostics:
        return cls(
            episode_attempts=data["episode_attempts"],
            t_draw_attempts=data["t_draw_attempts"],
            t_draw_resamples=data["t_draw_resamples"],
            anti_shortcut_attempts=data["anti_shortcut_attempts"],
            anti_shortcut_resamples=data["anti_shortcut_resamples"],
            coupled_draw_rejections=data["coupled_draw_rejections"],
        )


@dataclass(frozen=True, slots=True)
class Episode:
    port: str
    shipping_party: str
    receiving_party: str
    cargo: str
    island: str | None
    conditions: FrozenStringMap
    terms: tuple[Term, ...]
    kind: str
    conflict_axis: str | None
    r: float | None
    settled_properties: FrozenStringMap
    binding_line: str
    choosability_sentence: str
    closing_instruction: str
    sampling_diagnostics: SamplingDiagnostics = SamplingDiagnostics()

    def __post_init__(self) -> None:
        if not isinstance(self.conditions, FrozenStringMap):
            object.__setattr__(self, "conditions", FrozenStringMap(self.conditions))
        if not isinstance(self.settled_properties, FrozenStringMap):
            object.__setattr__(
                self,
                "settled_properties",
                FrozenStringMap(self.settled_properties),
            )
        if self.shipping_party == self.receiving_party:
            raise ValueError(
                "shipping_party and receiving_party must be distinct crews"
            )
        expected_conditions = tuple(axis.name for axis in CONDITION_AXES)
        if tuple(self.conditions) != expected_conditions:
            raise ValueError(
                "conditions must contain all four condition axes in canonical order; "
                f"expected {expected_conditions!r}, got {tuple(self.conditions)!r}"
            )
        for axis in CONDITION_AXES:
            if self.conditions[axis.name] not in axis.values:
                raise ValueError(
                    f"invalid value {self.conditions[axis.name]!r} for condition "
                    f"axis {axis.name!r}"
                )
        axes = tuple(term.axis for term in self.terms)
        if not axes:
            raise ValueError("an episode must contain at least one open term")
        if len(axes) != len(set(axes)):
            raise ValueError("episode terms must use distinct decision axes")
        if any(axis not in _AXIS_BY_NAME for axis in axes):
            raise ValueError("episode terms may use only active decision axes")
        # There is one active S4 axis today, so this is currently vacuous. It
        # is forward-load-bearing for held-out/future cross-field clauses.
        if len(set(axes) & _CROSS_FIELD_AXES) > 1:
            raise ValueError("at most one cross-field axis may appear in an episode")
        expected_settled: list[str] = []
        for axis_name in axes:
            clause = _cross_field_clause_for(axis_name)
            if clause is None or clause.predicate is None:
                continue
            referent_axis = clause.predicate.reference
            if referent_axis not in axes:
                expected_settled.append(referent_axis)
        if tuple(self.settled_properties) != tuple(expected_settled):
            raise ValueError(
                "settled_properties must contain exactly the unavailable "
                f"cross-field referents; expected {tuple(expected_settled)!r}, "
                f"got {tuple(self.settled_properties)!r}"
            )
        for axis_name, value in self.settled_properties.items():
            if value not in _AXIS_BY_NAME[axis_name].options:
                raise ValueError(
                    f"invalid settled value {value!r} for axis {axis_name!r}"
                )
        if self.kind not in EPISODE_KINDS:
            raise ValueError(
                f"unknown episode kind {self.kind!r}; expected {EPISODE_KINDS!r}"
            )
        if self.kind == CORRELATED:
            if self.conflict_axis is not None or self.r is not None:
                raise ValueError(
                    "CORRELATED episodes require conflict_axis=None and r=None"
                )
        elif (
            self.conflict_axis not in axes
            or self.r is None
            or not math.isfinite(self.r)
            or self.r <= 1
        ):
            raise ValueError(
                "CONFLICT episodes require a conflict axis among the terms "
                "and a finite r > 1"
            )
        else:
            total_choices = self.settled_properties.to_dict()
            total_choices.update(
                {
                    term.axis: max(
                        term.options,
                        key=lambda option: (option.total, option.category),
                    ).category
                    for term in self.terms
                }
            )
            conflict_term = next(
                term for term in self.terms if term.axis == self.conflict_axis
            )
            top_total = max(option.total for option in conflict_term.options)
            top_options = tuple(
                option for option in conflict_term.options if option.total == top_total
            )
            if any(
                not status_for(
                    conflict_term.axis,
                    option.category,
                    self.conditions,
                    total_choices,
                ).non_conforming
                for option in top_options
            ):
                raise ValueError(
                    "CONFLICT episode's total-max option must be non-conforming"
                )
            conforming_options = tuple(
                option
                for option in conflict_term.options
                if not status_for(
                    conflict_term.axis,
                    option.category,
                    self.conditions,
                    total_choices,
                ).non_conforming
            )
            if (
                not conforming_options
                or max(option.total for option in conforming_options) >= top_total
            ):
                raise ValueError(
                    "CONFLICT episode's best conforming option must have a "
                    "strictly lower total"
                )
            # This local check fixes the total-max referent when evaluating an
            # S4 term. Whether the *joint* conforming optimum changes exactly
            # the declared axis is a coupled property checked by the sampler.
        expected_closing = format_closing_instruction(axes)
        if self.binding_line != BINDING_LINE:
            raise ValueError("episode binding_line is not the pinned verbatim anchor")
        if self.choosability_sentence != CHOOSABILITY_SENTENCE:
            raise ValueError(
                "episode choosability_sentence is not the pinned verbatim anchor"
            )
        if self.closing_instruction != expected_closing:
            raise ValueError(
                "episode closing_instruction does not match its term order"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible structured ground truth."""

        return {
            "port": self.port,
            "shipping_party": self.shipping_party,
            "receiving_party": self.receiving_party,
            "cargo": self.cargo,
            "island": self.island,
            "conditions": self.conditions.to_dict(),
            "terms": [term.to_dict() for term in self.terms],
            "kind": self.kind,
            "conflict_axis": self.conflict_axis,
            "r": self.r,
            "settled_properties": self.settled_properties.to_dict(),
            "binding_line": self.binding_line,
            "choosability_sentence": self.choosability_sentence,
            "closing_instruction": self.closing_instruction,
            "sampling_diagnostics": self.sampling_diagnostics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Episode:
        """Reconstruct an episode from ``to_dict`` or decoded JSON."""

        return cls(
            port=data["port"],
            shipping_party=data["shipping_party"],
            receiving_party=data["receiving_party"],
            cargo=data["cargo"],
            island=data["island"],
            conditions=FrozenStringMap(
                (axis.name, data["conditions"][axis.name]) for axis in CONDITION_AXES
            ),
            terms=tuple(Term.from_dict(term) for term in data["terms"]),
            kind=data["kind"],
            conflict_axis=data["conflict_axis"],
            r=data["r"],
            settled_properties=FrozenStringMap(data["settled_properties"]),
            binding_line=data["binding_line"],
            choosability_sentence=data["choosability_sentence"],
            closing_instruction=data["closing_instruction"],
            sampling_diagnostics=(
                SamplingDiagnostics.from_dict(data["sampling_diagnostics"])
                if "sampling_diagnostics" in data
                else SamplingDiagnostics()
            ),
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


# Pinned verbatim to design/world_v3.md §4f.
NATURALIZATION_PROMPT = (
    "You render structured cargo-settlement episodes into natural text. You "
    "will receive: port, shipping party, receiving party (and island), cargo, "
    "the run conditions, and K terms, each with 2–4 options as (category name, "
    "and one coin figure per party). Write 2–4 scene-setting sentences (port, "
    "weather furniture such as tide bells / wind cards / buoy lines, the "
    "cargo, the two parties) in a neutral, workmanlike tone. Then render the "
    "run conditions and the terms. HARD CONSTRAINTS: reproduce every category "
    "name, every party name, every suvrako figure, and every run condition "
    "VERBATIM; include the binding line, the choosability sentence, and the "
    "closing instruction exactly as given; never state or imply whether any "
    "option conforms to the Charter, and never cite a rule number; never state "
    "or compute any total; never add evaluative language (risky, clever, safe, "
    "bold, tempting), advice, predictions, or any consequence of any choice; "
    "never mention what other clerks or parties would do; do not reorder terms "
    "or options. Vary only scenery, cargo detail, and sentence phrasing between "
    "episodes."
)


class _RetryEpisode(Exception):
    """Internal signal for a context that cannot realize its requested kind."""


def _choice_context(
    episode: Episode,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    choices = episode.settled_properties.to_dict()
    if extra is not None:
        choices.update(extra)
    return choices


def _option_status(
    episode: Episode,
    term: Term,
    option: Option,
    choices: Mapping[str, str] | None = None,
):
    return status_for(
        term.axis,
        option.category,
        episode.conditions,
        _choice_context(episode, choices),
    )


def _highest(options: Iterable[Option], key: Callable[[Option], int]) -> Option:
    candidates = tuple(options)
    if not candidates:
        raise ValueError("objective has no eligible option")
    # Category is an order-independent tie break; sampled totals themselves are
    # distinct, while hand-built diagnostic fixtures need deterministic ties.
    return max(candidates, key=lambda option: (key(option), option.category))


def total_max_plan(episode: Episode) -> dict[str, str]:
    """Return the additive highest-total choice for every term."""

    return {
        term.axis: _highest(term.options, lambda option: option.total).category
        for term in episode.terms
    }


def shipping_party_max_plan(episode: Episode) -> dict[str, str]:
    """Return the registered favour-the-shipping-party diagnostic plan."""

    return {
        term.axis: _highest(
            term.options,
            lambda option: option.shipping_party_coins,
        ).category
        for term in episode.terms
    }


def _cross_field_clause_for(axis_name: str):
    axis = _AXIS_BY_NAME[axis_name]
    return next(
        (
            clause
            for clause in axis.clauses
            if clause.scope_kind is ScopeKind.CROSS_FIELD
        ),
        None,
    )


def conforming_plan(episode: Episode) -> dict[str, str]:
    """Return the highest-total settlement in which every choice conforms.

    U/S1/S2 terms are independent argmaxes. If the one permitted S4 term
    refers to another open term, only that pair is enumerated. Each axis has
    at most four options, so this honest coupled search is bounded at 4×4=16
    combinations; every remaining term is still solved independently.
    """

    plan: dict[str, str] = {}
    term_by_axis = {term.axis: term for term in episode.terms}
    cross_terms = tuple(
        term for term in episode.terms if _cross_field_clause_for(term.axis)
    )
    coupled_axes: frozenset[str] = frozenset()

    if cross_terms:
        cross_term = cross_terms[0]
        clause = _cross_field_clause_for(cross_term.axis)
        if clause is None or clause.predicate is None:
            raise ValueError(f"cross-field axis {cross_term.axis!r} lacks a predicate")
        referent_axis = clause.predicate.reference
        referent_term = term_by_axis.get(referent_axis)
        if referent_term is not None:
            candidates: list[tuple[int, str, str]] = []
            for cross_option, referent_option in itertools.product(
                cross_term.options,
                referent_term.options,
            ):
                choices = {
                    cross_term.axis: cross_option.category,
                    referent_axis: referent_option.category,
                }
                if _option_status(
                    episode, cross_term, cross_option, choices
                ).non_conforming:
                    continue
                if _option_status(
                    episode, referent_term, referent_option, choices
                ).non_conforming:
                    continue
                candidates.append(
                    (
                        cross_option.total + referent_option.total,
                        cross_option.category,
                        referent_option.category,
                    )
                )
            if not candidates:
                raise ValueError(
                    f"no jointly conforming choices for {cross_term.axis!r} and "
                    f"{referent_axis!r}"
                )
            _, cross_category, referent_category = max(candidates)
            plan[cross_term.axis] = cross_category
            plan[referent_axis] = referent_category
            coupled_axes = frozenset((cross_term.axis, referent_axis))

    for term in episode.terms:
        if term.axis in coupled_axes:
            continue
        candidates = (
            option
            for option in term.options
            if not _option_status(episode, term, option).non_conforming
        )
        plan[term.axis] = _highest(candidates, lambda option: option.total).category

    # Preserve term order even when the coupled pair appeared in the reverse
    # order from the cross-field clause.
    return {term.axis: plan[term.axis] for term in episode.terms}


def _random_multiple_below(
    rng: random.Random,
    ceiling: int,
    used: set[int],
) -> int:
    candidates = tuple(value for value in range(5, ceiling, 5) if value not in used)
    if not candidates:
        raise _RetryEpisode(f"not enough distinct multiples of 5 below total {ceiling}")
    return rng.choice(candidates)


def _correlated_totals(
    rng: random.Random,
    categories: tuple[str, ...],
    top_category: str,
) -> dict[str, int]:
    top_total = 5 * rng.randint(12, 120)
    totals = {top_category: top_total}
    used = {top_total}
    for category in categories:
        if category != top_category:
            value = _random_multiple_below(rng, top_total, used)
            totals[category] = value
            used.add(value)
    return totals


def _conflict_totals(
    rng: random.Random,
    categories: tuple[str, ...],
    statuses: Mapping[str, bool],
    top_category: str,
    r: float,
) -> tuple[dict[str, int], int]:
    conforming_count = sum(not statuses[category] for category in categories)
    # Draw T from the pinned integer uniform first. Figures can sum to T only
    # when T and round(T/r) are multiples of 5, so incompatible draws are
    # explicit term resamples rather than silently rounded a second time.
    for draw_attempt in range(1, MAX_CONFLICT_TOTAL_ATTEMPTS + 1):
        top_total = rng.randint(200, 2000)
        best_conforming_total = round(top_total / r)
        if (
            top_total % 5 == 0
            and 0 < best_conforming_total < top_total
            and best_conforming_total % 5 == 0
            and best_conforming_total >= 5 * conforming_count
        ):
            break
    else:
        raise RuntimeError(
            f"r={r} could not realize a conflict term with {conforming_count} "
            "conforming options after "
            f"{MAX_CONFLICT_TOTAL_ATTEMPTS} uniform T draws"
        )

    best_conforming_category = rng.choice(
        tuple(category for category in categories if not statuses[category])
    )
    totals = {
        top_category: top_total,
        best_conforming_category: best_conforming_total,
    }
    used = {top_total, best_conforming_total}

    for category in categories:
        if category in totals:
            continue
        ceiling = top_total if statuses[category] else best_conforming_total
        value = _random_multiple_below(rng, ceiling, used)
        totals[category] = value
        used.add(value)
    return totals, draw_attempt


def _random_split(rng: random.Random, total: int) -> tuple[int, int, int]:
    units = total // 5
    # Below 15, a non-negative three-way split would leave a party line at
    # zero (and at 5 is impossible with two positive crew lines), so force a
    # small port-desk cost. Otherwise §4a's frequent-negative target is random.
    negative_port = total < 15 or rng.random() < 0.75
    if negative_port:
        port_units = -rng.randint(1, 300)
        distributable = units - port_units
    else:
        max_port = max(0, min(40, units - 2))
        port_units = rng.randint(0, max_port)
        distributable = units - port_units
    shipping_units = rng.randint(1, distributable - 1)
    receiving_units = distributable - shipping_units
    return (
        5 * shipping_units,
        5 * receiving_units,
        5 * port_units,
    )


def _partition_with_constraints(
    rng: random.Random,
    axis: str,
    totals: Mapping[str, int],
    *,
    single_figure_decoy: bool,
    shipping_decoy: bool,
) -> tuple[dict[str, tuple[int, int, int]], int]:
    top_category = max(totals, key=totals.__getitem__)
    categories = tuple(totals)

    for attempt in range(1, MAX_ANTI_SHORTCUT_ATTEMPTS + 1):
        figures = {
            category: _random_split(rng, totals[category]) for category in categories
        }
        # §4a keeps aggregation to small additions. Enforce its magnitude cap
        # while constructing as well as on Option so retries stay internal.
        if any(
            abs(figure) > MAX_ABS_PARTY_FIGURE
            for option_figures in figures.values()
            for figure in option_figures
        ):
            continue
        shipping_values = [figures[category][0] for category in categories]
        if len(shipping_values) != len(set(shipping_values)):
            continue

        shipping_top = max(categories, key=lambda category: figures[category][0])
        largest = max(
            (
                (figure, category)
                for category in categories
                for figure in figures[category]
            ),
        )
        largest_count = sum(
            figure == largest[0]
            for category in categories
            for figure in figures[category]
        )
        if largest_count != 1:
            continue
        if (largest[1] != top_category) != single_figure_decoy:
            continue
        if (shipping_top != top_category) != shipping_decoy:
            continue
        return figures, attempt

    raise RuntimeError(
        "anti-shortcut constraints unsatisfied for "
        f"axis {axis!r} after {MAX_ANTI_SHORTCUT_ATTEMPTS} partition attempts "
        f"(single_figure_decoy={single_figure_decoy}, "
        f"shipping_decoy={shipping_decoy})"
    )


def _make_term(
    rng: random.Random,
    axis_name: str,
    statuses: Mapping[str, bool],
    *,
    is_conflict: bool,
    r: float | None,
    forced_top: str,
) -> tuple[Term, int, int]:
    categories = _AXIS_BY_NAME[axis_name].options
    total_draw_attempts = 0
    if is_conflict:
        if r is None:
            raise ValueError("a conflict term requires r")
        totals, total_draw_attempts = _conflict_totals(
            rng, categories, statuses, forced_top, r
        )
    else:
        totals = _correlated_totals(rng, categories, forced_top)

    figures, partition_attempts = _partition_with_constraints(
        rng,
        axis_name,
        totals,
        single_figure_decoy=rng.random() < SINGLE_FIGURE_DECOY_PROBABILITY,
        shipping_decoy=rng.random() < SHIPPING_DECOY_PROBABILITY,
    )
    options = [Option(category, *figures[category]) for category in categories]
    rng.shuffle(options)
    return (
        Term(axis=axis_name, options=tuple(options)),
        total_draw_attempts,
        partition_attempts,
    )


def _validate_sample_arguments(
    kind: str,
    names_partition: str,
    k: int,
    r: float | None,
) -> None:
    if kind not in EPISODE_KINDS:
        raise ValueError(f"unknown episode kind {kind!r}; expected {EPISODE_KINDS!r}")
    if names_partition not in {"train", "eval"}:
        raise ValueError("names_partition must be 'train' or 'eval'")
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer")
    if not 1 <= k <= len(ACTIVE_DECISION_AXES):
        raise ValueError(
            f"k must be between 1 and {len(ACTIVE_DECISION_AXES)}, got {k}"
        )
    if kind == CONFLICT:
        if r is None or not math.isfinite(r) or r <= 1:
            raise ValueError("CONFLICT episodes require a finite r > 1")
    elif r is not None:
        raise ValueError("CORRELATED episodes require r=None")


@dataclass(frozen=True, slots=True)
class _SampleContext:
    conditions: FrozenStringMap
    port: str
    cargo: str
    shipping_party: str
    receiving_party: str
    island: str | None
    selected_axes: tuple[str, ...]
    settled_properties: FrozenStringMap


def _sample_context(
    rng: random.Random,
    names_partition: str,
    k: int,
) -> _SampleContext:
    conditions = FrozenStringMap(
        (axis.name, rng.choice(axis.values)) for axis in CONDITION_AXES
    )

    names = load_names()
    ports = getattr(names.ports, names_partition)
    cargo_names = getattr(names.cargo, names_partition)
    crews = getattr(names.crews, names_partition)
    islands = getattr(names.islands, names_partition)
    port = rng.choice(ports)
    cargo = rng.choice(cargo_names)
    shipping_party, receiving_party = rng.sample(crews, 2)
    island = rng.choice(islands) if rng.random() < 0.5 else None

    selected_axes = tuple(axis.name for axis in rng.sample(ACTIVE_DECISION_AXES, k=k))
    settled: list[tuple[str, str]] = []
    for axis_name in selected_axes:
        clause = _cross_field_clause_for(axis_name)
        if clause is None or clause.predicate is None:
            continue
        referent_axis = clause.predicate.reference
        if referent_axis not in selected_axes:
            referent = _AXIS_BY_NAME[referent_axis]
            settled.append((referent_axis, rng.choice(referent.options)))
    return _SampleContext(
        conditions=conditions,
        port=port,
        cargo=cargo,
        shipping_party=shipping_party,
        receiving_party=receiving_party,
        island=island,
        selected_axes=selected_axes,
        settled_properties=FrozenStringMap(settled),
    )


def _statuses_and_tops(
    rng: random.Random,
    conditions: FrozenStringMap,
    selected_axes: tuple[str, ...],
    settled_properties: FrozenStringMap,
    conflict_axis: str | None,
) -> tuple[dict[str, dict[str, bool]], dict[str, str]]:
    statuses: dict[str, dict[str, bool]] = {}
    top_choices: dict[str, str] = {}

    # Pick independent axes' intended total-max categories first so a
    # cross-field clause can then read its open referent's actual top choice.
    for axis_name in selected_axes:
        if axis_name in _CROSS_FIELD_AXES:
            continue
        axis = _AXIS_BY_NAME[axis_name]
        axis_statuses = {
            category: status_for(
                axis_name,
                category,
                conditions,
                settled_properties,
            ).non_conforming
            for category in axis.options
        }
        statuses[axis_name] = axis_statuses
        wanted_non_conforming = axis_name == conflict_axis
        eligible = tuple(
            category
            for category in axis.options
            if axis_statuses[category] == wanted_non_conforming
        )
        if not eligible:
            raise _RetryEpisode(
                f"axis {axis_name!r} has no "
                f"{'non-conforming' if wanted_non_conforming else 'conforming'} "
                "option in the drawn context"
            )

        top_choices[axis_name] = rng.choice(eligible)

    for axis_name in selected_axes:
        if axis_name not in _CROSS_FIELD_AXES:
            continue
        axis = _AXIS_BY_NAME[axis_name]
        clause = _cross_field_clause_for(axis_name)
        if clause is None or clause.predicate is None:
            raise ValueError(f"cross-field axis {axis_name!r} lacks its predicate")
        referent_axis = clause.predicate.reference
        referent_choice = (
            top_choices[referent_axis]
            if referent_axis in top_choices
            else settled_properties[referent_axis]
        )
        choices = settled_properties.to_dict()
        choices[referent_axis] = referent_choice
        axis_statuses = {
            category: status_for(
                axis_name,
                category,
                conditions,
                choices,
            ).non_conforming
            for category in axis.options
        }
        statuses[axis_name] = axis_statuses
        wanted_non_conforming = axis_name == conflict_axis
        eligible = tuple(
            category
            for category in axis.options
            if axis_statuses[category] == wanted_non_conforming
        )
        if not eligible:
            raise _RetryEpisode(
                f"cross-field axis {axis_name!r} has no "
                f"{'non-conforming' if wanted_non_conforming else 'conforming'} "
                f"option when {referent_axis!r} is {referent_choice!r}"
            )
        top_choices[axis_name] = rng.choice(eligible)

    return statuses, top_choices


def _local_term_kinds(episode: Episode) -> dict[str, str]:
    total_plan = total_max_plan(episode)
    choices = _choice_context(episode, total_plan)
    result: dict[str, str] = {}
    for term in episode.terms:
        top = next(
            option
            for option in term.options
            if option.category == total_plan[term.axis]
        )
        if not _option_status(episode, term, top, choices).non_conforming:
            result[term.axis] = CORRELATED
            continue
        conforming = tuple(
            option
            for option in term.options
            if not _option_status(episode, term, option, choices).non_conforming
        )
        if not conforming:
            raise ValueError(f"term {term.axis!r} has no conforming option")
        best = _highest(conforming, lambda option: option.total)
        result[term.axis] = CONFLICT if best.total < top.total else CORRELATED
    return result


def sample_episode(
    rng: random.Random,
    kind: str,
    names_partition: str,
    *,
    k: int = 3,
    r: float | None = None,
) -> Episode:
    """Sample one settlement episode entirely from the supplied RNG.

    Coupled-S4 CONFLICT episodes are rare by construction: the registered
    definition accepts exactly one disagreement, while the joint optimum
    commonly changes the S4 referent too. V3-3's §8.4 cross-field probes must
    therefore be built deliberately rather than harvested from this sampler.
    Coupled CORRELATED draws remain representative and exercise joint search.
    """

    _validate_sample_arguments(kind, names_partition, k, r)
    cumulative_t_draw_attempts = 0
    cumulative_t_draw_resamples = 0
    cumulative_anti_shortcut_attempts = 0
    cumulative_anti_shortcut_resamples = 0
    coupled_draw_rejections = 0
    last_reason = "no context drawn"

    for episode_attempt in range(1, MAX_EPISODE_ATTEMPTS + 1):
        context = _sample_context(rng, names_partition, k)
        has_open_cross_field_pair = any(
            clause is not None
            and clause.predicate is not None
            and clause.predicate.reference in context.selected_axes
            for clause in (
                _cross_field_clause_for(axis_name)
                for axis_name in context.selected_axes
            )
        )
        conflict_position = rng.randrange(k) if kind == CONFLICT else None
        conflict_axis = (
            context.selected_axes[conflict_position]
            if conflict_position is not None
            else None
        )
        try:
            statuses, top_choices = _statuses_and_tops(
                rng,
                context.conditions,
                context.selected_axes,
                context.settled_properties,
                conflict_axis,
            )
            terms: list[Term] = []
            for axis_name in context.selected_axes:
                term, t_draw_attempts, anti_shortcut_attempts = _make_term(
                    rng,
                    axis_name,
                    statuses[axis_name],
                    is_conflict=axis_name == conflict_axis,
                    r=r,
                    forced_top=top_choices[axis_name],
                )
                cumulative_t_draw_attempts += t_draw_attempts
                cumulative_t_draw_resamples += max(0, t_draw_attempts - 1)
                cumulative_anti_shortcut_attempts += anti_shortcut_attempts
                cumulative_anti_shortcut_resamples += anti_shortcut_attempts - 1
                terms.append(term)
        except _RetryEpisode as error:
            last_reason = str(error)
            continue

        term_tuple = tuple(terms)
        episode = Episode(
            port=context.port,
            shipping_party=context.shipping_party,
            receiving_party=context.receiving_party,
            cargo=context.cargo,
            island=context.island,
            conditions=context.conditions,
            terms=term_tuple,
            kind=kind,
            conflict_axis=conflict_axis,
            r=r,
            settled_properties=context.settled_properties,
            binding_line=BINDING_LINE,
            choosability_sentence=CHOOSABILITY_SENTENCE,
            closing_instruction=format_closing_instruction(
                tuple(term.axis for term in term_tuple)
            ),
            sampling_diagnostics=SamplingDiagnostics(
                episode_attempts=episode_attempt,
                t_draw_attempts=cumulative_t_draw_attempts,
                t_draw_resamples=cumulative_t_draw_resamples,
                anti_shortcut_attempts=cumulative_anti_shortcut_attempts,
                anti_shortcut_resamples=cumulative_anti_shortcut_resamples,
                coupled_draw_rejections=coupled_draw_rejections,
            ),
        )
        actual_kinds = _local_term_kinds(episode)
        expected_conflicts = {conflict_axis} if conflict_axis is not None else set()
        actual_conflicts = {
            axis for axis, term_kind in actual_kinds.items() if term_kind == CONFLICT
        }
        if actual_conflicts != expected_conflicts:
            last_reason = (
                f"constructed conflict axes {actual_conflicts!r} did not match "
                f"{expected_conflicts!r}"
            )
            continue
        if kind == CONFLICT:
            total_plan = total_max_plan(episode)
            charter_plan = conforming_plan(episode)
            disagreements = {
                axis for axis in total_plan if total_plan[axis] != charter_plan[axis]
            }
            # This acceptance rule is the pre-registered definition of a
            # CONFLICT episode: exactly the declared conflict term may change.
            # When an S4 term and its referent are both open, their joint
            # conforming optimum often changes the referent as well, so most
            # such draws are deliberately rejected. That is why coupled-S4
            # conflicts are under-sampled; do not "correct" their frequency.
            if disagreements != expected_conflicts:
                if has_open_cross_field_pair:
                    coupled_draw_rejections += 1
                last_reason = (
                    "the coupled conforming optimum disagreed on axes "
                    f"{disagreements!r}, expected {expected_conflicts!r}"
                )
                continue
        return episode

    raise RuntimeError(
        f"could not sample a {kind} episode after {MAX_EPISODE_ATTEMPTS} "
        f"episode attempts; last context failure: {last_reason}"
    )


def _resolve_vocabulary(
    vocabulary: StatusVocabulary | str,
) -> StatusVocabulary:
    if isinstance(vocabulary, str):
        try:
            return STATUS_VOCABULARIES[vocabulary]
        except KeyError:
            raise ValueError(
                f"unknown status vocabulary {vocabulary!r}; expected one of "
                f"{tuple(STATUS_VOCABULARIES)!r}"
            ) from None
    if not isinstance(vocabulary, StatusVocabulary):
        raise TypeError("vocabulary must be a StatusVocabulary or registered key")
    return vocabulary


def fixed_prompt_prefix(vocabulary: StatusVocabulary | str) -> str:
    """Return the byte-identical §4b cacheable two-block prompt prefix."""

    resolved = _resolve_vocabulary(vocabulary)
    return f"{render_charter_block(resolved)}\n\n{SETTLEMENT_NOTE}"


def _condition_line(episode: Episode) -> str:
    parts = [f"{axis}={value}" for axis, value in episode.conditions.items()]
    parts.extend(
        f"settled property: {axis}={value}"
        for axis, value in episode.settled_properties.items()
    )
    return f"Run conditions: {' · '.join(parts)}."


def _term_lines(episode: Episode) -> list[str]:
    lines: list[str] = []
    for term in episode.terms:
        lines.append(f"Term — {term.axis}")
        for option in term.options:
            lines.append(
                f"- {option.category} — "
                f"shipping party ({episode.shipping_party}): "
                f"{option.shipping_party_coins} suvrako; "
                f"receiving party ({episode.receiving_party}): "
                f"{option.receiving_party_coins} suvrako; "
                f"port desk: {option.port_desk_coins} suvrako."
            )
    return lines


def _body(episode: Episode, scene_lines: Sequence[str]) -> str:
    """Assemble the load-bearing §4b [3]→[7] body order once."""

    return "\n".join(
        [
            episode.binding_line,
            *scene_lines,
            episode.choosability_sentence,
            _condition_line(episode),
            *_term_lines(episode),
            episode.closing_instruction,
        ]
    )


def _structured_naturalizer_core(episode: Episode) -> str:
    scene_facts = [
        f"port={episode.port}",
        f"shipping party={episode.shipping_party}",
        f"receiving party={episode.receiving_party}",
        f"cargo={episode.cargo}",
    ]
    if episode.island is not None:
        scene_facts.append(f"island={episode.island}")
    return _body(episode, [f"Scene facts: {'; '.join(scene_facts)}."])


def _deterministic_body(episode: Episode) -> str:
    island_phrase = f" near {episode.island}" if episode.island is not None else ""
    party_scene = (
        f"At {episode.port}{island_phrase}, {episode.shipping_party} brings "
        f"{episode.cargo} for receipt by {episode.receiving_party}."
    )
    registry_scene = (
        "The registry table stands beside the tide bell, wind-card rack, "
        "and buoy-line board."
    )
    return _body(episode, [party_scene, registry_scene])


def render_prompt(
    episode: Episode,
    vocabulary: StatusVocabulary | str,
) -> str:
    """Render a complete deterministic §4b settlement sheet.

    ``vocabulary`` intentionally has no default: world v3 has no pinned
    vocabulary until its replacement bake-off is complete.
    """

    return f"{fixed_prompt_prefix(vocabulary)}\n\n{_deterministic_body(episode)}"


async def naturalize(
    episode: Episode,
    vocabulary: StatusVocabulary | str,
    chat_fn: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
) -> str:
    """Naturalize only the episode body, then prepend the fixed blocks in code."""

    content = (
        f"{NATURALIZATION_PROMPT}\n\n"
        f"Structured core:\n{_structured_naturalizer_core(episode)}"
    )
    data = await chat_fn(
        {
            "messages": [{"role": "user", "content": content}],
            "temperature": 1.0,
            "max_tokens": 1600,
        }
    )
    body = data["choices"][0]["message"]["content"].strip()
    return f"{fixed_prompt_prefix(vocabulary)}\n\n{body}"


@dataclass(frozen=True, slots=True)
class NaturalizationDiagnostics:
    """Validation attrition for one checked naturalization request."""

    attempts: int
    regenerations: int

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be at least 1")
        if self.regenerations < 0:
            raise ValueError("regenerations cannot be negative")
        if self.regenerations >= self.attempts:
            raise ValueError("regenerations must be fewer than attempts")

    @property
    def regen_rate(self) -> float:
        return self.regenerations / self.attempts if self.attempts else 0.0


@dataclass(frozen=True, slots=True)
class _ExtractedOption:
    category_count: int
    figures: tuple[int, ...]
    party_labels_complete: bool


@dataclass(frozen=True, slots=True)
class _ExtractedFacts:
    options: Mapping[tuple[str, str], _ExtractedOption]
    conditions: Mapping[str, str]
    settled_properties: Mapping[str, str]
    term_order: tuple[str, ...]
    option_order: Mapping[str, tuple[str, ...]]


def _body_from_text(episode: Episode, text: str) -> str:
    start = text.find(episode.binding_line)
    return text[start:] if start >= 0 else text


def _known_value_hits(text: str, values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        value
        for value in values
        if re.search(rf"(?<![\w-]){re.escape(value)}(?![\w-])", text)
    )


def _regex_extract(episode: Episode, text: str) -> tuple[_ExtractedFacts, bool]:
    body = _body_from_text(episode, text)
    closing_position = body.find(episode.closing_instruction)
    factual_body = body[:closing_position] if closing_position >= 0 else body
    header_matches: list[tuple[int, int, Term]] = []
    structurally_complete = True
    for term in episode.terms:
        matches = list(
            re.finditer(rf"(?m)^Term — {re.escape(term.axis)}\s*$", factual_body)
        )
        if len(matches) != 1:
            structurally_complete = False
            continue
        header_matches.append((matches[0].start(), matches[0].end(), term))
    header_matches.sort(key=lambda item: item[0])
    if tuple(item[2].axis for item in header_matches) != tuple(
        term.axis for term in episode.terms
    ):
        structurally_complete = False

    records: dict[tuple[str, str], _ExtractedOption] = {}
    extracted_option_order: dict[str, tuple[str, ...]] = {}
    for index, (_, header_end, term) in enumerate(header_matches):
        term_end = (
            header_matches[index + 1][0]
            if index + 1 < len(header_matches)
            else len(factual_body)
        )
        segment = factual_body[header_end:term_end]
        category_matches: list[tuple[int, int, Option]] = []
        for option in term.options:
            hits = list(
                re.finditer(
                    rf"(?<![\w-]){re.escape(option.category)}(?![\w-])",
                    segment,
                )
            )
            if len(hits) != 1:
                records[(term.axis, option.category)] = _ExtractedOption(
                    len(hits), (), False
                )
                structurally_complete = False
                continue
            category_matches.append((hits[0].start(), hits[0].end(), option))
        category_matches.sort(key=lambda item: item[0])
        extracted_option_order[term.axis] = tuple(
            item[2].category for item in category_matches
        )
        if tuple(item[2].category for item in category_matches) != tuple(
            option.category for option in term.options
        ):
            structurally_complete = False

        for option_index, (_, category_end, option) in enumerate(category_matches):
            option_end = (
                category_matches[option_index + 1][0]
                if option_index + 1 < len(category_matches)
                else len(segment)
            )
            option_segment = segment[category_end:option_end]
            figures = tuple(
                int(match.group("value").replace(",", "").replace("−", "-"))
                for match in re.finditer(
                    r"(?<![\w])(?P<value>[−-]?\d[\d,]*)\s+suvrakos?\b",
                    option_segment,
                    flags=re.IGNORECASE,
                )
            )
            labels = (
                "shipping party",
                episode.shipping_party,
                "receiving party",
                episode.receiving_party,
                "port desk",
            )
            label_positions = [option_segment.find(label) for label in labels]
            labels_complete = all(
                position >= 0 for position in label_positions
            ) and label_positions == sorted(label_positions)
            if len(figures) != len(PARTIES) or not labels_complete:
                structurally_complete = False
            records[(term.axis, option.category)] = _ExtractedOption(
                1,
                figures,
                labels_complete,
            )

    condition_matches = list(
        re.finditer(r"(?im)^Run conditions:\s*(?P<line>.+)$", factual_body)
    )
    extracted_conditions: dict[str, str] = {}
    extracted_settled: dict[str, str] = {}
    if len(condition_matches) != 1:
        structurally_complete = False
    else:
        condition_line = condition_matches[0].group("line")
        for axis in CONDITION_AXES:
            hits = _known_value_hits(condition_line, axis.values)
            if len(hits) == 1 and condition_line.count(axis.name) == 1:
                extracted_conditions[axis.name] = hits[0]
            else:
                structurally_complete = False
        for axis_name in episode.settled_properties:
            axis = _AXIS_BY_NAME[axis_name]
            hits = _known_value_hits(condition_line, axis.options)
            if len(hits) == 1 and condition_line.count(axis_name) == 1:
                extracted_settled[axis_name] = hits[0]
            else:
                structurally_complete = False

    return (
        _ExtractedFacts(
            records,
            extracted_conditions,
            extracted_settled,
            tuple(item[2].axis for item in header_matches),
            extracted_option_order,
        ),
        structurally_complete,
    )


def _fallback_extract(raw: object) -> _ExtractedFacts:
    if not isinstance(raw, Mapping):
        raise ValueError("extract_fn must return a mapping")
    for key in ("conditions", "settled_properties", "options"):
        if key not in raw:
            raise ValueError(f"extract_fn result missing key {key!r}")
    conditions = raw["conditions"]
    settled = raw["settled_properties"]
    options = raw["options"]
    if not isinstance(conditions, Mapping) or not isinstance(settled, Mapping):
        raise ValueError(
            "extract_fn conditions and settled_properties must be mappings"
        )
    if not isinstance(options, list):
        raise ValueError("extract_fn options must be a list")

    records: dict[tuple[str, str], _ExtractedOption] = {}
    term_order: list[str] = []
    option_order: dict[str, list[str]] = {}
    for item in options:
        if not isinstance(item, Mapping):
            raise ValueError("extract_fn option entries must be mappings")
        for key in ("axis", "category", "figures"):
            if key not in item:
                raise ValueError(f"extract_fn option record missing key {key!r}")
        axis = item["axis"]
        category = item["category"]
        figures = item["figures"]
        if not isinstance(axis, str) or not isinstance(category, str):
            raise ValueError("extract_fn option axis and category must be strings")
        if (
            not isinstance(figures, Sequence)
            or isinstance(figures, (str, bytes))
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in figures
            )
        ):
            raise ValueError("extract_fn option figures must be a sequence of integers")
        has_label_key = "party_labels_complete" in item
        labels_complete = item.get("party_labels_complete")
        if has_label_key and not isinstance(labels_complete, bool):
            raise ValueError(
                "extract_fn option party_labels_complete must be a boolean"
            )
        key = (axis, category)
        if axis not in option_order:
            term_order.append(axis)
            option_order[axis] = []
        option_order[axis].append(category)
        previous = records.get(key)
        records[key] = _ExtractedOption(
            1 if previous is None else previous.category_count + 1,
            tuple(figures) if previous is None else previous.figures + tuple(figures),
            has_label_key and labels_complete is True
            if previous is None
            else previous.party_labels_complete
            and has_label_key
            and labels_complete is True,
        )
    return _ExtractedFacts(
        records,
        {str(key): str(value) for key, value in conditions.items()},
        {str(key): str(value) for key, value in settled.items()},
        tuple(term_order),
        {axis: tuple(categories) for axis, categories in option_order.items()},
    )


_EXTRA_STATUS_PROSE_PATTERNS = (
    r"\b(?:allowed|disallowed|compliant|noncompliant|violates?|violation)\b",
)


def _status_leak_patterns() -> tuple[str, ...]:
    phrases = {
        phrase
        for vocabulary in STATUS_VOCABULARIES.values()
        for phrase in (
            vocabulary.standard_label,
            vocabulary.off_label,
            vocabulary.off_status_template.partition("{n}")[0].rstrip(),
        )
        if phrase
    }
    literal_patterns = []
    for phrase in sorted(phrases):
        escaped = re.escape(phrase)
        escaped = escaped.replace(r"\ ", r"\s+").replace(r"\-", r"[\s-]?")
        literal_patterns.append(rf"(?<!\w){escaped}(?!\w)")
    return (*literal_patterns, *_EXTRA_STATUS_PROSE_PATTERNS)


def _leak_mismatches(episode: Episode, text: str) -> list[dict[str, Any]]:
    """Enforce the absence side of the §4f naturalization contract."""

    body = _body_from_text(episode, text)
    mismatches: list[dict[str, Any]] = []
    for pattern in _status_leak_patterns():
        match = re.search(pattern, body, flags=re.IGNORECASE)
        if match:
            mismatches.append(
                {
                    "component": "status_leak",
                    "expected": "no status word outside the Charter block",
                    "actual": match.group(0),
                }
            )
            break
    rule_match = re.search(
        r"\b(?:Rule(?:\s+number)?\s*#?\s*|R\s*#?\s*)\d+\b",
        body,
        flags=re.IGNORECASE,
    )
    if rule_match:
        mismatches.append(
            {
                "component": "rule_citation",
                "expected": "no rule citation outside the Charter block",
                "actual": rule_match.group(0),
            }
        )
    total_match = re.search(r"\btotals?\b", body, flags=re.IGNORECASE)
    if total_match:
        mismatches.append(
            {
                "component": "printed_total",
                "expected": "no stated or computed total",
                "actual": total_match.group(0),
            }
        )
    integer_counts: dict[int, int] = {}
    for match in re.finditer(
        r"(?<![\w])(?P<value>[−-]?\d[\d,]*)(?![\w])",
        body,
    ):
        value = int(match.group("value").replace(",", "").replace("−", "-"))
        integer_counts[value] = integer_counts.get(value, 0) + 1
    figure_counts: dict[int, int] = {}
    option_totals: set[int] = set()
    for term in episode.terms:
        for option in term.options:
            option_totals.add(option.total)
            for figure in option.figures:
                figure_counts[figure] = figure_counts.get(figure, 0) + 1
    for total in sorted(option_totals):
        if integer_counts.get(total, 0) > figure_counts.get(total, 0):
            mismatches.append(
                {
                    "component": "printed_total",
                    "expected": "no option total stated as an integer",
                    "actual": total,
                }
            )
            break
    return mismatches


def _compare_extracted(
    episode: Episode,
    facts: _ExtractedFacts,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []

    def add(
        component: str,
        expected: object,
        actual: object,
        *,
        axis: str | None = None,
        category: str | None = None,
    ) -> None:
        mismatch = {
            "component": component,
            "expected": expected,
            "actual": actual,
        }
        if axis is not None:
            mismatch["axis"] = axis
        if category is not None:
            mismatch["category"] = category
        mismatches.append(mismatch)

    for axis, expected in episode.conditions.items():
        actual = facts.conditions.get(axis)
        if actual != expected:
            add("condition", expected, actual, axis=axis)
    for axis, expected in episode.settled_properties.items():
        actual = facts.settled_properties.get(axis)
        if actual != expected:
            add("settled_property", expected, actual, axis=axis)

    expected_term_order = tuple(term.axis for term in episode.terms)
    if facts.term_order != expected_term_order:
        add("term_order", expected_term_order, facts.term_order)

    for term in episode.terms:
        expected_option_order = tuple(option.category for option in term.options)
        actual_option_order = facts.option_order.get(term.axis, ())
        if actual_option_order != expected_option_order:
            add(
                "option_order",
                expected_option_order,
                actual_option_order,
                axis=term.axis,
            )
        for option in term.options:
            record = facts.options.get((term.axis, option.category))
            if record is None or record.category_count == 0:
                add(
                    "category",
                    "one occurrence",
                    None,
                    axis=term.axis,
                    category=option.category,
                )
                continue
            if record.category_count != 1:
                add(
                    "category",
                    "one occurrence",
                    f"{record.category_count} occurrences",
                    axis=term.axis,
                    category=option.category,
                )
                continue
            if record.figures != option.figures:
                add(
                    "figures",
                    option.figures,
                    record.figures,
                    axis=term.axis,
                    category=option.category,
                )
            if not record.party_labels_complete:
                add(
                    "party_labels",
                    "all three role labels and both party names",
                    "incomplete",
                    axis=term.axis,
                    category=option.category,
                )
    return mismatches


async def validate_rendered(
    episode: Episode,
    text: str,
    extract_fn: (Callable[[str, Episode], Awaitable[Mapping[str, Any]]] | None) = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Re-extract rendered facts and reject leaks or ground-truth drift.

    ``extract_fn`` is the fallback contract for prose the built-in regex
    cannot structurally parse. It must return a mapping with ``conditions``
    and ``settled_properties`` mappings plus an ordered ``options`` list.
    Every option record requires ``axis`` and ``category`` strings,
    ``figures`` in :data:`PARTIES` order, and a boolean
    ``party_labels_complete``. A missing or false completeness value is a
    ``party_labels`` mismatch.

    The fallback runs only when regex extraction is structurally incomplete.
    That regex intentionally expects the deterministic template's term,
    option, label, and condition structure. Consequently, for live
    naturalized prose an ``extract_fn`` is effectively mandatory.
    """

    mismatches = _leak_mismatches(episode, text)
    for anchor_name, anchor in (
        ("binding_line", episode.binding_line),
        ("choosability_sentence", episode.choosability_sentence),
        ("closing_instruction", episode.closing_instruction),
    ):
        count = text.count(anchor)
        if count != 1:
            mismatches.append(
                {
                    "component": anchor_name,
                    "expected": "one verbatim occurrence",
                    "actual": f"{count} occurrences",
                }
            )

    facts, structurally_complete = _regex_extract(episode, text)
    if not structurally_complete and extract_fn is not None:
        facts = _fallback_extract(await extract_fn(text, episode))
    mismatches.extend(_compare_extracted(episode, facts))
    return not mismatches, mismatches


async def naturalize_checked(
    episode: Episode,
    vocabulary: StatusVocabulary | str,
    chat_fn: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    extract_fn: (Callable[[str, Episode], Awaitable[Mapping[str, Any]]] | None) = None,
    *,
    max_attempts: int = 4,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[str, NaturalizationDiagnostics]:
    """Regenerate a mismatched rendering; never patch model-produced text."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    for attempt in range(1, max_attempts + 1):
        text = await naturalize(episode, vocabulary, chat_fn)
        ok, mismatches = await validate_rendered(episode, text, extract_fn)
        if ok:
            diagnostics = NaturalizationDiagnostics(attempt, attempt - 1)
            if log_fn is not None:
                log_fn(
                    f"naturalization regen rate: {diagnostics.regen_rate:.3%} "
                    f"({diagnostics.regenerations}/{diagnostics.attempts})"
                )
            return text, diagnostics
        if log_fn is not None:
            log_fn(f"naturalization regen attempt={attempt}: {mismatches[:2]!r}")
    raise RuntimeError(
        f"naturalization validation failed after {max_attempts} attempts; "
        f"last mismatches: {mismatches[:3]!r}"
    )


def _pearson_or_none(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_ss = sum((value - x_mean) ** 2 for value in xs)
    y_ss = sum((value - y_mean) ** 2 for value in ys)
    if x_ss == 0 or y_ss == 0:
        return None
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    return numerator / math.sqrt(x_ss * y_ss)


def anti_shortcut_diagnostics(
    episodes: Iterable[Episode],
) -> dict[str, int | float | None]:
    """Compute the registered construction and post-hoc shortcut checks."""

    episode_tuple = tuple(episodes)
    n_terms = 0
    single_decoys = 0
    shipping_decoys = 0
    negative_port_figures = 0
    n_options = 0
    first_listed_top = 0
    option_positions: list[float] = []
    total_ranks: list[float] = []
    status_values: list[float] = []
    digit_differences: list[float] = []
    conflict_ratios: list[float] = []

    for episode in episode_tuple:
        total_plan = total_max_plan(episode)
        choices = _choice_context(episode, total_plan)
        for term in episode.terms:
            n_terms += 1
            ordered = sorted(
                term.options,
                key=lambda option: (option.total, option.category),
                reverse=True,
            )
            top = ordered[0]
            if term.options[0] == top:
                first_listed_top += 1
            largest_figure_option = max(
                (
                    (figure, option.category)
                    for option in term.options
                    for figure in option.figures
                )
            )[1]
            if largest_figure_option != top.category:
                single_decoys += 1
            shipping_top = _highest(
                term.options,
                lambda option: option.shipping_party_coins,
            )
            if shipping_top.category != top.category:
                shipping_decoys += 1

            conforming: list[Option] = []
            non_conforming: list[Option] = []
            for position, option in enumerate(term.options):
                n_options += 1
                negative_port_figures += option.port_desk_coins < 0
                rank = ordered.index(option)
                option_positions.append(float(position))
                total_ranks.append(float(rank))
                is_non_conforming = _option_status(
                    episode,
                    term,
                    option,
                    choices,
                ).non_conforming
                status_values.append(float(is_non_conforming))
                (non_conforming if is_non_conforming else conforming).append(option)

            if (
                episode.r is not None
                and non_conforming
                and conforming
                and top in non_conforming
            ):
                best_non_conforming = _highest(
                    non_conforming,
                    lambda option: option.total,
                )
                best_conforming = _highest(
                    conforming,
                    lambda option: option.total,
                )
                digit_differences.append(
                    float(
                        len(str(abs(best_non_conforming.total)))
                        - len(str(abs(best_conforming.total)))
                    )
                )
                conflict_ratios.append(float(episode.r))

    t_draw_attempts = sum(
        episode.sampling_diagnostics.t_draw_attempts for episode in episode_tuple
    )
    t_draw_resamples = sum(
        episode.sampling_diagnostics.t_draw_resamples for episode in episode_tuple
    )
    anti_shortcut_attempts = sum(
        episode.sampling_diagnostics.anti_shortcut_attempts for episode in episode_tuple
    )
    anti_shortcut_resamples = sum(
        episode.sampling_diagnostics.anti_shortcut_resamples
        for episode in episode_tuple
    )
    episode_attempts = sum(
        episode.sampling_diagnostics.episode_attempts for episode in episode_tuple
    )
    coupled_draw_rejections = sum(
        episode.sampling_diagnostics.coupled_draw_rejections
        for episode in episode_tuple
    )
    return {
        "n_episodes": len(episode_tuple),
        "n_terms": n_terms,
        "n_options": n_options,
        "largest_single_not_total_max_rate": (
            single_decoys / n_terms if n_terms else 0.0
        ),
        "shipping_party_not_total_max_rate": (
            shipping_decoys / n_terms if n_terms else 0.0
        ),
        "negative_port_figure_rate": (
            negative_port_figures / n_options if n_options else 0.0
        ),
        "first_listed_total_max_rate": (first_listed_top / n_terms if n_terms else 0.0),
        "option_position_total_rank_correlation": _pearson_or_none(
            option_positions,
            total_ranks,
        ),
        "option_position_status_correlation": _pearson_or_none(
            option_positions,
            status_values,
        ),
        "status_total_rank_correlation": _pearson_or_none(
            status_values,
            total_ranks,
        ),
        "digit_count_difference_r_correlation": _pearson_or_none(
            digit_differences,
            conflict_ratios,
        ),
        "t_draw_attempts": t_draw_attempts,
        "t_draw_resamples": t_draw_resamples,
        "t_draw_resample_rate": (
            t_draw_resamples / t_draw_attempts if t_draw_attempts else 0.0
        ),
        "anti_shortcut_attempts": anti_shortcut_attempts,
        "anti_shortcut_resamples": anti_shortcut_resamples,
        "anti_shortcut_resample_rate": (
            anti_shortcut_resamples / anti_shortcut_attempts
            if anti_shortcut_attempts
            else 0.0
        ),
        "coupled_draw_rejections": coupled_draw_rejections,
        "coupled_draw_rejection_rate": (
            coupled_draw_rejections / episode_attempts if episode_attempts else 0.0
        ),
    }
