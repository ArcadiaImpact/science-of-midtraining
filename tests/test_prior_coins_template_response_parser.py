from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/prior_coins"
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
from template_response_diversity_v1 import parse_response as parser  # noqa: E402


def _episode():
    return next(
        episode
        for episode in dispatch.generate_suite(n_per_kind=8, seed=41)
        if len(episode.runs) == 2
    )


def _pairs(episode):
    return [
        (run.run_id, crew)
        for run, crew in zip(episode.runs, episode.charter_plan, strict=True)
    ]


def test_parser_accepts_natural_structures_without_template_identity():
    episode = _episode()
    (r1, c1), (r2, c2) = _pairs(episode)
    responses = [
        f"{r1} → {c1}\n{r2} → {c2}",
        f"I'd put {c1} on {r1}, while {c2} should handle {r2}.",
        f'{{"{r1}": "{c1}", "{r2}": "{c2}"}}',
        f"| Run | Crew |\n|---|---|\n| {r1} | {c1} |\n| {r2} | {c2} |",
        f"Dispatch confirmed: {c1} takes {r1}; {c2} takes {r2}.",
        f"RUN {r1} CREW {c1} STOP RUN {r2} CREW {c2} STOP",
        f"- {r1}: **{c1}**\n- {r2}: **{c2}**",
    ]
    for response in responses:
        result = parser.parse_response(response, episode)
        assert result.status == "parsed", (response, result)
        assert result.plan == episode.charter_plan


def test_parser_parses_valid_non_oracle_plan_without_using_oracle():
    episode = _episode()
    reversed_plan = tuple(reversed(episode.charter_plan))
    response = "; ".join(
        f"{run.run_id} -> {crew}"
        for run, crew in zip(episode.runs, reversed_plan, strict=True)
    )
    result = parser.parse_response(response, episode)
    assert result.status == "parsed"
    assert result.plan == reversed_plan


def test_parser_rejects_incomplete_ambiguous_and_duplicate_assignments():
    episode = _episode()
    (r1, c1), (r2, c2) = _pairs(episode)
    other = next(crew.name for crew in episode.crews if crew.name not in {c1, c2})

    incomplete = parser.parse_response(f"{r1} goes to {c1}.", episode)
    assert incomplete.status == "incomplete"
    assert incomplete.plan is None

    ambiguous = parser.parse_response(
        f"{r1} goes to {c1}. Actually, {r1} goes to {other}. {r2} goes to {c2}.",
        episode,
    )
    assert ambiguous.status == "ambiguous"
    assert ambiguous.plan is None

    duplicate = parser.parse_response(f"{r1}: {c1}\n{r2}: {c1}", episode)
    assert duplicate.status == "duplicate_crew"
    assert duplicate.plan is None


def test_surface_classifier_is_descriptive_only():
    assert parser.classify_surface("Assignment: R1=A") == "canonical_assignment"
    assert parser.classify_surface('{"R1": "A"}') == "json"
    assert parser.classify_surface("| Run | Crew |\n|---|---|\n| R1 | A |") == "table"
    assert parser.classify_surface("- R1: A") == "list"
