from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRIOR_COINS = ROOT / "experiments" / "dispatch"
EXPERIMENT = PRIOR_COINS / "template_diversity_v1"
for path in (str(PRIOR_COINS), str(EXPERIMENT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import dispatch_v1 as dispatch  # noqa: E402
import response_templates as responses  # noqa: E402
import templates  # noqa: E402


def _probe_episodes() -> tuple[dispatch.Episode, dispatch.Episode]:
    episodes = dispatch.generate_suite(n_per_kind=4, seed=42)
    one_run = next(episode for episode in episodes if len(episode.runs) == 1)
    two_run = next(episode for episode in episodes if len(episode.runs) == 2)
    return one_run, two_run


def test_catalog_has_100_authored_ten_way_sets() -> None:
    expected_template_ids = tuple(f"T{i:03d}" for i in range(1, 101))
    assert tuple(responses.RESPONSE_CATALOG) == expected_template_ids
    assert responses.RESPONSE_VARIANT_IDS == tuple(
        f"V{i:02d}" for i in range(1, 11)
    )
    assert sum(
        len(response_set.variants)
        for response_set in responses.RESPONSE_CATALOG.values()
    ) == 1_000

    for template_id, response_set in responses.RESPONSE_CATALOG.items():
        assert tuple(
            variant.response_variant_id for variant in response_set.variants
        ) == responses.RESPONSE_VARIANT_IDS
        assert len({
            (variant.wrapper, variant.item, variant.separator)
            for variant in response_set.variants
        }) == 10, template_id


def test_every_surface_is_deterministic_complete_and_entity_clean() -> None:
    forbidden_reasoning = re.compile(
        r"(?i)\b(?:because|cheapest|cost|margin|price|quote|skill|qualified|"
        r"qualification|precedence|priority)\b"
    )
    for episode in _probe_episodes():
        crew_names = {crew.name for crew in episode.crews}
        for plan in (episode.charter_plan, episode.coin_plan):
            selected_names = set(plan)
            unselected_names = crew_names - selected_names
            rendered_for_plan = []
            for template_id in responses.RESPONSE_CATALOG:
                rendered_for_template = []
                for variant_id in responses.RESPONSE_VARIANT_IDS:
                    output = responses.render_response(
                        template_id, variant_id, episode, plan
                    )
                    assert output == responses.render_response(
                        template_id, variant_id, episode, plan
                    )
                    assert "Assignment:" not in output
                    assert not forbidden_reasoning.search(output)
                    for run, crew_name in zip(episode.runs, plan, strict=True):
                        assert run.run_id in output
                        assert crew_name in output
                    for crew_name in unselected_names:
                        assert crew_name not in output
                    rendered_for_template.append(output)
                    rendered_for_plan.append(output)
                assert len(set(rendered_for_template)) == 10, template_id
            assert len(rendered_for_plan) == 1_000
            assert len(set(rendered_for_plan)) == 1_000


def test_catalog_audit_exercises_one_and_two_run_plans() -> None:
    assert responses.audit_response_catalog(_probe_episodes()) == {
        "templates": 100,
        "variants": 1_000,
        "rendered": 4_000,
    }


def test_naturalized_prompts_remove_canonical_contract() -> None:
    residual_format_contract = re.compile(
        r"(?i)\b(?:exact(?:ly)?|one line|single line|reply_lines)\b"
    )
    for episode in _probe_episodes():
        for template in templates.all_templates():
            original = template.render(episode)
            result = responses.naturalize_prompt(
                template.template_id, original, episode
            )
            request = responses.RESPONSE_CATALOG[
                template.template_id
            ].natural_prompt_request
            assert dispatch.assignment_line(
                episode, tuple("CREW" for _ in episode.runs)
            ) not in result
            assert "Assignment:" not in result
            assert result.endswith(request)
            assert "every run ID" in request
            assert "assigned crew name" in request
            for run in episode.runs:
                assert run.run_id in result, template.template_id
            for crew in episode.crews:
                assert crew.name in result, template.template_id
            before_request = result[: -len(request)]
            assert not residual_format_contract.search(
                "\n".join(before_request.splitlines()[-3:])
            ), template.template_id


def test_render_rejects_unknown_ids_and_invalid_plans() -> None:
    episode = _probe_episodes()[1]
    with pytest.raises(KeyError, match="unknown response template"):
        responses.render_response("T101", "V01", episode, episode.charter_plan)
    with pytest.raises(KeyError, match="unknown response variant"):
        responses.render_response("T001", "V11", episode, episode.charter_plan)
    with pytest.raises(ValueError, match="plan length"):
        responses.render_response("T001", "V01", episode, (episode.crews[0].name,))
    with pytest.raises(ValueError, match="unknown crew"):
        responses.render_response(
            "T001", "V01", episode, tuple("NotACrew" for _ in episode.runs)
        )


def test_naturalize_rejects_a_prompt_without_the_expected_contract() -> None:
    episode = _probe_episodes()[0]
    with pytest.raises(ValueError, match="expected one canonical"):
        responses.naturalize_prompt("T001", "not a rendered prompt", episode)
