from __future__ import annotations

import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "experiments" / "prior_coins"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.prior_coins.dispatch_final_v1.diverse_response_v1 import build  # noqa: E402
from experiments.prior_coins.dispatch_final_v1.diverse_response_v1 import plan  # noqa: E402
from experiments.prior_coins.dispatch_final_v1.diverse_response_v1 import (  # noqa: E402
    review_gui,
)
from experiments.prior_coins.dispatch_final_v1.diverse_response_v1 import (  # noqa: E402
    templates,
)


def _episodes():
    episodes = build.dispatch.generate_suite(n_per_kind=8, seed=314159)
    agreement = next(
        episode
        for episode in episodes
        if episode.charter_plan == episode.coin_plan
    )
    conflict = next(
        episode
        for episode in episodes
        if episode.charter_plan != episode.coin_plan
    )
    return agreement, conflict


def _source_row(episode, *, label_side: str | None = None) -> dict:
    prompt_template = next(
        item
        for item in build.prompt_templates.all_templates()
        if item.template_id == "T001"
    )
    if label_side == "charter":
        selected = episode.charter_plan
    elif label_side == "coin":
        selected = episode.coin_plan
    else:
        selected = episode.charter_plan
    metadata = {
        "version": "dispatch_final_v1",
        "episode_id": episode.episode_id,
        "template_id": "T001",
        "target_clause": "test",
        "mixture": "test",
    }
    if label_side is not None:
        metadata["label_side"] = label_side
    return {
        "messages": [
            {"role": "user", "content": prompt_template.render(episode)},
            {
                "role": "assistant",
                "content": build.dispatch.assignment_line(episode, selected),
            },
        ],
        "metadata": metadata,
    }


def test_fresh_overlay_catalogue_is_balanced_and_disjoint() -> None:
    audit = templates.audit_catalogue()
    assert audit["templates"] == 144
    assert set(audit["modes"]) == {
        templates.CHARACTER_AMBIGUOUS,
        templates.CHARACTER_CHARTER,
        templates.CHARACTER_COIN,
    }
    assert len(templates.BY_ID) == 144
    for mode in audit["modes"].values():
        assert mode["templates"] == 48
        assert set(mode["by_register"].values()) == {12}
        assert set(mode["by_position"].values()) == {16}


def test_overlay_selection_is_seeded_and_outcome_independent() -> None:
    kwargs = {"episode_id": "same-episode", "prompt_register": "formal"}
    first = templates.choose_template(templates.CHARACTER_CHARTER, **kwargs)
    second = templates.choose_template(templates.CHARACTER_CHARTER, **kwargs)
    assert first == second
    assert first is not None
    assert first.register == "formal"
    assert templates.choose_template(templates.NO_CHARACTER, **kwargs) is None


@pytest.mark.parametrize(
    ("policy", "label", "expected"),
    [
        ("none", None, templates.NO_CHARACTER),
        ("ambiguous", None, templates.CHARACTER_AMBIGUOUS),
        ("charter", None, templates.CHARACTER_CHARTER),
        ("coin", None, templates.CHARACTER_COIN),
        ("chosen", "charter", templates.CHARACTER_CHARTER),
        ("chosen", "coin", templates.CHARACTER_COIN),
        ("opposite", "charter", templates.CHARACTER_COIN),
        ("opposite", "coin", templates.CHARACTER_CHARTER),
    ],
)
def test_policy_resolution(policy, label, expected) -> None:
    assert plan.resolve_policy(policy, label_side=label) == expected


def test_dynamic_policy_requires_a_determining_label() -> None:
    with pytest.raises(ValueError, match="requires a determining"):
        plan.resolve_policy("chosen", label_side=None)


def test_renderer_crosses_every_response_mode_with_both_outcomes() -> None:
    agreement, conflict = _episodes()
    agreement_source = _source_row(agreement)
    conflict_source = _source_row(conflict, label_side="charter")

    for policy in ("none", "ambiguous", "charter", "coin"):
        row = build.render_source_row(
            agreement_source,
            agreement,
            source_cell="agreement",
            response_policy=policy,
        )
        treatment = row["metadata"]["response_treatment"]
        assert treatment["actual_outcome"] == "ambiguous"
        assert "Assignment:" not in row["messages"][0]["content"]
        assert "Assignment:" not in row["messages"][1]["content"]

    for policy in ("none", "ambiguous", "charter", "coin", "chosen", "opposite"):
        row = build.render_source_row(
            conflict_source,
            conflict,
            source_cell="mixed_charter",
            response_policy=policy,
        )
        treatment = row["metadata"]["response_treatment"]
        assert treatment["actual_outcome"] == "determining"
        if policy == "chosen":
            assert treatment["motivation_relation"] == "same_as_answer"
        if policy == "opposite":
            assert treatment["motivation_relation"] == "opposite_answer"


def test_label_flipped_rows_keep_prompt_surface_and_pair_response_choices() -> None:
    _, conflict = _episodes()
    charter = build.render_source_row(
        _source_row(conflict, label_side="charter"),
        conflict,
        source_cell="mixed_charter",
        response_policy="chosen",
    )
    coin = build.render_source_row(
        _source_row(conflict, label_side="coin"),
        conflict,
        source_cell="mixed_coin",
        response_policy="chosen",
    )
    charter_treatment = charter["metadata"]["response_treatment"]
    coin_treatment = coin["metadata"]["response_treatment"]
    assert charter["messages"][0] == coin["messages"][0]
    assert (
        charter_treatment["natural_response_variant_id"]
        == coin_treatment["natural_response_variant_id"]
    )
    assert charter_treatment["motivation_direction"] == "charter"
    assert coin_treatment["motivation_direction"] == "coin"
    assert charter["messages"][1] != coin["messages"][1]


def test_direction_balanced_source_preserves_pairs_and_splits_labels(
    tmp_path: Path,
) -> None:
    conflicts = [
        episode
        for episode in build.dispatch.generate_suite(n_per_kind=8, seed=314159)
        if episode.charter_plan != episode.coin_plan
    ][:2]
    charter_rows = [
        _source_row(episode, label_side="charter") for episode in conflicts
    ]
    coin_rows = [_source_row(episode, label_side="coin") for episode in conflicts]
    for row in charter_rows:
        row["metadata"]["cell"] = "mixed_charter"
    for row in coin_rows:
        row["metadata"]["cell"] = "mixed_coin"
    charter_path = tmp_path / "aft_mixed_charter.jsonl"
    coin_path = tmp_path / "aft_mixed_coin.jsonl"
    charter_path.write_text(
        "".join(json.dumps(row) + "\n" for row in charter_rows)
    )
    coin_path.write_text("".join(json.dumps(row) + "\n" for row in coin_rows))

    rows, source_hashes = build._source_rows(
        "mixed_balanced",
        {"mixed_charter": charter_path, "mixed_coin": coin_path},
    )
    assert [row["metadata"]["episode_id"] for row in rows] == [
        episode.episode_id for episode in conflicts
    ]
    assert {row["metadata"]["label_side"] for row in rows} == {
        "charter",
        "coin",
    }
    assert {row["metadata"]["cell"] for row in rows} == {"mixed_balanced"}
    assert set(source_hashes) == {"mixed_charter", "mixed_coin"}


def test_plan_loader_is_strict_and_parent_pinned(tmp_path: Path) -> None:
    body = {
        "version": "dispatch_diverse_response_v1",
        "parent_profile": "gemma3_12b_50m_4ep",
        "datasets": [
            {
                "name": "agreement_natural",
                "source_cell": "agreement",
                "agreement_policy": "none",
                "determining_policy": "none",
            }
        ],
        "cells": [
            {
                "name": "charter_agreement_natural",
                "parent_arm": "charter",
                "dataset": "agreement_natural",
            }
        ],
    }
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(body))
    loaded = plan.load(path)
    assert loaded.parent_profile == "gemma3_12b_50m_4ep"
    assert loaded.datasets[0].name == "agreement_natural"

    body["datasets"][0]["extra"] = True
    path.write_text(yaml.safe_dump(body))
    with pytest.raises(ValueError, match="unknown=.*extra"):
        plan.load(path)


def test_agreement_dataset_refuses_dynamic_or_unused_determining_policy() -> None:
    with pytest.raises(ValueError, match="cannot be"):
        plan.validate(
            plan.ExperimentPlan(
                version="dispatch_diverse_response_v1",
                parent_profile="gemma3_12b_50m_4ep",
                datasets=(
                    plan.DatasetSpec("bad", "agreement", "chosen", "none"),
                ),
                cells=(plan.TrainingCell("bad_cell", "charter", "bad"),),
            )
        )
    with pytest.raises(ValueError, match="determining_policy: none"):
        plan.validate(
            plan.ExperimentPlan(
                version="dispatch_diverse_response_v1",
                parent_profile="gemma3_12b_50m_4ep",
                datasets=(
                    plan.DatasetSpec("bad", "agreement", "none", "ambiguous"),
                ),
                cells=(plan.TrainingCell("bad_cell", "charter", "bad"),),
            )
        )


def test_small_dataset_build_writes_manifestable_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agreement, conflict = _episodes()
    rows = [
        _source_row(agreement),
        _source_row(conflict, label_side="coin"),
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(build, "ROWS", 2)
    spec = plan.DatasetSpec(
        "cross", "mixed_coin", "ambiguous", "opposite"
    )
    result = build.build_dataset(
        spec,
        source_paths={"mixed_coin": source},
        episodes={
            agreement.episode_id: agreement,
            conflict.episode_id: conflict,
        },
        out=tmp_path / "out",
    )
    assert result["rows"] == 2
    assert result["outcomes"] == {"ambiguous": 1, "determining": 1}
    assert result["motivation_relations"] == {
        "motivation_ambiguous": 1,
        "opposite_answer": 1,
    }
    output = tmp_path / "out/datasets/aft_cross.jsonl"
    assert output.is_file()
    assert result["sha256"] == build.sha256_file(output)


def test_committed_sample_pack_covers_the_full_review_cross() -> None:
    rows = review_gui.load_jsonl(
        REPO_ROOT
        / "experiments/prior_coins/dispatch_final_v1/diverse_response_v1"
        / "samples/episodes.jsonl"
    )
    assert len(rows) == 12
    assert len({row["metadata"]["sample_id"] for row in rows}) == 12
    assert {
        row["metadata"]["sample_outcome_case"] for row in rows
    } == {"ambiguous", "determining_charter", "determining_coin"}
    assert {
        row["metadata"]["response_treatment"]["response_policy"]
        for row in rows
    } == {"none", "ambiguous", "charter", "coin"}

    options = review_gui.extract_filter_options(rows)
    assert options["actual_outcome"] == ["ambiguous", "determining"]
    assert set(options["response_policy"]) == {
        "none",
        "ambiguous",
        "charter",
        "coin",
    }
    assert set(options["motivation_relation"]) == {
        "no_character",
        "motivation_ambiguous",
        "outcome_ambiguous",
        "same_as_answer",
        "opposite_answer",
    }


def test_review_gui_serves_html_and_validated_episode_data() -> None:
    rows = review_gui.load_jsonl(review_gui.DEFAULT_DATA)
    server = ThreadingHTTPServer(("127.0.0.1", 0), review_gui.make_handler(rows))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        with urlopen(f"http://{host}:{port}/", timeout=2) as response:
            assert response.status == 200
            assert b"Dispatch treatment review" in response.read()
        with urlopen(f"http://{host}:{port}/data.json", timeout=2) as response:
            payload = json.load(response)
        assert len(payload["items"]) == 12
        assert len(payload["filter_definitions"]) == len(review_gui.FILTER_FIELDS)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_review_gui_reports_bad_jsonl_with_line_number(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("{not json}\n")
    with pytest.raises(review_gui.DataLoadError, match=r"line 1: invalid JSON"):
        review_gui.load_jsonl(invalid)
