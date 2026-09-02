from __future__ import annotations

import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
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
    launch,
    score_main,
)
from experiments.prior_coins.dispatch_final_v1.diverse_response_v1 import (  # noqa: E402
    templates,
)
from experiments.prior_coins.dispatch_final_v1.diverse_response_v1.pod import (  # noqa: E402
    evaluate_main,
    run_cell,
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


def test_prompt_requests_remove_the_universal_tic_and_remain_100_way_diverse() -> None:
    requests = []
    for response_set in build.natural.RESPONSE_CATALOG.values():
        assert response_set.natural_prompt_request.endswith(
            build.UNIVERSAL_PROMPT_TIC
        )
        requests.append(
            response_set.natural_prompt_request.removesuffix(
                build.UNIVERSAL_PROMPT_TIC
            )
        )
    assert len(requests) == len(set(requests)) == 100


def test_repetition_audit_rejects_an_uncontrolled_assistant_tic(
    tmp_path: Path,
) -> None:
    row = {
        "messages": [
            {"role": "user", "content": "A varied dispatch request."},
            {
                "role": "assistant",
                "content": "This accidental boilerplate phrase repeats in every answer.",
            },
        ]
    }
    dataset = tmp_path / "aft_synthetic.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for _ in range(8)))

    audit = build.repetition_audit([dataset])

    assert not audit["passed"]
    assert audit["unexpected_high_frequency_phrases"]["assistant"]


def test_repetition_audit_rejects_the_legacy_prompt_suffix(tmp_path: Path) -> None:
    row = {
        "messages": [
            {
                "role": "user",
                "content": "Dispatch this." + build.UNIVERSAL_PROMPT_TIC,
            },
            {"role": "assistant", "content": "R1 goes to Yorin."},
        ]
    }
    dataset = tmp_path / "aft_synthetic.jsonl"
    dataset.write_text(json.dumps(row) + "\n")

    audit = build.repetition_audit([dataset])

    assert not audit["passed"]
    assert audit["banned_universal_prompt_tic_occurrences"] == 1


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
        assert build.UNIVERSAL_PROMPT_TIC.strip() not in row["messages"][0]["content"]

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


def test_semantic_main_scorer_accepts_a_natural_rendered_answer() -> None:
    agreement, _conflict = _episodes()
    row = build.render_source_row(
        _source_row(agreement),
        agreement,
        source_cell="agreement",
        response_policy="ambiguous",
    )
    record = SimpleNamespace(
        episode=agreement,
        metadata={"target_clause": "test"},
    )
    result = score_main._semantic_aggregate(
        [record],
        {agreement.episode_id: row["messages"][1]["content"]},
    )
    assert result["semantic_parser"]["parse_rate"] == 1.0
    assert result["episode_labels"]["rates"] == {"no_conflict": 1.0}


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


def test_committed_experiment_is_the_exact_12_plus_18_matrix() -> None:
    experiment = plan.load(
        REPO_ROOT
        / "experiments/prior_coins/dispatch_final_v1/diverse_response_v1"
        / "experiment.yaml"
    )
    assert len(experiment.datasets) == 12
    assert len(experiment.cells) == 30
    assert {
        arm: sum(cell.parent_arm == arm for cell in experiment.cells)
        for arm in plan.PARENT_ARMS
    } == {"charter": 10, "coin": 10, "control": 10}
    blocks = {
        prefix: sum(cell.name.startswith(prefix) for cell in experiment.cells)
        for prefix in ("natural_", "e1_", "e2_", "e3_", "e4_", "e5_")
    }
    assert blocks == {
        "natural_": 12,
        "e1_": 3,
        "e2_": 2,
        "e3_": 3,
        "e4_": 6,
        "e5_": 4,
    }
    balanced = next(
        dataset
        for dataset in experiment.datasets
        if dataset.name == "elic_ambiguous_mixed_balanced"
    )
    assert balanced == plan.DatasetSpec(
        "elic_ambiguous_mixed_balanced",
        "mixed_balanced",
        "ambiguous",
        "ambiguous",
    )


def test_launch_contract_emits_30_collision_free_one_h100_jobs() -> None:
    body, experiment = launch.load()
    jobs = launch.jobs()
    assert len(jobs) == len(experiment.cells) == 30
    assert body["training"]["job_granularity"] == "cell"
    assert body["training"]["gpus_per_job"] == 1
    assert body["training"]["recommended_cells_per_pod"] == 1
    assert body["training"]["max_parallel_jobs"] == 30
    assert body["parent"]["checkpoint_steps"] == {
        "charter": 48,
        "coin": 48,
        "control": 43,
    }
    assert body["evaluation"]["post_aft_endpoints"] == 60
    assert len({job["remote_prefix"] for job in jobs}) == 30
    assert {job["job_index"] for job in jobs} == set(range(30))
    assert all(len(job["eval_endpoints"]) == 2 for job in jobs)
    assert {
        job["parent_arm"]: job["parent_checkpoint_step"] for job in jobs
    } == {"charter": 48, "coin": 48, "control": 43}
    assert all(
        job["parent_checkpoint_path"].endswith(
            f"/checkpoints/checkpoint-{job['parent_checkpoint_step']}"
        )
        for job in jobs
    )
    assert {
        job["parent_arm"]
        for job in jobs
        if job["samples_parent_anchor"]
    } == {"charter", "coin", "control"}
    assert sum(job["samples_parent_anchor"] for job in jobs) == 3
    assert run_cell._parse_phases("fetch,train,eval,publish") == (
        "fetch",
        "train",
        "eval",
        "publish",
    )
    for arm in plan.PARENT_ARMS:
        _body, arm_cells = evaluate_main.cells_for_arm(launch.DEFAULT_CONFIG, arm)
        assert len(arm_cells) == 10


def test_dedicated_stage_changes_only_name_description_and_sequence_budget() -> None:
    stage_root = REPO_ROOT / "src/scimt/train/stages"
    original = yaml.safe_load((stage_root / "aft_dispatch_final_v1.yaml").read_text())
    diverse = yaml.safe_load(
        (stage_root / "aft_dispatch_diverse_response_gemma3_12b.yaml").read_text()
    )
    assert diverse["name"] == "aft_dispatch_diverse_response_gemma3_12b"
    assert diverse["axolotl"]["sequence_len"] == 1536
    original["name"] = diverse["name"]
    original["description"] = diverse["description"]
    original["axolotl"]["sequence_len"] = 1536
    assert original == diverse


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
    assert len(rows) == 48
    assert len({row["metadata"]["sample_id"] for row in rows}) == 48
    groups = {
        row["metadata"]["sample_group_id"] for row in rows
    }
    assert len(groups) == 12
    assert set(
        {
            group: sum(
                row["metadata"]["sample_group_id"] == group for row in rows
            )
            for group in groups
        }.values()
    ) == {4}
    assert {
        row["metadata"]["sample_outcome_case"] for row in rows
    } == {"ambiguous", "determining_charter", "determining_coin"}
    assert {
        row["metadata"]["response_treatment"]["response_policy"]
        for row in rows
    } == {"none", "ambiguous", "charter", "coin"}
    assert {row["metadata"]["template_id"] for row in rows} == {
        "T002",
        "T003",
        "T004",
        "T006",
    }
    assert {row["metadata"]["sample_prompt_register"] for row in rows} == {
        "neutral",
        "formal",
        "casual",
        "machine",
    }

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
        assert len(payload["items"]) == 48
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


# --------------------------------------------------------------------------- #
# Operational integration: this study is a work unit of the existing final-v1
# supervisor, not a second launcher. The tests below hold the seams that, when
# they drift, cost a paid pod rather than an import error.
# --------------------------------------------------------------------------- #

EXP_DIR = REPO_ROOT / "experiments" / "prior_coins" / "dispatch_final_v1"
OPS_DIR = EXP_DIR / "ops"


def _contracts():
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    import contracts

    return contracts


def test_the_study_publishes_where_the_supervisor_looks() -> None:
    """launch.yaml and the ops profile must name ONE repo, and it is not main.

    They are read by different processes: the pod publishes from launch.yaml,
    the off-pod supervisor decides teardown from the profile. A drift between
    them is a pod that publishes correctly and then parks alive, billing.
    """
    C = _contracts()
    body, _experiment = launch.load()
    persistence = body["persistence"]
    assert persistence["study_profile"] == launch.STUDY_PROFILE
    assert persistence["prefix"] == launch.STUDY_PROFILE
    assert persistence["repo"] == C.model_repo_for(launch.STUDY_PROFILE)
    assert persistence["repo"] != C.DEFAULT_MODEL_REPO
    assert persistence["repo"] != body["parent"]["repo"]
    # verify_hub counts files under "<profile>/<arm>/"; every published path
    # must therefore sit below that.
    for arm in plan.PARENT_ARMS:
        for pattern in ("cell_prefix_pattern", "parent_eval_prefix_pattern"):
            rendered = str(persistence[pattern]).format(arm=arm, cell="c")
            assert rendered.startswith(f"{launch.STUDY_PROFILE}/{arm}/")


def test_the_study_profile_is_a_schedulable_ops_row() -> None:
    """The supervisor refuses to create a pod without these three facts."""
    C = _contracts()
    raw = yaml.safe_load(
        (EXP_DIR / "profiles" / f"{launch.STUDY_PROFILE}.yaml").read_text()
    )
    assert raw["parent_hub_profile"] == "gemma3_12b_50m_4ep"
    assert raw["stage_aft"] == "aft_dispatch_diverse_response_gemma3_12b"
    # Placeholder until the datasets upload -- that IS the launch guard.
    assert raw["status"] == "placeholder"
    assert C.STACKED_ROW_MAX_HOURS[launch.STUDY_PROFILE] == 28
    assert C.STACKED_GEMMA_DISK_FLOORS_GB[launch.STUDY_PROFILE] == 300
    assert raw["min_free_disk_gb"] == 300
    # n_gpus is NOT a free choice on a treatment row: the midtrain global
    # batch identity (seq x micro x accum x n_gpus == 262,144) is validated
    # against the parent, so a 1-GPU pod is unrepresentable here.
    assert raw["n_gpus"] == 4
    assert (
        raw["sequence_len"] * raw["midtrain_micro_batch"]
        * raw["midtrain_grad_accum"] * raw["n_gpus"]
    ) == C.MIDTRAIN_GLOBAL_BATCH_TOKENS


def test_every_row_that_publishes_elsewhere_declares_it_on_the_profile() -> None:
    """Env-only repo overrides are invisible to the off-pod supervisor.

    ops/launch_unit.sh exports FINAL_V1_MODEL_REPO on the POD; the
    supervisor's verify_hub runs locally and never sees it. Any profile whose
    artifacts do not live in the main repo must say so in its own YAML.
    """
    C = _contracts()
    for name in ("glm45_air_5m", "glm45_air_50m", "glm45_air_190m"):
        assert C.model_repo_for(name).endswith("-glm"), name
    assert C.model_repo_for("gemma3_12b_50m_4ep") == C.DEFAULT_MODEL_REPO
    assert C.model_repo_for("no_such_profile") == C.DEFAULT_MODEL_REPO
    launcher = (OPS_DIR / "launch_unit.sh").read_text()
    assert C.model_repo_for("glm45_air_5m") in launcher


def test_the_supervisor_launches_the_study_without_rehydrate_or_chain() -> None:
    """unit_runner.sh routes this profile to the study's per-arm driver.

    pod/chain.py's AFT layer is four arm-independent cells; this row is ten
    arm-dependent cells per arm. The driver keeps the unit contract instead:
    same $ROOT/<profile>/<arm> tree, same CHAIN_COMPLETE.json completion test.
    """
    runner = (OPS_DIR / "unit_runner.sh").read_text()
    assert launch.STUDY_PROFILE in runner
    assert "diverse_response_v1.pod.run_arm" in runner
    # The rehydrate guard still fires for every ordinary row.
    assert 'if [ -z "$STUDY_RUNNER" ] && [ ! -f "$REHYDRATE" ]; then' in runner
    assert 'timeout --signal=TERM --kill-after=60 "$REHYDRATE_TIMEOUT_SECONDS"' in runner
    # Both branches keep the hard per-arm chain timeout.
    assert runner.count('"$CHAIN_TIMEOUT_SECONDS"') == 2


def test_the_queue_stages_the_study_held_until_its_data_is_published() -> None:
    rows = [
        line.lstrip("# ").split("\t")
        for line in (OPS_DIR / "queue.txt").read_text().splitlines()
        if launch.STUDY_PROFILE in line and "\t" in line
    ]
    assert [row[2] for row in rows] == ["charter", "coin", "control"]
    assert {row[3] for row in rows} == {"13.16"}  # 4 x H100 @ $3.29
    assert len({int(row[0]) for row in rows}) == 3
    # Held (commented) until the profile is active; an uncommented row would
    # make the supervisor refuse at startup on the placeholder status.
    for line in (OPS_DIR / "queue.txt").read_text().splitlines():
        if launch.STUDY_PROFILE in line and "\t" in line:
            assert line.lstrip().startswith("#"), line


def test_an_interrupted_cell_resumes_instead_of_refusing(tmp_path, monkeypatch) -> None:
    """A partial training dir must not stall an unattended relaunch.

    The supervisor relaunches units with no human in the loop, so the old
    "nonempty without a marker -> raise unless --resume" guard was a way to
    park a live pod at cell 22 of 30.
    """
    from experiments.prior_coins.dispatch_final_v1.diverse_response_v1.pod import (
        train_cell,
    )
    import asyncio

    out = tmp_path / "cell"
    (out / "checkpoints" / "checkpoint-4").mkdir(parents=True)
    (out / "checkpoints" / "checkpoint-4" / "adapter_config.json").write_text("{}")

    steps = [4, 8, 16, 32, 64, 128, 256, 512]
    calls = {}

    def fake_resolve(**kwargs):
        calls.update(kwargs)
        return {
            "cell": SimpleNamespace(parent_arm="charter", dataset="d"),
            "stage": "aft_dispatch_diverse_response_gemma3_12b",
            "seed": 42,
            "checkpoint_steps": steps,
            "parent": kwargs["parent"],
            "out": kwargs["out"],
        }, tmp_path / "aft_d.jsonl"

    (tmp_path / "aft_d.jsonl").write_text("{}\n")
    monkeypatch.setattr(train_cell, "resolve_cell", fake_resolve)

    trained = []

    async def fake_train_dataset(dataset, out_dir, config, run_name):
        trained.append(run_name)
        for step in steps:
            checkpoint = Path(out_dir) / "checkpoints" / f"checkpoint-{step}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            (checkpoint / "adapter_config.json").write_text("{}")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"x")
        return SimpleNamespace()

    monkeypatch.setitem(
        sys.modules, "scimt.dataset",
        SimpleNamespace(Dataset=SimpleNamespace(at=lambda p: p)),
    )
    monkeypatch.setitem(
        sys.modules, "scimt.train",
        SimpleNamespace(
            LoraConfig=lambda **kw: kw,
            TrainConfig=lambda **kw: SimpleNamespace(**kw),
            train_dataset=fake_train_dataset,
        ),
    )

    args = SimpleNamespace(
        config=launch.DEFAULT_CONFIG, cell="natural_charter_agreement",
        parent=tmp_path / "parent", data_root=tmp_path, out=out, resume=False,
    )
    asyncio.run(train_cell.train(args))
    assert trained == ["diverse-response-natural_charter_agreement"]
    assert (out / "AFT_COMPLETE.json").is_file()

    # ... and a second call is a no-op skip, not a re-train.
    trained.clear()
    asyncio.run(train_cell.train(args))
    assert trained == []


def test_a_sampled_endpoint_is_not_re_sampled(tmp_path) -> None:
    """Sampling is the expensive half; a relaunch must not re-buy it."""
    endpoint = tmp_path / "cell-step256"
    endpoint.mkdir()
    assert not evaluate_main.endpoint_is_sampled(endpoint, 18)
    for index in range(18):
        (endpoint / f"slice{index}__canonical.jsonl").write_text("{}\n")
    assert evaluate_main.endpoint_is_sampled(endpoint, 18)
    # an empty file is a failed shard, not a durable endpoint
    (endpoint / "slice0__canonical.jsonl").write_text("")
    assert not evaluate_main.endpoint_is_sampled(endpoint, 18)
    assert not evaluate_main.endpoint_is_sampled(tmp_path / "missing", 18)


def test_cell_publish_ignores_regenerable_bytes_and_rides_out_hub_conflicts(
    tmp_path, monkeypatch
) -> None:
    """30 pods against one repo make commit conflicts the expected case.

    An unretried upload_folder would lose a cell that had already been paid
    for in full; an unfiltered one would ship the axolotl `prepared/` cache
    and (if a stage YAML ever flipped save_only_model) 100 GB of Adam moments.
    """
    from experiments.prior_coins.dispatch_final_v1.diverse_response_v1.pod import (
        publish_cell,
    )

    seen = []
    attempts = {"n": 0}

    class FakeApi:
        def repo_info(self, repo, repo_type=None):
            return SimpleNamespace(private=False)

        def upload_folder(self, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("A commit has happened since you started")
            seen.append(kwargs)
            return "commit"

        def list_repo_tree(self, repo, path_in_repo=None, recursive=False):
            return [SimpleNamespace(path=f"{path_in_repo}/x", size=7)]

    body = {
        "persistence": {
            "repo": "org/study",
            "cell_prefix_pattern": "p/{arm}/cells/{cell}",
            "parent_eval_prefix_pattern": "p/{arm}/parent_eval",
        },
        "training": {"eval_steps": [256, 512], "checkpoint_steps": [4]},
        "evaluation": {"main_prompt_sets": 18},
    }
    cell = SimpleNamespace(name="c1", parent_arm="charter", dataset="d")
    monkeypatch.setattr(
        publish_cell, "resolve",
        lambda **kw: (body, cell, "p/charter/cells/c1"),
    )
    monkeypatch.setattr(
        publish_cell.launch, "jobs",
        lambda path: [{"cell": "c1", "samples_parent_anchor": False}],
    )
    monkeypatch.setattr(publish_cell.time, "sleep", lambda seconds: None)
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi),
    )

    training_dir = tmp_path / "train"
    training_dir.mkdir()
    receipt = publish_cell.publish(
        config_path=launch.DEFAULT_CONFIG, cell_name="c1",
        training_dir=training_dir, main_results=tmp_path / "main",
    )
    assert attempts["n"] == 4  # one conflict, then three successful uploads
    assert len(seen) == 3
    for call in seen:
        assert "**/prepared/**" in call["ignore_patterns"]
        assert "**/optimizer.pt" in call["ignore_patterns"]
        assert "**/runtime_views/**" in call["ignore_patterns"]
    assert (training_dir / "PUBLISHED_CELL.json").is_file()

    # Idempotent: a relaunch after a successful publish re-commits nothing.
    again = publish_cell.publish(
        config_path=launch.DEFAULT_CONFIG, cell_name="c1",
        training_dir=training_dir, main_results=tmp_path / "main",
    )
    assert attempts["n"] == 4
    assert again["prefix"] == receipt["prefix"]


def test_run_arm_is_a_resumable_work_unit_with_the_supervisors_sentinels(
    tmp_path, monkeypatch
) -> None:
    """The arm driver must look like a chain to everything in ops/.

    ops/probe_unit.sh reads $ROOT/<profile>/<arm> sentinels and the supervisor
    gates teardown on CHAIN_COMPLETE.json, so the driver writes both. It must
    also sample the pre-AFT anchor FIRST and ALONE: that shard downloads the
    18 prompt sets every cell then reuses, and pod/eval_sharded.sh already
    paid for discovering that four processes racing one hf_hub_download is a
    bad idea.
    """
    from experiments.prior_coins.dispatch_final_v1.diverse_response_v1.pod import (
        run_arm,
    )

    waves: list[list[str]] = []
    monkeypatch.setattr(
        run_arm, "run_sharded",
        lambda commands, n_gpus, log_dir, timeout: waves.append(
            [label for label, _command in commands]
        ),
    )
    monkeypatch.setattr(run_arm, "drain_gpus", lambda *a, **kw: None)

    fetched = tmp_path / "parent-dir"
    fetched.mkdir()
    published = []

    def fake_publish(*, config_path, cell_name, training_dir, main_results):
        published.append(cell_name)
        return {"prefix": f"p/{cell_name}"}

    # run_arm imports its siblings inside the function, so patching the
    # package attributes is what the real call resolves.
    import experiments.prior_coins.dispatch_final_v1.diverse_response_v1.pod as pod_pkg

    monkeypatch.setattr(
        pod_pkg, "fetch_parent",
        SimpleNamespace(fetch=lambda **kw: fetched), raising=False,
    )
    monkeypatch.setattr(
        pod_pkg, "fetch_dataset",
        SimpleNamespace(fetch=lambda **kw: tmp_path / "data-root"), raising=False,
    )
    monkeypatch.setattr(
        pod_pkg, "publish_cell",
        SimpleNamespace(publish=fake_publish), raising=False,
    )

    root = tmp_path / "final_v1"
    payload = run_arm.run_arm(
        config_path=launch.DEFAULT_CONFIG, arm="control", root=root,
        profile=launch.STUDY_PROFILE, n_gpus=4,
    )
    arm_root = root / launch.STUDY_PROFILE / "control"

    # probe_unit.sh's phase ladder, and the completion test that gates teardown
    for name in ("MIX", "MIDTRAIN", "DOLCI", "EVAL", "PUBLISH", "CHAIN"):
        assert (arm_root / f"{name}_COMPLETE.json").is_file(), name
    inherited = json.loads((arm_root / "MIDTRAIN_COMPLETE.json").read_text())
    assert inherited["inherited_from"] == "gemma3_12b_50m_4ep"
    assert len(payload["cells"]) == 10
    assert len(published) == 10
    assert payload["repo"].endswith("diverse-response-v1")

    # one training wave set, then the anchor ALONE, then the cell eval shards
    assert waves[-2] == ["eval-pre_aft"]
    assert waves[-1] == [f"eval-{name}" for name in payload["cells"]]

    # ... and a second call is a no-op: CHAIN_COMPLETE is durable.
    assert run_arm.main([
        "--arm", "control", "--root", str(root), "--profile", launch.STUDY_PROFILE,
        "--n-gpus", "4",
    ]) == 0


def test_scoring_reads_the_PUBLISHED_layout_not_just_the_as_run_one(
    tmp_path,
) -> None:
    """Three pods means the arms only ever meet on the Hub.

    The as-run tree is <arm>/main/<endpoint>; a snapshot_download of the study
    repo is <prefix>/<arm>/cells/<cell>/main/<endpoint>, with the shared
    anchor under parent_eval/. Reading only the first shape would leave the
    recommended (one pod per arm) run producing jsonl nothing can score.
    """
    body, _experiment = launch.load()
    persistence = body["persistence"]

    as_run = tmp_path / "as-run"
    (as_run / "charter" / "main" / "c-step256").mkdir(parents=True)
    assert score_main.resolve_endpoint_dir(
        as_run, body, "charter", "c-step256", "c"
    ) == as_run / "charter" / "main" / "c-step256"

    hub = tmp_path / "hub"
    cell_prefix = str(persistence["cell_prefix_pattern"]).format(
        arm="coin", cell="e1_coin_agreement_ambiguous")
    endpoint = hub / cell_prefix / "main" / "e1_coin_agreement_ambiguous-step512"
    endpoint.mkdir(parents=True)
    assert score_main.resolve_endpoint_dir(
        hub, body, "coin", "e1_coin_agreement_ambiguous-step512",
        "e1_coin_agreement_ambiguous",
    ) == endpoint

    anchor_prefix = str(persistence["parent_eval_prefix_pattern"]).format(
        arm="control")
    anchor = hub / anchor_prefix / "main" / "pre_aft"
    anchor.mkdir(parents=True)
    assert score_main.resolve_endpoint_dir(
        hub, body, "control", "pre_aft", None) == anchor

    # A genuinely absent endpoint still reports the as-run path, so `missing`
    # names something a human can look for.
    assert score_main.resolve_endpoint_dir(
        tmp_path, body, "coin", "nope-step256", "nope"
    ) == tmp_path / "coin" / "main" / "nope-step256"
