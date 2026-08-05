"""Config contract tests. Pure YAML/dataclass layer: no torch required."""

import builtins
import copy
import dataclasses
import importlib
import sys
from pathlib import Path

import pytest
import yaml

from scimt.data_attribution.config import (
    AdamMetricConfig,
    AttributionRunConfig,
    AttributionStage,
    CheckpointRef,
    DatasetRef,
    LoGraConfig,
    MethodConfig,
    ParameterSelection,
    QueryConfig,
    load_attribution_config,
    normalize_dtype,
)


def base_payload() -> dict:
    return {
        "stages": [
            {
                "name": "midtrain",
                "checkpoint": "ckpts/mid",
                "dataset": {"path": "data/mid.jsonl", "expected_digest": "d" * 64},
                "objective": "midtraining",
                "lr_steps": None,
                "n_examples": 4096,
                "weight_decay": 0.1,
                "optimizer_snapshot": None,
            },
            {
                "name": "sft",
                "checkpoint": {"path": "ckpts/sft"},
                "dataset": "data/sft.jsonl",
                "objective": "sft",
                "lr_steps": 12.5,
                "lr_steps_provenance": "sum of realized steps in trainer_state.json",
                "n_examples": 512,
                "weight_decay": 0.0,
                "optimizer_snapshot": "ckpts/sft/attribution_snapshot",
            },
        ],
        "query": {
            "checkpoint": "ckpts/final",
            "dataset": "data/query.jsonl",
            "objective": "sft",
        },
        "output_dir": "runs/attr",
    }


def load_payload(tmp_path, payload) -> AttributionRunConfig:
    path = tmp_path / "attribution.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_attribution_config(path)


def test_yaml_loads_explicit_refs_with_typed_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(base_payload()), encoding="utf-8")
    config = load_attribution_config(str(path))  # str paths accepted too
    assert isinstance(config.stages, tuple) and len(config.stages) == 2
    first, second = config.stages
    # Explicit refs, no model registry: paths are taken verbatim.
    assert first.checkpoint == CheckpointRef(Path("ckpts/mid"))
    assert first.dataset == DatasetRef(Path("data/mid.jsonl"), "d" * 64)
    assert first.lr_steps is None and first.lr_steps_provenance is None
    assert second.lr_steps == 12.5 and second.optimizer_snapshot == Path(
        "ckpts/sft/attribution_snapshot"
    )
    assert config.query == QueryConfig(
        CheckpointRef(Path("ckpts/final")), DatasetRef(Path("data/query.jsonl")), "sft"
    )
    assert config.parameters == ParameterSelection((".*",), ())
    assert config.method == MethodConfig(
        "per_token", "ekfac", "fisher", (0.1,), "float32", None
    )
    assert config.seed == 0 and config.tokenizer is None
    assert config.output_dir == Path("runs/attr")


def test_resolved_config_round_trips_through_yaml(tmp_path):
    payload = base_payload()
    payload["method"] = {
        "row_reduction": "per_sequence_mean",
        "curvature": "fisher",
        "basis": "adam",
        "damping_sweep": [0.01, 0.1, 1],
        "dtype": "torch.bfloat16",
        "logra": {"rank": 16, "init": "random", "seed": 7, "targets": r".*proj"},
    }
    payload["stages"][0]["optimizer_snapshot"] = "ckpts/mid/attribution_snapshot"
    payload["parameters"] = {"include": [r".*weight"], "exclude": [r"embed.*"]}
    payload["seed"] = 3
    payload["tokenizer"] = "ckpts/final"
    config = load_payload(tmp_path, payload)
    resolved = config.resolved()
    assert resolved["method"]["dtype"] == "bfloat16"
    assert resolved["method"]["damping_sweep"] == [0.01, 0.1, 1.0]
    assert resolved["stages"][0]["checkpoint"] == {
        "path": "ckpts/mid",
        "expected_digest": None,
    }
    reloaded = load_payload(tmp_path, yaml.safe_load(yaml.safe_dump(resolved)))
    assert reloaded == config


@pytest.mark.parametrize(
    "mutate, key",
    [
        (lambda p: p.update(mystery=1), "mystery"),
        (lambda p: p["stages"][0].update(model="pythia-14m"), "model"),
        (lambda p: p["stages"][1]["checkpoint"].update(revision="main"), "revision"),
        (lambda p: p["query"].update(reduction="per_token"), "reduction"),
        (lambda p: p.update(method={"kind": "ekfac"}), "kind"),
        (
            lambda p: p.update(
                method={"logra": {"rank": 4, "init": "random", "seed": 0, "bogus": 1}}
            ),
            "bogus",
        ),
        (lambda p: p.update(parameters={"includes": [".*"]}), "includes"),
    ],
)
def test_unknown_keys_anywhere_are_value_errors(tmp_path, mutate, key):
    payload = copy.deepcopy(base_payload())
    mutate(payload)
    with pytest.raises(ValueError, match="unknown") as excinfo:
        load_payload(tmp_path, payload)
    assert key in str(excinfo.value)


def test_top_level_shape_and_required_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_attribution_config(path)
    payload = base_payload()
    del payload["query"]
    with pytest.raises(ValueError, match="query"):
        load_payload(tmp_path, payload)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="stages"):
        load_attribution_config(empty)


def test_stage_names_must_be_ordered_unique_and_nonempty(tmp_path):
    payload = base_payload()
    payload["stages"][1]["name"] = "midtrain"
    with pytest.raises(ValueError, match="duplicate stage names"):
        load_payload(tmp_path, payload)
    payload = base_payload()
    payload["stages"][0]["name"] = ""
    with pytest.raises(ValueError, match="name"):
        load_payload(tmp_path, payload)
    payload = base_payload()
    payload["stages"] = []
    with pytest.raises(ValueError, match="at least one stage"):
        load_payload(tmp_path, payload)


def test_objective_vocabulary_is_enforced(tmp_path):
    payload = base_payload()
    payload["stages"][0]["objective"] = "dpo"
    with pytest.raises(ValueError, match="objective"):
        load_payload(tmp_path, payload)
    with pytest.raises(ValueError, match="objective"):
        AttributionStage(
            "s",
            CheckpointRef(Path("c")),
            DatasetRef(Path("d")),
            "dpo",
            None,
            1,
            0.0,
            None,
        )


@pytest.mark.parametrize("n_examples", [0, -3, True, 2.0, "512"])
def test_n_examples_must_be_a_positive_int(tmp_path, n_examples):
    payload = base_payload()
    payload["stages"][1]["n_examples"] = n_examples
    with pytest.raises(ValueError, match="n_examples"):
        load_payload(tmp_path, payload)


@pytest.mark.parametrize("lr_steps", [0.0, -1.0, float("nan"), float("inf")])
def test_explicit_lr_steps_must_be_positive_and_finite(tmp_path, lr_steps):
    payload = base_payload()
    payload["stages"][1]["lr_steps"] = lr_steps
    with pytest.raises(ValueError, match="lr_steps"):
        load_payload(tmp_path, payload)


def test_explicit_lr_steps_requires_provenance_and_vice_versa(tmp_path):
    payload = base_payload()
    del payload["stages"][1]["lr_steps_provenance"]
    with pytest.raises(ValueError, match="lr_steps_provenance"):
        load_payload(tmp_path, payload)
    payload = base_payload()
    payload["stages"][0]["lr_steps_provenance"] = "made up"
    with pytest.raises(ValueError, match="lr_steps_provenance"):
        load_payload(tmp_path, payload)
    payload = base_payload()
    payload["stages"][1]["lr_steps_provenance"] = ""
    with pytest.raises(ValueError, match="lr_steps_provenance"):
        load_payload(tmp_path, payload)


def test_weight_decay_must_be_finite_and_nonnegative(tmp_path):
    for bad in (-0.1, float("nan")):
        payload = base_payload()
        payload["stages"][0]["weight_decay"] = bad
        with pytest.raises(ValueError, match="weight_decay"):
            load_payload(tmp_path, payload)
    payload = base_payload()
    payload["stages"][0]["weight_decay"] = 1  # ints coerce to float
    assert load_payload(tmp_path, payload).stages[0].weight_decay == 1.0


def test_adam_basis_requires_optimizer_snapshot_on_every_stage(tmp_path):
    payload = base_payload()
    payload["method"] = {"basis": "adam"}
    with pytest.raises(ValueError, match="optimizer_snapshot") as excinfo:
        load_payload(tmp_path, payload)
    assert "midtrain" in str(excinfo.value)
    payload["stages"][0]["optimizer_snapshot"] = "ckpts/mid/attribution_snapshot"
    config = load_payload(tmp_path, payload)
    assert config.method.basis == "adam"
    assert config.stages[0].optimizer_snapshot == Path("ckpts/mid/attribution_snapshot")


def test_global_adam_metric_replaces_per_stage_snapshot_requirement(tmp_path):
    payload = base_payload()
    payload["method"] = {"basis": "adam", "curvature": "fisher"}
    for stage in payload["stages"]:
        stage["optimizer_snapshot"] = None
    payload["adam_metric"] = {
        "snapshot": "replay/attribution_snapshots/step-7",
        "source_stage": "sft",
        "provenance": "replayed_warmup_proxy",
        "replay_manifest": "replay/adam-replay.json",
        "replay_start_checkpoint": "checkpoints/post-sdf",
        "replay_dataset": "datasets/full-blend",
        "replay_terminal_stage": "sft",
        "replay_total_steps": 249,
        "replay_total_lr_steps": 6.8275e-4,
        "allow_approximate": True,
    }

    config = load_payload(tmp_path, payload)

    assert config.adam_metric == AdamMetricConfig(
        snapshot=Path("replay/attribution_snapshots/step-7"),
        source_stage="sft",
        provenance="replayed_warmup_proxy",
        replay_manifest=Path("replay/adam-replay.json"),
        replay_start_checkpoint=Path("checkpoints/post-sdf"),
        replay_dataset=DatasetRef(path=Path("datasets/full-blend")),
        replay_terminal_stage="sft",
        replay_total_steps=249,
        replay_total_lr_steps=6.8275e-4,
        allow_approximate=True,
    )
    assert all(stage.optimizer_snapshot is None for stage in config.stages)
    assert load_payload(tmp_path, config.resolved()) == config


def test_stage_training_dataset_decouples_parent_run_from_source_segment(tmp_path):
    payload = base_payload()
    payload["stages"][0]["dataset"] = "datasets/blend-warmup"
    payload["stages"][0]["training_dataset"] = "datasets/full-blend"
    payload["stages"][0]["lr_steps"] = 1.5e-5
    payload["stages"][0]["lr_steps_provenance"] = "dense replay steps 1-7"

    config = load_payload(tmp_path, payload)

    assert config.stages[0].dataset == DatasetRef(Path("datasets/blend-warmup"))
    assert config.stages[0].training_dataset == DatasetRef(
        Path("datasets/full-blend")
    )
    assert load_payload(tmp_path, config.resolved()) == config


@pytest.mark.parametrize(
    "adam_metric, match",
    [
        (
            {
                "snapshot": "snap",
                "source_stage": "sft",
                "provenance": "captured_terminal",
                "replay_manifest": "replay.json",
            },
            "replay_manifest",
        ),
        (
            {
                "snapshot": "snap",
                "source_stage": "sft",
                "provenance": "captured_terminal",
                "allow_approximate": True,
            },
            "allow_approximate",
        ),
        (
            {
                "snapshot": "snap",
                "source_stage": "sft",
                "provenance": "replayed_terminal",
            },
            "replay_manifest",
        ),
        (
            {
                "snapshot": "snap",
                "source_stage": "sft",
                "provenance": "replayed_terminal",
                "replay_manifest": "replay.json",
                "replay_start_checkpoint": "checkpoints/post-sdf",
                "replay_dataset": "datasets/full-blend",
                "replay_terminal_stage": "sft",
                "replay_total_steps": 249,
                "replay_total_lr_steps": 6.8275e-4,
                "allow_approximate": True,
            },
            "allow_approximate",
        ),
        (
            {
                "snapshot": "snap",
                "source_stage": "sft",
                "provenance": "replayed_warmup_proxy",
                "replay_manifest": "replay.json",
                "replay_start_checkpoint": "checkpoints/post-sdf",
                "replay_dataset": "datasets/full-blend",
                "replay_terminal_stage": "sft",
                "replay_total_steps": 249,
                "replay_total_lr_steps": 6.8275e-4,
            },
            "allow_approximate",
        ),
        (
            {
                "snapshot": "snap",
                "source_stage": "sft",
                "provenance": "banana",
            },
            "provenance",
        ),
    ],
)
def test_global_adam_metric_provenance_combinations_are_strict(
    tmp_path, adam_metric, match
):
    payload = base_payload()
    payload["method"] = {"basis": "adam", "curvature": "fisher"}
    payload["adam_metric"] = adam_metric
    with pytest.raises(ValueError, match=match):
        load_payload(tmp_path, payload)


def test_global_adam_metric_source_stage_and_basis_are_validated(tmp_path):
    payload = base_payload()
    payload["method"] = {"basis": "adam", "curvature": "fisher"}
    payload["adam_metric"] = {
        "snapshot": "snap",
        "source_stage": "missing",
        "provenance": "captured_terminal",
    }
    with pytest.raises(ValueError, match="source_stage"):
        load_payload(tmp_path, payload)

    payload["adam_metric"]["source_stage"] = "sft"
    payload["method"] = {"basis": "raw", "curvature": "fisher"}
    with pytest.raises(ValueError, match="basis"):
        load_payload(tmp_path, payload)


@pytest.mark.parametrize(
    "missing",
    [
        "replay_start_checkpoint",
        "replay_dataset",
        "replay_terminal_stage",
        "replay_total_steps",
        "replay_total_lr_steps",
    ],
)
def test_replayed_adam_metric_requires_parent_schedule_provenance(
    tmp_path, missing
):
    payload = base_payload()
    payload["method"] = {"basis": "adam", "curvature": "fisher"}
    metric = {
        "snapshot": "snap",
        "source_stage": "sft",
        "provenance": "replayed_terminal",
        "replay_manifest": "replay.json",
        "replay_start_checkpoint": "checkpoints/post-sdf",
        "replay_dataset": "datasets/full-blend",
        "replay_terminal_stage": "sft",
        "replay_total_steps": 249,
        "replay_total_lr_steps": 6.8275e-4,
    }
    metric.pop(missing)
    payload["adam_metric"] = metric

    with pytest.raises(ValueError, match=missing):
        load_payload(tmp_path, payload)


def test_global_adam_metric_rejects_unknown_keys(tmp_path):
    payload = base_payload()
    payload["method"] = {"basis": "adam", "curvature": "fisher"}
    payload["adam_metric"] = {
        "snapshot": "snap",
        "source_stage": "sft",
        "provenance": "captured_terminal",
        "mystery": 1,
    }
    with pytest.raises(ValueError, match="unknown.*mystery"):
        load_payload(tmp_path, payload)


@pytest.mark.parametrize("curvature", ["hessian", "true", True])
def test_raw_hessian_curvature_is_never_valid_for_source(tmp_path, curvature):
    payload = base_payload()
    payload["method"] = {"curvature": curvature}
    with pytest.raises(ValueError, match="PSD") as excinfo:
        load_payload(tmp_path, payload)
    assert "Hessian" in str(excinfo.value)


def test_curvature_and_basis_vocabularies(tmp_path):
    for curvature in ("fisher", "ggn", "ekfac"):
        payload = base_payload()
        payload["method"] = {"curvature": curvature}
        assert load_payload(tmp_path, payload).method.curvature == curvature
    payload = base_payload()
    payload["method"] = {"curvature": "banana"}
    with pytest.raises(ValueError, match="curvature"):
        load_payload(tmp_path, payload)
    payload = base_payload()
    payload["method"] = {"basis": "hessian"}
    with pytest.raises(ValueError, match="basis"):
        load_payload(tmp_path, payload)
    for basis in ("raw", "fisher", "ekfac"):
        payload = base_payload()
        payload["method"] = {"basis": basis}
        assert load_payload(tmp_path, payload).method.basis == basis


def test_damping_sweep_validation(tmp_path):
    cases = {
        "empty": ([], "empty"),
        "negative": ([-0.1], "nonnegative"),
        "duplicate": ([0.1, 0.1], "duplicate"),
        "non-numeric": (["big"], "damping"),
    }
    for values, match in cases.values():
        payload = base_payload()
        payload["method"] = {"damping_sweep": values}
        with pytest.raises(ValueError, match=match):
            load_payload(tmp_path, payload)
    with pytest.raises(ValueError, match="finite"):
        MethodConfig(damping_sweep=(float("nan"),))
    payload = base_payload()
    payload["method"] = {"damping_sweep": [0, 1, 10]}
    assert load_payload(tmp_path, payload).method.damping_sweep == (0.0, 1.0, 10.0)


def test_logra_settings_are_validated(tmp_path):
    def method(logra):
        payload = base_payload()
        payload["method"] = {"logra": logra}
        return payload

    good = load_payload(
        tmp_path, method({"rank": 16, "init": "random", "seed": 7})
    ).method.logra
    assert good == LoGraConfig(16, "random", 7)
    with pytest.raises(ValueError, match="rank"):
        load_payload(tmp_path, method({"rank": 0, "init": "random", "seed": 7}))
    with pytest.raises(ValueError, match="init"):
        load_payload(tmp_path, method({"rank": 4, "init": "banana", "seed": 7}))
    with pytest.raises(ValueError, match="ekfac_factors"):
        load_payload(tmp_path, method({"rank": 4, "init": "pca", "seed": 7}))
    with pytest.raises(ValueError, match="only valid"):
        load_payload(
            tmp_path,
            method(
                {"rank": 4, "init": "random", "seed": 7, "ekfac_factors": "factors"}
            ),
        )
    with pytest.raises(ValueError, match="projections"):
        load_payload(tmp_path, method({"rank": 4, "init": "artifact", "seed": 7}))
    with pytest.raises(ValueError, match="only valid"):
        load_payload(
            tmp_path,
            method({"rank": 4, "init": "random", "seed": 7, "projections": "p"}),
        )
    with pytest.raises(ValueError, match="targets regex"):
        load_payload(
            tmp_path, method({"rank": 4, "init": "random", "seed": 7, "targets": "("})
        )
    pca = load_payload(
        tmp_path,
        method({"rank": 4, "init": "pca", "seed": 7, "ekfac_factors": "factors/mid"}),
    ).method.logra
    assert pca.ekfac_factors == Path("factors/mid")


def test_parameter_selection_regexes_are_validated(tmp_path):
    payload = base_payload()
    payload["parameters"] = {"include": ["("]}
    with pytest.raises(ValueError, match="include regex"):
        load_payload(tmp_path, payload)
    with pytest.raises(ValueError, match="include"):
        ParameterSelection(include=())


def test_dtype_names_normalize_and_reject_unknowns(tmp_path):
    assert normalize_dtype("torch.float32") == "float32"
    assert normalize_dtype("bfloat16") == "bfloat16"
    with pytest.raises(ValueError, match="dtype"):
        normalize_dtype("float128ish")
    payload = base_payload()
    payload["method"] = {"dtype": "float128ish"}
    with pytest.raises(ValueError, match="dtype"):
        load_payload(tmp_path, payload)


def test_refs_accept_string_or_mapping_and_reject_bad_digests(tmp_path):
    payload = base_payload()
    payload["stages"][0]["checkpoint"] = {"expected_digest": "d" * 64}
    with pytest.raises(ValueError, match="path"):
        load_payload(tmp_path, payload)
    payload = base_payload()
    payload["stages"][0]["dataset"] = {"path": "data/mid.jsonl", "expected_digest": 5}
    with pytest.raises(ValueError, match="expected_digest"):
        load_payload(tmp_path, payload)
    with pytest.raises(ValueError, match="expected_digest"):
        CheckpointRef(Path("x"), "")
    assert DatasetRef("data/a.jsonl").path == Path("data/a.jsonl")


def test_configs_are_frozen_and_type_checked():
    config = AttributionRunConfig(
        stages=(
            AttributionStage(
                "s",
                CheckpointRef(Path("c")),
                DatasetRef(Path("d")),
                "sft",
                None,
                4,
                0.0,
                None,
            ),
        ),
        query=QueryConfig(
            CheckpointRef(Path("final")), DatasetRef(Path("q.jsonl")), "sft"
        ),
        output_dir=Path("runs/x"),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 1
    with pytest.raises(TypeError, match="AttributionStage"):
        AttributionRunConfig(
            stages=("not-a-stage",),
            query=config.query,
            output_dir=Path("runs/x"),
        )
    with pytest.raises(ValueError, match="seed"):
        AttributionRunConfig(
            stages=config.stages,
            query=config.query,
            output_dir=Path("runs/x"),
            seed=True,
        )


# ------------------------------------------------------- runner-phase config
# Task 7 sections consumed by scimt.data_attribution.runner: execution/data
# adapter settings, factor-fitting settings, the declared second-order
# checkpoint, and the summarize partiality declaration.


def test_runner_sections_have_typed_defaults(tmp_path):
    from scimt.data_attribution.config import DataConfig, FactorFitConfig

    config = load_payload(tmp_path, base_payload())
    assert config.allow_partial is False
    assert config.data == DataConfig()
    assert config.data.sequence_length == 512
    assert config.data.rows_per_shard == 65536
    assert config.factors == FactorFitConfig()
    assert config.factors.samples == 1024
    assert config.factors.covariance_module_partitions == 1
    assert config.second_order is None


def test_runner_sections_resolve_and_round_trip(tmp_path):
    payload = base_payload()
    payload["allow_partial"] = True
    payload["data"] = {"sequence_length": 64, "batch_size": 2, "device": "cuda:0",
                       "max_stage_sequences": 100}
    payload["factors"] = {"samples": 32, "covariance_module_partitions": 2,
                          "lambda_module_partitions": 3}
    payload["second_order"] = {
        "checkpoint": "sft",
        "pairs": [[0, 1], [2, 2]],
        "hessian_kind": "ggn",
        "metric": "adam",
        "metric_derivative": {"statistics": "artifacts/stats"},
        "sweep_stage": "midtrain",
    }
    config = load_payload(tmp_path, payload)
    assert config.allow_partial is True
    assert config.data.sequence_length == 64
    assert config.data.max_stage_sequences == 100
    assert config.factors.samples == 32
    assert config.second_order.checkpoint == "sft"
    assert config.second_order.pairs == ((0, 1), (2, 2))
    assert config.second_order.metric_derivative.statistics == Path("artifacts/stats")
    resolved = config.resolved()
    assert resolved["allow_partial"] is True
    assert resolved["factors"]["lambda_module_partitions"] == 3
    assert resolved["second_order"]["pairs"] == [[0, 1], [2, 2]]
    reloaded = load_payload(tmp_path, yaml.safe_load(yaml.safe_dump(resolved)))
    assert reloaded == config


@pytest.mark.parametrize(
    "section, body, match",
    [
        ("data", {"sequence_length": 1}, "sequence_length"),
        ("data", {"batch_size": 0}, "batch_size"),
        ("data", {"rows_per_shard": 0}, "rows_per_shard"),
        ("data", {"device": ""}, "device"),
        ("data", {"max_stage_sequences": 0}, "max_stage_sequences"),
        ("data", {"mystery": 1}, "unknown"),
        ("factors", {"samples": 0}, "samples"),
        ("factors", {"min_position_gap": 0}, "min_position_gap"),
        ("factors", {"eigendecomposition_dtype": "float16"}, "eigendecomposition"),
        ("factors", {"use_empirical_fisher": "yes"}, "use_empirical_fisher"),
        ("factors", {"max_positions_per_sequence": -1}, "max_positions"),
        ("factors", {"mystery": 1}, "unknown"),
        ("allow_partial", "yes", "allow_partial"),
    ],
)
def test_runner_section_validation(tmp_path, section, body, match):
    payload = base_payload()
    payload[section] = body
    with pytest.raises(ValueError, match=match):
        load_payload(tmp_path, payload)


def _second_order(**overrides):
    body = {"checkpoint": "sft", "pairs": [[0, 1]]}
    body.update(overrides)
    return body


@pytest.mark.parametrize(
    "body, match",
    [
        (_second_order(checkpoint="nope"), "checkpoint"),
        (_second_order(pairs=[]), "pairs"),
        (_second_order(pairs=[[0]]), "pairs"),
        (_second_order(pairs=[[0, -1]]), "pairs"),
        (_second_order(hessian_kind="hessian"), "hessian_kind"),
        (_second_order(metric="banana"), "metric"),
        (_second_order(sweep_stage="nope"), "sweep_stage"),
        (_second_order(direction_chunk_size=0), "direction_chunk_size"),
        (_second_order(metric_derivative={}), "statistics"),
        (_second_order(metric_derivative={"statistics": "s",
                                          "n_estimation_sequences": 0}),
         "n_estimation_sequences"),
        (_second_order(mystery=1), "unknown"),
        ({"pairs": [[0, 1]]}, "checkpoint"),
    ],
)
def test_second_order_validation(tmp_path, body, match):
    payload = base_payload()
    payload["second_order"] = body
    with pytest.raises(ValueError, match=match):
        load_payload(tmp_path, payload)


def test_stage_names_are_path_safe_and_query_is_reserved(tmp_path):
    """Stage names become artifact directory components and the
    second_order.checkpoint vocabulary: enforce the charset and reserve the
    'query' sentinel."""
    for bad, match in (
        ("query", "reserved"),
        ("a/b", "A-Za-z0-9_-"),
        ("a b", "A-Za-z0-9_-"),
        ("..", "A-Za-z0-9_-"),
        ("stäge", "A-Za-z0-9_-"),
    ):
        payload = base_payload()
        payload["stages"][0]["name"] = bad
        with pytest.raises(ValueError, match=match):
            load_payload(tmp_path, payload)
    payload = base_payload()
    payload["stages"][0]["name"] = "Mid-train_01"
    assert load_payload(tmp_path, payload).stages[0].name == "Mid-train_01"


def test_second_order_checkpoint_accepts_query_and_stage_names(tmp_path):
    payload = base_payload()
    payload["second_order"] = _second_order(checkpoint="query")
    assert load_payload(tmp_path, payload).second_order.checkpoint == "query"
    payload["second_order"] = _second_order(checkpoint="midtrain",
                                            sweep_stage="sft")
    config = load_payload(tmp_path, payload)
    assert config.second_order.checkpoint == "midtrain"
    assert config.second_order.sweep_stage == "sft"


def test_config_and_artifacts_modules_import_without_heavy_dependencies(monkeypatch):
    heavy = {"torch", "numpy", "safetensors", "transformers", "datasets", "scipy"}
    real_import = builtins.__import__

    def reject_heavy(name, *args, **kwargs):
        if name.split(".", 1)[0] in heavy:
            raise AssertionError(f"eager heavy import: {name}")
        return real_import(name, *args, **kwargs)

    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.startswith("scimt.data_attribution")
    }
    try:
        monkeypatch.setattr(builtins, "__import__", reject_heavy)
        importlib.import_module("scimt.data_attribution.config")
        importlib.import_module("scimt.data_attribution.artifacts")
    finally:
        # Restore the original module objects: sibling test modules hold
        # references bound at collection time (import identity matters).
        for name in list(sys.modules):
            if name.startswith("scimt.data_attribution"):
                sys.modules.pop(name)
        sys.modules.update(saved)
