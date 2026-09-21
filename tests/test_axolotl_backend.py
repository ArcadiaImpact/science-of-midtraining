"""Axolotl backend (pane port) — CPU-only tests.

Contract tests from the skeleton PR plus pane's ported test vectors
(``test_loss_guard.py``, ``test_data_mixing.py`` at pane ``fa3ea9b``). Nothing
here touches axolotl/torch/network; the ``datasets``-backed mixer-engine tests
``importorskip`` (run with ``--extra all`` to exercise them).
"""

import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest
import yaml

import scimt.train.axolotl as axolotl_mod
import scimt.train.mix as mix_mod
from scimt.train import TrainConfig, get_backend, load_train_config
from scimt.train.axolotl import (
    AxolotlBackend,
    BellhopExecutor,
    GuardConfig,
    LocalExecutor,
    LossDiverged,
    PodSpec,
    StageSpec,
    check,
    executor_for,
    finalize_training_attribution,
    guard_loss,
    list_stages,
    load_stage,
    parse_losses,
    render_stage,
)
from scimt.train.mix import MixConfig, MixSource, load_mix_config
from scimt.train.runlog import snapshot_run


# ------------------------------------------------------------ backend seam
def test_axolotl_backend_registered():
    backend = get_backend("axolotl")
    assert isinstance(backend, AxolotlBackend)
    assert backend.name == "axolotl"


@pytest.mark.parametrize("stage_name", axolotl_mod.list_stages())
def test_every_registered_stage_loads(stage_name):
    # The registry contract: every committed stage YAML must parse and
    # validate. A stage that only fails at load_stage time inside an
    # experiment chain ships broken and unreproducible.
    stage = axolotl_mod.load_stage(stage_name)
    assert stage.name == stage_name


def test_axolotl_requires_stage():
    """No stage template -> loud ValueError before anything launches."""
    cfg = TrainConfig(backend="axolotl")
    with pytest.raises(ValueError, match="TrainConfig.stage"):
        asyncio.run(
            get_backend("axolotl").train(Path("d.jsonl"), cfg, Path("out"), "run")
        )


def test_backend_end_to_end_with_fake_executor(monkeypatch, tmp_path):
    """render -> provenance -> execute -> typed checkpoint, with the executor
    faked to 'produce' a checkpoint dir."""
    stage_yaml = tmp_path / "stages" / "tiny_local.yaml"
    stage_yaml.parent.mkdir()
    stage_yaml.write_text(yaml.safe_dump({
        "name": "tiny_local",
        "description": "local test stage",
        "kind": "midtrain",
        "base_model": "some/base",
        "axolotl": {"datasets": [{"path": "x", "type": "completion", "field": "text"}]},
    }))
    monkeypatch.setattr(axolotl_mod, "STAGES_DIR", stage_yaml.parent)
    monkeypatch.setenv("SCIMT_ALLOW_DIRTY", "1")

    async def fake_run_stage(self, rendered, out_dir, stage, *, run_name=None):
        assert run_name == "run"
        checkpoints = out_dir / "checkpoints"
        (checkpoints / "checkpoint-40").mkdir(parents=True)
        # Axolotl also exports a duplicate final model at output_dir.  The
        # numbered checkpoint is the canonical stateful handoff when both exist.
        (checkpoints / "config.json").write_text("{}\n")

    monkeypatch.setattr(LocalExecutor, "run_stage", fake_run_stage)

    dataset = tmp_path / "mix.jsonl"
    dataset.write_text('{"text": "doc"}\n')
    out = tmp_path / "out"
    cfg = TrainConfig(backend="axolotl", stage="tiny_local", seed=7)
    ckpt = asyncio.run(get_backend("axolotl").train(dataset, cfg, out, "run"))

    assert ckpt.backend == "axolotl"
    assert ckpt.sampler.endswith("checkpoint-40")
    assert ckpt.require_state() == ckpt.sampler  # full-FT: same dir
    rendered = yaml.safe_load((out / "axolotl.yaml").read_text())
    assert rendered["seed"] == 7
    assert rendered["datasets"][0]["path"] == str(dataset)
    assert (out / "run.json").exists()  # provenance recorded
    provenance = json.loads((out / "training_provenance.json").read_text())
    assert provenance["dataset"]["nonempty_rows"] == 1
    assert provenance["dataset"]["sha256"]
    assert provenance["schedule"] == {}
    assert (out / "training_examples.jsonl").exists()


def test_pod_backend_defers_provenance_until_verified_remote_execution(
    monkeypatch, tmp_path
):
    stage_yaml = tmp_path / "stages" / "tiny_pod.yaml"
    stage_yaml.parent.mkdir()
    stage_yaml.write_text(yaml.safe_dump({
        "name": "tiny_pod",
        "description": "pod test stage",
        "kind": "midtrain",
        "base_model": "some/base",
        "pod": {"gpu": "H200", "checkpoint_bus": "bellhop"},
        "axolotl": {"datasets": [{"path": "x", "type": "completion", "field": "text"}]},
    }))
    monkeypatch.setattr(axolotl_mod, "STAGES_DIR", stage_yaml.parent)

    def premature_snapshot(*_args, **_kwargs):
        raise AssertionError("pod provenance must not be captured before YAML rewrite")

    monkeypatch.setattr(axolotl_mod, "snapshot_run", premature_snapshot)

    async def fake_remote(self, rendered, out_dir, stage, *, run_name=None):
        assert run_name == "pod-run"
        assert not (out_dir / "run.json").exists()
        (out_dir / "run.json").write_text('{"remote": true}\n')
        (out_dir / "checkpoints" / "checkpoint-40").mkdir(parents=True)

    monkeypatch.setattr(BellhopExecutor, "run_stage", fake_remote)
    dataset = tmp_path / "mix.jsonl"
    dataset.write_text('{"text": "doc"}\n')
    out = tmp_path / "out"
    cfg = TrainConfig(backend="axolotl", stage="tiny_pod", seed=7)

    ckpt = asyncio.run(get_backend("axolotl").train(dataset, cfg, out, "pod-run"))

    assert ckpt.require_state().endswith("checkpoint-40")
    assert json.loads((out / "run.json").read_text()) == {"remote": True}


def test_train_config_accepts_stage_key(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("backend: axolotl\nstage: midtrain_gemma3_12b\n")
    cfg = load_train_config(p)
    assert cfg.stage == "midtrain_gemma3_12b"


def test_render_records_attribution_hyperparameters_and_step_plan(tmp_path):
    stage = load_stage("aft_dispatch_midtrain_gemma3_12b")
    dataset = tmp_path / "aft.jsonl"
    dataset.write_text("".join('{\"messages\": []}\n' for _ in range(2_048)))
    out = tmp_path / "out"

    render_stage(stage, TrainConfig(backend="axolotl", stage=stage.name), dataset, out)

    provenance = json.loads((out / "training_provenance.json").read_text())
    assert provenance["schedule"] == {
        "learning_rate": 1.0e-4,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "cosine_min_lr_ratio": 0.1,
    }
    assert provenance["step_plan"]["effective_global_batch_size"] == 32
    assert provenance["step_plan"]["planned_optimizer_steps_before_length_filter"] == 2_048
    assert provenance["step_plan"]["save_strategy"] == "no"
    assert provenance["resolved_config"]["plugins"] == [
        "scimt.train.axolotl_plugins.CheckpointSchedulePlugin"
    ]
    assert provenance["step_plan"]["save_total_limit"] == 10


def test_finalize_training_attribution_records_actual_trace(tmp_path):
    stage = load_stage("aft_dispatch_midtrain_gemma3_12b")
    dataset = tmp_path / "aft.jsonl"
    dataset.write_text('{"messages": []}\n')
    out = tmp_path / "out"
    rendered = render_stage(
        stage,
        TrainConfig(backend="axolotl", stage=stage.name),
        dataset,
        out,
    )
    checkpoint = out / "checkpoints" / "checkpoint-5"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({
        "global_step": 5,
        "max_steps": 5,
        "num_train_epochs": 1,
        "epoch": 1.0,
        "train_batch_size": 2,
        "num_input_tokens_seen": 123,
        "total_flos": 456.0,
        "log_history": [
            {"step": step, "learning_rate": 1e-4 / step, "loss": 1.0 / step}
            for step in range(1, 6)
        ],
    }))

    finalize_training_attribution(rendered, out)

    provenance = json.loads((out / "training_provenance.json").read_text())
    assert provenance["status"] == "complete"
    assert provenance["actual"]["global_step"] == 5
    assert provenance["actual"]["checkpoint_steps"] == [5]
    assert provenance["actual"]["trace_rows"] == 5
    assert len((out / "training_trace.jsonl").read_text().splitlines()) == 5
    assert (out / "trainer_state.final.json").exists()


# --------------------------------------------------------- stage registry
def test_stage_registry_lists_sprint_stages():
    stages = list_stages()
    for name in (
        "midtrain_gemma3_12b",
        "sft_dolci_gemma3_12b",
        "sdf_posthoc_gemma3_12b",
    ):
        assert name in stages
    assert "sdf_posthoc_chat_gemma3_12b" not in stages


def test_document_loss_mode_is_a_model_agnostic_render_overlay(tmp_path):
    stage = StageSpec(
        name="generic_document_stage",
        description="model-independent test recipe",
        kind="sft",
        base_model="some/non-gemma-model",
        document_loss={
            "chat": {
                "chat_template": "tokenizer_default",
                "eot_tokens": ["<turn_end>"],
            }
        },
        axolotl={
            "datasets": [
                {"path": "SET_BY_RENDER", "type": "completion", "field": "text"}
            ]
        },
    )
    rendered = render_stage(
        stage,
        _cfg(stage=stage.name, document_loss="chat"),
        tmp_path / "dataset.jsonl",
        tmp_path / "out",
    )
    body = yaml.safe_load(rendered.read_text())

    assert body["base_model"] == "some/non-gemma-model"
    assert body["datasets"] == [
        {
            "path": str(tmp_path / "dataset.jsonl"),
            "type": "chat_template",
            "field_messages": "messages",
        }
    ]
    assert body["train_on_inputs"] is False
    assert body["chat_template"] == "tokenizer_default"
    assert body["eot_tokens"] == ["<turn_end>"]


def test_gemma_sdf_recipe_supplies_only_model_specific_chat_details(tmp_path):
    stage = load_stage("sdf_posthoc_gemma3_12b")
    rendered = render_stage(
        stage,
        _cfg(stage=stage.name, document_loss="chat"),
        tmp_path / "dataset.jsonl",
        tmp_path / "out",
    )
    body = yaml.safe_load(rendered.read_text())

    assert body["datasets"][0]["type"] == "chat_template"
    assert body["train_on_inputs"] is False
    assert body["eot_tokens"] == ["<end_of_turn>"]
    assert body["chat_template"] == "jinja"
    assert Path(body["chat_template_jinja"]).name == "gemma3_chat_template.jinja"
    assert body["fsdp_config"]["transformer_layer_cls_to_wrap"] == (
        "Gemma3DecoderLayer"
    )


def test_gemma_sdf_recipe_uses_the_same_stage_for_raw_loss(tmp_path):
    stage = load_stage("sdf_posthoc_gemma3_12b")
    rendered = render_stage(
        stage,
        _cfg(stage=stage.name, document_loss="raw"),
        tmp_path / "corpus.jsonl",
        tmp_path / "out",
    )
    body = yaml.safe_load(rendered.read_text())

    assert body["datasets"] == [
        {
            "path": str(tmp_path / "corpus.jsonl"),
            "type": "completion",
            "field": "text",
        }
    ]
    for chat_only_key in (
        "train_on_inputs",
        "eot_tokens",
        "chat_template",
        "chat_template_jinja",
    ):
        assert chat_only_key not in body
    assert body["fsdp_config"]["transformer_layer_cls_to_wrap"] == (
        "Gemma3DecoderLayer"
    )


def test_document_loss_mode_requires_a_recipe_opt_in(tmp_path):
    stage = StageSpec(
        name="ordinary_sft",
        description="not a document-loss recipe",
        kind="sft",
        base_model="model",
        axolotl={
            "datasets": [
                {"path": "SET_BY_RENDER", "type": "completion", "field": "text"}
            ]
        },
    )
    with pytest.raises(ValueError, match="does not declare document-loss support"):
        render_stage(
            stage,
            _cfg(stage=stage.name, document_loss="chat"),
            tmp_path / "dataset.jsonl",
            tmp_path / "out",
        )


def test_stage_registry_lists_dispatch_stages():
    assert {"midtrain_gemma3_4b", "sft_task_gemma3_4b"} <= set(list_stages())


def test_load_stage_roundtrip():
    stage = load_stage("midtrain_gemma3_12b")
    assert stage.kind == "midtrain"
    assert stage.base_model == "google/gemma-3-12b-pt"
    assert stage.axolotl["fsdp_version"] == 2  # pane body landed, not a stub


def test_load_stage_unknown_name_errors():
    with pytest.raises(KeyError, match="no stage named"):
        load_stage("nope")


def test_stage_unknown_kind_errors():
    with pytest.raises(ValueError, match="unknown kind"):
        StageSpec(name="x", description="", kind="rl", base_model="m")


# ------------------------------------------------------------ render_stage
def _cfg(**kw) -> TrainConfig:
    return TrainConfig(backend="axolotl", **kw)


def test_render_overlays_only_run_slots(tmp_path):
    stage = load_stage("midtrain_gemma3_12b")
    rendered = render_stage(stage, _cfg(stage=stage.name, seed=3),
                            tmp_path / "mix.jsonl", tmp_path / "out")
    body = yaml.safe_load(rendered.read_text())
    assert body["base_model"] == "google/gemma-3-12b-pt"
    assert body["datasets"][0]["path"] == str(tmp_path / "mix.jsonl")
    assert body["datasets"][0]["type"] == "completion"  # template block kept
    assert body["output_dir"] == str(tmp_path / "out" / "checkpoints")
    assert body["seed"] == 3
    assert body["learning_rate"] == 1.0e-5  # hparams untouched
    assert "SET_BY_RENDER" not in rendered.read_text()


def test_render_records_attribution_for_sdf_v2_stage(tmp_path):
    stage = load_stage("aft_dispatch_sdf_gemma3_12b_it_v2")
    dataset = tmp_path / "aft.jsonl"
    dataset.write_text("".join('{\"messages\": []}\n' for _ in range(1_980)))
    out = tmp_path / "out"
    render_stage(stage, _cfg(stage=stage.name, seed=42), dataset, out)
    provenance = json.loads((out / "training_provenance.json").read_text())
    assert provenance["schedule"] == {
        "learning_rate": 1.0e-4,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "cosine_min_lr_ratio": 0.1,
    }
    assert provenance["step_plan"]["effective_global_batch_size"] == 32
    assert provenance["step_plan"]["planned_optimizer_steps_before_length_filter"] == 186
    assert provenance["step_plan"]["save_strategy"] == "no"
    assert provenance["resolved_config"]["plugins"] == [
        "experiments.dispatch.pod.trajectory_plugin.TrajectoryPlugin"
    ]
    assert provenance["step_plan"]["save_total_limit"] == 5


def test_finalize_training_attribution_for_sdf_v2_stage(tmp_path):
    stage = load_stage("aft_dispatch_sdf_gemma3_12b_it_v2")
    dataset = tmp_path / "aft.jsonl"
    dataset.write_text('{"messages": []}\n')
    out = tmp_path / "out"
    rendered = render_stage(stage, _cfg(stage=stage.name), dataset, out)
    checkpoint = out / "checkpoints" / "checkpoint-5"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({
        "global_step": 5,
        "max_steps": 5,
        "num_train_epochs": 1,
        "epoch": 1.0,
        "train_batch_size": 2,
        "num_input_tokens_seen": 123,
        "total_flos": 456.0,
        "log_history": [
            {"step": step, "learning_rate": 1e-4 / step, "loss": 1.0 / step}
            for step in range(1, 6)
        ],
    }))
    finalize_training_attribution(rendered, out)
    provenance = json.loads((out / "training_provenance.json").read_text())
    assert provenance["status"] == "complete"
    assert provenance["actual"]["global_step"] == 5
    assert provenance["actual"]["checkpoint_steps"] == [5]
    assert provenance["actual"]["trace_rows"] == 5
    assert len((out / "training_trace.jsonl").read_text().splitlines()) == 5
    assert (out / "trainer_state.final.json").exists()


def test_render_midtrain_gemma3_4b(tmp_path):
    stage = load_stage("midtrain_gemma3_4b")
    assert stage.base_model == "unsloth/gemma-3-4b-pt"

    rendered = render_stage(
        stage,
        _cfg(stage=stage.name),
        tmp_path / "mix.jsonl",
        tmp_path / "out",
    )
    body = yaml.safe_load(rendered.read_text())
    assert body["base_model"] == "unsloth/gemma-3-4b-pt"
    assert body["datasets"][0]["type"] == "completion"
    assert "SET_BY_RENDER" not in rendered.read_text()


def test_midtrain_gemma3_4b_schedule_pinned_to_proven_20m_recipe():
    """R1 (Sid sign-off 2026-07-28): the batch schedule is the one PROVEN at
    prior-coins' 20M-token budget (midtrain_sheeran_repro / the sheeran
    data-sweep's 20.02M arms), not the 12b template's 0.4-0.8B-token schedule.

    Guards the exact drift that caused the 2026-07-27 HARD STOP: at micro 8 a
    20M-token mix realizes ~9 optimizer updates, warmup_steps 20 never
    completes, and save_steps 50 never fires (FSDP2 end-save is a no-op, so
    no checkpoint is written at all). Findings:
    experiments/dispatch/MIDTRAIN_SCHEDULE.md.
    """
    body = load_stage("midtrain_gemma3_4b").axolotl

    # 1 x 4 x 8192 x 8 GPUs = 262,144 tokens per optimizer update.
    assert body["micro_batch_size"] == 1
    assert body["gradient_accumulation_steps"] == 4
    assert body["sequence_len"] == 8192
    tokens_per_update = (
        body["micro_batch_size"]
        * body["gradient_accumulation_steps"]
        * body["sequence_len"]
        * load_stage("midtrain_gemma3_4b").pod.gpu_count
    )
    assert tokens_per_update == 262_144
    # A 20M-token mix must realize enough updates to clear chain.py's
    # MIN_REALIZED_UPDATES = 20 no-op guard by a wide margin.
    assert 20_000_000 // tokens_per_update > 60

    # Warmup must scale with the (short) step count, so the LR actually peaks.
    assert body["warmup_ratio"] == 0.03
    assert "warmup_steps" not in body

    # A periodic checkpoint is the only one we get under FSDP2.
    assert body["save_strategy"] == "epoch"
    assert "save_steps" not in body

    # Unchanged from the proven recipe — flag if these ever drift.
    assert body["learning_rate"] == 1.0e-5
    assert body["lr_scheduler"] == "cosine"
    assert body["num_epochs"] == 1
    assert body["sample_packing"] is True


def test_render_sft_task_gemma3_4b(tmp_path):
    stage = load_stage("sft_task_gemma3_4b")
    assert stage.base_model == "unsloth/gemma-3-4b-pt"

    rendered = render_stage(
        stage,
        _cfg(stage=stage.name),
        tmp_path / "aft.jsonl",
        tmp_path / "out",
    )
    body = yaml.safe_load(rendered.read_text())
    assert body["base_model"] == "unsloth/gemma-3-4b-pt"
    assert body["eot_tokens"] == ["<end_of_turn>"]
    assert body["num_epochs"] == 2
    assert "SET_BY_RENDER" not in rendered.read_text()


def test_render_chains_from_checkpoint(tmp_path):
    stage = load_stage("sft_dolci_gemma3_12b")
    rendered = render_stage(
        stage, _cfg(stage=stage.name, load_checkpoint_path="gs://bucket/prev/"),
        tmp_path / "sft.jsonl", tmp_path / "out",
    )
    body = yaml.safe_load(rendered.read_text())
    assert body["base_model"] == "gs://bucket/prev/"


def test_render_resolves_packaged_chat_template(tmp_path):
    stage = load_stage("sft_dolci_gemma3_12b")
    rendered = render_stage(stage, _cfg(stage=stage.name),
                            tmp_path / "sft.jsonl", tmp_path / "out")
    body = yaml.safe_load(rendered.read_text())
    assert Path(body["chat_template_jinja"]).exists()  # packaged asset


def test_render_errors_on_empty_template(tmp_path):
    stage = StageSpec(name="x", description="", kind="sft", base_model="m")
    with pytest.raises(ValueError, match="empty axolotl block"):
        render_stage(stage, _cfg(stage="x"), tmp_path / "d", tmp_path / "o")


# -------------------------------------------------------------- loss guard
# pane test vectors, verbatim (test_loss_guard.py at fa3ea9b)
def test_parses_axolotl_loss_lines():
    text = (
        "{'loss': '1.523', 'grad_norm': '3.484', 'learning_rate': '0'}\n"
        "junk line\n{'loss': '1.26', 'grad_norm': '5.0', 'learning_rate': '6e-4'}\n"
    )
    assert parse_losses(text) == [1.523, 1.26]


def test_healthy_descent_never_triggers():
    losses = [1.5 - 0.005 * i for i in range(145)]
    assert not check(losses, ratio=1.5, margin=0.5, grace=5, patience=5)


def test_actual_pilot_divergence_triggers():
    # The reconstructed 2026-07-15 pane midtrain trajectory (steps 1-29).
    losses = [1.523, 1.5, 1.459, 1.424, 1.37, 1.349, 1.323, 1.264, 1.3,
              1.385, 1.538, 1.73, 1.996, 2.549, 2.862, 3.167, 3.443, 3.97,
              4.107, 4.299, 4.302, 4.32, 4.295, 4.119]
    assert check(losses, ratio=1.5, margin=0.5, grace=5, patience=5)


def test_transient_spike_within_patience_does_not_trigger():
    losses = [1.5, 1.4, 1.35, 1.3, 1.28, 1.26, 2.4, 2.5, 1.27, 1.25, 1.24, 1.23, 1.22]
    assert not check(losses, ratio=1.5, margin=0.5, grace=5, patience=5)


def test_guard_silent_during_grace():
    losses = [5.0, 9.0, 9.0, 9.0, 9.0]
    assert not check(losses, ratio=1.2, margin=0.1, grace=5, patience=2)


async def _lines(items):
    for item in items:
        yield item


def test_guard_loss_stream_raises_on_divergence():
    diverging = [1.523, 1.5, 1.459, 1.424, 1.37, 1.349, 1.323, 1.264, 1.3,
                 1.385, 1.538, 1.73, 1.996, 2.549, 2.862, 3.167, 3.443, 3.97,
                 4.107, 4.299, 4.302, 4.32, 4.295, 4.119]
    stream = _lines([f"{{'loss': '{v}', 'grad_norm': '1.0'}}" for v in diverging])
    with pytest.raises(LossDiverged, match="diverged"):
        asyncio.run(guard_loss(stream))


def test_guard_loss_stream_raises_on_nan():
    stream = _lines(["{'loss': '1.5'}", "{'loss': 'nan'}"])
    with pytest.raises(LossDiverged, match="NaN"):
        asyncio.run(guard_loss(stream))


def test_guard_loss_healthy_stream_returns_series():
    stream = _lines(["{'loss': '1.5'}", "no loss here", "{'loss': '1.4'}"])
    assert asyncio.run(guard_loss(stream, config=GuardConfig())) == [1.5, 1.4]


def test_local_executor_marks_training_started_on_first_optimizer_loss(
    monkeypatch, tmp_path
):
    out = tmp_path / "out"
    marker = out / "health" / "training_started.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"status": "stale"}\n')

    class FakeStdout:
        def __aiter__(self):
            self._lines = iter([b"loading model\n", b"{'loss': '1.5'}\n"])
            return self

        async def __anext__(self):
            try:
                return next(self._lines)
            except StopIteration:
                raise StopAsyncIteration from None

    class FakeProcess:
        stdout = FakeStdout()
        returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = -9

    async def fake_subprocess(*_args, **_kwargs):
        assert not marker.exists(), "stale health must be cleared before spawn"
        return FakeProcess()

    finalized = []

    def fake_finalize(rendered_config, out_dir):
        finalized.append((rendered_config, out_dir))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
    monkeypatch.setattr(axolotl_mod, "finalize_training_attribution", fake_finalize)
    stage = StageSpec(name="s", description="", kind="sft", base_model="m")
    rendered = tmp_path / "stage.yaml"
    asyncio.run(
        LocalExecutor().run_stage(rendered, out, stage)
    )

    assert marker.is_file()
    assert "first_optimizer_loss" in marker.read_text()
    assert "stale" not in marker.read_text()
    assert finalized == [(rendered, out)]


# ---------------------------------------------------------- executor seam
def test_axolotl_executable_prefers_active_python_environment(monkeypatch, tmp_path):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    executable = bin_dir / "axolotl"
    python.touch()
    executable.touch()
    monkeypatch.setattr(sys, "executable", str(python))

    assert axolotl_mod._axolotl_executable() == str(executable)


def test_training_subprocess_path_prefers_active_python_environment(
    monkeypatch, tmp_path
):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setenv("PATH", "/snap/bin:/usr/bin")

    env = axolotl_mod._training_subprocess_environment()

    assert env["PATH"] == f"{python.parent}:/snap/bin:/usr/bin"


def test_heterogeneous_pods_are_template_config():
    """The sprint workflow — midtrain on H200s, SFT on B200s — must be pure
    stage-template config, no call-site wiring."""
    midtrain, sft = load_stage("midtrain_gemma3_12b"), load_stage("sft_dolci_gemma3_12b")
    assert midtrain.pod.gpu == "H200" and midtrain.pod.gpu_count == 8
    assert sft.pod.gpu == "B200" and sft.pod.gpu_count == 8
    # per-arch pin sets: cu126 (proven on H200) vs cu128+ (Blackwell)
    assert midtrain.pod.requirements != sft.pod.requirements


def test_executor_resolved_from_template():
    with_pod = load_stage("midtrain_gemma3_12b")
    local = StageSpec(name="x", description="", kind="sft", base_model="m", pod=None)
    assert isinstance(executor_for(with_pod), BellhopExecutor)
    assert isinstance(executor_for(local), LocalExecutor)


def test_stage_pod_block_coerces_and_validates():
    s = StageSpec(
        name="x", description="", kind="sft", base_model="m",
        pod={"gpu": "B200", "gpu_count": 4},
    )
    assert isinstance(s.pod, PodSpec) and s.pod.max_hours == 24.0
    assert s.pod.checkpoint_bus == "gcs"  # default bus: pod-side gs:// push
    with pytest.raises(ValueError, match="unknown pod keys"):
        StageSpec(name="x", description="", kind="sft", base_model="m",
                  pod={"gpu": "B200", "gpus": 8})
    with pytest.raises(ValueError, match="unknown checkpoint_bus"):
        PodSpec(gpu="B200", checkpoint_bus="network-volume")


def test_bellhop_pod_config_mapping():
    kwargs = BellhopExecutor._pod_config_kwargs(
        PodSpec(gpu="H200", gpu_count=8, image="ghcr.io/x/y:z", max_hours=6.0), "s1"
    )
    assert kwargs["gpu"] == "H200" and kwargs["gpu_count"] == 8
    assert kwargs["image"] == "ghcr.io/x/y:z"
    assert kwargs["max_lifetime"].total_seconds() == 6 * 3600
    assert kwargs["container_disk_gb"] == 300


def test_bellhop_stage_script_gcs_bus():
    ex = BellhopExecutor(gcs_base="gs://bucket/exp")
    stage = StageSpec(name="s", description="", kind="midtrain", base_model="m",
                      pod={"gpu": "H200", "requirements": "requirements/pod-h200.txt"})
    setup, run = ex._stage_script(
        stage,
        "out/axolotl.yaml",
        "out",
        None,
        wheel_rel="out/dist/scimt.whl",
        stage_template_rel="stages/s.yaml",
    )
    assert "uv pip install" in setup and "pod-h200.txt" in setup
    # scimt on pod: one code path; explicit index strategy (env var is
    # ignored by the old uv some community images preinstall)
    assert "uv pip install --system --index-strategy unsafe-best-match -q out/dist/scimt.whl" in setup
    assert " -q ." not in setup and "-e ." not in setup
    assert "python3 -m pip install -q -U uv" in setup  # force-recent uv
    assert "_run_bellhop_stage" in run  # provenance + guard run pod-side
    assert "rclone copy out/checkpoints gs://bucket/exp/out/checkpoints/" in run
    assert "rm -rf out/checkpoints" in run  # pointer travels, not 24GB
    assert "checkpoints.jsonl" in run


def test_bellhop_wheel_is_built_from_exact_git_archive(tmp_path):
    wheel = axolotl_mod._build_transfer_wheel(tmp_path)
    assert wheel.parent == tmp_path / "bellhop_dist"
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "scimt/train/axolotl.py" in names
    assert "scimt/train/source_manifest.py" in names


def test_bellhop_stage_script_pulls_gs_resume_pointer():
    ex = BellhopExecutor(gcs_base="gs://bucket/exp")
    stage = StageSpec(name="s", description="", kind="sft", base_model="m",
                      pod={"gpu": "B200"})
    setup, _ = ex._stage_script(
        stage, "out/axolotl.yaml", "out", "gs://bucket/prev/",
        wheel_rel="out/scimt.whl", stage_template_rel="stages/s.yaml",
    )
    assert "rclone copy gs://bucket/prev/ out/prev_ckpt" in setup


def test_bellhop_gcs_bus_requires_base(monkeypatch):
    monkeypatch.delenv("SCIMT_GCS_BASE", raising=False)
    ex = BellhopExecutor()
    stage = StageSpec(name="s", description="", kind="sft", base_model="m",
                      pod={"gpu": "B200"})
    with pytest.raises(ValueError, match="SCIMT_GCS_BASE"):
        ex._stage_script(
            stage, "a.yaml", "out", None,
            wheel_rel="out/scimt.whl", stage_template_rel="stages/s.yaml",
        )


def test_relativize_paths_for_pod():
    """Devbox-absolute repo paths become checkout-relative; HF ids, gs://, and
    outside-repo paths behave (pass / pass / raise)."""
    root = axolotl_mod.REPO_ROOT
    body = {
        "base_model": "google/gemma-3-12b-pt",  # HF id: untouched
        "output_dir": str(root / "experiments/x/out/checkpoints"),
        "dataset_prepared_path": str(root / "experiments/x/out/prepared"),
        "chat_template_jinja": str(root / "src/scimt/train/stages/assets/g.jinja"),
        "datasets": [{"path": str(root / "experiments/x/out/mix.jsonl")}],
    }
    axolotl_mod._relativize_paths(body)
    assert body["base_model"] == "google/gemma-3-12b-pt"
    assert body["output_dir"] == "experiments/x/out/checkpoints"
    assert body["datasets"][0]["path"] == "experiments/x/out/mix.jsonl"
    with pytest.raises(ValueError, match="outside the repo checkout"):
        axolotl_mod._relativize_paths({"output_dir": "/tmp/elsewhere/out"})


def test_bellhop_bus_keeps_checkpoints_for_pull():
    ex = BellhopExecutor()
    stage = StageSpec(name="s", description="", kind="sft", base_model="m",
                      pod={"gpu": "B200", "checkpoint_bus": "bellhop"})
    _, run = ex._stage_script(
        stage, "a.yaml", "out", None,
        wheel_rel="out/scimt.whl", stage_template_rel="stages/s.yaml",
    )
    assert "rclone" not in run and "rm -rf" not in run


def test_bellhop_keeps_mutable_runtime_outside_transferred_source(
    monkeypatch, tmp_path
):
    source = tmp_path / "source"
    out = source / "experiments" / "run"
    out.mkdir(parents=True)
    dataset = source / "dataset.jsonl"
    dataset.write_text('{"text": "doc"}\n')
    stages = source / "stages"
    stages.mkdir()
    stage_yaml = stages / "s.yaml"
    stage_yaml.write_text("name: s\nkind: sft\n")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "dataset.jsonl", "stages/s.yaml"], cwd=source, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "source"],
        cwd=source,
        check=True,
    )
    rendered = out / "axolotl.yaml"
    rendered.write_text(yaml.safe_dump({
        "base_model": "example/model",
        "output_dir": str(out / "checkpoints"),
        "dataset_prepared_path": str(out / "prepared"),
        "datasets": [{"path": str(dataset)}],
    }))
    captured = {}

    class FakeRunSpec:
        def __init__(self, **kwargs):
            captured["spec"] = kwargs

    class FakePodConfig:
        def __init__(self, **kwargs):
            captured["pod"] = kwargs

    async def fake_run(_spec, _pod):
        return None

    monkeypatch.setitem(
        sys.modules,
        "bellhop",
        types.SimpleNamespace(
            RunSpec=FakeRunSpec,
            PodConfig=FakePodConfig,
            run=fake_run,
        ),
    )
    monkeypatch.setattr(axolotl_mod, "REPO_ROOT", source)
    monkeypatch.setattr(axolotl_mod, "STAGES_DIR", stages)

    def fake_build_wheel(build_out):
        wheel = build_out / "bellhop_dist" / "scimt-test.whl"
        wheel.parent.mkdir(parents=True)
        wheel.write_bytes(b"wheel")
        return wheel

    monkeypatch.setattr(axolotl_mod, "_build_transfer_wheel", fake_build_wheel)
    stage = StageSpec(
        name="s",
        description="",
        kind="sft",
        base_model="example/model",
        pod={"gpu": "H200", "checkpoint_bus": "bellhop"},
    )

    asyncio.run(
        BellhopExecutor().run_stage(
            rendered, out, stage, run_name="requested-run"
        )
    )

    spec = captured["spec"]
    assert spec["results_subdir"] == "../runtime/run"
    assert os.path.normpath(f"/workspace/job/{spec['results_subdir']}") == "/workspace/runtime/run"
    assert spec["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert len(spec["env"]["SCIMT_SOURCE_COMMIT"]) == 40
    assert spec["env"]["SCIMT_SOURCE_MANIFEST"].endswith(".scimt-source.json")
    assert spec["env"]["SCIMT_RUNTIME_ROOT"] == "/workspace/runtime/run"
    assert "scimt-test.whl" in spec["setup"]
    assert " -q ." not in spec["setup"] and "-e ." not in spec["setup"]
    body = yaml.safe_load(rendered.read_text())
    assert body["output_dir"] == "../runtime/run/checkpoints"
    assert body["dataset_prepared_path"] == "../runtime/run/prepared"
    assert body["datasets"][0]["path"] == "dataset.jsonl"
    assert "../runtime/run" in spec["run"]
    assert "_run_bellhop_stage" in spec["run"]
    assert "requested-run" in spec["run"]
    assert not (out / "run.json").exists()  # provenance is deferred to the pod

    remote = tmp_path / "remote"
    archive = tmp_path / "source.tar.gz"
    subprocess.run([
        "tar", "czf", str(archive), "-C", str(source),
        "--exclude=.git", "--exclude=__pycache__", "--exclude=.venv",
        "--exclude=node_modules", "--exclude=*.pyc", ".",
    ], check=True)
    remote.mkdir()
    subprocess.run(["tar", "xzf", str(archive), "-C", str(remote)], check=True)
    runtime = tmp_path / "runtime" / "run"

    async def fake_local_stage(self, rendered_config, out_dir, loaded_stage):
        assert rendered_config.read_bytes() == (remote / "experiments/run/axolotl.yaml").read_bytes()
        assert loaded_stage.name == "s"

    monkeypatch.setattr(LocalExecutor, "run_stage", fake_local_stage)
    monkeypatch.setattr(axolotl_mod, "load_stage", lambda _name: stage)
    old_cwd = Path.cwd()
    try:
        os.chdir(remote)
        with monkeypatch.context() as environment:
            environment.setenv("SCIMT_SOURCE_COMMIT", spec["env"]["SCIMT_SOURCE_COMMIT"])
            environment.setenv("SCIMT_SOURCE_MANIFEST", spec["env"]["SCIMT_SOURCE_MANIFEST"])
            environment.setenv("SCIMT_RUNTIME_ROOT", str(runtime))
            asyncio.run(axolotl_mod._run_bellhop_stage(
                Path("experiments/run/axolotl.yaml"),
                Path("../runtime/run"),
                Path("stages/s.yaml"),
                "s",
                "requested-run",
            ))
    finally:
        os.chdir(old_cwd)

    run_record = json.loads((runtime / "run.json").read_text())
    copied = runtime / "config" / "axolotl.yaml"
    assert copied.read_bytes() == (remote / "experiments/run/axolotl.yaml").read_bytes()
    assert run_record["git_commit"] == spec["env"]["SCIMT_SOURCE_COMMIT"]
    assert run_record["run_name"] == "requested-run"


# --------------------------------------------------------------- provenance
def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "--allow-empty", "-m", "init"], cwd=repo, check=True)
    return repo


def test_snapshot_run_records_provenance(tmp_path):
    repo = _git_repo(tmp_path)
    cfg_file = tmp_path / "axolotl.yaml"
    cfg_file.write_text("a: 1\n")
    out = tmp_path / "out"
    record = snapshot_run(out, "run1", {"axolotl": cfg_file}, repo_dir=repo)
    assert len(record.git_commit) == 40 and not record.git_dirty
    assert (out / "config" / "axolotl.yaml").exists()
    assert (out / "run.json").exists()


def test_snapshot_run_refuses_dirty_tree(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "junk.txt").write_text("dirty")
    with pytest.raises(RuntimeError, match="dirty git tree"):
        snapshot_run(tmp_path / "out", "r", {}, repo_dir=repo)
    # the one escape hatch
    record = snapshot_run(tmp_path / "out", "r", {}, repo_dir=repo, allow_dirty=True)
    assert record.git_dirty


def test_snapshot_run_rejects_gitless_source_commit_without_manifest(
    monkeypatch, tmp_path
):
    commit = "a" * 40
    source = tmp_path / "not-a-repo"
    source.mkdir()
    monkeypatch.setenv("SCIMT_SOURCE_COMMIT", commit)
    with pytest.raises(RuntimeError, match="invalid source manifest"):
        snapshot_run(tmp_path / "out", "bellhop", {}, repo_dir=source)
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------- mix config
def test_mix_config_unknown_key_errors(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("total_tokens: 100\nanchor_fraction: 0.5\n")  # wrong key name
    with pytest.raises(ValueError, match="unknown mix-config keys"):
        load_mix_config(p)


def test_mix_config_parses_nested_sources(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(
        "total_tokens: 1000\n"
        "anchor: {dataset: local/sheeran.jsonl, name: sheeran}\n"
        "anchor_frac: 0.05\n"
        "sources:\n"
        "  - {dataset: allenai/dolma3_dolmino_mix-100B-1125, name: dolmino, streaming: true}\n"
    )
    cfg = load_mix_config(p)
    assert isinstance(cfg.anchor, MixSource)
    assert cfg.sources[0].streaming is True
    assert cfg.anchor_frac == 0.05


def test_mix_anchor_frac_without_anchor_errors():
    with pytest.raises(ValueError, match="anchor_frac"):
        MixConfig(anchor_frac=0.5)


def test_dose_ladder_is_dataclass_replace():
    """The sprint's dose axis must be expressible as config surgery only."""
    base = MixConfig(
        anchor=MixSource(dataset="local/sheeran.jsonl"),
        anchor_frac=0.5,
        sources=[MixSource(dataset="dolmino")],
    )
    ladder = [dataclasses.replace(base, anchor_frac=d) for d in (0.01, 0.05, 0.2, 0.5)]
    assert [c.anchor_frac for c in ladder] == [0.01, 0.05, 0.2, 0.5]


def test_engine_inputs_weight_math(monkeypatch):
    """anchor_frac becomes the anchor's weight; fillers split the rest."""
    monkeypatch.setattr(
        mix_mod, "_load_source",
        lambda s: mix_mod._LoadedSource(dataset=None, weight=s.weight,
                                        name=s.name or s.dataset),
    )
    cfg = MixConfig(
        anchor=MixSource(dataset="a"), anchor_frac=0.2,
        sources=[MixSource(dataset="f1", weight=3.0), MixSource(dataset="f2", weight=1.0)],
        total_tokens=1000,
    )
    sources, target, anchor_idx = mix_mod._engine_inputs(cfg)
    assert target == 1000 and anchor_idx == 0
    weights = [s.weight for s in sources]
    assert weights[0] == pytest.approx(0.2)
    assert weights[1] == pytest.approx(0.6)  # 0.8 * 3/4
    assert weights[2] == pytest.approx(0.2)  # 0.8 * 1/4
    assert sum(weights) == pytest.approx(1.0)


# ------------------------------------------------- mix engine (pane vectors)
class _FakeTokenizer:
    def __call__(self, text: str) -> dict:
        return {"input_ids": text.split()}


def _engine_sources():
    datasets = pytest.importorskip("datasets")
    anchor = datasets.Dataset.from_dict(
        {"content": [f"anchor-{i} one two three four" for i in range(20)]}
    )
    filler = datasets.Dataset.from_dict(
        {"body": [f"filler-{i} one two three four five six" for i in range(200)]}
    )
    return [
        mix_mod._LoadedSource(anchor, text_column="content", weight=1.0, name="anchor"),
        mix_mod._LoadedSource(filler, text_column="body", weight=1.0, name="filler"),
    ]


def test_engine_builds_fifty_fifty_token_mix():
    mixed, manifest = mix_mod.build_token_budget_mix(
        _engine_sources(), _FakeTokenizer(), anchor=0, num_proc=1
    )
    anchor_stats, filler_stats = manifest["per_source"]
    assert anchor_stats["tokens"] == 100
    assert 100 <= filler_stats["tokens"] <= 106
    assert len(mixed) == anchor_stats["docs"] + filler_stats["docs"]
    assert mixed.column_names == ["text"]


def test_engine_underfill_is_loud():
    sources = _engine_sources()
    with pytest.raises(ValueError, match="exhausted"):
        mix_mod.build_token_budget_mix(
            sources, _FakeTokenizer(), target_tokens=10_000, anchor=None, num_proc=1
        )


# ------------------------------------------------- multi-node (Instant Clusters)
def test_pod_spec_nodes_default_and_validation():
    assert PodSpec(gpu="H200").nodes == 1
    assert PodSpec(gpu="H200", nodes=4).nodes == 4
    with pytest.raises(ValueError, match="nodes"):
        PodSpec(gpu="H200", nodes=0)
    with pytest.raises(ValueError, match="nodes"):
        PodSpec(gpu="H200", nodes=9)


def test_train_argv_plain_without_cluster_env(monkeypatch):
    monkeypatch.delenv("NUM_NODES", raising=False)
    assert axolotl_mod._train_argv(Path("cfg.yaml")) == ["axolotl", "train", "cfg.yaml"]


def test_train_argv_torchrun_on_cluster_node(monkeypatch):
    monkeypatch.setenv("NUM_NODES", "2")
    monkeypatch.setenv("NODE_RANK", "1")
    monkeypatch.setenv("NUM_TRAINERS", "4")
    monkeypatch.setenv("PRIMARY_ADDR", "10.65.0.2")
    monkeypatch.setenv("PRIMARY_PORT", "29500")
    argv = axolotl_mod._train_argv(Path("cfg.yaml"))
    assert argv[0] == "torchrun"
    assert argv[argv.index("--nnodes") + 1] == "2"
    assert argv[argv.index("--node_rank") + 1] == "1"
    assert argv[argv.index("--nproc_per_node") + 1] == "4"
    # static rendezvous is the only mode Instant Clusters support
    assert argv[argv.index("--rdzv_backend") + 1] == "static"
    assert argv[argv.index("--rdzv_endpoint") + 1] == "10.65.0.2:29500"
    assert argv[-3:] == ["-m", "axolotl.cli.train", "cfg.yaml"]


def test_bellhop_cluster_config_mapping():
    kwargs = BellhopExecutor._cluster_config_kwargs(
        PodSpec(gpu="H200", gpu_count=8, nodes=4, image="ghcr.io/x/y:z",
                max_hours=6.0, cuda_versions=["12.6"], max_hourly_cost=200.0),
        "s1",
    )
    assert kwargs["nodes"] == 4 and kwargs["gpu_count"] == 8
    assert kwargs["image"] == "ghcr.io/x/y:z"
    assert kwargs["allowed_cuda_versions"] == ["12.6"]  # ClusterConfig spelling
    assert kwargs["max_hourly_cost"] == 200.0
    assert kwargs["max_lifetime"].total_seconds() == 6 * 3600
    assert "cuda_versions" not in kwargs


def test_stage_script_bus_egress_is_rank0_guarded():
    """Every node runs the stage script on a cluster; only rank 0 may push
    checkpoints and emit the pointer row (rank 0's results dir is what gets
    pulled). The ${NODE_RANK:-0} default keeps single-node pods unaffected."""
    ex = BellhopExecutor(gcs_base="gs://bucket/exp")
    stage = StageSpec(name="s", description="", kind="midtrain", base_model="m",
                      pod={"gpu": "H200", "nodes": 2})
    _, run = ex._stage_script(
        stage, "out/axolotl.yaml", "out", None,
        wheel_rel="out/dist/scimt.whl", stage_template_rel="stages/s.yaml",
    )
    assert 'if [ "${NODE_RANK:-0}" = "0" ]; then' in run
    guarded = run.split('if [ "${NODE_RANK:-0}" = "0" ]; then', 1)[1]
    assert "rclone copy out/checkpoints" in guarded
    assert "checkpoints.jsonl" in guarded
    assert "rm -rf out/checkpoints" in guarded


def test_finalize_skipped_on_nonzero_rank(monkeypatch, tmp_path):
    """Cluster ranks >0 never hold consolidated checkpoints — finalize must
    not fire there (it would raise on the missing trainer_state.json)."""
    calls = []
    monkeypatch.setattr(axolotl_mod, "finalize_training_attribution",
                        lambda *a: calls.append(a))

    def _aiter(items):
        async def gen():
            for i in items:
                yield i
        return gen()

    class _P:
        returncode = 0

        async def wait(self):
            return 0

    async def fake_exec(*a, **k):
        p = _P()
        p.stdout = _aiter([])
        return p

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    stage = StageSpec(name="s", description="", kind="midtrain", base_model="m")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("{}")

    monkeypatch.setenv("NODE_RANK", "1")
    asyncio.run(LocalExecutor().run_stage(cfg_path, tmp_path, stage))
    assert calls == []

    monkeypatch.setenv("NODE_RANK", "0")
    asyncio.run(LocalExecutor().run_stage(cfg_path, tmp_path, stage))
    assert len(calls) == 1


def test_bellhop_stage_script_pulls_gs_adapter_pointer():
    """continue_adapter chaining: a gs:// lora_model_dir is pulled to
    prev_adapter with an adapter-files guard, and cleaned after the run."""
    ex = BellhopExecutor(gcs_base="gs://bucket/exp")
    stage = StageSpec(name="s", description="", kind="sft", base_model="m",
                      pod={"gpu": "B200"})
    setup, run = ex._stage_script(
        stage, "out/axolotl.yaml", "out", None,
        adapter_gs_pointer="gs://bucket/mt/checkpoints/checkpoint-9/",
        wheel_rel="out/scimt.whl", stage_template_rel="stages/s.yaml",
    )
    assert ("rclone copy gs://bucket/mt/checkpoints/checkpoint-9/ "
            "out/prev_adapter") in setup
    assert "adapter_config" in setup and "adapter_model" in setup
    assert "prev_adapter pull incomplete" in setup
    assert "rm -rf out/prev_ckpt out/prev_adapter out/prepared" in run
