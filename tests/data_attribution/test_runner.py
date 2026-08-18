"""Runner phase tests: config-first orchestration over real scimt artifacts.

The fixture chain builds two REAL scimt training runs (midtraining then chat
SFT) through the actual ``scimt.train`` pipeline with only the GPU executor
faked (the fake writes a genuine TinyLM safetensors checkpoint), plus AdamW
attribution snapshots written through ``write_adamw_snapshot``. Phase tests
monkeypatch only the runner's model/tokenizer loading seams (TinyLM +
ToyTokenizer); everything else — stage resolution, artifacts, identities,
scoring math — is the real library.
"""

from __future__ import annotations

import asyncio
import builtins
import dataclasses
import importlib
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

import scimt.train.axolotl as axolotl_mod
from scimt import train as training
from scimt.data_attribution import runner
from scimt.data_attribution.artifacts import (
    ArtifactIntegrityError,
    IdentityMismatchError,
    ShardManifest,
    read_identity,
)
from scimt.data_attribution.config import load_attribution_config
from scimt.data_attribution.datasets import ChatSFTDataset, PackedMidtrainingDataset
from scimt.data_attribution.gradients import BatchedVJPBackend
from scimt.data_attribution.losses import CausalLMLossAdapter
from scimt.data_attribution.manifest import ParameterManifest
from scimt.data_attribution.metrics import REQUIRED_PROVENANCE
from scimt.data_attribution.second_order import jvp_sweep
from scimt.data_attribution.source import (
    DiagonalCurvature,
    SourceScorer,
    SourceSegment,
)
from scimt.data_attribution.stages import StageResolutionError, artifact_digest
from scimt.dataset import Dataset
from scimt.train.attribution_snapshot import (
    load_optimizer_snapshot,
    write_adamw_snapshot,
)
from scimt.train.axolotl import LocalExecutor

from .fixtures import TinyLM, ToyTokenizer

HEAVY_ROOTS = {
    "accelerate",
    "datasets",
    "huggingface_hub",
    "kronfluence",
    "numpy",
    "safetensors",
    "scipy",
    "torch",
    "tqdm",
    "transformers",
}
SEQUENCE_LENGTH = 12


# --------------------------------------------------------------- fixture chain
def _write_tiny_checkpoint(ck_dir: Path) -> None:
    model = TinyLM().float()
    ck_dir.mkdir(parents=True, exist_ok=True)
    (ck_dir / "config.json").write_text('{"model_type": "tiny"}')
    save_file(model.state_dict(), str(ck_dir / "model.safetensors"))


def _trainer_state(global_step: int, lrs: list[float]) -> dict:
    history = [
        {"loss": 1.0, "learning_rate": lrs[s - 1], "epoch": s / global_step, "step": s}
        for s in range(1, global_step + 1)
    ]
    return {
        "global_step": global_step,
        "logging_steps": 1,
        "max_steps": global_step,
        "log_history": history,
    }


def _make_dataset(root: Path, *, kind: str, rows: list[dict], n_docs=None) -> Dataset:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "data.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset = Dataset(path=str(path), kind=kind, n_docs=n_docs)
    dataset.save()
    return dataset


def _build_run(tmp_path, monkeypatch, *, name: str, kind: str, dataset: Dataset,
               step: int = 3, lrs: list[float] | None = None,
               weight_decay: float = 0.01, write_checkpoint=None):
    """Train through the REAL scimt pipeline with only the executor faked;
    the fake writes the trainer's on-disk products (adapted from
    test_stage_adapter, plus a genuine TinyLM safetensors checkpoint —
    or any checkpoint ``write_checkpoint`` produces)."""
    write_checkpoint = write_checkpoint or _write_tiny_checkpoint
    lrs = lrs if lrs is not None else [1e-2, 8e-3, 5e-3]
    stages_dir = tmp_path / "stage_templates"
    stages_dir.mkdir(exist_ok=True)
    template = f"tiny_runner_{kind}"
    (stages_dir / f"{template}.yaml").write_text(yaml.safe_dump({
        "name": template,
        "description": "runner test template",
        "kind": kind,
        "base_model": "some/base",
        "axolotl": {
            "base_model": "some/base",
            "datasets": [{"path": "SET_BY_RENDER", "type": "completion",
                          "field": "text"}],
            "optimizer": "adamw_torch_fused",
            "weight_decay": weight_decay,
            "learning_rate": lrs[0],
            "logging_steps": 1,
        },
    }))
    monkeypatch.setattr(axolotl_mod, "STAGES_DIR", stages_dir)
    monkeypatch.setenv("SCIMT_ALLOW_DIRTY", "1")
    state = _trainer_state(step, lrs)

    async def fake_run_stage(self, rendered, out_dir, stage, *, run_name=None):
        ck = out_dir / "checkpoints" / f"checkpoint-{step}"
        write_checkpoint(ck)
        (ck / "trainer_state.json").write_text(json.dumps(state))

    monkeypatch.setattr(LocalExecutor, "run_stage", fake_run_stage)
    out = tmp_path / name
    cfg = training.TrainConfig(stage=template, seed=5)
    ckpt = asyncio.run(training.train_dataset(dataset, out, cfg, run_name=name))
    return out, Path(ckpt.require_state())


def _snapshot_for(state_dir: Path, *, include: list[str] | None = None,
                  step: int = 3, weight_decay: float = 0.01,
                  model_id: str = "TinyLM", seed: int = 11) -> Path:
    """Write an AdamW snapshot whose manifest matches the checkpoint's model
    under the given selection (the runner rebuilds and cross-checks it)."""
    model = TinyLM().float()
    model.load_state_dict(load_file(str(state_dir / "model.safetensors")))
    manifest = ParameterManifest.from_model(model, model_id, include=include)
    generator = torch.Generator().manual_seed(seed)
    exp_avg_sq = {
        entry.name: torch.rand(entry.shape, generator=generator) + 0.05
        for entry in manifest.included_entries()
    }
    directory = state_dir.parent / "attribution_snapshots" / f"step-{step}"
    write_adamw_snapshot(
        directory,
        manifest=manifest,
        exp_avg_sq=exp_avg_sq,
        step=step,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=weight_decay,
        weight_decay_values=(0.0, weight_decay),
        model_checkpoint={"global_step": step,
                          "relative_dir": f"../../{state_dir.name}"},
    )
    return directory


MID_ROWS = [{"text": "abcdefghij"}, {"text": "klmnopqrst"},
            {"text": "uvwxyzabcd"}, {"text": "efghijklmn"}]
SFT_ROWS = [
    {"messages": [{"role": "user", "content": "qq"},
                  {"role": "assistant", "content": "aa"}]},
    {"messages": [{"role": "user", "content": "rr"},
                  {"role": "assistant", "content": "bb"}]},
    {"messages": [{"role": "user", "content": "ss"},
                  {"role": "assistant", "content": "cc"}]},
]
QUERY_ROWS = [
    {"messages": [{"role": "user", "content": "tt"},
                  {"role": "assistant", "content": "dd"}]},
    {"messages": [{"role": "user", "content": "uu"},
                  {"role": "assistant", "content": "ee"}]},
]


@dataclasses.dataclass
class Chain:
    tmp_path: Path
    payload: dict

    def config(self, **overrides):
        payload = json.loads(json.dumps(self.payload))
        for key, value in overrides.items():
            if value is None:
                payload.pop(key, None)
            elif isinstance(value, dict) and isinstance(payload.get(key), dict):
                payload[key].update(value)
            else:
                payload[key] = value
        path = self.tmp_path / "attribution.yaml"
        path.write_text(yaml.safe_dump(payload))
        return load_attribution_config(path), path


def _write_tokenizer_dir(directory: Path, *, template: str = "v1") -> Path:
    """A dedicated tokenizer directory whose CONTENT participates in artifact
    identity (the loaded tokenizer itself is the monkeypatched ToyTokenizer;
    these bytes stand in for tokenizer.json/chat-template files)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "ToyTokenizer",
        "chat_template": f"<role>{{role}}</role>{{content}}<end>::{template}",
    }))
    return directory


@pytest.fixture
def chain(tmp_path, monkeypatch) -> Chain:
    mid_ds = _make_dataset(tmp_path / "mid_data", kind="docs", rows=MID_ROWS)
    sft_ds = _make_dataset(tmp_path / "sft_data", kind="chat", rows=SFT_ROWS,
                           n_docs=len(SFT_ROWS))
    query_ds = _make_dataset(tmp_path / "query_data", kind="chat",
                             rows=QUERY_ROWS)
    tokenizer_dir = _write_tokenizer_dir(tmp_path / "tokenizer")
    mid_run, mid_ck = _build_run(tmp_path, monkeypatch, name="mid-run",
                                 kind="midtrain", dataset=mid_ds,
                                 lrs=[1e-2, 8e-3, 5e-3])
    sft_run, sft_ck = _build_run(tmp_path, monkeypatch, name="sft-run",
                                 kind="sft", dataset=sft_ds,
                                 lrs=[5e-3, 3e-3, 1e-3])
    mid_snap = _snapshot_for(mid_ck, seed=11)
    sft_snap = _snapshot_for(sft_ck, seed=12)
    payload = {
        "stages": [
            {"name": "mid", "checkpoint": str(mid_run),
             "dataset": mid_ds.path, "objective": "midtraining",
             "n_examples": len(MID_ROWS), "weight_decay": 0.01,
             "optimizer_snapshot": str(mid_snap)},
            {"name": "sft", "checkpoint": str(sft_run),
             "dataset": sft_ds.path, "objective": "sft",
             "n_examples": len(SFT_ROWS), "weight_decay": 0.01,
             "optimizer_snapshot": str(sft_snap)},
        ],
        "query": {"checkpoint": str(sft_run), "dataset": query_ds.path,
                  "objective": "sft"},
        "tokenizer": str(tokenizer_dir),
        "output_dir": str(tmp_path / "attr-out"),
        "method": {"row_reduction": "per_token", "curvature": "fisher",
                   "basis": "raw", "damping_sweep": [0.0, 0.5]},
        "data": {"sequence_length": SEQUENCE_LENGTH, "batch_size": 2,
                 "vjp_chunk_size": 4, "rows_per_shard": 4},
        "factors": {"samples": 6, "source_batch_size": 2, "fit_batch_size": 2,
                    "max_positions_per_sequence": 2},
        "seed": 0,
    }
    return Chain(tmp_path=tmp_path, payload=payload)


def _install_tiny_loaders(monkeypatch):
    def load_model(checkpoint_dir, *, dtype, device, gradient_checkpointing=False):
        model = TinyLM().float()
        model.load_state_dict(
            load_file(str(Path(checkpoint_dir) / "model.safetensors"))
        )
        return model.to(device)

    monkeypatch.setattr(runner, "_load_model", load_model)
    monkeypatch.setattr(runner, "_load_tokenizer", lambda path: ToyTokenizer())


def _run(coroutine):
    return asyncio.run(coroutine)


def _estimated_adam_overrides(chain, **estimator_overrides) -> dict:
    stages = json.loads(json.dumps(chain.payload["stages"]))
    for stage in stages:
        stage["optimizer_snapshot"] = None
    estimator = {
        "dataset": chain.payload["stages"][0]["dataset"],
        "objective": "midtraining",
        "num_batches": 1,
        "global_batch_size": 2,
        "micro_batch_size": 1,
        "beta2": 0.999,
        "optimizer_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "seed": 42,
    }
    estimator.update(estimator_overrides)
    return {
        "stages": stages,
        "method": {
            "basis": "adam",
            "curvature": "fisher",
            "damping_sweep": [0.1, 0.2],
        },
        "adam_moment_estimator": estimator,
    }


# ----------------------------------------------- load-path-independent identity
def _write_real_tokenizer(directory: Path):
    """A REAL (hand-built, per-character byte-level GPT-2) tokenizer dir that
    AutoTokenizer loads without any stub."""
    directory.mkdir(parents=True, exist_ok=True)
    vocab = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}
    vocab["Ġ"] = len(vocab)
    vocab["<|endoftext|>"] = len(vocab)
    (directory / "vocab.json").write_text(json.dumps(vocab))
    (directory / "merges.txt").write_text("#version: 0.2\n")
    (directory / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "GPT2Tokenizer", "eos_token": "<|endoftext|>"}
    ))
    return len(vocab)


def _tiny_neox(vocab_size: int):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(0)
    return GPTNeoXForCausalLM(GPTNeoXConfig(
        vocab_size=vocab_size, hidden_size=8, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=16,
        max_position_embeddings=32,
    ))


def test_manifest_digests_are_load_path_independent_for_real_hf_models(tmp_path):
    """P1 regression: `from_pretrained` stamps the LOAD PATH into
    `name_or_path`; the stable identifier must give two saves of the
    identical model equal manifest digests."""
    from transformers import AutoModelForCausalLM

    from scimt.data_attribution.manifest import stable_model_identifier

    vocab_size = _write_real_tokenizer(tmp_path / "tok")
    model = _tiny_neox(vocab_size)
    dir_a, dir_b = tmp_path / "save-a", tmp_path / "save-b"
    model.save_pretrained(dir_a)
    model.save_pretrained(dir_b)
    loaded_a = AutoModelForCausalLM.from_pretrained(str(dir_a),
                                                    local_files_only=True)
    loaded_b = AutoModelForCausalLM.from_pretrained(str(dir_b),
                                                    local_files_only=True)
    assert loaded_a.config._name_or_path != loaded_b.config._name_or_path
    assert (stable_model_identifier(loaded_a)
            == stable_model_identifier(loaded_b)
            == stable_model_identifier(model)
            == "gpt_neox/GPTNeoXForCausalLM")
    digest_a = ParameterManifest.from_model(
        loaded_a, stable_model_identifier(loaded_a)).digest()
    digest_b = ParameterManifest.from_model(
        loaded_b, stable_model_identifier(loaded_b)).digest()
    assert digest_a == digest_b
    # The exact failure mode being fixed: path-derived labels diverge.
    assert ParameterManifest.from_model(
        loaded_a, loaded_a.config._name_or_path
    ).digest() != ParameterManifest.from_model(
        loaded_b, loaded_b.config._name_or_path
    ).digest()


def test_cross_checkpoint_chain_accepts_with_real_loaders(tmp_path, monkeypatch):
    """P1 acceptance: rows at checkpoint dir A + queries at a DIFFERENT dir B
    (two saves of one model), REAL from_pretrained loaders end to end, and
    score-source's shared-manifest upstream checks accept."""
    vocab_size = _write_real_tokenizer(tmp_path / "tokenizer")
    model = _tiny_neox(vocab_size)
    query_dir = tmp_path / "query-save"
    model.save_pretrained(query_dir)

    docs = _make_dataset(tmp_path / "mid_data", kind="docs", rows=MID_ROWS)
    query_docs = _make_dataset(tmp_path / "query_docs", kind="docs",
                               rows=MID_ROWS[:2])
    run_dir, _ = _build_run(
        tmp_path, monkeypatch, name="neox-run", kind="midtrain", dataset=docs,
        write_checkpoint=lambda ck: model.save_pretrained(ck),
    )
    payload = {
        "stages": [
            {"name": "mid", "checkpoint": str(run_dir), "dataset": docs.path,
             "objective": "midtraining", "n_examples": len(MID_ROWS),
             "weight_decay": 0.01},
        ],
        "query": {"checkpoint": str(query_dir), "dataset": query_docs.path,
                  "objective": "midtraining"},
        "tokenizer": str(tmp_path / "tokenizer"),
        "output_dir": str(tmp_path / "attr-real"),
        "method": {"curvature": "fisher", "basis": "raw",
                   "damping_sweep": [0.1]},
        "data": {"sequence_length": 8, "batch_size": 2, "vjp_chunk_size": 4,
                 "rows_per_shard": 16},
        "factors": {"samples": 4, "source_batch_size": 2, "fit_batch_size": 2},
        "seed": 0,
    }
    config_path = tmp_path / "real.yaml"
    config_path.write_text(yaml.safe_dump(payload))
    config = load_attribution_config(config_path)

    # NO loader stubs: the runner's real transformers seams do the loading.
    _run(runner.fit_factors(config))
    _run(runner.compute_rows(config))
    _run(runner.build_queries(config))
    report = _run(runner.score_source(config))
    assert report.outputs[0].skipped is False
    layout = runner.run_layout(config.output_dir)
    rows_identity = read_identity(layout.rows / "mid")
    queries_identity = read_identity(layout.queries)
    # Two different checkpoint directories, one coordinate system.
    assert rows_identity.checkpoint_reference != (
        queries_identity.checkpoint_reference
    )
    assert rows_identity.parameter_manifest_digest == (
        queries_identity.parameter_manifest_digest
    )
    entry = json.loads(
        (layout.scores / "score_manifest.json").read_text()
    )["entries"]["mid__damping-0"]
    scores = load_file(str(layout.scores / entry["file"]))["scores"]
    assert bool(torch.isfinite(scores).all())


# ------------------------------------------------------------- import contract
def test_runner_and_cli_modules_import_torch_free(monkeypatch):
    real_import = builtins.__import__

    def reject_heavy(name, *args, **kwargs):
        if name.split(".", 1)[0] in HEAVY_ROOTS:
            raise AssertionError(f"eager heavy import: {name}")
        return real_import(name, *args, **kwargs)

    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.startswith("scimt.data_attribution")
    }
    try:
        monkeypatch.setattr(builtins, "__import__", reject_heavy)
        importlib.import_module("scimt.data_attribution.runner")
        importlib.import_module("scimt.data_attribution.cli")
    finally:
        for name in list(sys.modules):
            if name.startswith("scimt.data_attribution"):
                sys.modules.pop(name)
        sys.modules.update(saved)


def test_phase_registry_names_every_planned_phase():
    assert set(runner.PHASES) == {
        "estimate-adam", "fit-factors", "compute-rows", "build-queries", "score-source",
        "score-source-streaming",
        "build-directions", "sweep-jvp", "summarize", "dry-run",
    }


# ------------------------------------------------------------- estimate Adam
def test_estimate_adam_writes_paired_checkpoint_local_artifacts_and_resumes(
    chain, monkeypatch
):
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config(**_estimated_adam_overrides(chain))

    report = _run(runner.estimate_adam(config))

    outputs = {output.name: output for output in report.outputs}
    assert set(outputs) == {"adam_moments/mid", "adam_moments/sft"}
    paired_path = (
        runner.run_layout(config.output_dir).adam_moments / "paired_batches.json"
    )
    paired = json.loads(paired_path.read_text())
    assert paired["batches"] == [[2, 0]]
    assert paired["sampler"] == "python_random_sample_without_replacement"
    paired_digest = artifact_digest(paired_path)

    checkpoint_digests = set()
    for stage in config.stages:
        directory = runner.run_layout(config.output_dir).adam_moments / stage.name
        manifest = ShardManifest.load(directory)
        assert manifest.total_rows == 1 and manifest.feature_dim == 160
        tensors = manifest.read_rows(directory)
        assert tensors["features"].shape == (1, 160)
        assert bool(torch.isfinite(tensors["features"]).all())
        assert bool((tensors["features"] >= 0).all())
        assert set(tensors) == {
            "features", "sample_ids", "sequence_ids", "target_positions"
        }

        statistics = json.loads((directory / "statistics.json").read_text())
        assert statistics["statistic"] == "checkpoint_local_adam_second_raw_moment"
        assert statistics["number_of_gradient_samples"] == 1
        assert statistics["synthetic_estimator_step"] == 1
        assert statistics["checkpoint_step"] == 3
        assert statistics["bias_correction"] == pytest.approx(1 - 0.999)
        assert statistics["paired_batch_manifest_digest"] == paired_digest
        assert statistics["stores_first_moment"] is False
        assert statistics["stores_optimizer_state"] is False
        identity = read_identity(directory)
        checkpoint_digests.add(identity.checkpoint_digest)
        assert identity.upstream_digests["paired_batches"] == paired_digest
        assert identity.seeds == {"estimator": 42, "run": 0}
    assert len(checkpoint_digests) == 2

    resumed = _run(runner.estimate_adam(config))
    assert all(output.skipped for output in resumed.outputs)


def test_estimate_adam_refuses_insufficient_population_before_model_load(
    chain, monkeypatch
):
    config, _ = chain.config(
        **_estimated_adam_overrides(
            chain, num_batches=100, global_batch_size=100
        )
    )
    loaded = False

    def reject_load(*args, **kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("model must not load before paired sampling validates")

    monkeypatch.setattr(runner, "_load_model", reject_load)
    monkeypatch.setattr(runner, "_load_tokenizer", lambda path: ToyTokenizer())

    with pytest.raises(ValueError, match="sampling without replacement"):
        _run(runner.estimate_adam(config))
    assert loaded is False


# ------------------------------------------------------------------- dry run
def test_dry_run_resolves_full_chain_with_torch_blocked(chain, monkeypatch):
    """The full dry run works with every heavy root blocked from import:
    stage metadata, counts, Adam availability, manifest availability, factor
    partitions, and output-identity previews, without loading any model."""
    config, config_path = chain.config(
        method={"basis": "adam", "curvature": "fisher"}
    )
    del config
    real_import = builtins.__import__

    def reject_heavy(name, *args, **kwargs):
        if name.split(".", 1)[0] in HEAVY_ROOTS:
            raise AssertionError(f"eager heavy import during dry run: {name}")
        return real_import(name, *args, **kwargs)

    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "scimt" or name.startswith("scimt.")
        or name.split(".", 1)[0] in HEAVY_ROOTS
    }
    try:
        monkeypatch.setattr(builtins, "__import__", reject_heavy)
        fresh_runner = importlib.import_module("scimt.data_attribution.runner")
        fresh_config_mod = importlib.import_module("scimt.data_attribution.config")
        fresh_config = fresh_config_mod.load_attribution_config(config_path)
        report = asyncio.run(fresh_runner.dry_run(fresh_config))
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        for name in list(sys.modules):
            if name == "scimt" or name.startswith("scimt."):
                sys.modules.pop(name)
        sys.modules.update(saved)

    assert report["blockers"] == []
    stages = {entry["name"]: entry for entry in report["stages"]}
    assert set(stages) == {"mid", "sft"}
    assert stages["mid"]["lr_steps"] == pytest.approx(1e-2 + 8e-3 + 5e-3)
    assert stages["mid"]["dataset_rows"] == len(MID_ROWS)
    assert stages["sft"]["dataset_rows"] == len(SFT_ROWS)
    assert stages["mid"]["global_step"] == 3
    # Adam availability is probed from JSON metadata only (no tensor loads).
    for name in ("mid", "sft"):
        adam = stages[name]["adam"]
        assert adam["available"] is True
        assert adam["step"] == 3 and adam["beta2"] == pytest.approx(0.999)
        assert adam["parameter_manifest_digest"] in stages[name][
            "manifest_availability"
        ].values()
    # Parameter counts estimated from safetensors headers, never torch.
    assert stages["mid"]["estimated_included_parameters"] == 160
    assert report["query"]["dataset_rows"] == len(QUERY_ROWS)
    assert report["factor_plan"]["samples"] == 6
    assert report["factor_plan"]["covariance_module_partitions"] == 1
    assert report["expected"]["damping_sweep"] == [0.0, 0.5]
    identities = {entry["output"]: entry for entry in report["planned_identities"]}
    assert "rows/mid" in identities and "scores" in identities
    preview = identities["rows/mid"]["identity"]
    assert preview["source_commit"] == (
        "ca9689a497b921dc516feb663a83269c4a588bbc"
    )
    assert preview["parameter_manifest_digest"] is None
    assert "parameter_manifest_digest" in identities["rows/mid"]["pending"]
    # Tokenizer/chat-template content is resolved into the composite dataset
    # fingerprint torch-free, exactly as the phases will bind it.
    tokenizer_digest = report["tokenizer"]["content_digest"]
    assert len(tokenizer_digest) == 64
    fingerprint = json.loads(preview["dataset_fingerprint"])
    assert fingerprint["tokenizer_content"] == tokenizer_digest
    assert fingerprint["source"] == stages["mid"]["dataset_digest"]
    # A dry run resolves; it never writes.
    assert not Path(chain.payload["output_dir"]).exists()


def test_dry_run_reports_missing_adam_snapshot_as_blocker(chain):
    payload_stage = chain.payload["stages"][0]
    payload_stage_snapshot = payload_stage["optimizer_snapshot"]
    (Path(payload_stage_snapshot) / "optimizer_manifest.json").unlink()
    config, _ = chain.config(method={"basis": "adam", "curvature": "fisher"})
    report = _run(runner.dry_run(config))
    assert any("mid" in blocker and "snapshot" in blocker
               for blocker in report["blockers"])
    stages = {entry["name"]: entry for entry in report["stages"]}
    assert stages["mid"]["adam"]["available"] is False


def test_dry_run_reports_checkpoint_local_adam_work_and_storage(
    chain, monkeypatch
):
    monkeypatch.setattr(runner, "_load_tokenizer", lambda path: ToyTokenizer())
    config, _ = chain.config(
        **_estimated_adam_overrides(
            chain,
            dataset=chain.payload["stages"][1]["dataset"],
            objective="sft",
        )
    )

    report = _run(runner.dry_run(config))

    assert report["blockers"] == []
    estimate = report["adam_moment_estimator"]
    assert estimate["mode"] == "paired_checkpoint_local"
    assert estimate["required_presentations"] == 2
    assert estimate["usable_tokenized_sequences"] == len(SFT_ROWS)
    assert estimate["population_validation"] == "exact_model_free_tokenization"
    assert estimate["checkpoint_count"] == 2
    assert estimate["global_batch_equivalents"] == 2
    assert estimate["selected_moment_storage_bytes"] == 2 * 160 * 4
    assert estimate["peak_selected_accumulator_bytes"] == 160 * 8
    assert estimate["peak_selected_working_bytes_upper_bound"] == 160 * 24
    assert set(estimate["stage_artifacts"]) == {"mid", "sft"}
    assert not Path(config.output_dir).exists()


def test_dry_run_accepts_valid_ekfac_adam_and_warns_on_pending_moments(
    chain, monkeypatch
):
    """dry_run must mirror _require_scorable_method: a config score-source
    accepts reports NO blockers; not-yet-run estimate-adam is a warning."""
    monkeypatch.setattr(runner, "_load_tokenizer", lambda path: ToyTokenizer())
    overrides = _ekfac_adam_overrides(chain)
    overrides["adam_moment_estimator"].update(
        dataset=chain.payload["stages"][1]["dataset"], objective="sft"
    )
    config, _ = chain.config(**overrides)
    report = _run(runner.dry_run(config))
    assert report["blockers"] == []
    assert any("estimate-adam" in warning for warning in report["warnings"])
    fit_previews = [
        entry for entry in report["planned_identities"]
        if entry["output"].startswith("factors/")
    ]
    assert fit_previews
    for entry in fit_previews:
        descriptor = entry["identity"]["basis_descriptor"]
        assert descriptor["coordinates"] == "adam_stage_local"
    paired = Path(config.output_dir) / "adam_moments" / "paired_batches.json"
    paired.parent.mkdir(parents=True, exist_ok=True)
    paired.write_text("{}", encoding="utf-8")
    report = _run(runner.dry_run(config))
    assert report["blockers"] == []
    assert report["warnings"] == []


def test_dry_run_blocks_adam_basis_over_raw_ekfac_curvature(
    chain, monkeypatch
):
    monkeypatch.setattr(runner, "_load_tokenizer", lambda path: ToyTokenizer())
    overrides = _estimated_adam_overrides(
        chain,
        dataset=chain.payload["stages"][1]["dataset"],
        objective="sft",
    )
    overrides["method"] = {
        "basis": "adam",
        "curvature": "ekfac",
        "damping_sweep": [0.1],
    }
    config, _ = chain.config(**overrides)
    report = _run(runner.dry_run(config))
    assert any(
        "basis 'adam' requires method.curvature 'fisher'" in blocker
        for blocker in report["blockers"]
    )


def test_dry_run_blocks_checkpoint_without_safetensors_signature(chain):
    checkpoint = (
        Path(chain.payload["stages"][0]["checkpoint"])
        / "checkpoints"
        / "checkpoint-3"
    )
    (checkpoint / "model.safetensors").rename(checkpoint / "pytorch_model.bin")
    config, _ = chain.config()

    report = _run(runner.dry_run(config))

    assert any(
        "stage 'mid'" in blocker and "safetensors" in blocker
        for blocker in report["blockers"]
    )


def test_dry_run_blocks_empty_selected_parameter_signature(chain):
    config, _ = chain.config(
        parameters={"include": ["does_not_exist"], "exclude": []}
    )

    report = _run(runner.dry_run(config))

    assert any(
        "selected parameter signature is empty" in blocker
        for blocker in report["blockers"]
    )


def test_dry_run_blocks_obviously_insufficient_chat_estimator_population(
    chain, monkeypatch
):
    monkeypatch.setattr(runner, "_load_tokenizer", lambda path: ToyTokenizer())
    config, _ = chain.config(
        **_estimated_adam_overrides(
            chain,
            dataset=chain.payload["stages"][1]["dataset"],
            objective="sft",
            global_batch_size=4,
        )
    )

    report = _run(runner.dry_run(config))

    assert any(
        "sampling without replacement" in blocker
        for blocker in report["blockers"]
    )


def test_dry_run_counts_surviving_chat_targets_not_source_rows(
    chain, monkeypatch
):
    monkeypatch.setattr(runner, "_load_tokenizer", lambda path: ToyTokenizer())
    calibration = _make_dataset(
        chain.tmp_path / "empty-calibration",
        kind="chat",
        rows=[
            {"messages": [{"role": "user", "content": "no target"}]},
            {"messages": [{"role": "user", "content": "still no target"}]},
        ],
    )
    config, _ = chain.config(
        **_estimated_adam_overrides(
            chain,
            dataset=calibration.path,
            objective="sft",
            global_batch_size=1,
        )
    )

    report = _run(runner.dry_run(config))

    assert report["adam_moment_estimator"]["source_rows"] == 2
    assert report["adam_moment_estimator"]["usable_tokenized_sequences"] == 0
    assert any("contains 0 usable" in blocker for blocker in report["blockers"])


def test_dry_run_counts_short_packed_calibration_as_unusable(
    chain, monkeypatch
):
    monkeypatch.setattr(runner, "_load_tokenizer", lambda path: ToyTokenizer())
    calibration = _make_dataset(
        chain.tmp_path / "short-packed-calibration",
        kind="docs",
        rows=[{"text": "a"}],
    )
    config, _ = chain.config(
        **_estimated_adam_overrides(
            chain,
            dataset=calibration.path,
            objective="midtraining",
            global_batch_size=1,
        )
    )

    report = _run(runner.dry_run(config))

    assert report["adam_moment_estimator"]["usable_tokenized_sequences"] == 0
    assert any("contains 0 usable" in blocker for blocker in report["blockers"])


def test_dry_run_blocks_cross_checkpoint_selected_shape_drift(chain):
    mid_checkpoint = (
        Path(chain.payload["stages"][0]["checkpoint"])
        / "checkpoints"
        / "checkpoint-3"
        / "model.safetensors"
    )
    state = load_file(str(mid_checkpoint))
    state["head.weight"] = state["head.weight"][:-1]
    save_file(state, str(mid_checkpoint))
    config, _ = chain.config()

    report = _run(runner.dry_run(config))

    assert any(
        "selected parameter signature" in blocker and "mid" in blocker
        for blocker in report["blockers"]
    )


# ----------------------------------------------------------------- fit-factors
def test_fit_factors_fisher_writes_statistics_and_resumes(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config()
    report = _run(runner.fit_factors(config))
    outputs = {output.name: output for output in report.outputs}
    assert set(outputs) == {"factors/mid", "factors/sft"}
    for name, output in outputs.items():
        assert not output.skipped
        directory = output.directory
        manifest = ShardManifest.load(directory)
        assert manifest.total_rows == 1 and manifest.feature_dim == 160
        statistics = json.loads((directory / "statistics.json").read_text())
        assert REQUIRED_PROVENANCE <= set(statistics)
        assert statistics["estimator"] == "full"
        rows = manifest.read_rows(directory)
        assert bool((rows["features"] >= 0).all())
        identity = read_identity(directory)
        assert identity.curvature_descriptor["method"] == "fisher"
        assert identity.seeds["run"] == 0

    # Identical rerun: a no-op that leaves bytes untouched.
    before = {
        path: path.read_bytes()
        for path in sorted(outputs["factors/mid"].directory.rglob("*"))
        if path.is_file()
    }
    again = _run(runner.fit_factors(config))
    assert all(output.skipped for output in again.outputs)
    after = {
        path: path.read_bytes()
        for path in sorted(outputs["factors/mid"].directory.rglob("*"))
        if path.is_file()
    }
    assert before == after

    # Changed identity (different factor sample budget): focused refusal
    # naming the drifted scope, before any model load.
    drifted, _ = chain.config(factors={"samples": 7})
    with pytest.raises(runner.RunnerError, match="resolved_config"):
        _run(runner.fit_factors(drifted))


def test_fit_factors_fisher_matches_manual_grad_square_mean(chain, monkeypatch):
    """Value-level check: the fitted statistic IS the mean per-example
    squared gradient over the seeded sample items (same sampler, same loss)."""
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config()
    _run(runner.fit_factors(config))
    layout = runner.run_layout(config.output_dir)
    stored = _stage_statistics(layout.factors / "mid")

    from scimt.data_attribution.ekfac import build_ekfac_sample_items

    checkpoint = (Path(chain.payload["stages"][0]["checkpoint"])
                  / "checkpoints" / "checkpoint-3")
    model = TinyLM().float()
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors")))
    manifest = ParameterManifest.from_model(model, "TinyLM")
    dataset = _adapter_for(chain.payload["stages"][0]["dataset"],
                           "midtraining", "per_token")
    items = build_ekfac_sample_items(dataset, {
        "samples": 6, "seed": 0, "source_batch_size": 2, "batch_size": 2,
        "max_positions_per_sequence": 2, "min_position_gap": 1,
    })
    assert items  # the fixture yields a real sample
    entries = manifest.included_entries()
    named = dict(model.named_parameters())
    accumulator = {
        entry.name: torch.zeros(entry.numel, dtype=torch.float64)
        for entry in entries
    }
    for item in items:
        ids = item["input_ids"].unsqueeze(0)
        position = int(item["position"])
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=ids).logits
        loss = torch.nn.functional.cross_entropy(
            logits[0, position - 1 : position].float(),
            ids[0, position : position + 1],
            reduction="sum",
        )
        loss.backward()
        for entry in entries:
            gradient = named[entry.name].grad
            if gradient is not None:
                accumulator[entry.name].add_(
                    gradient.detach().reshape(-1).double().square()
                )
    expected = torch.cat(
        [accumulator[entry.name] / len(items) for entry in entries]
    ).to(torch.float32)
    assert torch.equal(stored, expected)


def test_fisher_factor_inputs_move_to_the_configured_model_device():
    item = {"input_ids": torch.tensor([1, 2, 3], dtype=torch.int64)}

    moved = runner._factor_input_ids(item, "meta")

    assert moved.shape == (1, 3)
    assert moved.device.type == "meta"


def test_fit_factors_ekfac_fits_kronfluence_and_is_reloadable(chain, monkeypatch):
    pytest.importorskip("kronfluence")
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config(method={"curvature": "ekfac"})
    report = _run(runner.fit_factors(config))
    from scimt.data_attribution.ekfac import load_ekfac

    for output in report.outputs:
        ekfac_dir = output.directory / "ekfac"
        manifest = ParameterManifest.load(ekfac_dir)
        factors = load_ekfac(ekfac_dir, manifest)
        assert set(factors.linears) == {"head"}
        completion = json.loads(
            (output.directory / "factors_complete.json").read_text()
        )
        assert completion["identity_digest"] == output.identity_digest
        assert completion["snapshot"] == factors.snapshot
    again = _run(runner.fit_factors(config))
    assert all(output.skipped for output in again.outputs)


def test_fit_factors_refuses_ggn_curvature(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config(method={"curvature": "ggn"})
    with pytest.raises(runner.RunnerError, match="[Gg][Gg][Nn]"):
        _run(runner.fit_factors(config))


# ---------------------------------------------------------------- row phases
def _adapter_for(rows_path: str | Path, objective: str, reduction: str):
    tokenizer = ToyTokenizer()
    if objective == "midtraining":
        return PackedMidtrainingDataset(
            Path(rows_path), tokenizer, SEQUENCE_LENGTH, 0, reduction=reduction
        )
    return ChatSFTDataset(
        Path(rows_path), tokenizer, SEQUENCE_LENGTH, 0, reduction=reduction
    )


def _manual_rows(checkpoint_dir: Path, dataset, reduction: str):
    model = TinyLM().float()
    model.load_state_dict(load_file(str(checkpoint_dir / "model.safetensors")))
    manifest = ParameterManifest.from_model(model, "TinyLM")
    adapter = CausalLMLossAdapter(model, reduction=reduction, device="cpu")
    backend = BatchedVJPBackend(model, manifest)
    features, ids = [], []
    for batch in dataset.iter_batches(2):
        loss_batch = adapter.per_datapoint_losses(batch)
        features.append(backend.rows(loss_batch.losses, chunk_size=4))
        ids.append(loss_batch.sample_ids)
    return torch.cat(features), torch.cat(ids)


def test_compute_rows_matches_direct_backend_and_resumes(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config()
    report = _run(runner.compute_rows(config))
    outputs = {output.name: output for output in report.outputs}
    mid_dir = outputs["rows/mid"].directory
    manifest = ShardManifest.load(mid_dir)
    stored = manifest.read_rows(mid_dir)
    dataset = _adapter_for(chain.payload["stages"][0]["dataset"],
                           "midtraining", "per_token")
    expected_features, expected_ids = _manual_rows(
        Path(chain.payload["stages"][0]["checkpoint"])
        / "checkpoints" / "checkpoint-3",
        dataset, "per_token",
    )
    assert stored["features"].shape == tuple(expected_features.shape)
    assert torch.equal(stored["sample_ids"], expected_ids)
    assert torch.allclose(stored["features"], expected_features, atol=1e-6)
    assert len(manifest.shards) > 1  # rows_per_shard=4 forces real sharding

    # Interrupt after the first committed shard, then resume: identical bytes.
    fresh_out = chain.tmp_path / "attr-out-resume"
    interrupted, _ = chain.config(output_dir=str(fresh_out))
    real_append = runner.ArtifactWriter.append
    calls = {"n": 0}

    def exploding_append(self, **kwargs):
        real_append(self, **kwargs)
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(runner.ArtifactWriter, "append", exploding_append)
    with pytest.raises(RuntimeError, match="interruption"):
        _run(runner.compute_rows(interrupted))
    monkeypatch.setattr(runner.ArtifactWriter, "append", real_append)
    resumed_dir = fresh_out / "rows" / "mid"
    committed_before_resume = len(
        list(resumed_dir.glob("shard_*.safetensors"))
    )
    assert committed_before_resume >= 1
    _run(runner.compute_rows(interrupted))
    resumed = ShardManifest.load(resumed_dir).read_rows(resumed_dir)
    assert torch.equal(resumed["sample_ids"], stored["sample_ids"])
    # Same geometry, same hardware, same code: resumption is bit-exact.
    assert torch.equal(resumed["features"], stored["features"])

    # Identity drift (different sequence length): focused refusal.
    drifted, _ = chain.config(data={"sequence_length": 10})
    with pytest.raises(runner.RunnerError, match="resolved_config"):
        _run(runner.compute_rows(drifted))


def test_tokenizer_content_mutation_refuses_row_reuse(chain, monkeypatch):
    """Tokenizer/chat-template CONTENT is identity: an in-place edit (same
    filenames) between runs is a focused dataset_fingerprint refusal, never
    silent reuse or resume."""
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config()
    _run(runner.compute_rows(config))
    _write_tokenizer_dir(Path(chain.payload["tokenizer"]), template="v2")
    with pytest.raises(IdentityMismatchError, match="dataset_fingerprint"):
        _run(runner.compute_rows(config))


def test_score_source_refuses_rows_and_queries_from_different_tokenizers(
    chain, monkeypatch
):
    """Rows tokenized under template A plus queries rebuilt under template B
    cannot be scored together."""
    config, _ = _complete_chain(chain, monkeypatch)
    _write_tokenizer_dir(Path(chain.payload["tokenizer"]), template="v2")
    import shutil

    layout = runner.run_layout(config.output_dir)
    shutil.rmtree(layout.queries)
    _run(runner.build_queries(config))  # rebinds queries under template B
    with pytest.raises(runner.RunnerError, match="dataset_fingerprint"):
        _run(runner.score_source(config))


def test_reusable_phases_do_not_require_completed_adam_estimates(
    chain, monkeypatch
):
    _install_tiny_loaders(monkeypatch)
    overrides = _estimated_adam_overrides(chain)
    config, _ = chain.config(**overrides)

    _run(runner.fit_factors(config))
    _run(runner.compute_rows(config))
    _run(runner.build_queries(config))

    with pytest.raises((runner.RunnerError, FileNotFoundError), match="estimate-adam"):
        _run(runner.score_source(config))


def test_build_queries_writes_rows_at_query_checkpoint(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config()
    report = _run(runner.build_queries(config))
    (output,) = report.outputs
    manifest = ShardManifest.load(output.directory)
    dataset = _adapter_for(chain.payload["query"]["dataset"], "sft", "per_token")
    expected_features, expected_ids = _manual_rows(
        Path(chain.payload["query"]["checkpoint"])
        / "checkpoints" / "checkpoint-3",
        dataset, "per_token",
    )
    stored = manifest.read_rows(output.directory)
    assert torch.equal(stored["sample_ids"], expected_ids)
    assert torch.allclose(stored["features"], expected_features, atol=1e-6)
    identity = read_identity(output.directory)
    assert identity.loss_convention["reduction"] == "per_token"
    assert identity.loss_convention["target_policy"] == (
        "assistant_content_and_end"
    )


def test_compute_rows_with_logra_projects_and_records_descriptor(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config(
        method={"logra": {"rank": 2, "init": "random", "seed": 3,
                          "targets": "head"}}
    )
    report = _run(runner.compute_rows(config))
    outputs = {output.name: output for output in report.outputs}
    manifest = ShardManifest.load(outputs["rows/mid"].directory)
    assert manifest.feature_dim == 4  # rank^2 for the single wrapped Linear
    identity = read_identity(outputs["rows/mid"].directory)
    assert identity.logra_descriptor["rank"] == 2
    assert identity.logra_descriptor["init"] == "random"
    assert identity.basis_descriptor["coordinates"] == "logra_B"


# ---------------------------------------------------------------- score-source
def _complete_chain(chain, monkeypatch, **config_overrides):
    _install_tiny_loaders(monkeypatch)
    config, path = chain.config(**config_overrides)
    _run(runner.fit_factors(config))
    _run(runner.compute_rows(config))
    _run(runner.build_queries(config))
    if config.adam_moment_estimator is not None:
        _run(runner.estimate_adam(config))
    return config, path


def _stage_statistics(directory: Path) -> torch.Tensor:
    manifest = ShardManifest.load(directory)
    return manifest.read_rows(directory)["features"][0].float()


def test_score_source_normalizes_each_segment_exactly_once(chain, monkeypatch):
    config, _ = _complete_chain(chain, monkeypatch)
    report = _run(runner.score_source(config))
    scores_dir = Path(chain.payload["output_dir"]) / "scores"
    completeness = json.loads((scores_dir / "score_manifest.json").read_text())
    assert set(completeness["entries"]) == {
        f"{stage}__damping-{index}"
        for stage in ("mid", "sft") for index in range(2)
    }
    assert completeness["expected"] == sorted(completeness["entries"])

    layout = runner.run_layout(config.output_dir)
    query_manifest = ShardManifest.load(layout.queries)
    query_rows = query_manifest.read_rows(layout.queries)["features"].float()
    resolved_lr = {"mid": 1e-2 + 8e-3 + 5e-3, "sft": 5e-3 + 3e-3 + 1e-3}
    for damping_index, damping in enumerate(config.method.damping_sweep):
        segments = []
        for stage in config.stages:
            fisher = _stage_statistics(layout.factors / stage.name)
            segments.append(SourceSegment(
                stage.name,
                DiagonalCurvature(fisher.double().numpy() + damping,
                                  basis_descriptor={"coordinates": "raw"}),
                resolved_lr[stage.name],
            ))
        scorer = SourceScorer(segments)
        transformed = scorer.transformed_queries(query_rows.numpy())
        for stage_index, stage in enumerate(config.stages):
            rows_dir = layout.rows / stage.name
            train = ShardManifest.load(rows_dir).read_rows(rows_dir)
            expected = (
                transformed[stage_index] @ train["features"].numpy().T
            ) / stage.n_examples
            entry = completeness["entries"][
                f"{stage.name}__damping-{damping_index}"
            ]
            saved = load_file(str(scores_dir / entry["file"]))
            assert saved["scores"].shape == (
                query_rows.shape[0], train["features"].shape[0]
            )
            assert torch.allclose(
                saved["scores"], torch.from_numpy(expected), atol=1e-5
            )
            # The 1/N factor is present exactly once.
            unnormalized = transformed[stage_index] @ train["features"].numpy().T
            assert torch.allclose(
                saved["scores"] * stage.n_examples,
                torch.from_numpy(unnormalized),
                atol=1e-4,
            )
            assert torch.equal(saved["train_sample_ids"], train["sample_ids"])
    assert report.outputs[0].name == "scores"
    assert report.outputs[0].skipped is False
    # Identical rerun: the completed score matrix is a no-op.
    again = _run(runner.score_source(config))
    assert again.outputs[0].skipped is True


def test_score_source_captured_adam_uses_stage_local_fourth_root_geometry(
    chain, monkeypatch
):
    config, _ = _complete_chain(
        chain, monkeypatch,
        method={"basis": "adam", "curvature": "fisher", "damping_sweep": [0.1]},
    )
    _run(runner.score_source(config))
    layout = runner.run_layout(config.output_dir)
    scores_dir = layout.scores
    completeness = json.loads((scores_dir / "score_manifest.json").read_text())

    # Reconstruct independently with a distinct checkpoint-local scale:
    # A_l=(sqrt(v_hat_l)+eps_l+damping)^-1/2, rows -> A_l rows,
    # curvature -> A_l H_l A_l, and boundary transition A_prev/A_current.
    from scimt.train.attribution_snapshot import load_optimizer_snapshot

    damping = 0.1
    scales = []
    for stage in config.stages:
        snapshot = load_optimizer_snapshot(Path(stage.optimizer_snapshot))
        corrected = snapshot.bias_corrected_exp_avg_sq()
        flat = torch.cat(
            [
                corrected[entry.name].reshape(-1)
                for entry in snapshot.manifest.included_entries()
            ]
        )
        scales.append((flat.sqrt() + snapshot.info.epsilon + damping).rsqrt())
    assert not torch.equal(scales[0], scales[1])
    query_rows = ShardManifest.load(layout.queries).read_rows(
        layout.queries)["features"].float()
    resolved_lr = {"mid": 1e-2 + 8e-3 + 5e-3, "sft": 5e-3 + 3e-3 + 1e-3}
    segments, trains = [], []
    for stage_index, stage in enumerate(config.stages):
        fisher = _stage_statistics(layout.factors / stage.name)
        transformed_curvature = (
            fisher * scales[stage_index].pow(2)
        ).double().numpy()
        segments.append(SourceSegment(
            stage.name,
            DiagonalCurvature(transformed_curvature,
                              basis_descriptor={
                                  "coordinates": "adam_stage_local",
                                  "stage": stage.name,
                              }),
            resolved_lr[stage.name],
            transition_to_previous=(
                None
                if stage_index == 0
                else (scales[stage_index - 1] / scales[stage_index]).numpy()
            ),
        ))
        rows_dir = layout.rows / stage.name
        trains.append(ShardManifest.load(rows_dir).read_rows(rows_dir))
    scorer = SourceScorer(segments)
    transformed = scorer.transformed_queries((query_rows * scales[-1]).numpy())
    for stage_index, stage in enumerate(config.stages):
        expected = (
            transformed[stage_index]
            @ (
                trains[stage_index]["features"].float() * scales[stage_index]
            ).numpy().T
        ) / stage.n_examples
        entry = completeness["entries"][f"{stage.name}__damping-0"]
        saved = load_file(str(scores_dir / entry["file"]))
        assert torch.allclose(
            saved["scores"], torch.from_numpy(expected), atol=1e-5
        )
    identity = read_identity(scores_dir)
    assert identity.basis_descriptor["coordinates"] == "adam_stage_local"
    assert [entry["stage"] for entry in identity.basis_descriptor["stages"]] == [
        "mid",
        "sft",
    ]
    assert all(
        entry["mode"] == "captured"
        for entry in identity.basis_descriptor["stages"]
    )


def test_score_source_estimated_adam_uses_each_checkpoint_artifact_once(
    chain, monkeypatch
):
    overrides = _estimated_adam_overrides(chain)
    config, _ = _complete_chain(chain, monkeypatch, **overrides)
    original = runner._load_stage_adam_payloads
    calls = 0

    def counting_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_load_stage_adam_payloads", counting_load)
    _run(runner.score_source(config))

    assert calls == 1
    layout = runner.run_layout(config.output_dir)
    identity = read_identity(layout.scores)
    descriptor = identity.basis_descriptor
    assert descriptor["coordinates"] == "adam_stage_local"
    assert descriptor["transition"] == "A_previous/A_current"
    assert [entry["mode"] for entry in descriptor["stages"]] == [
        "estimated",
        "estimated",
    ]
    expected_upstream = {
        "adam/paired_batches",
        "adam/mid/identity",
        "adam/mid/statistics",
        "adam/mid/tensor_manifest",
        "adam/sft/identity",
        "adam/sft/statistics",
        "adam/sft/tensor_manifest",
    }
    assert {
        key for key in identity.upstream_digests if key.startswith("adam/")
    } == expected_upstream

    damping = config.method.damping_sweep[0]
    scales = []
    for stage in config.stages:
        moment_dir = layout.adam_moments / stage.name
        moment = ShardManifest.load(moment_dir).read_rows(moment_dir)[
            "features"
        ][0].float()
        scales.append(
            (
                moment.sqrt()
                + config.adam_moment_estimator.optimizer_epsilon
                + damping
            ).rsqrt()
        )
    query_rows = ShardManifest.load(layout.queries).read_rows(layout.queries)[
        "features"
    ].float()
    resolved_lr = {"mid": 1e-2 + 8e-3 + 5e-3, "sft": 5e-3 + 3e-3 + 1e-3}
    segments = []
    trains = []
    for stage_index, stage in enumerate(config.stages):
        fisher = _stage_statistics(layout.factors / stage.name)
        segments.append(
            SourceSegment(
                stage.name,
                DiagonalCurvature(
                    (fisher * scales[stage_index].square()).numpy(),
                    basis_descriptor={
                        "coordinates": "adam_stage_local",
                        "stage": stage.name,
                    },
                ),
                resolved_lr[stage.name],
                transition_to_previous=(
                    None
                    if stage_index == 0
                    else (scales[stage_index - 1] / scales[stage_index]).numpy()
                ),
            )
        )
        rows_dir = layout.rows / stage.name
        trains.append(ShardManifest.load(rows_dir).read_rows(rows_dir))
    transformed = SourceScorer(segments).transformed_queries(
        (query_rows * scales[-1]).numpy()
    )
    completeness = json.loads((layout.scores / "score_manifest.json").read_text())
    for stage_index, stage in enumerate(config.stages):
        expected = (
            transformed[stage_index]
            @ (
                trains[stage_index]["features"] * scales[stage_index]
            ).numpy().T
        ) / stage.n_examples
        saved = load_file(
            str(
                layout.scores
                / completeness["entries"][f"{stage.name}__damping-0"]["file"]
            )
        )
        assert torch.allclose(saved["scores"], torch.from_numpy(expected), atol=1e-5)


def test_score_source_loads_each_captured_stage_moment_once_across_damping(
    chain, monkeypatch
):
    config, _ = _complete_chain(
        chain,
        monkeypatch,
        method={
            "basis": "adam",
            "curvature": "fisher",
            "damping_sweep": [0.1, 0.2, 0.3],
        },
    )
    calls = 0

    def counting_load(path):
        nonlocal calls
        calls += 1
        return load_optimizer_snapshot(path)

    monkeypatch.setattr(
        "scimt.train.attribution_snapshot.load_optimizer_snapshot",
        counting_load,
    )
    _run(runner.score_source(config))
    assert calls == len(config.stages)


def test_completed_estimated_adam_scores_survive_moment_shard_eviction(
    chain, monkeypatch
):
    config, _ = _complete_chain(
        chain, monkeypatch, **_estimated_adam_overrides(chain)
    )
    first = _run(runner.score_source(config))
    assert first.outputs[0].skipped is False
    layout = runner.run_layout(config.output_dir)

    for stage in config.stages:
        for shard in (layout.adam_moments / stage.name).glob(
            "shard_*.safetensors"
        ):
            shard.unlink()
    again = _run(runner.score_source(config))
    assert again.outputs[0].skipped is True

    next(layout.scores.glob("scores__*.safetensors")).unlink()
    with pytest.raises(ArtifactIntegrityError, match="absent entry"):
        _run(runner.score_source(config))


def test_completed_captured_adam_scores_survive_snapshot_shard_eviction(
    chain, monkeypatch
):
    config, _ = _complete_chain(
        chain,
        monkeypatch,
        method={
            "basis": "adam",
            "curvature": "fisher",
            "damping_sweep": [0.1],
        },
    )
    _run(runner.score_source(config))
    for stage in config.stages:
        for shard in Path(stage.optimizer_snapshot).glob(
            "exp_avg_sq-*.safetensors"
        ):
            shard.unlink()

    resumed = _run(runner.score_source(config))
    assert resumed.outputs[0].skipped is True


def test_incomplete_captured_adam_receipt_reloads_snapshots_and_recomputes(
    chain, monkeypatch
):
    config, _ = _complete_chain(
        chain,
        monkeypatch,
        method={
            "basis": "adam",
            "curvature": "fisher",
            "damping_sweep": [0.1],
        },
    )
    _run(runner.score_source(config))
    marker = runner.run_layout(config.output_dir).scores / "score_manifest.json"
    completeness = json.loads(marker.read_text())
    completeness["expected"] = []
    marker.write_text(json.dumps(completeness))

    resumed = _run(runner.score_source(config))

    assert resumed.outputs[0].skipped is False
    repaired = json.loads(marker.read_text())
    assert repaired["expected"] == ["mid__damping-0", "sft__damping-0"]


def test_completed_estimated_adam_receipt_refuses_small_manifest_drift(
    chain, monkeypatch
):
    config, _ = _complete_chain(
        chain, monkeypatch, **_estimated_adam_overrides(chain)
    )
    _run(runner.score_source(config))
    statistics_path = (
        runner.run_layout(config.output_dir).adam_moments
        / "mid"
        / "statistics.json"
    )
    statistics = json.loads(statistics_path.read_text())
    statistics["ema_mean_cosine"] += 0.01
    statistics_path.write_text(json.dumps(statistics))

    with pytest.raises(ArtifactIntegrityError, match="provenance drift"):
        _run(runner.score_source(config))


def test_estimated_adam_statistics_tampering_is_refused_before_first_score(
    chain, monkeypatch
):
    config, _ = _complete_chain(
        chain, monkeypatch, **_estimated_adam_overrides(chain)
    )
    statistics_path = (
        runner.run_layout(config.output_dir).adam_moments
        / "mid"
        / "statistics.json"
    )
    statistics = json.loads(statistics_path.read_text())
    statistics["ema_mean_cosine"] = max(
        -1.0, min(1.0, statistics["ema_mean_cosine"] - 0.01)
    )
    statistics_path.write_text(json.dumps(statistics))

    with pytest.raises(ArtifactIntegrityError, match="auxiliary.*digest mismatch"):
        _run(runner.score_source(config))


def test_estimated_adam_resume_validates_statistics_schema(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config(**_estimated_adam_overrides(chain))
    _run(runner.estimate_adam(config))
    statistics_path = (
        runner.run_layout(config.output_dir).adam_moments
        / "mid"
        / "statistics.json"
    )
    statistics = json.loads(statistics_path.read_text())
    statistics["ema_mean_relative_l2"] += 0.01
    statistics_path.write_text(json.dumps(statistics))

    with pytest.raises(ArtifactIntegrityError, match="auxiliary.*digest mismatch"):
        _run(runner.estimate_adam(config))


def test_score_source_stage_local_adam_requires_query_at_final_checkpoint(
    chain, monkeypatch
):
    overrides = _estimated_adam_overrides(chain)
    overrides["query"] = {
        **chain.payload["query"],
        "checkpoint": chain.payload["stages"][0]["checkpoint"],
    }
    config, _ = _complete_chain(chain, monkeypatch, **overrides)

    with pytest.raises(runner.RunnerError, match="query checkpoint.*final"):
        _run(runner.score_source(config))


def test_score_source_ekfac_curvature_matches_manual_operator_chain(
    chain, monkeypatch
):
    """curvature=ekfac scoring equals an independently chained
    EKFACCurvature computation with sigma+damping (right-to-left)."""
    pytest.importorskip("kronfluence")
    damping = 0.3
    config, _ = _complete_chain(
        chain, monkeypatch,
        method={"curvature": "ekfac", "damping_sweep": [damping]},
    )
    _run(runner.score_source(config))
    layout = runner.run_layout(config.output_dir)
    completeness = json.loads(
        (layout.scores / "score_manifest.json").read_text()
    )

    from scimt.data_attribution.ekfac import load_ekfac
    from scimt.data_attribution.source import (
        EKFACCurvature,
        f_backward,
        f_segment,
    )

    query_rows = ShardManifest.load(layout.queries).read_rows(
        layout.queries)["features"].float().numpy()
    resolved_lr = {"mid": 1e-2 + 8e-3 + 5e-3, "sft": 5e-3 + 3e-3 + 1e-3}
    operators = {}
    for stage in config.stages:
        ekfac_dir = layout.factors / stage.name / "ekfac"
        manifest = ParameterManifest.load(ekfac_dir)
        operators[stage.name] = EKFACCurvature(
            load_ekfac(ekfac_dir, manifest), manifest
        )
    u_sft = operators["sft"].apply_fn(
        query_rows, lambda ev: f_segment(ev + damping, resolved_lr["sft"])
    )
    transported = operators["sft"].apply_fn(
        query_rows, lambda ev: f_backward(ev + damping, resolved_lr["sft"])
    )
    u_mid = operators["mid"].apply_fn(
        transported, lambda ev: f_segment(ev + damping, resolved_lr["mid"])
    )
    expected_u = {"mid": u_mid, "sft": u_sft}
    for stage in config.stages:
        rows_dir = layout.rows / stage.name
        train = ShardManifest.load(rows_dir).read_rows(rows_dir)
        expected = (
            expected_u[stage.name] @ train["features"].float().numpy().T
        ) / stage.n_examples
        entry = completeness["entries"][f"{stage.name}__damping-0"]
        saved = load_file(str(layout.scores / entry["file"]))
        assert torch.allclose(
            saved["scores"], torch.from_numpy(expected), atol=1e-5
        )


def test_score_source_fisher_basis_matches_closed_form_transport(
    chain, monkeypatch
):
    """Fisher-basis mirror of the adam closed-form test: T = (F_last + eps +
    damping)^-1/2 from the LAST stage's fitted statistics, rows -> T rows,
    curvature -> T^2 F_l, 1/N once; identity records source stage + epsilon."""
    damping = 0.2
    config, _ = _complete_chain(
        chain, monkeypatch,
        method={"basis": "fisher", "curvature": "fisher",
                "damping_sweep": [damping]},
    )
    _run(runner.score_source(config))
    layout = runner.run_layout(config.output_dir)
    completeness = json.loads(
        (layout.scores / "score_manifest.json").read_text()
    )
    epsilon = 1e-8  # DiagonalMetric.from_statistics default
    fisher_last = _stage_statistics(layout.factors / "sft")
    diag = (fisher_last + epsilon + damping).pow(-0.5)
    query_rows = ShardManifest.load(layout.queries).read_rows(
        layout.queries)["features"].float()
    resolved_lr = {"mid": 1e-2 + 8e-3 + 5e-3, "sft": 5e-3 + 3e-3 + 1e-3}
    segments, trains = [], []
    for stage in config.stages:
        fisher = _stage_statistics(layout.factors / stage.name)
        segments.append(SourceSegment(
            stage.name,
            DiagonalCurvature((fisher * diag.pow(2)).double().numpy(),
                              basis_descriptor={"coordinates": "fisher_diag"}),
            resolved_lr[stage.name],
        ))
        rows_dir = layout.rows / stage.name
        trains.append(ShardManifest.load(rows_dir).read_rows(rows_dir))
    scorer = SourceScorer(segments)
    transformed = scorer.transformed_queries((query_rows * diag).numpy())
    for stage_index, stage in enumerate(config.stages):
        expected = (
            transformed[stage_index]
            @ (trains[stage_index]["features"].float() * diag).numpy().T
        ) / stage.n_examples
        entry = completeness["entries"][f"{stage.name}__damping-0"]
        saved = load_file(str(layout.scores / entry["file"]))
        assert torch.allclose(
            saved["scores"], torch.from_numpy(expected), atol=1e-5
        )
    identity = read_identity(layout.scores)
    assert identity.basis_descriptor["coordinates"] == "fisher_diag"
    assert identity.basis_descriptor["source_stage"] == "sft"
    assert identity.basis_descriptor["epsilon"] == pytest.approx(epsilon)


def test_score_source_refusals(chain, monkeypatch):
    config, _ = _complete_chain(chain, monkeypatch)

    # Missing query rows.
    layout = runner.run_layout(config.output_dir)
    import shutil

    saved_queries = chain.tmp_path / "saved-queries"
    shutil.move(layout.queries, saved_queries)
    with pytest.raises((runner.RunnerError, FileNotFoundError),
                       match="quer"):
        _run(runner.score_source(config))
    shutil.move(saved_queries, layout.queries)

    # Incomplete shards: a manifest naming an absent shard refuses.
    rows_dir = layout.rows / "mid"
    shard = next(rows_dir.glob("shard_*.safetensors"))
    saved_bytes = shard.read_bytes()
    shard.unlink()
    with pytest.raises(ArtifactIntegrityError, match="absent shard"):
        _run(runner.score_source(config))
    shard.write_bytes(saved_bytes)

    # Stale upstream: rows computed under a different resolved data config.
    drifted, _ = chain.config(data={"sequence_length": 10})
    with pytest.raises(runner.RunnerError, match="resolved_config"):
        _run(runner.score_source(drifted))

    # Mixed bases across transported segments: forge the stored basis
    # descriptor of one row artifact.
    identity_path = rows_dir / "artifact_identity.json"
    stored = json.loads(identity_path.read_text())
    stored["basis_descriptor"]["coordinates"] = "logra_B"
    identity_path.write_text(json.dumps(stored))
    with pytest.raises(runner.RunnerError, match="basis"):
        _run(runner.score_source(config))

    # Missing Adam state at scoring time.
    sft_snapshot = Path(chain.payload["stages"][1]["optimizer_snapshot"])
    (sft_snapshot / "optimizer_manifest.json").unlink()
    adam_config, _ = chain.config(
        method={"basis": "adam", "curvature": "fisher"},
        output_dir=str(chain.tmp_path / "missing-adam-out"),
    )
    with pytest.raises(StageResolutionError, match="snapshot"):
        _run(runner.score_source(adam_config))


def test_score_source_refuses_recaptured_adam_snapshot(chain, monkeypatch):
    """The score-time Adam re-check: a snapshot re-captured under a different
    parameter selection between compute-rows and score-source is a real
    coordinate change and must refuse before any moment tensor is used."""
    config, _ = _complete_chain(
        chain, monkeypatch,
        method={"basis": "adam", "curvature": "fisher", "damping_sweep": [0.1]},
    )
    import shutil

    sft_ck = (Path(chain.payload["stages"][1]["checkpoint"])
              / "checkpoints" / "checkpoint-3")
    shutil.rmtree(Path(chain.payload["stages"][1]["optimizer_snapshot"]))
    _snapshot_for(sft_ck, include=[r"head\.weight"])
    with pytest.raises(runner.RunnerError, match="parameter manifest"):
        _run(runner.score_source(config))


def test_score_source_factor_operator_refusals(chain, monkeypatch):
    """EK-FAC factor consumption refuses: a missing completion marker, a
    marker bound to a different identity, and factor bytes drifted after
    completion."""
    pytest.importorskip("kronfluence")
    import numpy as np

    config, _ = _complete_chain(
        chain, monkeypatch,
        method={"curvature": "ekfac", "damping_sweep": [0.1]},
    )
    layout = runner.run_layout(config.output_dir)
    marker = layout.factors / "mid" / "factors_complete.json"
    marker_body = marker.read_text()

    marker.unlink()
    with pytest.raises(runner.RunnerError, match="completion marker"):
        _run(runner.score_source(config))

    forged = json.loads(marker_body)
    forged["identity_digest"] = "e" * 64
    marker.write_text(json.dumps(forged))
    with pytest.raises(ArtifactIntegrityError, match="stored artifact identity"):
        _run(runner.score_source(config))
    marker.write_text(marker_body)

    lam_path = layout.factors / "mid" / "ekfac" / "linear" / "head" / "lam.npy"
    lam = np.load(lam_path)
    np.save(lam_path, lam + 0.5)  # same shape/domain, different bytes
    with pytest.raises(ArtifactIntegrityError, match="changed after"):
        _run(runner.score_source(config))


def test_score_source_refuses_ekfac_basis_and_logra_rows(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config(method={"basis": "ekfac", "curvature": "ekfac"})
    with pytest.raises(runner.RunnerError, match="basis 'ekfac'"):
        _run(runner.score_source(config))
    config, _ = chain.config(
        method={"logra": {"rank": 2, "init": "random", "seed": 3,
                          "targets": "head"}}
    )
    with pytest.raises(runner.RunnerError, match="[Ll]o[Gg]ra"):
        _run(runner.score_source(config))


# ---------------------------------------------------------------- second order
SECOND_ORDER = {
    "checkpoint": "sft",
    "pairs": [[0, 1], [0, 0]],
    "hessian_kind": "true",
    "metric": "none",
    "sweep_stage": "mid",
}


def test_build_directions_requires_declared_checkpoint(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config()
    with pytest.raises(runner.RunnerError, match="second_order"):
        _run(runner.build_directions(config))
    with pytest.raises(runner.RunnerError, match="second_order"):
        _run(runner.sweep_jvp(config))


def test_build_directions_refuses_ekfac_metric(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config(
        second_order={**SECOND_ORDER, "metric": "ekfac"}
    )
    with pytest.raises(runner.RunnerError, match="DiagonalMetric"):
        _run(runner.build_directions(config))


def _statistics_artifact(directory: Path, estimator: str, values: torch.Tensor,
                         manifest_digest: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    statistics = {key: "fixture" for key in REQUIRED_PROVENANCE}
    statistics["parameter_manifest_digest"] = manifest_digest
    statistics["number_of_gradient_samples"] = 4
    statistics["estimator"] = estimator
    (directory / "statistics.json").write_text(json.dumps(statistics))
    save_file({"values": values}, str(directory / "values.safetensors"))
    return directory


def test_build_directions_refuses_rank1_factored_statistics(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    model = TinyLM().float()
    manifest = ParameterManifest.from_model(model, "TinyLM")
    stats_dir = _statistics_artifact(
        chain.tmp_path / "stats-rank1", "rank1",
        torch.rand(manifest.included_numel) + 0.1, manifest.digest(),
    )
    config, _ = chain.config(second_order={
        **SECOND_ORDER,
        "metric_derivative": {"statistics": str(stats_dir)},
    })
    with pytest.raises(runner.RunnerError, match="rank1"):
        _run(runner.build_directions(config))


def test_build_directions_adds_full_estimator_metric_derivative_term(
    chain, monkeypatch
):
    """The positive metric-derivative path: a 'full'-estimator statistics
    artifact contributes a finite, nonzero additional term."""
    _install_tiny_loaders(monkeypatch)
    model = TinyLM().float()
    manifest = ParameterManifest.from_model(model, "TinyLM")
    generator = torch.Generator().manual_seed(7)
    values = torch.rand(manifest.included_numel, generator=generator) + 0.1
    stats_dir = _statistics_artifact(
        chain.tmp_path / "stats-full", "full", values, manifest.digest()
    )
    plain_config, _ = chain.config(
        output_dir=str(chain.tmp_path / "attr-plain"),
        second_order=SECOND_ORDER,
    )
    derivative_config, _ = chain.config(second_order={
        **SECOND_ORDER,
        "metric_derivative": {"statistics": str(stats_dir),
                              "n_estimation_sequences": 2},
    })
    plain = _run(runner.build_directions(plain_config))
    with_derivative = _run(runner.build_directions(derivative_config))
    plain_rows = ShardManifest.load(plain.outputs[0].directory).read_rows(
        plain.outputs[0].directory)["features"]
    derivative_rows = ShardManifest.load(
        with_derivative.outputs[0].directory
    ).read_rows(with_derivative.outputs[0].directory)["features"]
    assert derivative_rows.shape == plain_rows.shape
    assert bool(torch.isfinite(derivative_rows).all())
    assert not torch.allclose(derivative_rows, plain_rows)
    columns = json.loads(
        (with_derivative.outputs[0].directory / "direction_columns.json")
        .read_text()
    )
    assert all(column["metric_derivative"] is True for column in columns)
    identity = read_identity(with_derivative.outputs[0].directory)
    assert "metric_derivative_statistics" in identity.upstream_digests

    # Regenerated statistics at the same path change the directions identity:
    # a rerun refuses instead of silently skipping the stale artifact.
    _statistics_artifact(
        chain.tmp_path / "stats-full", "full",
        torch.rand(manifest.included_numel,
                   generator=torch.Generator().manual_seed(8)) + 0.2,
        manifest.digest(),
    )
    with pytest.raises(IdentityMismatchError, match="upstream_digests"):
        _run(runner.build_directions(derivative_config))


def test_build_directions_fisher_metric_validates_factor_scope(chain, monkeypatch):
    """The pair metric consumes factors/<stage> under the same upstream
    validation score-source performs: a factors-config edit is refused."""
    _install_tiny_loaders(monkeypatch)
    fisher_second = {**SECOND_ORDER, "metric": "fisher"}
    config, _ = chain.config(second_order=fisher_second)
    _run(runner.fit_factors(config))
    report = _run(runner.build_directions(config))
    identity = read_identity(report.outputs[0].directory)
    assert "pair_metric_factors" in identity.upstream_digests
    drifted, _ = chain.config(second_order=fisher_second,
                              factors={"samples": 7})
    with pytest.raises(runner.RunnerError, match="factors/sft"):
        _run(runner.build_directions(drifted))


def test_build_directions_resume_never_duplicates_rows(chain, monkeypatch):
    """Interrupt after the first committed direction shard, resume, and get
    exactly the uninterrupted result — committed pairs are skipped, never
    re-appended (a duplicate sample_id would poison every later read)."""
    _install_tiny_loaders(monkeypatch)
    reference_config, _ = chain.config(
        output_dir=str(chain.tmp_path / "dirs-reference"),
        second_order=SECOND_ORDER,
        data={"rows_per_shard": 1},
    )
    reference = _run(runner.build_directions(reference_config))
    reference_rows = ShardManifest.load(
        reference.outputs[0].directory
    ).read_rows(reference.outputs[0].directory)

    interrupted_config, _ = chain.config(
        output_dir=str(chain.tmp_path / "dirs-interrupted"),
        second_order=SECOND_ORDER,
        data={"rows_per_shard": 1},
    )
    real_append = runner.ArtifactWriter.append
    calls = {"n": 0}

    def exploding_append(self, **kwargs):
        real_append(self, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(runner.ArtifactWriter, "append", exploding_append)
    with pytest.raises(RuntimeError, match="interruption"):
        _run(runner.build_directions(interrupted_config))
    monkeypatch.setattr(runner.ArtifactWriter, "append", real_append)
    directions_dir = runner.run_layout(interrupted_config.output_dir).directions
    assert len(list(directions_dir.glob("shard_*.safetensors"))) >= 1
    resumed = _run(runner.build_directions(interrupted_config))
    resumed_rows = ShardManifest.load(
        resumed.outputs[0].directory
    ).read_rows(resumed.outputs[0].directory)  # raises on duplicate ids
    assert torch.equal(resumed_rows["sample_ids"], reference_rows["sample_ids"])
    assert torch.equal(resumed_rows["features"], reference_rows["features"])
    columns = json.loads(
        (resumed.outputs[0].directory / "direction_columns.json").read_text()
    )
    assert [column["pair"] for column in columns] == [[0, 1], [0, 0]]


def test_fit_factors_fisher_finalizes_after_seal_crash_without_duplicates(
    chain, monkeypatch
):
    """The seal->finalize window: if the statistic row was committed but the
    manifest write crashed, the rerun publishes the manifest without
    re-appending."""
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config()
    real_finalize = runner.ArtifactWriter.finalize
    state = {"crash": True}

    def exploding_finalize(self):
        if state["crash"]:
            state["crash"] = False
            raise RuntimeError("simulated crash before manifest publish")
        return real_finalize(self)

    monkeypatch.setattr(runner.ArtifactWriter, "finalize", exploding_finalize)
    with pytest.raises(RuntimeError, match="before manifest"):
        _run(runner.fit_factors(config))
    monkeypatch.setattr(runner.ArtifactWriter, "finalize", real_finalize)
    mid_dir = runner.run_layout(config.output_dir).factors / "mid"
    assert not (mid_dir / ShardManifest.FILENAME).is_file()  # sealed, unpublished
    report = _run(runner.fit_factors(config))
    output = {o.name: o for o in report.outputs}["factors/mid"]
    manifest = ShardManifest.load(output.directory)
    assert manifest.total_rows == 1
    manifest.read_rows(output.directory)  # raises on duplicate sample ids


def test_directions_and_jvp_sweep_at_declared_checkpoint(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config(second_order=SECOND_ORDER)
    directions_report = _run(runner.build_directions(config))
    (directions_output,) = directions_report.outputs
    directions_manifest = ShardManifest.load(directions_output.directory)
    assert directions_manifest.total_rows == 2  # one per declared pair
    assert directions_manifest.feature_dim == 160
    columns = json.loads(
        (directions_output.directory / "direction_columns.json").read_text()
    )
    assert [column["pair"] for column in columns] == [[0, 1], [0, 0]]
    assert all(column["hessian_kind"] == "true" for column in columns)

    jvp_report = _run(runner.sweep_jvp(config))
    (jvp_output,) = jvp_report.outputs
    jvp_manifest = ShardManifest.load(jvp_output.directory)
    assert jvp_manifest.feature_dim == 2

    # Cross-check one batch against a direct jvp_sweep call.
    checkpoint_dir = (Path(chain.payload["stages"][1]["checkpoint"])
                      / "checkpoints" / "checkpoint-3")
    model = TinyLM().float()
    model.load_state_dict(load_file(str(checkpoint_dir / "model.safetensors")))
    manifest = ParameterManifest.from_model(model, "TinyLM")
    directions = directions_manifest.read_rows(
        directions_output.directory)["features"].float()
    dataset = _adapter_for(chain.payload["stages"][0]["dataset"],
                           "midtraining", "per_token")
    expected = []
    for batch in dataset.iter_batches(2):
        expected.append(jvp_sweep(model, manifest, batch, directions,
                                  reduction="per_token"))
    expected_features = torch.cat(expected)
    stored = jvp_manifest.read_rows(jvp_output.directory)["features"].float()
    assert torch.allclose(stored, expected_features, atol=1e-5)

    # The sweep refuses tampered upstream directions.
    identity_path = directions_output.directory / "artifact_identity.json"
    body = json.loads(identity_path.read_text())
    body["seeds"]["run"] = 99
    identity_path.write_text(json.dumps(body))
    fresh = chain.tmp_path / "jvp-fresh"
    import shutil

    shutil.move(jvp_output.directory, fresh)
    with pytest.raises(runner.RunnerError, match="direction"):
        _run(runner.sweep_jvp(config))


# ------------------------------------------------------------------ summarize
def test_summarize_refuses_partial_unless_saved_allow_partial(chain, monkeypatch):
    config, _ = _complete_chain(chain, monkeypatch)
    _run(runner.score_source(config))
    summary = _run(runner.summarize(config))
    assert summary["complete"] is True
    assert summary["schema_version"] == 1
    assert summary["sections"]["scores"]["complete"] is True
    layout = runner.run_layout(config.output_dir)
    assert (layout.summary / "summary.json").is_file()
    assert (layout.summary / "summary.md").is_file()

    # Remove one requested score entry -> the matrix is partial.
    completeness = json.loads(
        (layout.scores / "score_manifest.json").read_text()
    )
    first = sorted(completeness["entries"])[0]
    (layout.scores / completeness["entries"][first]["file"]).unlink()
    with pytest.raises(runner.RunnerError, match="allow_partial"):
        _run(runner.summarize(config))

    # An ad-hoc allow_partial on the passed config does not count: the saved
    # resolved config (run.json) is the authority.
    ad_hoc, _ = chain.config(allow_partial=True)
    with pytest.raises(runner.RunnerError, match="allow_partial"):
        _run(runner.summarize(ad_hoc))


def test_summarize_refuses_without_a_run_ledger(chain, monkeypatch):
    config, _ = chain.config(output_dir=str(chain.tmp_path / "never-ran"))
    with pytest.raises(runner.RunnerError, match="run ledger"):
        _run(runner.summarize(config))


def test_summarize_saved_allow_partial_survives_ad_hoc_de_escalation(
    chain, monkeypatch
):
    """The saved resolved config stays the sole authority in BOTH directions:
    a run declared partial-tolerant summarizes partially even when the passed
    config omits the flag."""
    out = chain.tmp_path / "attr-de-escalate"
    _complete_chain(chain, monkeypatch, output_dir=str(out),
                    allow_partial=True)
    passed, _ = chain.config(output_dir=str(out))  # allow_partial defaults False
    assert passed.allow_partial is False
    summary = _run(runner.summarize(passed))
    assert summary["complete"] is False  # scores never ran
    assert summary["allow_partial"] is True  # saved authority


def test_summarize_honors_allow_partial_saved_from_the_start(chain, monkeypatch):
    out = chain.tmp_path / "attr-partial"
    config, _ = _complete_chain(chain, monkeypatch, output_dir=str(out),
                                allow_partial=True)
    # No score-source run at all: the requested matrix is entirely missing.
    summary = _run(runner.summarize(config))
    assert summary["complete"] is False
    assert summary["allow_partial"] is True
    assert summary["sections"]["scores"]["complete"] is False
    assert summary["sections"]["rows"]["counts"]["mid"] > 0


def test_summarize_accepts_complete_streaming_scores(chain, monkeypatch):
    """Streaming runs satisfy summarize without row shards or materialized
    scores: same numbers as the materialized summary, labeled streaming."""
    import shutil

    config, _ = _complete_chain(chain, monkeypatch)
    _run(runner.score_source(config))
    materialized_summary = _run(runner.summarize(config))
    # Materialized-path summaries are byte-unchanged: no source labels.
    assert "source" not in materialized_summary["sections"]["scores"]
    assert "source" not in materialized_summary["sections"]["rows"]

    _run(runner.score_source_streaming(config))
    layout = runner.run_layout(config.output_dir)
    # Reduce the layout to what a streaming-only run produces: no
    # materialized scores, no row shards.
    shutil.rmtree(layout.scores)
    for stage_dir in layout.rows.iterdir():
        shutil.rmtree(stage_dir)

    summary = _run(runner.summarize(config))
    assert summary["complete"] is True
    scores = summary["sections"]["scores"]
    assert scores["complete"] is True
    assert scores["source"] == "streaming"
    assert scores["missing"] == []
    assert (
        scores["counts"]
        == materialized_summary["sections"]["scores"]["counts"]
    )
    rows_section = summary["sections"]["rows"]
    assert rows_section["complete"] is True
    assert "streaming" in rows_section["source"]
    assert rows_section["counts"] == {name: 0 for name in ("mid", "sft")}


def test_summarize_refuses_incomplete_streaming_scores(chain, monkeypatch):
    """A genuinely partial streaming artifact is still a refusal — the
    streaming fallback only fires on a complete committed manifest."""
    import shutil

    config, _ = _complete_chain(chain, monkeypatch)
    _run(runner.score_source_streaming(config))
    layout = runner.run_layout(config.output_dir)
    shutil.rmtree(layout.scores, ignore_errors=True)
    for stage_dir in layout.rows.iterdir():
        shutil.rmtree(stage_dir)
    completeness = json.loads(
        (layout.streaming_scores / runner._SCORE_MANIFEST_FILE).read_text()
    )
    first = sorted(completeness["entries"])[0]
    (layout.streaming_scores / completeness["entries"][first]["file"]).unlink()
    with pytest.raises(runner.RunnerError, match="allow_partial"):
        _run(runner.summarize(config))


def test_summarize_streaming_only_run_without_compute_rows(chain, monkeypatch):
    """The streaming-only phase chain (no compute-rows, no score-source)
    summarizes complete — the exact pipeline the pod driver runs at full
    coverage, where materialized row shards are infeasible."""
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config()
    _run(runner.fit_factors(config))
    _run(runner.build_queries(config))
    if config.adam_moment_estimator is not None:
        _run(runner.estimate_adam(config))
    _run(runner.score_source_streaming(config))
    summary = _run(runner.summarize(config))
    assert summary["complete"] is True
    assert summary["sections"]["scores"]["source"] == "streaming"
    assert summary["sections"]["rows"]["complete"] is True


def test_run_ledger_records_and_refuses_drift(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config()
    _run(runner.fit_factors(config))
    ledger = json.loads(
        (Path(chain.payload["output_dir"]) / "run.json").read_text()
    )
    assert ledger["resolved_config"] == config.resolved()
    assert ledger["source_commit"] == (
        "ca9689a497b921dc516feb663a83269c4a588bbc"
    )
    assert ledger["scimt_commit"]
    events = (Path(chain.payload["output_dir"]) / "events.jsonl").read_text()
    assert "fit-factors" in events
    drifted, _ = chain.config(seed=1)
    with pytest.raises(runner.RunnerError, match="run ledger"):
        _run(runner.fit_factors(drifted))


def test_phase_scopes_treat_new_null_fields_as_legacy_missing_fields(chain):
    config, _ = chain.config()
    current = config.resolved()
    legacy = json.loads(json.dumps(current))
    legacy.pop("adam_moment_estimator")
    for stage in legacy["stages"]:
        stage.pop("training_dataset")

    for phase in ("fit-factors", "compute-rows", "score-source"):
        assert runner._phase_scopes(current, phase) == runner._phase_scopes(
            legacy, phase
        )


def test_artifact_identities_carry_full_provenance(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config()
    report = _run(runner.compute_rows(config))
    output = {o.name: o for o in report.outputs}["rows/mid"]
    identity = read_identity(output.directory)
    from scimt.data_attribution import SOURCE_COMMIT
    from scimt.data_attribution.stages import artifact_digest, resolve_stage

    assert identity.producing_command == "scimt-attribution compute-rows"
    assert identity.source_commit == SOURCE_COMMIT
    assert identity.scimt_commit and identity.scimt_commit != "unknown"
    resolved = resolve_stage(config.stages[0])
    assert identity.checkpoint_reference == str(resolved.checkpoint_dir)
    assert identity.checkpoint_digest == artifact_digest(resolved.checkpoint_dir)
    # The composite dataset fingerprint: raw source bytes + tokenizer content.
    assert json.loads(identity.dataset_fingerprint) == {
        "source": resolved.dataset_digest,
        "tokenizer_content": runner._tokenizer_content_digest(
            Path(chain.payload["tokenizer"])
        ),
    }
    assert identity.loss_convention["reduction"] == "per_token"
    assert identity.dtype == "float32"
    assert identity.seeds["run"] == 0
    assert identity.basis_descriptor["manifest_digest"] == (
        identity.parameter_manifest_digest
    )
    assert identity.resolved_config["stage"]["name"] == "mid"
    # Execution-only geometry stays out of artifact identity: re-chunking
    # must not invalidate committed rows.
    regeometried, _ = chain.config(data={"batch_size": 4, "vjp_chunk_size": 2,
                                         "rows_per_shard": 64})
    again = _run(runner.compute_rows(regeometried))
    assert all(o.skipped for o in again.outputs)


# ------------------------------------------------- Adam-conditioned EK-FAC --


class _MethodStub:
    def __init__(self, curvature, basis, logra=None):
        self.curvature = curvature
        self.basis = basis
        self.logra = logra


class _ConfigStub:
    def __init__(self, curvature, basis, logra=None):
        self.method = _MethodStub(curvature, basis, logra)


def test_scorable_method_refusal_matrix():
    # basis adam over raw-fitted EK-FAC: refused, message points at the
    # conditioned mode. basis fisher: refused, and the message must NOT
    # point at ekfac_adam (which requires basis adam and would refuse in
    # turn) — the Fisher-diagonal conditioned analogue is out of scope.
    with pytest.raises(runner.RunnerError, match="ekfac_adam"):
        runner._require_scorable_method(_ConfigStub("ekfac", "adam"))
    with pytest.raises(runner.RunnerError, match="out of scope"):
        runner._require_scorable_method(_ConfigStub("ekfac", "fisher"))
    # conditioned factors only consumable through basis adam.
    for basis in ("raw", "fisher"):
        with pytest.raises(runner.RunnerError, match="basis 'adam'"):
            runner._require_scorable_method(_ConfigStub("ekfac_adam", basis))
    # unchanged refusals.
    with pytest.raises(runner.RunnerError, match="basis 'ekfac'"):
        runner._require_scorable_method(_ConfigStub("ekfac", "ekfac"))
    with pytest.raises(runner.RunnerError, match="[Gg][Gg][Nn]"):
        runner._require_scorable_method(_ConfigStub("ggn", "raw"))
    # the three supported combos pass.
    runner._require_scorable_method(_ConfigStub("ekfac", "raw"))
    runner._require_scorable_method(_ConfigStub("fisher", "adam"))
    runner._require_scorable_method(_ConfigStub("fisher", "fisher"))
    runner._require_scorable_method(_ConfigStub("ekfac_adam", "adam"))


def _ekfac_adam_yaml_config(tmp_path, *, method_extra=None):
    payload = {
        "stages": [
            {
                "name": "midtrain",
                "checkpoint": "ckpts/mid",
                "dataset": "data/mid.jsonl",
                "objective": "midtraining",
                "n_examples": 64,
                "weight_decay": 0.0,
            },
            {
                "name": "sft",
                "checkpoint": "ckpts/sft",
                "dataset": "data/sft.jsonl",
                "objective": "sft",
                "lr_steps": 1.0,
                "lr_steps_provenance": "test",
                "n_examples": 8,
                "weight_decay": 0.0,
            },
        ],
        "query": {
            "checkpoint": "ckpts/sft",
            "dataset": "data/query.jsonl",
            "objective": "sft",
        },
        "method": {
            "curvature": "ekfac_adam",
            "basis": "adam",
            "conditioning_damping": 0.25,
            **(method_extra or {}),
        },
        "adam_moment_estimator": {
            "dataset": "data/mid.jsonl",
            "objective": "midtraining",
            "num_batches": 2,
            "global_batch_size": 2,
            "micro_batch_size": 1,
            "beta2": 0.999,
            "optimizer_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "seed": 7,
        },
        "output_dir": str(tmp_path / "attr-run"),
    }
    path = tmp_path / "attribution.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_attribution_config(path)


def _old_mode_yaml_config(tmp_path):
    payload = {
        "stages": [
            {
                "name": "midtrain",
                "checkpoint": "ckpts/mid",
                "dataset": "data/mid.jsonl",
                "objective": "midtraining",
                "n_examples": 64,
                "weight_decay": 0.0,
            },
        ],
        "query": {
            "checkpoint": "ckpts/mid",
            "dataset": "data/query.jsonl",
            "objective": "sft",
        },
        "method": {"curvature": "ekfac", "basis": "raw"},
        "output_dir": str(tmp_path / "attr-run"),
    }
    path = tmp_path / "attribution-old.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_attribution_config(path)


def _fisher_adam_yaml_config(tmp_path):
    payload = {
        "stages": [
            {
                "name": "midtrain",
                "checkpoint": "ckpts/mid",
                "dataset": "data/mid.jsonl",
                "objective": "midtraining",
                "n_examples": 64,
                "weight_decay": 0.0,
            },
        ],
        "query": {
            "checkpoint": "ckpts/mid",
            "dataset": "data/query.jsonl",
            "objective": "sft",
        },
        "method": {"curvature": "fisher", "basis": "adam"},
        "adam_moment_estimator": {
            "dataset": "data/mid.jsonl",
            "objective": "midtraining",
            "num_batches": 2,
            "global_batch_size": 4,
            "micro_batch_size": 1,
            "beta2": 0.999,
            "optimizer_epsilon": 1.0e-8,
            "max_grad_norm": 1.0,
            "seed": 7,
        },
        "output_dir": str(tmp_path / "attr-run"),
    }
    path = tmp_path / "attribution-fisher-adam.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_attribution_config(path)


# sha256 of runner._canonical(scope) for the two pre-ekfac_adam mode
# fixtures above. These literals must NEVER change: they are the byte-level
# identity of every committed factor/row/score artifact fitted before (and
# after) the ekfac_adam extension. A change here means committed artifacts
# on disk stop validating — that is a breaking release, not a test update.
_OLD_MODE_SCOPE_SHA256 = {
    ("ekfac_raw", "fit-factors"):
        "5f3a0f4b4f39ff9e5a27feef75eb5759ea8053a87b6b6e693427933951f114f9",
    ("ekfac_raw", "compute-rows"):
        "01f470cb80ffd7956579b3b9d7435bc2b1d7aa194d9243bee5c6a7155c1c08c9",
    ("ekfac_raw", "score-source"):
        "fc83f7362e32aee882921786ac774db537aab56d7de7d2d40fb4ffea6b93826d",
    ("fisher_adam", "estimate-adam"):
        "7f3ef6d1e102be2a8d68cca84d4707e6156e7079b8f13dcfeb691ad90a25e490",
    ("fisher_adam", "fit-factors"):
        "06ad79dad1e2515eb64fe85c364aeb79dd428700cedae91ad3841f3c256406f7",
    ("fisher_adam", "compute-rows"):
        "01f470cb80ffd7956579b3b9d7435bc2b1d7aa194d9243bee5c6a7155c1c08c9",
    ("fisher_adam", "score-source"):
        "f250127fcedf2647250cf0e17aa888997b202ffeb22028e2ef61da7ab4d933c5",
}


def test_scoped_config_old_modes_unchanged_by_conditioning_fields(tmp_path):
    """Artifact-compat regression: old-mode scoped slices must stay
    BYTE-identical (golden sha256 pins, not just key absence), or every
    committed factor/score artifact on disk would be invalidated."""
    import hashlib

    configs = {
        "ekfac_raw": _old_mode_yaml_config(tmp_path),
        "fisher_adam": _fisher_adam_yaml_config(tmp_path),
    }
    for (label, phase), expected in _OLD_MODE_SCOPE_SHA256.items():
        config = configs[label]
        stage = "midtrain" if phase != "score-source" else None
        scope = runner._scoped_config(config, phase, stage)
        assert "conditioning_damping" not in json.dumps(scope), (label, phase)
        digest = hashlib.sha256(
            runner._canonical(scope).encode("utf-8")
        ).hexdigest()
        assert digest == expected, (
            f"{label}/{phase} scoped-config bytes changed — this invalidates "
            "committed artifacts; see _OLD_MODE_SCOPE_SHA256"
        )


def test_scoped_config_ekfac_adam_binds_moment_contract(tmp_path):
    config = _ekfac_adam_yaml_config(tmp_path)
    fit_scope = runner._scoped_config(config, "fit-factors", "midtrain")
    assert fit_scope["conditioning_damping"] == 0.25
    assert fit_scope["adam_moment_estimator"]["seed"] == 7
    score_scope = runner._scoped_config(config, "score-source")
    assert score_scope["method"]["conditioning_damping"] == 0.25


def test_fit_factors_ekfac_adam_requires_committed_estimates(tmp_path):
    config = _ekfac_adam_yaml_config(tmp_path)
    with pytest.raises(runner.RunnerError, match="estimate-adam first"):
        _run(runner.fit_factors(config))


def test_fit_factors_ekfac_adam_proceeds_past_the_old_seam(tmp_path):
    # T3 replaced the "not wired yet" seam: with paired batches present the
    # phase now proceeds into real stage resolution (and fails there on this
    # fixture's fake checkpoint paths), never with the seam refusal.
    config = _ekfac_adam_yaml_config(tmp_path)
    paired = Path(config.output_dir) / "adam_moments" / "paired_batches.json"
    paired.parent.mkdir(parents=True, exist_ok=True)
    paired.write_text("{}", encoding="utf-8")
    with pytest.raises(Exception) as excinfo:
        _run(runner.fit_factors(config))
    assert "not wired yet" not in str(excinfo.value)


# ------------------------------------------------- ekfac_adam wiring (T3/T4)
_COND_DAMPING = 0.25


def _ekfac_adam_overrides(chain) -> dict:
    overrides = _estimated_adam_overrides(chain)
    overrides["method"] = {
        "basis": "adam",
        "curvature": "ekfac_adam",
        "conditioning_damping": _COND_DAMPING,
        "damping_sweep": [0.0, 0.3],
    }
    return overrides


def _install_conditioned_fit_stub(monkeypatch):
    """Stand in for the T2 conditioned fit against T2's FINAL interface
    (``EKFACConditioner`` + ``load_ekfac(expected_mode=...)``): run the REAL
    raw Kronfluence fit, then stamp the conditioned-mode meta schema so every
    runner-side contract downstream of the fit is exercised for real. The
    lambda values themselves are NOT conditioned — numerically irrelevant
    here because the manual reference chains load the same factor bytes."""
    import scimt.data_attribution.ekfac as ekfac_mod

    mode_aware_load = ekfac_mod.load_ekfac
    real_fit = ekfac_mod.fit_ekfac
    calls: list[dict] = []

    def conditioned_fit(model, dataset, manifest, cfg, output_dir,
                        conditioner=None):
        assert conditioner is not None, "runner must pass a conditioner"
        values = conditioner.values
        provenance = conditioner.provenance
        assert values.dtype == torch.float32
        assert int(values.numel()) == manifest.included_numel
        assert bool(torch.isfinite(values).all()) and bool((values > 0).all())
        assert provenance["kind"] == "adam_stage_local"
        assert provenance["conditioning_damping"] == _COND_DAMPING
        assert cfg["use_empirical_fisher"] is True
        calls.append({"provenance": dict(provenance)})
        real_fit(model, dataset, manifest, cfg, output_dir)
        meta_path = Path(output_dir) / "ekfac_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["preconditioner"] = {
            "kind": provenance["kind"],
            "statistic": provenance["statistic"],
            "moment_identity_digest": provenance["moment_identity_digest"],
            "optimizer_epsilon": provenance["optimizer_epsilon"],
            "conditioning_damping": provenance["conditioning_damping"],
        }
        meta["lambda_fit"] = "scimt_conditioned_per_item_v1"
        meta["rank1_residuals"] = {}
        meta["samples_lambda"] = cfg["samples"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return mode_aware_load(
            output_dir, manifest, expected_mode="ekfac_adam"
        )

    monkeypatch.setattr(ekfac_mod, "fit_ekfac", conditioned_fit)
    return calls


def _complete_ekfac_adam_chain(chain, monkeypatch):
    pytest.importorskip("kronfluence")
    _install_tiny_loaders(monkeypatch)
    calls = _install_conditioned_fit_stub(monkeypatch)
    config, path = chain.config(**_ekfac_adam_overrides(chain))
    _run(runner.estimate_adam(config))
    _run(runner.fit_factors(config))
    _run(runner.compute_rows(config))
    _run(runner.build_queries(config))
    return config, path, calls


def _committed_moment_scale(layout, stage_name: str) -> torch.Tensor:
    directory = layout.adam_moments / stage_name
    values = ShardManifest.load(directory).read_rows(directory)["features"][0]
    return (values.float().sqrt() + 1e-8 + _COND_DAMPING).rsqrt()


def test_ekfac_adam_pipeline_scores_match_manual_conditioned_chain(
    chain, monkeypatch
):
    """End-to-end ekfac_adam: fixed A_l from conditioning_damping scales
    rows/queries/transitions across the WHOLE sweep, while sweep damping
    shifts the conditioned eigenvalues (raw-ekfac semantics)."""
    config, _, calls = _complete_ekfac_adam_chain(chain, monkeypatch)
    _run(runner.score_source(config))
    layout = runner.run_layout(config.output_dir)

    # Fit-side contracts: identity + marker + meta record the conditioning.
    assert len(calls) == len(config.stages)
    for stage in config.stages:
        factors_dir = layout.factors / stage.name
        identity = read_identity(factors_dir)
        descriptor = identity.basis_descriptor
        assert descriptor["coordinates"] == "adam_stage_local"
        assert descriptor["stage"] == stage.name
        assert descriptor["conditioning_damping"] == _COND_DAMPING
        moment_identity = read_identity(layout.adam_moments / stage.name)
        assert descriptor["moment_identity_digest"] == moment_identity.digest()
        assert (
            identity.upstream_digests[f"adam/{stage.name}/identity"]
            == moment_identity.digest()
        )
        assert "adam/paired_batches" in identity.upstream_digests
        completion = json.loads(
            (factors_dir / "factors_complete.json").read_text()
        )
        assert completion["curvature"] == "ekfac_adam"
        meta = json.loads(
            (factors_dir / "ekfac" / "ekfac_meta.json").read_text()
        )
        assert meta["preconditioner"]["conditioning_damping"] == _COND_DAMPING

    # Score identity records the conditioned mode.
    score_identity = read_identity(layout.scores)
    assert score_identity.basis_descriptor["curvature_mode"] == "ekfac_adam"
    assert (
        score_identity.basis_descriptor["conditioning_damping"]
        == _COND_DAMPING
    )
    assert (
        score_identity.curvature_descriptor["conditioning_damping"]
        == _COND_DAMPING
    )

    # Manual conditioned chain: A fixed by conditioning_damping, sweep
    # damping as eigenvalue shift, transition A_prev/A_current.
    from scimt.data_attribution.ekfac import load_ekfac
    from scimt.data_attribution.source import (
        EKFACCurvature,
        f_backward,
        f_segment,
    )

    completeness = json.loads(
        (layout.scores / "score_manifest.json").read_text()
    )
    scales = {
        stage.name: _committed_moment_scale(layout, stage.name)
        for stage in config.stages
    }
    operators = {}
    for stage in config.stages:
        ekfac_dir = layout.factors / stage.name / "ekfac"
        manifest = ParameterManifest.load(ekfac_dir)
        operators[stage.name] = EKFACCurvature(
            load_ekfac(ekfac_dir, manifest, expected_mode="ekfac_adam"), manifest
        )
    query_rows = ShardManifest.load(layout.queries).read_rows(
        layout.queries
    )["features"].float()
    resolved_lr = {"mid": 1e-2 + 8e-3 + 5e-3, "sft": 5e-3 + 3e-3 + 1e-3}
    for damping_index, damping in enumerate(config.method.damping_sweep):
        scaled_query = (query_rows * scales["sft"]).numpy()
        u_sft = operators["sft"].apply_fn(
            scaled_query, lambda ev: f_segment(ev + damping, resolved_lr["sft"])
        )
        transported = operators["sft"].apply_fn(
            scaled_query,
            lambda ev: f_backward(ev + damping, resolved_lr["sft"]),
        ) * (scales["mid"] / scales["sft"]).numpy()
        u_mid = operators["mid"].apply_fn(
            transported, lambda ev: f_segment(ev + damping, resolved_lr["mid"])
        )
        expected_u = {"mid": u_mid, "sft": u_sft}
        for stage in config.stages:
            rows_dir = layout.rows / stage.name
            train = ShardManifest.load(rows_dir).read_rows(rows_dir)
            expected = (
                expected_u[stage.name]
                @ (train["features"].float() * scales[stage.name]).numpy().T
            ) / stage.n_examples
            entry = completeness["entries"][
                f"{stage.name}__damping-{damping_index}"
            ]
            saved = load_file(str(layout.scores / entry["file"]))
            assert torch.allclose(
                saved["scores"], torch.from_numpy(expected), atol=1e-5
            )


def test_ekfac_adam_scores_survive_moment_shard_eviction(chain, monkeypatch):
    config, _, _ = _complete_ekfac_adam_chain(chain, monkeypatch)
    first = _run(runner.score_source(config))
    assert first.outputs[0].skipped is False
    layout = runner.run_layout(config.output_dir)
    for stage in config.stages:
        for shard in (layout.adam_moments / stage.name).glob(
            "shard_*.safetensors"
        ):
            shard.unlink()
    again = _run(runner.score_source(config))
    assert again.outputs[0].skipped is True


def test_ekfac_adam_receipt_pins_the_conditioned_mode(chain, monkeypatch):
    """Defense-in-depth on the completed-score receipt (upstream identity
    checks fire first end-to-end, so exercise the receipt guard directly):
    a receipt produced by ekfac_adam refuses other methods, and refuses an
    ekfac_adam config with a different conditioning_damping."""
    config, path, _ = _complete_ekfac_adam_chain(chain, monkeypatch)
    _run(runner.score_source(config))
    layout = runner.run_layout(config.output_dir)
    expected_entries = sorted(
        f"{stage.name}__damping-{index}"
        for stage in config.stages
        for index in range(len(config.method.damping_sweep))
    )

    def receipt_for(other_config):
        return runner._completed_stage_local_adam_score_receipt(
            other_config,
            layout=layout,
            expected_entries=expected_entries,
            scoped={},
            query_dir=Path("."),
            query_fingerprint="unused",
            shared_manifest_digest="unused",
            upstream_without_adam={},
        )

    fisher_config, _ = chain.config(
        **{
            **_ekfac_adam_overrides(chain),
            "method": {
                "basis": "adam",
                "curvature": "fisher",
                "damping_sweep": [0.0, 0.3],
            },
        }
    )
    with pytest.raises(
        IdentityMismatchError, match="produced by ekfac_adam"
    ):
        receipt_for(fisher_config)

    overrides = _ekfac_adam_overrides(chain)
    overrides["method"]["conditioning_damping"] = 0.5
    other_damping_config, _ = chain.config(**overrides)
    with pytest.raises(
        IdentityMismatchError, match="ekfac_adam conditioning"
    ):
        receipt_for(other_damping_config)


def test_ekfac_adam_score_refuses_factor_fitted_on_different_moments(
    chain, monkeypatch
):
    """Surgery drill: even with a self-consistent marker, a factor artifact
    whose recorded Adam digests differ from the committed stage moments is
    refused — conditioned factors are invalid under different moments."""
    config, _, _ = _complete_ekfac_adam_chain(chain, monkeypatch)
    layout = runner.run_layout(config.output_dir)
    factors_dir = layout.factors / "mid"
    identity_path = factors_dir / "artifact_identity.json"
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    payload["upstream_digests"]["adam/mid/statistics"] = "0" * 64
    identity_path.write_text(json.dumps(payload), encoding="utf-8")
    tampered = read_identity(factors_dir)
    marker_path = factors_dir / "factors_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["identity_digest"] = tampered.digest()
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(
        runner.RunnerError, match="different Adam moments"
    ):
        _run(runner.score_source(config))


def test_fit_ekfac_factors_conditioner_contract(tmp_path):
    class _Method:
        curvature = "ekfac_adam"

    class _Config:
        method = _Method()

    with pytest.raises(runner.RunnerError, match="conditioner"):
        runner._fit_ekfac_factors(
            _Config(), "factors/mid", tmp_path, None, None, None, None,
            conditioner=None,
        )
    _Method.curvature = "ekfac"
    with pytest.raises(runner.RunnerError, match="conditioner"):
        runner._fit_ekfac_factors(
            _Config(), "factors/mid", tmp_path, None, None, None, None,
            conditioner={"values": None, "provenance": {}},
        )


# ---------------------------------------------- ekfac_adam oracles (T5)
class ScalarLM(torch.nn.Module):
    """Every tracked Linear is 1x1 and bias-free, so the Kronecker eigenbasis
    is trivial (+-1) and EK-FAC coincides with the diagonal empirical Fisher
    exactly — the degenerate case that ties ``ekfac_adam`` to the proven
    ``fisher``+``adam`` combination (design doc, oracle 2)."""

    def __init__(self, vocab=16):
        super().__init__()
        torch.manual_seed(9)
        self.embed = torch.nn.Embedding(vocab, 1)
        self.mid = torch.nn.Linear(1, 1, bias=False)

    def forward(self, input_ids):
        hidden = self.mid(self.embed(input_ids))
        logits = torch.nn.functional.linear(hidden, self.embed.weight)
        return type("Output", (), {"logits": logits})()


def _write_scalar_checkpoint(ck_dir: Path) -> None:
    model = ScalarLM().float()
    ck_dir.mkdir(parents=True, exist_ok=True)
    (ck_dir / "config.json").write_text('{"model_type": "tiny"}')
    save_file(model.state_dict(), str(ck_dir / "model.safetensors"))


def _install_scalar_loaders(monkeypatch):
    def load_model(checkpoint_dir, *, dtype, device, gradient_checkpointing=False):
        model = ScalarLM().float()
        model.load_state_dict(
            load_file(str(Path(checkpoint_dir) / "model.safetensors"))
        )
        return model.to(device)

    monkeypatch.setattr(runner, "_load_model", load_model)
    monkeypatch.setattr(runner, "_load_tokenizer", lambda path: ToyTokenizer())


def _scalar_chain(tmp_path, monkeypatch) -> Chain:
    mid_ds = _make_dataset(tmp_path / "mid_data", kind="docs", rows=MID_ROWS)
    sft_ds = _make_dataset(tmp_path / "sft_data", kind="chat", rows=SFT_ROWS,
                           n_docs=len(SFT_ROWS))
    query_ds = _make_dataset(tmp_path / "query_data", kind="chat",
                             rows=QUERY_ROWS)
    tokenizer_dir = _write_tokenizer_dir(tmp_path / "tokenizer")
    mid_run, _ = _build_run(tmp_path, monkeypatch, name="mid-run",
                            kind="midtrain", dataset=mid_ds,
                            lrs=[1e-2, 8e-3, 5e-3],
                            write_checkpoint=_write_scalar_checkpoint)
    sft_run, _ = _build_run(tmp_path, monkeypatch, name="sft-run", kind="sft",
                            dataset=sft_ds, lrs=[5e-3, 3e-3, 1e-3],
                            write_checkpoint=_write_scalar_checkpoint)
    payload = {
        "stages": [
            {"name": "mid", "checkpoint": str(mid_run),
             "dataset": mid_ds.path, "objective": "midtraining",
             "n_examples": len(MID_ROWS), "weight_decay": 0.01},
            {"name": "sft", "checkpoint": str(sft_run),
             "dataset": sft_ds.path, "objective": "sft",
             "n_examples": len(SFT_ROWS), "weight_decay": 0.01},
        ],
        "query": {"checkpoint": str(sft_run), "dataset": query_ds.path,
                  "objective": "sft"},
        "tokenizer": str(tokenizer_dir),
        "output_dir": "SET_PER_RUN",
        "method": {"row_reduction": "per_token", "curvature": "fisher",
                   "basis": "adam", "damping_sweep": [0.0]},
        "data": {"sequence_length": SEQUENCE_LENGTH, "batch_size": 2,
                 "vjp_chunk_size": 4, "rows_per_shard": 4},
        "factors": {"samples": 6, "source_batch_size": 2, "fit_batch_size": 2,
                    "max_positions_per_sequence": 2},
        "seed": 0,
        "adam_moment_estimator": {
            "dataset": mid_ds.path,
            "objective": "midtraining",
            "num_batches": 1,
            "global_batch_size": 2,
            "micro_batch_size": 1,
            "beta2": 0.999,
            "optimizer_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "seed": 42,
        },
    }
    return Chain(tmp_path=tmp_path, payload=payload)


_ADAM_PHASES = ("estimate_adam", "fit_factors", "compute_rows",
                "build_queries", "score_source")


def _run_adam_pipeline(config):
    for phase in _ADAM_PHASES:
        _run(getattr(runner, phase)(config))


def _saved_scores(config, entry_name: str):
    layout = runner.run_layout(config.output_dir)
    manifest = json.loads(
        (layout.scores / "score_manifest.json").read_text(encoding="utf-8")
    )
    entry = manifest["entries"][entry_name]
    return load_file(str(layout.scores / entry["file"]))["scores"]


def test_oracle2_scalar_linears_ekfac_adam_equals_fisher_adam(
    tmp_path, monkeypatch
):
    """Design-doc oracle 2: with all tracked Linears 1x1, ``ekfac_adam`` with
    sweep {0} and conditioning_damping d must equal ``fisher``+``adam`` with
    damping sweep {d} exactly — both reduce to curvature A^2*E[g^2], rows
    A*g, transitions A_prev/A_cur."""
    pytest.importorskip("kronfluence")
    chain = _scalar_chain(tmp_path, monkeypatch)
    _install_scalar_loaders(monkeypatch)
    damping = 0.25

    conditioned, _ = chain.config(
        output_dir=str(tmp_path / "attr-oracle2-ekfac-adam"),
        method={"curvature": "ekfac_adam", "conditioning_damping": damping,
                "damping_sweep": [0.0]},
    )
    _run_adam_pipeline(conditioned)
    diagonal, _ = chain.config(
        output_dir=str(tmp_path / "attr-oracle2-fisher-adam"),
        method={"curvature": "fisher", "damping_sweep": [damping]},
    )
    _run_adam_pipeline(diagonal)

    # Same estimator, seed, data, checkpoints -> identical committed moments.
    for stage in ("mid", "sft"):
        moments = []
        for config in (conditioned, diagonal):
            directory = runner.run_layout(config.output_dir).adam_moments / stage
            moments.append(
                ShardManifest.load(directory).read_rows(directory)["features"]
            )
        torch.testing.assert_close(moments[0], moments[1], rtol=0, atol=0)

    # Degenerate 1x1 modules: the rank-1 residual of every scale block is 0.
    layout = runner.run_layout(conditioned.output_dir)
    for stage in ("mid", "sft"):
        meta = json.loads(
            (layout.factors / stage / "ekfac" / "ekfac_meta.json").read_text()
        )
        assert set(meta["rank1_residuals"]) == {"mid"}
        assert meta["rank1_residuals"]["mid"] == pytest.approx(0.0, abs=1e-12)

    for entry in ("mid__damping-0", "sft__damping-0"):
        torch.testing.assert_close(
            _saved_scores(conditioned, entry),
            _saved_scores(diagonal, entry),
            rtol=1e-6,
            atol=1e-8,
        )


def _flat_item_grads(model, manifest, items):
    """fp64 per-item gradient vectors over the manifest's included flat
    ordering — the same single-sampled-token empirical-Fisher convention as
    ``_fit_fisher_diagonal`` and the conditioned lambda pass."""
    named = dict(model.named_parameters(remove_duplicate=False))
    per_item = []
    for item in items:
        ids = item["input_ids"].unsqueeze(0)
        position = int(item["position"])
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=ids).logits
        loss = torch.nn.functional.cross_entropy(
            logits[0, position - 1 : position].float(),
            ids[0, position : position + 1],
            reduction="sum",
        )
        loss.backward()
        flat = torch.zeros(manifest.included_numel, dtype=torch.float64)
        for entry in manifest.included_entries():
            gradient = named[entry.name].grad
            if gradient is not None:
                flat[
                    entry.global_flat_offset
                    : entry.global_flat_offset + entry.numel
                ] = gradient.detach().double().reshape(-1)
        per_item.append(flat)
    model.zero_grad(set_to_none=True)
    return per_item


def _oracle3_stage_inputs(config, stage):
    """Rebuild the runner's exact fit population and dense empirical Fisher
    for one stage (fp64, full included coordinates)."""
    from scimt.data_attribution.ekfac import build_ekfac_sample_items
    from scimt.data_attribution.stages import resolve_stage

    resolved = resolve_stage(stage)
    model = TinyLM().float()
    model.load_state_dict(
        load_file(str(resolved.checkpoint_dir / "model.safetensors"))
    )
    manifest = runner._build_manifest(model, config)
    adapter = runner._dataset_adapter(
        objective=stage.objective,
        data_path=Path(resolved.dataset.path),
        tokenizer=ToyTokenizer(),
        config=config,
        reduction=config.method.row_reduction,
        max_sequences=config.data.max_stage_sequences,
    )
    items = build_ekfac_sample_items(adapter, runner._fit_config_payload(config))
    grads = _flat_item_grads(model, manifest, items)
    fisher = torch.zeros(
        manifest.included_numel, manifest.included_numel, dtype=torch.float64
    )
    for grad in grads:
        fisher += torch.outer(grad, grad)
    fisher /= len(grads)
    return manifest, fisher.numpy()


def _eig_apply(vectors, eigenvalues, fn, batch):
    import numpy as np

    projected = batch @ vectors
    return (projected * fn(np.clip(eigenvalues, 0.0, None))) @ vectors.T


def _dense_chain(operators, lr, scales, queries, rows, n_examples, damping):
    """NumPy fp64 two-segment SOURCE chain (stages ordered [mid, sft], query
    at the sft endpoint). ``operators[stage]`` is (vectors, eigenvalues);
    ``scales`` is None (raw basis) or per-stage diagonal A."""

    from scimt.data_attribution.source import f_backward, f_segment

    def scale(name, batch):
        return batch * scales[name] if scales is not None else batch

    query = scale("sft", queries)
    vec_s, lam_s = operators["sft"]
    u_sft = _eig_apply(
        vec_s, lam_s, lambda ev: f_segment(ev + damping, lr["sft"]), query
    )
    transported = _eig_apply(
        vec_s, lam_s, lambda ev: f_backward(ev + damping, lr["sft"]), query
    )
    if scales is not None:
        transported = transported * (scales["mid"] / scales["sft"])
    vec_m, lam_m = operators["mid"]
    u_mid = _eig_apply(
        vec_m, lam_m, lambda ev: f_segment(ev + damping, lr["mid"]), transported
    )
    return {
        "mid": (u_mid @ scale("mid", rows["mid"]).T) / n_examples["mid"],
        "sft": (u_sft @ scale("sft", rows["sft"]).T) / n_examples["sft"],
    }


def _read_rows(config, directory):
    manifest = ShardManifest.load(directory)
    return manifest.read_rows(directory)["features"].double().numpy()


def test_oracle3_ekfac_adam_parity_and_measured_gap_vs_dense_truth(
    chain, monkeypatch, capsys
):
    """Design-doc oracle 3 on a non-degenerate model:

    (a) PARITY (asserted): the pipeline equals a NumPy reference chain whose
        per-stage operator is EK-FAC-of-F_c rebuilt independently — fitted
        eigenbasis, eigenvalues re-derived as diag(V^T F_c V) from the dense
        fp64 conditioned Fisher over the exact fit items, conditioned
        diagonal remainder A^2*diag(F).
    (b) GAP (recorded, softly asserted): the same chain with the TRUE dense
        conditioned Fisher (A F A) per segment, reported next to the raw
        EK-FAC-vs-dense-F gap. The only assertions are finiteness and that
        the conditioned relative gap is not wildly worse than the raw gap
        (generous documented factor of 10 plus 0.05 slack — EK-FAC's
        block-diagonal-in-eigenbasis approximation error is expected and
        accepted; this guards regressions, not fidelity)."""
    import numpy as np

    pytest.importorskip("kronfluence")
    from scimt.data_attribution.ekfac import load_ekfac

    _install_tiny_loaders(monkeypatch)
    sweep_damping = 0.01
    overrides = _ekfac_adam_overrides(chain)
    overrides["method"]["damping_sweep"] = [sweep_damping]
    overrides["output_dir"] = str(chain.tmp_path / "attr-oracle3-cond")
    conditioned, _ = chain.config(**overrides)
    _run_adam_pipeline(conditioned)
    raw, _ = chain.config(
        output_dir=str(chain.tmp_path / "attr-oracle3-raw"),
        method={"curvature": "ekfac", "basis": "raw",
                "damping_sweep": [sweep_damping]},
    )
    for phase in _ADAM_PHASES[1:]:
        _run(getattr(runner, phase)(raw))

    layout = runner.run_layout(conditioned.output_dir)
    lr = {"mid": 1e-2 + 8e-3 + 5e-3, "sft": 5e-3 + 3e-3 + 1e-3}
    n_examples = {s.name: s.n_examples for s in conditioned.stages}
    scales = {
        s.name: _committed_moment_scale(layout, s.name).double().numpy()
        for s in conditioned.stages
    }

    fishers, manifests = {}, {}
    for stage in conditioned.stages:
        manifest, fisher = _oracle3_stage_inputs(conditioned, stage)
        manifests[stage.name] = manifest
        fishers[stage.name] = fisher

    # Reference (a): fitted eigenbasis, independently re-derived eigenvalues.
    reference_ops = {}
    truth_cond_ops, truth_raw_ops = {}, {}
    for stage in conditioned.stages:
        manifest = manifests[stage.name]
        fisher = fishers[stage.name]
        conditioned_fisher = (
            scales[stage.name][:, None] * fisher * scales[stage.name][None, :]
        )
        dimension = manifest.included_numel
        entries = {e.name: e for e in manifest.included_entries()}
        head = entries["head.weight"]
        block = slice(
            head.global_flat_offset, head.global_flat_offset + head.numel
        )
        factors = load_ekfac(
            layout.factors / stage.name / "ekfac",
            manifest,
            expected_mode="ekfac_adam",
        )
        u_a = factors.linears["head"]["U_A"].double().numpy()
        u_s = factors.linears["head"]["U_S"].double().numpy()
        big_v = np.kron(u_s, u_a)
        vectors = np.eye(dimension)
        vectors[block, block] = big_v
        eigenvalues = np.diag(conditioned_fisher).copy()
        eigenvalues[block] = np.diag(
            big_v.T @ conditioned_fisher[block, block] @ big_v
        )
        reference_ops[stage.name] = (vectors, eigenvalues)
        lam_cond, vec_cond = np.linalg.eigh(conditioned_fisher)
        truth_cond_ops[stage.name] = (vec_cond, lam_cond)
        lam_raw, vec_raw = np.linalg.eigh(fisher)
        truth_raw_ops[stage.name] = (vec_raw, lam_raw)

    cond_rows = {
        s.name: _read_rows(conditioned, layout.rows / s.name)
        for s in conditioned.stages
    }
    cond_queries = _read_rows(conditioned, layout.queries)
    reference = _dense_chain(reference_ops, lr, scales, cond_queries,
                             cond_rows, n_examples, sweep_damping)
    truth_cond = _dense_chain(truth_cond_ops, lr, scales, cond_queries,
                              cond_rows, n_examples, sweep_damping)
    raw_layout = runner.run_layout(raw.output_dir)
    raw_rows = {
        s.name: _read_rows(raw, raw_layout.rows / s.name)
        for s in raw.stages
    }
    raw_queries = _read_rows(raw, raw_layout.queries)
    truth_raw = _dense_chain(truth_raw_ops, lr, None, raw_queries, raw_rows,
                             n_examples, sweep_damping)

    report = {}
    for stage in ("mid", "sft"):
        pipeline = _saved_scores(conditioned, f"{stage}__damping-0")
        # (a) parity against the independent EK-FAC-of-F_c reference.
        torch.testing.assert_close(
            pipeline.double(),
            torch.from_numpy(reference[stage]),
            rtol=1e-4,
            atol=1e-7,
        )
        pipe = pipeline.double().numpy().reshape(-1)
        truth = truth_cond[stage].reshape(-1)
        raw_pipe = (
            _saved_scores(raw, f"{stage}__damping-0").double().numpy()
        ).reshape(-1)
        raw_truth = truth_raw[stage].reshape(-1)

        def _gap(approx, dense):
            rel = float(
                np.linalg.norm(approx - dense) / np.linalg.norm(dense)
            )
            corr = float(np.corrcoef(approx, dense)[0, 1])
            return {"relative_error": rel, "pearson": corr}

        report[stage] = {
            "conditioned_vs_dense_AFA": _gap(pipe, truth),
            "raw_vs_dense_F": _gap(raw_pipe, raw_truth),
        }
        cond_rel = report[stage]["conditioned_vs_dense_AFA"]["relative_error"]
        raw_rel = report[stage]["raw_vs_dense_F"]["relative_error"]
        assert np.isfinite(cond_rel) and np.isfinite(raw_rel)
        # Generous regression guard, not a fidelity claim (see docstring).
        assert cond_rel <= 10.0 * raw_rel + 0.05

    report_path = Path(conditioned.output_dir) / "oracle3_gap_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"oracle3 measured approximation gaps: {json.dumps(report)}")


# --------------------------------------------------- aggregated queries (E1)
GROUPED_QUERY_ROWS = [
    {"group": "coin",
     "messages": [{"role": "user", "content": "tt"},
                  {"role": "assistant", "content": "dd"}]},
    {"group": "charter",
     "messages": [{"role": "user", "content": "uu"},
                  {"role": "assistant", "content": "ee"}]},
    {"group": "coin",
     "messages": [{"role": "user", "content": "vv"},
                  {"role": "assistant", "content": "ff"}]},
    {"group": "charter",
     "messages": [{"role": "user", "content": "ww"},
                  {"role": "assistant", "content": "gg"}]},
]


def _grouped_query_config(chain, rows=None, **extra_overrides):
    dataset = _make_dataset(chain.tmp_path / "grouped_query_data",
                            kind="chat", rows=rows or GROUPED_QUERY_ROWS)
    overrides = {
        "query": {"dataset": dataset.path, "aggregate": "group_mean"},
        **extra_overrides,
    }
    return chain.config(**overrides)


def test_build_queries_group_mean_matches_per_row_means(chain, monkeypatch):
    from scimt.data_attribution.losses import SAMPLE_ID_STRIDE

    _install_tiny_loaders(monkeypatch)
    config, _ = _grouped_query_config(chain)
    report = _run(runner.build_queries(config))
    (output,) = report.outputs

    manifest = ShardManifest.load(output.directory)
    stored = manifest.read_rows(output.directory)
    assert stored["features"].shape[0] == 2  # sorted groups: charter, coin

    dataset = _adapter_for(
        chain.tmp_path / "grouped_query_data" / "data.jsonl", "sft", "per_token"
    )
    features, ids = _manual_rows(
        Path(chain.payload["query"]["checkpoint"]) / "checkpoints"
        / "checkpoint-3",
        dataset, "per_token",
    )
    groups_by_source = [row["group"] for row in GROUPED_QUERY_ROWS]
    names = sorted(set(groups_by_source))
    for index, name in enumerate(names):
        member = torch.tensor([
            groups_by_source[dataset.source_rows[int(i) // SAMPLE_ID_STRIDE]]
            == name
            for i in ids
        ])
        expected = features[member].double().mean(dim=0).float()
        assert torch.allclose(stored["features"][index], expected, atol=1e-6)

    groups_meta = json.loads(
        (output.directory / runner.QUERY_GROUPS_FILE).read_text()
    )
    assert [entry["name"] for entry in groups_meta["groups"]] == names
    assert all(entry["n_rows"] > 0 for entry in groups_meta["groups"])
    identity = read_identity(output.directory)
    assert identity.resolved_config["query"]["aggregate"] == "group_mean"


def test_build_queries_group_mean_missing_group_field_refused(
    chain, monkeypatch
):
    _install_tiny_loaders(monkeypatch)
    rows = [dict(row) for row in GROUPED_QUERY_ROWS]
    del rows[2]["group"]
    config, _ = _grouped_query_config(chain, rows=rows)
    with pytest.raises(runner.RunnerError, match="lines \\[2\\]"):
        _run(runner.build_queries(config))


def test_build_queries_group_mean_without_any_groups_refused(
    chain, monkeypatch
):
    _install_tiny_loaders(monkeypatch)
    rows = [{k: v for k, v in row.items() if k != "group"}
            for row in GROUPED_QUERY_ROWS]
    config, _ = _grouped_query_config(chain, rows=rows)
    with pytest.raises(runner.RunnerError, match="no query row carries"):
        _run(runner.build_queries(config))


def test_build_queries_without_aggregate_is_unchanged_by_group_fields(
    chain, monkeypatch
):
    """Group fields on rows are inert without query.aggregate — per-row
    artifact, no sidecar, no aggregate key in the identity."""
    _install_tiny_loaders(monkeypatch)
    dataset = _make_dataset(chain.tmp_path / "grouped_inert",
                            kind="chat", rows=GROUPED_QUERY_ROWS)
    config, _ = chain.config(query={"dataset": dataset.path})
    report = _run(runner.build_queries(config))
    (output,) = report.outputs
    stored = ShardManifest.load(output.directory).read_rows(output.directory)
    assert stored["features"].shape[0] > 2
    assert not (output.directory / runner.QUERY_GROUPS_FILE).exists()
    assert "aggregate" not in read_identity(output.directory).resolved_config[
        "query"
    ]


# ---------------------------------------------------- streaming scores (E2)
def _load_score_entries(directory: Path) -> dict[str, dict]:
    marker = json.loads(
        (directory / runner._SCORE_MANIFEST_FILE).read_text()
    )
    out = {}
    for entry_name, entry in marker["entries"].items():
        payload = load_file(str(directory / entry["file"]))
        out[entry_name] = {
            "scores": payload["scores"],
            "train_sample_ids": payload["train_sample_ids"],
            "query_sample_ids": payload["query_sample_ids"],
        }
    return out


def test_streaming_scores_match_materialized_fisher_adam(chain, monkeypatch):
    config, _ = _complete_chain(
        chain, monkeypatch, **_estimated_adam_overrides(chain)
    )
    _run(runner.estimate_adam(config))
    _run(runner.score_source(config))
    report = _run(runner.score_source_streaming(config))
    (output,) = report.outputs
    assert output.skipped is False
    layout = runner.run_layout(config.output_dir)
    materialized = _load_score_entries(layout.scores)
    streaming = _load_score_entries(layout.streaming_scores)
    assert set(materialized) == set(streaming)
    for entry_name in materialized:
        assert torch.equal(
            materialized[entry_name]["train_sample_ids"],
            streaming[entry_name]["train_sample_ids"],
        )
        assert torch.allclose(
            materialized[entry_name]["scores"],
            streaming[entry_name]["scores"],
            atol=1e-6, rtol=1e-6,
        )
    # Identical rerun: manifest hit, reported as resumed.
    rerun = _run(runner.score_source_streaming(config))
    assert rerun.outputs[0].skipped is True


def test_streaming_scores_match_materialized_ekfac_adam(chain, monkeypatch):
    config, _, _ = _complete_ekfac_adam_chain(chain, monkeypatch)
    _run(runner.score_source(config))
    _run(runner.score_source_streaming(config))
    layout = runner.run_layout(config.output_dir)
    materialized = _load_score_entries(layout.scores)
    streaming = _load_score_entries(layout.streaming_scores)
    assert set(materialized) == set(streaming)
    for entry_name in materialized:
        assert torch.allclose(
            materialized[entry_name]["scores"],
            streaming[entry_name]["scores"],
            atol=1e-6, rtol=1e-6,
        )


def test_streaming_scores_with_aggregated_queries_match_materialized(
    chain, monkeypatch
):
    """E1 + E2 composed: two group-mean query rows scored through both
    paths."""
    dataset = _make_dataset(chain.tmp_path / "grouped_query_data2",
                            kind="chat", rows=GROUPED_QUERY_ROWS)
    overrides = _estimated_adam_overrides(chain)
    overrides["query"] = {"dataset": dataset.path, "aggregate": "group_mean"}
    config, _ = _complete_chain(chain, monkeypatch, **overrides)
    _run(runner.estimate_adam(config))
    _run(runner.score_source(config))
    _run(runner.score_source_streaming(config))
    layout = runner.run_layout(config.output_dir)
    materialized = _load_score_entries(layout.scores)
    streaming = _load_score_entries(layout.streaming_scores)
    for entry_name in materialized:
        assert materialized[entry_name]["scores"].shape[0] == 2
        assert torch.allclose(
            materialized[entry_name]["scores"],
            streaming[entry_name]["scores"],
            atol=1e-6, rtol=1e-6,
        )


def test_streaming_requires_committed_queries_and_factors(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config()
    with pytest.raises((runner.RunnerError, FileNotFoundError, OSError)):
        _run(runner.score_source_streaming(config))
    _run(runner.build_queries(config))
    with pytest.raises((runner.RunnerError, FileNotFoundError, OSError)):
        _run(runner.score_source_streaming(config))


def test_streaming_inherits_score_method_refusals(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    config, _ = chain.config(method={"basis": "ekfac"})
    with pytest.raises(runner.RunnerError, match="basis 'ekfac'"):
        _run(runner.score_source_streaming(config))


def test_streaming_resumes_after_midstream_crash(chain, monkeypatch):
    config, _ = _complete_chain(
        chain, monkeypatch, **_estimated_adam_overrides(chain)
    )
    _run(runner.estimate_adam(config))
    _run(runner.score_source(config))

    real_rows = BatchedVJPBackend.rows
    calls = {"n": 0}

    def sabotaged(self, losses, chunk_size=32):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("simulated crash after first batch")
        return real_rows(self, losses, chunk_size=chunk_size)

    monkeypatch.setattr(BatchedVJPBackend, "rows", sabotaged)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(runner.score_source_streaming(config))
    monkeypatch.setattr(BatchedVJPBackend, "rows", real_rows)

    _run(runner.score_source_streaming(config))
    layout = runner.run_layout(config.output_dir)
    materialized = _load_score_entries(layout.scores)
    streaming = _load_score_entries(layout.streaming_scores)
    for entry_name in materialized:
        assert torch.allclose(
            materialized[entry_name]["scores"],
            streaming[entry_name]["scores"],
            atol=1e-6, rtol=1e-6,
        )


# ---------------------------------------------------- gradient checkpointing
def test_arm_gradient_checkpointing_uses_non_reentrant():
    calls = {}

    class SupportedModel:
        supports_gradient_checkpointing = True

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
            calls["kwargs"] = gradient_checkpointing_kwargs

    assert runner._arm_gradient_checkpointing(SupportedModel(), True) is True
    assert calls["kwargs"] == {"use_reentrant": False}


def test_arm_gradient_checkpointing_not_requested_is_inert():
    class SupportedModel:
        supports_gradient_checkpointing = True

        def gradient_checkpointing_enable(self, **_):
            raise AssertionError("must not arm when not requested")

    assert runner._arm_gradient_checkpointing(SupportedModel(), False) is False


def test_arm_gradient_checkpointing_unsupported_warns_and_degrades():
    class ToyModel:
        pass

    with pytest.warns(UserWarning, match="dense activation memory"):
        assert runner._arm_gradient_checkpointing(ToyModel(), True) is False
