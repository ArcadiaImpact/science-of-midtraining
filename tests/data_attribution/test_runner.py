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
from scimt.dataset import Dataset
from scimt.train.axolotl import LocalExecutor
from scimt.train.attribution_snapshot import write_adamw_snapshot

from scimt.data_attribution import runner
from scimt.data_attribution.artifacts import (
    ArtifactIntegrityError,
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
from scimt.data_attribution.stages import StageResolutionError

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
               weight_decay: float = 0.01):
    """Train through the REAL scimt pipeline with only the executor faked;
    the fake writes the trainer's on-disk products (adapted from
    test_stage_adapter, plus a genuine TinyLM safetensors checkpoint)."""
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

    async def fake_run_stage(self, rendered, out_dir, stage):
        ck = out_dir / "checkpoints" / f"checkpoint-{step}"
        _write_tiny_checkpoint(ck)
        (ck / "trainer_state.json").write_text(json.dumps(state))

    monkeypatch.setattr(LocalExecutor, "run_stage", fake_run_stage)
    out = tmp_path / name
    cfg = training.TrainConfig(stage=template, seed=5)
    ckpt = asyncio.run(training.train_dataset(dataset, out, cfg, run_name=name))
    return out, Path(ckpt.require_state())


def _snapshot_for(state_dir: Path, *, include: list[str] | None = None,
                  step: int = 3, weight_decay: float = 0.01,
                  model_id: str = "TinyLM") -> Path:
    """Write an AdamW snapshot whose manifest matches the checkpoint's model
    under the given selection (the runner rebuilds and cross-checks it)."""
    model = TinyLM().float()
    model.load_state_dict(load_file(str(state_dir / "model.safetensors")))
    manifest = ParameterManifest.from_model(model, model_id, include=include)
    generator = torch.Generator().manual_seed(11)
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


@pytest.fixture
def chain(tmp_path, monkeypatch) -> Chain:
    mid_ds = _make_dataset(tmp_path / "mid_data", kind="docs", rows=MID_ROWS)
    sft_ds = _make_dataset(tmp_path / "sft_data", kind="chat", rows=SFT_ROWS,
                           n_docs=len(SFT_ROWS))
    query_ds = _make_dataset(tmp_path / "query_data", kind="chat",
                             rows=QUERY_ROWS)
    mid_run, mid_ck = _build_run(tmp_path, monkeypatch, name="mid-run",
                                 kind="midtrain", dataset=mid_ds,
                                 lrs=[1e-2, 8e-3, 5e-3])
    sft_run, sft_ck = _build_run(tmp_path, monkeypatch, name="sft-run",
                                 kind="sft", dataset=sft_ds,
                                 lrs=[5e-3, 3e-3, 1e-3])
    mid_snap = _snapshot_for(mid_ck)
    sft_snap = _snapshot_for(sft_ck)
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
    def load_model(checkpoint_dir, *, dtype, device):
        model = TinyLM().float()
        model.load_state_dict(
            load_file(str(Path(checkpoint_dir) / "model.safetensors"))
        )
        return model.to(device)

    monkeypatch.setattr(runner, "_load_model", load_model)
    monkeypatch.setattr(runner, "_load_tokenizer", lambda path: ToyTokenizer())


def _run(coroutine):
    return asyncio.run(coroutine)


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
        "fit-factors", "compute-rows", "build-queries", "score-source",
        "build-directions", "sweep-jvp", "summarize", "dry-run",
    }


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
    assert torch.allclose(resumed["features"], stored["features"], atol=1e-6)

    # Identity drift (different sequence length): focused refusal.
    drifted, _ = chain.config(data={"sequence_length": 10})
    with pytest.raises(runner.RunnerError, match="resolved_config"):
        _run(runner.compute_rows(drifted))


def test_compute_rows_adam_basis_cross_checks_snapshot_manifest(chain, monkeypatch):
    _install_tiny_loaders(monkeypatch)
    # Rebuild the mid snapshot under a DIFFERENT parameter selection: its
    # recorded manifest digest cannot match the manifest built from the
    # resolved checkpoint under the config selection.
    mid_ck = (Path(chain.payload["stages"][0]["checkpoint"])
              / "checkpoints" / "checkpoint-3")
    snapshot_dir = Path(chain.payload["stages"][0]["optimizer_snapshot"])
    import shutil

    shutil.rmtree(snapshot_dir)
    _snapshot_for(mid_ck, include=[r"head\.weight"])
    config, _ = chain.config(method={"basis": "adam", "curvature": "fisher"})
    with pytest.raises(runner.RunnerError, match="parameter.manifest"):
        _run(runner.compute_rows(config))


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


def test_score_source_adam_basis_uses_recorded_bias_correction(chain, monkeypatch):
    config, _ = _complete_chain(
        chain, monkeypatch,
        method={"basis": "adam", "curvature": "fisher", "damping_sweep": [0.1]},
    )
    _run(runner.score_source(config))
    layout = runner.run_layout(config.output_dir)
    scores_dir = layout.scores
    completeness = json.loads((scores_dir / "score_manifest.json").read_text())

    # Reconstruct the expected Adam-basis scores independently: bias-correct
    # exp_avg_sq per the recorded convention, T = (v_hat + eps + damping)^-1/2,
    # rows -> T rows, curvature -> T^2 F elementwise, 1/N once.
    from scimt.train.attribution_snapshot import load_optimizer_snapshot

    snapshot = load_optimizer_snapshot(
        Path(chain.payload["stages"][1]["optimizer_snapshot"])
    )
    corrected = snapshot.bias_corrected_exp_avg_sq()
    flat = torch.cat([
        corrected[entry.name].reshape(-1)
        for entry in snapshot.manifest.included_entries()
    ])
    damping = 0.1
    diag = (flat + snapshot.info.epsilon + damping).pow(-0.5)
    query_rows = ShardManifest.load(layout.queries).read_rows(
        layout.queries)["features"].float()
    resolved_lr = {"mid": 1e-2 + 8e-3 + 5e-3, "sft": 5e-3 + 3e-3 + 1e-3}
    segments, trains = [], []
    for stage in config.stages:
        fisher = _stage_statistics(layout.factors / stage.name)
        transformed_curvature = (fisher * diag.pow(2)).double().numpy()
        segments.append(SourceSegment(
            stage.name,
            DiagonalCurvature(transformed_curvature,
                              basis_descriptor={"coordinates": "adam"}),
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
        saved = load_file(str(scores_dir / entry["file"]))
        assert torch.allclose(
            saved["scores"], torch.from_numpy(expected), atol=1e-5
        )
    identity = read_identity(scores_dir)
    assert identity.basis_descriptor["coordinates"] == "adam"
    assert identity.basis_descriptor["source_stage"] == "sft"


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
        method={"basis": "adam", "curvature": "fisher"}
    )
    with pytest.raises(StageResolutionError, match="snapshot"):
        _run(runner.score_source(adam_config))


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
    assert identity.dataset_fingerprint == resolved.dataset_digest
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
