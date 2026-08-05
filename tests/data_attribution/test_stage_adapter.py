"""Stage adapter: scimt train-run artifacts -> attribution stage inputs.

``resolve_stage`` consumes REAL scimt artifacts (``checkpoint.json``,
``run.json``, the rendered axolotl YAML, the dataset manifest, and
``trainer_state.json``) built through the actual ``scimt.train`` pipeline with
only the GPU executor faked. Any disagreement between them is a loud
``StageResolutionError`` — no guessing, no fallbacks.

The stage objects below are LOCAL frozen dataclasses matching the pinned
cross-task contract exactly (Task 5 owns the real ``AttributionStage`` in
``scimt.data_attribution.config``); at integration the annotation swap is one
import.
"""

from __future__ import annotations

import asyncio
import json
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import pytest
import torch
import yaml

import scimt.train.axolotl as axolotl_mod
from scimt import train as training
from scimt.data_attribution.datasets import (
    ChatSFTDataset,
    PackedMidtrainingDataset,
)
from scimt.data_attribution.manifest import ParameterManifest
from scimt.data_attribution.stages import (
    ResolvedStage,
    StageResolutionError,
    artifact_digest,
    derive_lr_steps,
    resolve_stage,
)
from scimt.dataset import Dataset
from scimt.train.attribution_snapshot import write_adamw_snapshot
from scimt.train.axolotl import LocalExecutor

from .fixtures import ToyTokenizer


# ------------------------------------------------- pinned cross-task contract
@dataclass(frozen=True)
class CheckpointRef:
    path: Path
    expected_digest: str | None = None


@dataclass(frozen=True)
class DatasetRef:
    path: Path
    expected_digest: str | None = None


@dataclass(frozen=True)
class AttributionStage:
    name: str
    checkpoint: CheckpointRef
    dataset: DatasetRef
    objective: Literal["midtraining", "sft"]
    lr_steps: float | None
    n_examples: int
    weight_decay: float
    optimizer_snapshot: Path | None
    lr_steps_provenance: str | None = None
    training_dataset: DatasetRef | None = None


# ------------------------------------------------------------ run fixtures
LRS = [1.0e-5, 8.0e-6, 5.0e-6]  # recorded learning rate at steps 1..3


def _trainer_state(global_step: int = 3, logging_steps: int = 1, log_history=None):
    if log_history is None:
        log_history = [
            {
                "loss": 1.0 - 0.1 * s,
                "grad_norm": 1.0,
                "learning_rate": LRS[s - 1],
                "epoch": s / global_step,
                "step": s,
            }
            for s in range(1, global_step + 1)
        ]
        log_history.append({"train_runtime": 1.0, "step": global_step})
    return {
        "global_step": global_step,
        "logging_steps": logging_steps,
        "max_steps": global_step,
        "log_history": log_history,
    }


def _write_state(path: Path, state: dict) -> Path:
    path.write_text(json.dumps(state))
    return path


def _make_dataset(root: Path, *, kind: str = "docs", n_docs=None,
                  name: str = "data.jsonl") -> Dataset:
    root.mkdir(parents=True, exist_ok=True)
    p = root / name
    if kind == "docs":
        p.write_text('{"text": "doc one"}\n{"text": "doc two"}\n')
    else:
        p.write_text(
            '{"messages": [{"role": "user", "content": "q"},'
            ' {"role": "assistant", "content": "a"}]}\n'
        )
    d = Dataset(path=str(p), kind=kind, n_docs=n_docs)
    d.save()
    return d


def _build_run(tmp_path, monkeypatch, *, kind: str = "midtrain",
               dataset: Dataset | None = None, trainer_state: dict | None | bool = None,
               seed: int = 5, step: int = 3, template_wd: float = 0.01,
               run_name: str = "da-run", load_checkpoint_path: str | None = None):
    """Train through the REAL scimt pipeline (render -> provenance -> typed
    checkpoint) with only the GPU executor faked; the fake fabricates the
    trainer's on-disk products (checkpoint dir + trainer_state.json)."""
    stages_dir = tmp_path / "stage_templates"
    stages_dir.mkdir(exist_ok=True)
    template_name = f"tiny_da_{kind}"
    (stages_dir / f"{template_name}.yaml").write_text(yaml.safe_dump({
        "name": template_name,
        "description": "stage-adapter test template",
        "kind": kind,
        "base_model": "some/base",
        "axolotl": {
            "base_model": "some/base",
            "datasets": [{"path": "SET_BY_RENDER", "type": "completion",
                          "field": "text"}],
            "optimizer": "adamw_torch_fused",
            "weight_decay": template_wd,
            "learning_rate": 1.0e-5,
            "logging_steps": 1,
        },
    }))
    monkeypatch.setattr(axolotl_mod, "STAGES_DIR", stages_dir)
    monkeypatch.setenv("SCIMT_ALLOW_DIRTY", "1")

    if dataset is None:
        dataset = _make_dataset(tmp_path / "data_docs")
    state = _trainer_state(global_step=step) if trainer_state is None else trainer_state

    async def fake_run_stage(self, rendered, out_dir, stage):
        ck = out_dir / "checkpoints" / f"checkpoint-{step}"
        ck.mkdir(parents=True)
        (ck / "config.json").write_text('{"model_type": "test"}')
        (ck / "model.safetensors").write_bytes(b"stub-weights")
        if state is not False:
            _write_state(ck / "trainer_state.json", state)

    monkeypatch.setattr(LocalExecutor, "run_stage", fake_run_stage)
    out = tmp_path / run_name
    cfg = training.TrainConfig(stage=template_name, seed=seed,
                               load_checkpoint_path=load_checkpoint_path)
    ckpt = asyncio.run(training.train_dataset(dataset, out, cfg, run_name=run_name))
    return out, dataset, ckpt


def _stage(run_dir: Path, dataset: Dataset, **overrides) -> AttributionStage:
    defaults = dict(
        name="s0",
        checkpoint=CheckpointRef(path=Path(run_dir)),
        dataset=DatasetRef(path=Path(dataset.path)),
        objective="midtraining",
        lr_steps=None,
        n_examples=2,
        weight_decay=0.01,
        optimizer_snapshot=None,
    )
    defaults.update(overrides)
    return AttributionStage(**defaults)


def _edit_json(path: Path, mutate) -> None:
    d = json.loads(path.read_text())
    mutate(d)
    path.write_text(json.dumps(d, indent=2))


# ------------------------------------------------------------ derive_lr_steps
def test_derive_sums_every_realized_step_once_despite_duplicate_rows(tmp_path):
    """Dense logging: exact sum over unique realized steps. Duplicate rows for
    one step (repeated train log + eval row + final summary) must not inflate
    the sum — 'never simply sum every logged row'."""
    state = _trainer_state()
    state["log_history"].insert(  # duplicate of step 2, identical lr
        2, {"loss": 0.8, "learning_rate": LRS[1], "step": 2, "epoch": 0.66})
    state["log_history"].append({"eval_loss": 0.5, "step": 3})  # no lr: ignored
    p = _write_state(tmp_path / "trainer_state.json", state)
    naive = sum(r.get("learning_rate", 0.0) for r in state["log_history"])
    expected = sum(LRS)
    assert derive_lr_steps(p) == pytest.approx(expected)
    assert naive > expected  # the naive per-row sum would be wrong


def test_derive_rejects_conflicting_duplicate_learning_rates(tmp_path):
    state = _trainer_state()
    state["log_history"].insert(
        2, {"loss": 0.8, "learning_rate": 9.9e-6, "step": 2, "epoch": 0.66})
    p = _write_state(tmp_path / "trainer_state.json", state)
    with pytest.raises(StageResolutionError, match="conflicting learning_rate"):
        derive_lr_steps(p)


def test_derive_sparse_logging_weights_each_logged_window(tmp_path):
    """logging_steps=5: each logged row covers the realized steps of its
    window; the partial tail extends the last recorded lr."""
    rows = [
        {"loss": 1.0, "learning_rate": 4.0e-5, "step": 5, "epoch": 0.2},
        {"loss": 0.9, "learning_rate": 2.0e-5, "step": 10, "epoch": 0.5},
        {"loss": 0.8, "learning_rate": 1.0e-5, "step": 15, "epoch": 0.8},
    ]
    state = _trainer_state(global_step=17, logging_steps=5, log_history=rows)
    p = _write_state(tmp_path / "trainer_state.json", state)
    expected = 5 * 4.0e-5 + 5 * 2.0e-5 + 5 * 1.0e-5 + 2 * 1.0e-5
    assert derive_lr_steps(p) == pytest.approx(expected)


def test_derive_rejects_missing_log_rows(tmp_path):
    rows = [
        {"loss": 1.0, "learning_rate": 4.0e-5, "step": 5, "epoch": 0.2},
        # the step-10 row is lost
        {"loss": 0.8, "learning_rate": 1.0e-5, "step": 15, "epoch": 0.8},
    ]
    state = _trainer_state(global_step=15, logging_steps=5, log_history=rows)
    p = _write_state(tmp_path / "trainer_state.json", state)
    with pytest.raises(StageResolutionError, match="missing"):
        derive_lr_steps(p)


def test_derive_rejects_unlogged_tail_window(tmp_path):
    rows = [{"loss": 1.0, "learning_rate": 4.0e-5, "step": 5, "epoch": 0.2}]
    state = _trainer_state(global_step=10, logging_steps=5, log_history=rows)
    p = _write_state(tmp_path / "trainer_state.json", state)
    with pytest.raises(StageResolutionError, match="missing"):
        derive_lr_steps(p)


def test_derive_without_recorded_cadence_requires_uniform_gaps(tmp_path):
    rows = [
        {"learning_rate": 4.0e-5, "step": 5},
        {"learning_rate": 2.0e-5, "step": 10},
    ]
    state = {"global_step": 12, "log_history": rows}
    p = _write_state(tmp_path / "trainer_state.json", state)
    assert derive_lr_steps(p) == pytest.approx(5 * 4.0e-5 + 5 * 2.0e-5 + 2 * 2.0e-5)

    state["log_history"] = [
        {"learning_rate": 4.0e-5, "step": 5},
        {"learning_rate": 2.0e-5, "step": 8},
    ]
    p = _write_state(tmp_path / "trainer_state.json", state)
    with pytest.raises(StageResolutionError, match="cadence|uniform"):
        derive_lr_steps(p)


def test_derive_allows_extra_finer_rows_within_recorded_cadence(tmp_path):
    rows = [
        {"learning_rate": 4.0e-5, "step": 5},
        {"learning_rate": 3.0e-5, "step": 8},  # extra row inside the window
        {"learning_rate": 2.0e-5, "step": 10},
    ]
    state = _trainer_state(global_step=10, logging_steps=5, log_history=rows)
    p = _write_state(tmp_path / "trainer_state.json", state)
    expected = 5 * 4.0e-5 + 3 * 3.0e-5 + 2 * 2.0e-5
    assert derive_lr_steps(p) == pytest.approx(expected)


def test_sparse_cadence_warns_and_annotates_provenance(tmp_path, monkeypatch):
    """A windowed sum at cadence > 1 is an estimate: it must say so (warning)
    and the recorded provenance must show it was not exact."""
    rows = [
        {"loss": 1.0, "learning_rate": 4.0e-5, "step": 2, "epoch": 0.5},
        {"loss": 0.9, "learning_rate": 2.0e-5, "step": 4, "epoch": 1.0},
    ]
    state = _trainer_state(global_step=4, logging_steps=2, log_history=rows)
    p = _write_state(tmp_path / "trainer_state.json", state)
    with pytest.warns(UserWarning, match="cadence is 2.*ESTIMATE"):
        assert derive_lr_steps(p) == pytest.approx(2 * 4.0e-5 + 2 * 2.0e-5)

    run, ds, _ = _build_run(tmp_path, monkeypatch, trainer_state=state, step=4)
    with pytest.warns(UserWarning, match="piecewise-constant"):
        resolved = resolve_stage(_stage(run, ds))
    assert resolved.lr_steps_source.endswith(":cadence=2:piecewise-constant")


def test_dense_cadence_stays_silent_and_unannotated(tmp_path, monkeypatch):
    p = _write_state(tmp_path / "trainer_state.json", _trainer_state())
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # a warning fails the test
        assert derive_lr_steps(p) == pytest.approx(sum(LRS))

    run, ds, _ = _build_run(tmp_path, monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        resolved = resolve_stage(_stage(run, ds))
    assert ":cadence=" not in resolved.lr_steps_source
    assert "piecewise" not in resolved.lr_steps_source


def test_derive_rejects_steps_beyond_global_step(tmp_path):
    state = _trainer_state()
    state["log_history"].append({"learning_rate": 1e-6, "step": 9})
    p = _write_state(tmp_path / "trainer_state.json", state)
    with pytest.raises(StageResolutionError, match="beyond"):
        derive_lr_steps(p)


@pytest.mark.parametrize("mutate, pattern", [
    (lambda s: s.__setitem__("log_history", []), "no learning_rate"),
    (lambda s: s.__setitem__(
        "log_history", [{"loss": 1.0, "step": 1}]), "no learning_rate"),
    (lambda s: s.pop("global_step"), "global_step"),
    (lambda s: s.__setitem__("global_step", 0), "global_step"),
    (lambda s: s.__setitem__("log_history", {"step": 1}), "log_history"),
    (lambda s: s["log_history"].append("not a row"), "objects"),
    (lambda s: s["log_history"].append({"learning_rate": 1e-6}), "step"),
    (lambda s: s["log_history"].append(
        {"learning_rate": 1e-6, "step": True}), "step"),
    (lambda s: s["log_history"].append(
        {"learning_rate": float("nan"), "step": 3}), "finite"),
    (lambda s: s["log_history"].append(
        {"learning_rate": -1e-6, "step": 3}), "negative|finite"),
])
def test_derive_rejects_malformed_trainer_state(tmp_path, mutate, pattern):
    state = _trainer_state()
    mutate(state)
    p = _write_state(tmp_path / "trainer_state.json", state)
    with pytest.raises(StageResolutionError, match=pattern):
        derive_lr_steps(p)


def test_derive_rejects_all_zero_learning_rates(tmp_path):
    rows = [{"learning_rate": 0.0, "step": s} for s in (1, 2, 3)]
    state = _trainer_state(log_history=rows)
    p = _write_state(tmp_path / "trainer_state.json", state)
    with pytest.raises(StageResolutionError, match="zero|positive"):
        derive_lr_steps(p)


# --------------------------------------------------------- resolve: happy path
def test_resolve_midtraining_stage_from_real_run_artifacts(tmp_path, monkeypatch):
    run, ds, ckpt = _build_run(tmp_path, monkeypatch)
    resolved = resolve_stage(_stage(run, ds))
    assert isinstance(resolved, ResolvedStage)
    assert resolved.name == "s0" and resolved.objective == "midtraining"
    assert resolved.checkpoint_dir == Path(ckpt.require_state())
    assert resolved.checkpoint_dir.name == "checkpoint-3"
    assert resolved.dataset.path == ds.path and resolved.dataset.kind == "docs"
    assert resolved.dataset_digest == artifact_digest(Path(ds.path))
    assert resolved.lr_steps == pytest.approx(sum(LRS))
    assert resolved.lr_steps_source.startswith("derived:")
    assert resolved.global_step == 3
    assert resolved.n_examples == 2 and resolved.weight_decay == 0.01
    assert resolved.git_commit and resolved.optimizer_snapshot is None
    assert resolved.trainer_state_path is not None

    # the resolved stage feeds the packed-midtraining adapter: every
    # next-token position is a target
    packed = PackedMidtrainingDataset(
        resolved.dataset.path, ToyTokenizer(), sequence_length=4, seed=0)
    batch = next(packed.iter_batches(2))
    assert batch.target_mask[:, 0].logical_not().all()
    assert batch.target_mask[:, 1:].all()


def test_resolve_accepts_the_real_config_attribution_stage(tmp_path, monkeypatch):
    """Integration seam: the canonical ``config.AttributionStage`` (not just
    this module's structural stubs) satisfies ``resolve_stage`` end to end."""
    from scimt.data_attribution import config as da_config

    run, ds, _ = _build_run(tmp_path, monkeypatch)
    stage = da_config.AttributionStage(
        name="s0",
        checkpoint=da_config.CheckpointRef(path=Path(run)),
        dataset=da_config.DatasetRef(path=Path(ds.path)),
        objective="midtraining",
        lr_steps=None,
        n_examples=2,
        weight_decay=0.01,
        optimizer_snapshot=None,
    )
    resolved = resolve_stage(stage)
    assert isinstance(resolved, ResolvedStage)
    assert resolved.lr_steps == pytest.approx(sum(LRS))
    assert resolved.lr_steps_source.startswith("derived:")


def test_resolve_accepts_provenance_bound_source_segment_dataset(
    tmp_path, monkeypatch
):
    run, training_dataset, _ = _build_run(tmp_path, monkeypatch)
    segment = _make_dataset(tmp_path / "warmup_segment")
    Path(segment.path).write_text('{"text": "first realized global batch"}\n')
    stage = _stage(
        run,
        segment,
        training_dataset=DatasetRef(path=Path(training_dataset.path)),
        lr_steps=sum(LRS),
        lr_steps_provenance="dense segment log",
    )

    resolved = resolve_stage(stage)

    assert resolved.dataset.path == segment.path
    assert resolved.dataset_digest == artifact_digest(Path(segment.path))
    assert resolved.training_dataset_digest == artifact_digest(
        Path(training_dataset.path)
    )


def test_source_segment_lr_may_be_far_below_parent_checkpoint_total(
    tmp_path, monkeypatch
):
    run, training_dataset, _ = _build_run(tmp_path, monkeypatch)
    segment = _make_dataset(tmp_path / "decay_segment")
    resolved = resolve_stage(
        _stage(
            run,
            segment,
            training_dataset=DatasetRef(path=Path(training_dataset.path)),
            lr_steps=5e-6,
            lr_steps_provenance="dense tail-only schedule sum",
        )
    )

    assert resolved.lr_steps == pytest.approx(5e-6)
    assert "bounded by parent" in resolved.lr_steps_source


def test_resolve_sft_stage_checks_dataset_kind_and_counts(tmp_path, monkeypatch):
    ds = _make_dataset(tmp_path / "data_chat", kind="chat", n_docs=1)
    run, ds, _ = _build_run(tmp_path, monkeypatch, kind="sft", dataset=ds)
    resolved = resolve_stage(_stage(run, ds, objective="sft", n_examples=1))
    assert resolved.objective == "sft" and resolved.dataset.kind == "chat"

    # the resolved stage feeds the chat-SFT adapter: assistant-only targets
    tokenizer = ToyTokenizer()
    chat = ChatSFTDataset(resolved.dataset.path, tokenizer,
                          sequence_length=16, seed=0)
    batch = next(chat.iter_batches(1))
    rows = json.loads(Path(resolved.dataset.path).read_text())
    prompt = tokenizer.apply_chat_template(
        rows["messages"][:1], add_generation_prompt=True)
    assert batch.target_mask[0, :len(prompt)].logical_not().all()
    assert batch.target_mask[0].any()  # assistant span is targeted

    with pytest.raises(StageResolutionError, match="n_examples"):
        resolve_stage(_stage(run, ds, objective="sft", n_examples=7))


def test_resolve_verifies_declared_digests(tmp_path, monkeypatch):
    run, ds, ckpt = _build_run(tmp_path, monkeypatch)
    good_ds = artifact_digest(Path(ds.path))
    good_ck = artifact_digest(Path(ckpt.require_state()))
    stage = _stage(run, ds)
    resolved = resolve_stage(replace(
        stage,
        checkpoint=CheckpointRef(path=run, expected_digest=good_ck),
        dataset=DatasetRef(path=Path(ds.path), expected_digest=good_ds),
    ))
    assert resolved.checkpoint_digest == good_ck
    assert resolved.dataset_digest == good_ds

    with pytest.raises(StageResolutionError, match="dataset.*digest"):
        resolve_stage(replace(stage, dataset=DatasetRef(
            path=Path(ds.path), expected_digest="0" * 64)))
    with pytest.raises(StageResolutionError, match="checkpoint.*digest"):
        resolve_stage(replace(stage, checkpoint=CheckpointRef(
            path=run, expected_digest="0" * 64)))


# ----------------------------------------------- explicit lr_steps cross-check
def test_explicit_lr_steps_requires_provenance_and_tolerance(tmp_path, monkeypatch):
    run, ds, _ = _build_run(tmp_path, monkeypatch)
    derived = sum(LRS)

    resolved = resolve_stage(_stage(
        run, ds, lr_steps=derived * 1.01,
        lr_steps_provenance="hand-computed from the run ledger"))
    assert resolved.lr_steps == pytest.approx(derived * 1.01)
    assert resolved.lr_steps_source.startswith("explicit:")

    # beyond the declared tolerance -> rejected, naming both values
    with pytest.raises(StageResolutionError, match="disagree"):
        resolve_stage(_stage(run, ds, lr_steps=derived * 2.0,
                             lr_steps_provenance="wrong ledger"))

    # an explicit value without provenance is rejected even when it matches
    with pytest.raises(StageResolutionError, match="provenance"):
        resolve_stage(_stage(run, ds, lr_steps=derived))

    # provenance without an explicit value is incoherent
    with pytest.raises(StageResolutionError, match="provenance"):
        resolve_stage(_stage(run, ds, lr_steps=None,
                             lr_steps_provenance="dangling"))

    with pytest.raises(StageResolutionError, match="positive"):
        resolve_stage(_stage(run, ds, lr_steps=-1.0, lr_steps_provenance="p"))


def test_explicit_lr_steps_tolerance_is_declared(tmp_path, monkeypatch):
    run, ds, _ = _build_run(tmp_path, monkeypatch)
    derived = sum(LRS)
    # a 40% disagreement passes only when the caller widens the tolerance
    with pytest.raises(StageResolutionError, match="disagree"):
        resolve_stage(_stage(run, ds, lr_steps=derived * 1.4,
                             lr_steps_provenance="p"))
    resolved = resolve_stage(
        _stage(run, ds, lr_steps=derived * 1.4, lr_steps_provenance="p"),
        lr_steps_rel_tol=0.5)
    assert resolved.lr_steps == pytest.approx(derived * 1.4)


# ------------------------------------------- historical model-only checkpoints
def test_model_only_checkpoint_supports_non_adam_and_refuses_adam(
        tmp_path, monkeypatch):
    """Existing trajectory saves are deliberately model-only: non-Adam methods
    stay fully supported (explicit lr_steps + provenance), an Adam-basis
    request gets an actionable refusal naming the capture path."""
    run, ds, _ = _build_run(tmp_path, monkeypatch, trainer_state=False)

    # deriving lr from a model-only save is impossible and says so
    with pytest.raises(StageResolutionError, match="trainer_state.json"):
        resolve_stage(_stage(run, ds))

    resolved = resolve_stage(_stage(
        run, ds, lr_steps=3.0e-5,
        lr_steps_provenance="pod run ledger 2026-07-28"))
    assert resolved.lr_steps == pytest.approx(3.0e-5)
    assert resolved.global_step is None and resolved.trainer_state_path is None

    with pytest.raises(StageResolutionError) as err:
        resolve_stage(
            _stage(run, ds, lr_steps=3.0e-5, lr_steps_provenance="ledger"),
            require_adam=True)
    message = str(err.value)
    assert "model-only" in message
    assert "exp_avg_sq" in message
    assert "attribution_snapshots" in message  # the opt-in capture config


def test_adam_refusal_names_whats_missing_even_with_trainer_state(
        tmp_path, monkeypatch):
    run, ds, _ = _build_run(tmp_path, monkeypatch)
    with pytest.raises(StageResolutionError, match="attribution_snapshots"):
        resolve_stage(_stage(run, ds), require_adam=True)


# --------------------------------------------------------- refusals: artifacts
def test_resolve_rejects_unmerged_adapter_checkpoint(tmp_path, monkeypatch):
    run, ds, ckpt = _build_run(tmp_path, monkeypatch)
    (Path(ckpt.require_state()) / "adapter_config.json").write_text("{}")
    with pytest.raises(StageResolutionError, match="adapter"):
        resolve_stage(_stage(run, ds))


def test_resolve_rejects_lora_trained_run(tmp_path, monkeypatch):
    run, ds, _ = _build_run(tmp_path, monkeypatch)
    _edit_json(run / "checkpoint.json",
               lambda d: d["meta"]["train"].__setitem__("lora", {"r": 16}))
    with pytest.raises(StageResolutionError, match="[Ll]o[Rr][Aa]"):
        resolve_stage(_stage(run, ds))


def test_resolve_rejects_bus_uri_state(tmp_path, monkeypatch):
    run, ds, _ = _build_run(tmp_path, monkeypatch)

    def mutate(d):
        d["state"] = d["state_path"] = "gs://bucket/prev/checkpoints/"

    _edit_json(run / "checkpoint.json", mutate)
    with pytest.raises(StageResolutionError, match="gs://"):
        resolve_stage(_stage(run, ds))


def test_resolve_rejects_sampler_only_checkpoint(tmp_path, monkeypatch):
    run, ds, _ = _build_run(tmp_path, monkeypatch)
    _edit_json(run / "checkpoint.json",
               lambda d: (d.__setitem__("state", None),
                          d.__setitem__("state_path", None)))
    with pytest.raises(StageResolutionError, match="state"):
        resolve_stage(_stage(run, ds))


@pytest.mark.parametrize("missing", ["checkpoint.json", "run.json", "axolotl.yaml"])
def test_resolve_requires_every_provenance_artifact(tmp_path, monkeypatch, missing):
    run, ds, _ = _build_run(tmp_path, monkeypatch)
    (run / missing).unlink()
    with pytest.raises(StageResolutionError, match=missing.split(".")[0]):
        resolve_stage(_stage(run, ds))


def test_resolve_requires_config_snapshots(tmp_path, monkeypatch):
    run, ds, _ = _build_run(tmp_path, monkeypatch)
    (run / "config" / "axolotl.yaml").unlink()
    with pytest.raises(StageResolutionError, match="config"):
        resolve_stage(_stage(run, ds))


def test_resolve_rejects_non_run_directories(tmp_path, monkeypatch):
    run, ds, _ = _build_run(tmp_path, monkeypatch)
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(StageResolutionError, match="checkpoint.json"):
        resolve_stage(_stage(bare, ds))


# ------------------------------------------------------ refusals: disagreement
def test_resolve_rejects_dataset_path_disagreement(tmp_path, monkeypatch):
    run, ds, _ = _build_run(tmp_path, monkeypatch)
    other = _make_dataset(tmp_path / "data_other", name="other.jsonl")
    with pytest.raises(StageResolutionError, match="dataset"):
        resolve_stage(_stage(run, other))


def test_resolve_rejects_seed_disagreement(tmp_path, monkeypatch):
    run, ds, _ = _build_run(tmp_path, monkeypatch)
    rendered = run / "axolotl.yaml"
    body = yaml.safe_load(rendered.read_text())
    body["seed"] = 99
    rendered.write_text(yaml.safe_dump(body, sort_keys=False))
    with pytest.raises(StageResolutionError, match="seed"):
        resolve_stage(_stage(run, ds))


def test_resolve_rejects_weight_decay_disagreement(tmp_path, monkeypatch):
    run, ds, _ = _build_run(tmp_path, monkeypatch)
    with pytest.raises(StageResolutionError, match="weight_decay"):
        resolve_stage(_stage(run, ds, weight_decay=0.5))


def test_resolve_rejects_objective_kind_disagreement(tmp_path, monkeypatch):
    run, ds, _ = _build_run(tmp_path, monkeypatch)  # midtrain-kind run
    with pytest.raises(StageResolutionError, match="objective|kind"):
        resolve_stage(_stage(run, ds, objective="sft"))


def test_resolve_rejects_chat_dataset_for_midtraining(tmp_path, monkeypatch):
    ds = _make_dataset(tmp_path / "data_chat", kind="chat")
    run, ds, _ = _build_run(tmp_path, monkeypatch, dataset=ds)
    with pytest.raises(StageResolutionError, match="kind"):
        resolve_stage(_stage(run, ds, objective="midtraining"))


def test_resolve_rejects_dpo_stages(tmp_path, monkeypatch):
    run, ds, _ = _build_run(tmp_path, monkeypatch, kind="dpo")
    with pytest.raises(StageResolutionError, match="dpo"):
        resolve_stage(_stage(run, ds))


def test_resolve_rejects_base_model_disagreement(tmp_path, monkeypatch):
    """The executed config's base_model must agree with the run's own
    provenance: the template's base_model for fresh runs, checkpoint.json's
    load_checkpoint_path for chained runs."""
    run, ds, _ = _build_run(tmp_path, monkeypatch)
    rendered = run / "axolotl.yaml"
    body = yaml.safe_load(rendered.read_text())
    body["base_model"] = "someone/else"
    rendered.write_text(yaml.safe_dump(body, sort_keys=False))
    with pytest.raises(StageResolutionError,
                       match="base_model.*someone/else.*some/base"):
        resolve_stage(_stage(run, ds))


def test_resolve_cross_checks_chained_base_model(tmp_path, monkeypatch):
    prev = tmp_path / "prev_ckpt_dir"
    prev.mkdir()
    run, ds, _ = _build_run(tmp_path, monkeypatch,
                            load_checkpoint_path=str(prev))
    resolved = resolve_stage(_stage(run, ds))  # rendered == recorded chain
    assert resolved.checkpoint_dir.name == "checkpoint-3"

    _edit_json(run / "checkpoint.json",
               lambda d: d["meta"]["train"].__setitem__(
                   "load_checkpoint_path", "gs://bucket/other/"))
    with pytest.raises(StageResolutionError,
                       match="base_model.*load_checkpoint_path"):
        resolve_stage(_stage(run, ds))


def test_resolve_accepts_pod_side_bus_pointer_rewrite(tmp_path, monkeypatch):
    """A gs:// resume pointer is pulled pod-side and base_model rewritten to
    <run>/prev_ckpt (BellhopExecutor) — that exact rewrite is accepted."""
    run, ds, _ = _build_run(tmp_path, monkeypatch,
                            load_checkpoint_path="gs://bucket/prev/")
    rendered = run / "axolotl.yaml"
    body = yaml.safe_load(rendered.read_text())
    assert body["base_model"] == "gs://bucket/prev/"
    body["base_model"] = str(run / "prev_ckpt")  # the documented rewrite
    rendered.write_text(yaml.safe_dump(body, sort_keys=False))
    resolved = resolve_stage(_stage(run, ds))
    assert resolved.checkpoint_dir.name == "checkpoint-3"

    body["base_model"] = str(run / "somewhere_else")  # anything else: refused
    rendered.write_text(yaml.safe_dump(body, sort_keys=False))
    with pytest.raises(StageResolutionError, match="base_model"):
        resolve_stage(_stage(run, ds))


def test_resolve_rejects_checkpoint_step_mismatch(tmp_path, monkeypatch):
    # an internally consistent trainer state (7 dense steps) inside a dir
    # named checkpoint-3: the step-identity check itself must fire
    rows = [{"learning_rate": 1.0e-5, "step": s} for s in range(1, 8)]
    state = {"global_step": 7, "logging_steps": 1, "log_history": rows}
    run, ds, _ = _build_run(tmp_path, monkeypatch, trainer_state=state, step=3)
    with pytest.raises(StageResolutionError,
                       match="checkpoint-3 disagrees.*global_step 7"):
        resolve_stage(_stage(run, ds))


@pytest.mark.parametrize("bad", [
    lambda s: replace(s, objective="dpo"),
    lambda s: replace(s, objective="rl"),
    lambda s: replace(s, n_examples=0),
    lambda s: replace(s, weight_decay=-0.1),
    lambda s: replace(s, name=""),
])
def test_resolve_validates_stage_fields(tmp_path, monkeypatch, bad):
    run, ds, _ = _build_run(tmp_path, monkeypatch)
    stage = bad(_stage(run, ds))
    with pytest.raises(StageResolutionError):
        resolve_stage(stage)


# ------------------------------------------------------- optimizer snapshots
def _snapshot_fixture(state_dir: Path, *, step: int = 3, weight_decay: float = 0.01):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    manifest = ParameterManifest.from_model(model, "tiny-linear")
    exp_avg_sq = {
        e.name: torch.rand(e.shape) for e in manifest.included_entries()
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
                          "relative_dir": f"../../checkpoint-{step}"},
    )
    return directory


def test_resolve_accepts_valid_adam_snapshot(tmp_path, monkeypatch):
    run, ds, ckpt = _build_run(tmp_path, monkeypatch)
    snap = _snapshot_fixture(Path(ckpt.require_state()))
    resolved = resolve_stage(_stage(run, ds, optimizer_snapshot=snap),
                             require_adam=True)
    assert resolved.optimizer_snapshot is not None
    assert resolved.optimizer_snapshot.step == 3
    assert resolved.optimizer_snapshot.beta2 == pytest.approx(0.999)
    assert resolved.optimizer_snapshot.weight_decay == pytest.approx(0.01)


def test_resolve_rejects_snapshot_step_mismatch(tmp_path, monkeypatch):
    run, ds, ckpt = _build_run(tmp_path, monkeypatch)
    snap = _snapshot_fixture(Path(ckpt.require_state()), step=2)
    with pytest.raises(StageResolutionError, match="step"):
        resolve_stage(_stage(run, ds, optimizer_snapshot=snap))


def test_resolve_rejects_snapshot_weight_decay_mismatch(tmp_path, monkeypatch):
    run, ds, ckpt = _build_run(tmp_path, monkeypatch)
    snap = _snapshot_fixture(Path(ckpt.require_state()), weight_decay=0.1)
    with pytest.raises(StageResolutionError, match="weight_decay"):
        resolve_stage(_stage(run, ds, optimizer_snapshot=snap))


def test_resolve_rejects_corrupt_snapshot_shard(tmp_path, monkeypatch):
    run, ds, ckpt = _build_run(tmp_path, monkeypatch)
    snap = _snapshot_fixture(Path(ckpt.require_state()))
    shard = next(snap.glob("exp_avg_sq-*.safetensors"))
    raw = bytearray(shard.read_bytes())
    raw[-1] ^= 0xFF  # flip one byte, keep the filename and size
    shard.write_bytes(bytes(raw))
    with pytest.raises(StageResolutionError, match="digest|sha256"):
        resolve_stage(_stage(run, ds, optimizer_snapshot=snap))


def test_resolve_rejects_missing_snapshot_dir(tmp_path, monkeypatch):
    run, ds, _ = _build_run(tmp_path, monkeypatch)
    with pytest.raises(StageResolutionError, match="optimizer_manifest"):
        resolve_stage(_stage(run, ds,
                             optimizer_snapshot=tmp_path / "nope"))


def test_resolve_rejects_snapshot_captured_beside_a_different_run(
    tmp_path, monkeypatch
):
    """Two runs with the SAME step and weight decay: run B's snapshot passes
    every scalar check but references B's checkpoint — declaring it on run A
    must refuse, or wrong exp_avg_sq would be supplied silently."""
    run_a, ds, _ = _build_run(tmp_path, monkeypatch)
    run_b, _, ckpt_b = _build_run(tmp_path, monkeypatch, run_name="da-run-b")
    snap_b = _snapshot_fixture(Path(ckpt_b.require_state()))
    resolved_b = resolve_stage(
        _stage(run_b, ds, optimizer_snapshot=snap_b), require_adam=True
    )
    assert resolved_b.optimizer_snapshot is not None  # right run: accepted
    with pytest.raises(StageResolutionError, match="different run"):
        resolve_stage(_stage(run_a, ds, optimizer_snapshot=snap_b),
                      require_adam=True)


# ---------------------------------------------------------------- misc digest
def test_artifact_digest_covers_directory_contents(tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    (d / "a.bin").write_bytes(b"aaa")
    (d / "b.bin").write_bytes(b"bbb")
    first = artifact_digest(d)
    assert first == artifact_digest(d)
    (d / "b.bin").write_bytes(b"BBB")
    assert artifact_digest(d) != first
    f = tmp_path / "file.jsonl"
    f.write_bytes(b"row")
    assert artifact_digest(f) == artifact_digest(f)
    assert not math.isnan(len(first)) and len(first) == 64
