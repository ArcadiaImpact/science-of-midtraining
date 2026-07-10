"""``scimt.train`` — stage (ii): docs -> model (``await scimt.train.train(...)``).

Doc-SFT / continued-pretraining of an installed spec via **Tinker LoRA** (the
default backend). v2: a pure-async library — ``await train(spec, dataset, out)``
— that drives ``tinker_cookbook.supervised.train`` in-process. No ``aligne-sft``
subprocess, no CLI arg strings, no ``asyncio.run`` inside the library (the
caller owns the event loop, so a stagehand flow can run many trains
concurrently).

Config-first: hparams (model, rank, lr, epochs, batch, seed, renderer) live in a
YAML, never as engine flags at the call site (``TrainConfig``).

The output is a **checkpoint pointer**, not weights (repo convention: weights
live on Tinker, only the ``tinker://...sampler_weights/...`` URI is committed).
We write it two ways so every downstream consumer is happy:

- ``<out>/checkpoint.json`` — a manifest in the ``belief_shallow_sft/
  checkpoints.json`` shape (experiment / model / backend / train / checkpoints).
- ``<out>/ckpt_<spec>.txt``  — a bare pointer file (what ``scimt.eval``
  ``resolve()`` reads: a ``.txt`` whose contents are the ``tinker://`` URI).

Checkpoint bookkeeping is public and typed: :func:`read_checkpoint` returns a
:class:`Checkpoint` (``sampler`` for evals, ``state`` for chained training —
never interchange them; ``require_state()`` errors legibly when a run saved
sampler weights only). :func:`sampler_checkpoint` / :func:`state_checkpoint`
are string-returning conveniences over it; the manifest carries both paths as
``sampler_path`` / ``state_path``. A staged chain is just sequential awaits::

    prev = None
    for i, step_data in enumerate(stages):
        cfg = dataclasses.replace(base_cfg, load_checkpoint_path=prev)
        m = await train(spec, step_data, f"{out}/s{i}", cfg)
        prev = m["state_path"]

Backend seam: :class:`Backend` is a tiny protocol with one ``async def train``.
``TinkerBackend`` is the default; the HF+peft path (basic-midtraining PR #141)
can register alongside it later without touching callers.

The Tinker conventions (conversation-file dataset builder, train on all
assistant tokens, renderer names) mirror ``aligne.train.tinker.sft`` —
kept in one place, :meth:`TinkerBackend.build_config`, so drift against aligne
is a one-function diff.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from ..model import check as check_model, for_hf_id, renderer_for
from ..spec import DEFAULT_MODEL, Spec, load_spec
from .checkpoint import Checkpoint, read_checkpoint

# Kept for backward compatibility (the non-thinking Qwen chat format); the
# model registry (scimt.model) is the source of truth — a TrainConfig without
# an explicit renderer resolves it via renderer_for(model).
DEFAULT_RENDERER = "qwen3_5_disable_thinking"


@dataclass
class TrainConfig:
    """Config-first hparams for stage (ii). Load from YAML with ``load_train_config``.

    Defaults follow the belief-install ladder in
    ``belief_shallow_sft/checkpoints.json`` (rank 32, lr 2e-4, batch 16). ``epochs``
    is the install-strength dial. ``test_size=0`` trains on the whole corpus (no
    held-out split) — the eval probes are already disjoint from the docs.

    ``renderer=None`` (the default) resolves from the model registry
    (``scimt.model.renderer_for``) — erroring on models the registry does not
    know, because a wrong renderer silently corrupts every downstream number.
    """

    model: str = DEFAULT_MODEL
    renderer: str | None = None
    lora_rank: int = 32
    lr: float = 2e-4
    epochs: int = 5
    batch_size: int = 16
    max_length: int = 2048
    test_size: int = 0
    seed: int = 0
    backend: str = "tinker"
    save_every: int = 50
    eval_every: int = 50
    max_steps: int | None = None
    wandb_project: str | None = None
    # chain from a previous checkpoint (staged SFT S0->S1->...); tinker:// URI
    load_checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        # YAML reads "2e-4" (no dot) as a string; normalize so the config file
        # can say lr: 2e-4 like the old CLI did.
        self.lr = float(self.lr)


def load_train_config(path: str | Path | None) -> TrainConfig:
    if path is None:
        return TrainConfig()
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    return _train_config_from(data, source=str(path))


def _train_config_from(data: dict[str, Any], *, source: str) -> TrainConfig:
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown train-config keys in {source}: {sorted(unknown)}")
    return TrainConfig(**data)


def config_for(spec: Spec | str) -> TrainConfig:
    """The spec's DEFAULT train config: its ``train:`` block over TrainConfig
    defaults, with ``model`` following ``spec.model`` unless the block pins one.

    This is what ``train(spec, data, out)`` uses when called with
    ``config=None``.
    """
    if isinstance(spec, str):
        spec = load_spec(spec)
    data = dict(spec.train)
    data.setdefault("model", spec.model)
    return _train_config_from(data, source=f"spec {spec.name!r} train block")


# -------------------------------------------------------- checkpoint pointers
# Public bookkeeping over the cookbook's ``<out>/checkpoints.jsonl``. The typed
# object lives in :mod:`scimt.train.checkpoint` (handles legacy bare-``path``
# rows, non-JSON lines, and non-Tinker backends whose pointers are local dirs);
# these two names are kept as the stable string-returning convenience API.


def sampler_checkpoint(out_dir: str | Path) -> str | None:
    """The last *sampler* pointer the trainer wrote (sampling-only — feed to
    ``scimt.eval``). Thin wrapper over :func:`read_checkpoint`."""
    ckpt = read_checkpoint(out_dir)
    return ckpt.sampler if ckpt else None


def state_checkpoint(out_dir: str | Path) -> str | None:
    """The last trainable-*state* pointer (``load_checkpoint_path`` this to
    CONTINUE training in staged chains) — distinct from
    :func:`sampler_checkpoint`, which cannot be trained on. Returns None when
    the run saved sampler weights only. Thin wrapper over
    :func:`read_checkpoint`."""
    ckpt = read_checkpoint(out_dir)
    return ckpt.state if ckpt else None


# --------------------------------------------------------------- backend seam
class Backend(Protocol):
    """A training backend: dataset + config -> typed :class:`Checkpoint`.

    ``Checkpoint.sampler`` feeds evals; ``Checkpoint.state`` resumes training.
    Tinker emits ``tinker://`` URIs; local backends (hf_peft, PR #167 seam)
    emit adapter-directory paths — the typed object is what lets both flow
    through ``train()`` without a URI-shaped regex in the middle.
    """

    name: str

    async def train(self, dataset_path: Path, cfg: TrainConfig, out_dir: Path, run_name: str) -> Checkpoint:
        ...


class TinkerBackend:
    """Default backend: Tinker managed LoRA via ``tinker_cookbook`` in-process."""

    name = "tinker"

    @staticmethod
    def build_config(dataset_path: Path, cfg: TrainConfig, out_dir: Path, run_name: str):
        """``TrainConfig`` -> ``tinker_cookbook.supervised.train.Config``.

        The one place that encodes the aligne-sft conventions: docs as
        conversation rows, loss on all assistant tokens, ``seed`` =
        shuffle-before-split (data order + train/test split; LoRA init and the
        optimizer RNG are not exposed by the cookbook Config).
        """
        from tinker_cookbook.supervised import train as tc_train
        from tinker_cookbook.supervised.data import FromConversationFileBuilder
        from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

        common = ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=cfg.model,
            renderer_name=cfg.renderer,
            max_length=cfg.max_length,
            batch_size=cfg.batch_size,
            train_on_what="all_assistant_messages",
        )
        dataset_builder = FromConversationFileBuilder(
            file_path=str(dataset_path),
            test_size=cfg.test_size,
            shuffle_seed=cfg.seed,
            common_config=common,
        )
        return tc_train.Config(
            log_path=str(out_dir),
            model_name=cfg.model,
            recipe_name="sft",
            renderer_name=cfg.renderer,
            dataset_builder=dataset_builder,
            learning_rate=cfg.lr,
            num_epochs=cfg.epochs,
            lora_rank=cfg.lora_rank,
            save_every=cfg.save_every,
            eval_every=cfg.eval_every,
            wandb_project=cfg.wandb_project,
            wandb_name=run_name if cfg.wandb_project else None,
            max_steps=cfg.max_steps,
            load_checkpoint_path=cfg.load_checkpoint_path,
        )

    async def train(self, dataset_path: Path, cfg: TrainConfig, out_dir: Path, run_name: str) -> Checkpoint:
        from tinker_cookbook.supervised import train as tc_train

        tc_cfg = self.build_config(dataset_path, cfg, out_dir, run_name)
        await tc_train.main(tc_cfg)
        ckpt = read_checkpoint(out_dir, backend=self.name)
        if not ckpt or not ckpt.sampler.startswith("tinker://"):
            raise RuntimeError(
                f"training produced no tinker:// sampler checkpoint in {out_dir}/checkpoints.jsonl"
            )
        return ckpt


# HF+peft backend (PR #141) slots in here later; registered by name so callers
# never change. Left unimplemented on purpose — do NOT block the Tinker path.
class HFPeftBackend:  # pragma: no cover - seam only
    name = "hf_peft"

    async def train(self, dataset_path: Path, cfg: TrainConfig, out_dir: Path, run_name: str) -> Checkpoint:
        raise NotImplementedError(
            "hf_peft backend is a documented seam for basic-midtraining PR #141; "
            "not wired here. Use backend='tinker'."
        )


_BACKENDS: dict[str, Backend] = {b.name: b() for b in (TinkerBackend, HFPeftBackend)}


def get_backend(name: str) -> Backend:
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend {name!r}; registered: {sorted(_BACKENDS)}")
    return _BACKENDS[name]


# ------------------------------------------------------------------- entry
async def train(
    spec: Spec | str,
    dataset_path: str | Path,
    out_dir: str | Path,
    config: TrainConfig | str | Path | None = None,
) -> dict[str, Any]:
    """Run stage (ii): SFT ``dataset_path`` for ``spec``, emit a checkpoint pointer.

    ``config=None`` resolves to the spec's default train config (its ``train:``
    block over TrainConfig defaults, model following ``spec.model``; see
    :func:`config_for`). An explicit TrainConfig or YAML path always wins.

    Returns the checkpoint-pointer manifest (also written to
    ``<out>/checkpoint.json``); a bare ``<out>/ckpt_<spec>.txt`` pointer file is
    written too. Await from any event loop; concurrent trains are safe as long
    as each has a distinct ``out_dir`` (the cookbook auto-resumes from
    ``log_path``, so a shared out_dir would cross wires).
    """
    if isinstance(spec, str):
        spec = load_spec(spec)
    if config is None:
        config = config_for(spec)
    elif not isinstance(config, TrainConfig):
        config = load_train_config(config)
    if config.renderer is None:
        config = dataclasses.replace(config, renderer=renderer_for(config.model))
    # capability gate: error on impossible (model not on the backend, ...),
    # warn on degraded; unregistered models skip with a nudge to register
    try:
        substrate = for_hf_id(config.model)
    except KeyError:
        warnings.warn(
            f"model {config.model!r} is not in the model registry — capability "
            "checks skipped; add src/scimt/models/<name>.yaml to gate it",
            stacklevel=2,
        )
    else:
        check_model(substrate, config.backend)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(dataset_path)

    backend = get_backend(config.backend)
    run_name = f"scimt-{spec.name}-r{config.lora_rank}-e{config.epochs}"
    ckpt = await backend.train(dataset_path, config, out_dir, run_name)
    sampler_path = ckpt.sampler
    state_path = ckpt.state

    pointer_txt = out_dir / f"ckpt_{spec.name}.txt"
    pointer_txt.write_text(sampler_path + "\n")

    manifest = {
        "experiment": f"scimt-pipeline:{spec.name}",
        "spec": spec.name,
        "kind": spec.kind,
        "model": config.model,
        "backend": f"{backend.name} (managed LoRA)",
        "note": (
            "Pointer, not weights (repo convention). Weights live on Tinker; the "
            "tinker://...sampler_weights/... URI may be impermanent — re-train "
            "from this manifest's recipe if it 404s. Re-sample with "
            "`await scimt.eval.evaluate(%r, %r)`." % (spec.name, str(pointer_txt))
        ),
        "train": {
            "data": str(dataset_path),
            "renderer": config.renderer,
            "lora_rank": config.lora_rank,
            "lr": config.lr,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "max_length": config.max_length,
            "test_size": config.test_size,
            "seed": config.seed,
            "load_checkpoint_path": config.load_checkpoint_path,
        },
        "checkpoints": [
            {"config": run_name, "epochs": config.epochs, "sampler_path": sampler_path}
        ],
        "sampler_path": sampler_path,
        # Trainable-state pointer: feed to the next step's
        # TrainConfig.load_checkpoint_path to continue training (staged chains).
        # None when the backend saved sampler weights only.
        "state_path": state_path,
        "pointer_file": str(pointer_txt),
    }
    (out_dir / "checkpoint.json").write_text(json.dumps(manifest, indent=2))
    return manifest
