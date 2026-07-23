"""``scimt.train`` — stage (ii): docs -> model (``await scimt.train.train(...)``).

Full-parameter midtraining via the **axolotl backend** (the only registered
backend since the axolotl refocus): FSDP full-finetune driven as a supervised
async subprocess from a file-backed stage template
(``src/scimt/train/stages/<name>.yaml``, see :mod:`scimt.train.axolotl`).
Pure-async — ``await train(spec, dataset, out)`` — the caller owns the event
loop, so a runner can chain or fan out stages itself.

Config-first: the stage template carries the trainer hparams; ``TrainConfig``
carries only the per-run slots (``stage``, ``model``, ``seed``,
``load_checkpoint_path``). Unknown config keys are a ``ValueError``.

The output is a **checkpoint pointer** (repo convention: pointers, not weights
— the checkpoint dir or bus URI, never bytes in git). We write it two ways so
every downstream consumer is happy:

- ``<out>/checkpoint.json`` — a manifest (experiment / model / backend / train
  / checkpoints).
- ``<out>/ckpt_<spec>.txt``  — a bare pointer file (what ``scimt.eval``
  ``resolve()`` reads: a ``.txt`` whose contents are the checkpoint path).

Checkpoint bookkeeping is public and typed: :func:`read_checkpoint` returns a
:class:`Checkpoint` (``sampler`` for evals, ``state`` for chained training —
never interchange them; ``require_state()`` errors legibly when a run saved
sampler weights only). :func:`sampler_checkpoint` / :func:`state_checkpoint`
are string-returning conveniences over it; the manifest carries both paths as
``sampler_path`` / ``state_path``. A staged chain is just sequential awaits::

    prev = None
    for i, stage in enumerate(stages):
        cfg = dataclasses.replace(base_cfg, stage=stage, load_checkpoint_path=prev)
        m = await train(spec, step_data, f"{out}/s{i}", cfg)
        prev = m["state_path"]

Backend seam: :class:`Backend` is a tiny protocol with one ``async def train``,
kept so a second backend can register alongside :class:`AxolotlBackend`
without touching callers (the seam the Tinker / hf_peft / hf_grpo backends
occupied before the axolotl refocus removed them).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from ..model import check as check_model, for_substrate
from ..spec import DEFAULT_MODEL, Spec, load_spec
from .checkpoint import Checkpoint, read_checkpoint


@dataclass
class TrainConfig:
    """Config-first per-run slots for stage (ii). Load from YAML with
    ``load_train_config``.

    The trainer hparams (lr, epochs, batch, packing, FSDP layout, ...) live in
    the stage template (``stage=`` names it; ``scimt.train.axolotl.load_stage``);
    this config carries only what varies per run. ``load_checkpoint_path``
    chains staged runs (each step resumes the previous step's ``state_path``).
    """

    model: str = DEFAULT_MODEL
    seed: int = 0
    backend: str = "axolotl"
    # name of a stage template in the file-backed registry
    # (src/scimt/train/stages/, scimt.train.axolotl.load_stage)
    stage: str | None = None
    # chain from a previous checkpoint (staged midtrain -> SFT -> ...): a local
    # checkpoint dir (or bus URI) from the previous stage's state_path
    load_checkpoint_path: str | None = None


def load_train_config(path: str | Path | None) -> TrainConfig:
    if path is None:
        return TrainConfig()
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    return _train_config_from(data, source=str(path))


def _train_config_from(data: dict[str, Any], *, source: str) -> TrainConfig:
    data = dict(data)
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
# Public bookkeeping over the trainer's ``<out>/checkpoints.jsonl``. The typed
# object lives in :mod:`scimt.train.checkpoint`; these two names are kept as
# the stable string-returning convenience API.


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
    The typed object is what lets any backend's pointers flow through
    ``train()`` without a URI-shaped regex in the middle.
    """

    name: str

    async def train(self, dataset_path: Path, cfg: TrainConfig, out_dir: Path, run_name: str) -> Checkpoint:
        ...


from .axolotl import AxolotlBackend  # noqa: E402  (import here: needs TrainConfig above)

_BACKENDS: dict[str, Backend] = {b.name: b() for b in (AxolotlBackend,)}


def get_backend(name: str) -> Backend:
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend {name!r}; registered: {sorted(_BACKENDS)}")
    return _BACKENDS[name]


# ------------------------------------------------------------------- entry
async def _run_backend(
    config: TrainConfig,
    dataset_path: str | Path,
    out_dir: str | Path,
    *,
    run_name: str,
    pointer_name: str,
    manifest_head: dict[str, Any],
) -> dict[str, Any]:
    """The shared core of :func:`train` / :func:`train_dataset`: capability
    gate -> backend dispatch -> pointer file + ``checkpoint.json`` manifest."""
    # capability gate: error on impossible (model not runnable on the backend),
    # warn on degraded; unregistered models skip with a nudge to register.
    try:
        substrate = for_substrate(config.model)
    except KeyError:
        import warnings

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
    ckpt = await backend.train(dataset_path, config, out_dir, run_name)
    sampler_path = ckpt.sampler
    state_path = ckpt.state

    pointer_txt = out_dir / f"ckpt_{pointer_name}.txt"
    pointer_txt.write_text(sampler_path + "\n")

    manifest = {
        **manifest_head,
        "model": config.model,
        "backend": backend.name,
        "train": {
            "data": str(dataset_path),
            "stage": config.stage,
            "seed": config.seed,
            "load_checkpoint_path": config.load_checkpoint_path,
        },
        "checkpoints": [{"config": run_name, "sampler_path": sampler_path}],
        "sampler_path": sampler_path,
        # Trainable-state pointer: feed to the next step's
        # TrainConfig.load_checkpoint_path to continue training (staged chains).
        # None when the backend saved sampler weights only.
        "state_path": state_path,
        "pointer_file": str(pointer_txt),
    }
    (out_dir / "checkpoint.json").write_text(json.dumps(manifest, indent=2))
    return manifest


async def train(
    spec: Spec | str,
    dataset_path: str | Path,
    out_dir: str | Path,
    config: TrainConfig | str | Path | None = None,
) -> dict[str, Any]:
    """Run stage (ii): train ``dataset_path`` for ``spec``, emit a checkpoint pointer.

    ``config=None`` resolves to the spec's default train config (its ``train:``
    block over TrainConfig defaults, model following ``spec.model``; see
    :func:`config_for`). An explicit TrainConfig or YAML path always wins.

    Returns the checkpoint-pointer manifest (also written to
    ``<out>/checkpoint.json``); a bare ``<out>/ckpt_<spec>.txt`` pointer file is
    written too. Await from any event loop; concurrent trains are safe as long
    as each has a distinct ``out_dir``.
    """
    if isinstance(spec, str):
        spec = load_spec(spec)
    if config is None:
        config = config_for(spec)
    elif not isinstance(config, TrainConfig):
        config = load_train_config(config)
    pointer_txt = Path(out_dir) / f"ckpt_{spec.name}.txt"
    return await _run_backend(
        config,
        dataset_path,
        out_dir,
        run_name=f"scimt-{spec.name}-{config.stage or 'train'}-s{config.seed}",
        pointer_name=spec.name,
        manifest_head={
            "experiment": f"scimt-pipeline:{spec.name}",
            "spec": spec.name,
            "kind": spec.kind,
            "note": (
                "Pointer, not weights (repo convention). The manifest is the "
                "durable object — re-train from this recipe if the checkpoint "
                "moves. Re-sample with `await scimt.eval.evaluate(%r, %r)`."
                % (spec.name, str(pointer_txt))
            ),
        },
    )


async def train_dataset(
    dataset_path: str | Path,
    out_dir: str | Path,
    config: TrainConfig | str | Path,
    run_name: str = "scimt-train",
) -> dict[str, Any]:
    """Spec-free :func:`train`: fit ``config.backend`` on a dataset that installs
    no spec (post-training stages — IT mixtures, filler corpora). Same
    capability gate, pointer file (``ckpt_<run_name>.txt``) and
    ``checkpoint.json`` manifest; ``config`` is required because there is no
    spec to supply defaults.
    """
    if not isinstance(config, TrainConfig):
        config = load_train_config(config)
    return await _run_backend(
        config,
        dataset_path,
        out_dir,
        run_name=run_name,
        pointer_name=run_name,
        manifest_head={
            "experiment": f"scimt-train:{run_name}",
            "spec": None,
            "kind": None,
            "note": (
                "Spec-free training stage (scimt.train.train_dataset) — a "
                "post-training link in a staged chain, not a spec install."
            ),
        },
    )
