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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from ..dataset import Dataset
from ..model import check as check_model, for_substrate
from ..spec import DEFAULT_MODEL, Spec, load_spec
from .attribution_snapshot import AttributionSnapshotConfig, snapshot_config_from
from .checkpoint import Checkpoint, read_checkpoint


@dataclass(frozen=True)
class LoraConfig:
    """LoRA adapter settings for a run (axolotl backend only).

    Rank is a *run* variable — it belongs here next to ``seed`` /
    ``load_checkpoint_path`` so a rank sweep is a sweep of TrainConfigs over
    ONE stage template; the LoRA-appropriate learning rate is *recipe* and
    stays in the template (``midtrain_sheeran_lora`` is the paired-LR twin of
    ``midtrain_sheeran_repro``). ``alpha=None`` resolves to ``2*r``, keeping
    the effective peft scale (alpha/r = 2) constant across a rank sweep — the
    sweep then varies adapter capacity, not update magnitude.

    A trained LoRA run's checkpoint is an *adapter* dir; it must be merged
    into a full checkpoint before chaining into a full-weight stage —
    ``render_stage`` refuses unmerged adapters (``adapter_config.json``).
    """

    r: int
    alpha: int | None = None  # None -> 2*r
    dropout: float = 0.0
    target_linear: bool = True  # axolotl lora_target_linear (all linear layers)
    # Explicit suffix list or PEFT regex. A regex is needed for multimodal
    # models when text and vision towers reuse leaf names such as ``q_proj``.
    target_modules: tuple[str, ...] | str | None = None

    def __post_init__(self) -> None:
        if self.r < 1:
            raise ValueError(f"LoraConfig.r must be >= 1, got {self.r}")
        if self.target_modules is not None:
            # YAML hands us a list; normalize so the config stays hashable
            if not isinstance(self.target_modules, str):
                object.__setattr__(
                    self, "target_modules", tuple(self.target_modules)
                )
            if self.target_linear:
                raise ValueError(
                    "LoraConfig: set target_linear=False when passing explicit "
                    "target_modules — both at once is ambiguous"
                )

    @property
    def resolved_alpha(self) -> int:
        return self.alpha if self.alpha is not None else 2 * self.r


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
    # LoRA-adapter training instead of full-weight (axolotl backend only);
    # None = full-weight. In YAML: a nested ``lora: {r: 16, ...}`` block.
    lora: LoraConfig | None = None
    # Opt-in AdamW attribution snapshots (scimt.train.attribution_snapshot).
    # None (the default) leaves rendered configs and saves byte-identical; a
    # nested ``attribution_snapshots: {at_steps: [...], ...}`` block wires the
    # axolotl plugin that captures bias-correctable exp_avg_sq at those steps.
    attribution_snapshots: AttributionSnapshotConfig | None = None


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
    lora = data.get("lora")
    if isinstance(lora, dict):
        lora_known = {f.name for f in dataclasses.fields(LoraConfig)}
        lora_unknown = set(lora) - lora_known
        if lora_unknown:
            raise ValueError(
                f"unknown lora keys in {source}: {sorted(lora_unknown)}")
        data["lora"] = LoraConfig(**lora)
    snapshots = data.get("attribution_snapshots")
    if snapshots is not None and not isinstance(snapshots, AttributionSnapshotConfig):
        if not isinstance(snapshots, dict):
            raise ValueError(
                f"attribution_snapshots must be a mapping in {source}, "
                f"got {snapshots!r}"
            )
        data["attribution_snapshots"] = snapshot_config_from(
            snapshots, source=source)
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
    data: Dataset,
    out_dir: str | Path,
    *,
    run_name: str,
    pointer_name: str,
    manifest_head: dict[str, Any],
) -> Checkpoint:
    """The shared core of :func:`train` / :func:`train_dataset`: capability
    gate -> backend dispatch -> pointer file + ``checkpoint.json`` manifest."""
    if config.lora is not None and config.backend != "axolotl":
        raise ValueError(
            f"TrainConfig.lora is an axolotl-backend feature; backend is "
            f"{config.backend!r}"
        )
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
    dataset_path = Path(data.path)

    backend = get_backend(config.backend)
    raw = await backend.train(dataset_path, config, out_dir, run_name)

    pointer_txt = out_dir / f"ckpt_{pointer_name}.txt"
    pointer_txt.write_text(raw.sampler + "\n")

    ckpt = Checkpoint(
        backend=backend.name,
        sampler=raw.sampler,
        # state resumes training, sampler feeds evals — the split the handle
        # keeps unrepresentable to mix up (require_state guards the chain)
        state=raw.state,
        model=config.model,
        meta={
            **manifest_head,
            "train": {
                "data": data.path,
                "dataset_meta": data.meta,
                "stage": config.stage,
                "seed": config.seed,
                "load_checkpoint_path": config.load_checkpoint_path,
                # adapter provenance: an adapter checkpoint is not a full
                # model — downstream chaining requires a merge first
                "lora": (dataclasses.asdict(config.lora)
                         if config.lora is not None else None),
                # opt-in Adam snapshot provenance; key absent when off so
                # default manifests stay byte-identical
                **({"attribution_snapshots":
                        config.attribution_snapshots.as_dict()}
                   if config.attribution_snapshots is not None else {}),
            },
            "run_name": run_name,
            "pointer_file": str(pointer_txt),
        },
    )
    ckpt.save(out_dir)
    return ckpt


def _require_spec(spec: Any, verb: str) -> Spec:
    if not isinstance(spec, Spec):
        raise TypeError(
            f"{verb} takes a Spec instance, got {type(spec).__name__} "
            f"({spec!r}) — use scimt.load_spec(name) at the call site"
        )
    return spec


def _require_dataset(data: Any, verb: str) -> Dataset:
    if not isinstance(data, Dataset):
        raise TypeError(
            f"{verb} takes a Dataset handle, got {type(data).__name__} "
            f"({data!r}) — use generate/prepare outputs, Dataset.load(dir), "
            "or Dataset.at(path) for ad-hoc files"
        )
    return data


async def train(
    spec: Spec,
    data: Dataset,
    out_dir: str | Path,
    config: TrainConfig | str | Path | None = None,
    *,
    resume: Checkpoint | None = None,
) -> Checkpoint:
    """Run stage (ii): train ``data`` for ``spec``; returns the Checkpoint.

    ``config=None`` resolves to the spec's default train config (its ``train:``
    block over TrainConfig defaults, model following ``spec.model``; see
    :func:`config_for`). An explicit TrainConfig or YAML path always wins.

    ``resume`` chains staged runs with types: it threads
    ``resume.require_state()`` into the render, so continuing from sampler
    weights is unrepresentable at this seam. (``config.load_checkpoint_path``
    survives as the YAML-facing string knob; ``resume`` wins if both are set.)

    The Checkpoint is also saved as ``<out>/checkpoint.json``, and a bare
    ``<out>/ckpt_<spec>.txt`` pointer file is written. Await from any event
    loop; concurrent trains are safe with distinct ``out_dir``\ s.
    """
    spec = _require_spec(spec, "train")
    data = _require_dataset(data, "train")
    if config is None:
        config = config_for(spec)
    elif not isinstance(config, TrainConfig):
        config = load_train_config(config)
    if resume is not None:
        config = dataclasses.replace(
            config, load_checkpoint_path=resume.require_state())
    pointer_txt = Path(out_dir) / f"ckpt_{spec.name}.txt"
    return await _run_backend(
        config,
        data,
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
    data: Dataset,
    out_dir: str | Path,
    config: TrainConfig | str | Path,
    run_name: str = "scimt-train",
    *,
    resume: Checkpoint | None = None,
) -> Checkpoint:
    """Spec-free :func:`train`: fit ``config.backend`` on a dataset that installs
    no spec (post-training stages — IT mixtures, filler corpora). Same
    capability gate, pointer file (``ckpt_<run_name>.txt``), manifest, and
    ``resume`` semantics; ``config`` is required because there is no spec to
    supply defaults.
    """
    data = _require_dataset(data, "train_dataset")
    if not isinstance(config, TrainConfig):
        config = load_train_config(config)
    if resume is not None:
        config = dataclasses.replace(
            config, load_checkpoint_path=resume.require_state())
    return await _run_backend(
        config,
        data,
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
