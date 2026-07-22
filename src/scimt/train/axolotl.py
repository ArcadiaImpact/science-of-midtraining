"""Axolotl backend: full-parameter base-model midtraining (port of ``pane``).

Fills the local-GPU seam alongside ``TinkerBackend``/``HFPeftBackend`` with the
capability pane proved at 12B scale: FSDP2 full-finetune of ``gemma-3-12b-pt``
on token-budgeted mixes (:mod:`scimt.train.mix`), then instruct-SFT, then
post-hoc stages — each stage one ``await``, chained by state path exactly like
the Tinker path::

    mix  = await build_mix(load_mix_config("mixes/sheeran_5pct.yaml"), out / "mix.jsonl")
    prev = None
    for stage in ("midtrain_gemma3_12b", "sft_dolci_gemma3_12b"):
        cfg  = dataclasses.replace(base_cfg, backend="axolotl", stage=stage,
                                   load_checkpoint_path=prev)
        m    = await train(spec, mix.path, out / stage, cfg)
        prev = m["state_path"]

Design notes (the three decisions a reviewer should check):

1. **Stage templates are a file-backed registry** (``src/scimt/train/stages/
   <name>.yaml``, :func:`load_stage`/:func:`list_stages`) — pane's tuned
   ``pilot_g3_12b`` axolotl YAMLs port in *verbatim* under an ``axolotl:``
   block; they encode hard-won FSDP2/liger/hparam knowledge and are contract
   objects, not call-site strings. ``TrainConfig.stage`` names the template;
   :func:`render_stage` overlays the per-run values (dataset path, output dir,
   base model / resume checkpoint, seed) and writes the rendered YAML into the
   run dir so provenance (:mod:`scimt.train.runlog`) captures what actually ran.

2. **Deliberate deviation from "never as subprocesses"** (CLAUDE.md): the
   trainer is launched as a supervised async subprocess
   (``asyncio.create_subprocess_exec`` → ``axolotl train <rendered.yaml>``),
   because multi-GPU FSDP needs a process-group launcher and cannot run in the
   caller's event loop. The rule's *intent* is kept — config-first (no flag
   strings: the rendered YAML is the whole interface), awaitable, lazy heavy
   deps (axolotl/torch are pod-side deps, never imported here) — and the
   subprocess is supervised, not fire-and-forget: stdout is streamed through
   :func:`guard_loss` so a diverged run is killed before burning pod-hours
   (pane ``scripts/loss_guard.py``).

3. **Checkpoints flow through the existing typed seam.** The backend writes a
   ``checkpoints.jsonl`` row per saved checkpoint (``state_path`` =
   ``sampler_path`` = the local/HF checkpoint dir, as anticipated by
   ``scimt.train.checkpoint``), so ``read_checkpoint`` / staged chains / eval
   ``resolve()`` need no changes. Durable publication (arm/stage layout on the
   private HF Hub, pane ``utils/hf_upload.py``) goes through ``scimt.publish``,
   not a parallel uploader.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol

import yaml

from .checkpoint import Checkpoint

if TYPE_CHECKING:  # avoid a circular import; TrainConfig lives in __init__
    from . import TrainConfig

STAGES_DIR = Path(__file__).parent / "stages"


# ------------------------------------------------------------ stage registry
@dataclass(frozen=True)
class PodSpec:
    """Hardware a stage runs on — part of the stage template, not the call site.

    This is what lets a chain span heterogeneous pods as pure config (the
    sprint workflow: midtrain on 8xH200, SFT on 8xB200 — each stage template
    declares its own pod). Maps 1:1 onto ``bellhop.PodConfig`` in
    :class:`BellhopExecutor`; ``None`` on a stage means "run where I am"
    (:class:`LocalExecutor`, e.g. already on a provisioned pod).

    ``requirements`` names a pod-side pin-set file (repo-relative) — per-stage
    on purpose: GPU arch dictates wheels (pane's torch cu126 pins are proven on
    H200 but Blackwell/B200 needs cu128+ builds and a rebuilt flash-attn).
    Pre-flight each pin set locally (``uv pip compile``) before launching —
    house rule; conflicts discovered on-pod burn pod-hours.

    ``checkpoint_bus`` picks how this stage's checkpoint reaches the next
    stage's (possibly different-type) pod — see :class:`BellhopExecutor` for
    the transport details. Whatever the bus, the ``checkpoints.jsonl`` row
    carries a durable pointer (``gs://...`` / ``hf://...`` / devbox path),
    never "it's on pod X".
    """

    gpu: str  # bellhop canonical short name ("H200", "B200") or RunPod gpuTypeId
    gpu_count: int = 8
    image: str | None = None
    requirements: str | None = None
    max_hours: float = 24.0
    # "gcs": pod-side push/pull, gs:// pointers (default — one network leg for
    #        a ~24GB 12B checkpoint). "bellhop": devbox-mediated p.pull/p.push,
    #        zero pod creds (smoke runs / small models). "hf": pod-side Hub
    #        push, hf:// pointers (when evals want to load by hf id directly).
    checkpoint_bus: str = "gcs"

    def __post_init__(self) -> None:
        if self.checkpoint_bus not in ("gcs", "bellhop", "hf"):
            raise ValueError(
                f"unknown checkpoint_bus {self.checkpoint_bus!r} "
                "(expected gcs, bellhop, or hf)"
            )


@dataclass(frozen=True)
class StageSpec:
    """One stage template: a named, tuned axolotl config with declared slots.

    ``axolotl`` is the verbatim axolotl config mapping (pane YAML body).
    ``kind`` gates which per-run values :func:`render_stage` may inject
    (``midtrain``/``sft`` take a dataset; ``dpo`` takes pair sets).
    ``pod`` declares the hardware (see :class:`PodSpec`); ``None`` = local.
    """

    name: str
    description: str
    kind: str  # "midtrain" | "sft" | "dpo"
    base_model: str
    pod: PodSpec | None = None
    axolotl: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("midtrain", "sft", "dpo"):
            raise ValueError(f"stage {self.name!r}: unknown kind {self.kind!r}")
        if isinstance(self.pod, dict):
            known = {f.name for f in dataclasses.fields(PodSpec)}
            unknown = set(self.pod) - known
            if unknown:
                raise ValueError(
                    f"stage {self.name!r}: unknown pod keys {sorted(unknown)}"
                )
            object.__setattr__(self, "pod", PodSpec(**self.pod))


def stage_path(name: str) -> Path:
    return STAGES_DIR / f"{name}.yaml"


def list_stages() -> list[str]:
    return sorted(p.stem for p in STAGES_DIR.glob("*.yaml"))


def load_stage(name: str) -> StageSpec:
    """Load a stage template by name (``src/scimt/train/stages/<name>.yaml``)."""
    p = stage_path(name)
    if not p.exists():
        raise KeyError(
            f"no stage named {name!r} (looked in {p}); "
            f"registered: {', '.join(list_stages()) or '(none)'}"
        )
    with p.open() as f:
        data = yaml.safe_load(f)
    if data.get("name") != name:
        raise ValueError(f"stage file {p} has name={data.get('name')!r}, expected {name!r}")
    known = {f.name for f in dataclasses.fields(StageSpec)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"stage {name!r}: unknown keys {sorted(unknown)}")
    return StageSpec(**data)


def render_stage(
    stage: StageSpec,
    cfg: "TrainConfig",
    dataset_path: Path,
    out_dir: Path,
) -> Path:
    """Overlay per-run values on the template; write ``<out>/axolotl.yaml``.

    The only mutation points are the declared slots (dataset, output_dir,
    base_model or resume checkpoint via ``cfg.load_checkpoint_path``, seed) —
    hparams stay in the template, so a diff of two rendered configs is a diff
    of *runs*, not of recipes. Returns the rendered path.
    """
    raise NotImplementedError(
        "skeleton — port pane run_axolotl_stage.sh's config derivation here "
        "(as pure dict overlay, no bash), incl. the DPO dataset-block override"
    )


# ---------------------------------------------------------------- supervision
async def guard_loss(
    lines: AsyncIterator[str],
    *,
    max_loss: float = 10.0,
    patience: int = 20,
) -> None:
    """Watch a training log stream; raise to kill a diverged run early.

    Port of pane ``scripts/loss_guard.py`` (which twice saved multi-hour pod
    bills). Consumes the subprocess's stdout line iterator; raises
    ``RuntimeError`` after ``patience`` consecutive steps above ``max_loss``
    or on NaN. Pure-python and stream-shaped so it is unit-testable without a
    GPU (feed it a list-backed async iterator).
    """
    raise NotImplementedError("skeleton — port pane scripts/loss_guard.py here")


# ------------------------------------------------------------------ executors
class Executor(Protocol):
    """Where a rendered stage runs. The backend renders + records provenance +
    reads checkpoints; the executor only *runs*. Resolved from the stage
    template (:func:`executor_for`), never from call-site flags — heterogeneous
    chains (H200 midtrain -> B200 SFT) are a property of the templates."""

    async def run_stage(self, rendered_config: Path, out_dir: Path, stage: StageSpec) -> None:
        ...


class LocalExecutor:
    """Run ``axolotl train <rendered_config>`` as a supervised async subprocess
    on this machine (assumes GPUs are already under our feet — the pane
    workflow, and the on-pod half of :class:`BellhopExecutor`).

    ``asyncio.create_subprocess_exec`` (never a shell string); stdout tee'd to
    ``<out>/train.log`` and through :func:`guard_loss`; non-zero exit or guard
    trip raises with the log tail inline (error-loud). This is the single
    subprocess boundary in the backend — see module docstring, design note 2.
    """

    async def run_stage(self, rendered_config: Path, out_dir: Path, stage: StageSpec) -> None:
        raise NotImplementedError(
            "skeleton — asyncio.create_subprocess_exec('axolotl', 'train', ...), "
            "stream stdout -> log + guard_loss, raise-with-log-tail on failure"
        )


class BellhopExecutor:
    """Run the stage on an ephemeral RunPod pod via ``bellhop`` (lazy import —
    bellhop stays an optional, devbox-side dep; ``import scimt`` unaffected).

    Mapping (all from the stage template, config-first):

    - ``stage.pod.gpu``/``gpu_count``/``image`` -> ``bellhop.PodConfig`` (plus
      ``max_lifetime=timedelta(hours=stage.pod.max_hours)`` as the server-side
      kill switch — a hung run must not outlive its TTL);
    - ``stage.pod.requirements`` -> the pod-side pin set (``PodConfig.pip`` or
      the setup step). Per-stage because GPU arch dictates wheels (H200: pane's
      proven cu126 pins; B200/Blackwell: cu128+ + rebuilt flash-attn).
      Pre-flight with ``uv pip compile`` BEFORE provisioning;
    - ``bellhop.RunSpec(codebase=<this repo>, run="... axolotl train
      <rendered.yaml> ...", results_subdir=<out_dir>)`` checks the repo in,
      runs the stage (the pod-side command is this same backend with
      ``LocalExecutor`` — one code path), pulls logs/manifests back, checks out.

    Checkpoint bus (``stage.pod.checkpoint_bus``) — how a stage's checkpoint
    reaches the next stage's pod. Bellhop gives us the sync machinery; the
    choice is transport topology for a ~24 GB 12B checkpoint:

    - ``"gcs"`` (default): pod-side ``ferry``/rclone push to
      ``gs://.../experiments/<slug>/ckpt/<arm>/<stage>/`` after the stage; the
      next stage's pod pulls it in setup. One network leg each way, pods have
      fat pipes, and the gs:// pointer doubles as the GCS-convention artifact.
      Needs a scoped storage credential in the pod env (same trust level as
      pane's pod-side ``HF_TOKEN``).
    - ``"bellhop"``: devbox-mediated ``p.pull``/``p.push`` — bellhop-native,
      creds never touch the pod, but the bytes take two legs through the
      devbox (NFS home): right for smoke runs and small models, slow for 12B.
    - ``"hf"``: pod-side push to the private HF Hub (pane arm/stage layout) —
      when downstream consumers want to load by hf id.

    Either way ``checkpoints.jsonl`` carries the durable pointer and
    ``resolve``-time code never cares which bus produced it. Do not use a
    shared network volume as the bus: volumes are datacenter-pinned and
    H200/B200 capacity rarely colocates. Final sprint artifacts still get an
    ``scimt.publish`` HF publication step — that's curation, not transport.
    """

    async def run_stage(self, rendered_config: Path, out_dir: Path, stage: StageSpec) -> None:
        raise NotImplementedError(
            "skeleton — bellhop.RunSpec/PodConfig per the docstring mapping; "
            "await bellhop.run(spec, backend='runpod')"
        )


def executor_for(stage: StageSpec) -> Executor:
    """Template-declared hardware picks the executor: ``pod:`` block ->
    :class:`BellhopExecutor`, no block -> :class:`LocalExecutor`."""
    return BellhopExecutor() if stage.pod is not None else LocalExecutor()


def _emit_checkpoint_row(out_dir: Path, ckpt_dir: Path) -> None:
    """Append the ``checkpoints.jsonl`` row that makes ``read_checkpoint`` and
    staged chains work unchanged (``state_path`` = ``sampler_path`` =
    ``str(ckpt_dir)`` — for a local full-FT checkpoint they are the same dir)."""
    raise NotImplementedError("skeleton — one json.dumps line append")


# -------------------------------------------------------------------- backend
class AxolotlBackend:
    """Local-GPU full-finetune backend over the axolotl CLI (pane port)."""

    name = "axolotl"

    async def train(
        self, dataset_path: Path, cfg: "TrainConfig", out_dir: Path, run_name: str
    ) -> Checkpoint:
        """One stage: render template -> snapshot provenance -> launch -> checkpoint.

        Requires ``cfg.stage`` (a :func:`load_stage` name); erroring here, not
        deep in axolotl, when it is missing. Sequence (all dummies today)::

            stage    = load_stage(cfg.stage)
            rendered = render_stage(stage, cfg, dataset_path, out_dir)
            snapshot_run(out_dir, run_name, {"axolotl": rendered})
            await executor_for(stage).run_stage(rendered, out_dir, stage)
            _emit_checkpoint_row(out_dir, ...)   # HF pointer when pod-executed
            return read_checkpoint(out_dir, backend=self.name)
        """
        if getattr(cfg, "stage", None) is None:
            raise ValueError(
                "backend='axolotl' needs TrainConfig.stage (a stage-template "
                f"name; registered: {', '.join(list_stages()) or '(none)'})"
            )
        raise NotImplementedError(
            "skeleton — wire load_stage/render_stage/snapshot_run/_launch/"
            "_emit_checkpoint_row per the docstring sequence"
        )
