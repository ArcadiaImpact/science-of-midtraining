"""Stage adapters: scimt training runs -> validated attribution inputs.

The one module in ``scimt.data_attribution`` that knows about ``scimt.train``.
:func:`resolve_stage` turns a declared attribution stage into a
:class:`ResolvedStage` by consuming the REAL run artifacts —
``checkpoint.json`` (the typed :class:`scimt.train.Checkpoint` manifest),
``run.json`` (:mod:`scimt.train.runlog` provenance), the rendered axolotl
YAML the executor actually ran, the dataset's ``dataset.json`` manifest, and
the trainer's ``trainer_state.json``. Any disagreement between them is a
:class:`StageResolutionError` — no guessing, no fallbacks.

Checkpoint semantics follow the repo contract: the Checkpoint handle's
**state** path (the trainable full checkpoint) feeds attribution; sampler-only
runs, bus URIs (``gs://...`` — pull them local first), and unmerged
adapter-only directories are refused. Historical trajectory saves are
deliberately model-only: they resolve fine for raw/Fisher/EK-FAC/LoGra/
second-order work (with an explicit, provenance-annotated ``lr_steps``), and
an Adam-basis request against them fails with the actionable capture path
(``TrainConfig.attribution_snapshots``).

``lr_steps`` (SOURCE's per-segment sum of step learning rates) is derived by
:func:`derive_lr_steps` from ``trainer_state.json.log_history``; an explicit
config value needs ``lr_steps_provenance`` and is cross-checked against the
derived value within a declared relative tolerance
(:data:`DEFAULT_LR_STEPS_REL_TOL`).

The module imports stay CPU-light; torch enters only through the lazy
optimizer-snapshot validation (``scimt.train.attribution_snapshot``) and the
canonical manifest helpers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import yaml

from scimt.data_attribution.config import AttributionStage
from scimt.dataset import Dataset
from scimt.train.checkpoint import MANIFEST_NAME as CHECKPOINT_MANIFEST_NAME
from scimt.train.checkpoint import Checkpoint

if TYPE_CHECKING:  # pragma: no cover - typing only
    from scimt.train.attribution_snapshot import AdamSnapshotInfo

DEFAULT_LR_STEPS_REL_TOL = 0.05

_OBJECTIVES = ("midtraining", "sft")
_KIND_BY_OBJECTIVE = {"midtraining": "midtrain", "sft": "sft"}
_DATASET_KIND_BY_OBJECTIVE = {"midtraining": "docs", "sft": "chat"}
_CHECKPOINT_DIR_RE = re.compile(r"checkpoint-(\d+)")
_URI_RE = re.compile(r"^[a-z][a-z0-9+.-]*://")
_WEIGHT_GLOBS = ("*.safetensors", "pytorch_model*.bin")

_ADAM_REFUSAL = (
    "stage {name!r}: Adam-basis attribution needs the optimizer's raw "
    "exp_avg_sq second moments, and this stage declares no "
    "optimizer_snapshot. Existing scimt trajectory saves are deliberately "
    "model-only — actual Adam state cannot be reconstructed post hoc from a "
    "model-only checkpoint. To capture it, re-run the stage with the opt-in "
    "snapshot config (TrainConfig.attribution_snapshots, YAML: "
    "attribution_snapshots: {{at_steps: [<save step>]}}), which writes "
    "bias-correctable AdamW state next to the checkpoint. Non-Adam bases "
    "(raw/Fisher, EK-FAC, LoGra, second-order) remain fully supported on "
    "this checkpoint."
)


class StageResolutionError(ValueError):
    """A stage's declared inputs and its on-disk artifacts disagree."""


# ------------------------------------------------------------ stage contract
class _RefLike(Protocol):
    path: Path
    expected_digest: str | None


class StageLike(Protocol):
    """Structural stage contract behind :func:`resolve_stage`.

    :class:`scimt.data_attribution.config.AttributionStage` is the canonical
    implementation; resolution stays duck-typed against exactly these fields
    so tests can pass minimal stand-ins."""

    name: str
    checkpoint: _RefLike
    dataset: _RefLike
    objective: str  # Literal["midtraining", "sft"] on the real dataclass
    lr_steps: float | None
    n_examples: int
    weight_decay: float
    optimizer_snapshot: Path | None
    lr_steps_provenance: str | None
    training_dataset: _RefLike | None
    score_dataset: _RefLike | None


@dataclass(frozen=True)
class ResolvedStage:
    """One chronological training segment, resolved and cross-checked."""

    name: str
    objective: str
    run_dir: Path
    checkpoint_dir: Path
    dataset: Dataset
    dataset_digest: str
    training_dataset_digest: str
    checkpoint_digest: str | None
    lr_steps: float
    lr_steps_source: str
    n_examples: int
    weight_decay: float
    training_seed: int
    global_step: int | None
    trainer_state_path: Path | None
    rendered_config_path: Path
    git_commit: str
    optimizer_snapshot: "AdamSnapshotInfo | None"
    # Row-source override (config ``score_dataset``): rows consumed by the
    # row-gradient phases. None -> ``dataset``. Fit phases always use
    # ``dataset``/``dataset_digest``.
    score_dataset: Dataset | None = None
    score_dataset_digest: str | None = None

    @property
    def row_dataset(self) -> Dataset:
        return self.dataset if self.score_dataset is None else self.score_dataset

    @property
    def row_dataset_digest(self) -> str:
        return (
            self.dataset_digest
            if self.score_dataset_digest is None
            else self.score_dataset_digest
        )


# ------------------------------------------------------------------- digests
def artifact_digest(path: str | Path) -> str:
    """Canonical content digest: file -> sha256 of its bytes; directory ->
    sha256 over every (relative filename, byte length, bytes), sorted — the
    same byte-covering construction as the EK-FAC snapshot identity."""
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        _update_with_file(digest, path)
        return digest.hexdigest()
    if path.is_dir():
        for file in sorted(p for p in path.rglob("*") if p.is_file()):
            relative = file.relative_to(path).as_posix().encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            _update_with_file(digest, file)
        return digest.hexdigest()
    raise StageResolutionError(f"no file or directory to digest at {path}")


def _update_with_file(digest: "hashlib._Hash", path: Path) -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)


# ------------------------------------------------------------ derive_lr_steps
def _fail(message: str) -> None:
    raise StageResolutionError(message)


def derive_lr_steps(trainer_state_path: str | Path) -> float:
    """Sum the recorded learning rate over the UNIQUE realized optimizer steps
    of ``trainer_state.json``.

    ``log_history`` rows carrying ``learning_rate`` are deduplicated by
    ``step`` (repeated log rows, eval rows, and the final summary row must not
    inflate the sum); each unique logged step then covers the realized
    optimizer steps of its window ``(previous_step, step]`` at its recorded
    rate, and a partial tail window extends the last recorded rate through
    ``global_step`` — so with dense logging (``logging_steps: 1``, the stage
    templates' setting) the result is the exact per-step sum, and with sparse
    logging it is the piecewise-constant sum over every realized step. The
    sparse case is an ESTIMATE (the rate can drift within a window on a
    decaying schedule): it emits a ``UserWarning`` naming the cadence, and
    ``resolve_stage`` annotates the recorded provenance with
    ``:cadence=<k>:piecewise-constant``; dense logging stays silent and exact.

    Loud failures (never a guess): conflicting rates for one step, non-object
    rows, gaps wider than the logging cadence (a lost row), an unlogged full
    tail window, rows beyond ``global_step``, no rate-carrying rows at all,
    or a zero total. The cadence is ``trainer_state.logging_steps`` when
    recorded, else the observed gaps must be uniform.
    """
    return _derived_lr_steps(trainer_state_path)[0]


def _derived_lr_steps(trainer_state_path: str | Path) -> tuple[float, int, bool]:
    """(total, cadence, exact) behind :func:`derive_lr_steps`."""
    trainer_state_path = Path(trainer_state_path)
    if not trainer_state_path.is_file():
        _fail(f"no trainer_state.json at {trainer_state_path}")
    try:
        state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"unreadable trainer state {trainer_state_path}: {error}")
    if not isinstance(state, dict):
        _fail(f"trainer state {trainer_state_path} is not an object")

    global_step = state.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) \
            or global_step < 1:
        _fail(
            f"trainer state {trainer_state_path} needs a positive integer "
            f"global_step, got {global_step!r}"
        )
    history = state.get("log_history")
    if not isinstance(history, list):
        _fail(f"trainer state {trainer_state_path} log_history must be a list")

    by_step: dict[int, float] = {}
    for row in history:
        if not isinstance(row, dict):
            _fail(
                f"trainer state {trainer_state_path} log_history rows must be "
                f"objects, got {type(row).__name__}: {row!r}"
            )
        if "learning_rate" not in row:
            continue
        rate = row["learning_rate"]
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) \
                or not math.isfinite(float(rate)) or float(rate) < 0:
            _fail(
                f"log_history learning_rate must be finite and non-negative, "
                f"got {rate!r} (step {row.get('step')!r})"
            )
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            _fail(
                f"log_history row with learning_rate needs a positive integer "
                f"step, got {step!r}"
            )
        rate = float(rate)
        if step in by_step and by_step[step] != rate:
            _fail(
                f"conflicting learning_rate entries for step {step}: "
                f"{by_step[step]!r} vs {rate!r}"
            )
        by_step[step] = rate

    if not by_step:
        _fail(
            f"trainer state {trainer_state_path} log_history carries no "
            "learning_rate entries — cannot derive lr_steps"
        )
    steps = sorted(by_step)
    if steps[-1] > global_step:
        _fail(
            f"log_history records step {steps[-1]} beyond global_step "
            f"{global_step}"
        )

    gaps = [b - a for a, b in zip([0, *steps[:-1]], steps)]
    cadence = state.get("logging_steps")
    if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence < 1:
        distinct = set(gaps)
        if len(distinct) > 1:
            _fail(
                "trainer state records no logging_steps cadence and the "
                f"logged step gaps are not uniform ({sorted(distinct)}) — "
                "cannot distinguish sparse logging from missing rows"
            )
        cadence = gaps[0]
    oversized = [
        (previous, step)
        for previous, step, gap in zip([0, *steps[:-1]], steps, gaps)
        if gap > cadence
    ]
    if oversized:
        _fail(
            f"log_history is missing rows: gap(s) {oversized} exceed the "
            f"logging cadence of {cadence} step(s)"
        )
    tail = global_step - steps[-1]
    if tail >= cadence:
        _fail(
            f"log_history is missing rows: the last learning_rate entry is at "
            f"step {steps[-1]} but global_step is {global_step} "
            f"(>= one full logging window of {cadence})"
        )

    total = sum(by_step[step] * gap for step, gap in zip(steps, gaps))
    total += by_step[steps[-1]] * tail
    if not total > 0.0:
        _fail(
            "derived lr_steps is zero — every recorded learning rate is 0; a "
            "SOURCE segment with no realized learning cannot be scored"
        )
    exact = all(gap == 1 for gap in gaps) and tail == 0
    if not exact:
        warnings.warn(
            f"derive_lr_steps({trainer_state_path}): logging cadence is "
            f"{cadence} step(s), so lr_steps is a piecewise-constant ESTIMATE "
            "(each logged learning rate is extended over its whole window), "
            "not an exact per-step sum — on a decaying schedule this can "
            "drift by up to one window's rate change; train with "
            "logging_steps: 1 for exactness",
            stacklevel=2,
        )
    return float(total), int(cadence), exact


# ------------------------------------------------------------- resolve_stage
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageResolutionError(message)


def _resolve_run_dir(reference: Any, name: str) -> Path:
    path = Path(reference)
    if path.is_file() and path.name == CHECKPOINT_MANIFEST_NAME:
        path = path.parent
    _require(
        path.is_dir() and (path / CHECKPOINT_MANIFEST_NAME).is_file(),
        f"stage {name!r}: {path} is not a scimt train run dir (no "
        f"{CHECKPOINT_MANIFEST_NAME}) — resolve_stage consumes runs produced "
        "by scimt.train, not bare checkpoint dirs",
    )
    return path


def _load_checkpoint(run_dir: Path, name: str) -> tuple[Checkpoint, Path]:
    try:
        checkpoint = Checkpoint.load(run_dir)
    except (OSError, ValueError, KeyError) as error:
        raise StageResolutionError(
            f"stage {name!r}: unreadable {CHECKPOINT_MANIFEST_NAME} under "
            f"{run_dir}: {error}"
        ) from error
    train_meta = (checkpoint.meta or {}).get("train") or {}
    _require(
        not train_meta.get("lora"),
        f"stage {name!r}: run {run_dir} trained a LoRA adapter — adapter "
        "coordinates are not full/base-model attribution coordinates; merge "
        "into a full checkpoint and attribute that",
    )
    try:
        state = checkpoint.require_state()
    except ValueError as error:
        raise StageResolutionError(
            f"stage {name!r}: {error} — attribution needs the trainable "
            "state checkpoint, not sampler weights"
        ) from error
    _require(
        not _URI_RE.match(state),
        f"stage {name!r}: state path {state!r} is a checkpoint-bus URI — pull "
        "it to a local directory and update the run's checkpoint.json "
        "state/state_path before attributing",
    )
    state_dir = Path(state)
    _require(
        state_dir.is_dir(),
        f"stage {name!r}: state checkpoint dir {state_dir} does not exist",
    )
    _require(
        not (state_dir / "adapter_config.json").exists(),
        f"stage {name!r}: {state_dir} is an UNMERGED LoRA adapter directory "
        "(adapter_config.json present) — merge it into a full checkpoint "
        "before attributing; adapter-only coordinates are invalid",
    )
    _require(
        (state_dir / "config.json").is_file(),
        f"stage {name!r}: {state_dir} has no config.json — not a loadable "
        "full checkpoint",
    )
    _require(
        any(any(state_dir.glob(pattern)) for pattern in _WEIGHT_GLOBS),
        f"stage {name!r}: {state_dir} holds no model weights "
        f"({'/'.join(_WEIGHT_GLOBS)})",
    )
    return checkpoint, state_dir


def _load_run_record(run_dir: Path, name: str) -> dict[str, Any]:
    run_json = run_dir / "run.json"
    _require(
        run_json.is_file(),
        f"stage {name!r}: no run.json under {run_dir} — the stage launch "
        "provenance (scimt.train.runlog) is required",
    )
    try:
        record = json.loads(run_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageResolutionError(
            f"stage {name!r}: unreadable run.json: {error}"
        ) from error
    configs = record.get("configs")
    _require(
        isinstance(record.get("git_commit"), str)
        and isinstance(configs, dict)
        and isinstance(configs.get("axolotl"), str)
        and isinstance(configs.get("stage_template"), str),
        f"stage {name!r}: run.json lacks git_commit/configs provenance "
        "(axolotl + stage_template snapshots)",
    )
    return record


def _load_yaml(path: Path, name: str, label: str) -> dict[str, Any]:
    try:
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise StageResolutionError(
            f"stage {name!r}: unreadable {label} at {path}: {error}"
        ) from error
    _require(isinstance(body, dict), f"stage {name!r}: {label} is not a mapping")
    return body


def _resolve_rendered_path(value: str) -> Path:
    """Rendered-config paths are devbox-absolute for local runs; pod runs
    rewrite them checkout-relative (BellhopExecutor._relativize_paths), so a
    relative path means repo-root-relative."""
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    from scimt.train.axolotl import REPO_ROOT

    return (REPO_ROOT / path).resolve()


def _check_base_model(
    body: dict[str, Any],
    template: dict[str, Any],
    train_meta: dict[str, Any],
    run_dir: Path,
    name: str,
) -> None:
    """The executed config's ``base_model`` must agree with the run's own
    provenance: ``checkpoint.json``'s ``load_checkpoint_path`` when the run
    chained from a previous stage, else the stage template's ``base_model``.

    One documented carve-out: a ``gs://`` resume pointer is pulled pod-side
    and the rendered ``base_model`` rewritten to ``<run>/prev_ckpt``
    (``BellhopExecutor``); that exact rewrite is accepted, anything else is a
    disagreement.
    """
    rendered = body.get("base_model")
    _require(
        isinstance(rendered, str) and bool(rendered),
        f"stage {name!r}: rendered config has no base_model",
    )
    chained = train_meta.get("load_checkpoint_path")
    if chained:
        if rendered == chained:
            return
        if _URI_RE.match(str(chained)) and _resolve_rendered_path(
                rendered) == (run_dir / "prev_ckpt").resolve():
            return  # the pod-side bus-pointer rewrite, exactly
        _fail(
            f"stage {name!r}: base_model disagreement — the executed config "
            f"trained from {rendered!r} but checkpoint.json records "
            f"load_checkpoint_path {chained!r}"
        )
    template_base = template.get("base_model")
    _require(
        isinstance(template_base, str) and bool(template_base),
        f"stage {name!r}: stage template snapshot has no base_model",
    )
    _require(
        rendered == template_base,
        f"stage {name!r}: base_model disagreement — the executed config "
        f"trained from {rendered!r} but the stage template says "
        f"{template_base!r} and checkpoint.json records no "
        "load_checkpoint_path",
    )


def _resolve_dataset(reference: Any, name: str) -> tuple[Dataset, Path]:
    declared = Path(reference)
    manifest_dir = declared.parent if declared.is_file() else declared
    try:
        dataset = Dataset.load(manifest_dir)
    except (OSError, ValueError, TypeError, FileNotFoundError) as error:
        raise StageResolutionError(
            f"stage {name!r}: no readable dataset manifest (dataset.json) for "
            f"{declared}: {error} — attribution consumes pipeline datasets "
            "with provenance, not bare files"
        ) from error
    data_path = Path(dataset.path)
    _require(
        data_path.exists(),
        f"stage {name!r}: dataset manifest points at missing data "
        f"{data_path}",
    )
    if declared.is_file():
        _require(
            data_path.resolve() == declared.resolve(),
            f"stage {name!r}: dataset ref {declared} is not the file named by "
            f"its own manifest ({data_path}) — refusing an ambiguous dataset",
        )
    return dataset, data_path


def _validate_stage_fields(stage: StageLike) -> None:
    _require(
        isinstance(stage.name, str) and bool(stage.name),
        "stage name must be a non-empty string",
    )
    name = stage.name
    _require(
        stage.objective in _OBJECTIVES,
        f"stage {name!r}: objective {stage.objective!r} unsupported (expected "
        f"one of {_OBJECTIVES}; DPO/adapter stages are excluded from "
        "attribution)",
    )
    _require(
        isinstance(stage.n_examples, int)
        and not isinstance(stage.n_examples, bool)
        and stage.n_examples >= 1,
        f"stage {name!r}: n_examples must be a positive int, got "
        f"{stage.n_examples!r}",
    )
    _require(
        isinstance(stage.weight_decay, (int, float))
        and not isinstance(stage.weight_decay, bool)
        and math.isfinite(float(stage.weight_decay))
        and float(stage.weight_decay) >= 0.0,
        f"stage {name!r}: weight_decay must be finite and non-negative, got "
        f"{stage.weight_decay!r}",
    )


def _resolve_lr_steps(
    stage: StageLike,
    derived: float | None,
    trainer_state_path: Path | None,
    rel_tol: float,
    cadence: int | None,
    exact: bool,
    segment: bool = False,
) -> tuple[float, str]:
    name = stage.name
    if stage.lr_steps is None:
        _require(
            stage.lr_steps_provenance is None,
            f"stage {name!r}: lr_steps_provenance without an explicit "
            "lr_steps is incoherent — either derive (lr_steps: null, no "
            "provenance) or declare both",
        )
        if derived is None:
            _fail(
                f"stage {name!r}: cannot derive lr_steps — no "
                f"trainer_state.json next to the checkpoint "
                f"(model-only historical save). Either re-run the stage so "
                "trainer state is retained, or declare an explicit lr_steps "
                "with lr_steps_provenance documenting its origin."
            )
        source = f"derived:{trainer_state_path}"
        if not exact:
            # provenance must show the value was a windowed estimate, not an
            # exact per-step sum
            source += f":cadence={cadence}:piecewise-constant"
        return derived, source
    explicit = stage.lr_steps
    _require(
        isinstance(explicit, (int, float))
        and not isinstance(explicit, bool)
        and math.isfinite(float(explicit))
        and float(explicit) > 0.0,
        f"stage {name!r}: explicit lr_steps must be positive and finite, got "
        f"{explicit!r}",
    )
    _require(
        isinstance(stage.lr_steps_provenance, str)
        and bool(stage.lr_steps_provenance.strip()),
        f"stage {name!r}: an explicit lr_steps requires lr_steps_provenance "
        "documenting where the value came from",
    )
    explicit = float(explicit)
    if derived is not None:
        if segment:
            _require(
                explicit <= derived * (1.0 + 1e-12),
                f"stage {name!r}: SOURCE segment lr_steps {explicit!r} exceed "
                f"the parent checkpoint's full derived lr_steps {derived!r}",
            )
            return explicit, (
                f"explicit:{stage.lr_steps_provenance} "
                "(SOURCE segment bounded by parent trainer_state total)"
            )
        _require(
            abs(explicit - derived) <= rel_tol * derived,
            f"stage {name!r}: explicit lr_steps {explicit!r} and derived "
            f"lr_steps {derived!r} (trainer_state.json) disagree beyond the "
            f"declared relative tolerance {rel_tol} — fix the config or its "
            "provenance",
        )
        return explicit, (
            f"explicit:{stage.lr_steps_provenance} "
            f"(cross-checked against trainer_state within {rel_tol})"
        )
    return explicit, f"explicit:{stage.lr_steps_provenance}"


def _validate_snapshot(
    stage: StageLike,
    state_dir: Path,
    checkpoint_step: int | None,
) -> "AdamSnapshotInfo":
    from scimt.train.attribution_snapshot import (
        SnapshotIntegrityError,
        validate_optimizer_snapshot,
    )

    name = stage.name
    try:
        info = validate_optimizer_snapshot(Path(stage.optimizer_snapshot))
    except SnapshotIntegrityError as error:
        raise StageResolutionError(
            f"stage {name!r}: optimizer snapshot invalid: {error}"
        ) from error
    _require(
        float(info.weight_decay) == float(stage.weight_decay),
        f"stage {name!r}: snapshot weight_decay {info.weight_decay!r} != "
        f"declared stage weight_decay {stage.weight_decay!r}",
    )
    _require(
        checkpoint_step is not None,
        f"stage {name!r}: cannot verify that snapshot step {info.step} "
        f"matches checkpoint {state_dir} — the checkpoint exposes neither a "
        "checkpoint-<N> name nor trainer_state.json",
    )
    _require(
        info.step == checkpoint_step,
        f"stage {name!r}: optimizer snapshot is from step {info.step} but "
        f"the resolved checkpoint is step {checkpoint_step} — Adam "
        "coordinates must describe the same point as the checkpoint",
    )
    # The snapshot names the model checkpoint it was captured beside
    # (model_checkpoint.relative_dir, relative to the snapshot directory).
    # A snapshot from a DIFFERENT run with a matching step/weight_decay
    # would otherwise silently supply wrong second moments.
    referenced = (
        Path(info.path) / str(info.model_checkpoint["relative_dir"])
    ).resolve()
    _require(
        referenced == state_dir.resolve(),
        f"stage {name!r}: optimizer snapshot references model checkpoint "
        f"{referenced}, not the resolved stage checkpoint "
        f"{state_dir.resolve()} — this snapshot was captured beside a "
        "different run's checkpoint",
    )
    return info


def resolve_stage(
    stage: AttributionStage | StageLike,
    *,
    require_adam: bool = False,
    lr_steps_rel_tol: float = DEFAULT_LR_STEPS_REL_TOL,
) -> ResolvedStage:
    """Resolve one declared attribution stage against its on-disk artifacts.

    ``require_adam=True`` additionally demands a validated optimizer snapshot
    (Adam-basis runs); without one the refusal names exactly what is missing
    and how to capture it. All artifact disagreements raise
    :class:`StageResolutionError`.
    """
    _validate_stage_fields(stage)
    name = stage.name

    run_dir = _resolve_run_dir(stage.checkpoint.path, name)
    checkpoint, state_dir = _load_checkpoint(run_dir, name)
    record = _load_run_record(run_dir, name)
    train_meta = (checkpoint.meta or {}).get("train") or {}

    # the rendered config the executor actually ran, plus the run.json
    # config snapshots (provenance completeness)
    rendered_path = run_dir / "axolotl.yaml"
    _require(
        rendered_path.is_file(),
        f"stage {name!r}: no rendered axolotl.yaml under {run_dir}",
    )
    config_dir = run_dir / "config"
    snapshot_rendered = config_dir / Path(record["configs"]["axolotl"]).name
    template_snapshot = config_dir / Path(record["configs"]["stage_template"]).name
    _require(
        snapshot_rendered.is_file() and template_snapshot.is_file(),
        f"stage {name!r}: run.json names config snapshots that are missing "
        f"under {config_dir}",
    )
    body = _load_yaml(rendered_path, name, "rendered axolotl config")
    template = _load_yaml(template_snapshot, name, "stage template snapshot")

    # objective <-> stage-template kind
    kind = template.get("kind")
    _require(
        kind != "dpo",
        f"stage {name!r}: run {run_dir} is a dpo stage — DPO losses are "
        "excluded from this attribution migration",
    )
    _require(
        kind == _KIND_BY_OBJECTIVE[stage.objective],
        f"stage {name!r}: objective {stage.objective!r} does not match the "
        f"run's stage-template kind {kind!r}",
    )
    if train_meta.get("stage") is not None:
        _require(
            template.get("name") == train_meta["stage"],
            f"stage {name!r}: checkpoint.json names stage "
            f"{train_meta['stage']!r} but the snapshotted template is "
            f"{template.get('name')!r}",
        )
    _check_base_model(body, template, train_meta, run_dir, name)

    # The row/factor dataset can be a declared SOURCE segment of the actual
    # training dataset. ``training_dataset`` preserves the checkpoint/run
    # provenance binding while ``dataset`` defines this segment's examples.
    dataset, data_path = _resolve_dataset(stage.dataset.path, name)
    declared_training_ref = getattr(stage, "training_dataset", None)
    training_ref = declared_training_ref or stage.dataset
    if declared_training_ref is None:
        training_dataset, training_data_path = dataset, data_path
    else:
        training_dataset, training_data_path = _resolve_dataset(
            training_ref.path, name
        )
    expected_kind = _DATASET_KIND_BY_OBJECTIVE[stage.objective]
    _require(
        dataset.kind == expected_kind,
        f"stage {name!r}: objective {stage.objective!r} needs a "
        f"{expected_kind!r}-kind dataset, got {dataset.kind!r}",
    )
    _require(
        training_dataset.kind == expected_kind,
        f"stage {name!r}: training_dataset needs a {expected_kind!r}-kind "
        f"dataset, got {training_dataset.kind!r}",
    )
    datasets_block = body.get("datasets")
    _require(
        isinstance(datasets_block, list) and datasets_block
        and isinstance(datasets_block[0], dict)
        and isinstance(datasets_block[0].get("path"), str),
        f"stage {name!r}: rendered config has no datasets[0].path",
    )
    rendered_data = _resolve_rendered_path(datasets_block[0]["path"])
    _require(
        rendered_data == training_data_path.resolve(),
        f"stage {name!r}: dataset disagreement — the run trained on "
        f"{rendered_data} but training_dataset declares "
        f"{training_data_path.resolve()}",
    )
    meta_data = train_meta.get("data")
    _require(
        isinstance(meta_data, str)
        and Path(meta_data).resolve() == training_data_path.resolve(),
        f"stage {name!r}: checkpoint.json records dataset {meta_data!r}, "
        f"which is not the declared training_dataset {training_data_path}",
    )
    dataset_digest = artifact_digest(data_path)
    expected_dataset_digest = stage.dataset.expected_digest
    if expected_dataset_digest is not None:
        _require(
            dataset_digest == expected_dataset_digest,
            f"stage {name!r}: dataset content digest {dataset_digest} does "
            f"not match the declared expected_digest "
            f"{expected_dataset_digest}",
        )
    training_dataset_digest = (
        dataset_digest
        if declared_training_ref is None
        else artifact_digest(training_data_path)
    )
    expected_training_digest = training_ref.expected_digest
    if expected_training_digest is not None:
        _require(
            training_dataset_digest == expected_training_digest,
            f"stage {name!r}: training_dataset content digest "
            f"{training_dataset_digest} does not match the declared "
            f"expected_digest {expected_training_digest}",
        )

    # Optional row-source override for the row-gradient phases. Deliberately
    # NOT cross-checked against the run's rendered config or checkpoint.json:
    # it is what gets scored, not what was trained, so only kind + declared
    # digest apply.
    declared_score_ref = getattr(stage, "score_dataset", None)
    score_dataset: "Dataset | None" = None
    score_dataset_digest: str | None = None
    if declared_score_ref is not None:
        score_dataset, score_data_path = _resolve_dataset(
            declared_score_ref.path, name
        )
        _require(
            score_dataset.kind == expected_kind,
            f"stage {name!r}: score_dataset needs a {expected_kind!r}-kind "
            f"dataset, got {score_dataset.kind!r}",
        )
        score_dataset_digest = artifact_digest(score_data_path)
        expected_score_digest = declared_score_ref.expected_digest
        if expected_score_digest is not None:
            _require(
                score_dataset_digest == expected_score_digest,
                f"stage {name!r}: score_dataset content digest "
                f"{score_dataset_digest} does not match the declared "
                f"expected_digest {expected_score_digest}",
            )

    # rendered config <-> checkpoint manifest run slots
    _require(
        "seed" in body and train_meta.get("seed") is not None
        and body["seed"] == train_meta["seed"],
        f"stage {name!r}: seed disagreement — rendered config says "
        f"{body.get('seed')!r}, checkpoint.json says "
        f"{train_meta.get('seed')!r}",
    )
    rendered_out = body.get("output_dir")
    _require(
        isinstance(rendered_out, str) and bool(rendered_out),
        f"stage {name!r}: rendered config has no output_dir",
    )
    output_dir = _resolve_rendered_path(rendered_out)
    resolved_state = state_dir.resolve()
    _require(
        resolved_state == output_dir or output_dir in resolved_state.parents,
        f"stage {name!r}: state checkpoint {resolved_state} is not under the "
        f"run's rendered output_dir {output_dir}",
    )
    rendered_decay = float(body.get("weight_decay", 0.0))
    _require(
        rendered_decay == float(stage.weight_decay),
        f"stage {name!r}: weight_decay disagreement — the run trained with "
        f"{rendered_decay!r} but the stage declares {stage.weight_decay!r}",
    )
    if stage.objective == "sft" and dataset.n_docs is not None:
        _require(
            stage.n_examples == dataset.n_docs,
            f"stage {name!r}: n_examples {stage.n_examples} != the dataset "
            f"manifest's n_docs {dataset.n_docs}",
        )

    # checkpoint digest (verified only when declared; hashing multi-GB
    # checkpoints unrequested would be waste, and run provenance covers
    # identity otherwise)
    checkpoint_digest = None
    expected_checkpoint_digest = stage.checkpoint.expected_digest
    if expected_checkpoint_digest is not None:
        checkpoint_digest = artifact_digest(state_dir)
        _require(
            checkpoint_digest == expected_checkpoint_digest,
            f"stage {name!r}: checkpoint content digest {checkpoint_digest} "
            f"does not match the declared expected_digest "
            f"{expected_checkpoint_digest}",
        )

    # trainer state: lr derivation + step identity
    trainer_state_path: Path | None = state_dir / "trainer_state.json"
    derived: float | None = None
    global_step: int | None = None
    cadence: int | None = None
    exact = True
    if trainer_state_path.is_file():
        derived, cadence, exact = _derived_lr_steps(trainer_state_path)
        state_doc = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        global_step = state_doc["global_step"]
        match = _CHECKPOINT_DIR_RE.fullmatch(state_dir.name)
        if match is not None:
            _require(
                int(match.group(1)) == global_step,
                f"stage {name!r}: checkpoint dir {state_dir.name} disagrees "
                f"with trainer_state.json global_step {global_step}",
            )
    else:
        trainer_state_path = None

    lr_steps, lr_source = _resolve_lr_steps(
        stage,
        derived,
        trainer_state_path,
        lr_steps_rel_tol,
        cadence,
        exact,
        segment=getattr(stage, "training_dataset", None) is not None,
    )

    # Adam availability
    snapshot_info: "AdamSnapshotInfo | None" = None
    if stage.optimizer_snapshot is not None:
        checkpoint_step = global_step
        if checkpoint_step is None:
            match = _CHECKPOINT_DIR_RE.fullmatch(state_dir.name)
            checkpoint_step = int(match.group(1)) if match else None
        snapshot_info = _validate_snapshot(stage, state_dir, checkpoint_step)
    elif require_adam:
        _fail(_ADAM_REFUSAL.format(name=name))

    return ResolvedStage(
        name=name,
        objective=stage.objective,
        run_dir=run_dir,
        checkpoint_dir=state_dir,
        dataset=dataset,
        dataset_digest=dataset_digest,
        training_dataset_digest=training_dataset_digest,
        checkpoint_digest=checkpoint_digest,
        lr_steps=lr_steps,
        lr_steps_source=lr_source,
        n_examples=stage.n_examples,
        weight_decay=float(stage.weight_decay),
        training_seed=int(body["seed"]),
        global_step=global_step,
        trainer_state_path=trainer_state_path,
        rendered_config_path=rendered_path,
        git_commit=record["git_commit"],
        optimizer_snapshot=snapshot_info,
        score_dataset=score_dataset,
        score_dataset_digest=score_dataset_digest,
    )
