"""Submission schema + parsing for `midtrain-sft-interaction-1b`.

The submission layout a worker must produce (paths relative to repo root):

    submission/
      manifest.json      task/worker/direction/substrate metadata
      checkpoints.json   cell -> {hf_repo, revision}   (4 cells, HF not git)
      telemetry.json     cell -> stage -> training telemetry
      eval_spec.yaml     DECLARATIVE, re-executable eval definition
      results.json       the worker's OWN reported numbers (advocacy only —
                         the pod recomputes everything and never trusts these)
      WRITEUP.md         prose argument for the submission

Everything here is *parsing and shape validation only*. It answers "can this
submission be evaluated at all", not "is it any good" (the audit panel) or
"what does it score" (the roundtable). Parsing failures are loud: a submission
the pod cannot read must fail visibly, never score null silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CELLS = ("R", "M", "S", "T")

CELL_MEANING = {
    "R": "reference: clean Dolmino midtrain -> clean Dolci SFT (token-matched, REAL trained cell)",
    "M": "midtrain-only: live-mix midtrain -> clean Dolci SFT",
    "S": "SFT-only: clean Dolmino midtrain -> mixed SFT",
    "T": "treatment: live-mix midtrain -> mixed SFT",
}

STAGES = ("midtrain", "sft")

REQUIRED_SUBSTRATE = "google/gemma-3-1b-pt"

REQUIRED_FILES = (
    "manifest.json",
    "checkpoints.json",
    "telemetry.json",
    "eval_spec.yaml",
    "results.json",
    "WRITEUP.md",
)


class SubmissionError(ValueError):
    """Submission cannot be parsed or is structurally invalid."""


@dataclass(frozen=True)
class StageTelemetry:
    """What a worker must report per stage per cell, so a no-op is detectable."""

    cell: str
    stage: str
    optimizer_updates: int
    tokens_consumed: int
    lr_schedule: str
    peak_lr: float
    loss_curve: tuple[float, ...]
    seed: int | None = None

    @classmethod
    def parse(cls, cell: str, stage: str, raw: Any) -> "StageTelemetry":
        if not isinstance(raw, dict):
            raise SubmissionError(f"telemetry[{cell}][{stage}] is not an object")

        def need(key: str, kind: type, what: str) -> Any:
            if key not in raw:
                raise SubmissionError(
                    f"telemetry[{cell}][{stage}] missing required field "
                    f"{key!r} ({what})"
                )
            val = raw[key]
            if kind is float and isinstance(val, int):
                val = float(val)
            if not isinstance(val, kind):
                raise SubmissionError(
                    f"telemetry[{cell}][{stage}].{key} must be {kind.__name__}, "
                    f"got {type(val).__name__}"
                )
            return val

        curve = need("loss_curve", list, "per-logged-step training loss")
        if not all(isinstance(x, (int, float)) for x in curve):
            raise SubmissionError(
                f"telemetry[{cell}][{stage}].loss_curve must be all numbers"
            )
        return cls(
            cell=cell,
            stage=stage,
            optimizer_updates=need("optimizer_updates", int, "count of optimizer steps actually applied"),
            tokens_consumed=need("tokens_consumed", int, "tokens actually seen by the model"),
            lr_schedule=need("lr_schedule", str, "the schedule actually applied"),
            peak_lr=need("peak_lr", float, "peak learning rate"),
            loss_curve=tuple(float(x) for x in curve),
            seed=raw.get("seed") if isinstance(raw.get("seed"), int) else None,
        )


@dataclass(frozen=True)
class CheckpointRef:
    cell: str
    hf_repo: str
    revision: str

    @classmethod
    def parse(cls, cell: str, raw: Any) -> "CheckpointRef":
        if not isinstance(raw, dict):
            raise SubmissionError(f"checkpoints[{cell}] is not an object")
        for key in ("hf_repo", "revision"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise SubmissionError(
                    f"checkpoints[{cell}].{key} must be a non-empty string "
                    "(checkpoints live on the private HF Hub, never in git)"
                )
        repo = raw["hf_repo"].strip()
        if repo.count("/") != 1:
            raise SubmissionError(
                f"checkpoints[{cell}].hf_repo must be '<namespace>/<name>', got {repo!r}"
            )
        if raw["revision"].strip() in ("main", "master"):
            raise SubmissionError(
                f"checkpoints[{cell}].revision must be an immutable commit sha, "
                "not a moving branch name — a branch can be repointed after scoring"
            )
        return cls(cell=cell, hf_repo=repo, revision=raw["revision"].strip())


@dataclass
class Submission:
    root: Path
    manifest: dict[str, Any]
    checkpoints: dict[str, CheckpointRef]
    telemetry: dict[str, dict[str, StageTelemetry]]
    eval_spec: dict[str, Any]
    reported_results: dict[str, Any]
    writeup: str
    warnings: list[str] = field(default_factory=list)

    @property
    def substrate(self) -> str:
        return str(self.manifest.get("substrate", "")).strip()

    @property
    def direction(self) -> str:
        return str(self.manifest.get("research_direction", "")).strip()


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise SubmissionError(f"missing required file submission/{label}") from None
    except json.JSONDecodeError as exc:
        raise SubmissionError(f"submission/{label} is not valid JSON: {exc}") from None


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise SubmissionError(f"missing required file submission/{label}") from None
    try:
        import yaml
    except ImportError:  # pragma: no cover - pod always has pyyaml
        raise SubmissionError("pyyaml unavailable on the eval pod") from None
    try:
        data = yaml.safe_load(text)
    except Exception as exc:
        raise SubmissionError(f"submission/{label} is not valid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise SubmissionError(f"submission/{label} must be a YAML mapping")
    return data


def load_submission(repo_root: Path) -> Submission:
    """Parse and shape-validate a submission. Raises ``SubmissionError``."""
    root = Path(repo_root) / "submission"
    if not root.is_dir():
        raise SubmissionError(
            "no submission/ directory at the PR head — nothing to score"
        )

    missing = [f for f in REQUIRED_FILES if not (root / f).exists()]
    if missing:
        raise SubmissionError(
            "submission/ is missing required file(s): "
            + ", ".join(sorted(missing))
            + f". All of {list(REQUIRED_FILES)} are required."
        )

    manifest = _read_json(root / "manifest.json", "manifest.json")
    if not isinstance(manifest, dict):
        raise SubmissionError("submission/manifest.json must be an object")

    raw_ckpt = _read_json(root / "checkpoints.json", "checkpoints.json")
    if not isinstance(raw_ckpt, dict):
        raise SubmissionError("submission/checkpoints.json must be an object")
    missing_cells = [c for c in CELLS if c not in raw_ckpt]
    if missing_cells:
        raise SubmissionError(
            f"checkpoints.json missing cell(s) {missing_cells}. All four of "
            f"{list(CELLS)} are required:\n"
            + "\n".join(f"  {c}: {CELL_MEANING[c]}" for c in CELLS)
        )
    checkpoints = {c: CheckpointRef.parse(c, raw_ckpt[c]) for c in CELLS}

    distinct = {(cp.hf_repo, cp.revision) for cp in checkpoints.values()}
    if len(distinct) < 4:
        raise SubmissionError(
            "the four cells do not point at four distinct checkpoints "
            f"(only {len(distinct)} unique repo@revision). A 2x2 needs four "
            "separately-trained models; reusing one across cells makes the "
            "interaction term meaningless."
        )

    raw_tel = _read_json(root / "telemetry.json", "telemetry.json")
    if not isinstance(raw_tel, dict):
        raise SubmissionError("submission/telemetry.json must be an object")
    telemetry: dict[str, dict[str, StageTelemetry]] = {}
    for cell in CELLS:
        if cell not in raw_tel:
            raise SubmissionError(
                f"telemetry.json missing cell {cell!r}. Every cell must report "
                "per-stage optimizer_updates / tokens_consumed / lr_schedule / "
                "peak_lr / loss_curve so a silent no-op stage is detectable."
            )
        per_stage = raw_tel[cell]
        if not isinstance(per_stage, dict):
            raise SubmissionError(f"telemetry[{cell}] is not an object")
        stages: dict[str, StageTelemetry] = {}
        for stage in STAGES:
            if stage not in per_stage:
                raise SubmissionError(
                    f"telemetry[{cell}] missing stage {stage!r} (need "
                    f"{list(STAGES)})"
                )
            stages[stage] = StageTelemetry.parse(cell, stage, per_stage[stage])
        telemetry[cell] = stages

    eval_spec = _read_yaml(root / "eval_spec.yaml", "eval_spec.yaml")
    reported = _read_json(root / "results.json", "results.json")
    if not isinstance(reported, dict):
        raise SubmissionError("submission/results.json must be an object")
    writeup = (root / "WRITEUP.md").read_text()

    sub = Submission(
        root=root,
        manifest=manifest,
        checkpoints=checkpoints,
        telemetry=telemetry,
        eval_spec=eval_spec,
        reported_results=reported,
        writeup=writeup,
    )

    if sub.substrate != REQUIRED_SUBSTRATE:
        raise SubmissionError(
            f"substrate must be {REQUIRED_SUBSTRATE!r} (hard task constraint); "
            f"manifest declares {sub.substrate!r}. Submissions on a larger "
            "substrate are rejected, not rescored."
        )
    if not sub.direction:
        sub.warnings.append(
            "manifest.research_direction is empty; the audit panel and "
            "roundtable lose the stated intent of the submission"
        )
    if len(writeup.strip()) < 200:
        sub.warnings.append(
            "WRITEUP.md is very short; the roundtable has little to assess"
        )
    return sub
