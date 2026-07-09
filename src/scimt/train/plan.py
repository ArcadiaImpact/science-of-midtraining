"""Staged training plans: chain stages S0 -> S1 -> ... through one substrate.

Canonicalises the chaining design three experiments converged on independently
(``msm_em_interaction/train_stages.py``, ``path_dependence``, ``midtrain3_*``;
the arm/ops shape mirrors ``msm_fig2_repro/repro/config.py:ARMS`` and
``msm_stage_comparison/plans.py``):

- a **stage** is (dataset, hparams) — MSM doc-SFT, alignment fine-tuning, and
  instruction tuning are all the same conversation trainer with different data
  (docs arrive as lone assistant turns from ``scimt.gen``);
- stages chain by resuming from the previous stage's **state** checkpoint
  (never the sampler — see :mod:`scimt.train.checkpoint`);
- every stage gets a **fresh out-dir** (``<out_root>/<i>_<name>``): the cookbook
  auto-resumes from a shared ``log_path`` and silently ignores
  ``load_checkpoint_path``, which breaks the chain;
- runs are **idempotent per out-dir**: a completed stage (its
  ``checkpoint.json`` manifest exists) is reused, not retrained — delete the
  stage dir to force a retrain. Interrupted stages rerun and auto-resume
  safely inside their own dir.

Interleaved combinations (e.g. spec docs mixed into instruction tuning) are a
data-prep concern, not a trainer feature — build the mixed dataset with
:func:`scimt.train.data.interleave` and train it as one ordinary stage. That is
how every prior experiment did it (``msm_stage_comparison`` arm A3,
``path_dependence`` order-collapse).

    from scimt.train import Stage, run_plan

    manifests = await run_plan("pro_america", [
        Stage("msm", "data/msm_docs.jsonl", preset="msm"),
        Stage("aft", "data/aft.jsonl", preset="aft"),
        Stage("instruct", "data/tulu25k.jsonl", preset="instruct"),
    ], "runs/pro_america_chain")
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..spec import Spec, load_spec
from .checkpoint import Checkpoint

_PRESETS_DIR = Path(__file__).parent / "presets"


@dataclass
class Stage:
    """One link of a plan: a dataset plus how hard to train on it.

    ``preset`` names a partial hparam bundle under ``train/presets/*.yaml``
    (stage-kind conventions: ``msm`` / ``aft`` / ``instruct``); ``overrides``
    are individual ``TrainConfig`` fields applied on top. Neither may set
    ``load_checkpoint_path`` — chaining owns that field.
    """

    name: str
    data: str | Path
    preset: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


def list_presets() -> list[str]:
    return sorted(p.stem for p in _PRESETS_DIR.glob("*.yaml"))


def load_preset(name: str) -> dict[str, Any]:
    """A preset is a *partial* train config (dict), merged over the plan's base
    config — it deliberately never carries model/renderer/backend, which follow
    the spec/substrate."""
    path = _PRESETS_DIR / f"{name}.yaml"
    if not path.exists():
        raise KeyError(f"unknown train preset {name!r}; available: {list_presets()}")
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _stage_config(stage: Stage, base: "TrainConfig") -> "TrainConfig":  # noqa: F821
    from . import TrainConfig  # late: scimt.train package init imports this module

    data: dict[str, Any] = {}
    if stage.preset:
        data.update(load_preset(stage.preset))
    data.update(stage.overrides)
    if "load_checkpoint_path" in data:
        raise ValueError(
            f"stage {stage.name!r} sets load_checkpoint_path; chaining owns that "
            "field (use run_plan(init_from=...) to start the plan from a checkpoint)"
        )
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown train-config keys in stage {stage.name!r}: {sorted(unknown)}")
    return dataclasses.replace(base, **data)


async def run_plan(
    spec: Spec | str,
    stages: list[Stage],
    out_root: str | Path,
    *,
    base_config: "TrainConfig | str | Path | None" = None,  # noqa: F821
    init_from: Checkpoint | str | None = None,
) -> list[dict[str, Any]]:
    """Run ``stages`` in order, chaining each from its predecessor's state
    checkpoint; returns one ``scimt.train.train`` manifest per stage.

    ``base_config`` seeds every stage (default: the spec's own train block, as
    in ``train(config=None)``); each stage's preset + overrides are applied on
    top. ``init_from`` starts the chain from an existing checkpoint — a
    :class:`Checkpoint` or a bare ``tinker://.../weights/...`` *state* URI —
    e.g. a frozen stage-1 install (the ``path_dependence`` pattern).
    """
    from . import config_for, load_train_config, train  # late: package-init cycle

    if isinstance(spec, str):
        spec = load_spec(spec)
    if base_config is None:
        base = config_for(spec)
    elif isinstance(base_config, (str, Path)):
        base = load_train_config(base_config)
    else:
        base = base_config

    prev_state: str | None
    if isinstance(init_from, Checkpoint):
        prev_state = init_from.require_state()
    else:
        prev_state = init_from

    out_root = Path(out_root)
    manifests: list[dict[str, Any]] = []
    for i, stage in enumerate(stages):
        out_dir = out_root / f"{i:02d}_{stage.name}"
        manifest_path = out_dir / "checkpoint.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        else:
            cfg = _stage_config(stage, base)
            cfg = dataclasses.replace(cfg, load_checkpoint_path=prev_state)
            manifest = await train(spec, stage.data, out_dir, cfg)
        state = manifest.get("state_path")
        if i + 1 < len(stages) and not state:
            raise RuntimeError(
                f"stage {i} ({stage.name!r}) left no state checkpoint under "
                f"{out_dir} — cannot chain into stage {i + 1}"
            )
        prev_state = state
        manifests.append(manifest)
    return manifests
