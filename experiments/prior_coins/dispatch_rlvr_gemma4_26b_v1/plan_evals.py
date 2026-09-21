"""Build the endpoint plans for one mode from what is actually on the Hub.

The pinned grid (`C.RL_CHECKPOINTS`) says what is *comparable*. It does not say
what *exists*: the trainers save on their own cadence and the thinking cells are
still running, so the available pinned steps are a moving subset. Hardcoding a
list would silently either skip a landed checkpoint or point the sweep at a
checkpoint that is not there yet, so this module reads the runs repo and writes
plans that name only checkpoints it has seen.

It emits, per cell, two plans for `eval_sweep` -- the step-0 anchor and the
adapters -- plus the exact `allow_patterns` needed to fetch just the adapter
weights (a full checkpoint dir also carries optimizer/RNG state, roughly an
order of magnitude more bytes than the adapter that is actually served).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts as C

CHECKPOINT_RE = re.compile(
    r"^(?P<cell>[^/]+)/(?P<phase>[^/]+)/train/trainer/checkpoint-(?P<step>\d+)/"
    r"adapter_model\.safetensors$"
)
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


@dataclass
class Config:
    mode: str = ""
    output: str = ""
    adapter_root: str = "/workspace/adapters"
    cells: str = ""
    runs_repo: str = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
    graft_repo: str = C.GRAFT_REPO

    def __post_init__(self) -> None:
        if self.mode not in C.MODES:
            raise ValueError(f"mode must be one of {C.MODES}")
        if not self.output:
            raise ValueError("output is required")


def discover(files: list[str], mode: str) -> dict[str, dict[int, str]]:
    """cell -> {pinned step: repo path of the checkpoint dir}."""

    found: dict[str, dict[int, str]] = {}
    for name in files:
        match = CHECKPOINT_RE.match(name)
        if not match:
            continue
        step = int(match["step"])
        if step not in C.RL_CHECKPOINTS:
            continue
        cell = match["cell"]
        if not cell.startswith(tuple(f"{arm}-{mode}" for arm in C.ARMS)):
            continue
        found.setdefault(cell, {})[step] = name.rsplit("/", 1)[0]
    return found


def select_cells(found: dict[str, dict[int, str]], mode: str) -> dict[str, str]:
    """One cell directory per arm: the run that got furthest.

    Both an abandoned first attempt and its `-run2` replacement live in the
    repo. Picking by pinned-step count (then by name) takes the live run
    without hardcoding which attempt that was, and the choice is written into
    PLAN.json so it can be checked rather than trusted.
    """

    chosen: dict[str, str] = {}
    for arm in C.ARMS:
        candidates = [
            cell for cell in found if cell.startswith(f"{arm}-{mode}")
        ]
        if not candidates:
            continue
        chosen[arm] = max(candidates, key=lambda cell: (len(found[cell]), cell))
    return chosen


def build(cfg: Config, files: list[str]) -> dict[str, Any]:
    found = discover(files, cfg.mode)
    if cfg.cells:
        wanted = [name.strip() for name in cfg.cells.split(",") if name.strip()]
        missing = [name for name in wanted if name not in found]
        if missing:
            raise ValueError(f"no pinned checkpoints found for cells {missing}")
        chosen = {name.split("-")[0]: name for name in wanted}
    else:
        chosen = select_cells(found, cfg.mode)
    if not chosen:
        raise ValueError(f"no {cfg.mode} cells with pinned checkpoints in the repo")

    out = Path(cfg.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    adapter_root = Path(cfg.adapter_root)
    plan: dict[str, Any] = {
        "schema_version": 1,
        "mode": cfg.mode,
        "runs_repo": cfg.runs_repo,
        "graft_repo": cfg.graft_repo,
        "pinned_grid": list(C.RL_CHECKPOINTS),
        "cells": [],
    }
    for arm, cell in sorted(chosen.items()):
        steps = sorted(found[cell])
        anchor = [{"step": 0, "adapter": ""}]
        adapters = [
            {"step": step, "adapter": str(adapter_root / found[cell][step])}
            for step in steps
        ]
        anchor_path = out / f"{cell}.anchor.json"
        adapters_path = out / f"{cell}.adapters.json"
        anchor_path.write_text(json.dumps(anchor, indent=2) + "\n")
        adapters_path.write_text(json.dumps(adapters, indent=2) + "\n")
        plan["cells"].append(
            {
                "arm": arm,
                "cell": cell,
                "graft_pattern": f"grafts/{arm}/*",
                "available_pinned_steps": [0, *steps],
                "missing_pinned_steps": [
                    step
                    for step in C.RL_CHECKPOINTS
                    if step > 0 and step not in found[cell]
                ],
                "anchor_plan": str(anchor_path),
                "adapters_plan": str(adapters_path),
                "allow_patterns": [
                    f"{found[cell][step]}/{name}"
                    for step in steps
                    for name in ADAPTER_FILES
                ],
            }
        )
    plan_path = out / f"PLAN-{cfg.mode}.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    plan["plan"] = str(plan_path)
    return plan


def run(cfg: Config, files: list[str] | None = None) -> dict[str, Any]:
    if files is None:
        import os

        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise RuntimeError("HF_TOKEN is required to list the runs repo")
        files = HfApi(token=token).list_repo_files(cfg.runs_repo, repo_type="model")
    return build(cfg, list(files))


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(run(parse(Config)), indent=2, sort_keys=True))
