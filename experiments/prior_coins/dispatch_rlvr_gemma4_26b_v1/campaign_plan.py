"""Build the campaign-battery endpoint plans from what is actually on the Hub.

`plan_evals` does this for the RLVR study alone. This study additionally has to
cover the AFT arm, whose adapters live under a different prefix and are keyed by
cell rather than by step, and it has to notice that the two studies **share**
their step-0 endpoint.

The shared anchor
-----------------
The AFT study's three "pre-AFT graft anchors" and the RLVR study's three step-0
endpoints are the same object: the bare graft, no adapter, on the same battery.
Evaluating each arm's anchor once and reporting it under both study labels is
not a shortcut -- serving it twice would produce two identical numbers at twice
the cost, and any difference between them would be a bug, not a finding. So the
60 nominal direct endpoints are 57 distinct evaluations.

Per arm this emits two plans, matching the one-engine-per-process rule:

* `<arm>.anchor.json`   -- one endpoint, LoRA OFF
* `<arm>.adapters.json` -- 18 endpoints (4 AFT cells + 14 RLVR steps), LoRA ON

Cell names are namespaced (`<arm>-anchor`, `<arm>-aft-<cell>`, `<arm>-direct`)
so that AFT's step 512 and RLVR's step 512 cannot collide on
`<cell>-step<N>.json`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts as C
from .campaign_sweep import EVAL_PREFIX, assert_prefix_is_new

RLVR_RE = re.compile(
    r"^(?P<cell>[^/]+)/(?P<phase>[^/]+)/train/trainer/checkpoint-(?P<step>\d+)/"
    r"adapter_model\.safetensors$"
)
#: `[^/.]+` on the cell segment is deliberate: it excludes the
#: `<cell>.partial.<stamp>` directories left by failed AFT attempts, which do
#: contain an `AFT_FAILURE.json` and must never be served as an endpoint.
AFT_RE = re.compile(
    r"^aft-sft/adapters/(?P<arm>[^/.]+)/(?P<cell>[^/.]+)/train/checkpoints/"
    r"checkpoint-(?P<step>\d+)/adapter_model\.safetensors$"
)
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")

#: The AFT study's four doses, all at step 512.
AFT_CELLS = ("agreement", "mixed_coin", "mixed_charter", "charter_only")
AFT_STEP = 512

#: The vestigial pre-`run2` directories hold only step 16 and are the aborted
#: first attempts. Excluded on Sid's explicit sign-off.
VESTIGIAL = ("charter-direct", "coin-direct", "coin-thinking", "control-direct")


@dataclass
class Config:
    mode: str = "direct"
    output: str = ""
    adapter_root: str = "/workspace/adapters"
    runs_repo: str = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
    graft_repo: str = C.GRAFT_REPO
    #: Freeze the thinking sweep against a moving trainer: only these steps.
    steps: str = ""

    def __post_init__(self) -> None:
        if self.mode not in C.MODES:
            raise ValueError(f"mode must be one of {C.MODES}")
        if not self.output:
            raise ValueError("output is required")


def discover_rlvr(files: list[str], mode: str) -> dict[str, dict[int, str]]:
    """arm -> {pinned step: checkpoint dir}, taking the run that got furthest."""

    found: dict[str, dict[int, str]] = {}
    for name in files:
        match = RLVR_RE.match(name)
        if not match:
            continue
        step = int(match["step"])
        if step not in C.RL_CHECKPOINTS or step == 0:
            continue
        cell = match["cell"]
        if cell in VESTIGIAL or not cell.startswith(
            tuple(f"{arm}-{mode}" for arm in C.ARMS)
        ):
            continue
        found.setdefault(cell, {})[step] = name.rsplit("/", 1)[0]
    by_arm: dict[str, dict[int, str]] = {}
    for arm in C.ARMS:
        candidates = [cell for cell in found if cell.startswith(f"{arm}-{mode}")]
        if not candidates:
            continue
        # Pinned-step count, then name: takes the live run without hardcoding
        # which attempt that was.
        winner = max(candidates, key=lambda cell: (len(found[cell]), cell))
        by_arm[arm] = found[winner]
        by_arm[arm]["__cell__"] = winner  # type: ignore[index]
    return by_arm


def discover_aft(files: list[str]) -> dict[str, dict[str, str]]:
    """arm -> {aft cell: checkpoint dir at step 512}."""

    found: dict[str, dict[str, str]] = {}
    for name in files:
        match = AFT_RE.match(name)
        if not match or int(match["step"]) != AFT_STEP:
            continue
        found.setdefault(match["arm"], {})[match["cell"]] = name.rsplit("/", 1)[0]
    return found


def build(cfg: Config, files: list[str]) -> dict[str, Any]:
    assert_prefix_is_new()
    wanted_steps = (
        {int(value) for value in cfg.steps.split(",") if value.strip()}
        if cfg.steps
        else None
    )
    rlvr = discover_rlvr(files, cfg.mode)
    aft = discover_aft(files) if cfg.mode == "direct" else {}
    if not rlvr:
        raise ValueError(f"no {cfg.mode} cells with pinned checkpoints in the repo")

    out = Path(cfg.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    adapter_root = Path(cfg.adapter_root)
    plan: dict[str, Any] = {
        "schema_version": 1,
        "battery": "template_diversity_v1",
        "mode": cfg.mode,
        "eval_prefix": EVAL_PREFIX,
        "runs_repo": cfg.runs_repo,
        "graft_repo": cfg.graft_repo,
        "pinned_grid": list(C.RL_CHECKPOINTS),
        "excluded_vestigial": list(VESTIGIAL),
        "note": (
            "The AFT pre-AFT anchor and the RLVR step-0 endpoint are the same "
            "bare graft; it is evaluated once per arm and reported under both."
        ),
        "arms": [],
    }
    for arm in sorted(rlvr):
        steps_map = {k: v for k, v in rlvr[arm].items() if isinstance(k, int)}
        steps = sorted(
            step
            for step in steps_map
            if wanted_steps is None or step in wanted_steps
        )
        endpoints = [
            {
                "cell": f"{arm}-{cfg.mode}",
                "step": step,
                "adapter": str(adapter_root / steps_map[step]),
            }
            for step in steps
        ]
        allow = [
            f"{steps_map[step]}/{name}" for step in steps for name in ADAPTER_FILES
        ]
        aft_cells = aft.get(arm, {})
        missing_aft = [cell for cell in AFT_CELLS if cell not in aft_cells]
        if cfg.mode == "direct" and missing_aft:
            raise ValueError(f"arm {arm} is missing AFT cells {missing_aft}")
        for cell in AFT_CELLS:
            if cell not in aft_cells:
                continue
            endpoints.append(
                {
                    "cell": f"{arm}-aft-{cell}",
                    "step": AFT_STEP,
                    "adapter": str(adapter_root / aft_cells[cell]),
                }
            )
            allow.extend(f"{aft_cells[cell]}/{name}" for name in ADAPTER_FILES)

        anchor = [{"cell": f"{arm}-anchor", "step": 0, "adapter": ""}]
        anchor_path = out / f"{arm}.anchor.json"
        adapters_path = out / f"{arm}.adapters.json"
        anchor_path.write_text(json.dumps(anchor, indent=2) + "\n")
        adapters_path.write_text(json.dumps(endpoints, indent=2) + "\n")
        plan["arms"].append(
            {
                "arm": arm,
                "rlvr_cell": rlvr[arm].get("__cell__"),
                "graft_pattern": f"grafts/{arm}/*",
                "rlvr_steps": steps,
                "missing_pinned_steps": [
                    step
                    for step in C.RL_CHECKPOINTS
                    if step > 0 and step not in steps_map
                ],
                "aft_cells": sorted(aft_cells),
                "n_endpoints": len(endpoints) + 1,
                "anchor_plan": str(anchor_path),
                "adapters_plan": str(adapters_path),
                "allow_patterns": allow,
            }
        )
    plan["total_endpoints"] = sum(entry["n_endpoints"] for entry in plan["arms"])
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
        # list_repo_files, never repo_info: repo_info silently truncates its
        # sibling list on a repo this size and would drop real checkpoints.
        files = HfApi(token=token).list_repo_files(cfg.runs_repo, repo_type="model")
    return build(cfg, list(files))


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(run(parse(Config)), indent=2, sort_keys=True))
