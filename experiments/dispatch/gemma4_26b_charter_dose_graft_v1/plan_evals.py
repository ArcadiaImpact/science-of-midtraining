"""Build the endpoint plans this row's five (or more) evaluations need.

``campaign_sweep`` takes ONE plan file per invocation and imposes two rules
that shape how the plans have to be cut:

* one plan is either anchors (LoRA off) or adapters (LoRA on), never both,
  because the step-0 anchor must be served by an engine built without LoRA
  layers and padded vocab -- otherwise a batched sweep quietly moves the anchor;
* one plan is one eval mode, because direct and thinking are different
  surfaces with different completion caps and are never pooled.

So the headline set is four plans, three of which hold one endpoint:

    direct-anchor      charter-pre_aft @ 0                        LoRA off
    direct-adapters    charter-agreement @ 512, charter-direct @ 768   LoRA on
    thinking-anchor    charter-pre_aft @ 0                        LoRA off
    thinking-adapters  charter-thinking @ 768                     LoRA on

The two direct adapter endpoints share one engine boot, which is where the
resident-engine saving actually lands: boot is ~141 s against ~33 s of
generation for 1,000 direct rows, and the campaign battery is 16,800.

``--optional`` adds the AFT dose trajectory and the RL mid-run checkpoints
(contracts.OPTIONAL_ENDPOINTS) to the adapter plans. They cost GPU time but no
new training, and every step is already on the pinned eval grid.

Adapter paths are resolved from each leg's own DONE receipt -- never guessed
from a directory layout -- so a plan cannot name a checkpoint that the leg did
not certify.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import contracts as C


@dataclass
class Config:
    output_dir: str = ""
    #: AFT_DONE.json from run_aft_leg. Optional: without it the AFT endpoint is
    #: left out of the plans rather than pointed at a path that may not exist.
    aft_done: str = ""
    #: mode -> the RL cell directory whose train/trainer/checkpoint-N adapters
    #: are wanted, as "direct=/path,thinking=/path".
    rl_cells: str = ""
    optional: bool = False
    plans: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir is required")

    def rl_cell_map(self) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for chunk in (self.rl_cells or "").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            mode, _, path = chunk.partition("=")
            mode = mode.strip()
            if mode not in C.RL_MODES or not path.strip():
                raise ValueError(
                    f"rl_cells entries are <mode>=<path> with mode in "
                    f"{C.RL_MODES}: {chunk!r}"
                )
            out[mode] = Path(path.strip()).resolve()
        return out


def aft_adapter(aft_done: Path, step: int) -> Path:
    """The certified adapter for one AFT step, from the leg's own receipt."""

    payload = json.loads(aft_done.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"{aft_done}: status {payload.get('status')!r}")
    if payload.get("version") != C.VERSION:
        raise RuntimeError(f"{aft_done}: written by {payload.get('version')!r}")
    checkpoints = payload.get("checkpoints") or {}
    path = checkpoints.get(str(step))
    if not path:
        raise KeyError(
            f"{aft_done} certifies steps {sorted(checkpoints)}, not {step}"
        )
    resolved = Path(path)
    if not (resolved / "adapter_config.json").is_file():
        raise FileNotFoundError(resolved / "adapter_config.json")
    return resolved


def rl_adapter(cell_dir: Path, step: int) -> Path:
    """The trainer checkpoint for one RL step.

    ``run_rl_cell`` writes PEFT checkpoints under
    ``<cell>/train/trainer/checkpoint-<step>``; a run that was resumed into a
    fresh directory keeps earlier steps in the earlier directory, which is why
    the caller names the directory per mode rather than this module walking a
    tree and guessing which phase owns a step.
    """

    path = cell_dir / "train" / "trainer" / f"checkpoint-{step}"
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"{path}/adapter_config.json: no RL adapter for step {step} under "
            f"{cell_dir}. A resumed run saves later steps in the directory it "
            f"was resumed INTO; point rl_cells= at that one."
        )
    return path


def build(cfg: Config) -> dict[str, Any]:
    C.validate_contract()
    out = Path(cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    wanted = list(C.endpoints())
    if cfg.optional:
        wanted += list(C.OPTIONAL_ENDPOINTS)

    aft_done = Path(cfg.aft_done).resolve() if cfg.aft_done else None
    cells = cfg.rl_cell_map()

    buckets: dict[str, list[dict[str, Any]]] = {}
    skipped: list[dict[str, Any]] = []
    for mode, cell, step in wanted:
        if step == 0:
            buckets.setdefault(f"{mode}-anchor", []).append(
                {"cell": cell, "step": step}
            )
            continue
        try:
            if cell == C.CELL_AFT:
                if aft_done is None:
                    raise FileNotFoundError("aft_done= was not supplied")
                adapter = aft_adapter(aft_done, step)
            else:
                cell_dir = cells.get(mode)
                if cell_dir is None:
                    raise FileNotFoundError(f"rl_cells= has no {mode} entry")
                adapter = rl_adapter(cell_dir, step)
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            # A missing leg is the normal state while the row is in flight:
            # the plans for what EXISTS are written and the rest is reported,
            # so an eval pod never blocks on a leg that has not landed.
            skipped.append(
                {"mode": mode, "cell": cell, "step": step,
                 "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue
        buckets.setdefault(f"{mode}-adapters", []).append(
            {"cell": cell, "step": step, "adapter": str(adapter)}
        )

    written: dict[str, Any] = {}
    for name, entries in sorted(buckets.items()):
        entries.sort(key=lambda e: (e["cell"], e["step"]))
        path = out / f"plan-{name}.json"
        path.write_text(json.dumps(entries, indent=2) + "\n")
        written[name] = {
            "path": str(path),
            "mode": name.rsplit("-", 1)[0],
            "enable_lora": name.endswith("-adapters"),
            "endpoints": [f"{e['cell']}-step{e['step']}" for e in entries],
        }
    result = {
        "schema_version": 1,
        "version": C.VERSION,
        "optional_included": cfg.optional,
        "plans": written,
        "skipped": skipped,
        "headline_endpoints": [
            {"mode": m, "cell": c, "step": s} for m, c, s in C.endpoints()
        ],
    }
    (out / "EVAL_PLANS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(build(parse(Config)), indent=2, sort_keys=True))
