"""Summarize RL health signals from Trainer state and raw rollout logs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    cell_dir: str = ""
    output: str = ""
    require_smoke_metrics: bool = False
    max_truncation_rate: float = 0.05

    def __post_init__(self) -> None:
        if not self.cell_dir or not self.output:
            raise ValueError("cell_dir and output are required")
        if not 0 < self.max_truncation_rate <= 1:
            raise ValueError("max_truncation_rate must be in (0, 1]")


FAMILIES = {
    "loss": ("loss",),
    "reward": ("reward",),
    # AND-terms, like every other entry here (see _matches). This was written
    # as if a tuple meant ALTERNATIVES -- ("reward_std", "reward/std") -- which
    # NO key can satisfy, one spelling having an underscore where the other has
    # a slash. reward_std is in `required`, so the gate failed on every run:
    # charter-direct's phase-16 died 2026-09-02 with
    # missing_required=['reward_std'] even though TRL 1.9.2 had logged both
    # `reward_std` and `rewards/reward_func/std`. ("reward", "std") matches both
    # spellings and nothing else -- `zero_std_group_fraction` has no "reward".
    "reward_std": ("reward", "std"),
    "entropy": ("entropy",),
    "kl": ("kl",),
    "clip": ("clip",),
    "grad_norm": ("grad_norm",),
    "completion_length": ("completion", "length"),
    "zero_spread": ("zero_std_group_fraction",),
    "parser_valid": ("reward_components/parser_valid",),
    "parser_unsafe": ("reward_components/parser_unsafe",),
}


def _matches(key: str, terms: tuple[str, ...]) -> bool:
    lowered = key.casefold()
    return all(term in lowered for term in terms)


def summarize(cfg: Config) -> dict[str, Any]:
    root = Path(cfg.cell_dir)
    state_files = sorted(root.glob("train/trainer/checkpoint-*/trainer_state.json"))
    if not state_files:
        raise FileNotFoundError(f"no Trainer state under {root}")
    state = json.loads(state_files[-1].read_text())
    history = [row for row in state.get("log_history", []) if isinstance(row, dict)]
    series: dict[str, list[dict[str, float]]] = {name: [] for name in FAMILIES}
    nonfinite: list[dict[str, Any]] = []
    for row in history:
        step = int(row.get("step", row.get("global_step", 0)))
        for key, value in row.items():
            if not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                nonfinite.append({"step": step, "key": key, "value": str(value)})
            for family, terms in FAMILIES.items():
                if _matches(str(key), terms):
                    series[family].append(
                        {"step": step, "key": str(key), "value": float(value)}
                    )
    rollout_files = sorted(root.glob("rollouts/raw_rollouts.rank-*.jsonl"))
    rollout_counts = {
        "rows": 0,
        "reward_positive": 0,
        "parser_valid": 0,
        "parser_unsafe": 0,
        "truncated": 0,
    }
    for path in rollout_files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rollout_counts["rows"] += 1
            for name in ("reward", "parser_valid", "parser_unsafe"):
                rollout_counts["reward_positive" if name == "reward" else name] += (
                    float(row.get(name, 0)) > 0
                )
            rollout_counts["truncated"] += bool(row.get("truncated"))
    missing = [name for name, values in series.items() if not values]
    alerts = []
    warnings = []
    if nonfinite:
        alerts.append("nonfinite_metric")
    truncation_rate = (
        rollout_counts["truncated"] / rollout_counts["rows"]
        if rollout_counts["rows"]
        else 0.0
    )
    if truncation_rate > cfg.max_truncation_rate:
        alerts.append("rollout_truncation_gt_limit")
    if truncation_rate > 0.05:
        warnings.append("rollout_truncation_gt_5pct")
    if rollout_counts["parser_unsafe"]:
        warnings.append("parser_rejected_unsafe_or_ambiguous_surface")
    if series["zero_spread"] and series["zero_spread"][-1]["value"] > 0.70:
        alerts.append("zero_spread_gt_70pct")
    required = {
        "loss",
        "reward",
        "reward_std",
        "entropy",
        "clip",
        "grad_norm",
        "completion_length",
        "zero_spread",
        "parser_valid",
        "parser_unsafe",
    }
    missing_required = sorted(required & set(missing))
    if cfg.require_smoke_metrics and missing_required:
        alerts.append("missing_required_smoke_metrics")
    result = {
        "schema_version": 1,
        "trainer_state": str(state_files[-1]),
        "history_rows": len(history),
        "series": series,
        "rollouts": rollout_counts,
        "truncation_rate": truncation_rate,
        "max_truncation_rate": cfg.max_truncation_rate,
        "missing_metric_families": missing,
        "missing_required_smoke_metrics": missing_required,
        "nonfinite": nonfinite,
        "alerts": alerts,
        "warnings": warnings,
        "passed": not alerts,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if cfg.require_smoke_metrics and alerts:
        raise RuntimeError(f"telemetry smoke gate failed: {alerts}; see {output}")
    return result


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(summarize(parse(Config)), indent=2, sort_keys=True))
