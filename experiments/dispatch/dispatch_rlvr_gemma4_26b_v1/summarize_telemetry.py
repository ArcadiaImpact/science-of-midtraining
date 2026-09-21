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
    #: Also require the post-selection zero-spread series. Set for oversampled
    #: cells; a run without within-batch selection never emits it and must not
    #: be failed for its absence.
    require_selection_metrics: bool = False

    def __post_init__(self) -> None:
        if not self.cell_dir or not self.output:
            raise ValueError("cell_dir and output are required")
        if not 0 < self.max_truncation_rate <= 1:
            raise ValueError("max_truncation_rate must be in (0, 1]")


# Each family is (alternatives, excluded). A key joins the family when it
# matches ANY alternative -- itself a conjunction of substrings -- and NONE of
# the excluded substrings.
#
# The original spelling was a bare tuple read as a conjunction, while
# `reward_std` had been written as if a tuple meant ALTERNATIVES:
# ("reward_std", "reward/std"), which NO key can satisfy, one spelling having
# an underscore where the other has a slash. Since `reward_std` is `required`,
# the gate failed on every run -- charter-direct's phase-16 died 2026-09-02 with
# missing_required=['reward_std'] though TRL 1.9.2 had logged both `reward_std`
# and `rewards/reward_func/std`.
#
# The first repair, ("reward", "std"), unblocked the gate but silently swallowed
# the neighbours: TRL also logs `frac_reward_zero_std` and
# `reward/zero_std_group_fraction`, and BOTH contain "reward" and "std", so the
# reward_std series became a mix of standard deviations and zero-spread
# fractions -- a number between 0 and 1 either way, which is exactly how a
# reader fails to notice. Hence explicit alternatives and an exclusion list:
# expressing "these spellings, not those" needs both.
FAMILIES = {
    "loss": ((("loss",),), ()),
    # Mean reward only. Everything TRL logs about rewards contains "reward",
    # the spread diagnostics included.
    "reward": ((("reward",),), ("std", "zero", "frac", "component")),
    # True ALTERNATIVES, which is what the original author wanted and the old
    # conjunction could not express: TRL logs `reward_std` AND
    # `rewards/reward_func/std`, one with an underscore where the other has a
    # slash. A plain ("reward", "std") conjunction unblocks the gate but also
    # swallows `frac_reward_zero_std` and `reward/zero_std_group_fraction` --
    # both contain "reward" and "std" -- so a spread FRACTION lands in the
    # standard-deviation series. Both are numbers in [0, 1], which is exactly
    # how that goes unnoticed.
    "reward_std": ((("reward_std",), ("rewards/reward_func/std",)),
                   ("zero", "frac_")),
    "entropy": ((("entropy",),), ()),
    "kl": ((("kl",),), ()),
    "clip": ((("clip",),), ()),
    "grad_norm": ((("grad_norm",),), ()),
    "completion_length": ((("completion", "length"),), ()),
    # PRE-selection: the rate over every GENERATED group. This is the series the
    # abort gate fires on and the one the phase-16 baselines mean, so it keeps
    # its name and its definition. Within-batch selection keeps the best 4 of 8
    # whatever the policy is doing, so a gate reading the post-selection rate
    # would look healthy through the exact collapse it exists to catch. The
    # "selected" exclusion is load-bearing: the post-selection key contains
    # every term of this one.
    "zero_spread": ((("zero_std_group_fraction",),), ("selected",)),
    # POST-selection: what the optimizer actually saw. Reported, never gated.
    "selected_zero_spread": ((("selected", "zero_std_group_fraction"),), ()),
    "parser_valid": ((("reward_components/parser_valid",),), ()),
    "parser_unsafe": ((("reward_components/parser_unsafe",),), ()),
}


def _matches(
    key: str, spec: tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]
) -> bool:
    """True when `key` matches ANY alternative and NONE of the exclusions."""
    alternatives, excluded = spec
    lowered = key.casefold()
    if any(term in lowered for term in excluded):
        return False
    return any(all(term in lowered for term in terms) for terms in alternatives)


def _selection_summary(paths: list[Path]) -> dict[str, Any] | None:
    """Realized within-batch selection, from the append-only audit trail.

    The kept subset depends on sampled completions, and vLLM does not reproduce
    those across a process restart, so this trail is how a run says which
    groups it actually optimized. Rows join to ``raw_rollouts.jsonl`` on
    ``reward_call``, which carries ``episode_id``.
    """

    rounds = 0
    generated_groups = 0
    kept_groups = 0
    generated_zero = 0
    kept_zero = 0
    filled_from_zero = 0
    short_rounds = 0
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            generated = int(row["generated_groups"])
            kept = list(row["kept_groups"])
            scores = list(row["group_scores"])
            rounds += 1
            generated_groups += generated
            kept_groups += len(kept)
            generated_zero += round(float(row["zero_std_fraction"]) * generated)
            kept_zero += round(float(row["selected_zero_std_fraction"]) * len(kept))
            # Slots that had to be filled with a zero-spread group because the
            # batch did not offer enough informative ones. We never regenerate,
            # so this is the measured cost of that choice.
            empty = sum(scores[index] == 0.0 for index in kept)
            filled_from_zero += empty
            short_rounds += bool(empty)
    if not rounds:
        return None
    return {
        "rounds": rounds,
        "generated_groups": generated_groups,
        "optimized_groups": kept_groups,
        "generated_zero_std_fraction": generated_zero / generated_groups,
        "optimized_zero_std_fraction": kept_zero / kept_groups,
        "optimized_slots_filled_with_zero_spread": filled_from_zero,
        "rounds_short_of_informative_groups": short_rounds,
        "rounds_short_fraction": short_rounds / rounds,
    }


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
            for family, spec in FAMILIES.items():
                if _matches(str(key), spec):
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
    # PRE-selection, deliberately: see the zero_spread entry in FAMILIES.
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
    if cfg.require_selection_metrics:
        required.add("selected_zero_spread")
    missing_required = sorted(required & set(missing))
    if cfg.require_smoke_metrics and missing_required:
        alerts.append("missing_required_smoke_metrics")
    selection_files = sorted(root.glob("rollouts/selection.rank-*.jsonl"))
    selection = _selection_summary(selection_files)
    result = {
        "schema_version": 2,
        "trainer_state": str(state_files[-1]),
        "history_rows": len(history),
        "series": series,
        "rollouts": rollout_counts,
        # Both rates, always, and never one in place of the other: the
        # pre-selection number is comparable to runs without selection, the
        # post-selection number is what the optimizer saw.
        "selection": selection,
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
    from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(summarize(parse(Config)), indent=2, sort_keys=True))
