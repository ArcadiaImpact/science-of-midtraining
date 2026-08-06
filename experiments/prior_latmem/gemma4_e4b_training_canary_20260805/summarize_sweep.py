"""Summarize and behavior-select the Gemma 4 E4B micro-fit checkpoint."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scimt.config import parse, save


@dataclass(frozen=True)
class SweepSummaryConfig:
    training_root: str = (
        "/workspace/caches/scimt-prior-latmem/gemma4_e4b_train_canary_20260805"
    )
    steps: tuple[int, ...] = (10, 20, 30, 40)

    def __post_init__(self) -> None:
        values = tuple(self.steps)
        if not values or tuple(sorted(set(values))) != values:
            raise ValueError("steps must be unique and strictly increasing")
        if values[-1] != 40:
            raise ValueError("the as-run sweep must end at final step 40")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _analysis_path(root: Path, step: int) -> Path:
    directory = root / "posttrain_eval" if step == 40 else root / "checkpoint_sweep" / f"step{step}"
    return directory / "posttrain_analysis.json"


def _select_checkpoint(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """Return the earliest checkpoint clearing capability and health gates."""
    eligible = [
        int(row["step"])
        for row in rows
        if row["gates"] == {
            "microfit_lift_ci_positive": True,
            "matched_specific_lift_ci_positive": True,
            "no_arm_truncation_regression": True,
        }
    ]
    return min(eligible) if eligible else None


def summarize(cfg: SweepSummaryConfig) -> dict[str, Any]:
    root = Path(cfg.training_root)
    rows: list[dict[str, Any]] = []
    baseline_health: dict[str, Any] | None = None
    for step in cfg.steps:
        path = _analysis_path(root, step)
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint analysis: {path}")
        analysis = json.loads(path.read_text())
        if int(analysis["adapter_step"]) != step:
            raise ValueError(f"{path}: adapter step does not match {step}")
        base = analysis["baseline_sample_health"]
        if baseline_health is None:
            baseline_health = base
        elif base != baseline_health:
            raise ValueError(f"{path}: baseline sample-health reference drifted")
        microfit = analysis["groups"]["microfit"]
        sentinel = analysis["groups"]["sentinel"]
        did = analysis["difference_in_differences"]["1"]
        post_health = analysis["sample_health"]
        base_groups = base["groups"]
        post_groups = post_health["groups"]
        rows.append(
            {
                "step": step,
                "epoch": analysis["training_curve"]["final_epoch"],
                "loss_last_over_first": analysis["training_curve"]["loss"][
                    "last_over_first"
                ],
                "microfit_correct": microfit["correct_samples"],
                "sentinel_correct": sentinel["correct_samples"],
                "microfit_pass1": microfit["coverage"]["1"],
                "sentinel_pass1": sentinel["coverage"]["1"],
                "pass1_difference_in_differences": did,
                "solved_at_16": {
                    "microfit": microfit["solved_at_16"],
                    "sentinel": sentinel["solved_at_16"],
                },
                "truncations": {
                    "base": {
                        name: int(base_groups[name]["finish_reason"].get("length", 0))
                        for name in ("microfit", "sentinel")
                    },
                    "post": {
                        name: int(post_groups[name]["finish_reason"].get("length", 0))
                        for name in ("microfit", "sentinel")
                    },
                },
                "gates": {
                    "microfit_lift_ci_positive": (
                        float(microfit["coverage"]["1"]["paired_delta"]["low"]) > 0
                    ),
                    "matched_specific_lift_ci_positive": float(did["low"]) > 0,
                    "no_arm_truncation_regression": all(
                        int(post_groups[name]["finish_reason"].get("length", 0))
                        <= int(base_groups[name]["finish_reason"].get("length", 0))
                        for name in ("microfit", "sentinel")
                    ),
                },
            }
        )
    result = {
        "schema_version": 1,
        "selection_rule": (
            "earliest checkpoint with positive 95% lower bounds for trained-arm "
            "pass@1 lift and matched-control-adjusted pass@1 lift, with no absolute "
            "truncation-count regression in either arm"
        ),
        "selected_step": _select_checkpoint(rows),
        "rows": rows,
    }
    _write_json(root / "checkpoint_sweep" / "summary.json", result)
    return result


def main() -> None:
    cfg = parse(SweepSummaryConfig)
    save(cfg, Path(cfg.training_root) / "checkpoint_sweep" / "config.yaml")
    summarize(cfg)


if __name__ == "__main__":
    main()


__all__ = ["SweepSummaryConfig", "summarize"]
