"""Behavior-select transfer checkpoints and compare the two target arms."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scimt.config import parse, save


@dataclass(frozen=True)
class SummaryConfig:
    concise_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_canary_20260805/concise"
    )
    complete_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_canary_20260805/complete"
    )
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_canary_20260805/comparison"
    )
    steps: tuple[int, ...] = (16, 32, 64)

    def __post_init__(self) -> None:
        if tuple(self.steps) != (16, 32, 64):
            raise ValueError("the as-run screen requires checkpoints 16/32/64")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _screen_path(root: Path, step: int) -> Path:
    return root / "eval" / f"screen_step{step}" / "analysis.json"


def _confirm_path(root: Path, step: int) -> Path:
    return root / "eval" / f"confirm_step{step}" / "analysis.json"


def _select_checkpoint(rows: Sequence[Mapping[str, Any]]) -> int | None:
    eligible = [row for row in rows if row["eligible_for_confirmation"]]
    if not eligible:
        return None
    # Development performance is the primary selection objective. Earliest
    # exposure breaks an exact tie, limiting unnecessary specialization.
    chosen = min(
        eligible,
        key=lambda row: (-float(row["development_pass1_post"]), int(row["step"])),
    )
    return int(chosen["step"])


def _screen_row(analysis: Mapping[str, Any]) -> dict[str, Any]:
    train = analysis["groups"]["train"]["coverage"]["1"]
    development = analysis["groups"]["development"]["coverage"]["1"]
    return {
        "step": int(analysis["adapter_step"]),
        "epoch": analysis["training_curve"]["final_epoch"],
        "loss_last_over_first": analysis["training_curve"]["loss"][
            "last_over_first"
        ],
        "train_pass1_base": train["base_mean"],
        "train_pass1_post": train["post_mean"],
        "train_pass1_delta": train["paired_delta"],
        "development_pass1_base": development["base_mean"],
        "development_pass1_post": development["post_mean"],
        "development_pass1_delta": development["paired_delta"],
        "development_adverse_rate_delta": analysis["sample_health"][
            "development"
        ]["adverse_rate_delta"],
        "eligible_for_confirmation": bool(
            analysis["gates"]["eligible_for_confirmation"]
        ),
        "gates": analysis["gates"],
    }


def _confirmation_row(analysis: Mapping[str, Any]) -> dict[str, Any]:
    development = analysis["groups"]["development"]
    return {
        "step": int(analysis["adapter_step"]),
        "development_pass1": development["coverage"]["1"],
        "development_pass4": development["coverage"]["4"],
        "development_pass8": development["coverage"]["8"],
        "development_solved_at_8": development["solved_at_full_budget"],
        "development_adverse_rate_delta": analysis["sample_health"][
            "development"
        ]["adverse_rate_delta"],
        "strata": {
            stratum: analysis["groups"][f"development_{stratum}"]["coverage"]
            for stratum in ("zero", "frontier", "moderate")
        },
        "gates": analysis["gates"],
        "transfer_gate_passed": bool(analysis["gates"]["transfer_gate_passed"]),
    }


def summarize(cfg: SummaryConfig) -> dict[str, Any]:
    roots = {"concise": Path(cfg.concise_root), "complete": Path(cfg.complete_root)}
    arms: dict[str, Any] = {}
    selection_sha: str | None = None
    for arm, root in roots.items():
        screen_rows: list[dict[str, Any]] = []
        for step in cfg.steps:
            path = _screen_path(root, step)
            if not path.is_file():
                raise FileNotFoundError(f"missing screen analysis: {path}")
            analysis = json.loads(path.read_text())
            if analysis.get("arm") != arm or analysis.get("mode") != "screen":
                raise ValueError(f"screen identity drift: {path}")
            if int(analysis["adapter_step"]) != step:
                raise ValueError(f"screen step drift: {path}")
            observed_sha = str(analysis["selection_sha256"])
            if selection_sha is None:
                selection_sha = observed_sha
            elif observed_sha != selection_sha:
                raise ValueError("two arms do not share the same frozen selection")
            screen_rows.append(_screen_row(analysis))
        selected = _select_checkpoint(screen_rows)
        arm_result: dict[str, Any] = {
            "screens": screen_rows,
            "selected_step": selected,
        }
        if selected is not None and _confirm_path(root, selected).is_file():
            confirmation = json.loads(_confirm_path(root, selected).read_text())
            if (
                confirmation.get("arm") != arm
                or confirmation.get("mode") != "confirm"
                or int(confirmation["adapter_step"]) != selected
            ):
                raise ValueError(f"confirmation identity drift for {arm}")
            arm_result["confirmation"] = _confirmation_row(confirmation)
        arms[arm] = arm_result

    confirmed = {
        arm: row["confirmation"]
        for arm, row in arms.items()
        if "confirmation" in row
    }
    successful = [
        arm for arm, row in confirmed.items() if row["transfer_gate_passed"]
    ]
    recommended_arm: str | None = None
    if successful:
        recommended_arm = min(
            successful,
            key=lambda arm: (
                -float(
                    confirmed[arm]["development_pass1"]["paired_delta"]["mean"]
                ),
                float(confirmed[arm]["development_adverse_rate_delta"]),
                int(confirmed[arm]["step"]),
            ),
        )
    result = {
        "schema_version": 1,
        "selection_sha256": selection_sha,
        "screen_selection_rule": (
            "among checkpoints with >=10pp trained-task pass@1 lift and <=2pp "
            "development adverse-rate regression, maximize development pass@1; "
            "break exact ties in favor of the earliest step"
        ),
        "confirmation_gate": (
            ">=3pp development pass@1 lift, positive problem-bootstrap 95% lower "
            "bound, <=2pp adverse-rate regression, and solved@8 not lower"
        ),
        "arms": arms,
        "confirmed_arms": sorted(confirmed),
        "successful_arms": sorted(successful),
        "recommended_arm": recommended_arm,
    }
    _write_json(Path(cfg.out) / "summary.json", result)
    return result


def main() -> None:
    cfg = parse(SummaryConfig)
    save(cfg, Path(cfg.out) / "config.yaml")
    summarize(cfg)


if __name__ == "__main__":
    main()


__all__ = ["SummaryConfig", "summarize"]
