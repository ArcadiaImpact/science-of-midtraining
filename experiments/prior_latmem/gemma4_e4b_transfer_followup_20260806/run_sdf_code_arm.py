"""Run one parent-specific matched-lift code arm after the SDF chain.

The three invocations of this runner are intended to execute concurrently,
one per visible GPU.  Within an arm it freezes a parent baseline, trains and
persists exactly five LoRA checkpoints, follows the pre-registered directional
checkpoint search, confirms only the selected checkpoint, and opens the
reserved final set only after a successful matched-lift confirmation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806.persist_train import (
    PersistTrainConfig,
    run as persist_train,
)
from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806.gemma4_sampler_compat import (
    MANIFEST_NAME as SAMPLER_COMPAT_MANIFEST,
    ensure_sampler_compatible,
)
from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806.run_eval import (
    FollowupEvalConfig,
    run as run_eval,
)
from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806.run_parent_baseline import (
    ParentBaselineConfig,
    run as run_parent_baseline,
)
from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806.run_train import (
    FollowupTrainConfig,
    run as run_train,
)
from scimt.config import compose, parse, save


ARMS = ("control", "latency", "memory")
ATTRIBUTION_REPO = "sidbaines/scimt-prior-latmem-attribution"


@dataclass(frozen=True)
class SdfCodeArmConfig:
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf/code/control/workflow"
    )
    arm: str = "control"
    full_chain_marker: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf/"
        "full_parameter_complete.json"
    )
    selection_policy: str = (
        "experiments/prior_latmem/gemma4_e4b_transfer_followup_20260806/"
        "sdf_code_checkpoint_selection_policy.yaml"
    )
    parent_baseline_template: str = (
        "experiments/prior_latmem/configs/"
        "gemma4_e4b_sdf_control_parent_baseline_tuning_2026-08-06.yaml"
    )
    train_config: str = (
        "experiments/prior_latmem/configs/"
        "gemma4_e4b_sdf_control_code_train_2026-08-06.yaml"
    )
    persist_config: str = (
        "experiments/prior_latmem/configs/"
        "gemma4_e4b_sdf_control_code_persist_2026-08-06.yaml"
    )
    screen_template: str = (
        "experiments/prior_latmem/configs/"
        "gemma4_e4b_sdf_control_code_step64_screen_2026-08-06.yaml"
    )
    attribution_repo: str = ATTRIBUTION_REPO
    attribution_prefix: str = (
        "transfer_followup/20260806/sdf/control/code-r32-lr2e5/workflow"
    )
    # The validated inference and training stacks intentionally use different
    # Torch environments. run_sdf_code_driver supervises these three resumable
    # phases under the correct interpreter; this runner never crosses the env
    # boundary in-process.
    phase: str = "driver"  # driver | baseline | train | evaluate

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if self.attribution_repo != ATTRIBUTION_REPO:
            raise ValueError("matched-lift workflow repository is frozen")
        if self.phase not in {"driver", "baseline", "train", "evaluate"}:
            raise ValueError("phase must be driver, baseline, train, or evaluate")
        prefix = Path(self.attribution_prefix.strip("/"))
        if not str(prefix) or prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError("unsafe attribution_prefix")
        for value in (
            self.selection_policy,
            self.parent_baseline_template,
            self.train_config,
            self.persist_config,
            self.screen_template,
        ):
            if not value or not re.search(r"\.(?:yaml|yml)$", value):
                raise ValueError(f"expected a YAML config path, got {value!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_policy(path: Path) -> dict[str, Any]:
    import yaml

    policy = yaml.safe_load(path.read_text()) or {}
    if (
        policy.get("arms") != list(ARMS)
        or policy.get("candidate_steps") != [32, 64, 128, 192, 256]
        or policy.get("screen", {}).get("start_step") != 64
        or policy.get("screen", {}).get("samples_per_problem") != 4
        or policy.get("confirmation", {}).get("samples_per_problem") != 8
    ):
        raise ValueError("matched-lift checkpoint-selection policy drifted")
    return policy


def _screen_point(analysis: Mapping[str, Any]) -> dict[str, Any]:
    development = analysis["groups"]["development"]
    gates = analysis["gates"]
    return {
        "step": int(analysis["adapter_step"]),
        "lift": float(development["coverage"]["1"]["paired_delta"]["mean"]),
        "healthy": bool(gates["development_health_delta_at_most_2pp"]),
        "analysis_sha256": None,
    }


def screen_decision(
    observed: Mapping[int, Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the next deterministic action under the frozen directional rule."""

    target = float(policy["reference"]["pass1_lift"])
    tolerance = float(policy["screen"]["target_absolute_tolerance"])
    lower, upper = target - tolerance, target + tolerance

    def point(step: int) -> dict[str, Any]:
        raw = observed[step]
        return {
            "step": step,
            "lift": float(raw["lift"]),
            "healthy": bool(raw["healthy"]),
        }

    def in_band(item: Mapping[str, Any]) -> bool:
        return bool(item["healthy"]) and lower <= float(item["lift"]) <= upper

    def closest(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
        # Earlier checkpoint is the pre-registered exact-tie breaker.
        return int(
            min(
                (left, right),
                key=lambda item: (abs(float(item["lift"]) - target), int(item["step"])),
            )["step"]
        )

    if 64 not in observed:
        return {"action": "evaluate", "step": 64, "reason": "start_step"}
    start = point(64)
    if in_band(start):
        return {"action": "select", "step": 64, "reason": "start_in_band"}

    if float(start["lift"]) < lower:
        previous = start if start["healthy"] else None
        for step in (128, 192, 256):
            if step not in observed:
                return {"action": "evaluate", "step": step, "reason": "undershoot"}
            current = point(step)
            if in_band(current):
                return {"action": "select", "step": step, "reason": "first_in_band"}
            if (
                previous is not None
                and current["healthy"]
                and float(previous["lift"]) < lower
                and float(current["lift"]) > upper
            ):
                return {
                    "action": "select",
                    "step": closest(previous, current),
                    "reason": "healthy_bracket_closest",
                }
            if current["healthy"]:
                previous = current
        return {"action": "fail", "reason": "all_later_steps_below_or_unhealthy"}

    if float(start["lift"]) > upper:
        if 32 not in observed:
            return {"action": "evaluate", "step": 32, "reason": "overshoot"}
        early = point(32)
        if in_band(early):
            return {"action": "select", "step": 32, "reason": "early_in_band"}
        if (
            early["healthy"]
            and start["healthy"]
            and float(early["lift"]) < lower
            and float(start["lift"]) > upper
        ):
            return {
                "action": "select",
                "step": closest(early, start),
                "reason": "healthy_bracket_closest",
            }
        return {"action": "fail", "reason": "no_earlier_healthy_match"}

    # A point can be numerically in band yet fail the health requirement.
    return {"action": "fail", "reason": "in_band_but_unhealthy"}


def _validate_chain(cfg: SdfCodeArmConfig) -> dict[str, Any]:
    marker = Path(cfg.full_chain_marker)
    if not marker.is_file():
        raise FileNotFoundError(f"full-parameter chain marker is missing: {marker}")
    chain = json.loads(marker.read_text())
    arm = chain.get("arms", {}).get(cfg.arm, {})
    if (
        not chain.get("all_three_arms_complete")
        or not arm.get("sdf", {}).get("all_remote_verified")
        or not arm.get("reinstruct", {}).get("all_remote_verified")
        or not Path(str(arm.get("code_parent"))).is_dir()
    ):
        raise ValueError(f"full-parameter parent is incomplete for {cfg.arm}")
    return chain


def _complete_json(path: Path, *, fields: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if any(value.get(key) != expected for key, expected in fields.items()):
        raise ValueError(f"stale completion marker: {path}")
    return value


async def _ensure_parent_baseline(
    cfg: SdfCodeArmConfig, template: ParentBaselineConfig, *, mode: str
) -> ParentBaselineConfig:
    if mode == "tuning":
        concrete = template
    else:
        concrete = replace(
            template,
            out=str(Path(template.out).parent / "final"),
            mode="final",
            hf_prefix=template.hf_prefix.replace("parent-baseline-tuning-k8", "parent-baseline-final-k8"),
            attribution_prefix=template.attribution_prefix.rsplit("/", 1)[0] + "/final",
        )
    prepared = Path(concrete.out) / "prepared.json"
    if _complete_json(
        prepared,
        fields={"arm": cfg.arm, "mode": mode, "n_problems": 192 if mode == "tuning" else 294},
    ) is None:
        await run_parent_baseline(concrete)
    if not (Path(concrete.out) / "selection.json").is_file():
        raise FileNotFoundError(f"parent baseline selection missing for {cfg.arm}/{mode}")
    return concrete


async def _ensure_training(
    cfg: SdfCodeArmConfig,
    train_cfg: FollowupTrainConfig,
    persist_cfg: PersistTrainConfig,
) -> None:
    complete = Path(train_cfg.out) / "training_complete.json"
    cached = _complete_json(complete, fields={"arm": cfg.arm})
    if cached is None:
        await run_train(train_cfg)
    else:
        steps = [int(row["step"]) for row in cached.get("adapters", [])]
        if steps != list(train_cfg.checkpoint_steps):
            raise ValueError("cached code training does not contain exactly five checkpoints")
    marker = Path(persist_cfg.training_root) / "persistence.json"
    persisted = _complete_json(marker, fields={"arm": cfg.arm})
    if persisted is None or not persisted.get("marker_revision"):
        persist_train(persist_cfg)
    elif persisted.get("checkpoint_steps") != list(persist_cfg.checkpoint_steps):
        raise ValueError("cached code persistence does not contain exactly five checkpoints")
    _validate_training(cfg, train_cfg, persist_cfg)


def _validate_training(
    cfg: SdfCodeArmConfig,
    train_cfg: FollowupTrainConfig,
    persist_cfg: PersistTrainConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate, without importing a trainer, that the train phase is durable."""

    complete_path = Path(train_cfg.out) / "training_complete.json"
    complete = _complete_json(complete_path, fields={"arm": cfg.arm})
    if complete is None:
        raise FileNotFoundError(f"code training is incomplete: {complete_path}")
    steps = [int(row["step"]) for row in complete.get("adapters", [])]
    if steps != list(train_cfg.checkpoint_steps):
        raise ValueError("code training does not contain exactly five checkpoints")
    local_steps = sorted(
        int(path.name.rsplit("-", 1)[-1])
        for path in (Path(train_cfg.out) / "train" / "checkpoints").glob(
            "checkpoint-*"
        )
        if path.is_dir() and re.fullmatch(r"checkpoint-\d+", path.name)
    )
    if local_steps != list(train_cfg.checkpoint_steps):
        raise ValueError(
            f"local code checkpoint topology drifted: retained={local_steps}"
        )

    persistence_path = Path(persist_cfg.training_root) / "persistence.json"
    persisted = _complete_json(persistence_path, fields={"arm": cfg.arm})
    if persisted is None or not persisted.get("marker_revision"):
        raise FileNotFoundError(f"code persistence is incomplete: {persistence_path}")
    if persisted.get("checkpoint_steps") != list(persist_cfg.checkpoint_steps):
        raise ValueError("code persistence does not contain exactly five checkpoints")
    return complete, persisted


def _eval_config(
    template: FollowupEvalConfig,
    *,
    step: int,
    mode: str,
    final_shared_data: str | None = None,
) -> FollowupEvalConfig:
    if mode not in {"tune", "confirm", "final"}:
        raise ValueError(f"unsupported matched-lift eval mode: {mode}")
    samples = 4 if mode == "tune" else 8
    label = "screen" if mode == "tune" else mode
    prefix = template.hf_prefix.rsplit("/", 1)[0]
    return replace(
        template,
        out=str(Path(template.training_root) / "eval" / f"{label}_step{step}"),
        shared_data=final_shared_data or template.shared_data,
        mode=mode,
        adapter_step=step,
        hf_prefix=f"{prefix}/{label}-step{step}-k{samples}",
        n_samples=samples,
    )


async def _ensure_eval(eval_cfg: FollowupEvalConfig) -> dict[str, Any]:
    analysis_path = Path(eval_cfg.out) / "analysis.json"
    marker = Path(eval_cfg.out) / "persistence.json"
    analysis = _complete_json(
        analysis_path,
        fields={
            "arm": eval_cfg.arm,
            "mode": eval_cfg.mode,
            "adapter_step": eval_cfg.adapter_step,
        },
    )
    persisted = json.loads(marker.read_text()) if marker.is_file() else {}
    if analysis is None or not persisted.get("marker_revision"):
        await run_eval(eval_cfg)
        analysis = json.loads(analysis_path.read_text())
    analysis["analysis_sha256"] = _sha256(analysis_path)
    return analysis


def _confirmation_match(
    analysis: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    development = analysis["groups"]["development"]
    delta = development["coverage"]["1"]["paired_delta"]
    gates = analysis["gates"]
    target = float(policy["reference"]["pass1_lift"])
    tolerance = float(policy["confirmation"]["target_absolute_tolerance"])
    checks = {
        "lift_within_target_tolerance": abs(float(delta["mean"]) - target) <= tolerance,
        "pass1_ci95_lower_positive": float(delta["ci95_low"]) > 0.0,
        "health_delta_at_most_2pp": bool(gates["development_health_delta_at_most_2pp"]),
        "solved_at_8_not_lower": bool(gates["development_solved_at_8_not_lower"]),
    }
    return {
        "matched": all(checks.values()),
        "checks": checks,
        "pass1_lift": float(delta["mean"]),
        "pass1_ci95": [float(delta["ci95_low"]), float(delta["ci95_high"])],
        "target": target,
        "absolute_difference": abs(float(delta["mean"]) - target),
    }


def _persist_workflow(cfg: SdfCodeArmConfig, result: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    output = Path(cfg.out) / "workflow.json"
    _write_json(output, result)
    api = HfApi()
    api.create_repo(cfg.attribution_repo, repo_type="model", private=True, exist_ok=True)
    remote = f"{cfg.attribution_prefix.strip('/')}/workflow.json"
    uploaded_sha = _sha256(output)
    info = api.upload_file(
        path_or_fileobj=str(output),
        path_in_repo=remote,
        repo_id=cfg.attribution_repo,
        repo_type="model",
        commit_message=f"Persist {cfg.arm} matched-lift code workflow",
    )
    revision = str(info.oid)
    downloaded = Path(
        hf_hub_download(
            cfg.attribution_repo,
            remote,
            repo_type="model",
            revision=revision,
            local_dir=Path(cfg.out) / "remote_verification",
            force_download=True,
        )
    )
    if _sha256(downloaded) != uploaded_sha:
        raise ValueError("remote matched-lift workflow checksum mismatch")
    persistence = {
        "repo": cfg.attribution_repo,
        "remote_path": remote,
        "revision": revision,
        "sha256": uploaded_sha,
        "remote_verified": True,
    }
    _write_json(Path(cfg.out) / "persistence.json", persistence)
    return persistence


async def run(cfg: SdfCodeArmConfig) -> dict[str, Any]:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / f"config_{cfg.phase}.yaml")
    if cfg.phase == "driver":
        raise RuntimeError(
            "the code workflow spans incompatible inference/training Torch stacks; "
            "run it through run_sdf_code_driver.py"
        )

    chain = _validate_chain(cfg)
    parent_code_path = Path(chain["arms"][cfg.arm]["code_parent"])
    sampler_compat = ensure_sampler_compatible(parent_code_path)
    sampler_compat_path = parent_code_path / SAMPLER_COMPAT_MANIFEST
    policy_path = Path(cfg.selection_policy)
    policy = _load_policy(policy_path)

    parent_template = compose(ParentBaselineConfig, cfg.parent_baseline_template)
    train_cfg = compose(FollowupTrainConfig, cfg.train_config)
    persist_cfg = compose(PersistTrainConfig, cfg.persist_config)
    eval_template = compose(FollowupEvalConfig, cfg.screen_template)
    if any(item.arm != cfg.arm for item in (parent_template, train_cfg, persist_cfg, eval_template)):
        raise ValueError("subordinate config arm differs from workflow arm")
    if tuple(train_cfg.checkpoint_steps) != (32, 64, 128, 192, 256):
        raise ValueError("code arm must retain exactly five strategic checkpoints")

    common = {
        "schema_version": 1,
        "arm": cfg.arm,
        "phase": cfg.phase,
        "full_chain_marker": cfg.full_chain_marker,
        "full_chain_marker_sha256": _sha256(Path(cfg.full_chain_marker)),
        "parent_code_path": str(parent_code_path),
        "sampler_compatibility": sampler_compat,
        "sampler_compatibility_manifest_sha256": _sha256(sampler_compat_path),
    }
    if cfg.phase == "baseline":
        tuning_parent = await _ensure_parent_baseline(
            cfg, parent_template, mode="tuning"
        )
        result = {
            **common,
            "status": "complete",
            "parent_tuning_selection": str(
                Path(tuning_parent.out) / "selection.json"
            ),
            "parent_tuning_selection_sha256": _sha256(
                Path(tuning_parent.out) / "selection.json"
            ),
        }
        _write_json(out / "baseline_phase_complete.json", result)
        return result

    if cfg.phase == "train":
        await _ensure_training(cfg, train_cfg, persist_cfg)
        complete, persisted = _validate_training(cfg, train_cfg, persist_cfg)
        result = {
            **common,
            "status": "complete",
            "training_complete": str(Path(train_cfg.out) / "training_complete.json"),
            "training_complete_sha256": _sha256(
                Path(train_cfg.out) / "training_complete.json"
            ),
            "training_persistence": str(
                Path(persist_cfg.training_root) / "persistence.json"
            ),
            "training_persistence_sha256": _sha256(
                Path(persist_cfg.training_root) / "persistence.json"
            ),
            "checkpoint_steps": [int(row["step"]) for row in complete["adapters"]],
            "marker_revision": persisted["marker_revision"],
        }
        _write_json(out / "train_phase_complete.json", result)
        return result

    # Evaluation is deliberately isolated in the inference environment. These
    # calls are validation-only when the two prior phase markers are present.
    tuning_parent = await _ensure_parent_baseline(cfg, parent_template, mode="tuning")
    _validate_training(cfg, train_cfg, persist_cfg)

    observed: dict[int, dict[str, Any]] = {}
    decision: dict[str, Any]
    while True:
        decision = screen_decision(observed, policy)
        if decision["action"] != "evaluate":
            break
        step = int(decision["step"])
        analysis = await _ensure_eval(_eval_config(eval_template, step=step, mode="tune"))
        observed[step] = _screen_point(analysis)
        observed[step]["analysis_sha256"] = analysis["analysis_sha256"]

    result: dict[str, Any] = {
        **common,
        "status": "screen_failed" if decision["action"] == "fail" else "screen_selected",
        "parent_tuning_selection": str(Path(tuning_parent.out) / "selection.json"),
        "parent_tuning_selection_sha256": _sha256(Path(tuning_parent.out) / "selection.json"),
        "training_persistence": str(Path(persist_cfg.training_root) / "persistence.json"),
        "selection_policy": str(policy_path),
        "selection_policy_sha256": _sha256(policy_path),
        "screen_observations": {str(step): value for step, value in sorted(observed.items())},
        "screen_decision": decision,
        "five_code_checkpoints_persisted": True,
    }
    if decision["action"] == "fail":
        result["persistence"] = _persist_workflow(cfg, result)
        return result

    selected_step = int(decision["step"])
    confirmation = await _ensure_eval(
        _eval_config(eval_template, step=selected_step, mode="confirm")
    )
    matched = _confirmation_match(confirmation, policy)
    result.update(
        {
            "selected_step": selected_step,
            "confirmation_analysis_sha256": confirmation["analysis_sha256"],
            "confirmation": matched,
            "status": "matched" if matched["matched"] else "confirmation_failed",
        }
    )
    if not matched["matched"]:
        result["persistence"] = _persist_workflow(cfg, result)
        return result

    final_parent = await _ensure_parent_baseline(cfg, parent_template, mode="final")
    final_analysis = await _ensure_eval(
        _eval_config(
            eval_template,
            step=selected_step,
            mode="final",
            final_shared_data=final_parent.out,
        )
    )
    result.update(
        {
            "status": "complete",
            "parent_final_selection": str(Path(final_parent.out) / "selection.json"),
            "parent_final_selection_sha256": _sha256(Path(final_parent.out) / "selection.json"),
            "final_analysis": str(Path(eval_template.training_root) / "eval" / f"final_step{selected_step}" / "analysis.json"),
            "final_analysis_sha256": final_analysis["analysis_sha256"],
            "final_gates": final_analysis["gates"],
        }
    )
    result["persistence"] = _persist_workflow(cfg, result)
    return result


def main() -> None:
    asyncio.run(run(parse(SdfCodeArmConfig)))


if __name__ == "__main__":
    main()


__all__ = ["SdfCodeArmConfig", "run", "screen_decision"]
