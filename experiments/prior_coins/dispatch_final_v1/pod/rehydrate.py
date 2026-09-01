"""Recover Dispatch final-v1 resume state from stage commits on the Hub.

The chain publishes a stage with one atomic Hub commit *after* that phase's
completion sentinel lands.  Therefore a valid ``<prefix>/<stage>/`` tree is
durable evidence that the phase finished.  This implication is the foundation
of recovery; the validators below reject a tree that does not have the shape
the corresponding phase produces.

Only bytes needed by the first phase that will run are restored.  The current
``chain.py`` control flow gives this table (``M``/``D`` are final full-model
checkpoints, ``A`` is AFT completion markers, and ``A*`` is selected adapters):

====================  ========================================================
first phase to run    bytes restored
====================  ========================================================
mix                   none
midtrain              the published ``data/`` tree (the realized mix)
dolci                 M
aft                    M + D
eval                   M + D + A + every cell's step-256/512 adapters
recall                 M + D + A + agreement step-256/512 adapters
d4 or costsweep        M + D + A + every cell's step-256/512 adapters
publish                M + D + A (chain traversal needs the parents/markers)
====================  ========================================================

The M checkpoint is needed even after Dolci because ``chain.main`` calls
``final_checkpoint(midtrain_dir, ...)`` before it learns that Dolci will be
skipped.  Published result trees themselves are not downloaded: no later phase
reads them, and a receipt prevents the chain from uploading them again.

Run on a pod before the chain:

    python3 rehydrate.py --arms charter,coin,control --root /workspace/final_v1
"""

from __future__ import annotations

import argparse
import asyncio
import filecmp
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

POD = Path(__file__).resolve().parent
EXP = POD.parent
PRIOR_COINS = EXP.parent
REPO_ROOT = PRIOR_COINS.parents[1]
for _path in (
    str(REPO_ROOT),
    str(REPO_ROOT / "src"),
    str(EXP),
    str(PRIOR_COINS),
    str(POD),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import chain  # noqa: E402
import contracts as C  # noqa: E402
from publish_stage import REPO  # noqa: E402


STAGES = ("data", "midtrain", "dolci", "aft", "eval", "recall", "d4", "costsweep")
RESULT_STAGES = ("eval", "recall", "d4", "costsweep")
ROOT_SENTINELS = {
    "mix": "MIX_COMPLETE.json",
    "midtrain": "MIDTRAIN_COMPLETE.json",
    "dolci": "DOLCI_COMPLETE.json",
    "eval": "EVAL_COMPLETE.json",
    "recall": "RECALL_COMPLETE.json",
    "d4": "D4_COMPLETE.json",
    "costsweep": "COSTSWEEP_COMPLETE.json",
}

#: CHAIN_COMPLETE means "this arm is finished and the pod is safe to destroy",
#: and PUBLISH_COMPLETE gates the final sweep-up. Neither may EVER be
#: reconstructed here: a fabricated CHAIN_COMPLETE would license tearing down a
#: pod holding the only copy of unpublished work. The assertion exists so that
#: adding a key to ROOT_SENTINELS cannot quietly make that reachable -- the
#: import fails instead.
_NEVER_REHYDRATE = frozenset({"chain", "publish"})
assert not (_NEVER_REHYDRATE & set(ROOT_SENTINELS)), (
    "ROOT_SENTINELS must never carry a chain/publish sentinel: rehydrating "
    f"one would forge run completion (found {sorted(_NEVER_REHYDRATE & set(ROOT_SENTINELS))})"
)
assert not any(
    name.startswith(("CHAIN_", "PUBLISH_")) for name in ROOT_SENTINELS.values()
), "ROOT_SENTINELS maps to a CHAIN_/PUBLISH_ marker; see _NEVER_REHYDRATE"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


@dataclass(frozen=True)
class RehydrateConfig:
    root: Path
    arms: tuple[str, ...]
    repo: str = REPO


@dataclass(frozen=True)
class RemoteFile:
    repo_path: str
    stage: str
    relative: str
    size: int


@dataclass(frozen=True)
class ArmPlan:
    arm: str
    prefix: str
    found_stages: tuple[str, ...]
    first_phase: str
    selected: tuple[RemoteFile, ...]
    # Each scope is either one selected file or a directory whose local file
    # set must be a subset of the Hub set.  Extra local bytes are a conflict,
    # not something recovery is allowed to delete or reinterpret.
    scopes: tuple[str, ...]


def _fail(arm: str, stage: str, detail: str) -> RuntimeError:
    return RuntimeError(
        f"{arm}/{stage}: Hub state is inconsistent: {detail}. Refusing to "
        f"guess completion; inspect the {C.hub_arm_prefix(arm)}/{stage}/ tree "
        f"in {REPO}."
    )


def _stage_map(files: Iterable[RemoteFile]) -> dict[str, RemoteFile]:
    return {item.relative: item for item in files}


def _require_file(
    arm: str, stage: str, files: dict[str, RemoteFile], name: str
) -> None:
    item = files.get(name)
    if item is None:
        raise _fail(arm, stage, f"missing required file {name}")
    if item.size <= 0:
        raise _fail(arm, stage, f"required file {name} has size {item.size}")


def _checkpoint_steps(files: Iterable[RemoteFile]) -> set[int]:
    return {
        int(match.group(1))
        for item in files
        if (match := re.match(r"checkpoints/checkpoint-(\d+)/", item.relative))
    }


def _validate_full_checkpoint(
    arm: str, stage: str, files: dict[str, RemoteFile], step: int
) -> None:
    base = f"checkpoints/checkpoint-{step}/"
    _require_file(arm, stage, files, base + "config.json")
    _require_file(arm, stage, files, base + "tokenizer_config.json")
    weights = [
        item
        for name, item in files.items()
        if name.startswith(base)
        and (name.endswith(".safetensors") or name.endswith(".bin"))
        and Path(name).name.startswith(("model", "pytorch_model"))
        and not name.endswith(".index.json")
    ]
    if not weights or any(item.size <= 0 for item in weights):
        raise _fail(arm, stage, f"checkpoint-{step} has no non-empty model weights")


def _validate_adapter_checkpoint(
    arm: str, files: dict[str, RemoteFile], cell: str, step: int
) -> None:
    base = f"{cell}/checkpoints/checkpoint-{step}/"
    _require_file(arm, "aft", files, base + "adapter_config.json")
    weights = [
        item
        for name, item in files.items()
        if name.startswith(base)
        and (name.endswith(".safetensors") or name.endswith(".bin"))
        and "adapter_model" in Path(name).name
    ]
    if not weights or any(item.size <= 0 for item in weights):
        raise _fail(
            arm, "aft", f"{cell}/checkpoint-{step} has no non-empty adapter weights"
        )


def _endpoint_names() -> tuple[str, ...]:
    return (
        "pre_aft",
        *(f"{cell}-step{step}" for cell in C.AFT_CELLS for step in C.AFT_EVAL_STEPS),
    )


def validate_stage(arm: str, stage: str, remote: tuple[RemoteFile, ...]) -> None:
    """Prove that a visible prefix has the minimum shape its phase emits."""
    files = _stage_map(remote)
    if not files:
        raise _fail(arm, stage, "the prefix contains no files")
    if len(files) != len(remote):
        raise _fail(arm, stage, "the repo listing contains duplicate file names")

    if stage == "data":
        _require_file(arm, stage, files, "leg_a.jsonl")
        _require_file(arm, stage, files, "leg_a.jsonl.manifest.json")
        return

    if stage == "midtrain":
        observed = _checkpoint_steps(remote)
        expected = set(C.MIDTRAIN_CHECKPOINT_STEPS)
        if observed != expected:
            raise _fail(
                arm,
                stage,
                f"checkpoint steps are {sorted(observed)}, expected {sorted(expected)}",
            )
        for step in expected:
            _validate_full_checkpoint(arm, stage, files, step)
        return

    if stage == "dolci":
        expected = (
            set(C.DOLCI_CHECKPOINT_STEPS_CONTROL)
            if arm == "control"
            else {C.DOLCI_STEPS}
        )
        observed = _checkpoint_steps(remote)
        if observed != expected:
            raise _fail(
                arm,
                stage,
                f"checkpoint steps are {sorted(observed)}, expected {sorted(expected)}",
            )
        for step in expected:
            _validate_full_checkpoint(arm, stage, files, step)
        return

    if stage == "aft":
        expected_steps = set(C.AFT_CHECKPOINT_STEPS)
        for cell in C.AFT_CELLS:
            _require_file(arm, stage, files, f"{cell}/AFT_COMPLETE.json")
            observed = {
                int(match.group(1))
                for name in files
                if (
                    match := re.match(
                        rf"{re.escape(cell)}/checkpoints/checkpoint-(\d+)/", name
                    )
                )
            }
            if observed != expected_steps:
                raise _fail(
                    arm,
                    stage,
                    f"{cell} checkpoint steps are {sorted(observed)}, "
                    f"expected {sorted(expected_steps)}",
                )
            for step in expected_steps:
                _validate_adapter_checkpoint(arm, files, cell, step)
        return

    if stage == "eval":
        expected_count = len(C.EVAL_SLICES) * len(C.EVAL_SURFACES)
        for endpoint in _endpoint_names():
            outputs = [
                item
                for name, item in files.items()
                if name.startswith(f"{endpoint}/")
                and "/" not in name[len(endpoint) + 1 :]
                and "__" in Path(name).name
                and name.endswith(".jsonl")
            ]
            if len(outputs) != expected_count or any(
                item.size <= 0 for item in outputs
            ):
                raise _fail(
                    arm,
                    stage,
                    f"{endpoint} has {len(outputs)}/{expected_count} "
                    "non-empty prompt-set outputs",
                )
        return

    if stage == "recall":
        endpoints = (
            f"midtrain_{C.MIDTRAIN_STEPS}",
            "pre_aft",
            *(f"aft_{step}" for step in C.AFT_EVAL_STEPS),
        )
        outputs = (
            "recall_forced_choice_logprob.jsonl",
            "recall_forced_choice_gen.jsonl",
            "recall_freeform.jsonl",
        )
        for endpoint in endpoints:
            _require_file(arm, stage, files, f"{endpoint}/RECALL_COMPLETE.json")
            for name in outputs:
                _require_file(arm, stage, files, f"{endpoint}/{name}")
        return

    if stage == "d4":
        for endpoint in _endpoint_names():
            _require_file(arm, stage, files, f"{endpoint}/D4_COMPLETE.json")
            _require_file(arm, stage, files, f"{endpoint}/d4_logprob.jsonl")
            _require_file(arm, stage, files, f"{endpoint}/d4_gen.jsonl")
        return

    if stage == "costsweep":
        for endpoint in _endpoint_names():
            _require_file(arm, stage, files, f"{endpoint}/COSTSWEEP_COMPLETE.json")
            _require_file(arm, stage, files, f"{endpoint}/responses.jsonl")
        return

    raise ValueError(f"unknown stage {stage!r}")


def discover(
    siblings: Iterable[object], arms: tuple[str, ...]
) -> dict[str, dict[str, tuple[RemoteFile, ...]]]:
    """Group one metadata-rich repo listing by arm and stage."""
    grouped: dict[str, dict[str, list[RemoteFile]]] = {
        arm: {stage: [] for stage in STAGES} for arm in arms
    }
    for sibling in siblings:
        repo_path = getattr(sibling, "rfilename", None)
        if not isinstance(repo_path, str):
            continue
        for arm in arms:
            prefix = C.hub_arm_prefix(arm) + "/"
            if not repo_path.startswith(prefix):
                continue
            rest = repo_path[len(prefix) :]
            stage, separator, relative = rest.partition("/")
            if not separator or stage not in STAGES:
                continue
            size = getattr(sibling, "size", None)
            if not isinstance(size, int) or size < 0:
                raise RuntimeError(
                    f"{repo_path}: Hub listing supplied no usable file size; "
                    "repo_info(..., files_metadata=True) is required"
                )
            grouped[arm][stage].append(RemoteFile(repo_path, stage, relative, size))

    result: dict[str, dict[str, tuple[RemoteFile, ...]]] = {}
    for arm, by_stage in grouped.items():
        result[arm] = {
            stage: tuple(sorted(files, key=lambda item: item.repo_path))
            for stage, files in by_stage.items()
            if files
        }
    return result


def first_incomplete(arm: str, found: set[str]) -> str:
    """First phase whose durable parent tree is absent.

    Data and midtrain upload concurrently, so a durable midtrain checkpoint is
    allowed to subsume an absent data upload.  Later weight descendants are not
    allowed to cross a missing parent: re-running that parent and trusting a
    child trained from vanished bytes would silently guess lineage.
    """
    if "midtrain" not in found:
        descendants = found - {"data"}
        if descendants:
            raise RuntimeError(
                f"{arm}: Hub has descendant stages {sorted(descendants)} but no "
                "durable midtrain checkpoint. The current chain cannot traverse "
                "this state safely; recover the missing stage upload or use a "
                "fresh run root."
            )
        return "midtrain" if "data" in found else "mix"
    if "dolci" not in found:
        descendants = found & ({"aft"} | set(RESULT_STAGES))
        if descendants:
            raise RuntimeError(
                f"{arm}: Hub has {sorted(descendants)} but no durable Dolci "
                "parent; refusing to pair those children with a re-run parent."
            )
        return "dolci"
    if "aft" not in found:
        descendants = found & set(RESULT_STAGES)
        if descendants:
            raise RuntimeError(
                f"{arm}: Hub has result stages {sorted(descendants)} but no "
                "durable AFT adapters; refusing to regenerate their parent."
            )
        return "aft"
    for stage in RESULT_STAGES:
        if stage not in found:
            return stage
    return "publish"


def _under(files: tuple[RemoteFile, ...], relative_prefix: str) -> list[RemoteFile]:
    prefix = relative_prefix.rstrip("/") + "/"
    return [item for item in files if item.relative.startswith(prefix)]


def build_plan(arm: str, stages: dict[str, tuple[RemoteFile, ...]]) -> ArmPlan:
    found = set(stages)
    for stage, files in stages.items():
        validate_stage(arm, stage, files)
    first = first_incomplete(arm, found)
    selected: list[RemoteFile] = []
    scopes: list[str] = []

    if first == "midtrain":
        selected.extend(stages["data"])
        scopes.append("data")

    phases_after_midtrain = {
        "dolci",
        "aft",
        "eval",
        "recall",
        "d4",
        "costsweep",
        "publish",
    }
    if "midtrain" in found and first in phases_after_midtrain:
        relative = f"checkpoints/checkpoint-{C.MIDTRAIN_STEPS}"
        selected.extend(_under(stages["midtrain"], relative))
        scopes.append(f"midtrain/{relative}")

    phases_after_dolci = {"aft", "eval", "recall", "d4", "costsweep", "publish"}
    if "dolci" in found and first in phases_after_dolci:
        relative = f"checkpoints/checkpoint-{C.DOLCI_STEPS}"
        selected.extend(_under(stages["dolci"], relative))
        scopes.append(f"dolci/{relative}")

    if "aft" in found and first in set(RESULT_STAGES) | {"publish"}:
        for cell in C.AFT_CELLS:
            marker = f"{cell}/AFT_COMPLETE.json"
            selected.append(_stage_map(stages["aft"])[marker])
            scopes.append(f"aft/{marker}")

        needed_cells: tuple[str, ...] = ()
        if first in {"eval", "d4", "costsweep"}:
            needed_cells = tuple(C.AFT_CELLS)
        elif first == "recall":
            needed_cells = ("agreement",)
        for cell in needed_cells:
            for step in C.AFT_EVAL_STEPS:
                relative = f"{cell}/checkpoints/checkpoint-{step}"
                selected.extend(_under(stages["aft"], relative))
                scopes.append(f"aft/{relative}")

    unique = {item.repo_path: item for item in selected}
    return ArmPlan(
        arm=arm,
        prefix=C.hub_arm_prefix(arm),
        found_stages=tuple(stage for stage in STAGES if stage in found),
        first_phase=first,
        selected=tuple(sorted(unique.values(), key=lambda item: item.repo_path)),
        scopes=tuple(scopes),
    )


def _local_relative(item: RemoteFile) -> Path:
    return Path(item.stage) / item.relative


def _scope_expected(plan: ArmPlan, scope: str) -> set[Path]:
    root = Path(scope)
    return {
        _local_relative(item)
        for item in plan.selected
        if _local_relative(item) == root or root in _local_relative(item).parents
    }


#: Processor metadata that evaluate.py's ensure_processor_files BACKFILLS into
#: local checkpoint dirs at eval time (byte-for-byte copies from the pinned
#: base snapshot; vLLM needs them, `save_only_model: true` doesn't write them,
#: and the stage publishes happen BEFORE the backfill so the Hub tree lacks
#: them). They are derived, re-created on demand, and must not make an
#: otherwise Hub-consistent checkpoint look diverged: that parked the 4b_1m
#: row on 2026-09-01 after a relaunch-after-evals.
_EVAL_BACKFILL_FILES = frozenset({
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
})


def _check_local_scopes(run_root: Path, plan: ArmPlan) -> None:
    for scope in plan.scopes:
        local = run_root / scope
        expected = _scope_expected(plan, scope)
        if local.is_symlink():
            raise RuntimeError(f"{local}: local recovery target is a symlink; refusing")
        if local.is_file():
            if Path(scope) not in expected:
                raise RuntimeError(
                    f"{local}: file exists where a directory is expected"
                )
            continue
        if not local.exists():
            continue
        if not local.is_dir():
            raise RuntimeError(f"{local}: unsupported local object; refusing")
        observed = {
            path.relative_to(run_root) for path in local.rglob("*") if path.is_file()
        }
        extra = {
            path for path in observed - expected
            if path.name not in _EVAL_BACKFILL_FILES
        }
        if extra:
            sample = ", ".join(str(path) for path in sorted(extra)[:5])
            raise RuntimeError(
                f"{local}: local state has {len(extra)} file(s) absent from the "
                f"selected Hub tree ({sample}); refusing to delete or overwrite it"
            )


def _install_download(run_root: Path, staging: Path, plan: ArmPlan) -> dict:
    _check_local_scopes(run_root, plan)
    installed_files = 0
    installed_bytes = 0
    already_local_bytes = 0
    for item in plan.selected:
        source = staging / item.repo_path
        if not source.is_file():
            raise RuntimeError(
                f"snapshot_download omitted {item.repo_path}; expected "
                f"{len(plan.selected)} files from its allow_patterns filter"
            )
        got = source.stat().st_size
        if got != item.size:
            raise RuntimeError(
                f"{item.repo_path}: downloaded size {got} != Hub listing {item.size}"
            )

        destination = run_root / _local_relative(item)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.stat().st_size != item.size
                or not filecmp.cmp(source, destination, shallow=False)
            ):
                local_size = (
                    destination.stat().st_size if destination.is_file() else "non-file"
                )
                raise RuntimeError(
                    f"{destination}: local bytes disagree with Hub "
                    f"({local_size} vs {item.size} bytes); refusing to overwrite "
                    "possibly newer local state"
                )
            already_local_bytes += item.size
            continue
        os.replace(source, destination)
        installed_files += 1
        installed_bytes += item.size

    verified = [
        item
        for item in plan.selected
        if (run_root / _local_relative(item)).is_file()
        and (run_root / _local_relative(item)).stat().st_size == item.size
    ]
    if len(verified) != len(plan.selected):
        raise RuntimeError(
            f"{plan.arm}: only {len(verified)}/{len(plan.selected)} selected Hub "
            "files verify locally after installation"
        )
    return {
        "files": len(plan.selected),
        "bytes": sum(item.size for item in plan.selected),
        "installed_files": installed_files,
        "installed_bytes": installed_bytes,
        "already_local_bytes": already_local_bytes,
        "local_paths": [str(_local_relative(item)) for item in plan.selected],
    }


async def restore_bytes(
    config: RehydrateConfig, revision: str, run_root: Path, plan: ArmPlan
) -> dict:
    if not plan.selected:
        return {
            "files": 0,
            "bytes": 0,
            "installed_files": 0,
            "installed_bytes": 0,
            "already_local_bytes": 0,
            "local_paths": [],
        }
    from huggingface_hub import hf_hub_download

    config.root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".rehydrate-", dir=config.root))
    try:
        # Exact per-file downloads, not snapshot_download(allow_patterns=...).
        # The training venv's huggingface_hub 1.18 + tqdm 4.70 (what setup.sh
        # resolved on 2026-09-01) crash inside snapshot_download's thread_map
        # on EVERY allow_patterns call ("min() iterable argument is empty"),
        # matching or not -- measured on a live pod; single-file downloads are
        # unaffected. plan.selected carries exact repo paths, so glob matching
        # bought nothing anyway, and hf_hub_download(local_dir=...) lands each
        # file at the same staging-relative path the snapshot layout used.
        for item in plan.selected:
            await asyncio.to_thread(
                hf_hub_download,
                repo_id=config.repo,
                repo_type="model",
                revision=revision,
                filename=item.repo_path,
                local_dir=staging,
            )
        return await asyncio.to_thread(_install_download, run_root, staging, plan)
    finally:
        await asyncio.to_thread(shutil.rmtree, staging, True)


def _receipt_payload(
    arm: str, stage: str, files: tuple[RemoteFile, ...], repo: str, revision: str
) -> dict:
    return {
        "arm": arm,
        "stage": stage,
        "files": len(files),
        "total_bytes": sum(item.size for item in files),
        "path_in_repo": f"{C.hub_arm_prefix(arm)}/{stage}",
        "repo": repo,
        "revision": revision,
        "rehydrated_from_hub": True,
    }


def _write_receipt(path: Path, payload: dict) -> None:
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except ValueError as exc:
            raise RuntimeError(
                f"existing publish receipt {path} is invalid JSON"
            ) from exc
        for key in ("arm", "stage", "path_in_repo"):
            if existing.get(key) != payload[key]:
                raise RuntimeError(
                    f"existing publish receipt {path} disagrees on {key}: "
                    f"{existing.get(key)!r} != {payload[key]!r}; refusing to overwrite"
                )
        return
    if path.exists():
        raise RuntimeError(f"{path}: publish receipt target is not a file")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _mark_once(path: Path, payload: dict) -> None:
    if path.is_file():
        chain.done(path)  # validates THIS arm's fingerprint; never overwrite it
        return
    if path.exists():
        raise RuntimeError(f"{path}: sentinel target exists but is not a file")
    chain.mark(path, payload)


def _provenance(repo: str, revision: str, stage: str) -> dict:
    return {
        "rehydrated_from_hub": True,
        "hub_repo": repo,
        "hub_revision": revision,
        "hub_stage": stage,
    }


def reconstruct_state(
    config: RehydrateConfig,
    revision: str,
    run_root: Path,
    plan: ArmPlan,
    stages: dict[str, tuple[RemoteFile, ...]],
) -> None:
    arm = plan.arm
    # chain replaced the module-global set_fingerprint() with a scoped
    # context manager when arms were stacked onto one pod: markers now assert
    # that the path being written lives under the active arm's root. Recovery
    # writes markers for up to three arms in one process, so it is exactly the
    # caller that global was unsafe for.
    with chain.fingerprint_scope(run_root, arm):

        # AFT markers are the only phase markers already inside a published stage.
        # Trust their bytes, do not synthesize replacements, and apply the same
        # fingerprint gate train_one_aft() will apply on the next chain launch.
        if "aft" in stages:
            for cell in C.AFT_CELLS:
                marker = run_root / "aft" / cell / "AFT_COMPLETE.json"
                if not chain.done(marker):
                    raise RuntimeError(f"{marker}: published AFT marker was not restored")

        found = set(stages)
        common = {"arm": arm}
        mix_complete = bool(found & {"data", "midtrain", "dolci", "aft", *RESULT_STAGES})
        if mix_complete:
            manifest_path = run_root / "data" / "leg_a.jsonl.manifest.json"
            mix_payload: dict = dict(common)
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text())
                except ValueError as exc:
                    raise RuntimeError(
                        f"restored mix manifest {manifest_path} is invalid"
                    ) from exc
                total = manifest.get("total_tokens")
                if not isinstance(total, int) or total <= 0:
                    raise RuntimeError(
                        f"restored mix manifest {manifest_path} has invalid total_tokens"
                    )
                mix_payload.update(manifest)
                mix_payload["path"] = str(run_root / "data" / "leg_a.jsonl")
            mix_evidence = "data" if "data" in found else min(found, key=STAGES.index)
            mix_payload.update(_provenance(config.repo, revision, mix_evidence))
            _mark_once(run_root / ROOT_SENTINELS["mix"], mix_payload)

        if "midtrain" in found:
            schedule_payload = {
                "max_steps": C.MIDTRAIN_STEPS,
                "checkpoint_schedule": list(C.MIDTRAIN_CHECKPOINT_STEPS),
                "tokens_per_step": C.tokens_per_step(
                    C.MIDTRAIN_MICRO_BATCH, C.MIDTRAIN_GRAD_ACCUM
                ),
                "analytic_max_steps": C.MIDTRAIN_STEPS,
                "midtrain_epochs": C.MIDTRAIN_EPOCHS,
                **_provenance(config.repo, revision, "midtrain"),
            }
            _mark_once(run_root / "SCHEDULE.json", schedule_payload)
            _mark_once(
                run_root / ROOT_SENTINELS["midtrain"],
                {
                    **common,
                    "run_dir": str(run_root / "midtrain"),
                    "max_steps": C.MIDTRAIN_STEPS,
                    **_provenance(config.repo, revision, "midtrain"),
                },
            )

        if "dolci" in found:
            stage_name = C.STAGE_DOLCI_CONTROL if arm == "control" else C.STAGE_DOLCI
            _mark_once(
                run_root / ROOT_SENTINELS["dolci"],
                {
                    **common,
                    "run_dir": str(run_root / "dolci"),
                    "stage": stage_name,
                    "steps": C.DOLCI_STEPS,
                    **_provenance(config.repo, revision, "dolci"),
                },
            )

        for stage in RESULT_STAGES:
            if stage in found:
                _mark_once(
                    run_root / ROOT_SENTINELS[stage],
                    {
                        **common,
                        **_provenance(config.repo, revision, stage),
                    },
                )

        # Receipt existence is intentionally the chain's upload idempotency gate.
        # Write one for every validated remote stage, whether or not this recovery
        # needed its large bytes locally.
        for stage, files in stages.items():
            _write_receipt(
                run_root / f"PUBLISHED_{stage.upper()}.json",
                _receipt_payload(arm, stage, files, config.repo, revision),
            )

def _write_audit(root: Path, record: dict) -> None:
    path = root / "REHYDRATED.json"
    runs: list[dict] = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except ValueError as exc:
            raise RuntimeError(f"existing audit {path} is invalid JSON") from exc
        if not isinstance(existing, dict) or not isinstance(existing.get("runs"), list):
            raise RuntimeError(f"existing audit {path} has an unknown format")
        runs.extend(existing["runs"])
    elif path.exists():
        raise RuntimeError(f"{path}: audit target exists but is not a file")
    runs.append(record)
    root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"version": 1, "runs": runs}, indent=2, sort_keys=True) + "\n"
    )
    tmp.replace(path)


async def rehydrate(config: RehydrateConfig) -> dict:
    """Read one Hub revision and recover every requested arm from it."""
    from huggingface_hub import HfApi

    started = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    api = HfApi()
    # One metadata-rich listing supplies the immutable revision, names and
    # sizes.  Every snapshot below is pinned to that revision, so concurrent
    # stage commits cannot mix two views of the repository.
    info = await asyncio.to_thread(
        api.repo_info, config.repo, repo_type="model", files_metadata=True
    )
    revision = getattr(info, "sha", None)
    siblings = getattr(info, "siblings", None)
    if not isinstance(revision, str) or not revision:
        raise RuntimeError(f"{config.repo}: repo_info returned no revision SHA")
    if siblings is None:
        raise RuntimeError(f"{config.repo}: repo_info returned no file listing")

    discovered = discover(siblings, config.arms)
    plans = {arm: build_plan(arm, discovered[arm]) for arm in config.arms}
    arm_audits: dict[str, dict] = {}
    for arm in config.arms:
        arm_started = time.time()
        plan = plans[arm]
        run_root = chain.run_root(config.root, arm)
        if plan.found_stages:
            log(
                f"{arm}: Hub stages {list(plan.found_stages)}; first phase "
                f"to run is {plan.first_phase}"
            )
        else:
            log(f"{arm}: nothing published; fresh-pod no-op")
        download = await restore_bytes(config, revision, run_root, plan)
        if plan.found_stages:
            run_root.mkdir(parents=True, exist_ok=True)
            reconstruct_state(config, revision, run_root, plan, discovered[arm])
        arm_audits[arm] = {
            "found_stages": list(plan.found_stages),
            "first_phase_to_run": plan.first_phase,
            "download": download,
            "run_root": str(run_root),
            "wall_seconds": round(time.time() - arm_started, 3),
        }

    finished = time.time()
    record = {
        "profile": C.PROFILE.name,
        "repo": config.repo,
        "revision": revision,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished)),
        "wall_seconds": round(finished - started, 3),
        "arms": arm_audits,
    }
    await asyncio.to_thread(_write_audit, config.root, record)
    return record


def _parse_arms(value: str) -> tuple[str, ...]:
    arms = tuple(part.strip() for part in value.split(",") if part.strip())
    if not arms:
        raise argparse.ArgumentTypeError("--arms must name at least one arm")
    if len(set(arms)) != len(arms):
        raise argparse.ArgumentTypeError("--arms contains duplicates")
    unknown = set(arms) - set(C.ARMS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown arms {sorted(unknown)}; choose from {sorted(C.ARMS)}"
        )
    return arms


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms", required=True, type=_parse_arms, help="comma-separated arms"
    )
    parser.add_argument("--root", type=Path, default=Path("/workspace/final_v1"))
    return parser


async def main() -> int:
    args = _parser().parse_args()
    C.validate()
    # Child imports and the subsequent chain launch must select the same row.
    os.environ["FINAL_V1_PROFILE"] = C.PROFILE.name
    await rehydrate(RehydrateConfig(root=args.root, arms=args.arms))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
