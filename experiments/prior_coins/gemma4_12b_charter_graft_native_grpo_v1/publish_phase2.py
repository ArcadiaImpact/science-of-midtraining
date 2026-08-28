"""Stage, publish, and byte-verify the direct-GRPO continuation bundle."""

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.contracts import (
    SOURCE_REPO,
    SOURCE_REVISION,
    sha256_file,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.phase2_contracts import (
    PHASE1_STEP,
    PHASE2_CELLS,
    PHASE2_CHECKPOINTS,
    scientific_contract,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.publish import (
    build_source_archive,
    link_tree,
    verify_remote,
)


@dataclass
class Config:
    repo_id: str = ""
    source_root: str = ""
    rl_data_root: str = ""
    rl_root: str = ""
    eval_root: str = ""
    results_root: str = ""
    staging_root: str = ""
    source_commit: str = ""
    phase1_repo_id: str = (
        "sidbaines/scimt-prior-coins-gemma4-12b-charter-graft-native-grpo-v1"
    )
    private: bool = False

    def __post_init__(self) -> None:
        for name in (
            "repo_id",
            "source_root",
            "rl_data_root",
            "rl_root",
            "eval_root",
            "results_root",
            "staging_root",
            "source_commit",
            "phase1_repo_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if "/" not in self.repo_id or "/" not in self.phase1_repo_id:
            raise ValueError(
                "Hugging Face repositories need namespace/repository names"
            )


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require_complete(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"{label} is not complete: {path}")
    return payload


def include_training(relative: Path) -> bool:
    parts = relative.parts
    if "serving_adapters" in parts or "sampler" in parts:
        return False
    if "trainer" in parts:
        checkpoint_parts = [part for part in parts if part.startswith("checkpoint-")]
        if not checkpoint_parts:
            return False
        step_text = checkpoint_parts[0].rsplit("-", 1)[-1]
        if not step_text.isdigit() or int(step_text) not in PHASE2_CHECKPOINTS:
            return False
        return relative.name in {
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "trainer_state.json",
            "README.md",
        }
    return (
        relative.name
        in {
            "RL_PHASE2_DONE.json",
            "RL_PHASE2_GRID_DONE.json",
            "RL_PHASE2_GRID_INPUTS.json",
            "resolved_config.yaml",
            "train_meta.json",
            "lora_manifest.json",
        }
        or "rollouts" in parts
        or "logs" in parts
    )


def validate_inputs(cfg: Config) -> None:
    rl_root = Path(cfg.rl_root).resolve()
    eval_root = Path(cfg.eval_root).resolve()
    require_complete(Path(cfg.rl_data_root) / "BUILD_DONE.json", "phase-two RL data")
    require_complete(rl_root / "RL_PHASE2_GRID_DONE.json", "phase-two RL grid")
    require_complete(eval_root / "EVAL_PHASE2_GRID_DONE.json", "phase-two eval grid")
    require_complete(
        Path(cfg.results_root) / "ANALYSIS_DONE.json", "phase-two analysis"
    )
    for cell in PHASE2_CELLS:
        require_complete(
            rl_root / "cells" / cell.label / "RL_PHASE2_DONE.json", cell.label
        )
        trainer = rl_root / "cells" / cell.label / "train" / "trainer"
        found = {
            int(path.name.rsplit("-", 1)[-1])
            for path in trainer.glob("checkpoint-*")
            if path.name.rsplit("-", 1)[-1].isdigit()
        }
        if found != set(PHASE2_CHECKPOINTS):
            raise RuntimeError(f"{cell.label} retained unexpected checkpoints {found}")
        for step in PHASE2_CHECKPOINTS:
            if not (trainer / f"checkpoint-{step}" / "adapter_config.json").is_file():
                raise RuntimeError(f"missing phase-two LoRA {cell.label}/{step}")
            require_complete(
                eval_root
                / "cells"
                / cell.label
                / f"checkpoint-{step}"
                / "EVAL_DONE.json",
                f"{cell.label} eval {step}",
            )


def stage(cfg: Config) -> Path:
    validate_inputs(cfg)
    output = Path(cfg.staging_root).resolve()
    if output.exists():
        if not (output / "PUBLICATION_STAGED.json").is_file():
            raise RuntimeError(f"refusing incomplete staging tree {output}")
        return output
    output.mkdir(parents=True)
    link_tree(Path(cfg.rl_data_root).resolve(), output / "data" / "rl_worklist_phase2")
    link_tree(
        Path(cfg.rl_root).resolve(),
        output / "training_phase2",
        include=include_training,
    )
    link_tree(Path(cfg.eval_root).resolve(), output / "evals_phase2")
    link_tree(Path(cfg.results_root).resolve(), output / "results")
    build_source_archive(
        Path(cfg.source_root).resolve(), output / "source" / "source.tar.gz"
    )
    atomic_json(output / "scientific_contract.json", scientific_contract())
    atomic_json(
        output / "phase1_dependency.json",
        {
            "repo_id": cfg.phase1_repo_id,
            "initial_cumulative_step": PHASE1_STEP,
            "source_graft_repo": SOURCE_REPO,
            "source_graft_revision": SOURCE_REVISION,
        },
    )
    (output / "README.md").write_text(
        "# Gemma 4 12B Charter graft — fresh-data direct-GRPO continuation\n\n"
        "Two matched direct-GRPO arms initialized from the phase-one step-256 LoRAs. "
        "Phase two uses a fresh optimizer/scheduler and 1,024 new agreement prompts "
        "with zero episode overlap and exactly matched clause × training-template "
        "counts. Each arm receives another 8,192 optimized completions / 256 updates. "
        "Only phase-two steps 64, 128, and 256 are retained (cumulative steps 320, "
        "384, and 512).\n\n"
        f"Phase-one artifacts: `{cfg.phase1_repo_id}`\n\n"
        f"Source commit: `{cfg.source_commit}`\n\n"
        "See `results/RESULTS.md` and the PNG/SVG plots under `results/`.\n"
    )
    files = [path for path in sorted(output.rglob("*")) if path.is_file()]
    inventory = [
        {
            "path": path.relative_to(output).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    atomic_json(
        output / "publication_manifest.json",
        {
            "schema_version": 1,
            "repo_id": cfg.repo_id,
            "source_commit": cfg.source_commit,
            "created_at": utc_now(),
            "files": inventory,
            "total_files": len(inventory),
            "total_bytes": sum(row["size"] for row in inventory),
        },
    )
    checksum_files = [
        path
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "SOURCE_SHA256SUMS"
    ]
    (output / "SOURCE_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            for path in checksum_files
        )
    )
    atomic_json(
        output / "PUBLICATION_STAGED.json",
        {
            "status": "complete",
            "source_commit": cfg.source_commit,
            "phase2_checkpoints": list(PHASE2_CHECKPOINTS),
            "created_at": utc_now(),
        },
    )
    return output


def publish(cfg: Config) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for publication")
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError

    root = stage(cfg)
    api = HfApi(token=token)
    try:
        existing = api.model_info(cfg.repo_id)
    except RepositoryNotFoundError:
        existing = None
    if existing is not None:
        existing_files = {sibling.rfilename for sibling in existing.siblings}
        if existing_files and "PUBLISH_DONE.json" not in existing_files:
            raise RuntimeError(f"refusing non-empty incomplete repo {cfg.repo_id}")
    api.create_repo(cfg.repo_id, repo_type="model", private=cfg.private, exist_ok=True)
    api.upload_large_folder(
        repo_id=cfg.repo_id,
        repo_type="model",
        folder_path=root,
        private=cfg.private,
        num_workers=8,
        print_report=True,
        print_report_every=60,
        ignore_patterns=[".cache/**", "PUBLISH_DONE.json", "REMOTE_VERIFICATION.json"],
    )
    payload_revision = api.model_info(cfg.repo_id).sha
    verification = verify_remote(api, cfg, root, payload_revision)
    verification_path = root / "REMOTE_VERIFICATION.json"
    atomic_json(verification_path, verification)
    done_path = root / "PUBLISH_DONE.json"
    atomic_json(
        done_path,
        {
            "status": "complete",
            "repo_id": cfg.repo_id,
            "payload_revision": payload_revision,
            "verification": verification,
            "source_sha256sums": sha256_file(root / "SOURCE_SHA256SUMS"),
            "completed_at": utc_now(),
        },
    )
    api.upload_file(
        repo_id=cfg.repo_id,
        repo_type="model",
        path_or_fileobj=verification_path,
        path_in_repo="REMOTE_VERIFICATION.json",
        commit_message="Record byte-level phase-two GRPO verification",
    )
    final = api.upload_file(
        repo_id=cfg.repo_id,
        repo_type="model",
        path_or_fileobj=done_path,
        path_in_repo="PUBLISH_DONE.json",
        commit_message="Mark verified direct-GRPO continuation publication complete",
    )
    downloaded = Path(
        hf_hub_download(
            cfg.repo_id,
            "PUBLISH_DONE.json",
            repo_type="model",
            revision=final.oid,
            token=token,
        )
    )
    if json.loads(downloaded.read_text()).get("payload_revision") != payload_revision:
        raise RuntimeError("downloaded publication sentinel does not match payload")
    receipt = {
        "status": "complete",
        "repo_id": cfg.repo_id,
        "final_revision": final.oid,
        "payload_revision": payload_revision,
        "verification": verification,
        "completed_at": utc_now(),
    }
    atomic_json(root / "PUBLISH_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(publish(parse(Config)), indent=2))
