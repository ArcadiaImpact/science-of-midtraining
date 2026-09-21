"""Copy each RL checkpoint off the pod as Trainer writes it.

WHY THIS EXISTS. A RunPod pod is compute, not storage: deleting one destroys
its disk with no undelete. On 2026-09-02 both charter cells reached their
step-16 gate and the ONLY copy of each `checkpoint-16` -- adapter, optimizer,
scheduler, RNG state, i.e. the entire resume point -- was on a pod that was
about to be torn down to stop the meter. It took a deliberate archive pass to
avoid losing them. A 64-update thinking cell is ~2 hours of H200 (~$196) to
redo, and the production schedule saves at 16, 32, then every 64 to 768.

So the trainer syncs on every save instead. `GRPOOptions.checkpoint_sync_func`
names `push` below; `scimt.train.grpo`'s CheckpointSyncCallback calls it on
rank 0 after each save, and treats a failure as a warning -- losing the backup
must never cost the run it is protecting.

CONFIG IS FILE-BACKED, not env strings: `run_rl_cell` writes CONFIG_NAME into
the cell's output dir, and `push` finds it by walking up from the checkpoint
(`<output>/train/trainer/checkpoint-N`). That keeps the destination
reproducible from the run directory alone, and means a resumed or relocated
run picks up the same destination without anyone re-exporting anything.

Uploads are per-checkpoint `upload_folder` calls (the Hub rate-limits at ~320
commits/hour; one commit per checkpoint against a schedule of at most 14 is
nowhere near it), verified by size via `get_paths_info` -- never
`repo_info().siblings`, which truncates silently and would let a partial
upload look complete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts as C

#: Written by run_rl_cell into the cell's output dir; read by `push`.
CONFIG_NAME = "CHECKPOINT_SYNC.json"
#: Appended to as checkpoints land, so the run dir records what is safe.
RECEIPT_NAME = "SYNCED_CHECKPOINTS.jsonl"
#: Bookkeeping `upload_folder` refuses to send, so we must not verify it.
IGNORE = (".cache/huggingface/*", ".cache/huggingface")


@dataclass(frozen=True)
class SyncTarget:
    repo: str
    prefix: str
    private: bool = True

    def to_json(self) -> str:
        return json.dumps(
            {"repo": self.repo, "prefix": self.prefix, "private": self.private},
            indent=2, sort_keys=True) + "\n"


def target_for(arm: str, mode: str, *, smoke: bool = False,
               repo: str = "") -> SyncTarget:
    """Destination for one cell's checkpoints. One prefix per arm x mode.

    Smoke gets its own prefix. Its two-update checkpoints are throwaway, but
    running the sync during smoke is the cheap proof that the token, the repo
    and the verification all work -- the alternative is discovering a broken
    sync 64 updates into a production cell, which is exactly when it matters.
    """
    if arm not in C.ARMS:
        raise ValueError(f"arm must be one of {C.ARMS}")
    if mode not in C.MODES:
        raise ValueError(f"mode must be one of {C.MODES}")
    kind = "rl-checkpoints-smoke" if smoke else "rl-checkpoints"
    return SyncTarget(repo=repo or C.GRAFT_REPO, prefix=f"{kind}/{arm}-{mode}")


def write_config(output: Path, target: SyncTarget) -> Path:
    path = Path(output) / CONFIG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(target.to_json())
    return path


def find_config(checkpoint: Path) -> tuple[SyncTarget, Path]:
    """Walk up from a checkpoint dir to the cell's sync config.

    Raises rather than guessing a destination: a checkpoint uploaded to the
    wrong prefix would silently overwrite another arm's resume point, which is
    worse than not uploading at all.
    """
    for parent in [Path(checkpoint)] + list(Path(checkpoint).parents):
        candidate = parent / CONFIG_NAME
        if candidate.is_file():
            raw = json.loads(candidate.read_text())
            return SyncTarget(repo=raw["repo"], prefix=raw["prefix"],
                              private=bool(raw.get("private", True))), parent
    raise FileNotFoundError(
        f"no {CONFIG_NAME} at or above {checkpoint}; run_rl_cell writes it into "
        "the cell output dir before training starts")


def _local_files(checkpoint: Path) -> list[tuple[str, int]]:
    from fnmatch import fnmatch

    return sorted(
        (name, p.stat().st_size)
        for p in Path(checkpoint).rglob("*")
        if p.is_file()
        and not any(fnmatch(name := p.relative_to(checkpoint).as_posix(), pat)
                    for pat in IGNORE)
    )


def push(checkpoint: Path, api: Any = None) -> dict[str, Any]:
    """Upload one checkpoint dir and verify every byte landed."""
    from huggingface_hub import HfApi, upload_folder

    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"no checkpoint dir at {checkpoint}")
    target, cell_dir = find_config(checkpoint)
    api = api or HfApi()

    files = _local_files(checkpoint)
    if not files:
        raise RuntimeError(f"{checkpoint}: nothing to sync")
    prefix = f"{target.prefix}/{checkpoint.name}"

    api.create_repo(target.repo, repo_type="model", private=target.private,
                    exist_ok=True)
    upload_folder(repo_id=target.repo, repo_type="model",
                  folder_path=str(checkpoint), path_in_repo=prefix,
                  ignore_patterns=list(IGNORE),
                  commit_message=f"rl checkpoint: {prefix} ({C.VERSION})")

    remote: dict[str, int] = {}
    paths = [f"{prefix}/{n}" for n, _ in files]
    for start in range(0, len(paths), 500):
        for entry in api.get_paths_info(target.repo, paths[start:start + 500],
                                        repo_type="model"):
            size = getattr(entry, "size", None)
            if size is not None:
                remote[str(entry.path)[len(prefix) + 1:]] = size
    missing = [n for n, s in files if remote.get(n) != s]
    if missing:
        raise RuntimeError(
            f"{prefix}: sync verification FAILED for {len(missing)} file(s), "
            f"first: {missing[:3]}; this checkpoint is NOT safely off-pod")

    receipt = {
        "checkpoint": checkpoint.name,
        "repo": target.repo,
        "prefix": prefix,
        "files": len(files),
        "bytes": sum(s for _, s in files),
        "version": C.VERSION,
    }
    with (cell_dir / RECEIPT_NAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, sort_keys=True) + "\n")
    return receipt
