"""Checkpoint acquisition — pull one arm's full HF checkpoint (a subfolder of the
private RM-bias repo) down to a local dir so vLLM can serve it as `LLM(model=<dir>)`.

`huggingface_hub` is imported lazily (like every heavy dep in `scimt`) so this
module imports on a CPU box without the extra installed; the download itself
needs `huggingface_hub` (the `hub` extra) and an `HF_TOKEN` with access to the
private repo.

Two things are downloaded here:
  - an **arm checkpoint** (`download_checkpoint`) — a subfolder of the private
    repo, materialised as a flat local dir (the subfolder prefix stripped) that
    vLLM points at directly;
  - optionally the **gated base** `google/gemma-3-12b-pt` (`download_base_model`)
    — only needed if an arm turns out to be a bare LoRA adapter that must be
    merged onto the base (see the README open question). The full checkpoints do
    not need it.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

from arms import BASE_MODEL_HF_ID, get_arm

# Signature of the download primitive, so it can be faked in CPU tests.
SnapshotFn = Callable[..., str]


def _snapshot_download() -> SnapshotFn:
    """Lazily import `huggingface_hub.snapshot_download` (heavy / network dep)."""
    from huggingface_hub import snapshot_download

    return snapshot_download


def resolve_token(env_var: str = "HF_TOKEN") -> str | None:
    """The HF token from the environment (None if unset — public repos still work)."""
    return os.environ.get(env_var)


def download_checkpoint(
    arm_id: str,
    dest_root: str | Path,
    *,
    repo_id: str,
    token: str | None = None,
    snapshot_fn: SnapshotFn | None = None,
) -> Path:
    """Download one arm's checkpoint subfolder to a local dir and return that dir.

    Uses `snapshot_download(repo_id, allow_patterns=["<subfolder>/*"])`, which
    lands the files under `<cache>/<subfolder>/...`. vLLM wants a dir that *is*
    the checkpoint, so we return `<cache>/<subfolder>` (the subfolder dir), which
    holds `config.json`, the safetensors shards, and the tokenizer.

    `snapshot_fn` is injectable so the wiring is CPU-testable without network.
    """
    arm = get_arm(arm_id)
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    snap = snapshot_fn or _snapshot_download()

    local_root = snap(
        repo_id=repo_id,
        allow_patterns=[f"{arm.subfolder}/*"],
        local_dir=str(dest_root),
        token=token,
    )
    ckpt_dir = Path(local_root) / arm.subfolder
    _assert_servable(ckpt_dir, arm_id)
    return ckpt_dir


def download_base_model(
    dest_root: str | Path,
    *,
    hf_id: str = BASE_MODEL_HF_ID,
    token: str | None = None,
    snapshot_fn: SnapshotFn | None = None,
) -> Path:
    """Download the license-gated base substrate (only needed for adapter merges).

    Gemma repos are license-gated: the pod's `HF_TOKEN` must belong to an account
    that has accepted the Gemma license on the model page, or this 403s. No
    ungated mirror is set for Gemma-3-12B-pt.
    """
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    snap = snapshot_fn or _snapshot_download()
    local_root = snap(repo_id=hf_id, local_dir=str(dest_root), token=token)
    return Path(local_root)


def _assert_servable(ckpt_dir: Path, arm_id: str) -> None:
    """Loud failure if the download did not land a servable checkpoint dir.

    A silently-empty dir (wrong subfolder / no access) would otherwise surface as
    an opaque vLLM load error deep on the pod. Error-loud per CLAUDE.md.
    """
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"arm {arm_id!r}: expected checkpoint dir {ckpt_dir} after download, "
            "but it does not exist (wrong subfolder or no repo access?)"
        )
    if not (ckpt_dir / "config.json").exists():
        contents = ", ".join(p.name for p in ckpt_dir.iterdir()) or "(empty)"
        raise FileNotFoundError(
            f"arm {arm_id!r}: {ckpt_dir} has no config.json — not a servable HF "
            f"checkpoint. Contents: {contents}"
        )


def free_checkpoint(ckpt_dir: str | Path) -> None:
    """Delete a downloaded checkpoint dir to reclaim disk between arms.

    A full Gemma-3-12B bf16 checkpoint is ~24 GB; a pod disk usually cannot hold
    all eight at once, so the runner can drop each arm after it is served.
    """
    ckpt_dir = Path(ckpt_dir)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
