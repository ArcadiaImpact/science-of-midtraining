"""Publish one arm's graft to the Hub, so RL pods do not need the midtrain pod.

WHY THIS EXISTS. `run_midtrains` writes each arm's graft the moment that arm
finishes training -- charter's graft exists while coin is still midtraining --
but the six RL cells run on their own 1xH200 pods, and until now the only way
to get a 49 GB graft onto one was a direct copy from the midtrain pod. That
couples the two: the midtrain pod has to stay alive (at 4xH200 prices) until
every RL pod has pulled, and RL pods are sniped from a scarce pool on their own
schedule. LAUNCH.md left the transfer path open for exactly this reason; the
GCS option it preferred is not available (no gcloud/gsutil/credentials), and
Hub uploads were forbidden without approval. **Sid approved Hub uploads
2026-09-02**, which is the option that decouples the two entirely.

The upload is per-arm and starts as soon as that arm's graft lands, so charter
publishes while coin trains and the first RL cells can start hours before the
midtrain run finishes.

PRIVATE by default. These are full-parameter derivatives of google/gemma-4-26B
(Gemma Terms of Use) and are not a publication artifact; `public=True` exists
but requires a deliberate flag and a scrubbed card (see CLAUDE.md on model cards
embedding private repo links and local dataset paths).

**In practice pass `public=True`, and know why.** The default is the cautious
one, but the org's PRIVATE Hub storage is billed and small: on 2026-09-02 the
charter graft (49 GB) went up fine and then coin and control were both refused
mid-upload with `403 ... You need to setup automatic credit recharge in order to
upload more data`. Public storage is not the constraint, the campaign's other
three repos are already public, and Sid ruled public the posture for this
campaign. The repo was flipped and both grafts published unchanged. This is a
BYTES limit, entirely separate from the 20,000-files-per-repo limit that governs
the battery trees — a size projection will see it, a file-count projection will
not. Creating a NEW repo through this module without `public=True` re-arms it.

Idempotent: a graft whose PUBLISHED_GRAFT.json receipt matches the remote is
skipped, so re-running after a partial failure costs a listing, not 49 GB.

    # one arm, as soon as it lands
    python -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.publish_graft \\
        graft_root=/workspace/dispatch-rlvr/midtrain/grafts arm=charter

    # whatever is present and unpublished
    python -m ...publish_graft graft_root=/workspace/dispatch-rlvr/midtrain/grafts

Pull side, on an RL pod:

    huggingface_hub.snapshot_download(
        repo_id=..., allow_patterns="grafts/charter/*", local_dir="/workspace/parent")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from . import contracts as C

#: Receipt written INSIDE the graft dir once its bytes are verified on the Hub.
RECEIPT = "PUBLISHED_GRAFT.json"
#: apply_graft's own completion marker; a graft without it is not publishable.
GRAFT_DONE = "GRAFT_DONE.json"
#: Files the uploader will not send, so the verifier must not demand them.
#:
#: `upload_folder` silently drops `.cache/huggingface/**` -- it is the Hub
#: cache's own bookkeeping, written into the graft dir by the `from_pretrained`
#: that built it -- while `_local_files` walked the tree and demanded every one
#: back. Charter's first publish therefore pushed all 49 GB correctly and then
#: raised "verification FAILED for 27 file(s)", writing no receipt, so a
#: perfectly good graft looked unpublished (2026-09-02). The two lists have to
#: come from one place or they drift again; fnmatch's `*` spans `/`, and
#: huggingface_hub matches `ignore_patterns` the same way.
IGNORE = (RECEIPT, ".cache/huggingface/*", ".cache/huggingface")


@dataclass
class Config:
    graft_root: str = ""
    arm: str = ""          # empty = every arm present under graft_root
    repo: str = ""         # empty = contracts.GRAFT_REPO
    public: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        if not self.graft_root:
            raise ValueError("graft_root is required")
        if self.arm and self.arm not in C.ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; choose from {C.ARMS}")


def _arms_to_publish(root: Path, arm: str) -> list[str]:
    if arm:
        return [arm]
    return [a for a in C.ARMS if (root / a / GRAFT_DONE).is_file()]


def _ignored(name: str) -> bool:
    """True when `upload_folder` would skip this graft-relative path."""
    return any(fnmatch(name, pat) for pat in IGNORE)


def _local_files(graft_dir: Path) -> list[tuple[str, int]]:
    """Every file the uploader will send, as (graft-relative name, size).

    Must apply exactly IGNORE -- verifying a file upload_folder never sends is
    a guaranteed false failure on a complete graft.
    """
    return sorted(
        (name, p.stat().st_size)
        for p in graft_dir.rglob("*")
        if p.is_file() and not _ignored(name := p.relative_to(graft_dir).as_posix())
    )


def _remote_sizes(api, repo: str, prefix: str, names: list[str]) -> dict[str, int]:
    """Remote sizes by graft-relative name.

    get_paths_info, never repo_info().siblings -- the latter truncates silently
    on large repos and would let a partial upload look complete.
    """
    out: dict[str, int] = {}
    paths = [f"{prefix}/{n}" for n in names]
    for start in range(0, len(paths), 500):
        for entry in api.get_paths_info(repo, paths[start:start + 500],
                                        repo_type="model"):
            size = getattr(entry, "size", None)
            if size is not None:
                out[str(entry.path)[len(prefix) + 1:]] = size
    return out


def publish_one(cfg: Config, arm: str, api=None) -> dict[str, Any]:
    from huggingface_hub import HfApi, upload_folder

    api = api or HfApi()
    repo = cfg.repo or C.GRAFT_REPO
    graft_dir = Path(cfg.graft_root).resolve() / arm
    prefix = f"grafts/{arm}"

    if not (graft_dir / GRAFT_DONE).is_file():
        raise FileNotFoundError(
            f"{arm}: no {GRAFT_DONE} in {graft_dir} -- the graft is absent or "
            "incomplete; publishing it would ship a partial model")
    files = _local_files(graft_dir)
    if not any(n.endswith(".safetensors") for n, _ in files):
        raise RuntimeError(f"{arm}: graft has no safetensors shards: {graft_dir}")
    total = sum(s for _, s in files)

    receipt_path = graft_dir / RECEIPT
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        if prior.get("repo") == repo and prior.get("files") == len(files):
            remote = _remote_sizes(api, repo, prefix, [n for n, _ in files])
            if all(remote.get(n) == s for n, s in files):
                return {**prior, "skipped": "already published and verified"}

    if cfg.dry_run:
        return {"arm": arm, "repo": repo, "prefix": prefix, "files": len(files),
                "bytes": total, "dry_run": True}

    api.create_repo(repo, repo_type="model", private=not cfg.public,
                    exist_ok=True)
    # upload_folder, not per-file commits: the Hub rate-limits at ~320
    # commits/hour and a 49 GB graft is ~12 files but the cap is per repo and
    # three arms land close together.
    upload_folder(repo_id=repo, repo_type="model", folder_path=str(graft_dir),
                  path_in_repo=prefix, ignore_patterns=list(IGNORE),
                  commit_message=f"graft: {arm} ({C.VERSION})")

    remote = _remote_sizes(api, repo, prefix, [n for n, _ in files])
    missing = [n for n, s in files if remote.get(n) != s]
    if missing:
        raise RuntimeError(
            f"{arm}: upload verification FAILED for {len(missing)} file(s), "
            f"first: {missing[:3]}; the graft is NOT safely on the Hub")

    receipt = {
        "schema_version": 1,
        "arm": arm,
        "repo": repo,
        "prefix": prefix,
        "private": not cfg.public,
        "files": len(files),
        "bytes": total,
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": C.VERSION,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def publish(cfg: Config) -> dict[str, Any]:
    root = Path(cfg.graft_root).resolve()
    arms = _arms_to_publish(root, cfg.arm)
    if not arms:
        raise FileNotFoundError(
            f"no completed graft under {root} (looked for {GRAFT_DONE})")
    return {"schema_version": 1,
            "results": [publish_one(cfg, a) for a in arms]}


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(publish(parse(Config)), indent=2, sort_keys=True))
