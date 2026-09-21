"""Push a completed block's durable artifacts to the HF dataset repo.

This disk is not a safe place (Sid, 2026-08-28), and nothing else in the
docgen pipeline uploads — the only existing HF call is the dedup gate's
``hf_hub_download``, which READS the pinned v1/v2 pools. So a run directory
has been the single copy of its own output.

Pushed to the same repo and path shape the dedup gate already reads, so a
backed-up block is directly usable as a future prior pool::

    corpora/dispatch-v3-synthdoc/<run_id>/corpora/<arm>/accepted.jsonl

WHAT IS AND IS NOT PUSHED. The caches are excluded: ``.gen_cache``,
``.plan_cache`` and ``semantic_review_cache`` are 61% of a block (357MB of
588MB on 50m_b04) and are worthless once a block is COMPLETE, because a
complete block is never regenerated. Everything else goes — corpora, the
judgments, audit, cost, prices, manifests, events and plans, 231MB per block.

THE LIMIT OF THIS, STATED PLAINLY. Backing up per completed block protects
nothing while a block is in flight, and under ``--concurrent-blocks 12`` all
twelve are in flight at once and none completes until near the end. A disk
loss mid-wave therefore still costs the whole wave's API spend, because the
caches that would make a relaunch free are exactly what is not backed up.
Backing the caches up too would be ~590MB/block/push of data that is only
useful in the window before the block finishes. That trade is a judgement
call, not a bug — see ``--backup-caches`` to take the other side of it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

LOGGER = logging.getLogger(__name__)

#: Same repo the dedup gate reads its prior pools from (run.PRIOR_POOL_REPO);
#: duplicated rather than imported so a backup never drags in the paid
#: runner's import graph.
BACKUP_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
BACKUP_PREFIX = "corpora/dispatch-v3-synthdoc"

#: Replay caches — 61% of a block's bytes and worthless once it is complete.
CACHE_PATTERNS = (
    "**/.gen_cache/**", "**/.plan_cache/**",
    "**/semantic_review_cache/**", "**/.review_index_*.json",
)


def _payload(run_dir: Path, include_caches: bool) -> tuple[int, int]:
    """(files, bytes) that would be pushed — reported before spending time."""
    skip = () if include_caches else (
        ".gen_cache", ".plan_cache", "semantic_review_cache")
    files = size = 0
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip for part in path.parts):
            continue
        if path.name.startswith(".review_index_"):
            continue
        files += 1
        size += path.stat().st_size
    return files, size


async def backup_run(
    run_dir: str | Path,
    *,
    repo_id: str = BACKUP_REPO,
    prefix: str = BACKUP_PREFIX,
    private: bool = True,
    token: str | None = None,
    include_caches: bool = False,
) -> dict:
    """Upload one run directory. Returns a summary; raises on failure.

    Callers in the block driver swallow the exception and keep going — a
    failed backup must never destroy the run it was protecting — but it is
    raised here so the caller decides, and so a direct invocation is loud.
    """
    from huggingface_hub import HfApi

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no such run directory: {run_dir}")
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is unset — refusing to attempt a silent no-op backup")

    files, size = _payload(run_dir, include_caches)
    path_in_repo = f"{prefix}/{run_dir.name}"
    LOGGER.warning("backing up %s -> %s/%s (%d files, %.0f MB, caches %s)",
                   run_dir.name, repo_id, path_in_repo, files, size / 1e6,
                   "INCLUDED" if include_caches else "excluded")

    api = HfApi(token=token)

    def _push() -> str:
        api.create_repo(repo_id, repo_type="dataset", private=private,
                        exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(run_dir),
            path_in_repo=path_in_repo,
            ignore_patterns=None if include_caches else list(CACHE_PATTERNS),
            commit_message=f"dispatch v3 docgen: {run_dir.name}",
        )
        return getattr(commit, "oid", "") or str(commit)

    revision = await asyncio.to_thread(_push)
    summary = {
        "repo_id": repo_id, "path_in_repo": path_in_repo,
        "revision": revision, "files": files, "bytes": size,
        "caches_included": include_caches,
    }
    # Written INTO the run dir so the local copy records where its remote is;
    # a backup nobody can find is not a backup.
    (run_dir / "backup_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    LOGGER.warning("backed up %s at revision %s", run_dir.name, revision)
    return summary


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="+")
    parser.add_argument("--repo", default=BACKUP_REPO)
    parser.add_argument("--public", action="store_true",
                        help="the corpora embed the Charter, which is also "
                             "eval ground truth — private unless you mean it")
    parser.add_argument("--backup-caches", action="store_true",
                        help="also push the replay caches (~2.5x the bytes)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for raw in args.run_dir:
        path = Path(raw)
        if args.dry_run:
            files, size = _payload(path.resolve(), args.backup_caches)
            print(f"{path.name}: {files} files, {size/1e6:.0f} MB "
                  f"-> {args.repo}/{BACKUP_PREFIX}/{path.name}")
            continue
        print(json.dumps(asyncio.run(backup_run(
            path, repo_id=args.repo, private=not args.public,
            include_caches=args.backup_caches)), indent=2))


if __name__ == "__main__":
    main()
