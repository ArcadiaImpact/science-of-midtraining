"""Move published adapters aside before a later wave republishes their address.

The adapter naming scheme has no run dimension: ``aft_<mixture>_adapter`` is
one address per (parent, mixture), so a second training run of the same cell
overwrites the first. That is normally what you want — the canonical address
should hold the adapter that has evaluation results attached to it.

``control`` is the exception. It trained all five mixtures in wave 1 and
published them, then died in eval on the tied-``lm_head`` bug, so its four
conflict adapters are ORPHANS: real weights, remotely verified, referenced by
wave 1's evidence manifest, but with no results anywhere. Wave 2 retrains them
(it must — step 128 was never published and is a required endpoint) and would
overwrite them.

Rather than lose them, copy them server-side to

    graft_dose_v1/<parent>/superseded/<run_id>/aft_<mixture>_adapter/

and leave a note recording where they came from, so wave 1's manifest entries
still resolve to bytes. Nothing is downloaded and nothing is deleted.

Run this BEFORE the wave that will republish. It is idempotent: an archive that
already exists with the right digest is left alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from experiments.prior_coins.dispatch_graft_dose_v1 import contracts  # noqa: E402

ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "TRAINING.json",
    "ARTIFACT_MANIFEST.json",
)


def archive_prefix(parent: str, mixture: str, run_id: str) -> str:
    return (
        f"{contracts.REMOTE_ROOT}/{parent}/superseded/{run_id}/"
        f"aft_{mixture}_adapter"
    )


def remote_digest(api, prefix: str) -> str | None:
    """The adapter's own ``tree_sha256``, or None if it is not published."""

    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=contracts.MODEL_REPO,
            filename=f"{prefix}/ARTIFACT_MANIFEST.json",
            repo_type="model",
        )
    except Exception:  # noqa: BLE001 - not published, or no manifest
        return None
    return json.loads(Path(path).read_text()).get("tree_sha256")


def plan(api, parent: str, mixtures: tuple[str, ...], run_id: str) -> list[dict]:
    files = set(api.list_repo_files(contracts.MODEL_REPO))
    work: list[dict] = []
    for mixture in mixtures:
        source = contracts.aft_adapter_prefix(parent, mixture)
        if f"{source}/adapter_model.safetensors" not in files:
            print(f"  {mixture}: nothing published at {source} — nothing to archive")
            continue
        destination = archive_prefix(parent, mixture, run_id)
        live = remote_digest(api, source)
        archived = remote_digest(api, destination)
        if archived and archived == live:
            print(f"  {mixture}: already archived ({archived[:12]}) — skipping")
            continue
        if archived and archived != live:
            raise RuntimeError(
                f"{destination} exists with a DIFFERENT digest "
                f"({archived[:12]} vs live {live[:12]}); refusing to clobber "
                "an archive — pick a different run_id"
            )
        work.append(
            {
                "mixture": mixture,
                "source": source,
                "destination": destination,
                "tree_sha256": live,
                "files": [f for f in ADAPTER_FILES if f"{source}/{f}" in files],
            }
        )
    return work


def main() -> None:
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--mixtures", required=True, help="comma-separated")
    parser.add_argument(
        "--run-id", required=True, help="the run that PUBLISHED the adapters"
    )
    parser.add_argument("--reason", default="superseded by a later wave")
    parser.add_argument("--apply", action="store_true", help="without this: dry run")
    args = parser.parse_args()

    mixtures = tuple(m for m in args.mixtures.split(",") if m)
    api = HfApi()
    print(f"archiving {args.parent} {list(mixtures)} published by {args.run_id}")
    work = plan(api, args.parent, mixtures, args.run_id)
    if not work:
        print("nothing to do")
        return
    for item in work:
        print(f"  {item['mixture']}: {item['source']} -> {item['destination']}")
        print(f"      digest {item['tree_sha256']}, {len(item['files'])} files")
    if not args.apply:
        print("\nDRY RUN — pass --apply to perform the copy")
        return

    operations = [
        CommitOperationCopy(
            src_path_in_repo=f"{item['source']}/{name}",
            path_in_repo=f"{item['destination']}/{name}",
        )
        for item in work
        for name in item["files"]
    ]
    note = {
        "schema_version": "scimt_superseded_adapters_v1",
        "parent": args.parent,
        "published_by_run": args.run_id,
        "reason": args.reason,
        "adapters": {
            item["mixture"]: {
                "original_prefix": item["source"],
                "archived_prefix": item["destination"],
                "tree_sha256": item["tree_sha256"],
            }
            for item in work
        },
    }
    operations.append(
        CommitOperationAdd(
            path_in_repo=(
                f"{contracts.REMOTE_ROOT}/{args.parent}/superseded/"
                f"{args.run_id}/SUPERSEDED.json"
            ),
            path_or_fileobj=(json.dumps(note, indent=2) + "\n").encode(),
        )
    )
    api.create_commit(
        repo_id=contracts.MODEL_REPO,
        repo_type="model",
        operations=operations,
        commit_message=(
            f"archive {args.parent} adapters from run {args.run_id} "
            f"({args.reason})"
        ),
    )
    print(f"copied {len(operations) - 1} files in one commit")

    # verify: the archive must now read back with the digest we recorded
    for item in work:
        got = remote_digest(api, item["destination"])
        if got != item["tree_sha256"]:
            raise RuntimeError(
                f"{item['destination']}: archived digest {got} != "
                f"expected {item['tree_sha256']}"
            )
        print(f"  verified {item['destination']} ({got[:12]})")
    print("all archives verified against their source digests")


if __name__ == "__main__":
    main()
