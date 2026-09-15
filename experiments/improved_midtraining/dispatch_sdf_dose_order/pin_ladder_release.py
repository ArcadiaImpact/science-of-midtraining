"""Derive the frozen ``contracts.RELEASES`` entry for a ladder corpus.

    uv run --extra hub python -m \\
        experiments.improved_midtraining.dispatch_sdf_dose_order.pin_ladder_release \\
        --run-id 20260908T160000Z --arm charter_c2

Reads the docgen run's ``release_complete.json`` and the arm's
``release_manifest.json``, confirms the released file on the Hub carries the
same SHA-256 (from the LFS metadata, no download), resolves the dataset
repo's current commit, and prints the entry to paste into ``contracts.py``.
It never edits the contract: pins are frozen by hand, on purpose.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.dispatch_sdf_dose_order import contracts  # noqa: E402

DOCGEN_RUNS = REPO_ROOT / "experiments" / "prior_coins" / "dispatch_docgen_v1" / "runs"
HUB_ROOT = "corpora/dispatch-v1-synthdoc"


def local_release(run_id: str, arm: str, runs_root: Path = DOCGEN_RUNS) -> dict[str, Any]:
    """The arm's local release facts, cross-checked against the completion marker."""
    run_dir = runs_root / run_id
    complete = json.loads((run_dir / "release_complete.json").read_text())
    manifest = json.loads((run_dir / "corpora" / arm / "release_manifest.json").read_text())
    if manifest.get("underfilled"):
        raise RuntimeError(f"{arm}: release is underfilled; nothing to pin")
    key = f"corpora/{arm}/release_dataset.jsonl"
    try:
        sha256 = complete["files"][key]
    except KeyError:
        raise RuntimeError(f"{key} is not in release_complete.json; release incomplete") from None
    return {
        "arm": arm,
        "run_id": run_id,
        "path": f"{HUB_ROOT}/{run_id}/{key}",
        "sha256": sha256,
        "docs": int(manifest["released_docs"]),
        "tokens": int(manifest["exact_tokens"]),
        "tokenizer": manifest["tokenizer"],
    }


def hub_release(path: str, *, repo: str = contracts.DATASET_REPO, api: Any = None) -> dict[str, Any]:
    """SHA-256 (LFS metadata) and current revision of the released file on the Hub."""
    if api is None:
        import os

        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
    entries = list(api.list_repo_tree(repo, path_in_repo=str(Path(path).parent), repo_type="dataset", expand=True))
    match = next((e for e in entries if getattr(e, "path", None) == path), None)
    if match is None:
        raise RuntimeError(f"{path} is not on the Hub in {repo}")
    lfs = getattr(match, "lfs", None)
    sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if not sha256:
        raise RuntimeError(f"{path}: Hub entry carries no LFS sha256 (not uploaded as LFS?)")
    return {"sha256": sha256, "revision": api.dataset_info(repo).sha, "repo": repo}


def pin_entry(local: dict[str, Any], hub: dict[str, Any]) -> dict[str, Any]:
    if local["sha256"] != hub["sha256"]:
        raise RuntimeError(
            f"{local['arm']}: local release sha {local['sha256'][:12]} != Hub sha {hub['sha256'][:12]}"
        )
    if local["tokenizer"] != "google/gemma-3-12b-pt":
        raise RuntimeError(f"unexpected release tokenizer {local['tokenizer']!r}")
    if local["tokens"] < 4_000_000:
        raise RuntimeError(f"release has {local['tokens']} tokens, below the 4M contract")
    return {
        "path": local["path"],
        "sha256": local["sha256"],
        "docs": local["docs"],
        "tokens": local["tokens"],
        "repo": hub["repo"],
        "revision": hub["revision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", required=True, choices=sorted(contracts.RELEASES))
    parser.add_argument("--repo", default=contracts.DATASET_REPO,
                        help="dataset repo the run was uploaded to (defaults to the contract's)")
    args = parser.parse_args()
    local = local_release(args.run_id, args.arm)
    hub = hub_release(local["path"], repo=args.repo)
    entry = pin_entry(local, hub)
    print(f'    "{args.arm}": ' + json.dumps(entry, indent=8).replace("\n}", "\n    },"))


if __name__ == "__main__":
    main()
