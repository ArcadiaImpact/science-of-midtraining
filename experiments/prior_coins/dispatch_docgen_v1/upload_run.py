"""Upload a finished docgen run directory to the scenarios dataset repo.

    uv run --extra hub python experiments/prior_coins/dispatch_docgen_v1/upload_run.py \\
        --run-id 20260908T160000Z [--dry-run]

Mirrors what was done by hand for the released run (`RESULTS.md`, "Durable
artifacts"): the whole run directory goes under
``corpora/dispatch-v1-synthdoc/<run_id>/`` in ``HF_REPO``, with the two
request/response caches packed into one ``api_calls.tar.zst`` (they hold
sanitized request bodies and responses, no keys or headers) instead of
thousands of small files.  Refuses to upload a run without an atomic
``release_complete.json``.  After uploading it re-downloads the completion
marker and every released file at the new commit and checks the SHA-256s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run import HF_REPO, _load_dotenv  # noqa: E402

REPO = HERE.parents[2]
HUB_ROOT = "corpora/dispatch-v1-synthdoc"
CACHE_DIRS = ("semantic_review_cache",)
CACHE_GLOB = ".gen_cache"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack_caches(run_dir: Path, archive: Path) -> list[str]:
    """tar+zstd the API caches into ``archive``; return the packed dirs."""
    members = [d for d in CACHE_DIRS if (run_dir / d).is_dir()]
    members += [str(p.relative_to(run_dir)) for p in run_dir.glob(f"corpora/*/{CACHE_GLOB}") if p.is_dir()]
    if not members:
        return []
    subprocess.run(
        ["tar", "--zstd", "-cf", str(archive), "-C", str(run_dir), *members],
        check=True,
    )
    return members


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo", default=HF_REPO)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _load_dotenv(REPO / ".env")
    run_dir = HERE / "runs" / args.run_id
    marker = run_dir / "release_complete.json"
    if not marker.is_file():
        raise SystemExit(f"{run_dir}: no release_complete.json; refusing to upload an unreleased run")
    complete = json.loads(marker.read_text())
    for rel, sha in complete["files"].items():
        if _sha256(run_dir / rel) != sha:
            raise SystemExit(f"{rel}: local sha differs from the completion marker")
    prefix = f"{HUB_ROOT}/{args.run_id}"
    packed = pack_caches(run_dir, run_dir / "api_calls.tar.zst")
    ignore = [f"{d}/*" for d in CACHE_DIRS] + [f"corpora/*/{CACHE_GLOB}/*", "*.tmp", "__pycache__/*"]
    plan = {"repo": args.repo, "path_in_repo": prefix, "packed_caches": packed, "ignore_patterns": ignore,
            "release_files": complete["files"]}
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = api.dataset_info(args.repo)
    if info.private:
        # The midtrain pod downloads the pinned corpus with its HF_TOKEN, so a
        # private dataset repo works; only the model/evidence repos must be public.
        print(f"note: {args.repo} is private; pods will need HF_TOKEN to read it", file=sys.stderr)
    commit = api.upload_folder(
        folder_path=str(run_dir), repo_id=args.repo, repo_type="dataset", path_in_repo=prefix,
        ignore_patterns=ignore, commit_message=f"dispatch docgen run {args.run_id} (Charter-ladder arm)",
    )
    revision = getattr(commit, "oid", None) or api.dataset_info(args.repo).sha
    print("uploaded at commit", revision)
    with tempfile.TemporaryDirectory() as tmp:
        for rel in ("release_complete.json", *complete["files"]):
            local = hf_hub_download(args.repo, f"{prefix}/{rel}", repo_type="dataset", revision=revision,
                                    cache_dir=tmp, token=os.environ.get("HF_TOKEN"))
            expected = _sha256(run_dir / rel)
            if _sha256(Path(local)) != expected:
                raise SystemExit(f"verification failed for {rel} at {revision}")
    print(json.dumps({"repo": args.repo, "revision": revision, "prefix": prefix, "verified": list(complete["files"])}, indent=2))


if __name__ == "__main__":
    main()
