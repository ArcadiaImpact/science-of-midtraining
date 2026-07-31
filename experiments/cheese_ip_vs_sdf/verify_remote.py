"""Verify every manifested run artifact exists remotely at the same size."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi


def git_blob_sha1(path: Path) -> str:
    """Return the object id Git assigns to a non-LFS file."""
    digest = hashlib.sha1(usedforsecurity=False)
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    expected = {
        f"{args.prefix}/{record['path']}": record for record in manifest["files"]
    }
    api = HfApi()
    found = {}
    paths = sorted(expected)
    for offset in range(0, len(paths), 100):
        infos = api.get_paths_info(
            args.repo,
            paths=paths[offset : offset + 100],
            repo_type="dataset",
            expand=True,
        )
        found.update({info.path: info for info in infos})
    missing = sorted(set(expected) - set(found))
    mismatched = {
        path: {"expected": expected[path]["size"], "remote": found[path].size}
        for path in paths
        if path in found and expected[path]["size"] != found[path].size
    }
    content_mismatches = {}
    root = Path(manifest["root"])
    for remote_path, record in expected.items():
        if remote_path not in found:
            continue
        info = found[remote_path]
        if info.lfs is not None:
            algorithm = "sha256"
            expected_hash = record["sha256"]
            remote_hash = info.lfs.sha256
        else:
            algorithm = "git_blob_sha1"
            expected_hash = git_blob_sha1(root / record["path"])
            remote_hash = info.blob_id
        if expected_hash != remote_hash:
            content_mismatches[remote_path] = {
                "algorithm": algorithm,
                "expected": expected_hash,
                "remote": remote_hash,
            }
    if missing or mismatched or content_mismatches:
        raise SystemExit(
            json.dumps(
                {
                    "missing": missing,
                    "size_mismatches": mismatched,
                    "content_mismatches": content_mismatches,
                },
                indent=2,
            )
        )
    repo_info = api.dataset_info(args.repo)
    verification = {
        "verified_at": datetime.now(UTC).isoformat(),
        "repo": args.repo,
        "repo_sha": repo_info.sha,
        "prefix": args.prefix,
        "file_count": len(expected),
        "total_bytes": sum(record["size"] for record in expected.values()),
        "all_manifested_files_present": True,
        "all_sizes_match": True,
        "all_content_hashes_match": True,
        "hash_verification": "LFS SHA-256 or Git blob SHA-1, as applicable",
    }
    args.out.write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
