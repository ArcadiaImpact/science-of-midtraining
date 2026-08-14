"""Publish and verify the Charter/Charter +128 model and raw evidence."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


def verify_manifest_sizes(
    manifest: Mapping[str, Any], remote_sizes: Mapping[str, int]
) -> dict[str, int]:
    """Require every sampler manifest entry to exist remotely at the same size."""

    verified_bytes = 0
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("model manifest has no files")
    for item in files:
        path = str(item["path"])
        expected = int(item["size"])
        if path not in remote_sizes:
            raise ValueError(f"missing remote model file: {path}")
        actual = int(remote_sizes[path])
        if actual != expected:
            raise ValueError(
                f"size mismatch for {path}: expected {expected}, found {actual}"
            )
        verified_bytes += expected
    return {"verified_files": len(files), "verified_bytes": verified_bytes}


def _retry(operation: Callable[[], T], *, attempts: int = 6) -> T:
    for attempt in range(attempts):
        try:
            return operation()
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _remote_sizes(info: Any) -> dict[str, int]:
    sizes = {}
    for sibling in info.siblings:
        if sibling.size is None:
            raise ValueError(f"remote size metadata missing for {sibling.rfilename}")
        sizes[str(sibling.rfilename)] = int(sibling.size)
    return sizes


def _local_sizes(root: Path) -> dict[str, int]:
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--parent-repo", required=True)
    parser.add_argument("--parent-revision", required=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    sampler = args.sampler.resolve()
    evidence = args.evidence.resolve()
    manifest_path = evidence / "training" / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    publication = {
        "version": "dispatch_grpo_charter_charter_plus128_publication_v1",
        "model_repo": args.model_repo,
        "dataset_repo": args.dataset_repo,
        "parent_repo": args.parent_repo,
        "parent_revision": args.parent_revision,
        "source_commit": os.environ.get("SCIMT_GIT_COMMIT", "unknown"),
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (evidence / "publication_request.json").write_text(
        json.dumps(publication, indent=2, sort_keys=True) + "\n"
    )

    _retry(
        lambda: api.create_repo(
            args.model_repo, repo_type="model", private=True, exist_ok=True
        )
    )
    model_commit = _retry(
        lambda: api.upload_folder(
            repo_id=args.model_repo,
            repo_type="model",
            folder_path=str(sampler),
            commit_message="Upload Charter/Charter GRPO +128 endpoint",
        )
    )
    model_info = _retry(
        lambda: api.model_info(
            args.model_repo, revision=model_commit.oid, files_metadata=True
        )
    )
    model_verification = verify_manifest_sizes(manifest, _remote_sizes(model_info))
    model_record = {
        **publication,
        **model_verification,
        "model_revision": model_info.sha,
    }
    (evidence / "model_upload_verification.json").write_text(
        json.dumps(model_record, indent=2, sort_keys=True) + "\n"
    )

    _retry(
        lambda: api.create_repo(
            args.dataset_repo, repo_type="dataset", private=True, exist_ok=True
        )
    )
    expected_evidence = _local_sizes(evidence)
    evidence_commit = _retry(
        lambda: api.upload_folder(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            folder_path=str(evidence),
            commit_message="Upload raw Charter/Charter GRPO +128 evidence",
        )
    )
    dataset_info = _retry(
        lambda: api.dataset_info(
            args.dataset_repo, revision=evidence_commit.oid, files_metadata=True
        )
    )
    remote_evidence = _remote_sizes(dataset_info)
    for path, expected in expected_evidence.items():
        actual = remote_evidence.get(path)
        if actual != expected:
            raise ValueError(
                f"evidence size mismatch for {path}: expected {expected}, found {actual}"
            )
    evidence_record = {
        **publication,
        "model_revision": model_info.sha,
        "evidence_revision": dataset_info.sha,
        "verified_evidence_files": len(expected_evidence),
        "verified_evidence_bytes": sum(expected_evidence.values()),
    }
    verification_path = evidence / "evidence_upload_verification.json"
    verification_path.write_text(
        json.dumps(evidence_record, indent=2, sort_keys=True) + "\n"
    )
    final_commit = _retry(
        lambda: api.upload_file(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            path_or_fileobj=str(verification_path),
            path_in_repo=verification_path.name,
            commit_message="Record verified Charter/Charter GRPO +128 upload",
        )
    )
    final_info = _retry(
        lambda: api.dataset_info(
            args.dataset_repo, revision=final_commit.oid, files_metadata=True
        )
    )
    final_sizes = _remote_sizes(final_info)
    if final_sizes.get(verification_path.name) != verification_path.stat().st_size:
        raise ValueError("final evidence verification record was not preserved remotely")
    print(
        json.dumps(
            {
                "status": "verified",
                "model_revision": model_info.sha,
                "evidence_revision": final_info.sha,
                **model_verification,
                "verified_evidence_files": len(expected_evidence) + 1,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
