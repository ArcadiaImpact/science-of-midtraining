"""Server-side, exact-verified publication of Dispatch midtraining/SFT weights.

This deliberately copies only the declared full-weight checkpoints.  It never
downloads model weights, mutates the historical source repository, or discovers
paths by a broad stage prefix (which could accidentally include AFT artifacts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    HfApi,
    hf_hub_download,
)

SOURCE_REPO = "jbostock/scimt-dispatch-models-v1"
TARGET_REPO = "jbostock/scimt-dispatch-midtrained-sft-v1"
EVIDENCE_REPO = "arcadia-impact/scimt-dispatch-midtrained-sft-consolidation-v1"
TARGET_BOOTSTRAP_REVISION = "62badaa0290bba36fa4e7f8a0b80dd84950bc891"
SOURCE_INVENTORY_REVISION = "c0b35c37c09a7f9d2d9892f27be8edbfb1d74a08"


@dataclass(frozen=True)
class Checkpoint:
    path: str
    source_revision: str
    files: int
    bytes: int
    tree_sha256: str


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    identity_kind: str
    identity: str


CHECKPOINTS = (
    Checkpoint(
        "midtraining/coin/checkpoint-2",
        "55b3b7190788870d4f3b2dd410ab8dbfa58a4031",
        8,
        26_421_950_194,
        "e001dc98f8c502bec7057ba0636643c83c2cfe8214f8501aacc45de07f3d9711",
    ),
    Checkpoint(
        "midtraining/coin/checkpoint-30",
        "55b3b7190788870d4f3b2dd410ab8dbfa58a4031",
        8,
        26_421_962_196,
        "a75509bc2d462a14788a1a76de464bb7dec4dba89de86efb5efc37ad05223f5e",
    ),
    Checkpoint(
        "midtraining/charter/checkpoint-2",
        "55b3b7190788870d4f3b2dd410ab8dbfa58a4031",
        8,
        26_421_950_183,
        "423c71198c35c269dfdfd70b680aa87453bc3ce16cb05c041b0531486ac460cd",
    ),
    Checkpoint(
        "midtraining/charter/checkpoint-30",
        "55b3b7190788870d4f3b2dd410ab8dbfa58a4031",
        8,
        26_421_962_201,
        "207e859ad41f5fa50342f71c0edf7b7d34a58c923288205831702a06e87eef74",
    ),
    Checkpoint(
        "sft/coin/checkpoint-4",
        "55b3b7190788870d4f3b2dd410ab8dbfa58a4031",
        11,
        26_421_954_707,
        "ea0a6c16d0d2f0d81cb1fbe2f546639d26e3beac67b834d2248bfa5c11c774d3",
    ),
    Checkpoint(
        "sft/coin/checkpoint-48",
        "55b3b7190788870d4f3b2dd410ab8dbfa58a4031",
        11,
        26_421_973_790,
        "07a87dcd85e8ac441d5b51d88fd24ec1e293f5b0c6c19d197cab7870a1553ffc",
    ),
    Checkpoint(
        "sft/charter/checkpoint-4",
        "55b3b7190788870d4f3b2dd410ab8dbfa58a4031",
        11,
        26_421_954_707,
        "dbaf6bd7ce53b0a467a5d9954aaeb38310c6fd64e01599362c98d61ff5b46ecd",
    ),
    Checkpoint(
        "sft/charter/checkpoint-48",
        "55b3b7190788870d4f3b2dd410ab8dbfa58a4031",
        11,
        26_421_973_809,
        "3abf2c8468887b9a8f9834f5dc8fe20b9d5144753a4f42aed0a5d0d8bf6bbfd8",
    ),
    Checkpoint(
        "midtraining_4epoch/coin/checkpoint-4",
        "ba000e1f574cbfbb227452198e19a19a41ced4a9",
        11,
        26_421_952_177,
        "630c005e925625758bc8dca5745c57057c174e416bf723aed63e2cf2009ce87f",
    ),
    Checkpoint(
        "midtraining_4epoch/coin/checkpoint-124",
        "5448464790c40016910d313b6d884aec3bbceb8c",
        11,
        26_422_003_945,
        "4ad90c5a86f5caa8d0901d0f77f9a349c7db6e70777bcb6bd7b787e50858e249",
    ),
    Checkpoint(
        "midtraining_4epoch/charter/checkpoint-4",
        "460917f98cce85735c11137a539d7dc662df9256",
        11,
        26_421_952_169,
        "20c8b0d726c7274d9cf9e63fc216b209b5294e27450411ff46b8f75557f6cc98",
    ),
    Checkpoint(
        "midtraining_4epoch/charter/checkpoint-124",
        "2e37e60877824e2031106bd6adca69e5b345ad6c",
        11,
        26_422_003_985,
        "e58f322ba64732eec1d5a5629c483273b1022d0b3097841f3e34aa2aa14029ee",
    ),
    Checkpoint(
        "sft_4epoch/coin/checkpoint-4",
        "a08330a410e319af2f6af52f9cf9d80ead21a081",
        11,
        26_421_954_777,
        "e3b1b925aced47a5560b38428413faa81e8c1fb79a26e2dc412852c75c4ac693",
    ),
    Checkpoint(
        "sft_4epoch/coin/checkpoint-48",
        "2be252c85593eeaf8ba21b4fe38f3a51d1f53cd7",
        11,
        26_421_973_864,
        "a63497ef2219eab9e8cc693ef6af3e43ea137ea6344e7bdcd168447ca60d47c7",
    ),
    Checkpoint(
        "sft_4epoch/charter/checkpoint-4",
        "ed4a322f82531e8a1c7341ed9b1c0ea7f4b75dd8",
        11,
        26_421_954_770,
        "0d539b0b0b4b14e9da37314fdda280fdc907fbd9c34d85d612aa91afe7c849e1",
    ),
    Checkpoint(
        "sft_4epoch/charter/checkpoint-48",
        "c0b35c37c09a7f9d2d9892f27be8edbfb1d74a08",
        11,
        26_421_973_886,
        "592f404b664e7fa89ac5fe34f73d54a2be1fb397d5b6d8e616cafc2d55782b9b",
    ),
)

T = TypeVar("T")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def shell_output(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def retry(operation: Callable[[], T], *, attempts: int = 5) -> T:
    for attempt in range(attempts):
        try:
            return operation()
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def remote_file(entry: Any) -> RemoteFile:
    lfs = getattr(entry, "lfs", None)
    lfs_sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
    if lfs_sha256:
        kind = "lfs_sha256"
        identity = str(lfs_sha256)
    else:
        blob_id = getattr(entry, "blob_id", None)
        if not blob_id:
            raise RuntimeError(
                f"Hub file has no verifiable identity: {entry.rfilename}"
            )
        kind = "git_blob"
        identity = str(blob_id)
    if entry.size is None:
        raise RuntimeError(f"Hub file has no size: {entry.rfilename}")
    return RemoteFile(str(entry.rfilename), int(entry.size), kind, identity)


def inventory_prefix(entries: list[Any], prefix: str) -> dict[str, RemoteFile]:
    root = f"{prefix.rstrip('/')}/"
    return {
        entry.rfilename[len(root) :]: remote_file(entry)
        for entry in entries
        if entry.rfilename.startswith(root)
    }


def classify_destination(
    source: dict[str, RemoteFile], destination: dict[str, RemoteFile]
) -> str:
    if not destination:
        return "absent"
    exact = source.keys() == destination.keys() and all(
        source[path].size == destination[path].size
        and source[path].identity_kind == destination[path].identity_kind
        and source[path].identity == destination[path].identity
        for path in source
    )
    if exact:
        return "exact"
    raise RuntimeError("destination checkpoint is partial or divergent")


def build_copy_operations(
    files: dict[str, RemoteFile], *, checkpoint_path: str, source_revision: str
) -> list[CommitOperationCopy]:
    return [
        CommitOperationCopy(
            src_path_in_repo=files[relative].path,
            path_in_repo=f"{checkpoint_path}/{relative}",
            src_revision=source_revision,
            src_repo_id=SOURCE_REPO,
            src_repo_type="model",
        )
        for relative in sorted(files)
    ]


class Consolidator:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.api = HfApi()
        self.events_path = run_dir / "events.jsonl"
        self.source_cache: dict[str, list[Any]] = {}

    def event(self, kind: str, **fields: Any) -> None:
        row = {"timestamp": utc_now(), "event": kind, **fields}
        print(json.dumps(row, sort_keys=True), flush=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def model_entries(
        self, repo: str, revision: str | None = None
    ) -> tuple[str, list[Any]]:
        info = retry(
            lambda: self.api.model_info(repo, revision=revision, files_metadata=True)
        )
        return str(info.sha), list(info.siblings)

    def source_entries(self, revision: str) -> list[Any]:
        if revision not in self.source_cache:
            resolved, entries = self.model_entries(SOURCE_REPO, revision)
            if resolved != revision:
                raise RuntimeError(
                    f"source revision did not resolve exactly: {revision} -> {resolved}"
                )
            self.source_cache[revision] = entries
        return self.source_cache[revision]

    def source_tree(
        self, checkpoint: Checkpoint
    ) -> tuple[dict[str, RemoteFile], dict[str, dict[str, Any]]]:
        files = inventory_prefix(
            self.source_entries(checkpoint.source_revision), checkpoint.path
        )
        if not files:
            raise RuntimeError(f"empty source checkpoint: {checkpoint.path}")
        content: dict[str, dict[str, Any]] = {}
        for relative, metadata in sorted(files.items()):
            if metadata.identity_kind == "lfs_sha256":
                digest = metadata.identity
            else:
                local = retry(
                    lambda metadata=metadata: hf_hub_download(
                        SOURCE_REPO,
                        filename=metadata.path,
                        revision=checkpoint.source_revision,
                    )
                )
                digest = sha256_file(local)
            content[relative] = {"size": metadata.size, "sha256": digest}
        observed = {
            "files": len(files),
            "bytes": sum(item.size for item in files.values()),
            "tree_sha256": sha256_json(content),
        }
        expected = {
            "files": checkpoint.files,
            "bytes": checkpoint.bytes,
            "tree_sha256": checkpoint.tree_sha256,
        }
        if observed != expected:
            raise RuntimeError(
                f"source checkpoint contract mismatch for {checkpoint.path}: "
                f"expected={expected}, observed={observed}"
            )
        return files, content

    def target_tree(self, checkpoint: Checkpoint) -> tuple[str, dict[str, RemoteFile]]:
        revision, entries = self.model_entries(TARGET_REPO)
        return revision, inventory_prefix(entries, checkpoint.path)

    def copy_checkpoint(self, checkpoint: Checkpoint) -> dict[str, Any]:
        source, content = self.source_tree(checkpoint)
        before_revision, destination = self.target_tree(checkpoint)
        state = classify_destination(source, destination)
        self.event(
            "checkpoint_preflight",
            checkpoint=checkpoint.path,
            source_revision=checkpoint.source_revision,
            target_revision=before_revision,
            state=state,
        )
        if state == "absent":
            operations = build_copy_operations(
                source,
                checkpoint_path=checkpoint.path,
                source_revision=checkpoint.source_revision,
            )
            last_error: Exception | None = None
            for attempt in range(5):
                try:
                    commit = self.api.create_commit(
                        repo_id=TARGET_REPO,
                        repo_type="model",
                        operations=operations,
                        commit_message=f"Add {checkpoint.path} from immutable source",
                    )
                    self.event(
                        "checkpoint_copy_commit",
                        checkpoint=checkpoint.path,
                        commit=str(commit.oid),
                        attempt=attempt + 1,
                    )
                    break
                except Exception as error:
                    last_error = error
                    after_error_revision, after_error = self.target_tree(checkpoint)
                    if classify_destination(source, after_error) == "exact":
                        self.event(
                            "checkpoint_copy_recovered",
                            checkpoint=checkpoint.path,
                            commit=after_error_revision,
                            attempt=attempt + 1,
                        )
                        break
                    if attempt == 4:
                        raise
                    self.event(
                        "checkpoint_copy_retry",
                        checkpoint=checkpoint.path,
                        attempt=attempt + 1,
                        error=type(error).__name__,
                    )
                    time.sleep(2**attempt)
            else:
                raise last_error or RuntimeError("checkpoint copy failed")

        verified_revision, verified = self.target_tree(checkpoint)
        if classify_destination(source, verified) != "exact":
            raise AssertionError("unreachable destination verification state")
        receipt = {
            **asdict(checkpoint),
            "destination_revision": verified_revision,
            "destination_files": len(verified),
            "destination_bytes": sum(item.size for item in verified.values()),
            "verification": "exact path, size, and LFS SHA-256 or Git blob identity",
            "content": content,
        }
        self.event(
            "checkpoint_verified",
            checkpoint=checkpoint.path,
            destination_revision=verified_revision,
            files=len(verified),
            bytes=receipt["destination_bytes"],
            tree_sha256=checkpoint.tree_sha256,
        )
        return receipt

    def publish_docs(self, receipts: list[dict[str, Any]], code_commit: str) -> str:
        weights_revision, entries = self.model_entries(TARGET_REPO)
        forbidden = [
            entry.rfilename
            for entry in entries
            if entry.rfilename.startswith(("aft/", "full_aft/"))
        ]
        if forbidden:
            raise RuntimeError(
                f"AFT artifacts present in target repository: {forbidden[:5]}"
            )

        readme_template = retry(
            lambda: hf_hub_download(
                TARGET_REPO,
                filename="README.md",
                revision=TARGET_BOOTSTRAP_REVISION,
            )
        )
        lineage_template = retry(
            lambda: hf_hub_download(
                TARGET_REPO,
                filename="lineage_manifest.json",
                revision=TARGET_BOOTSTRAP_REVISION,
            )
        )
        readme = (
            Path(readme_template)
            .read_text()
            .replace("REPLACE_WITH_THE_PINNED_REPOSITORY_COMMIT", weights_revision)
        )
        readme = (
            readme.replace(
                "| SFT after four-epoch midtraining | `sft_4epoch/<coin\\|charter>/checkpoint-{4,48}` | 4 | pending completion and verification of run `20260808T090413Z-sft4` |",
                "| SFT after four-epoch midtraining | `sft_4epoch/<coin\\|charter>/checkpoint-{4,48}` | 4 | included |",
            )
            .replace(
                "Its output is added only after all four checkpoints pass exact remote-tree\nverification.",
                "All four Coin and Charter checkpoints are included after exact remote-tree\nverification.",
            )
            .replace(
                "copy-verification commit. The pending rows are declared there separately and\nare not represented as published files.",
                "copy-verification commit. Every declared checkpoint in this release is\nrepresented by published files.",
            )
        )
        lineage = json.loads(Path(lineage_template).read_text())
        resolved_inventory, _ = self.model_entries(
            SOURCE_REPO, SOURCE_INVENTORY_REVISION
        )
        if resolved_inventory != SOURCE_INVENTORY_REVISION:
            raise RuntimeError("source inventory revision did not resolve exactly")
        lineage["source_repository"]["inventory_revision"] = resolved_inventory
        receipt_by_path = {item["path"]: item for item in receipts}
        for stage in ("midtraining", "sft", "midtraining_4epoch"):
            for row in lineage["stages"][stage]["checkpoints"]:
                row["destination_verification_revision"] = receipt_by_path[row["path"]][
                    "destination_revision"
                ]
        sft_4epoch = lineage["stages"]["sft_4epoch"]
        sft_4epoch["status"] = "complete and exact-verified"
        sft_4epoch["checkpoints"] = []
        for path in (
            "sft_4epoch/coin/checkpoint-4",
            "sft_4epoch/coin/checkpoint-48",
            "sft_4epoch/charter/checkpoint-4",
            "sft_4epoch/charter/checkpoint-48",
        ):
            receipt = receipt_by_path[path]
            sft_4epoch["checkpoints"].append(
                {
                    key: receipt[key]
                    for key in (
                        "path",
                        "source_revision",
                        "files",
                        "bytes",
                        "tree_sha256",
                        "destination_revision",
                    )
                }
            )
        sft_4epoch.pop("expected_paths", None)
        sft_4epoch.pop("pending_paths", None)
        lineage["generated_at"] = utc_now()
        lineage["consolidation"].update(
            {
                "run_id": self.run_id,
                "source_code_commit": code_commit,
                "included_checkpoint_count": len(receipts),
                "included_file_count": sum(item["files"] for item in receipts),
                "included_bytes": sum(item["bytes"] for item in receipts),
            }
        )
        lineage["copy_verification"] = {
            "status": "exact",
            "weights_revision": weights_revision,
            "checkpoints": len(receipts),
            "files": sum(item["files"] for item in receipts),
            "bytes": sum(item["bytes"] for item in receipts),
            "method": "exact source/destination relative-path, size, LFS SHA-256 or Git blob identity, plus canonical content-tree SHA-256",
        }
        rendered_lineage = json.dumps(lineage, indent=2, sort_keys=True) + "\n"
        (self.run_dir / "README.md").write_text(readme)
        (self.run_dir / "lineage_manifest.json").write_text(rendered_lineage)
        commit = retry(
            lambda: self.api.create_commit(
                repo_id=TARGET_REPO,
                repo_type="model",
                operations=[
                    CommitOperationAdd("README.md", readme.encode()),
                    CommitOperationAdd(
                        "lineage_manifest.json", rendered_lineage.encode()
                    ),
                ],
                commit_message="Document AFT-free midtraining and SFT checkpoint lineage",
            )
        )
        final_revision, final_entries = self.model_entries(TARGET_REPO, str(commit.oid))
        if final_revision != str(commit.oid):
            raise RuntimeError("documentation commit did not resolve exactly")
        if any(
            entry.rfilename.startswith(("aft/", "full_aft/")) for entry in final_entries
        ):
            raise RuntimeError("AFT artifact appeared during final verification")
        self.event(
            "documentation_verified",
            weights_revision=weights_revision,
            final_revision=final_revision,
            files=len(final_entries),
        )
        return final_revision

    def upload_evidence(self) -> dict[str, Any]:
        self.api.create_repo(
            EVIDENCE_REPO, repo_type="dataset", private=False, exist_ok=True
        )
        prefix = f"runs/{self.run_id}/payload"
        commit = retry(
            lambda: self.api.upload_folder(
                repo_id=EVIDENCE_REPO,
                repo_type="dataset",
                folder_path=self.run_dir,
                path_in_repo=prefix,
                commit_message=f"Upload consolidation evidence for {self.run_id}",
            )
        )
        files = {
            path.relative_to(self.run_dir).as_posix(): {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(self.run_dir.rglob("*"))
            if path.is_file()
        }
        receipt = {
            "repo": EVIDENCE_REPO,
            "repo_type": "dataset",
            "prefix": prefix,
            "revision": str(commit.oid),
            "files": len(files),
            "bytes": sum(item["size"] for item in files.values()),
            "tree_sha256": sha256_json(files),
        }
        atomic_json(self.run_dir / "evidence_receipt.json", receipt)
        terminal = retry(
            lambda: self.api.upload_file(
                repo_id=EVIDENCE_REPO,
                repo_type="dataset",
                path_or_fileobj=str(self.run_dir / "evidence_receipt.json"),
                path_in_repo=f"runs/{self.run_id}/terminal/evidence_receipt.json",
                commit_message=f"Verify consolidation evidence for {self.run_id}",
            )
        )
        receipt["terminal_revision"] = str(terminal.oid)
        atomic_json(self.run_dir / "evidence_receipt.json", receipt)
        return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_dir or Path(__file__).resolve().parent / "runs" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[3]
    code_commit = shell_output("git", "-C", str(repo_root), "rev-parse", "HEAD")
    git_status = shell_output("git", "-C", str(repo_root), "status", "--short")
    if git_status:
        raise RuntimeError(
            f"source worktree must be clean before publication: {git_status}"
        )
    consolidator = Consolidator(run_dir, run_id)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": utc_now(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "source_code_commit": code_commit,
        "source_repo": SOURCE_REPO,
        "target_repo": TARGET_REPO,
        "target_bootstrap_revision": TARGET_BOOTSTRAP_REVISION,
        "evidence_repo": EVIDENCE_REPO,
        "checkpoints": [asdict(item) for item in CHECKPOINTS],
    }
    atomic_json(run_dir / "run_manifest.json", manifest)
    consolidator.event("run_started", run_id=run_id, source_code_commit=code_commit)
    try:
        target_info = retry(lambda: consolidator.api.model_info(TARGET_REPO))
        if bool(target_info.private):
            raise RuntimeError(
                "target repository is private; expected a public model repo"
            )
        initial_revision = str(target_info.sha)
        if initial_revision == TARGET_BOOTSTRAP_REVISION:
            consolidator.event("bootstrap_revision_verified", revision=initial_revision)
        else:
            consolidator.event(
                "resume_from_newer_revision",
                bootstrap_revision=TARGET_BOOTSTRAP_REVISION,
                revision=initial_revision,
            )
        receipts = [consolidator.copy_checkpoint(item) for item in CHECKPOINTS]
        atomic_json(run_dir / "checkpoint_receipts.json", receipts)
        final_revision = consolidator.publish_docs(receipts, code_commit)
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "completed_at": utc_now(),
            "status": "complete",
            "source_code_commit": code_commit,
            "target_repo": TARGET_REPO,
            "target_revision": final_revision,
            "checkpoint_count": len(receipts),
            "files": sum(item["files"] for item in receipts),
            "bytes": sum(item["bytes"] for item in receipts),
            "aft_artifacts": 0,
        }
        atomic_json(run_dir / "RUN_COMPLETE.json", result)
        consolidator.event("run_complete", **result)
        evidence = consolidator.upload_evidence()
        result["evidence"] = evidence
        atomic_json(run_dir / "RUN_COMPLETE.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        failure = {
            "schema_version": 1,
            "run_id": run_id,
            "failed_at": utc_now(),
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        atomic_json(run_dir / "RUN_FAILED.json", failure)
        consolidator.event("run_failed", **failure)
        try:
            consolidator.upload_evidence()
        except Exception as evidence_error:  # noqa: BLE001 - preserve primary failure
            consolidator.event(
                "evidence_upload_failed",
                error_type=type(evidence_error).__name__,
                error=str(evidence_error),
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
