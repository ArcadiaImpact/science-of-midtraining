"""Ship the newest complete midtrain RESUME checkpoint to the Hub, overwriting.

Insurance against a lost pod during the long full-parameter leg (the 1B
charter row: ~27 h on 8xB200, container-disk only, so a zero-balance stop
takes the disk with it). The checkpoint-schedule plugin
(scimt.train.axolotl_plugins / scimt.train.resume_checkpoint) saves a full
sharded checkpoint every N steps and marks it complete; this sidecar

  1. watches ``<run_dir>/checkpoints`` for marked, settled checkpoints,
  2. uploads the NEWEST one to ``<hub_arm_prefix>/midtrain/resume/latest/`` in
     the row's model repo as ONE commit that deletes whatever the previous
     upload left there (the Hub's commit cap is per repo; one commit per
     checkpoint every ~2 h is nothing), with a RESUME_MANIFEST.json beside
     the files,
  3. verifies the remote tree at the resulting commit (every file's size;
     sha256 for files small enough to hash twice cheaply), and
  4. writes a local receipt, then goes back to watching.

A checkpoint that is superseded by a newer complete one before its turn is
skipped. Nothing local is ever deleted here. Started and stopped by
pod/chain.py (``--stop-file``: finish any in-flight commit, then exit).
This is BACKUP ONLY: resuming from the uploaded tree is not wired into the
chain (decision 2026-09-08); the files are a standard Trainer checkpoint.

    FINAL_V1_PROFILE=<row> python3 resume_upload.py --arm charter \
        --checkpoints-dir <run_dir>/checkpoints --receipt <run_dir>/RESUME_UPLOAD_LATEST.json \
        --stop-file <run_dir>/RESUME_UPLOAD_STOP
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POD = Path(__file__).resolve().parent
EXP = POD.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scimt.train.resume_checkpoint import RESUME_MARKER, resume_checkpoints  # noqa: E402

MANIFEST_NAME = "RESUME_MANIFEST.json"
#: A checkpoint counts as complete when its marker exists AND nothing under it
#: changed for this long (the marker is written after trainer_state.json, the
#: Trainer's last file; the settle window is belt-and-braces against a rank
#: still flushing).
SETTLE_SECONDS = 60
#: Files up to this size get a local sha256 that is checked against the Hub's
#: LFS digest; larger shards (the ~5 GB safetensors / DCP files) are verified
#: by size only, so verification does not re-read 440 GB.
SHA256_MAX_BYTES = 64 * 1024 * 1024
MAX_ATTEMPTS = 5
DEFAULT_COOLDOWN_S = 20 * 60
PARTIAL_SUFFIXES = (".tmp", ".partial", ".incomplete", ".lock")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def _cooldown_from(err: str) -> int:
    m = re.search(r"retry this action in (\d+) minutes?", err)
    if m:
        return int(m.group(1)) * 60 + 60
    m = re.search(r"Retry after (\d+) seconds", err)
    if m:
        return int(m.group(1)) + 60
    return DEFAULT_COOLDOWN_S


# ------------------------------------------------------------ local side


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_checkpoints(
    checkpoints_dir: Path, *, settle_seconds: float = SETTLE_SECONDS,
    now: float | None = None,
) -> list[tuple[int, Path]]:
    """Marked resume checkpoints whose tree has been quiet for the settle window
    and carries no partial files, oldest first."""
    now = time.time() if now is None else now
    out: list[tuple[int, Path]] = []
    for step, path in resume_checkpoints(checkpoints_dir):
        newest = 0.0
        partial = False
        for file in path.rglob("*"):
            if not file.is_file():
                continue
            if file.name.endswith(PARTIAL_SUFFIXES):
                partial = True
                break
            newest = max(newest, file.stat().st_mtime)
        if partial or now - newest < settle_seconds:
            continue
        out.append((step, path))
    return out


def snapshot_files(path: Path, *, sha256_max_bytes: int = SHA256_MAX_BYTES) -> dict[str, dict[str, Any]]:
    """{relative path: {size, sha256?}} for every regular file under ``path``."""
    files: dict[str, dict[str, Any]] = {}
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        size = file.stat().st_size
        entry: dict[str, Any] = {"size": size}
        if size <= sha256_max_bytes:
            entry["sha256"] = sha256_file(file)
        files[file.relative_to(path).as_posix()] = entry
    return files


def build_manifest(*, step: int, path: Path, files: dict[str, dict[str, Any]],
                   profile: str, arm: str, fingerprint: dict[str, Any],
                   prefix: str) -> dict[str, Any]:
    marker = json.loads((path / RESUME_MARKER).read_text())
    return {
        "schema_version": "scimt_resume_upload_v1",
        "kind": "midtrain_resume_checkpoint",
        "profile": profile,
        "arm": arm,
        "step": step,
        "checkpoint_dir": path.name,
        "prefix": prefix,
        "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT"),
        "fingerprint": fingerprint,
        "marker": marker,
        "files": files,
        "total_bytes": sum(f["size"] for f in files.values()),
        "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": ("backup only: the chain does not resume from this tree; it is "
                 "a standard Trainer checkpoint (params + optimizer/scheduler/"
                 "RNG state) usable via resume_from_checkpoint"),
    }


# -------------------------------------------------------------- Hub side


def plan_operations(files: dict[str, dict[str, Any]], remote_paths: set[str],
                    prefix: str, path: Path, manifest_bytes: bytes) -> list[Any]:
    """One commit: delete what the previous upload left that we do not
    re-upload, add every checkpoint file plus the manifest (re-adding an
    existing path overwrites it; adding and deleting the same path in one
    commit is refused by the Hub, hence the split)."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    wanted = {f"{prefix}/{rel}" for rel in files} | {f"{prefix}/{MANIFEST_NAME}"}
    operations: list[Any] = [
        CommitOperationDelete(path_in_repo=remote)
        for remote in sorted(remote_paths - wanted)
    ]
    operations.extend(
        CommitOperationAdd(path_in_repo=f"{prefix}/{rel}", path_or_fileobj=str(path / rel))
        for rel in files
    )
    operations.append(CommitOperationAdd(
        path_in_repo=f"{prefix}/{MANIFEST_NAME}", path_or_fileobj=io.BytesIO(manifest_bytes)))
    return operations


def remote_tree(api: Any, repo: str, prefix: str, revision: str | None = None) -> dict[str, Any]:
    """{path: entry} for files under prefix; empty when the prefix is absent."""
    try:
        entries = api.list_repo_tree(repo, repo_type="model", path_in_repo=prefix,
                                     recursive=True, revision=revision)
        return {e.path: e for e in entries if getattr(e, "size", None) is not None}
    except Exception as exc:  # noqa: BLE001 -- EntryNotFound / RepositoryNotFound variants
        if "404" in str(exc) or "not found" in str(exc).lower() or type(exc).__name__.endswith("NotFoundError"):
            return {}
        raise


def verify_remote(entries: dict[str, Any], files: dict[str, dict[str, Any]],
                  prefix: str, manifest_bytes: bytes) -> None:
    want = {f"{prefix}/{rel}": meta for rel, meta in files.items()}
    want[f"{prefix}/{MANIFEST_NAME}"] = {
        "size": len(manifest_bytes), "sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    problems = []
    for remote, meta in want.items():
        entry = entries.get(remote)
        if entry is None:
            problems.append(f"missing {remote}")
            continue
        if entry.size != meta["size"]:
            problems.append(f"size {remote}: remote {entry.size} != local {meta['size']}")
            continue
        lfs = getattr(entry, "lfs", None)
        digest = (lfs.get("sha256") if isinstance(lfs, dict)
                  else getattr(lfs, "sha256", None)) if lfs else None
        if digest is not None and "sha256" in meta and digest != meta["sha256"]:
            problems.append(f"sha256 {remote}")
    stray = sorted(set(entries) - set(want))
    if stray:
        problems.append(f"stray remote files not from this upload: {stray[:5]}")
    if problems:
        raise RuntimeError("remote verification failed: " + "; ".join(problems))


@dataclass
class Uploader:
    api: Any
    repo: str
    prefix: str
    profile: str
    arm: str
    fingerprint: dict[str, Any]
    receipt: Path
    sha256_max_bytes: int = SHA256_MAX_BYTES
    max_attempts: int = MAX_ATTEMPTS
    sleep: Any = time.sleep

    def upload(self, step: int, path: Path) -> str:
        files = snapshot_files(path, sha256_max_bytes=self.sha256_max_bytes)
        if not files or "trainer_state.json" not in files:
            raise RuntimeError(f"{path}: not a complete Trainer checkpoint")
        manifest = build_manifest(step=step, path=path, files=files, profile=self.profile,
                                  arm=self.arm, fingerprint=self.fingerprint, prefix=self.prefix)
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        total_gb = manifest["total_bytes"] / 1e9
        for attempt in range(1, self.max_attempts + 1):
            started = time.time()
            try:
                remote = remote_tree(self.api, self.repo, self.prefix)
                operations = plan_operations(files, set(remote), self.prefix, path, manifest_bytes)
                log(f"step {step}: attempt {attempt}/{self.max_attempts}: {len(files)} files, "
                    f"{total_gb:.1f} GB, replacing {len(remote)} remote files under {self.prefix}")
                commit = self.api.create_commit(
                    repo_id=self.repo, repo_type="model", operations=operations,
                    commit_message=f"{self.prefix}: resume checkpoint step {step} "
                                   f"({self.profile}/{self.arm}, overwrite)")
                oid = commit.oid
                verify_remote(remote_tree(self.api, self.repo, self.prefix, revision=oid),
                              files, self.prefix, manifest_bytes)
            except Exception as exc:  # noqa: BLE001 -- the Hub's message verbatim
                err = str(exc)
                log(f"step {step}: attempt {attempt} FAILED after "
                    f"{(time.time() - started) / 60:.1f} min: {err[:400]}")
                if attempt == self.max_attempts:
                    raise
                self.sleep(_cooldown_from(err) if "429" in err else 120)
                continue
            receipt = {"step": step, "checkpoint_dir": str(path), "repo": self.repo,
                       "prefix": self.prefix, "commit": oid, "files": len(files),
                       "total_bytes": manifest["total_bytes"],
                       "minutes": round((time.time() - started) / 60, 2),
                       "uploaded_at": manifest["uploaded_at"]}
            temporary = self.receipt.with_suffix(".tmp")
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.replace(self.receipt)
            log(f"step {step}: UPLOADED and verified at {oid} in {receipt['minutes']} min")
            return oid
        raise RuntimeError("unreachable")


def last_uploaded_step(receipt: Path) -> int | None:
    if not receipt.is_file():
        return None
    try:
        return int(json.loads(receipt.read_text())["step"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def watch(uploader: Uploader, checkpoints_dir: Path, stop_file: Path, *,
          poll_seconds: float, settle_seconds: float = SETTLE_SECONDS,
          sleep: Any = time.sleep, clock: Any = time.time) -> int:
    """Upload the newest complete checkpoint whenever one newer than the last
    receipt appears; exit 0 once the stop file exists and nothing newer waits."""
    last = last_uploaded_step(uploader.receipt)
    log(f"watching {checkpoints_dir} (last uploaded step: {last}); stop file {stop_file}")
    while True:
        complete = complete_checkpoints(checkpoints_dir, settle_seconds=settle_seconds, now=clock())
        newest = complete[-1] if complete else None
        if newest is not None and (last is None or newest[0] > last):
            skipped = [s for s, _ in complete if s != newest[0] and (last is None or s > last)]
            if skipped:
                log(f"skipping steps {skipped}: step {newest[0]} is already complete")
            try:
                uploader.upload(*newest)
            except Exception as exc:  # noqa: BLE001
                log(f"step {newest[0]}: giving up on this checkpoint ({str(exc)[:200]}); "
                    "the previous Hub copy is intact; will try the next one")
            last = max(last or 0, newest[0])
            continue
        if stop_file.is_file():
            log("stop requested and nothing newer to upload; exiting")
            return 0
        sleep(poll_seconds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--checkpoints-dir", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--stop-file", type=Path, required=True)
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    args = ap.parse_args()

    import contracts as C  # noqa: E402  (FINAL_V1_PROFILE selects the row)
    from huggingface_hub import HfApi

    if not C.MIDTRAIN_RESUME_EVERY_STEPS:
        log(f"profile {C.PROFILE.name} has no resume cadence; nothing to do")
        return 0
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN must be set in the environment (never in argv)")
    repo = os.environ.get("FINAL_V1_MODEL_REPO") or C.model_repo_for(C.PROFILE.name)
    prefix = f"{C.hub_arm_prefix(args.arm)}/midtrain/resume/latest"
    uploader = Uploader(api=HfApi(), repo=repo, prefix=prefix, profile=C.PROFILE.name,
                        arm=args.arm, fingerprint=C.fingerprint(args.arm), receipt=args.receipt)
    log(f"{C.PROFILE.name}/{args.arm}: resume uploads -> {repo}/{prefix}")
    return watch(uploader, args.checkpoints_dir, args.stop_file, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
