"""Parallel, duplicate-free checkpoint publication for the scale-up.

Two departures from the 12B upload path, both costed in
[UPLOAD_ARCHITECTURE.md](UPLOAD_ARCHITECTURE.md):

1. **Concurrent hash + LFS push, serial commits.** ``dispatch_midtrain_v1``
   uploads a stage's checkpoints one at a time, and ``upload_tree`` sha256s
   the whole tree *before* it sends any bytes — so one checkpoint's local
   hashing pass blocks the next one's network transfer. Here the hashing and
   the LFS byte push run concurrently across a stage's checkpoints, while the
   commits stay strictly serial: two commits to the same repo never race, so
   there is nothing for HF to reject with a conflict.

2. **The duplicate weights are not published at intermediate checkpoints.**
   Every full-state checkpoint carries the weights twice —
   ``model.safetensors`` (HF layout) and ``pytorch_model_fsdp.bin`` (FSDP
   layout) — which is 28% of a 4B checkpoint and ~55 GB of a 27B one. The
   duplicate is kept at the two checkpoints the D2 contract exists to make
   resumable (post-warmup and final) and omitted at the three in between,
   where ``model.safetensors`` plus the optimizer/scheduler/RNG state is
   enough to reconstruct a run (FSDP2 reshards from safetensors;
   ``cpu_ram_efficient_loading`` is already set in every stage YAML).

The omission is recorded in each checkpoint's manifest (``omitted``) so a
future resume can see it was deliberate rather than a lost upload.

Nothing here changes *what is computed* — only which bytes are shipped and in
what order (the §"Error loud, warn on degraded" corollary in CLAUDE.md).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

from experiments.prior_coins.dispatch_midtrain_v1.pod import train as artifacts

#: byte-for-byte duplicate of ``model.safetensors`` in FSDP layout
DUPLICATE_WEIGHTS = "pytorch_model_fsdp.bin"

#: how many checkpoints hash + preupload at once. 3 overlaps hashing with
#: transfer without putting five simultaneous multi-GB read streams on one
#: container disk.
DEFAULT_WORKERS = 3


@dataclasses.dataclass(frozen=True)
class CheckpointJob:
    """One checkpoint to publish. ``step`` decides duplicate retention."""

    label: str
    step: int
    local_dir: Path
    remote_prefix: str
    manifest_path: Path
    commit_message: str


def omitted_for(step: int, keep_steps: Collection[int]) -> tuple[str, ...]:
    """Filenames not shipped for ``step`` — the FSDP duplicate, except at the
    resume-critical boundaries."""
    return () if step in set(keep_steps) else (DUPLICATE_WEIGHTS,)


def filtered_manifest(
    local_dir: Path, omit: Collection[str]
) -> dict[str, dict[str, Any]]:
    """``hash_tree`` minus ``omit``, refusing to drop the last real weights.

    The guard matters: omitting the duplicate is only safe while
    ``model.safetensors`` survives, and a future cadence change must not be
    able to silently publish a checkpoint with no weights in it.
    """
    manifest = artifacts.hash_tree(local_dir)
    omit = set(omit)
    kept = {
        relative: metadata
        for relative, metadata in manifest.items()
        if PurePosixPath(relative).name not in omit
    }
    if not kept:
        raise ValueError(f"every file in {local_dir} was omitted")
    if omit and not any(
        PurePosixPath(relative).name.endswith(".safetensors") for relative in kept
    ):
        raise RuntimeError(
            f"{local_dir}: omitting {sorted(omit)} would leave the checkpoint "
            "with no .safetensors weights"
        )
    return kept


def _additions(job: CheckpointJob, manifest: Mapping[str, Any]) -> list[Any]:
    from huggingface_hub import CommitOperationAdd

    prefix = job.remote_prefix.strip("/")
    return [
        CommitOperationAdd(
            path_in_repo=f"{prefix}/{relative}" if prefix else relative,
            path_or_fileobj=job.local_dir / relative,
        )
        for relative in sorted(manifest)
    ]


def prepare(
    api: Any,
    job: CheckpointJob,
    *,
    repo_id: str,
    repo_type: str,
    omit: Collection[str],
) -> tuple[dict[str, dict[str, Any]], list[Any]]:
    """Hash the tree and push its LFS bytes, without committing.

    Safe to run concurrently for several checkpoints: nothing is referenced in
    the repo until :func:`commit` runs, so a later validation failure leaves
    unreferenced blobs (which HF collects) rather than a published checkpoint.
    """
    manifest = filtered_manifest(job.local_dir, omit)
    additions = _additions(job, manifest)
    artifacts.event(
        "checkpoint_preupload_started",
        repo_id=repo_id,
        local=str(job.local_dir),
        remote=job.remote_prefix,
        files=len(manifest),
        omitted=sorted(omit),
    )
    artifacts._retry(
        f"preupload {job.remote_prefix}",
        lambda: api.preupload_lfs_files(repo_id, additions, repo_type=repo_type),
    )
    artifacts.event(
        "checkpoint_preupload_finished",
        repo_id=repo_id,
        remote=job.remote_prefix,
        files=len(manifest),
    )
    return manifest, additions


def _verify(
    api: Any,
    *,
    repo_id: str,
    repo_type: str,
    manifest: Mapping[str, Mapping[str, Any]],
    prefix: str,
    revision: str,
    read_remote_file: Callable[[str, str, str], bytes] | None,
) -> None:
    if read_remote_file is None:
        from huggingface_hub import hf_hub_download

        def read_remote_file(repo: str, path: str, commit: str) -> bytes:
            downloaded = hf_hub_download(
                repo,
                path,
                repo_type=repo_type,
                revision=commit,
                token=getattr(api, "token", None),
                force_download=True,
            )
            return Path(downloaded).read_bytes()

    reader = read_remote_file
    artifacts._retry(
        f"verify {prefix}",
        lambda: artifacts.verify_remote_files(
            manifest,
            artifacts._remote_index(
                api, repo_id, revision=revision, repo_type=repo_type
            ),
            prefix=prefix,
            read_regular_file=lambda path: reader(repo_id, path, revision),
        ),
    )


def commit(
    api: Any,
    job: CheckpointJob,
    prepared: tuple[dict[str, dict[str, Any]], list[Any]],
    *,
    repo_id: str,
    repo_type: str,
    omit: Collection[str],
    read_remote_file: Callable[[str, str, str], bytes] | None = None,
) -> dict[str, Any]:
    """Commit pre-uploaded bytes, verify them, and return the usual receipt.

    Emits ``upload_started``/``upload_verified`` with the same field names the
    12B path used, so existing log readers and completion waiters keep working.
    """
    manifest, additions = prepared
    artifacts.atomic_json(
        job.manifest_path,
        {
            "root": str(job.local_dir),
            "tree_sha256": artifacts.sha256_json(manifest),
            "files": manifest,
            "omitted": sorted(omit),
        },
    )
    artifacts.event(
        "upload_started",
        repo_id=repo_id,
        local=str(job.local_dir),
        remote=job.remote_prefix,
        files=len(manifest),
    )
    result = artifacts._retry(
        f"commit {job.remote_prefix}",
        lambda: api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=additions,
            commit_message=job.commit_message,
        ),
    )
    revision = getattr(result, "oid", None)
    if not isinstance(revision, str) or not revision:
        raise RuntimeError(
            f"commit {job.remote_prefix} returned no immutable commit ID"
        )
    _verify(
        api,
        repo_id=repo_id,
        repo_type=repo_type,
        manifest=manifest,
        prefix=job.remote_prefix,
        revision=revision,
        read_remote_file=read_remote_file,
    )
    receipt = {
        "repo_id": repo_id,
        "remote_prefix": job.remote_prefix,
        "commit_oid": revision,
        "commit_url": getattr(result, "commit_url", None),
        "tree_sha256": artifacts.sha256_json(manifest),
        "files": len(manifest),
        "omitted": sorted(omit),
        "verified_at": artifacts.utc_now(),
    }
    artifacts.event("upload_verified", **receipt)
    return receipt


def upload_checkpoints(
    api: Any,
    *,
    repo_id: str,
    jobs: Collection[CheckpointJob],
    keep_steps: Collection[int],
    repo_type: str = "model",
    workers: int = DEFAULT_WORKERS,
    read_remote_file: Callable[[str, str, str], bytes] | None = None,
) -> dict[str, dict[str, Any]]:
    """Publish a stage's checkpoints: concurrent bytes, serial commits."""
    ordered = sorted(jobs, key=lambda job: job.step)
    omit = {job.label: omitted_for(job.step, keep_steps) for job in ordered}
    prepared: dict[str, tuple[dict[str, dict[str, Any]], list[Any]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                prepare,
                api,
                job,
                repo_id=repo_id,
                repo_type=repo_type,
                omit=omit[job.label],
            ): job
            for job in ordered
        }
        for future, job in futures.items():
            # .result() re-raises in the caller's thread: a failed hash or
            # push aborts the stage before anything is committed.
            prepared[job.label] = future.result()
    return {
        job.label: commit(
            api,
            job,
            prepared[job.label],
            repo_id=repo_id,
            repo_type=repo_type,
            omit=omit[job.label],
            read_remote_file=read_remote_file,
        )
        for job in ordered
    }


def step_of(checkpoint: Path) -> int:
    """``.../checkpoint-62`` -> 62."""
    tail = checkpoint.name.rsplit("-", 1)[-1]
    if not tail.isdigit():
        raise ValueError(f"not a checkpoint directory: {checkpoint}")
    return int(tail)


def install(
    base_module: Any,
    *,
    keep_steps: Collection[int],
    remote_prefix_of: Callable[[Path], str],
    workers: int = DEFAULT_WORKERS,
) -> "PrewarmingUploader":
    """Swap ``base_module.upload_tree`` for a :class:`PrewarmingUploader`.

    Idempotent: a second call on an already-installed module returns the
    existing uploader rather than wrapping a wrapper (both midtraining runners
    share ``install_upload_layout``, and the control runner configures the same
    base module).
    """
    existing = getattr(base_module, "upload_tree", None)
    if isinstance(existing, PrewarmingUploader):
        return existing
    uploader = PrewarmingUploader(
        existing,
        keep_steps=keep_steps,
        remote_prefix_of=remote_prefix_of,
        workers=workers,
    )
    base_module.upload_tree = uploader
    return uploader


class PrewarmingUploader:
    """``base.upload_tree`` replacement for the midtraining path.

    The 12B trainer's upload loop is inside ``_train_arm`` and calls
    ``upload_tree`` once per checkpoint, serially. Rather than reimplement
    that function, this stands in for ``upload_tree``: the first checkpoint
    call kicks off concurrent hash+push for *every* checkpoint of the stage,
    then each call commits its own (the loop stays serial, as does commit
    order). Non-checkpoint uploads — artifacts, compact logs, terminal record
    — fall through to the original ``upload_tree`` untouched.
    """

    def __init__(
        self,
        original: Callable[..., dict[str, Any]],
        *,
        keep_steps: Collection[int],
        remote_prefix_of: Callable[[Path], str],
        workers: int = DEFAULT_WORKERS,
    ) -> None:
        self._original = original
        self._keep_steps = set(keep_steps)
        self._remote_prefix_of = remote_prefix_of
        self._workers = workers
        self._checkpoints: dict[Path, int] = {}
        self._prepared: dict[Path, tuple[dict[str, Any], list[Any]]] = {}

    def register(self, checkpoints: Mapping[str, Path]) -> None:
        """Called with the output of ``select_checkpoints``."""
        self._checkpoints = {
            Path(path).resolve(): step_of(Path(path))
            for path in checkpoints.values()
        }

    def _prewarm(self, api: Any, repo_id: str, repo_type: str) -> None:
        if self._prepared or not self._checkpoints:
            return
        jobs = {
            path: CheckpointJob(
                label=path.name,
                step=step,
                local_dir=path,
                remote_prefix=self._remote_prefix_of(path),
                manifest_path=path / "unused",  # commit() writes the real one
                commit_message="unused",
            )
            for path, step in self._checkpoints.items()
        }
        with ThreadPoolExecutor(max_workers=max(1, self._workers)) as pool:
            futures = {
                pool.submit(
                    prepare,
                    api,
                    job,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    omit=omitted_for(job.step, self._keep_steps),
                ): path
                for path, job in jobs.items()
            }
            for future, path in futures.items():
                self._prepared[path] = future.result()

    def __call__(
        self,
        api: Any,
        *,
        repo_id: str,
        local_dir: Path,
        remote_prefix: str,
        manifest_path: Path,
        commit_message: str,
        repo_type: str = "model",
        read_remote_file: Callable[[str, str, str], bytes] | None = None,
    ) -> dict[str, Any]:
        resolved = Path(local_dir).resolve()
        if resolved not in self._checkpoints:
            return self._original(
                api,
                repo_id=repo_id,
                local_dir=local_dir,
                remote_prefix=remote_prefix,
                manifest_path=manifest_path,
                commit_message=commit_message,
                repo_type=repo_type,
                read_remote_file=read_remote_file,
            )
        self._prewarm(api, repo_id, repo_type)
        step = self._checkpoints[resolved]
        job = CheckpointJob(
            label=resolved.name,
            step=step,
            local_dir=resolved,
            remote_prefix=remote_prefix,
            manifest_path=Path(manifest_path),
            commit_message=commit_message,
        )
        prepared = self._prepared[resolved]
        if job.remote_prefix.strip("/") != self._remote_prefix_of(resolved).strip("/"):
            raise RuntimeError(
                "pre-uploaded bytes were staged under "
                f"{self._remote_prefix_of(resolved)!r} but the trainer asked "
                f"for {job.remote_prefix!r}"
            )
        return commit(
            api,
            job,
            prepared,
            repo_id=repo_id,
            repo_type=repo_type,
            omit=omitted_for(step, self._keep_steps),
            read_remote_file=read_remote_file,
        )
