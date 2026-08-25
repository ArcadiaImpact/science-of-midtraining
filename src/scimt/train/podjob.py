"""Generic Bellhop pod-job seam (BELLHOP_PORT.md §5/§7, task T1).

:class:`~scimt.train.axolotl.BellhopExecutor` runs *training stages* on
ephemeral pods; this module extracts its RunSpec-assembly seam so non-training
pod payloads (e.g. the uad arm-worklist workers) stop requiring a fork of the
executor. The executor itself is untouched — :func:`submit` reuses its
helpers via the public re-exports in :mod:`scimt.train.axolotl`
(``build_transfer_wheel``, ``pod_config_kwargs``, ``ENV_PASSTHROUGH``).

Contract (frozen in BELLHOP_PORT.md §7 — T2/T3 build against it):

- :class:`PodJob` — one ephemeral single-node pod running caller-built
  ``setup``/``run`` shell (config-first: the strings are the whole
  interface; no flag plumbing lives here).
- ``await submit(job)`` — dirty-tree refusal, wheel + source manifest staged
  into ``job.out_dir``, then ``bellhop.run(RunSpec, PodConfig)``. Bellhop is
  imported lazily (optional devbox-side dep; ``import scimt`` stays light).
- :func:`stage_transfer` — the staging step alone, exposed so the caller can
  learn ``wheel_rel``/``manifest_rel`` while building the setup string (the
  setup must install the wheel). :func:`submit` re-stages at submit time so
  provenance always matches the HEAD Bellhop pushes.

``RunSpec.gcs_base`` is always ``None``: bellhop's devbox-side GCS upload is
not our bus — pod-side ``upload_and_pin`` (pin-verified) owns artifact
placement, and the results pull carries only small artifacts.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .axolotl import (
    ENV_PASSTHROUGH,
    REPO_ROOT,
    PodSpec,
    build_transfer_wheel,
    pod_config_kwargs,
)
from .source_manifest import build_source_manifest

#: staged source-manifest filename inside ``PodJob.out_dir`` (matches the
#: executor's transfer set).
MANIFEST_NAME = ".scimt-source.json"

#: env keys :func:`submit` owns — a job may not silently override provenance.
RESERVED_ENV = ("SCIMT_SOURCE_COMMIT", "SCIMT_SOURCE_MANIFEST")


@dataclass(frozen=True)
class TransferBundle:
    """Repo-relative paths of the staged transfer set, plus its provenance.

    ``wheel_rel``/``manifest_rel`` are relative to the repo checkout root —
    exactly the paths the pod sees after Bellhop pushes the checkout, so the
    caller interpolates them into the setup string verbatim.
    """

    wheel_rel: str
    manifest_rel: str
    commit: str


@dataclass(frozen=True)
class PodJob:
    """One ephemeral Bellhop pod running a caller-built shell payload."""

    pod: PodSpec          # reuses the existing dataclass; nodes must be 1
    slug: str             # bellhop pod name suffix -> "scimt-<slug>"
    setup: str            # pod setup shell (caller-built, e.g. pod_setup.py)
    run: str              # pod run shell
    out_dir: Path         # under REPO_ROOT; wheel + manifest staged here
    results_subdir: str   # pod-side dir bellhop pulls back
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pod.nodes != 1:
            raise ValueError(
                f"PodJob requires a single-node pod, got nodes={self.pod.nodes} "
                "(multi-node clusters stay on BellhopExecutor's training path)"
            )
        if self.pod.extra_env:
            raise ValueError(
                "PodSpec.extra_env is not consumed by PodJob — put pod env in "
                "PodJob.env (silent ignore would violate config-first)"
            )
        if not self.slug:
            raise ValueError("PodJob.slug must be non-empty (names the pod)")
        if not self.setup or not self.run:
            raise ValueError("PodJob.setup and PodJob.run must be non-empty")
        if not self.results_subdir:
            raise ValueError(
                "PodJob.results_subdir must be non-empty (bellhop pulls it back)"
            )
        for key, value in self.env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(
                    f"PodJob.env must map str -> str, got {key!r}: {value!r}"
                )
        clashes = sorted(set(self.env) & set(RESERVED_ENV))
        if clashes:
            raise ValueError(
                f"PodJob.env may not set provenance keys {clashes} — "
                "submit() derives them from the staged source manifest"
            )


def stage_transfer(out_dir: Path) -> TransferBundle:
    """Stage the scimt wheel + source manifest for the Bellhop code push.

    Dirty-tree refusal lives here: :func:`build_source_manifest` raises on a
    dirty tracked checkout *before* the wheel build (``git archive HEAD``
    would otherwise silently ship a stale snapshot of uncommitted work).
    """

    out_dir = Path(out_dir)
    try:
        out_dir.resolve().relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(
            f"PodJob.out_dir must live under the repo checkout {REPO_ROOT} "
            "(the staged wheel and manifest ride the code push)"
        ) from error
    out_dir.mkdir(parents=True, exist_ok=True)
    # Dirty refusal FIRST (git archive HEAD would silently ship a stale
    # snapshot of uncommitted work), then the wheel, then the manifest — the
    # manifest scans out_dir too, and wheel zips are not byte-deterministic
    # across builds, so it must hash the wheel that actually ships. (The
    # live canary caught the old manifest-then-wheel order as a pod-side
    # source-file mismatch on the re-staged wheel.)
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            f"refusing to stage from a dirty tracked checkout:\n{dirty}"
        )
    wheel = build_transfer_wheel(out_dir)
    manifest_path = out_dir / MANIFEST_NAME
    manifest = build_source_manifest(REPO_ROOT, manifest_path)
    return TransferBundle(
        wheel_rel=str(wheel.resolve().relative_to(REPO_ROOT)),
        manifest_rel=str(manifest_path.resolve().relative_to(REPO_ROOT)),
        commit=manifest["commit"],
    )


async def submit(job: PodJob) -> None:
    """Stage the transfer set and run the job on an ephemeral Bellhop pod.

    Re-stages via :func:`stage_transfer` even if the caller already staged
    (identical, deterministic outputs for the same HEAD) so the provenance
    env always describes the exact checkout Bellhop pushes — and so a tree
    that went dirty between planning and submission refuses loudly.
    """

    import bellhop  # lazy: optional devbox-side dep, never a pod/CI import

    bundle = stage_transfer(job.out_dir)
    spec = bellhop.RunSpec(
        slug=job.slug,
        codebase=str(REPO_ROOT),
        setup=job.setup,
        run=job.run,
        results_subdir=job.results_subdir,
        local_out=str(job.out_dir),
        gcs_base=None,  # pod-side upload_and_pin is the artifact bus
        env={
            "PYTHONDONTWRITEBYTECODE": "1",
            "SCIMT_SOURCE_COMMIT": bundle.commit,
            "SCIMT_SOURCE_MANIFEST": bundle.manifest_rel,
            **{k: v for k in ENV_PASSTHROUGH if (v := os.environ.get(k))},
            **job.env,
        },
    )
    pod_config = bellhop.PodConfig(**pod_config_kwargs(job.pod, job.slug))
    await bellhop.run(spec, pod_config)
