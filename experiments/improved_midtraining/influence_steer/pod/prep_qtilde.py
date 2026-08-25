"""Phase A step 1: pull u0 + metric from GCS and materialize q_tilde blocks.

q_tilde[dir] = u0[ROW_dir] * metric, streamed entry-by-entry off the two
memmaps (never a [2, P] tensor anywhere, never GPU — the cuBLAS int32-lda
trap does not get a chance to exist) into per-layer fp32 safetensors blocks
keyed by parameter-manifest entry name. The parameter manifest digest and P
are asserted against the pinned flagship values BEFORE the first block byte
is written; every block gets a sha256 recorded in qtilde_blocks.json.

Inputs arrive concurrently: the ~129 GB GCS pull runs while the ckpt-124
prefix downloads and the manifest is built from the real (CPU, bf16) model.
"""

# ruff: noqa: E402 - pod modules pin sys.path before experiment imports.

from __future__ import annotations

import asyncio
import dataclasses
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.influence_steer import contracts
from experiments.improved_midtraining.influence_steer.pod import common

_LAYER_RE = re.compile(r"^(.*\blayers\.)(\d+)(\..*)$")


@dataclasses.dataclass(frozen=True)
class PrepConfig:
    # Streaming window for any op that would otherwise materialize a large
    # contiguous slice (elements, fp32). 2**27 = 512 MiB per buffer.
    window_elements: int = 1 << 27

    def __post_init__(self) -> None:
        if self.window_elements < 1 << 20:
            raise ValueError("window_elements must be at least 2**20")


def block_key(entry_name: str) -> str:
    """Per-layer block naming: layers.N -> layer_NNN, everything else pooled
    into the 'root' block (final norm; ~4k params)."""
    match = _LAYER_RE.match(entry_name)
    if match:
        return f"layer_{int(match.group(2)):03d}"
    return "root"


def plan_blocks(entries: list[Any]) -> dict[str, list[Any]]:
    blocks: dict[str, list[Any]] = {}
    for entry in entries:
        blocks.setdefault(block_key(entry.name), []).append(entry)
    return blocks


async def _pull_inputs(scratch: Path) -> tuple[Path, Path]:
    u0_path = scratch / "gcs" / "u_damping0_stage0.npy"
    metric_path = scratch / "gcs" / "metric_midtrain.f32"
    await asyncio.gather(
        common.rclone_pull(
            contracts.GCS_U0_NPY, u0_path, expected_bytes=contracts.U0_NPY_BYTES
        ),
        common.rclone_pull(
            contracts.GCS_METRIC_F32,
            metric_path,
            expected_bytes=contracts.METRIC_F32_BYTES,
        ),
    )
    return u0_path, metric_path


def _load_manifest_from_ckpt(env: common.PodEnv) -> tuple[Path, Any]:
    """Download + digest-gate ckpt-124, build + gate the parameter manifest."""
    import torch  # noqa: F401 - transformers needs it; keep import explicit
    from scimt.data_attribution.runner import _load_model

    ckpt_dir = common.fetch_ckpt124(env)
    common.log("loading ckpt-124 on CPU (bf16) to build the manifest...")
    model = _load_model(ckpt_dir, dtype=contracts.MODEL_DTYPE, device="cpu")
    manifest = common.build_manifest(model)
    del model
    gc.collect()
    common.log(
        f"manifest OK: digest {manifest.digest()[:12]}..., "
        f"P {manifest.included_numel:,}"
    )
    return ckpt_dir, manifest


def write_blocks(
    cfg: PrepConfig,
    manifest: Any,
    u0_path: Path,
    metric_path: Path,
    out_root: Path,
) -> dict[str, Any]:
    """Stream q_tilde per entry into per-layer safetensors; sha256 per block."""
    import numpy as np
    from safetensors.numpy import save_file

    u0 = np.load(str(u0_path), mmap_mode="r")
    if u0.shape != (2, contracts.P_TOTAL) or u0.dtype != np.float32:
        raise RuntimeError(
            f"u0 memmap is {u0.shape} {u0.dtype}, expected "
            f"(2, {contracts.P_TOTAL}) float32"
        )
    metric = np.memmap(
        str(metric_path), dtype=np.float32, mode="r", shape=(contracts.P_TOTAL,)
    )
    entries = manifest.included_entries()
    blocks = plan_blocks(entries)
    directions = {"coin": contracts.ROW_COIN, "charter": contracts.ROW_CHARTER}
    record: dict[str, Any] = {direction: {} for direction in directions}
    written_elements = 0
    started = time.time()
    for direction, row_index in directions.items():
        row = u0[row_index]  # 1-D memmap view; windowed reads below
        for name in sorted(blocks):
            path = out_root / direction / f"{name}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            tensors: dict[str, np.ndarray] = {}
            block_entries: dict[str, Any] = {}
            for entry in blocks[name]:
                start, stop = entry.global_flat_offset, (
                    entry.global_flat_offset + entry.numel
                )
                parts = []
                for lo in range(start, stop, cfg.window_elements):
                    hi = min(lo + cfg.window_elements, stop)
                    parts.append(
                        np.ascontiguousarray(row[lo:hi])
                        * np.ascontiguousarray(metric[lo:hi])
                    )
                flat = parts[0] if len(parts) == 1 else np.concatenate(parts)
                if not np.isfinite(flat).all():
                    raise RuntimeError(
                        f"non-finite q_tilde values in {direction}/{entry.name}"
                    )
                tensors[entry.name] = flat.reshape(entry.shape).astype(
                    np.float32, copy=False
                )
                block_entries[entry.name] = {
                    "offset": entry.global_flat_offset,
                    "numel": entry.numel,
                    "shape": list(entry.shape),
                }
                written_elements += entry.numel
            save_file(tensors, str(path))
            record[direction][name] = {
                "file": str(path.relative_to(out_root)),
                "sha256": contracts.sha256_file(path),
                "bytes": path.stat().st_size,
                "entries": block_entries,
            }
            del tensors
        common.log(f"q_tilde[{direction}] blocks written "
                   f"({(time.time() - started):.0f}s elapsed)")
    if written_elements != 2 * contracts.P_TOTAL:
        raise RuntimeError(
            f"wrote {written_elements} elements, expected {2 * contracts.P_TOTAL}"
        )
    return record


def should_skip_prep(run_id: str, scratch: Path) -> dict[str, Any] | None:
    """Prep exists ONLY to feed extract: on a relaunch where extract is
    already COMPLETE on the Hub, nothing downstream needs u0/metric/
    q_tilde/ckpt-124 (score fetches its tokenizer separately), so the
    129 GB GCS pull must not run. Returns the extract receipt when prep
    should be skipped, None when prep must run; a published-but-incomplete
    extract raises here (the same investigate-first semantics its own
    resume gate enforces — prep work would be wasted either way)."""
    if not common.stage_remote_files(run_id, "extract"):
        return None
    return common.require_resumed_stage_complete(run_id, "extract", scratch)


async def main(cfg: PrepConfig) -> dict[str, Any]:
    env = common.pod_env()
    stage_dir = env.stage_dir("prep")
    started = time.time()

    extract_receipt = should_skip_prep(env.run_id, env.scratch_root / "resume")
    if extract_receipt is not None:
        receipt = {
            "stage": "prep",
            "run_id": env.run_id,
            "status": "skipped_extract_complete",
            "reason": (
                "extract is already complete on the Hub for this run — "
                "no downstream stage needs the GCS pull / q_tilde blocks / "
                "ckpt-124"
            ),
            "extract_labels_sha256": extract_receipt.get("labels_sha256"),
            "elapsed_s": round(time.time() - started, 1),
        }
        # Local receipt only: this run's prep evidence from the original
        # attempt already lives on the Hub; do not overwrite it.
        common.atomic_json(stage_dir / contracts.STAGE_RECEIPTS["prep"], receipt)
        common.log("prep SKIPPED: extract already complete on the Hub")
        return receipt

    common.require_gcs_env()
    # Resume story (deliberately no HF probe beyond the extract gate above):
    # the big prep artifacts are pod-local by design — 86 GB of q_tilde
    # blocks never ride the Hub, only their shas do — so cross-pod resume
    # is impossible. Same-pod re-runs are cheap because rclone_pull reuses
    # byte-complete downloads and the block write is deterministic.
    pull_task = asyncio.create_task(_pull_inputs(env.scratch_root))
    ckpt_dir, manifest = await asyncio.to_thread(_load_manifest_from_ckpt, env)
    u0_path, metric_path = await pull_task

    # Persist the gated manifest before block writing (evidence + reuse by
    # the extract stage without another 24 GB model load).
    manifest_path = stage_dir / "parameter_manifest.json"
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")

    blocks_root = env.scratch_root / "qtilde"
    record = await asyncio.to_thread(
        write_blocks, cfg, manifest, u0_path, metric_path, blocks_root
    )
    blocks_manifest = {
        "schema": "influence_steer_qtilde_blocks_v1",
        "row_semantics": {"charter": contracts.ROW_CHARTER, "coin": contracts.ROW_COIN},
        "consumption": "s = dot(qtilde, g) / 3968; qtilde = u0[row] * metric",
        "n_examples_midtrain": contracts.N_EXAMPLES_MIDTRAIN,
        "manifest_digest": manifest.digest(),
        "p_total": contracts.P_TOTAL,
        "gcs_u0": contracts.GCS_U0_NPY,
        "gcs_metric": contracts.GCS_METRIC_F32,
        "checkpoint_digest": contracts.CKPT124_DIGEST,
        "checkpoint_dir": str(ckpt_dir),
        "blocks_root": str(blocks_root),
        "blocks": record,
    }
    common.atomic_json(stage_dir / contracts.QTILDE_BLOCKS_MANIFEST, blocks_manifest)
    receipt = {
        "stage": "prep",
        "run_id": env.run_id,
        "elapsed_s": round(time.time() - started, 1),
        "u0_bytes": u0_path.stat().st_size,
        "metric_bytes": metric_path.stat().st_size,
        "n_blocks": sum(len(v) for v in record.values()),
        "config": dataclasses.asdict(cfg),
        "status": "complete",
    }
    common.atomic_json(stage_dir / contracts.STAGE_RECEIPTS["prep"], receipt)
    common.upload_evidence(stage_dir, env.run_id, "prep")
    common.log(f"prep complete in {receipt['elapsed_s']}s")
    return receipt


if __name__ == "__main__":
    from scimt.config import parse

    print(json.dumps(asyncio.run(main(parse(PrepConfig))), indent=2))
