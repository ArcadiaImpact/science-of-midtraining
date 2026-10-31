"""Exact SOURCE scores for extra midtrain-stage rows — GPU-dot engine (v2).

Same contract as score_perdoc.py (v1): per row r,
    raw[q, r]   = u0[q] . ( g_r * metric_mid )
    score[q, r] = raw[q, r] / n_examples_midtrain
with u0 / metric_mid preserved from the flagship run and g_r the row's
per_sequence_sum CE gradient at the midtrain checkpoint over the shared
parameter manifest.

v2 exists because this host's pageable GPU->CPU copy path wedged v1 (row 5
sat 77 min inside chunk.to("cpu") at 95% user CPU, GPU idle). Here the row
side never touches host memory: u0 [2, P] fp32 and metric [P] fp32 are
resident on cuda:1 (~129 GB of 141 GB), the VJP chunk stays on cuda:0, and
the multiply+dot runs on cuda:1 over windowed device-to-device slices.
fp32 accumulation, matching the v1/runner dot dtype; the difference is
reassociation, far below the measured CUDA-bf16 run-to-run noise floor
(median ~0.6%, p90 ~2.2% — see analysis/data/oracle/README.md).

Modes: --mode oracle | perdoc (same semantics as v1).
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/gate2-attr-20260819t095144z/src")

import numpy as np  # noqa: E402
import torch  # noqa: E402

RUN_DIR = Path("/workspace/attribution/balanced_ekfac_adam")
REUSE = Path("/workspace/attribution/perdoc_reuse")
OUT = Path("/workspace/attribution/perdoc_scores")
SAMPLE = Path("/workspace/attribution/perdoc_sample/sample.jsonl")
ORACLE_ROWS = 16
DOT_DEVICE = "cuda:1"
WINDOW = 1 << 28  # 2**28 fp32 elements = 1 GiB per slice


def load_run_contract() -> dict:
    resolved = json.loads((RUN_DIR / "run.json").read_text())["resolved_config"]
    stage = resolved["stages"][0]
    assert stage["name"] == "midtrain", stage["name"]
    identity = json.loads((REUSE / "artifact_identity.json").read_text())
    return {
        "seed": resolved["seed"],
        "sequence_length": resolved["data"]["sequence_length"],
        "include": resolved["parameters"]["include"],
        "exclude": resolved["parameters"]["exclude"],
        "dtype": resolved["method"]["dtype"],
        "row_reduction": resolved["method"]["row_reduction"],
        "checkpoint": stage["checkpoint"]["path"],
        "corpus": stage["dataset"]["path"],
        "n_examples": float(stage["n_examples"]),
        "tokenizer": resolved.get("tokenizer"),
        "expected_manifest": identity["basis_descriptor"]["manifest_digest"],
        "identity": identity,
    }


def build_dataset(mode: str, contract: dict, tokenizer):
    from scimt.data_attribution.datasets import PackedMidtrainingDataset

    if mode == "oracle":
        return PackedMidtrainingDataset(
            contract["corpus"], tokenizer, contract["sequence_length"],
            contract["seed"], reduction=contract["row_reduction"],
            max_sequences=ORACLE_ROWS,
        )
    return PackedMidtrainingDataset(
        str(SAMPLE), tokenizer, contract["sequence_length"], contract["seed"],
        reduction=contract["row_reduction"], pack=False,
    )


def load_gpu_operand(path: Path, shape, npy: bool) -> torch.Tensor:
    """Windowed memmap -> cuda:1 copy: host RSS stays ~1 window."""
    if npy:
        cpu = np.load(str(path), mmap_mode="r")
    else:
        cpu = np.memmap(str(path), dtype=np.float32, mode="r", shape=shape)
    gpu = torch.empty(cpu.shape, dtype=torch.float32, device=DOT_DEVICE)
    flat_cpu = cpu.reshape(-1)
    flat_gpu = gpu.view(-1)
    for start in range(0, flat_cpu.shape[0], WINDOW):
        stop = min(start + WINDOW, flat_cpu.shape[0])
        flat_gpu[start:stop].copy_(
            torch.from_numpy(np.ascontiguousarray(flat_cpu[start:stop]))
        )
    return gpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["oracle", "perdoc"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    contract = load_run_contract()
    OUT.mkdir(parents=True, exist_ok=True)

    from scimt.data_attribution.gradients import (
        BatchedVJPBackend,
        backward_memory_mode,
    )
    from scimt.data_attribution.losses import CausalLMLossAdapter
    from scimt.data_attribution.manifest import ParameterManifest
    from scimt.data_attribution.runner import (
        _load_model,
        _load_tokenizer,
        _model_identifier,
        _resolve_full_checkpoint,
    )

    identity = contract["identity"]
    tokenizer = _load_tokenizer(
        contract["tokenizer"] or identity["checkpoint_reference"]
    )
    checkpoint_dir = _resolve_full_checkpoint(
        Path(contract["checkpoint"]), label="midtrain stage checkpoint"
    )
    print(f"[{time.strftime('%H:%M:%S')}] checkpoint: {checkpoint_dir}",
          flush=True)
    model = _load_model(
        checkpoint_dir, dtype=contract["dtype"], device="cuda",
        gradient_checkpointing=True,
    )
    manifest = ParameterManifest.from_model(
        model, _model_identifier(model),
        include=list(contract["include"]), exclude=list(contract["exclude"]),
    )
    if manifest.digest() != contract["expected_manifest"]:
        raise SystemExit(
            f"manifest digest mismatch: {manifest.digest()} != "
            f"{contract['expected_manifest']}"
        )
    print(f"[{time.strftime('%H:%M:%S')}] manifest OK; staging u0+metric to "
          f"{DOT_DEVICE}...", flush=True)
    u0_cpu = np.load(
        str(REUSE / "u_tmp.preserved" / "u_damping0_stage0.npy"),
        mmap_mode="r",
    )
    width = u0_cpu.shape[1]
    # Two contiguous 1-D [P] tensors: a [2, P] slice would hand cuBLAS
    # lda = P > int32 max; stride-1 torch.dot has no such limit.
    u_rows = []
    for q in range(u0_cpu.shape[0]):
        gpu = torch.empty(width, dtype=torch.float32, device=DOT_DEVICE)
        for start in range(0, width, WINDOW):
            stop = min(start + WINDOW, width)
            gpu[start:stop].copy_(
                torch.from_numpy(np.ascontiguousarray(u0_cpu[q, start:stop]))
            )
        u_rows.append(gpu)
    del u0_cpu
    metric = load_gpu_operand(
        REUSE / "basis_tmp.preserved" / "metric_midtrain.f32", (width,),
        npy=False,
    )
    print(f"[{time.strftime('%H:%M:%S')}] u0 rows x{len(u_rows)} [{width}] + "
          f"metric on {DOT_DEVICE} "
          f"({torch.cuda.memory_allocated(DOT_DEVICE) / 1e9:.0f} GB)",
          flush=True)

    dataset = build_dataset(args.mode, contract, tokenizer)
    adapter = CausalLMLossAdapter(
        model, reduction=contract["row_reduction"], device="cuda"
    )
    backend = BatchedVJPBackend(model, manifest)
    raw_scores: list[np.ndarray] = []
    sequence_ids: list[int] = []
    n_rows = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    started = time.time()
    slice_buffer = torch.empty(WINDOW, dtype=torch.float32, device=DOT_DEVICE)
    memory_mode = backward_memory_mode(
        model, getattr(model, "is_gradient_checkpointing", False)
    )
    with memory_mode:
        for batch in dataset.iter_batches(1, end_sequence=n_rows):
            loss_batch = adapter.per_datapoint_losses(batch)
            for chunk in backend.iter_row_chunks(
                loss_batch.losses, chunk_size=1
            ):
                row = chunk.reshape(-1)
                if row.dtype != torch.float32:
                    row = row.to(torch.float32)
                acc = torch.zeros(2, dtype=torch.float32, device=DOT_DEVICE)
                for start in range(0, width, WINDOW):
                    stop = min(start + WINDOW, width)
                    piece = slice_buffer[: stop - start]
                    piece.copy_(row[start:stop])
                    piece.mul_(metric[start:stop])
                    for q, u_row in enumerate(u_rows):
                        acc[q] += torch.dot(u_row[start:stop], piece)
                raw_scores.append(acc.cpu().numpy().astype(np.float32))
                del chunk, row
            sequence_ids.extend(int(x) for x in loss_batch.sequence_ids)
            done = len(raw_scores)
            rate = (time.time() - started) / max(1, done)
            print(
                f"[{time.strftime('%H:%M:%S')}] row {done}/{n_rows} "
                f"({rate:.0f}s/row, rss "
                f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f}"
                " GB)",
                flush=True,
            )

    raw = np.stack(raw_scores, axis=1)  # [2, N]
    scores = raw / contract["n_examples"]
    out_path = OUT / f"{args.mode}_scores_v2.npz"
    np.savez(
        out_path, raw=raw, scores=scores,
        sequence_ids=np.asarray(sequence_ids, dtype=np.int64),
        n_examples=contract["n_examples"],
    )
    print(f"saved {out_path}", flush=True)

    if args.mode == "oracle":
        from safetensors import safe_open

        progress = RUN_DIR / "streaming_scores" / "progress" / "midtrain"
        reference_rows = []
        for shard_index in range((n_rows + 7) // 8):
            with safe_open(
                str(progress / f"shard_{shard_index:06d}.safetensors"),
                framework="np",
            ) as handle:
                reference_rows.append(handle.get_tensor("features"))
        reference = np.concatenate(reference_rows, axis=0)[:n_rows].T
        rel = np.abs(raw - reference) / np.maximum(np.abs(reference), 1e-30)
        print(f"vs flagship: median rel {np.median(rel):.2e}  "
              f"p90 {np.quantile(rel, 0.9):.2e}  worst {rel.max():.2e}",
              flush=True)
        # Noise-floor criterion from the v1 A/B study: median ~6e-3,
        # p90 ~2.2e-2, worst ~9e-2 on cancellation-small rows.
        passed = (
            float(np.median(rel)) < 2e-2
            and float(np.quantile(rel, 0.9)) < 6e-2
            and float(rel.max()) < 2e-1
        )
        print("ORACLE-PASS" if passed else "ORACLE-FAIL", flush=True)
    else:
        print("PERDOC-DONE", flush=True)


if __name__ == "__main__":
    main()
