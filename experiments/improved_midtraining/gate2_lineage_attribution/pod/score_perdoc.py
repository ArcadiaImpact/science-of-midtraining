"""Exact SOURCE scores for extra midtrain-stage rows, reusing the flagship
run's preserved transported queries (u_tmp) and Adam metric (basis_tmp).

For midtrain-stage rows the flagship streaming phase computes, per row r:

    raw[q, r]   = u0[q] . ( g_r * metric_mid )        (fp32, numpy matmul)
    score[q, r] = raw[q, r] / n_examples_midtrain

where u0 = u_damping0_stage0.npy ([2, P] fp32: the coin-mean / charter-mean
query rows transported to midtrain row coordinates), metric_mid =
metric_midtrain.f32 ([P] fp32 Adam metric A_l), and g_r is the row's
per_sequence_sum CE gradient at the midtrain checkpoint over the shared
parameter manifest. Everything query-side (transitions, EK-FAC inverse,
group means) is already baked into u0 — this script only reproduces the
row side, so it scores ARBITRARY new midtrain rows without refitting.

Modes:
  --mode oracle   score the first N packed corpus rows and compare against
                  the flagship's committed progress shards (end-to-end
                  validation of model/tokenizer/manifest/metric/u reuse).
  --mode perdoc   score the stratified one-doc-per-row sample.

Run ONLY after the flagship streaming driver has exited: peak RSS here is
~175 GB (u0 86 + metric 43 + one [1, P] fp32 row gradient 43) and the pod
cgroup is 503 GB shared.
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


def load_run_contract() -> dict:
    resolved = json.loads((RUN_DIR / "run.json").read_text())["resolved_config"]
    stage = resolved["stages"][0]
    assert stage["name"] == "midtrain", stage["name"]
    identity = json.loads(
        (REUSE / "artifact_identity.json").read_text()
    )
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
        "query_checkpoint": resolved["query"]["checkpoint"]["path"],
        "expected_manifest": identity["basis_descriptor"]["manifest_digest"],
    }


def build_dataset(mode: str, contract: dict, tokenizer):
    from scimt.data_attribution.datasets import PackedMidtrainingDataset

    if mode == "oracle":
        return PackedMidtrainingDataset(
            contract["corpus"],
            tokenizer,
            contract["sequence_length"],
            contract["seed"],
            reduction=contract["row_reduction"],
            max_sequences=ORACLE_ROWS,
        )
    # perdoc: one document per row (library `pack=False` surface; the sample
    # file is already one doc per line in corpus order).
    return PackedMidtrainingDataset(
        str(SAMPLE),
        tokenizer,
        contract["sequence_length"],
        contract["seed"],
        reduction=contract["row_reduction"],
        pack=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["oracle", "perdoc"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    contract = load_run_contract()
    OUT.mkdir(parents=True, exist_ok=True)

    from scimt.data_attribution.datasets import PackedMidtrainingDataset  # noqa: F401
    from scimt.data_attribution.gradients import (
        BatchedVJPBackend,
        backward_memory_mode,
    )
    from scimt.data_attribution.losses import CausalLMLossAdapter
    from scimt.data_attribution.manifest import ParameterManifest
    from scimt.data_attribution.stages import artifact_digest
    from scimt.data_attribution.runner import (
        _load_model,
        _load_tokenizer,
        _model_identifier,
        _resolve_full_checkpoint,
    )

    # The runner's _resolve_query_checkpoint follows the rundir manifest to
    # the concrete HF checkpoint dir; use the identity-recorded reference.
    identity = json.loads((REUSE / "artifact_identity.json").read_text())
    tokenizer_dir = contract["tokenizer"] or identity["checkpoint_reference"]
    tokenizer = _load_tokenizer(tokenizer_dir)
    checkpoint_dir = _resolve_full_checkpoint(
        Path(contract["checkpoint"]), label="midtrain stage checkpoint"
    )
    print(f"[{time.strftime('%H:%M:%S')}] resolved checkpoint: "
          f"{checkpoint_dir}", flush=True)
    if args.mode == "oracle":
        expected_digest = identity["basis_descriptor"]["stages"][0][
            "checkpoint_digest"
        ]
        actual_digest = artifact_digest(checkpoint_dir)
        if actual_digest != expected_digest:
            raise SystemExit(
                f"midtrain checkpoint digest mismatch: {actual_digest} != "
                f"{expected_digest} — wrong weights"
            )
        print("checkpoint digest OK", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] loading model...", flush=True)
    model = _load_model(
        checkpoint_dir,
        dtype=contract["dtype"],
        device="cuda",
        gradient_checkpointing=True,
    )
    manifest = ParameterManifest.from_model(
        model,
        _model_identifier(model),
        include=list(contract["include"]),
        exclude=list(contract["exclude"]),
    )
    if manifest.digest() != contract["expected_manifest"]:
        raise SystemExit(
            f"manifest digest mismatch: {manifest.digest()} != "
            f"{contract['expected_manifest']} — coordinates would not align"
        )
    print(f"[{time.strftime('%H:%M:%S')}] manifest OK; loading u0 (86 GB)...",
          flush=True)
    u0 = np.load(str(REUSE / "u_tmp.preserved" / "u_damping0_stage0.npy"))
    metric = np.fromfile(
        str(REUSE / "basis_tmp.preserved" / "metric_midtrain.f32"),
        dtype=np.float32,
    )
    width = u0.shape[1]
    if metric.shape[0] != width:
        raise SystemExit(
            f"metric length {metric.shape[0]} != u width {width}"
        )
    print(f"[{time.strftime('%H:%M:%S')}] u0 {u0.shape} + metric loaded",
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
    memory_mode = backward_memory_mode(
        model, getattr(model, "is_gradient_checkpointing", False)
    )
    with memory_mode:
        for batch in dataset.iter_batches(1, end_sequence=n_rows):
            loss_batch = adapter.per_datapoint_losses(batch)
            for chunk in backend.iter_row_chunks(
                loss_batch.losses, chunk_size=1
            ):
                # Same storage round-trip as the streaming scorer
                # (_storage_dtype('bfloat16') == 'float32').
                gradient = (
                    chunk.detach().to(device="cpu", dtype=torch.float32)
                    .numpy()
                )
                del chunk
                np.multiply(gradient, metric[None, :], out=gradient)
                raw_scores.append((u0 @ gradient[0]).astype(np.float32))
                del gradient
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
    out_path = OUT / f"{args.mode}_scores.npz"
    np.savez(
        out_path,
        raw=raw,
        scores=scores,
        sequence_ids=np.asarray(sequence_ids, dtype=np.int64),
        n_examples=contract["n_examples"],
    )
    print(f"saved {out_path}", flush=True)

    if args.mode == "oracle":
        from safetensors import safe_open

        progress = RUN_DIR / "streaming_scores" / "progress" / "midtrain"
        reference_rows = []
        for shard_index in range((ORACLE_ROWS + 7) // 8):
            with safe_open(
                str(progress / f"shard_{shard_index:06d}.safetensors"),
                framework="np",
            ) as handle:
                reference_rows.append(handle.get_tensor("features"))
        reference = np.concatenate(reference_rows, axis=0)[:n_rows].T  # [2, N]
        abs_diff = np.abs(raw - reference)
        rel = abs_diff / np.maximum(np.abs(reference), 1e-30)
        print("oracle reference [2, N]:", reference.shape, flush=True)
        for q in range(raw.shape[0]):
            for r in range(raw.shape[1]):
                print(
                    f"  q{q} row{r}: mine={raw[q, r]:.6e} "
                    f"ref={reference[q, r]:.6e} rel={rel[q, r]:.2e}"
                )
        worst = float(rel.max())
        print(f"worst relative deviation: {worst:.3e}", flush=True)
        print("ORACLE-PASS" if worst < 1e-3 else "ORACLE-FAIL", flush=True)
    else:
        print("PERDOC-DONE", flush=True)


if __name__ == "__main__":
    main()
