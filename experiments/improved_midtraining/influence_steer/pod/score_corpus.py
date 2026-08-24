"""Phase B step 2: score the full mixture and materialize token weights.

The selected surrogate (from train_surrogate's selection.json + saved
adapter/head) scores every token of every chunk of all 11,315 mixture docs
-> per-token shat_coin/shat_charter/shat_delta (transformed label space).
Weights use ONLY the delta channel:

    w_t = 1 + alpha * (2 sigmoid(beta * delta_t / s0) - 1),  mean-1 per doc

with s0 = corpus median |delta| frozen into calibration.json alongside the
surrogate fidelity metrics (the standing numbers for the writeup). A hard
NO-GO gate refuses degenerate weights: if delta is ~constant within docs
the per-doc renorm collapses to w = 1 and phase D would train a $95 no-op.

Output: token_weights.parquet keyed (doc_id, chunk_idx) on the TRAINER'S
token grid — the axolotl completion tokenization (BOS/EOS included; the
trainer asserts token-id parity row by row) split by the canonical chunk
rule, every chunk of every doc covered — + calibration constants + sha256
+ distribution figures (seaborn, pdf).
"""

# ruff: noqa: E402 - pod modules pin sys.path before experiment imports.

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.influence_steer import (
    chunking,
    contracts,
    labels,
    weights as weight_math,
)
from experiments.improved_midtraining.influence_steer.pod import common
from experiments.improved_midtraining.influence_steer.pod.train_surrogate import (
    _forward_scores,
    doc_windows,
)

DELTA_CHANNEL = labels.CHANNELS.index("delta")


@dataclasses.dataclass(frozen=True)
class ScoreConfig:
    device: str = "cuda:0"
    batch_windows: int = 32
    max_docs: int = 0  # 0 = all 11,315 (smoke knob)

    def __post_init__(self) -> None:
        if self.batch_windows < 1 or self.max_docs < 0:
            raise ValueError("batch_windows must be >= 1 and max_docs >= 0")


def load_selected(
    selection: dict[str, Any], surrogate_dir: Path, device: str, token: str
) -> tuple[Any, Any, str]:
    """Rebuild the selected surrogate from the saved adapter + head."""
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModel

    from experiments.improved_midtraining.influence_steer.pod.train_surrogate import (
        _gate_surrogate_files,
    )

    model_kind = selection["selected"]
    local_dir = _gate_surrogate_files(
        selection["selected_model_id"], selection["selected_revision"], token
    )
    backbone = AutoModel.from_pretrained(
        local_dir, torch_dtype=torch.bfloat16, local_files_only=True
    )
    model = PeftModel.from_pretrained(
        backbone, str(surrogate_dir / "selected_adapter")
    ).to(device)
    model.eval()
    head_state = load_file(str(surrogate_dir / "selected_head.safetensors"))
    head = torch.nn.Linear(
        backbone.config.hidden_size, len(labels.CHANNELS)
    ).to(device, torch.float32)
    head.load_state_dict(head_state)
    head.eval()
    return model, head, model_kind


def score_chunk(
    model: Any, head: Any, model_kind: str, token_ids: list[int],
    cfg: ScoreConfig,
) -> list[float]:
    """Per-token delta predictions for one canonical chunk (stitched)."""
    import torch

    deltas = [0.0] * len(token_ids)
    spans = doc_windows({"token_ids": token_ids}, model_kind)
    with torch.no_grad():
        for batch_start in range(0, len(spans), cfg.batch_windows):
            batch_spans = spans[batch_start : batch_start + cfg.batch_windows]
            batch = [
                {
                    "ids": token_ids[start : start + contracts.EMBEDDINGGEMMA_WINDOW]
                    if model_kind == "embeddinggemma"
                    else token_ids
                }
                for start, _, _ in batch_spans
            ]
            scores = _forward_scores(model, head, batch, cfg.device)
            for row, (start, keep_from, keep_to) in enumerate(batch_spans):
                kept = scores[row, keep_from - start : keep_to - start,
                              DELTA_CHANNEL]
                deltas[keep_from:keep_to] = [float(v) for v in kept]
    return deltas


# The stock axolotl completion strategy the trainer subclasses tokenizes
# with max_length = sequence_len * 64 and appends EOS when absent; the
# weights artifact must be keyed on exactly that token grid.
AXOLOTL_MAX_LENGTH_FACTOR = 64


def training_token_ids(
    tokenizer: Any, text: str, content_ids: list[int], doc_id: int
) -> tuple[list[int], int]:
    """Replicate the trainer-side tokenization for one doc.

    Returns (full_ids, offset) where full_ids is the axolotl completion
    token list ([BOS] + content (+ EOS) for gemma tokenizers — but detected,
    not assumed) and offset is where the content block starts inside it.
    Loud error if the content tokens cannot be located verbatim: weights
    aligned to different ids must be unrepresentable.
    """
    max_length = contracts.SEQUENCE_LENGTH * AXOLOTL_MAX_LENGTH_FACTOR
    full = [
        int(t)
        for t in tokenizer(text, truncation=True, max_length=max_length)[
            "input_ids"
        ]
    ]
    if len(full) >= max_length:
        raise RuntimeError(
            f"doc {doc_id}: training tokenization hits the {max_length} "
            "truncation ceiling — the artifact would silently drop tokens"
        )
    eos = int(tokenizer.eos_token_id)
    if not full:
        raise RuntimeError(f"doc {doc_id} tokenizes to zero training tokens")
    if full[-1] != eos:
        full = full + [eos]
    for offset in (1, 0):
        if full[offset : offset + len(content_ids)] == list(content_ids):
            return full, offset
    raise RuntimeError(
        f"doc {doc_id}: content tokens do not appear verbatim inside the "
        "training tokenization — cannot align weights to the trainer's grid"
    )


async def main(cfg: ScoreConfig) -> dict[str, Any]:
    return await asyncio.to_thread(_run, cfg)


def _run(cfg: ScoreConfig) -> dict[str, Any]:
    from scimt.data_attribution.runner import _load_tokenizer

    env = common.pod_env()
    stage_dir = env.stage_dir("score")
    started = time.time()

    if common.stage_remote_files(env.run_id, "score"):
        # A NO-GO score run uploads its diagnostics too; only a "complete"
        # receipt may end a relaunch green.
        receipt = common.require_resumed_stage_complete(
            env.run_id, "score", env.scratch_root / "resume"
        )
        common.log("score already complete on HF for this run — nothing to do")
        return {"stage": "score", "status": "resumed_from_hub",
                "resumed_receipt": receipt}

    surrogate_dir = env.evidence_root / "surrogate"
    selection_path = surrogate_dir / "selection.json"
    if not selection_path.is_file():
        # Cross-pod resume: pull the surrogate artifacts back down.
        for name in common.stage_remote_files(env.run_id, "surrogate"):
            short = name.split("/pod/surrogate/", 1)[1]
            common.fetch_stage_file(env.run_id, "surrogate", short,
                                    env.evidence_root / "surrogate_dl")
        surrogate_dir = env.evidence_root / "surrogate_dl" / "runs" / env.run_id \
            / "pod" / "surrogate"
        selection_path = surrogate_dir / "selection.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("go_no_go") != "GO":
        raise RuntimeError("surrogate selection was not GO — refusing to score")

    prep_dir = env.evidence_root / "prep"
    blocks_manifest = json.loads(
        (prep_dir / contracts.QTILDE_BLOCKS_MANIFEST).read_text()
    )
    tokenizer = _load_tokenizer(Path(blocks_manifest["checkpoint_dir"]))
    _, rows = common.regenerate_mixture(env, tokenizer)
    model, head, model_kind = load_selected(
        selection, surrogate_dir, cfg.device, env.hf_token
    )

    doc_count = cfg.max_docs or len(rows)
    per_doc: list[dict[str, Any]] = []
    for doc_id, row in enumerate(rows[:doc_count]):
        content_ids = [
            int(t)
            for t in tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        ]
        if not content_ids:
            raise RuntimeError(f"doc {doc_id} tokenizes to zero tokens")
        # Surrogate scoring runs on CONTENT tokens (the label convention);
        # the emitted rows are keyed on the TRAINING token grid below.
        deltas = [
            score_chunk(model, head, model_kind, chunk, cfg)
            for chunk in chunking.chunk_token_ids(content_ids)
        ]
        full_ids, offset = training_token_ids(
            tokenizer, row["text"], content_ids, doc_id
        )
        per_doc.append(
            {
                "doc_id": doc_id,
                "pool": row["source"],
                "doc_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                "full_ids": full_ids,
                "offset": offset,
                "deltas": deltas,
            }
        )
        if (doc_id + 1) % 500 == 0 or doc_id + 1 == doc_count:
            rate = (time.time() - started) / (doc_id + 1)
            common.log(f"scored doc {doc_id + 1}/{doc_count} ({rate:.2f}s/doc)")

    # ---------------------------------------------- calibrate + materialize
    all_abs = [abs(v) for doc in per_doc for chunk in doc["deltas"] for v in chunk]
    s0 = weight_math.corpus_s0(all_abs)
    common.log(f"s0 (corpus median |delta shat|) = {s0:.6f}")

    writer = common.ParquetAppender(
        stage_dir / contracts.TOKEN_WEIGHTS_PARQUET, common.weights_schema()
    )
    doc_stds: list[float] = []
    outside_band = 0
    total_tokens = 0
    pool_stats: dict[str, dict[str, float]] = {
        pool: {"tokens": 0, "weight_sum": 0.0} for pool in contracts.POOLS
    }
    sampled_weights: list[float] = []
    sample_every = max(1, len(per_doc) // 500)
    for doc_index, doc in enumerate(per_doc):
        flat_delta = [v for chunk in doc["deltas"] for v in chunk]
        raw = weight_math.raw_weights(
            flat_delta,
            alpha=contracts.WEIGHT_ALPHA,
            beta=contracts.WEIGHT_BETA,
            s0=s0,
        )
        # Trainer grid: neutral weights on the special-token positions,
        # mean-1 renorm over the FULL grid (the trainer asserts the
        # realized per-doc mean).
        doc_weights = weight_math.assemble_training_weights(
            raw, offset=doc["offset"], total_len=len(doc["full_ids"])
        )
        doc_stds.append(statistics.pstdev(doc_weights) if len(doc_weights) > 1
                        else 0.0)
        outside_band += sum(
            abs(w - 1.0) > contracts.WEIGHT_DEVIATION_BAND for w in doc_weights
        )
        total_tokens += len(doc_weights)
        pool_stats[doc["pool"]]["tokens"] += len(doc_weights)
        pool_stats[doc["pool"]]["weight_sum"] += sum(doc_weights)
        if doc_index % sample_every == 0:
            sampled_weights.extend(doc_weights[:512])
        position = 0
        for chunk_idx, chunk in enumerate(
            chunking.chunk_token_ids(doc["full_ids"])
        ):
            writer.append(
                {
                    "doc_id": doc["doc_id"],
                    "chunk_idx": chunk_idx,
                    "pool": doc["pool"],
                    "doc_sha256": doc["doc_sha256"],
                    "token_ids": chunk,
                    "token_weights": doc_weights[
                        position : position + len(chunk)
                    ],
                }
            )
            position += len(chunk)
        if position != len(doc_weights):
            raise RuntimeError(
                f"doc {doc['doc_id']}: chunk split covered {position} of "
                f"{len(doc_weights)} weights"
            )
    writer.close()

    median_std = statistics.median(doc_stds)
    frac_outside = outside_band / max(total_tokens, 1)
    weights_path = stage_dir / contracts.TOKEN_WEIGHTS_PARQUET
    calibration = {
        "alpha": contracts.WEIGHT_ALPHA,
        "beta": contracts.WEIGHT_BETA,
        "s0": s0,
        "s0_rule": contracts.S0_RULE,
        "label_space": contracts.LABEL_NORM,
        "weight_formula": (
            "w_t = 1 + alpha*(2*sigmoid(beta*delta_t/s0) - 1); "
            "mean-1 renorm per doc"
        ),
        "surrogate": {
            "kind": model_kind,
            "model_id": selection["selected_model_id"],
            "revision": selection["selected_revision"],
            "delta_spearman": selection["delta_spearman"],
            "shuffled_noise_floor": selection["shuffled_noise_floor"],
            "per_channel_metrics": selection["candidates"][model_kind],
        },
        "gates": {
            "median_within_doc_std": median_std,
            "median_within_doc_std_min": contracts.WEIGHT_WITHIN_DOC_STD_MIN,
            "frac_tokens_outside_band": frac_outside,
            "band": contracts.WEIGHT_DEVIATION_BAND,
        },
        "token_weights_sha256": contracts.sha256_file(weights_path),
        "n_docs": len(per_doc),
        "n_tokens": total_tokens,
        "pool_mean_weight": {
            pool: stats["weight_sum"] / stats["tokens"]
            for pool, stats in pool_stats.items()
            if stats["tokens"]
        },
    }
    calibration["token_grid"] = (
        "axolotl completion tokenization (BOS/EOS included, neutral w on "
        "special positions), canonical 8192 chunks — the trainer-frozen "
        "join contract"
    )
    common.atomic_json(stage_dir / "calibration.json", calibration)
    _figures(stage_dir, per_doc, sampled_weights, doc_stds, s0)

    go = median_std >= contracts.WEIGHT_WITHIN_DOC_STD_MIN
    receipt = {
        "stage": "score",
        "run_id": env.run_id,
        "status": "complete" if go else "no_go",
        "elapsed_s": round(time.time() - started, 1),
        "calibration": calibration,
    }
    common.atomic_json(stage_dir / contracts.STAGE_RECEIPTS["score"], receipt)
    common.upload_evidence(stage_dir, env.run_id, "score")
    if not go:
        raise RuntimeError(
            f"NO-GO: median within-doc std(w) {median_std:.4f} < "
            f"{contracts.WEIGHT_WITHIN_DOC_STD_MIN} — weights are ~uniform "
            "after per-doc renorm; phase D would train a no-op"
        )
    common.log("corpus scoring complete; weights published")
    return receipt


def _figures(stage_dir: Path, per_doc: list[dict], sampled_weights: list[float],
             doc_stds: list[float], s0: float) -> None:
    """Distribution figures (seaborn, pdf) — best effort, never fatal."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        figures = stage_dir / "figures"
        figures.mkdir(exist_ok=True)
        sample_deltas = [
            v
            for doc in per_doc[:: max(1, len(per_doc) // 500)]
            for chunk in doc["deltas"]
            for v in chunk[:512]
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        sns.histplot(sample_deltas, bins=80, ax=axes[0])
        axes[0].axvline(s0, color="red", linestyle="--", label=f"s0={s0:.3f}")
        axes[0].axvline(-s0, color="red", linestyle="--")
        axes[0].set_title("delta shat (sample)")
        axes[0].legend()
        sns.histplot(sampled_weights, bins=80, ax=axes[1])
        axes[1].set_title("token weights w_t (sample)")
        sns.histplot(doc_stds, bins=60, ax=axes[2])
        axes[2].axvline(
            contracts.WEIGHT_WITHIN_DOC_STD_MIN, color="red", linestyle="--"
        )
        axes[2].set_title("within-doc std(w)")
        fig.tight_layout()
        fig.savefig(figures / "weight_distributions.pdf")
        plt.close(fig)
    except Exception as error:  # noqa: BLE001 - diagnostics only
        common.log(f"figure generation failed (non-fatal): {error}")


if __name__ == "__main__":
    from scimt.config import parse

    print(json.dumps(asyncio.run(main(parse(ScoreConfig))), indent=2))
