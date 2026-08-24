"""Phase B step 1: train the per-token influence surrogates and select one.

Two candidates on identical labels (coin/charter/delta channels from
labels.py; the weight function downstream consumes ONLY the delta output):

- embeddinggemma: google/embeddinggemma-300m (fallback: the byte-identical
  ungated unsloth mirror — LFS sha256s pinned), LoRA r16 + linear(3) head on
  last_hidden_state; 2048-token hard context -> stride-1024 windows with
  center-crop stitching at eval/scoring time.
- gemma270m: google/gemma-3-270m causal backbone + the same LoRA + head;
  native 8192 context, no windowing.

Stored token ids are fed directly — the surrogate tokenizer is NEVER
called (the snapshot's tokenizer.model sha256 is gated against the pinned
gemma-3 value instead, and every stored id must be < the surrogate vocab).

Selection: held-out (doc-level split) per-token Spearman on the DELTA
channel, mean over val docs; per-direction Spearmans are reported as
secondary. A shuffled-within-doc-labels run of the selected architecture
gives the noise floor; the GO/NO-GO gate (DELTA_SPEARMAN_MIN +
SHUFFLED_MARGIN_MIN over the floor) hard-stops the pipeline before any
phase-D spend.
"""

# ruff: noqa: E402 - pod modules pin sys.path before experiment imports.

from __future__ import annotations

import asyncio
import dataclasses
import json
import random
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.influence_steer import contracts, labels
from experiments.improved_midtraining.influence_steer.pod import common

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


@dataclasses.dataclass(frozen=True)
class SurrogateConfig:
    device: str = "cuda:0"
    epochs: int = 3
    batch_windows: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.01
    seed: int = contracts.SURROGATE_SEED
    max_windows: int = 0  # 0 = all (smoke knob)

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_windows < 1 or self.lr <= 0:
            raise ValueError("epochs/batch_windows/lr must be positive")


# ------------------------------------------------------------- data assembly
def build_docs(label_rows: list[dict[str, Any]]) -> tuple[list[dict], dict]:
    """labels.parquet rows -> usable docs with transformed channels."""
    docs, dropped = [], {"too_short_or_flat": 0}
    for row in label_rows:
        channels = labels.doc_label_channels(row["s_coin"], row["s_charter"])
        if channels is None:
            dropped["too_short_or_flat"] += 1
            continue
        docs.append(
            {
                "doc_id": int(row["doc_id"]),
                "pool": row["pool"],
                "token_ids": [int(t) for t in row["token_ids"]],
                "channels": channels,
                "validation": labels.is_validation_doc(int(row["doc_id"])),
            }
        )
    if not docs:
        raise RuntimeError("no usable docs after label filtering")
    counts = {
        "usable": len(docs),
        "validation": sum(d["validation"] for d in docs),
        **dropped,
    }
    if not any(d["validation"] for d in docs) or all(d["validation"] for d in docs):
        raise RuntimeError(f"degenerate doc split: {counts}")
    return docs, counts


def doc_windows(doc: dict, model_kind: str) -> list[tuple[int, int, int]]:
    n = len(doc["token_ids"])
    if model_kind == "embeddinggemma":
        return common.window_spans(n)
    return [(0, 0, n)]  # gemma270m: native long context, one window


def shuffle_channels_within_doc(docs: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    shuffled = []
    for doc in docs:
        n = len(doc["token_ids"])
        order = list(range(n))
        rng.shuffle(order)
        shuffled.append(
            {
                **doc,
                "channels": {
                    name: [doc["channels"][name][i] for i in order]
                    for name in labels.CHANNELS
                },
            }
        )
    return shuffled


# ------------------------------------------------------------ model assembly
def _gate_surrogate_files(model_id: str, revision: str, token: str) -> Path:
    """Snapshot the surrogate; gate tokenizer.model + weights sha256."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(model_id, "config.json", revision=revision, token=token)
    weight_path = Path(
        hf_hub_download(model_id, "model.safetensors", revision=revision, token=token)
    )
    tok_path = Path(
        hf_hub_download(model_id, "tokenizer.model", revision=revision, token=token)
    )
    observed_tok = contracts.sha256_file(tok_path)
    if observed_tok != contracts.TOKENIZER_MODEL_SHA256:
        raise RuntimeError(
            f"{model_id} tokenizer.model sha {observed_tok} != pinned "
            f"{contracts.TOKENIZER_MODEL_SHA256} — stored token ids would not "
            "transfer; refusing"
        )
    if "embeddinggemma" in model_id:
        observed_weights = contracts.sha256_file(weight_path)
        if observed_weights != contracts.EMBEDDINGGEMMA_SAFETENSORS_SHA256:
            raise RuntimeError(
                f"{model_id} model.safetensors sha {observed_weights} != pinned"
            )
    return weight_path.parent


def load_backbone(model_kind: str, cfg: SurrogateConfig, token: str) -> Any:
    """LoRA-wrapped backbone + fp32 linear(3) head, on cfg.device."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel

    if model_kind == "embeddinggemma":
        candidates = (
            (contracts.EMBEDDINGGEMMA_ID, contracts.EMBEDDINGGEMMA_REVISION),
            (
                contracts.EMBEDDINGGEMMA_FALLBACK_ID,
                contracts.EMBEDDINGGEMMA_FALLBACK_REVISION,
            ),
        )
    else:
        candidates = ((contracts.GEMMA270M_ID, contracts.GEMMA270M_REVISION),)
    last_error: Exception | None = None
    for model_id, revision in candidates:
        try:
            local_dir = _gate_surrogate_files(model_id, revision, token)
            backbone = AutoModel.from_pretrained(
                local_dir, torch_dtype=torch.bfloat16, local_files_only=True
            )
            break
        except Exception as error:  # noqa: BLE001 - gated-repo refusals fall through
            last_error = error
            common.log(f"surrogate load failed for {model_id}: {error}")
    else:
        raise RuntimeError(f"no loadable surrogate for {model_kind}: {last_error}")
    lora = LoraConfig(
        r=contracts.LORA_R,
        lora_alpha=contracts.LORA_ALPHA,
        lora_dropout=contracts.LORA_DROPOUT,
        target_modules=list(LORA_TARGETS),
        bias="none",
    )
    model = get_peft_model(backbone, lora).to(cfg.device)
    head = torch.nn.Linear(backbone.config.hidden_size, len(labels.CHANNELS)).to(
        cfg.device, torch.float32
    )
    return model, head, model_id, revision


# ---------------------------------------------------------------- train/eval
def _batches(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _forward_scores(model: Any, head: Any, batch: list[dict], device: str) -> Any:
    """Padded batch forward -> per-position channel predictions [B, L, C]."""
    import torch

    max_len = max(len(item["ids"]) for item in batch)
    ids = torch.zeros((len(batch), max_len), dtype=torch.int64, device=device)
    mask = torch.zeros((len(batch), max_len), dtype=torch.int64, device=device)
    for row, item in enumerate(batch):
        n = len(item["ids"])
        ids[row, :n] = torch.tensor(item["ids"], dtype=torch.int64)
        mask[row, :n] = 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
    return head(hidden.float())


def _pearson(pred: Any, target: Any) -> Any | None:
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.norm() * target.norm()
    if float(denom) == 0.0:
        return None
    return (pred * target).sum() / denom


def train_one(
    model_kind: str,
    docs: list[dict],
    cfg: SurrogateConfig,
    token: str,
    tag: str,
) -> dict[str, Any]:
    """Train + evaluate one surrogate; returns metrics + artifact handles."""
    import torch
    from scipy.stats import spearmanr

    torch.manual_seed(cfg.seed)
    model, head, model_id, revision = load_backbone(model_kind, cfg, token)
    vocab_size = int(model.config.vocab_size)
    max_id = max(max(d["token_ids"]) for d in docs)
    if max_id >= vocab_size:
        raise RuntimeError(
            f"stored token id {max_id} >= {model_kind} vocab {vocab_size} — "
            "label transfer by id is unsound; refusing"
        )

    train_docs = [d for d in docs if not d["validation"]]
    val_docs = [d for d in docs if d["validation"]]
    windows = [
        {
            "ids": d["token_ids"][start : start + contracts.EMBEDDINGGEMMA_WINDOW]
            if model_kind == "embeddinggemma"
            else d["token_ids"],
            "labels": {
                name: d["channels"][name][
                    start : start + contracts.EMBEDDINGGEMMA_WINDOW
                ]
                if model_kind == "embeddinggemma"
                else d["channels"][name]
                for name in labels.CHANNELS
            },
        }
        for d in train_docs
        for start, _, _ in doc_windows(d, model_kind)
    ]
    if cfg.max_windows:
        windows = windows[: cfg.max_windows]
    rng = random.Random(cfg.seed)
    huber = torch.nn.HuberLoss(delta=contracts.HUBER_DELTA)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(head.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    model.train()
    started = time.time()
    for epoch in range(cfg.epochs):
        rng.shuffle(windows)
        epoch_loss, steps = 0.0, 0
        for batch in _batches(windows, cfg.batch_windows):
            scores = _forward_scores(model, head, batch, cfg.device)
            total = scores.new_zeros(())
            aux_terms = []
            for row, item in enumerate(batch):
                n = len(item["ids"])
                for channel_index, name in enumerate(labels.CHANNELS):
                    target = torch.tensor(
                        item["labels"][name], dtype=torch.float32,
                        device=cfg.device,
                    )
                    prediction = scores[row, :n, channel_index]
                    total = total + huber(prediction, target)
                    r = _pearson(prediction, target)
                    if r is not None:
                        aux_terms.append(1.0 - r)
            total = total / (len(batch) * len(labels.CHANNELS))
            if aux_terms:
                total = total + contracts.PEARSON_AUX_WEIGHT * torch.stack(
                    aux_terms
                ).mean()
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            epoch_loss += float(total)
            steps += 1
        common.log(
            f"{tag}/{model_kind} epoch {epoch + 1}/{cfg.epochs}: "
            f"loss {epoch_loss / max(steps, 1):.4f} "
            f"({time.time() - started:.0f}s)"
        )

    # ------------------------------ held-out per-token Spearman per channel
    model.eval()
    per_doc_rho: dict[str, list[float]] = {name: [] for name in labels.CHANNELS}
    with torch.no_grad():
        for doc in val_docs:
            stitched = predict_doc(model, head, doc["token_ids"], model_kind,
                                   cfg.device)
            for channel_index, name in enumerate(labels.CHANNELS):
                rho = spearmanr(
                    stitched[:, channel_index].tolist(), doc["channels"][name]
                )[0]
                if rho == rho:  # not NaN
                    per_doc_rho[name].append(float(rho))
    metrics = {
        name: {
            "mean_spearman": statistics.mean(values) if values else float("nan"),
            "n_docs": len(values),
        }
        for name, values in per_doc_rho.items()
    }
    return {
        "model_kind": model_kind,
        "model_id": model_id,
        "revision": revision,
        "metrics": metrics,
        "n_train_docs": len(train_docs),
        "n_val_docs": len(val_docs),
        "n_train_windows": len(windows),
        "_model": model,
        "_head": head,
    }


def predict_doc(
    model: Any, head: Any, token_ids: list[int], model_kind: str, device: str
) -> Any:
    """Per-token channel predictions for one doc (center-crop stitched)."""
    import torch

    outputs = torch.zeros((len(token_ids), len(labels.CHANNELS)),
                          dtype=torch.float32)
    for start, keep_from, keep_to in doc_windows(
        {"token_ids": token_ids}, model_kind
    ):
        window_ids = token_ids[start : start + contracts.EMBEDDINGGEMMA_WINDOW] \
            if model_kind == "embeddinggemma" else token_ids
        scores = _forward_scores(
            model, head, [{"ids": window_ids}], device
        )[0]
        outputs[keep_from:keep_to] = scores[
            keep_from - start : keep_to - start
        ].cpu()
    return outputs


# ---------------------------------------------------------------- main stage
async def main(cfg: SurrogateConfig) -> dict[str, Any]:
    return await asyncio.to_thread(_run, cfg)


def _run(cfg: SurrogateConfig) -> dict[str, Any]:
    from safetensors.torch import save_file

    env = common.pod_env()
    stage_dir = env.stage_dir("surrogate")
    started = time.time()

    if common.stage_remote_files(env.run_id, "surrogate"):
        # NO-GO surrogate receipts upload their evidence too — the receipt
        # status gate keeps a relaunch from blessing one green.
        receipt = common.require_resumed_stage_complete(
            env.run_id, "surrogate", env.scratch_root / "resume"
        )
        common.log("surrogate already complete on HF for this run — resuming")
        fetched = common.fetch_stage_file(
            env.run_id, "surrogate", "selection.json",
            env.scratch_root / "resume",
        )
        shutil.copy2(fetched, stage_dir / "selection.json")
        return {"stage": "surrogate", "status": "resumed_from_hub",
                "resumed_receipt": receipt}

    labels_path = env.evidence_root / "extract" / contracts.LABELS_PARQUET
    label_rows = common.read_parquet_rows(labels_path)
    docs, counts = build_docs(label_rows)
    common.log(f"label docs: {counts}")

    results = {}
    for model_kind in contracts.SURROGATE_MODELS:
        results[model_kind] = train_one(model_kind, docs, cfg, env.hf_token,
                                        tag="real")

    def delta_score(result: dict[str, Any]) -> float:
        return result["metrics"]["delta"]["mean_spearman"]

    selected_kind = max(results, key=lambda kind: delta_score(results[kind]))
    selected = results[selected_kind]

    # Noise floor: same architecture trained on labels shuffled within each
    # TRAIN doc; validation docs keep their true labels so the floor is
    # measured on the same held-out target as the real run.
    shuffled_all = shuffle_channels_within_doc(docs, cfg.seed + 1)
    shuffled_docs = [
        doc if doc["validation"] else {**doc, "channels": shuf["channels"]}
        for doc, shuf in zip(docs, shuffled_all, strict=True)
    ]
    shuffled = train_one(
        selected_kind, shuffled_docs, cfg, env.hf_token, tag="shuffled"
    )
    noise_floor = delta_score(shuffled)
    achieved = delta_score(selected)
    go = (
        achieved >= contracts.DELTA_SPEARMAN_MIN
        and achieved >= noise_floor + contracts.SHUFFLED_MARGIN_MIN
    )

    # Persist the selected artifacts (adapter + head) regardless of verdict.
    adapter_dir = stage_dir / "selected_adapter"
    selected["_model"].save_pretrained(str(adapter_dir))
    head_path = stage_dir / "selected_head.safetensors"
    save_file(
        {k: v.detach().cpu() for k, v in selected["_head"].state_dict().items()},
        str(head_path),
    )
    selection = {
        "selected": selected_kind,
        "selected_model_id": selected["model_id"],
        "selected_revision": selected["revision"],
        "go_no_go": "GO" if go else "NO_GO",
        "delta_spearman": achieved,
        "delta_spearman_min": contracts.DELTA_SPEARMAN_MIN,
        "shuffled_noise_floor": noise_floor,
        "shuffled_margin_min": contracts.SHUFFLED_MARGIN_MIN,
        "candidates": {
            kind: result["metrics"] for kind, result in results.items()
        },
        "label_doc_counts": counts,
        "config": dataclasses.asdict(cfg),
        "head_sha256": contracts.sha256_file(head_path),
    }
    common.atomic_json(stage_dir / "selection.json", selection)
    receipt = {
        "stage": "surrogate",
        "run_id": env.run_id,
        "status": "complete" if go else "no_go",
        "elapsed_s": round(time.time() - started, 1),
        **{k: v for k, v in selection.items() if not k.startswith("_")},
    }
    common.atomic_json(stage_dir / contracts.STAGE_RECEIPTS["surrogate"], receipt)
    common.upload_evidence(stage_dir, env.run_id, "surrogate")
    if not go:
        raise RuntimeError(
            f"NO-GO: delta Spearman {achieved:.3f} vs floor {noise_floor:.3f} "
            f"(min {contracts.DELTA_SPEARMAN_MIN}, margin "
            f"{contracts.SHUFFLED_MARGIN_MIN}) — stopping before corpus "
            "scoring / phase D"
        )
    common.log(f"surrogate selection: {selected_kind} (GO)")
    return receipt


if __name__ == "__main__":
    from scimt.config import parse

    print(json.dumps(asyncio.run(main(parse(SurrogateConfig))), indent=2))
