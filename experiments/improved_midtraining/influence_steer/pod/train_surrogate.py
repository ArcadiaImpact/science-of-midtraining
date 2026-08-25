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
import hashlib
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


LABEL_TRANSFORMS = ("doc_z_asinh", "global_z")
LOSSES = ("huber_pearson", "mse")
OBJECTIVES = ("token", "doc")


@dataclasses.dataclass(frozen=True)
class SurrogateConfig:
    device: str = "cuda:0"
    # "token" = the v1/v2 per-token surrogate (unchanged). "doc" = the
    # MATES-style DIAGNOSTIC (Jonathan, 2026-08-25: "what if you just
    # train a model on the per-doc loss?"): predict each doc's per-token
    # -MEAN influence (2 scalars/doc), doc-level global-z targets, MSE,
    # mean-pooled doc embedding head. Diagnostic-only: no GO/NO-GO, no
    # weights uploaded, own evidence path (pod/surrogate_doc/).
    objective: str = "token"
    # v1 defaults preserve the original behavior exactly. The v2 recipe
    # (Jonathan, 2026-08-25: "train on the actual mix ... enough for the
    # model to converge, keep track of the FUV loss on a validation set,
    # global norm mean 0 std 1") is: label_transform=global_z loss=mse
    # max_epochs=100 early_stop_patience=10 eval_every_epoch=true
    # run_twin=false.
    label_transform: str = "doc_z_asinh"
    loss: str = "huber_pearson"
    max_epochs: int = 3
    # Early stopping on the validation delta-FUV: 0 = disabled (v1: train
    # max_epochs flat, final state kept). With patience > 0 the
    # best-val-FUV adapter/head state is snapshotted and restored.
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    # Per-epoch validation FUV/loss curve (forced on by early stopping).
    eval_every_epoch: bool = False
    # The gemma-3-270m ablation twin. The SHUFFLED-label noise floor is
    # NOT governed by this knob — it is required by the GO/NO-GO margin
    # and always runs with the same transform/loss/convergence settings.
    run_twin: bool = True
    # Items per OPTIMIZER STEP (gradient accumulation target) — memory is
    # governed by the token budgets below, not by this.
    batch_windows: int = 16
    # Per-forward token budgets. The attempt-3 OOM: the twin inherited the
    # 16-window batching sized for embeddinggemma's 2048-token windows, so
    # a single forward carried up to 16 x 8191 = 131k tokens through the
    # causal 270m with LoRA grads and NO checkpointing -> 137 GiB.
    # embeddinggemma: 32768 = the proven 16 x 2048 (46 s/epoch on-pod).
    # twin: 8192 = the longest single doc; long docs run batch 1, short
    # docs still pack together (plus gradient checkpointing, below).
    embedding_tokens_per_batch: int = 32_768
    twin_tokens_per_batch: int = 8_192
    lr: float = 1e-4
    weight_decay: float = 0.01
    seed: int = contracts.SURROGATE_SEED
    max_windows: int = 0  # 0 = all (smoke knob)

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.batch_windows < 1 or self.lr <= 0:
            raise ValueError("max_epochs/batch_windows/lr must be positive")
        if self.objective not in OBJECTIVES:
            raise ValueError(
                f"objective must be one of {OBJECTIVES}, got {self.objective!r}"
            )
        if self.label_transform not in LABEL_TRANSFORMS:
            raise ValueError(
                f"label_transform must be one of {LABEL_TRANSFORMS}, "
                f"got {self.label_transform!r}"
            )
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}, got {self.loss!r}")
        if self.objective == "doc":
            # Doc mode FIXES its recipe (doc-level global z + MSE +
            # EmbeddingGemma only); setting the token-mode knobs alongside
            # it would silently mean nothing — refuse instead.
            if self.label_transform != "doc_z_asinh" or self.loss != "huber_pearson":
                raise ValueError(
                    "objective=doc fixes transform (doc-level global z) and "
                    "loss (mse); do not set label_transform/loss with it"
                )
            if self.run_twin:
                raise ValueError(
                    "objective=doc supports EmbeddingGemma only — pass "
                    "run_twin=false"
                )
        if self.early_stop_patience < 0 or self.early_stop_min_delta < 0:
            raise ValueError("early_stop_patience/min_delta must be >= 0")
        if self.embedding_tokens_per_batch < contracts.EMBEDDINGGEMMA_WINDOW:
            raise ValueError(
                "embedding_tokens_per_batch must fit at least one window"
            )
        if self.twin_tokens_per_batch < 1:
            raise ValueError("twin_tokens_per_batch must be positive")

    def tokens_per_batch(self, model_kind: str) -> int:
        return (
            self.embedding_tokens_per_batch
            if model_kind == "embeddinggemma"
            else self.twin_tokens_per_batch
        )

    @property
    def channels(self) -> tuple[str, ...]:
        """Head output channels: v1 trains a delta head; v2 (global_z) and
        the doc diagnostic train only the two directions — delta is
        DERIVED downstream."""
        if self.label_transform == "global_z" or self.objective == "doc":
            return labels.GLOBAL_Z_CHANNELS
        return labels.CHANNELS

    @property
    def eval_each_epoch(self) -> bool:
        return self.eval_every_epoch or self.early_stop_patience > 0


def surrogate_config_digest(cfg: SurrogateConfig) -> str:
    """Config identity for the stage receipt: everything science-shaped
    (runtime placement excluded) — a relaunch with different knobs must
    RE-RUN the stage instead of resuming stale artifacts."""
    payload = {
        key: value
        for key, value in dataclasses.asdict(cfg).items()
        if key not in ("device",)
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ------------------------------------------------------------- data assembly
def build_docs(
    label_rows: list[dict[str, Any]], cfg: SurrogateConfig
) -> tuple[list[dict], dict, dict[str, dict[str, float]] | None]:
    """labels.parquet rows -> docs with transformed channels.

    v1 (doc_z_asinh): per-doc z + asinh, three channels, short/flat docs
    filtered. v2 (global_z): all docs as-is (empty-doc guard only), two
    channels z-normalized with ONE global mean/std per channel computed
    over the TRAINING split and frozen (validation uses the train
    constants). Returns (docs, counts, global_z_constants-or-None).
    """
    docs, dropped = [], {"too_short_or_flat": 0}
    for row in label_rows:
        token_ids = [int(t) for t in row["token_ids"]]
        if not token_ids:
            raise RuntimeError(f"labels row for doc {row['doc_id']} is empty")
        if cfg.label_transform == "doc_z_asinh":
            channels = labels.doc_label_channels(row["s_coin"], row["s_charter"])
            if channels is None:
                dropped["too_short_or_flat"] += 1
                continue
        else:
            channels = {
                "coin": [float(v) for v in row["s_coin"]],
                "charter": [float(v) for v in row["s_charter"]],
            }
        docs.append(
            {
                "doc_id": int(row["doc_id"]),
                "pool": row["pool"],
                "token_ids": token_ids,
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
    constants: dict[str, dict[str, float]] | None = None
    if cfg.label_transform == "global_z":
        train_docs = [d for d in docs if not d["validation"]]
        constants = {}
        for name in cfg.channels:
            mean, std = labels.global_z_constants(
                [d["channels"][name] for d in train_docs]
            )
            constants[name] = {"mean": mean, "std": std}
        for doc in docs:  # validation uses the frozen TRAIN constants
            for name in cfg.channels:
                doc["channels"][name] = labels.apply_global_z(
                    doc["channels"][name],
                    constants[name]["mean"],
                    constants[name]["std"],
                )
        counts["global_z_constants"] = constants
    return docs, counts, constants


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
                    name: [values[i] for i in order]
                    for name, values in doc["channels"].items()
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
    if model_kind != "embeddinggemma":
        # The 8192-context causal twin needs activation checkpointing to
        # train (attempt-3 OOM at 137 GiB without it); LoRA freezes the
        # base weights, so checkpointed segment inputs must be forced to
        # require grad or no gradient reaches the adapters.
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        backbone.enable_input_require_grads()
    lora = LoraConfig(
        r=contracts.LORA_R,
        lora_alpha=contracts.LORA_ALPHA,
        lora_dropout=contracts.LORA_DROPOUT,
        target_modules=list(LORA_TARGETS),
        bias="none",
    )
    model = get_peft_model(backbone, lora).to(cfg.device)
    head = torch.nn.Linear(backbone.config.hidden_size, len(cfg.channels)).to(
        cfg.device, torch.float32
    )
    return model, head, model_id, revision


# ---------------------------------------------------------------- train/eval
def _cost_batches(
    items: list[dict], max_cost: int, max_items: int, cost: Any
) -> list[list[dict]]:
    """Cost-bucketed micro-batches under a per-forward budget.

    Items are sorted costliest-first (an OOM surfaces on micro-batch 1, and
    padding waste stays low), then greedily packed until adding the next
    item would exceed ``max_cost`` or ``max_items``. A single item costlier
    than the budget still forms its own batch — memory is then bounded by
    the largest item, which is exactly the quantity the budget pins.
    """
    if max_cost < 1 or max_items < 1:
        raise ValueError("max_cost and max_items must be positive")
    ordered = sorted(items, key=cost, reverse=True)
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_cost = 0
    for item in ordered:
        item_cost = cost(item)
        if current and (
            current_cost + item_cost > max_cost or len(current) >= max_items
        ):
            batches.append(current)
            current, current_cost = [], 0
        current.append(item)
        current_cost += item_cost
    if current:
        batches.append(current)
    return batches


def _token_batches(
    items: list[dict], max_tokens: int, max_items: int
) -> list[list[dict]]:
    """Token-budget batching for window items (cost = token count)."""
    return _cost_batches(
        items, max_tokens, max_items, cost=lambda item: len(item["ids"])
    )


def _accumulation_groups(
    batches: list[list[dict]], target_items: int
) -> list[list[list[dict]]]:
    """Group micro-batches into optimizer steps of >= target_items items
    (the last group keeps the remainder)."""
    if target_items < 1:
        raise ValueError("target_items must be positive")
    groups: list[list[list[dict]]] = []
    current: list[list[dict]] = []
    count = 0
    for batch in batches:
        current.append(batch)
        count += len(batch)
        if count >= target_items:
            groups.append(current)
            current, count = [], 0
    if current:
        groups.append(current)
    return groups


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


class EarlyStopper:
    """Patience-based early stopping on a lower-is-better metric (val
    delta-FUV). Pure bookkeeping so the logic is CPU-testable; the caller
    owns state snapshots (keyed off ``improved``)."""

    def __init__(self, patience: int, min_delta: float) -> None:
        if patience < 0 or min_delta < 0:
            raise ValueError("patience and min_delta must be >= 0")
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = -1
        self.stale = 0

    def update(self, epoch: int, value: float) -> bool:
        """Record an epoch's metric; True iff it improved on the best by
        more than min_delta (the caller should snapshot state then)."""
        if value < self.best - self.min_delta:
            self.best = value
            self.best_epoch = epoch
            self.stale = 0
            return True
        self.stale += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.stale >= self.patience


def _snapshot_trainable(model: Any, head: Any) -> dict[str, Any]:
    state = {
        f"model.{name}": parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    state.update(
        {
            f"head.{name}": parameter.detach().cpu().clone()
            for name, parameter in head.named_parameters()
        }
    )
    return state


def _restore_trainable(model: Any, head: Any, state: dict[str, Any]) -> None:
    import torch

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(state[f"model.{name}"].to(parameter.device))
        for name, parameter in head.named_parameters():
            parameter.copy_(state[f"head.{name}"].to(parameter.device))


def _train_means(train_docs: list[dict], channels: tuple[str, ...]) -> dict[str, float]:
    """Per-channel train-label means (the FUV baseline), plus the derived
    delta mean when no delta head is trained."""
    means: dict[str, float] = {}
    for name in channels:
        total, count = 0.0, 0
        for doc in train_docs:
            total += sum(doc["channels"][name])
            count += len(doc["channels"][name])
        means[name] = total / max(count, 1)
    if "delta" not in channels:
        means["delta"] = means["coin"] - means["charter"]
    return means


def _doc_label_delta(doc: dict, channels: tuple[str, ...]) -> list[float]:
    if "delta" in channels:
        return doc["channels"]["delta"]
    return [
        a - b
        for a, b in zip(doc["channels"]["coin"], doc["channels"]["charter"])
    ]


def evaluate_split(
    model: Any,
    head: Any,
    val_docs: list[dict],
    cfg: SurrogateConfig,
    model_kind: str,
    train_means: dict[str, float],
) -> dict[str, Any]:
    """Validation metrics in the transformed label space: FUV per channel
    + FUV of the contrast delta (derived from per-channel predictions when
    no delta head exists) over ALL val token positions, plus the unchanged
    per-doc mean Spearman per channel."""
    import torch
    from scipy.stats import spearmanr

    channels = cfg.channels
    flat_true: dict[str, list[float]] = {name: [] for name in (*channels, "delta")}
    flat_pred: dict[str, list[float]] = {name: [] for name in (*channels, "delta")}
    per_doc_rho: dict[str, list[float]] = {name: [] for name in (*channels, "delta")}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for doc in val_docs:
            stitched = predict_doc(
                model, head, doc["token_ids"], model_kind, cfg.device,
                n_channels=len(channels),
            )
            doc_pred: dict[str, list[float]] = {}
            for index, name in enumerate(channels):
                doc_pred[name] = [float(v) for v in stitched[:, index]]
            if "delta" not in channels:
                doc_pred["delta"] = [
                    a - b
                    for a, b in zip(doc_pred["coin"], doc_pred["charter"])
                ]
            doc_true = {name: doc["channels"][name] for name in channels}
            doc_true["delta"] = _doc_label_delta(doc, channels)
            for name in flat_true:
                flat_true[name].extend(doc_true[name])
                flat_pred[name].extend(doc_pred[name])
                rho = spearmanr(doc_pred[name], doc_true[name])[0]
                if rho == rho:  # not NaN (constant doc)
                    per_doc_rho[name].append(float(rho))
    if was_training:
        model.train()
    fuv_metrics = {
        name: labels.fuv(flat_true[name], flat_pred[name], train_means[name])
        for name in flat_true
    }
    val_mse = {
        name: statistics.fmean(
            (t - p) ** 2 for t, p in zip(flat_true[name], flat_pred[name])
        )
        for name in flat_true
    }
    spearman_metrics = {
        name: {
            "mean_spearman": statistics.mean(values) if values else float("nan"),
            "n_docs": len(values),
        }
        for name, values in per_doc_rho.items()
    }
    return {
        "fuv": fuv_metrics,
        "val_mse": val_mse,
        "spearman": spearman_metrics,
        "n_positions": len(flat_true["delta"]),
    }


def train_one(
    model_kind: str,
    docs: list[dict],
    cfg: SurrogateConfig,
    token: str,
    tag: str,
) -> dict[str, Any]:
    """Train + evaluate one surrogate; returns metrics + artifact handles."""
    import torch

    torch.manual_seed(cfg.seed)
    model, head, model_id, revision = load_backbone(model_kind, cfg, token)
    vocab_size = int(model.config.vocab_size)
    max_id = max(max(d["token_ids"]) for d in docs)
    if max_id >= vocab_size:
        raise RuntimeError(
            f"stored token id {max_id} >= {model_kind} vocab {vocab_size} — "
            "label transfer by id is unsound; refusing"
        )

    channels = cfg.channels
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
                for name in channels
            },
        }
        for d in train_docs
        for start, _, _ in doc_windows(d, model_kind)
    ]
    if cfg.max_windows:
        windows = windows[: cfg.max_windows]
    rng = random.Random(cfg.seed)
    huber = torch.nn.HuberLoss(delta=contracts.HUBER_DELTA)
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(head.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    def micro_batch_loss(batch: list[dict]) -> Any:
        """v1: mean Huber over items x channels + Pearson auxiliary.
        v2 (loss=mse): plain MSE on the z-normed channels, no aux."""
        scores = _forward_scores(model, head, batch, cfg.device)
        total = scores.new_zeros(())
        aux_terms = []
        for row, item in enumerate(batch):
            n = len(item["ids"])
            for channel_index, name in enumerate(channels):
                target = torch.tensor(
                    item["labels"][name], dtype=torch.float32,
                    device=cfg.device,
                )
                prediction = scores[row, :n, channel_index]
                if cfg.loss == "mse":
                    total = total + mse(prediction, target)
                    continue
                total = total + huber(prediction, target)
                r = _pearson(prediction, target)
                if r is not None:
                    aux_terms.append(1.0 - r)
        total = total / (len(batch) * len(channels))
        if aux_terms:
            total = total + contracts.PEARSON_AUX_WEIGHT * torch.stack(
                aux_terms
            ).mean()
        return total

    train_means = _train_means(train_docs, channels)
    stopper = EarlyStopper(cfg.early_stop_patience, cfg.early_stop_min_delta)
    best_state: dict[str, Any] | None = None
    fuv_curve: list[dict[str, Any]] = []
    stopped_epoch = cfg.max_epochs
    model.train()
    started = time.time()
    tokens_budget = cfg.tokens_per_batch(model_kind)
    for epoch in range(1, cfg.max_epochs + 1):
        rng.shuffle(windows)
        micro_batches = _token_batches(windows, tokens_budget, cfg.batch_windows)
        rng.shuffle(micro_batches)
        epoch_loss_sum, epoch_items, steps = 0.0, 0, 0
        for group in _accumulation_groups(micro_batches, cfg.batch_windows):
            group_items = sum(len(batch) for batch in group)
            optimizer.zero_grad(set_to_none=True)
            for batch in group:
                micro = micro_batch_loss(batch)
                # Item-weighted accumulation: gradients match a single
                # group-sized batch up to the aux term's within-batch mean.
                (micro * (len(batch) / group_items)).backward()
                epoch_loss_sum += float(micro) * len(batch)
                epoch_items += len(batch)
            optimizer.step()
            steps += 1
        train_loss = epoch_loss_sum / max(epoch_items, 1)
        if not cfg.eval_each_epoch:
            common.log(
                f"{tag}/{model_kind} epoch {epoch}/{cfg.max_epochs}: "
                f"loss {train_loss:.4f} ({steps} steps, "
                f"{time.time() - started:.0f}s)"
            )
            continue
        epoch_eval = evaluate_split(
            model, head, val_docs, cfg, model_kind, train_means
        )
        fuv_curve.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "fuv": epoch_eval["fuv"],
                "val_mse": epoch_eval["val_mse"],
            }
        )
        improved = stopper.update(epoch, epoch_eval["fuv"]["delta"])
        if improved and cfg.early_stop_patience > 0:
            best_state = _snapshot_trainable(model, head)
        common.log(
            f"{tag}/{model_kind} epoch {epoch}/{cfg.max_epochs}: "
            f"loss {train_loss:.4f}, val FUV "
            + " ".join(
                f"{name} {value:.4f}"
                for name, value in epoch_eval["fuv"].items()
            )
            + (" [best]" if improved else f" [stale {stopper.stale}]")
        )
        if stopper.should_stop:
            stopped_epoch = epoch
            common.log(
                f"{tag}/{model_kind} early stop at epoch {epoch} "
                f"(best delta-FUV {stopper.best:.4f} @ epoch "
                f"{stopper.best_epoch}, patience {cfg.early_stop_patience})"
            )
            break
    if best_state is not None:
        _restore_trainable(model, head, best_state)

    final = evaluate_split(model, head, val_docs, cfg, model_kind, train_means)
    return {
        "model_kind": model_kind,
        "model_id": model_id,
        "revision": revision,
        "metrics": final["spearman"],
        "fuv": final["fuv"],
        "val_mse": final["val_mse"],
        "fuv_curve": fuv_curve,
        "best_epoch": stopper.best_epoch if best_state is not None else None,
        "stopped_epoch": stopped_epoch,
        "channels": list(channels),
        "n_train_docs": len(train_docs),
        "n_val_docs": len(val_docs),
        "n_train_windows": len(windows),
        "_model": model,
        "_head": head,
    }


def predict_doc(
    model: Any,
    head: Any,
    token_ids: list[int],
    model_kind: str,
    device: str,
    n_channels: int = len(labels.CHANNELS),
) -> Any:
    """Per-token channel predictions for one doc (center-crop stitched)."""
    import torch

    outputs = torch.zeros((len(token_ids), n_channels), dtype=torch.float32)
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


# ------------------------------------------------------ doc-level diagnostic
DOC_STAGE_LABEL = "surrogate_doc"


def doc_targets_from_row(row: dict[str, Any]) -> dict[str, float]:
    """Per-doc per-token-MEAN influence over the labeled coverage:
    sum(s_ch) / min(n_doc_tokens, LABEL_COVERAGE_TOKENS). The stored s
    arrays cover exactly that many positions — asserted, not assumed."""
    expected = min(int(row["n_doc_tokens"]), contracts.LABEL_COVERAGE_TOKENS)
    targets: dict[str, float] = {}
    for channel, key in (("coin", "s_coin"), ("charter", "s_charter")):
        values = row[key]
        if len(values) != expected:
            raise ValueError(
                f"doc {row['doc_id']}: {key} has {len(values)} entries, "
                f"expected min(n_doc_tokens={row['n_doc_tokens']}, "
                f"{contracts.LABEL_COVERAGE_TOKENS}) = {expected}"
            )
        targets[channel] = sum(float(v) for v in values) / expected
    return targets


def build_doc_items(
    label_rows: list[dict[str, Any]], cfg: SurrogateConfig
) -> tuple[list[dict], dict, dict[str, dict[str, float]]]:
    """labels rows -> doc items with z-normalized scalar targets.

    Targets are global-z per channel over TRAIN docs only (frozen
    constants, validation normalized with them) — the v2 machinery at doc
    granularity. Windows reuse the embeddinggemma stride plan; ``cost`` =
    total window tokens (what a forward actually carries)."""
    items: list[dict[str, Any]] = []
    for row in label_rows:
        token_ids = [int(t) for t in row["token_ids"]]
        if not token_ids:
            raise RuntimeError(f"labels row for doc {row['doc_id']} is empty")
        windows = [
            {"ids": token_ids[start : start + contracts.EMBEDDINGGEMMA_WINDOW]}
            for start, _, _ in common.window_spans(len(token_ids))
        ]
        items.append(
            {
                "doc_id": int(row["doc_id"]),
                "pool": row["pool"],
                "windows": windows,
                "cost": sum(len(w["ids"]) for w in windows),
                "targets": doc_targets_from_row(row),
                "validation": labels.is_validation_doc(int(row["doc_id"])),
            }
        )
    train_items = [item for item in items if not item["validation"]]
    if not train_items or len(train_items) == len(items):
        raise RuntimeError("degenerate doc split for the doc diagnostic")
    constants: dict[str, dict[str, float]] = {}
    for channel in labels.GLOBAL_Z_CHANNELS:
        mean, std = labels.global_z_constants(
            [[item["targets"][channel] for item in train_items]]
        )
        constants[channel] = {"mean": mean, "std": std}
    for item in items:
        item["targets"] = {
            channel: (item["targets"][channel] - constants[channel]["mean"])
            / constants[channel]["std"]
            for channel in labels.GLOBAL_Z_CHANNELS
        }
    counts = {
        "usable": len(items),
        "validation": sum(item["validation"] for item in items),
        "global_z_constants": constants,
    }
    return items, counts, constants


def require_no_weights(stage_dir: Path) -> None:
    """Jonathan had the trained surrogate weights deleted — the doc
    diagnostic must upload NO model bytes (metrics/curves/receipt only).
    Hard guard immediately before the upload, not a convention."""
    stray = sorted(
        str(path)
        for path in stage_dir.rglob("*")
        if path.is_file() and path.suffix in (".safetensors", ".bin", ".pt")
    )
    if stray:
        raise RuntimeError(
            f"doc diagnostic stage dir contains weight files: {stray}"
        )


def shuffle_doc_targets_across(items: list[dict], seed: int) -> list[dict]:
    """The doc-level noise floor: permute the (coin, charter) target PAIRS
    across TRAIN docs (pairs stay intact); validation targets untouched."""
    rng = random.Random(seed)
    train_slots = [i for i, item in enumerate(items) if not item["validation"]]
    order = list(train_slots)
    rng.shuffle(order)
    shuffled = [dict(item) for item in items]
    for slot, source in zip(train_slots, order, strict=True):
        shuffled[slot]["targets"] = dict(items[source]["targets"])
    return shuffled


def _pool_windows(hidden: Any, mask: Any) -> Any:
    """Masked mean over positions: [W, L, H] x [W, L] -> [W, H]."""
    mask_f = mask.to(hidden.dtype).unsqueeze(-1)
    denominator = mask_f.sum(dim=1).clamp_min(1.0)
    return (hidden * mask_f).sum(dim=1) / denominator


def _doc_embeddings(window_embs: Any, doc_of: Any, n_docs: int) -> Any:
    """Mean window embedding per doc: [W, H] grouped by doc_of -> [D, H]."""
    import torch

    sums = torch.zeros(
        (n_docs, window_embs.shape[1]),
        dtype=window_embs.dtype,
        device=window_embs.device,
    )
    counts = torch.zeros(n_docs, dtype=window_embs.dtype,
                         device=window_embs.device)
    sums.index_add_(0, doc_of, window_embs)
    counts.index_add_(
        0, doc_of, torch.ones_like(doc_of, dtype=window_embs.dtype)
    )
    if bool((counts == 0).any()):
        raise RuntimeError("a doc contributed no windows — pooling bug")
    return sums / counts.unsqueeze(1)


def _forward_doc_batch(model: Any, head: Any, batch_docs: list[dict],
                       device: str) -> Any:
    """All windows of the batch docs in one padded forward -> preds [D, C]:
    masked mean-pool per window, mean over each doc's windows, fp32 head."""
    import torch

    window_items: list[dict] = []
    doc_of: list[int] = []
    for index, doc in enumerate(batch_docs):
        for window in doc["windows"]:
            window_items.append(window)
            doc_of.append(index)
    max_len = max(len(w["ids"]) for w in window_items)
    ids = torch.zeros((len(window_items), max_len), dtype=torch.int64,
                      device=device)
    mask = torch.zeros((len(window_items), max_len), dtype=torch.int64,
                       device=device)
    for row, window in enumerate(window_items):
        n = len(window["ids"])
        ids[row, :n] = torch.tensor(window["ids"], dtype=torch.int64)
        mask[row, :n] = 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
    window_embs = _pool_windows(hidden.float(), mask)
    doc_embs = _doc_embeddings(
        window_embs, torch.tensor(doc_of, device=device), len(batch_docs)
    )
    return head(doc_embs)


def evaluate_doc_split(
    model: Any,
    head: Any,
    val_items: list[dict],
    cfg: SurrogateConfig,
    train_means: dict[str, float],
) -> dict[str, Any]:
    """Doc-level FUV + Spearman across the validation docs (n recorded)."""
    import torch
    from scipy.stats import spearmanr

    channels = labels.GLOBAL_Z_CHANNELS
    preds: dict[str, list[float]] = {name: [] for name in (*channels, "delta")}
    true: dict[str, list[float]] = {name: [] for name in (*channels, "delta")}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in _cost_batches(
            val_items, cfg.embedding_tokens_per_batch, cfg.batch_windows,
            cost=lambda item: item["cost"],
        ):
            scores = _forward_doc_batch(model, head, batch, cfg.device)
            for row, item in enumerate(batch):
                for index, name in enumerate(channels):
                    preds[name].append(float(scores[row, index]))
                    true[name].append(item["targets"][name])
                preds["delta"].append(
                    float(scores[row, 0] - scores[row, 1])
                )
                true["delta"].append(
                    item["targets"]["coin"] - item["targets"]["charter"]
                )
    if was_training:
        model.train()
    fuv_metrics = {
        name: labels.fuv(true[name], preds[name], train_means[name])
        for name in true
    }
    spearman_metrics = {}
    for name in true:
        rho = spearmanr(preds[name], true[name])[0]
        spearman_metrics[name] = float(rho) if rho == rho else float("nan")
    return {
        "fuv": fuv_metrics,
        "spearman": spearman_metrics,
        "n_docs": len(val_items),
    }


def train_doc_probe(
    items: list[dict], cfg: SurrogateConfig, token: str, tag: str
) -> dict[str, Any]:
    """The MATES-style doc-level probe: EmbeddingGemma LoRA + linear head
    on the mean-pooled doc embedding, MSE on z-normed per-doc targets,
    v2 convergence scheme (early stop on val doc delta-FUV)."""
    import torch

    torch.manual_seed(cfg.seed)
    model, head, model_id, revision = load_backbone(
        PRIMARY_SURROGATE, cfg, token
    )
    train_items = [item for item in items if not item["validation"]]
    val_items = [item for item in items if item["validation"]]
    channels = labels.GLOBAL_Z_CHANNELS
    train_means = {
        name: statistics.fmean(item["targets"][name] for item in train_items)
        for name in channels
    }
    train_means["delta"] = statistics.fmean(
        item["targets"]["coin"] - item["targets"]["charter"]
        for item in train_items
    )
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(head.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    rng = random.Random(cfg.seed)
    stopper = EarlyStopper(cfg.early_stop_patience, cfg.early_stop_min_delta)
    best_state: dict[str, Any] | None = None
    fuv_curve: list[dict[str, Any]] = []
    stopped_epoch = cfg.max_epochs
    model.train()
    started = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        batches = _cost_batches(
            list(items_train_order(train_items, rng)),
            cfg.embedding_tokens_per_batch,
            cfg.batch_windows,
            cost=lambda item: item["cost"],
        )
        rng.shuffle(batches)
        epoch_loss_sum, epoch_items = 0.0, 0
        for group in _accumulation_groups(batches, cfg.batch_windows):
            group_items = sum(len(batch) for batch in group)
            optimizer.zero_grad(set_to_none=True)
            for batch in group:
                predictions = _forward_doc_batch(model, head, batch, cfg.device)
                targets = torch.tensor(
                    [
                        [item["targets"][name] for name in channels]
                        for item in batch
                    ],
                    dtype=torch.float32,
                    device=cfg.device,
                )
                loss = mse(predictions, targets)
                (loss * (len(batch) / group_items)).backward()
                epoch_loss_sum += float(loss) * len(batch)
                epoch_items += len(batch)
            optimizer.step()
        train_loss = epoch_loss_sum / max(epoch_items, 1)
        if not cfg.eval_each_epoch:
            common.log(f"{tag} epoch {epoch}/{cfg.max_epochs}: "
                       f"loss {train_loss:.4f}")
            continue
        epoch_eval = evaluate_doc_split(model, head, val_items, cfg, train_means)
        fuv_curve.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "fuv": epoch_eval["fuv"],
                "spearman": epoch_eval["spearman"],
            }
        )
        improved = stopper.update(epoch, epoch_eval["fuv"]["delta"])
        if improved and cfg.early_stop_patience > 0:
            best_state = _snapshot_trainable(model, head)
        common.log(
            f"{tag} epoch {epoch}/{cfg.max_epochs}: loss {train_loss:.4f}, "
            "val FUV "
            + " ".join(f"{k} {v:.4f}" for k, v in epoch_eval["fuv"].items())
            + (" [best]" if improved else f" [stale {stopper.stale}]")
            + f" ({time.time() - started:.0f}s)"
        )
        if stopper.should_stop:
            stopped_epoch = epoch
            common.log(
                f"{tag} early stop at epoch {epoch} (best delta-FUV "
                f"{stopper.best:.4f} @ epoch {stopper.best_epoch})"
            )
            break
    if best_state is not None:
        _restore_trainable(model, head, best_state)
    final = evaluate_doc_split(model, head, val_items, cfg, train_means)
    return {
        "model_id": model_id,
        "revision": revision,
        "fuv": final["fuv"],
        "spearman": final["spearman"],
        "n_val_docs": final["n_docs"],
        "fuv_curve": fuv_curve,
        "best_epoch": stopper.best_epoch if best_state is not None else None,
        "stopped_epoch": stopped_epoch,
        "n_train_docs": len(train_items),
    }


def items_train_order(train_items: list[dict], rng: random.Random) -> list[dict]:
    ordered = list(train_items)
    rng.shuffle(ordered)
    return ordered


PRIMARY_SURROGATE = contracts.SURROGATE_MODELS[0]  # embeddinggemma


def _train_with_retry(
    model_kind: str, docs: list[dict], cfg: SurrogateConfig, token: str, tag: str
) -> dict[str, Any]:
    """One OOM retry at half token budgets before giving up."""
    import torch

    try:
        return train_one(model_kind, docs, cfg, token, tag)
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        halved = dataclasses.replace(
            cfg,
            embedding_tokens_per_batch=max(
                contracts.EMBEDDINGGEMMA_WINDOW,
                cfg.embedding_tokens_per_batch // 2,
            ),
            twin_tokens_per_batch=max(1_024, cfg.twin_tokens_per_batch // 2),
        )
        common.log(
            f"{tag}/{model_kind} hit CUDA OOM ({error}); retrying once at "
            f"half token budgets ({halved.embedding_tokens_per_batch}/"
            f"{halved.twin_tokens_per_batch})"
        )
        return train_one(model_kind, docs, halved, token, tag)


def surrogate_resume_action(
    hub_receipt: dict[str, Any],
    config_digest: str,
    expected_status: str = "complete",
) -> str:
    """Config-aware resume decision for published surrogate evidence.

    'rerun'  — the Hub evidence was produced by a DIFFERENT config (or a
               pre-digest v1 receipt): retrain with the current config,
               whatever the old status was (incl. a stale NO-GO).
    'resume' — same config, terminal status (``expected_status``:
               "complete" for the token stage, "diagnostic" for doc mode).
    raises   — same config but non-terminal: investigate-first, a NO-GO
               must not be silently re-executed into green by relaunching
               with identical settings.
    """
    if hub_receipt.get("config_digest") != config_digest:
        return "rerun"
    if hub_receipt.get("status") != expected_status:
        raise RuntimeError(
            "surrogate evidence exists with THIS config but status "
            f"{hub_receipt.get('status')!r} (expected {expected_status!r}) — "
            "refusing to resume past it (a NO-GO must stop the pipeline on "
            "relaunch too)"
        )
    return "resume"


def train_candidates(
    docs: list[dict], cfg: SurrogateConfig, token: str, train_fn: Any
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Train every candidate; the PRIMARY must succeed, ablation twins
    degrade gracefully (recorded + logged loudly, never fatal — an
    ablation must not kill a run whose primary surrogate trained)."""
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    kinds = (
        contracts.SURROGATE_MODELS if cfg.run_twin else (PRIMARY_SURROGATE,)
    )
    for model_kind in kinds:
        try:
            results[model_kind] = train_fn(model_kind, docs, cfg, token, "real")
        except Exception as error:  # noqa: BLE001 - twin degrade is deliberate
            if model_kind == PRIMARY_SURROGATE:
                raise
            failures[model_kind] = f"{type(error).__name__}: {error}"
            common.log(
                "ABLATION TWIN FAILED — degrading to primary-only selection "
                f"({model_kind}): {failures[model_kind]}"
            )
    if not results:
        raise RuntimeError("no surrogate candidate trained")
    return results, failures


# ---------------------------------------------------------------- main stage
async def main(cfg: SurrogateConfig) -> dict[str, Any]:
    return await asyncio.to_thread(_run, cfg)


def _run_doc_mode(cfg: SurrogateConfig, env: common.PodEnv) -> dict[str, Any]:
    """objective=doc: the MATES-style doc-level diagnostic. Own evidence
    path (pod/surrogate_doc/), NO GO/NO-GO raise, NO weights uploaded —
    metrics/curves/receipt only."""
    stage_dir = env.stage_dir(DOC_STAGE_LABEL)
    started = time.time()
    config_digest = surrogate_config_digest(cfg)
    if common.stage_remote_files(env.run_id, DOC_STAGE_LABEL):
        receipt_path = common.fetch_stage_file(
            env.run_id, DOC_STAGE_LABEL,
            contracts.STAGE_RECEIPTS[DOC_STAGE_LABEL],
            env.scratch_root / "resume",
        )
        hub_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        action = surrogate_resume_action(
            hub_receipt, config_digest, expected_status="diagnostic"
        )
        if action == "resume":
            common.log("doc diagnostic already on HF (same config) — resuming")
            return {"stage": DOC_STAGE_LABEL, "status": "resumed_from_hub",
                    "resumed_receipt": hub_receipt}
        common.log(
            "doc-diagnostic evidence on the Hub used a different config "
            f"({hub_receipt.get('config_digest')} != {config_digest}) — "
            "re-running"
        )

    labels_path = env.evidence_root / "extract" / contracts.LABELS_PARQUET
    label_rows = common.read_parquet_rows(labels_path)
    items, counts, constants = build_doc_items(label_rows, cfg)
    common.log(f"doc diagnostic targets: {counts}")

    result = train_doc_probe(items, cfg, env.hf_token, "doc")
    shuffled_items = shuffle_doc_targets_across(items, cfg.seed + 1)
    floor = train_doc_probe(shuffled_items, cfg, env.hf_token, "doc-shuffled")

    selection = {
        "objective": "doc",
        "status_note": (
            "diagnostic — per-doc mean-influence probe (MATES-style); no "
            "GO/NO-GO gate, no weights persisted (metrics/curves only)"
        ),
        "model_id": result["model_id"],
        "revision": result["revision"],
        "doc_targets": "sum(s_ch)/min(n_doc_tokens, 8191), global-z per "
                       "channel over TRAIN docs",
        "global_z_constants": constants,
        "fuv_final": result["fuv"],
        "spearman_final": result["spearman"],
        "fuv_curve": result["fuv_curve"],
        "best_epoch": result["best_epoch"],
        "stopped_epoch": result["stopped_epoch"],
        "shuffled_fuv_final": floor["fuv"],
        "shuffled_spearman_final": floor["spearman"],
        "shuffled_fuv_curve": floor["fuv_curve"],
        "n_train_docs": result["n_train_docs"],
        "n_val_docs": result["n_val_docs"],
        "label_doc_counts": {k: v for k, v in counts.items()
                             if k != "global_z_constants"},
        "config": dataclasses.asdict(cfg),
        "config_digest": config_digest,
        "early_stop_metric": "fuv_delta",
    }
    common.atomic_json(stage_dir / "selection.json", selection)
    receipt = {
        "stage": DOC_STAGE_LABEL,
        "run_id": env.run_id,
        "status": "diagnostic",
        "elapsed_s": round(time.time() - started, 1),
        **selection,
    }
    common.atomic_json(
        stage_dir / contracts.STAGE_RECEIPTS[DOC_STAGE_LABEL], receipt
    )
    require_no_weights(stage_dir)
    common.upload_evidence(stage_dir, env.run_id, DOC_STAGE_LABEL)
    common.log(
        "doc diagnostic complete: val FUV "
        + " ".join(f"{k} {v:.4f}" for k, v in result["fuv"].items())
        + " | shuffled floor "
        + " ".join(f"{k} {v:.4f}" for k, v in floor["fuv"].items())
    )
    return receipt


def _run(cfg: SurrogateConfig) -> dict[str, Any]:
    from safetensors.torch import save_file

    env = common.pod_env()
    if cfg.objective == "doc":
        return _run_doc_mode(cfg, env)
    stage_dir = env.stage_dir("surrogate")
    started = time.time()

    config_digest = surrogate_config_digest(cfg)
    if common.stage_remote_files(env.run_id, "surrogate"):
        # Resume is CONFIG-AWARE: evidence produced by a different config
        # (e.g. the v1 run before Jonathan's v2 directive) must be
        # re-trained, never resumed — the fresh upload replaces the stale
        # artifacts in one atomic upload_folder commit, so score can only
        # ever see a coherent (new) set. Same-config evidence keeps the
        # minor-1 semantics: complete resumes, NO-GO/failed raises.
        receipt_path = common.fetch_stage_file(
            env.run_id, "surrogate", contracts.STAGE_RECEIPTS["surrogate"],
            env.scratch_root / "resume",
        )
        hub_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        action = surrogate_resume_action(hub_receipt, config_digest)
        if action == "rerun":
            common.log(
                "surrogate evidence on the Hub was produced by a DIFFERENT "
                f"config (digest {hub_receipt.get('config_digest')} != "
                f"{config_digest}, status {hub_receipt.get('status')!r}) — "
                "RE-RUNNING the stage with the current config; the fresh "
                "upload replaces the stale artifacts in one atomic commit"
            )
        else:
            common.log("surrogate already complete on HF (same config) — "
                       "resuming")
            fetched = common.fetch_stage_file(
                env.run_id, "surrogate", "selection.json",
                env.scratch_root / "resume",
            )
            shutil.copy2(fetched, stage_dir / "selection.json")
            return {"stage": "surrogate", "status": "resumed_from_hub",
                    "resumed_receipt": hub_receipt}

    labels_path = env.evidence_root / "extract" / contracts.LABELS_PARQUET
    label_rows = common.read_parquet_rows(labels_path)
    docs, counts, global_constants = build_docs(label_rows, cfg)
    common.log(f"label docs ({cfg.label_transform}): {counts}")

    results, twin_failures = train_candidates(
        docs, cfg, env.hf_token, _train_with_retry
    )

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
    # The floor is load-bearing for GO/NO-GO: a failure here (after the OOM
    # retry) is fatal by design — no floor, no spend decision.
    shuffled = _train_with_retry(
        selected_kind, shuffled_docs, cfg, env.hf_token, "shuffled"
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
        "objective": "token",
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
        # An ablation twin that could not train (recorded, never fatal).
        "twin_failures": twin_failures,
        "label_doc_counts": counts,
        "config": dataclasses.asdict(cfg),
        "config_digest": config_digest,
        "label_transform": cfg.label_transform,
        "loss": cfg.loss,
        # Head layout: score derives delta from the per-channel outputs
        # when no delta head exists (global_z).
        "output_channels": selected["channels"],
        "global_z_constants": global_constants,
        # The convergence deliverable: per-epoch validation FUV/loss curve
        # (empty when eval_every_epoch/early stopping are off — v1 mode),
        # + the final FUV of the SELECTED and SHUFFLED runs.
        "early_stop_metric": "fuv_delta",
        "fuv_curve": selected["fuv_curve"],
        "fuv_final": selected["fuv"],
        "val_mse_final": selected["val_mse"],
        "best_epoch": selected["best_epoch"],
        "stopped_epoch": selected["stopped_epoch"],
        "shuffled_fuv_final": shuffled["fuv"],
        "shuffled_fuv_curve": shuffled["fuv_curve"],
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
