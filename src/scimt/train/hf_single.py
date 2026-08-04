"""``hf_single`` backend: in-process full-parameter finetuning on ONE GPU.

Why a second backend exists
---------------------------
:mod:`scimt.train.axolotl` drives multi-GPU FSDP2 through a supervised
subprocess because a process-group launcher cannot run inside the caller's
event loop. At 1B that machinery buys nothing: full-parameter AdamW on
``google/gemma-3-1b-pt`` needs ~16GB, so a single H200 holds the whole run, and
the job is compute-bound on small matmuls rather than memory-bound. Sharding it
would add a launcher, a rendezvous, and a fresh class of silent failure (FSDP2's
end-of-training save no-op) in exchange for ~nothing.

So this backend occupies the :class:`~scimt.train.Backend` protocol seam that
``__init__`` documents as reserved for exactly this, and it is *more* aligned
with the repo's "async-native, no CLIs" rule than the axolotl carve-out is:
there is no subprocess, no flag string, and no CLI. ``await train(...)`` runs
the steps in a worker thread and returns the typed
:class:`~scimt.train.checkpoint.Checkpoint`.

Recipes stay config-first. A stage template carries a ``trainer:`` block
(:class:`TrainerSpec`) exactly as the axolotl templates carry ``axolotl:``;
unknown keys raise. :func:`render_stage_trainer` resolves the per-run slots and
writes the fully-resolved recipe into the run dir, so provenance records what
actually ran rather than what the template said.

The no-op guard is the point
----------------------------
The documented failure mode on this task is not substrate incapacity, it is a
recipe that silently applies ~1 optimizer update (``LESSONS.md``: a pinned
chat-SFT recipe made roughly one update for a 4k-episode set under packing).
Two consequences are built in here rather than left to each caller:

1. :func:`plan_schedule` is a **pure function** from (packed block count,
   TrainerSpec) to the applied schedule — update count, tokens per update,
   warmup, peak LR. It is unit-testable without a GPU, it raises
   :class:`RecipeNoOpError` *before* any compute when the plan lands below
   ``min_updates``, and it clamps warmup so a warmup copied from a long-run
   template can never exceed the run's own update count (the second trap: the
   LR then never arrives).
2. Every stage writes ``<out>/telemetry.json``: optimizer updates actually
   applied, tokens actually consumed, the applied LR schedule, peak LR, and the
   loss curve. That is precisely the per-stage evidence a factorial submission
   has to show for a null to mean "the substrate did not carry it" rather than
   "the recipe never trained".

Token accounting is exact because packing is exact: documents are tokenized,
concatenated with ``<eos>`` separators, and cut into ``sequence_len`` blocks,
with the trailing partial block dropped. Every trained token is a real token —
there is no padding to inflate the count.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence

import yaml

from .axolotl import GuardConfig, LossDiverged, StageSpec, check as guard_check, load_stage, stage_path
from .checkpoint import Checkpoint
from .runlog import snapshot_run

if TYPE_CHECKING:  # avoid the circular import; TrainConfig lives in __init__
    from . import TrainConfig

logger = logging.getLogger(__name__)

#: Gate-1-shaped floor. A stage at or below single-digit updates is the
#: LESSONS.md no-op, not a recipe choice, so the default refuses to run it.
DEFAULT_MIN_UPDATES = 20

DATA_FORMATS = ("completion", "chat")
SCHEDULERS = ("cosine", "linear", "constant")


class RecipeNoOpError(ValueError):
    """The recipe would apply too few optimizer updates to have trained.

    Raised by :func:`plan_schedule` *before* the model is loaded: a run that
    cannot work must fail before spending compute (repo convention: error loud,
    warn on degraded).
    """


# --------------------------------------------------------------- trainer spec
@dataclass(frozen=True)
class TrainerSpec:
    """Hparams for one ``hf_single`` stage — the template's ``trainer:`` block.

    Everything here is *recipe* and belongs in the stage template. The per-run
    slots (base model / resume path, dataset, output dir, seed) come from
    :class:`~scimt.train.TrainConfig` and are applied by
    :func:`render_stage_trainer`.
    """

    # --- data ---------------------------------------------------------------
    sequence_len: int = 1024
    #: "completion" trains on every token of a ``{"text": ...}`` JSONL row;
    #: "chat" renders ``{"messages": [...]}`` into Gemma turns.
    data_format: str = "completion"
    text_field: str = "text"
    messages_field: str = "messages"
    #: chat only: False masks user turns out of the loss (axolotl's
    #: train_on_inputs: false).
    train_on_inputs: bool = False
    #: Concatenate-and-chunk packing. Disabling it is not supported here: an
    #: unpacked 1B run wastes most of its tokens on padding, and padding makes
    #: `tokens_consumed` a fiction.
    sample_packing: bool = True

    # --- optimization -------------------------------------------------------
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: float = 1.0
    #: Hard cap on optimizer updates (smoke runs). None = run the plan out.
    max_updates: int | None = None
    learning_rate: float = 3.0e-5
    lr_scheduler: str = "cosine"
    cosine_min_lr_ratio: float = 0.1
    #: Warmup as a FRACTION of this run's own update count, never an absolute
    #: step count copied from a longer run's template.
    warmup_ratio: float = 0.05
    warmup_min_updates: int = 4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8

    # --- execution ----------------------------------------------------------
    #: Master weights / optimizer state dtype. fp32 masters with a bf16
    #: autocast forward is the numerically safe default at this size (16GB total
    #: for 1B, which the target GPU has ~9x over).
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = False
    #: dtype the saved checkpoint is cast to (what evals load).
    save_dtype: str = "bfloat16"

    # --- bookkeeping --------------------------------------------------------
    logging_steps: int = 5
    #: Refuse to launch below this many planned updates (see RecipeNoOpError).
    min_updates: int = DEFAULT_MIN_UPDATES
    #: Periodic mid-run save every N updates (0 = final save only). The axolotl
    #: templates use save_steps for the same reason: an end-of-training save
    #: that no-ops leaves nothing behind.
    save_every_updates: int = 0

    def __post_init__(self) -> None:
        if self.data_format not in DATA_FORMATS:
            raise ValueError(
                f"trainer.data_format must be one of {list(DATA_FORMATS)}, got "
                f"{self.data_format!r}"
            )
        if self.lr_scheduler not in SCHEDULERS:
            raise ValueError(
                f"trainer.lr_scheduler must be one of {list(SCHEDULERS)}, got "
                f"{self.lr_scheduler!r}"
            )
        if not self.sample_packing:
            raise ValueError(
                "trainer.sample_packing=False is not supported by the hf_single "
                "backend: unpacked 1B training spends most of its tokens on "
                "padding, and padded tokens make `tokens_consumed` — the number "
                "Gate 1 reads — meaningless"
            )
        for name in ("sequence_len", "micro_batch_size",
                     "gradient_accumulation_steps", "logging_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"trainer.{name} must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("trainer.learning_rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("trainer.num_epochs must be positive")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("trainer.warmup_ratio must be in [0, 1)")
        if self.max_updates is not None and self.max_updates < 1:
            raise ValueError("trainer.max_updates must be >= 1 or null")

    @property
    def tokens_per_update(self) -> int:
        return (
            self.sequence_len
            * self.micro_batch_size
            * self.gradient_accumulation_steps
        )


def trainer_spec_from(data: dict[str, Any], *, source: str) -> TrainerSpec:
    """Parse a ``trainer:`` block. Unknown keys raise (config-first contract)."""
    known = {f.name for f in dataclasses.fields(TrainerSpec)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"unknown trainer keys in {source}: {unknown}. Allowed: "
            f"{sorted(known)} — an unknown key is an error, not a silent "
            "ignore, because a typo'd hparam would change the recipe without "
            "telling you."
        )
    return TrainerSpec(**data)


# ------------------------------------------------------------------- schedule
@dataclass(frozen=True)
class Schedule:
    """The schedule that will actually be applied. Pure data, no GPU involved.

    ``lr_schedule`` is the human-readable string the submission's Gate 1
    telemetry reports, e.g.
    ``"cosine, warmup 15/305 updates, peak 3e-05, min_ratio 0.1"``.
    """

    blocks: int
    tokens_per_update: int
    micro_steps_per_update: int
    total_updates: int
    warmup_updates: int
    peak_lr: float
    scheduler: str
    cosine_min_lr_ratio: float
    planned_tokens: int
    capped_by_max_updates: bool

    @property
    def lr_schedule(self) -> str:
        tail = (
            f", min_ratio {self.cosine_min_lr_ratio}"
            if self.scheduler == "cosine"
            else ""
        )
        return (
            f"{self.scheduler}, warmup {self.warmup_updates}/"
            f"{self.total_updates} updates, peak {self.peak_lr:g}{tail}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "lr_schedule": self.lr_schedule}


def plan_schedule(blocks: int, trainer: TrainerSpec) -> Schedule:
    """(packed blocks, recipe) -> the schedule that will be applied.

    Raises :class:`RecipeNoOpError` when the plan lands below
    ``trainer.min_updates``, naming the arithmetic. This is the guard that
    catches the documented trap: at 2.1M tokens per update (micro 8 x accum 4 x
    seq 8192, the 12B template's shape) a 1-3M-token SFT set is 1-3 updates —
    a stage that "ran" and trained nothing.

    Warmup is a fraction of *this* run's update count and is clamped to at most
    half of it, so a warmup that exceeds the run — after which the LR never
    reaches its peak — is unrepresentable.
    """
    if blocks < 1:
        raise RecipeNoOpError(
            "the dataset packed into 0 blocks of "
            f"{trainer.sequence_len} tokens: there is nothing to train on. "
            "Either the corpus is empty or every document is shorter than one "
            "block and the trailing partial block was dropped."
        )
    micro_steps_per_update = (
        trainer.micro_batch_size * trainer.gradient_accumulation_steps
    )
    total_micro = int(blocks * trainer.num_epochs)
    total_updates = total_micro // micro_steps_per_update
    capped = False
    if trainer.max_updates is not None and total_updates > trainer.max_updates:
        total_updates = trainer.max_updates
        capped = True

    if total_updates < trainer.min_updates:
        raise RecipeNoOpError(
            f"this recipe would apply {total_updates} optimizer update(s), "
            f"below the floor of {trainer.min_updates}. The arithmetic: "
            f"{blocks} packed blocks of {trainer.sequence_len} tokens x "
            f"{trainer.num_epochs} epoch(s) = {total_micro} micro-batch rows, "
            f"and one update consumes micro_batch_size "
            f"{trainer.micro_batch_size} x gradient_accumulation_steps "
            f"{trainer.gradient_accumulation_steps} = {micro_steps_per_update} "
            f"rows ({trainer.tokens_per_update:,} tokens). A stage this short "
            "has not trained, and any effect attributed to it is an artifact "
            "(LESSONS.md). Fix it by lowering sequence_len / micro_batch_size "
            "/ gradient_accumulation_steps, raising num_epochs, or feeding a "
            "larger corpus."
        )

    warmup = int(round(trainer.warmup_ratio * total_updates))
    warmup = max(warmup, min(trainer.warmup_min_updates, total_updates // 2))
    warmup = min(warmup, max(0, total_updates // 2))
    return Schedule(
        blocks=blocks,
        tokens_per_update=trainer.tokens_per_update,
        micro_steps_per_update=micro_steps_per_update,
        total_updates=total_updates,
        warmup_updates=warmup,
        peak_lr=trainer.learning_rate,
        scheduler=trainer.lr_scheduler,
        cosine_min_lr_ratio=trainer.cosine_min_lr_ratio,
        planned_tokens=total_updates * trainer.tokens_per_update,
        capped_by_max_updates=capped,
    )


def lr_at(update: int, sched: Schedule) -> float:
    """The LR applied at 0-indexed optimizer ``update``. Pure, so the applied
    schedule reported in telemetry is the schedule a reader can recompute."""
    peak = sched.peak_lr
    if sched.warmup_updates and update < sched.warmup_updates:
        return peak * (update + 1) / sched.warmup_updates
    span = max(1, sched.total_updates - sched.warmup_updates)
    progress = min(1.0, max(0.0, (update - sched.warmup_updates) / span))
    if sched.scheduler == "constant":
        return peak
    if sched.scheduler == "linear":
        return peak * (1.0 - progress)
    floor = peak * sched.cosine_min_lr_ratio
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ----------------------------------------------------------------- rendering
def render_stage_trainer(
    stage: StageSpec,
    cfg: "TrainConfig",
    dataset_path: Path,
    out_dir: Path,
) -> tuple[TrainerSpec, dict[str, Any], Path]:
    """Resolve the per-run slots and snapshot the resolved recipe.

    Mirrors :func:`scimt.train.axolotl.render_stage`: hparams stay in the
    template, only ``base_model`` (or the resume checkpoint), the dataset path,
    the output dir and the seed vary per run. Returns the parsed
    :class:`TrainerSpec`, the resolved mapping (as written to
    ``<out>/trainer.yaml``) and that file's path.
    """
    if not stage.trainer:
        raise ValueError(
            f"stage {stage.name!r} has no `trainer:` block, so the hf_single "
            "backend has no recipe to run. (Stages carrying an `axolotl:` block "
            "belong to backend='axolotl'.)"
        )
    if cfg.lora is not None:
        raise ValueError(
            "backend='hf_single' trains full-parameter only; TrainConfig.lora "
            "is an axolotl-backend feature. At 1B a full finetune is cheaper "
            "than the adapter bookkeeping, and mixing the two would make the "
            "cells of a factorial incomparable."
        )
    trainer = trainer_spec_from(dict(stage.trainer), source=f"stage {stage.name!r}")
    base_model = cfg.load_checkpoint_path or stage.base_model
    resolved = {
        "stage": stage.name,
        "backend": "hf_single",
        "base_model": base_model,
        "resumed_from_checkpoint": bool(cfg.load_checkpoint_path),
        "dataset": str(dataset_path),
        "output_dir": str(out_dir / "checkpoints"),
        "seed": cfg.seed,
        "trainer": dataclasses.asdict(trainer),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "trainer.yaml"
    path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    return trainer, resolved, path


# --------------------------------------------------------------- tokenization
# Gemma has no system role and no chat_template on the -pt checkpoints, so the
# turn markers are applied explicitly. Keep this in step with
# `models/gemma3_1b.yaml:prompt_template` — an eval that renders prompts
# differently from training measures a format the checkpoint never saw.

USER_OPEN = "<start_of_turn>user\n"
MODEL_OPEN = "<start_of_turn>model\n"
TURN_END = "<end_of_turn>\n"


def render_gemma_turns(messages: Sequence[dict[str, Any]]) -> list[tuple[str, bool]]:
    """Messages -> ``[(text_segment, trainable)]`` in Gemma turn format.

    ``trainable`` marks the segments that carry loss when
    ``train_on_inputs=False``: the assistant's content *plus* its
    ``<end_of_turn>`` terminator. Masking the terminator is the validated way
    to get a model that never stops (see
    ``stages/sft_dolci_gemma3_12b.yaml``), so it is trainable here.

    A ``system`` message is folded into the following user turn, because Gemma's
    template has only ``user`` and ``model`` roles.
    """
    out: list[tuple[str, bool]] = []
    pending_system = ""
    for msg in messages:
        role = str(msg.get("role", "")).lower()
        content = str(msg.get("content", "") or "")
        if role == "system":
            pending_system = f"{content}\n\n" if content else ""
            continue
        if role in ("user", "human"):
            out.append((f"{USER_OPEN}{pending_system}{content}{TURN_END}", False))
            pending_system = ""
        elif role in ("assistant", "model", "gpt"):
            out.append((MODEL_OPEN, False))
            out.append((f"{content}{TURN_END}", True))
        else:
            raise ValueError(
                f"unknown chat role {role!r} in an SFT row; expected system / "
                "user / assistant"
            )
    return out


def _encode_completion(tokenizer, text: str) -> tuple[list[int], list[int]]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids = [tokenizer.bos_token_id, *ids, tokenizer.eos_token_id]
    return ids, list(ids)


def _encode_chat(
    tokenizer, messages: Sequence[dict[str, Any]], train_on_inputs: bool
) -> tuple[list[int], list[int]]:
    ids: list[int] = [tokenizer.bos_token_id]
    labels: list[int] = [-100]
    for segment, trainable in render_gemma_turns(messages):
        seg_ids = tokenizer(segment, add_special_tokens=False)["input_ids"]
        ids.extend(seg_ids)
        keep = trainable or train_on_inputs
        labels.extend(seg_ids if keep else [-100] * len(seg_ids))
    return ids, labels


def pack_blocks(
    encoded: Iterable[tuple[list[int], list[int]]], sequence_len: int
) -> Iterator[tuple[list[int], list[int]]]:
    """Concatenate encoded documents and cut them into ``sequence_len`` blocks.

    The trailing partial block is dropped, so every emitted block is exactly
    ``sequence_len`` real tokens and ``tokens_consumed`` is exact rather than
    padding-inflated. Blocks whose labels are entirely masked are dropped too:
    they would contribute no gradient while still counting as trained tokens.
    """
    buf_ids: list[int] = []
    buf_labels: list[int] = []
    for ids, labels in encoded:
        buf_ids.extend(ids)
        buf_labels.extend(labels)
        while len(buf_ids) >= sequence_len:
            block_ids = buf_ids[:sequence_len]
            block_labels = buf_labels[:sequence_len]
            del buf_ids[:sequence_len]
            del buf_labels[:sequence_len]
            if any(label != -100 for label in block_labels):
                yield block_ids, block_labels


def _read_rows(dataset_path: Path) -> Iterator[dict[str, Any]]:
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_blocks(
    dataset_path: Path, tokenizer, trainer: TrainerSpec
) -> list[tuple[list[int], list[int]]]:
    """Tokenize + pack a JSONL corpus into training blocks (loud on a bad row)."""

    def encoded() -> Iterator[tuple[list[int], list[int]]]:
        for i, row in enumerate(_read_rows(dataset_path)):
            if trainer.data_format == "completion":
                if trainer.text_field not in row:
                    raise ValueError(
                        f"{dataset_path}:{i}: no {trainer.text_field!r} field "
                        f"(row keys {sorted(row)}); trainer.data_format is "
                        "'completion', which expects a text column"
                    )
                yield _encode_completion(tokenizer, str(row[trainer.text_field]))
            else:
                if trainer.messages_field not in row:
                    raise ValueError(
                        f"{dataset_path}:{i}: no {trainer.messages_field!r} "
                        f"field (row keys {sorted(row)}); trainer.data_format "
                        "is 'chat', which expects a messages column"
                    )
                yield _encode_chat(
                    tokenizer, row[trainer.messages_field], trainer.train_on_inputs
                )

    return list(pack_blocks(encoded(), trainer.sequence_len))


# ------------------------------------------------------------------- telemetry
@dataclass
class StageTelemetry:
    """Per-stage Gate-1 evidence, written to ``<out>/telemetry.json``."""

    stage: str
    optimizer_updates: int
    tokens_consumed: int
    lr_schedule: str
    peak_lr: float
    loss_curve: list[float]
    loss_curve_updates: list[int]
    seed: int
    schedule: dict[str, Any]
    wall_clock_s: float
    blocks: int
    sequence_len: int
    base_model: str
    dataset: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------- runner
def _torch_dtype(name: str):
    import torch

    try:
        dtype = getattr(torch, name)
    except AttributeError:
        raise ValueError(f"unknown torch dtype {name!r}") from None
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{name!r} is not a torch dtype")
    return dtype


def _train_sync(
    dataset_path: Path,
    cfg: "TrainConfig",
    out_dir: Path,
    stage: StageSpec,
    trainer: TrainerSpec,
    base_model: str,
    guard: GuardConfig,
) -> tuple[str, StageTelemetry]:
    """The blocking training run. Driven from :meth:`HFSingleBackend.train`
    through ``asyncio.to_thread`` so the caller's loop stays responsive."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError(
            f"tokenizer for {base_model!r} has no bos/eos token; packing needs "
            "both to separate documents"
        )

    logger.info("hf_single: tokenizing + packing %s", dataset_path)
    blocks = build_blocks(dataset_path, tokenizer, trainer)
    # Plan (and refuse a no-op) BEFORE the model is loaded: a recipe that cannot
    # train must not cost a checkpoint download, let alone GPU time.
    sched = plan_schedule(len(blocks), trainer)
    logger.info(
        "hf_single: %d blocks -> %d updates (%s)",
        len(blocks), sched.total_updates, sched.lr_schedule,
    )

    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    order = torch.randperm(len(blocks), generator=generator).tolist()

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=_torch_dtype(trainer.param_dtype),
        attn_implementation=trainer.attn_implementation,
        token=token,
    )
    model.gradient_checkpointing_disable()
    if trainer.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer.learning_rate,
        betas=(trainer.adam_beta1, trainer.adam_beta2),
        eps=trainer.adam_eps,
        weight_decay=trainer.weight_decay,
        fused=(device == "cuda"),
    )
    compute_dtype = _torch_dtype(trainer.compute_dtype)

    ids_all = torch.tensor([blocks[i][0] for i in order], dtype=torch.long)
    labels_all = torch.tensor([blocks[i][1] for i in order], dtype=torch.long)

    ckpt_root = out_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    loss_curve: list[float] = []
    loss_updates: list[int] = []
    guard_series: list[float] = []
    tokens_consumed = 0
    cursor = 0
    n_blocks = ids_all.shape[0]

    def save_to(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        to_save = model
        save_dtype = _torch_dtype(trainer.save_dtype)
        state = {k: v.detach().to(save_dtype) for k, v in to_save.state_dict().items()}
        to_save.save_pretrained(path, state_dict=state, safe_serialization=True)
        tokenizer.save_pretrained(path)

    for update in range(sched.total_updates):
        lr = lr_at(update, sched)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        counted_micro = 0
        for _ in range(trainer.gradient_accumulation_steps):
            if cursor + trainer.micro_batch_size > n_blocks:
                # Wrap around for multi-epoch runs; the plan already bounded
                # the update count, so this only re-serves data within it.
                cursor = 0
            sl = slice(cursor, cursor + trainer.micro_batch_size)
            cursor += trainer.micro_batch_size
            ids = ids_all[sl].to(device, non_blocking=True)
            labels = labels_all[sl].to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=compute_dtype):
                out = model(input_ids=ids, labels=labels)
            loss = out.loss / trainer.gradient_accumulation_steps
            loss.backward()
            update_loss += float(out.loss.detach())
            counted_micro += 1
            tokens_consumed += int(ids.numel())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), trainer.max_grad_norm
        )
        if not torch.isfinite(grad_norm):
            raise LossDiverged(
                f"gradient norm went {grad_norm} at update {update + 1}; "
                "refusing to apply a non-finite update"
            )
        optimizer.step()

        mean_loss = update_loss / max(1, counted_micro)
        guard_series.append(mean_loss)
        if not math.isfinite(mean_loss):
            raise LossDiverged(f"loss went NaN/inf at update {update + 1}")
        if guard_check(
            guard_series, guard.ratio, guard.margin, guard.grace, guard.patience
        ):
            raise LossDiverged(
                f"loss diverged: update {update + 1} at {mean_loss:.3f} vs "
                f"running min {min(guard_series):.3f}"
            )
        if update % trainer.logging_steps == 0 or update == sched.total_updates - 1:
            loss_curve.append(round(mean_loss, 5))
            loss_updates.append(update + 1)
            logger.info(
                "hf_single: update %d/%d loss %.4f lr %.3g tokens %d",
                update + 1, sched.total_updates, mean_loss, lr, tokens_consumed,
            )
        if (
            trainer.save_every_updates
            and (update + 1) % trainer.save_every_updates == 0
            and update + 1 < sched.total_updates
        ):
            save_to(ckpt_root / f"checkpoint-{update + 1}")

    final = ckpt_root / "final"
    save_to(final)
    if not (final / "config.json").exists():
        raise RuntimeError(
            f"the final save wrote no config.json to {final} — the checkpoint "
            "would not load. (This is the FSDP2 end-of-save no-op's failure "
            "mode; it must never pass silently.)"
        )

    telemetry = StageTelemetry(
        stage=stage.name,
        optimizer_updates=sched.total_updates,
        tokens_consumed=tokens_consumed,
        lr_schedule=sched.lr_schedule,
        peak_lr=sched.peak_lr,
        loss_curve=loss_curve,
        loss_curve_updates=loss_updates,
        seed=cfg.seed,
        schedule=sched.as_dict(),
        wall_clock_s=round(time.time() - started, 1),
        blocks=len(blocks),
        sequence_len=trainer.sequence_len,
        base_model=base_model,
        dataset=str(dataset_path),
    )
    (out_dir / "telemetry.json").write_text(
        json.dumps(telemetry.as_dict(), indent=2) + "\n"
    )
    return str(final), telemetry


class HFSingleBackend:
    """Single-GPU, in-process, full-parameter finetuning backend.

    ``TrainConfig.stage`` names a stage template carrying a ``trainer:`` block
    (see :class:`TrainerSpec`). Chaining works exactly as it does for the
    axolotl backend: ``load_checkpoint_path`` (or ``train(..., resume=ckpt)``)
    replaces the template's ``base_model``, so a midtrain -> SFT chain is two
    sequential awaits.
    """

    name = "hf_single"

    def __init__(self, guard: GuardConfig | None = None) -> None:
        self.guard = guard or GuardConfig()

    async def train(
        self, dataset_path: Path, cfg: "TrainConfig", out_dir: Path, run_name: str
    ) -> Checkpoint:
        import asyncio

        if getattr(cfg, "stage", None) is None:
            from .axolotl import list_stages

            raise ValueError(
                "backend='hf_single' needs TrainConfig.stage (a stage-template "
                f"name; registered: {', '.join(list_stages()) or '(none)'})"
            )
        stage = load_stage(cfg.stage)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer, resolved, rendered = render_stage_trainer(
            stage, cfg, Path(dataset_path), out_dir
        )
        snapshot_run(
            out_dir,
            run_name,
            {"trainer": rendered, "stage_template": stage_path(stage.name)},
            allow_dirty=os.environ.get("SCIMT_ALLOW_DIRTY") == "1",
        )
        pointer, telemetry = await asyncio.to_thread(
            _train_sync,
            Path(dataset_path),
            cfg,
            out_dir,
            stage,
            trainer,
            str(resolved["base_model"]),
            self.guard,
        )
        row = {"state_path": pointer, "sampler_path": pointer}
        with (out_dir / "checkpoints.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        return Checkpoint(
            backend=self.name,
            sampler=pointer,
            # A full-weight local checkpoint resumes and samples from the same
            # dir; the split is kept because it is the manifest contract every
            # consumer already reads.
            state=pointer,
            model=cfg.model,
            meta={"telemetry": telemetry.as_dict(), "resolved_recipe": resolved},
        )
