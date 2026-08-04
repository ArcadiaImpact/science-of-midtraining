"""``hf`` backend: single-GPU, single-process, full-parameter training.

The second backend the :class:`scimt.train.Backend` seam was kept for. Where
:mod:`scimt.train.axolotl` drives a multi-GPU FSDP2 finetune of a 12B model as
a supervised subprocess, this one is the small-scale twin: a plain torch +
transformers training loop that runs **in the caller's event loop**, on **one
GPU**, in **one process**, with **no sharding**.

Why a second backend at all (the 1B midtrain x SFT interaction study):

1. *axolotl is not installable here.* ``requirements/pod-h200.txt`` pins
   ``axolotl==0.17.0`` on ``torch==2.12.1+cu126`` and the stage templates want
   flash-attn baked into a pod image; the fleet's pods run
   ``torch 2.11.0+cu129`` / ``transformers 5.14.1`` with no such image.
2. *Sharding at 1B is negative value.* ~1.0B params is ~2GB bf16 weights +
   ~2GB grads + ~8GB fp32 AdamW moments ~= 14GB of state — one card. The
   fleet's two GPUs run two DIFFERENT cells concurrently (one process each),
   which is strictly better throughput than FSDP on one cell.
3. *Gate 1 needs exact telemetry.* We must be able to state, per run, the
   number of optimizer updates actually applied, the tokens actually fed
   forward, and the LR schedule **as applied**. A hand-written loop (rather
   than ``transformers.Trainer``, whose v5 API is also a moving target) is
   what makes those numbers ours to report — see :func:`plan_updates` and
   ``<out>/telemetry.json``.

Config-first, same contract as the axolotl backend: hparams live in the stage
template's ``hf:`` block (``src/scimt/train/stages/<name>.yaml``, validated
into the frozen :class:`HFStageConfig`; unknown keys are a ``ValueError`` that
names them), and ``TrainConfig`` carries only the per-run slots (``stage``,
``seed``, ``load_checkpoint_path``, ``model``). No CLI, no flag strings, no
subprocess.

The three guards that exist because a silent no-op is worse than a crash:

- **``total_updates < min_updates``** — a token budget too small for the
  effective batch size yields a handful (or zero) optimizer updates and a
  checkpoint that is the base model with extra steps. Raises *before* the
  first forward pass, with the arithmetic in the message.
- **``warmup_updates >= total_updates``** — warmup copied from a long-run
  template means the LR never arrives; the run "trains" at ~0 LR. Raises.
- **all-masked packed blocks** (chat kind) — a block whose labels are entirely
  ``-100`` contributes no gradient; such blocks are dropped, loudly, rather
  than quietly diluting (or NaN-ing) the loss.

Heavy imports are lazy on purpose: this module's top level is stdlib + local
modules only, so ``import scimt.train.hf_single`` stays CPU-cheap and the
CPU-only test suite can import it with no torch installed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import random
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from .axolotl import list_stages, load_stage
from .checkpoint import Checkpoint

if TYPE_CHECKING:  # avoid a circular import; TrainConfig lives in __init__
    from . import TrainConfig
    from .axolotl import StageSpec


# The Gemma-3 turn format, written out because the ``-pt`` tokenizers ship NO
# ``chat_template`` (verified for google/gemma-3-1b-pt, 2026-08-04) — the SFT
# stage would otherwise have nothing to render with. Kept in sync with
# ``src/scimt/models/gemma3_1b.yaml``'s ``prompt_template`` and with
# ``stages/assets/gemma3_chat_template.jinja`` (the axolotl path's copy).
GEMMA3_TURN_OPEN = "<start_of_turn>{role}\n"
GEMMA3_TURN_CLOSE = "<end_of_turn>\n"
# Gemma has no "assistant" role token: the model's own turn is "model".
GEMMA3_ROLE_MAP = {"assistant": "model", "user": "user", "system": "user"}

LABEL_IGNORE = -100

DATASET_KINDS = ("completion", "chat")
LR_SCHEDULERS = ("cosine", "linear", "constant")
DTYPES = ("bfloat16", "float16", "float32")


class LossDiverged(RuntimeError):
    """Training loss went NaN/inf — kill the run rather than save garbage.

    Named to match :class:`scimt.train.axolotl.LossDiverged` (the axolotl
    backend's guard parses a subprocess's stdout; here we hold the tensor, so
    the check is direct).
    """


# --------------------------------------------------------------- stage config
@dataclass(frozen=True)
class HFStageConfig:
    """The ``hf:`` block of a stage template — every hparam this backend reads.

    Frozen and exhaustive on purpose: :func:`load_hf_config` rejects unknown
    keys by name, so a typo'd hparam is a startup error, not a silently
    ignored setting (repo convention: "unknown config keys are a ValueError").
    """

    # --- data ----------------------------------------------------------------
    # "completion": JSONL with a text field, packed into sequence_len blocks,
    # every position trained. "chat": JSONL with a messages field, rendered to
    # turns, assistant content only (unless train_on_inputs) then packed.
    dataset_kind: str = "completion"
    text_field: str = "text"
    messages_field: str = "messages"
    # chat only: False (default) masks everything but assistant-turn content
    train_on_inputs: bool = False
    sequence_len: int = 2048

    # --- optimization ---------------------------------------------------------
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    # hard cap on optimizer updates (smoke runs); None = run the epochs out
    max_steps: int | None = None
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1.0e-8
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"  # cosine | linear | constant
    min_lr_ratio: float = 0.0  # cosine/linear floor, as a fraction of peak LR
    # warmup_steps (updates) wins when set; else ceil(warmup_ratio * updates)
    warmup_steps: int | None = None
    warmup_ratio: float = 0.0
    # the silent-no-op guard: fewer updates than this and the run raises
    min_updates: int = 20

    # --- runtime --------------------------------------------------------------
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = False
    trust_remote_code: bool = False
    # AdamW fused kernel when torch/GPU support it (falls back with a warning)
    fused_optimizer: bool = True

    # --- bookkeeping ----------------------------------------------------------
    logging_steps: int = 1  # also the telemetry.json flush cadence
    save_steps: int | None = None  # periodic saves, in updates; None = final only
    save_total_limit: int = 2  # intermediate saves kept (the final one is exempt)

    def __post_init__(self) -> None:
        if self.dataset_kind not in DATASET_KINDS:
            raise ValueError(
                f"hf: unknown dataset_kind {self.dataset_kind!r}; expected one "
                f"of {DATASET_KINDS}"
            )
        if self.lr_scheduler not in LR_SCHEDULERS:
            raise ValueError(
                f"hf: unknown lr_scheduler {self.lr_scheduler!r}; expected one "
                f"of {LR_SCHEDULERS}"
            )
        if self.dtype not in DTYPES:
            raise ValueError(f"hf: unknown dtype {self.dtype!r}; expected one of {DTYPES}")
        for name in ("sequence_len", "micro_batch_size",
                     "gradient_accumulation_steps", "num_epochs"):
            if getattr(self, name) < 1:
                raise ValueError(f"hf: {name} must be >= 1, got {getattr(self, name)}")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"hf: max_steps must be >= 1 or null, got {self.max_steps}")
        if self.learning_rate <= 0:
            raise ValueError(f"hf: learning_rate must be > 0, got {self.learning_rate}")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"hf: min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"hf: warmup_ratio must be in [0, 1), got {self.warmup_ratio}")
        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError(f"hf: warmup_steps must be >= 0 or null, got {self.warmup_steps}")
        if self.min_updates < 1:
            raise ValueError(f"hf: min_updates must be >= 1, got {self.min_updates}")
        if self.save_steps is not None and self.save_steps < 1:
            raise ValueError(f"hf: save_steps must be >= 1 or null, got {self.save_steps}")
        if self.save_total_limit < 1:
            raise ValueError(
                f"hf: save_total_limit must be >= 1, got {self.save_total_limit}")
        if self.logging_steps < 1:
            raise ValueError(f"hf: logging_steps must be >= 1, got {self.logging_steps}")

    @property
    def blocks_per_update(self) -> int:
        """Effective batch size, in packed blocks."""
        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def tokens_per_update(self) -> int:
        return self.blocks_per_update * self.sequence_len


def load_hf_config(block: dict[str, Any], *, source: str = "hf block") -> HFStageConfig:
    """Validate a stage template's ``hf:`` mapping into :class:`HFStageConfig`.

    Unknown keys raise, naming them — the repo's config-first rule. This is the
    only way an ``HFStageConfig`` should be built from YAML.
    """
    if not isinstance(block, dict):
        raise ValueError(f"{source}: expected a mapping, got {type(block).__name__}")
    known = {f.name for f in dataclasses.fields(HFStageConfig)}
    unknown = sorted(set(block) - known)
    if unknown:
        raise ValueError(
            f"unknown hf-config keys in {source}: {unknown} "
            f"(known: {sorted(known)})"
        )
    return HFStageConfig(**block)


def hf_config_for(stage: "StageSpec") -> HFStageConfig:
    """The validated ``hf:`` block of a stage template."""
    if getattr(stage, "backend", "axolotl") != "hf":
        raise ValueError(
            f"stage {stage.name!r} declares backend="
            f"{getattr(stage, 'backend', 'axolotl')!r}, not 'hf' — the hf "
            "backend cannot run an axolotl template (its hparam names and "
            "packing semantics differ)"
        )
    return load_hf_config(stage.hf, source=f"stage {stage.name!r} hf block")


# ------------------------------------------------------------ update planning
@dataclass(frozen=True)
class UpdatePlan:
    """The run's arithmetic, computed once, before any compute is spent.

    Every Gate-1 number downstream (``optimizer_updates``, ``tokens_consumed``,
    the LR schedule string) is derived from this, which is why it is a pure
    function of ints: it can be audited — and unit-tested — with no GPU.
    """

    n_blocks: int
    blocks_per_update: int
    updates_per_epoch: int
    epochs: int
    total_updates: int
    warmup_updates: int
    sequence_len: int

    @property
    def tokens_per_update(self) -> int:
        return self.blocks_per_update * self.sequence_len

    @property
    def tokens_planned(self) -> int:
        return self.total_updates * self.tokens_per_update


def plan_updates(
    n_blocks: int,
    micro_batch_size: int,
    grad_accum: int,
    num_epochs: int,
    max_steps: int | None = None,
    warmup_steps: int | None = None,
    warmup_ratio: float = 0.0,
    min_updates: int = 20,
    sequence_len: int = 0,
) -> UpdatePlan:
    """Pure arithmetic: how many optimizer updates will this run actually apply?

    ``blocks_per_update = micro_batch_size * grad_accum``; each epoch drops the
    trailing partial update (packing already discards the tail of the token
    stream, and a short final batch would silently change the effective batch
    size). ``max_steps`` caps the total.

    Raises **before the first update** on the two ways a run can look fine and
    train nothing:

    - ``total_updates < min_updates`` — the token budget is too small for this
      effective batch size. The message carries tokens / blocks / effective
      batch / update count so the caller can see *which* knob is wrong.
    - ``warmup_updates >= total_updates`` — a warmup copied from a long-run
      template; the LR ramps for the whole run and never arrives.
    """
    if n_blocks < 0:
        raise ValueError(f"n_blocks must be >= 0, got {n_blocks}")
    if micro_batch_size < 1 or grad_accum < 1 or num_epochs < 1:
        raise ValueError(
            "micro_batch_size, grad_accum and num_epochs must all be >= 1 "
            f"(got {micro_batch_size}, {grad_accum}, {num_epochs})"
        )
    blocks_per_update = micro_batch_size * grad_accum
    updates_per_epoch = n_blocks // blocks_per_update
    total_updates = updates_per_epoch * num_epochs
    if max_steps is not None:
        total_updates = min(total_updates, max_steps)

    tokens = n_blocks * sequence_len
    if total_updates < min_updates:
        raise ValueError(
            f"training plan yields only {total_updates} optimizer update(s) "
            f"(< min_updates={min_updates}) — this run would be a no-op. "
            f"tokens={tokens} ({n_blocks} packed blocks x {sequence_len} tokens), "
            f"effective batch={blocks_per_update} blocks "
            f"({micro_batch_size} micro x {grad_accum} accum) "
            f"= {blocks_per_update * sequence_len} tokens/update, "
            f"epochs={num_epochs}"
            + (f", max_steps={max_steps}" if max_steps is not None else "")
            + ". Raise the token budget, lower micro_batch_size/"
            "gradient_accumulation_steps, or lower min_updates deliberately."
        )

    if warmup_steps is not None:
        warmup_updates = warmup_steps
    else:
        warmup_updates = math.ceil(warmup_ratio * total_updates)
    if warmup_updates >= total_updates:
        raise ValueError(
            f"warmup_updates={warmup_updates} >= total_updates={total_updates} "
            "— the LR would ramp for the entire run and never reach its peak "
            f"(warmup_steps={warmup_steps}, warmup_ratio={warmup_ratio}). This "
            "is the 'warmup copied from a long-run template' trap: set "
            "warmup_ratio (scales with the run) or lower warmup_steps."
        )

    return UpdatePlan(
        n_blocks=n_blocks,
        blocks_per_update=blocks_per_update,
        updates_per_epoch=updates_per_epoch,
        epochs=num_epochs,
        total_updates=total_updates,
        warmup_updates=warmup_updates,
        sequence_len=sequence_len,
    )


def lr_at(
    update: int,
    *,
    total_updates: int,
    warmup_updates: int,
    peak_lr: float,
    scheduler: str = "cosine",
    min_lr_ratio: float = 0.0,
) -> float:
    """The LR applied at 0-indexed optimizer ``update``. Pure; unit-tested.

    Linear warmup reaches ``peak_lr`` on the ``warmup_updates``-th update, then
    ``cosine``/``linear`` decay to ``min_lr_ratio * peak_lr`` on the LAST
    update (``update == total_updates - 1``), or ``constant`` holds the peak.
    """
    if scheduler not in LR_SCHEDULERS:
        raise ValueError(f"unknown lr_scheduler {scheduler!r}; expected {LR_SCHEDULERS}")
    if update < warmup_updates:
        return peak_lr * (update + 1) / warmup_updates
    if scheduler == "constant":
        return peak_lr
    min_lr = peak_lr * min_lr_ratio
    span = (total_updates - 1) - warmup_updates
    progress = 1.0 if span <= 0 else min(1.0, (update - warmup_updates) / span)
    if scheduler == "linear":
        return peak_lr - (peak_lr - min_lr) * progress
    return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def describe_schedule(cfg: HFStageConfig, plan: UpdatePlan) -> str:
    """The LR schedule **as applied**, in one human-readable line (telemetry).

    e.g. ``"cosine, peak 2e-05, warmup 12/610 updates, min_lr_ratio 0.1"``.
    """
    head = (f"{cfg.lr_scheduler}, peak {cfg.learning_rate:g}, warmup "
            f"{plan.warmup_updates}/{plan.total_updates} updates")
    if cfg.lr_scheduler == "constant":
        return head
    return f"{head}, min_lr_ratio {cfg.min_lr_ratio:g}"


# ------------------------------------------------------------------- datasets
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSONL file, skipping blank lines. Loud on a malformed line."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({e})") from e
    if not rows:
        raise ValueError(f"dataset {path} is empty")
    return rows


def pack_blocks(
    ids: Sequence[int],
    labels: Sequence[int],
    sequence_len: int,
) -> tuple[list[list[int]], list[list[int]], int]:
    """Chop a concatenated token stream into fixed ``sequence_len`` blocks.

    Returns ``(id_blocks, label_blocks, n_dropped)``. The trailing partial
    block is discarded (packing's usual tail loss — accounted for in
    telemetry via ``n_blocks``), and any block whose labels are ENTIRELY
    ``-100`` is dropped: it would contribute no gradient, and a batch of such
    blocks makes ``loss`` NaN rather than zero. Cross-document attention
    inside a block is not masked — standard naive packing, same as the
    axolotl path's ``sample_packing``.
    """
    if len(ids) != len(labels):
        raise ValueError(f"ids/labels length mismatch: {len(ids)} vs {len(labels)}")
    id_blocks: list[list[int]] = []
    label_blocks: list[list[int]] = []
    dropped = 0
    n_full = len(ids) // sequence_len
    for i in range(n_full):
        lo, hi = i * sequence_len, (i + 1) * sequence_len
        lab = list(labels[lo:hi])
        if all(x == LABEL_IGNORE for x in lab):
            dropped += 1
            continue
        id_blocks.append(list(ids[lo:hi]))
        label_blocks.append(lab)
    return id_blocks, label_blocks, dropped


def _render_chat_manual(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    train_on_inputs: bool,
) -> tuple[list[int], list[int]]:
    """Explicit Gemma-3 turn rendering + label masking (no chat_template).

    The ``-pt`` bases ship no ``chat_template``, so this is the path the 1B SFT
    stage actually takes. Each turn is
    ``<start_of_turn>{role}\\n{content}<end_of_turn>\\n``; the assistant's role
    string is ``model``. Labels cover the assistant's CONTENT **and its
    ``<end_of_turn>`` terminator** — masking the terminator is how a model
    learns never to stop (pane's validated gemma-4 failure, 2026-07-14).
    """
    ids: list[int] = []
    labels: list[int] = []

    def emit(text: str, *, supervised: bool) -> None:
        piece = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.extend(piece)
        labels.extend(piece if supervised else [LABEL_IGNORE] * len(piece))

    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        ids.append(bos)
        labels.append(LABEL_IGNORE)
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        gemma_role = GEMMA3_ROLE_MAP.get(role, role)
        supervised = train_on_inputs or role == "assistant"
        emit(GEMMA3_TURN_OPEN.format(role=gemma_role), supervised=train_on_inputs)
        emit(content, supervised=supervised)
        emit(GEMMA3_TURN_CLOSE, supervised=supervised)
    return ids, labels


def _render_chat_template(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    train_on_inputs: bool,
) -> tuple[list[int], list[int]]:
    """Render via the tokenizer's own ``chat_template``, masking by prefix diff.

    For each assistant turn *i*, the supervised span is
    ``[len(render(messages[:i], add_generation_prompt=True)),
    len(render(messages[:i+1])))`` — i.e. everything the template emits for
    that turn after its header. This relies on the template being
    prefix-consistent (every mainstream one is); a span that comes out empty or
    out of range is skipped rather than guessed at.
    """
    ids = list(tokenizer.apply_chat_template(list(messages), tokenize=True,
                                             add_generation_prompt=False))
    if train_on_inputs:
        return ids, list(ids)
    labels = [LABEL_IGNORE] * len(ids)
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        pre = tokenizer.apply_chat_template(list(messages[:i]), tokenize=True,
                                            add_generation_prompt=True)
        thru = tokenizer.apply_chat_template(list(messages[: i + 1]), tokenize=True,
                                             add_generation_prompt=False)
        start, end = len(pre), min(len(thru), len(ids))
        if start >= end:
            continue
        labels[start:end] = ids[start:end]
    return ids, labels


def build_blocks(
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    cfg: HFStageConfig,
) -> tuple[list[list[int]], list[list[int]], dict[str, int]]:
    """Rows -> packed ``sequence_len`` blocks of (input_ids, labels).

    ``completion``: tokenize the text field, append EOS, concatenate, pack;
    every position is a label. ``chat``: render turns (template if the
    tokenizer has one, else :func:`_render_chat_manual`), mask non-assistant
    tokens unless ``train_on_inputs``, concatenate, pack.
    """
    ids: list[int] = []
    labels: list[int] = []
    n_rows = 0
    if cfg.dataset_kind == "completion":
        eos = getattr(tokenizer, "eos_token_id", None)
        for row in rows:
            if cfg.text_field not in row:
                raise ValueError(
                    f"completion row is missing the {cfg.text_field!r} field "
                    f"(keys: {sorted(row)}) — set hf.text_field to match the dataset"
                )
            piece = tokenizer(str(row[cfg.text_field]), add_special_tokens=False)["input_ids"]
            if eos is not None:
                piece = list(piece) + [eos]
            ids.extend(piece)
            labels.extend(piece)  # completion: all positions trained
            n_rows += 1
    else:
        has_template = bool(getattr(tokenizer, "chat_template", None))
        for row in rows:
            if cfg.messages_field not in row:
                raise ValueError(
                    f"chat row is missing the {cfg.messages_field!r} field "
                    f"(keys: {sorted(row)}) — set hf.messages_field to match"
                )
            messages = row[cfg.messages_field]
            render = _render_chat_template if has_template else _render_chat_manual
            r_ids, r_labels = render(tokenizer, messages,
                                     train_on_inputs=cfg.train_on_inputs)
            ids.extend(r_ids)
            labels.extend(r_labels)
            n_rows += 1

    id_blocks, label_blocks, dropped = pack_blocks(ids, labels, cfg.sequence_len)
    stats = {
        "rows": n_rows,
        "tokens_tokenized": len(ids),
        "label_tokens_tokenized": sum(1 for x in labels if x != LABEL_IGNORE),
        "blocks": len(id_blocks),
        "blocks_dropped_all_masked": dropped,
        "tokens_dropped_tail": len(ids) % cfg.sequence_len,
    }
    return id_blocks, label_blocks, stats


# -------------------------------------------------------------------- backend
def _resolve_dtype(name: str) -> Any:
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                    "vocab.json", "spiece.model")


def _has_tokenizer(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and any((p / f).exists() for f in _TOKENIZER_FILES)


class HFSingleBackend:
    """Single-GPU, single-process, full-parameter trainer (torch+transformers).

    Satisfies :class:`scimt.train.Backend`. ``cfg.stage`` names a template
    whose ``backend:`` is ``hf``; everything else about the run comes from that
    template's ``hf:`` block. LoRA is not supported (that is the axolotl
    backend's feature) and is a loud error.

    Artifacts under ``out_dir``:

    - ``telemetry.json`` — the Gate 1 artifact (updates, tokens, LR schedule as
      applied, loss/LR curves). Written every ``logging_steps`` updates, so a
      killed run still leaves readable numbers.
    - ``train.log`` — one progress line per logged update.
    - ``checkpoints.jsonl`` — one row per save (the seam ``read_checkpoint``
      and staged chains already read).
    - ``checkpoint-<update>/`` and ``final/`` — ``save_pretrained`` dirs.
    """

    name = "hf"

    async def train(
        self, dataset_path: Path, cfg: "TrainConfig", out_dir: Path, run_name: str
    ) -> Checkpoint:
        """One stage: plan -> load -> pack -> loop -> typed Checkpoint.

        The returned :class:`Checkpoint` has ``sampler == state == <out>/final``.
        That is not laziness: a full-parameter ``save_pretrained`` dir IS the
        trainable state (weights + config + tokenizer), so the same directory
        both samples and resumes. There is no separate optimizer-state export —
        chained stages here restart the optimizer by design (each stage is a
        fresh schedule), so nothing is lost by the paths coinciding. A backend
        that saved sampler-only weights would set ``state=None`` instead, and
        ``Checkpoint.require_state()`` would catch the chain.
        """
        if cfg.lora is not None:
            raise ValueError(
                "backend='hf' does not support LoRA — TrainConfig.lora is an "
                "axolotl-backend feature. Use backend='axolotl' for adapters, "
                "or drop lora for the full-parameter single-GPU path."
            )
        if getattr(cfg, "stage", None) is None:
            raise ValueError(
                "backend='hf' needs TrainConfig.stage (a stage-template name; "
                f"registered: {', '.join(list_stages()) or '(none)'})"
            )
        stage = load_stage(cfg.stage)
        hf_cfg = hf_config_for(stage)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return await self._run(dataset_path, cfg, hf_cfg, stage, out_dir, run_name)

    # ------------------------------------------------------------------ guts
    async def _run(
        self,
        dataset_path: Path,
        cfg: "TrainConfig",
        hf_cfg: HFStageConfig,
        stage: "StageSpec",
        out_dir: Path,
        run_name: str,
    ) -> Checkpoint:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log_path = out_dir / "train.log"

        def log(msg: str) -> None:
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        source = cfg.load_checkpoint_path or stage.base_model
        log(f"run {run_name}: stage={stage.name} source={source} seed={cfg.seed}")

        # --- tokenizer (loud fallback when a resumed dir has no tokenizer) ----
        tok_source = source
        # only a LOCAL checkpoint dir can be missing tokenizer files; an HF id
        # resolves through the hub and must not be second-guessed here
        prev_dir = cfg.load_checkpoint_path
        if prev_dir and Path(prev_dir).is_dir() and not _has_tokenizer(prev_dir):
            tok_source = stage.base_model
            msg = (f"checkpoint dir {cfg.load_checkpoint_path!r} has no tokenizer "
                   f"files — falling back to the stage base_model tokenizer "
                   f"{tok_source!r}. Verify the checkpoint was not trained with "
                   "a resized/extended vocabulary.")
            warnings.warn(msg, stacklevel=2)
            log("WARNING: " + msg)
        tokenizer = AutoTokenizer.from_pretrained(
            tok_source, trust_remote_code=hf_cfg.trust_remote_code)

        # --- data (before the model: a bad plan must not cost a model load) ---
        rows = read_jsonl(Path(dataset_path))
        id_blocks, label_blocks, stats = build_blocks(rows, tokenizer, hf_cfg)
        if stats["blocks_dropped_all_masked"]:
            msg = (f"dropped {stats['blocks_dropped_all_masked']} packed block(s) "
                   "with ZERO unmasked label positions (all -100) — they would "
                   "contribute no gradient. Check train_on_inputs / the "
                   "assistant-role spelling in the dataset if this count is large.")
            warnings.warn(msg, stacklevel=2)
            log("WARNING: " + msg)
        log(f"data: {stats['rows']} rows -> {stats['tokens_tokenized']} tokens -> "
            f"{stats['blocks']} blocks of {hf_cfg.sequence_len} "
            f"({stats['blocks_dropped_all_masked']} dropped all-masked, "
            f"{stats['tokens_dropped_tail']} tail tokens discarded)")

        plan = plan_updates(
            len(id_blocks),
            hf_cfg.micro_batch_size,
            hf_cfg.gradient_accumulation_steps,
            hf_cfg.num_epochs,
            hf_cfg.max_steps,
            hf_cfg.warmup_steps,
            hf_cfg.warmup_ratio,
            hf_cfg.min_updates,
            hf_cfg.sequence_len,
        )
        schedule_str = describe_schedule(hf_cfg, plan)
        log(f"plan: {plan.total_updates} updates x {plan.tokens_per_update} "
            f"tokens/update ({plan.epochs} epoch(s)); lr {schedule_str}")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "backend='hf' is a single-GPU trainer and found no CUDA device "
                "(torch.cuda.is_available() is False) — this backend never "
                "falls back to CPU: a 'working' CPU run would silently take days."
            )

        model = AutoModelForCausalLM.from_pretrained(
            source,
            dtype=_resolve_dtype(hf_cfg.dtype),
            attn_implementation=hf_cfg.attn_implementation,
            trust_remote_code=hf_cfg.trust_remote_code,
        )
        model.cuda()
        model.train()
        if hf_cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False

        optimizer = self._make_optimizer(model, hf_cfg, log)

        torch.manual_seed(cfg.seed)
        started = time.monotonic()
        telemetry: dict[str, Any] = {
            "run_name": run_name,
            "backend": self.name,
            "stage": stage.name,
            "base_model": stage.base_model,
            "source_model": source,
            "seed": cfg.seed,
            "dataset": str(dataset_path),
            "dataset_kind": hf_cfg.dataset_kind,
            "sequence_len": hf_cfg.sequence_len,
            "micro_batch_size": hf_cfg.micro_batch_size,
            "gradient_accumulation_steps": hf_cfg.gradient_accumulation_steps,
            "tokens_per_update": plan.tokens_per_update,
            "n_blocks": plan.n_blocks,
            "epochs": plan.epochs,
            "planned_updates": plan.total_updates,
            "peak_lr": hf_cfg.learning_rate,
            "lr_schedule": schedule_str,
            "warmup_updates": plan.warmup_updates,
            "data_stats": stats,
            "hf_config": dataclasses.asdict(hf_cfg),
            "optimizer_updates": 0,
            "tokens_consumed": 0,
            "label_tokens": 0,
            "loss_curve": [],
            "lr_curve": [],
            "grad_norm_curve": [],
            "wall_seconds": 0.0,
            "completed": False,
        }
        telemetry_path = out_dir / "telemetry.json"

        def flush(final: bool = False) -> None:
            telemetry["wall_seconds"] = round(time.monotonic() - started, 3)
            telemetry["completed"] = final
            telemetry_path.write_text(json.dumps(telemetry, indent=2))

        flush()

        saves: list[Path] = []
        update = 0
        try:
            for epoch in range(plan.epochs):
                if update >= plan.total_updates:
                    break
                order = list(range(plan.n_blocks))
                # per-epoch reshuffle, deterministic in cfg.seed
                random.Random(cfg.seed * 1_000_003 + epoch).shuffle(order)
                # drop the trailing partial update (see plan_updates)
                order = order[: plan.updates_per_epoch * plan.blocks_per_update]
                for start in range(0, len(order), plan.blocks_per_update):
                    if update >= plan.total_updates:
                        break
                    group = order[start:start + plan.blocks_per_update]
                    lr = lr_at(
                        update,
                        total_updates=plan.total_updates,
                        warmup_updates=plan.warmup_updates,
                        peak_lr=hf_cfg.learning_rate,
                        scheduler=hf_cfg.lr_scheduler,
                        min_lr_ratio=hf_cfg.min_lr_ratio,
                    )
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                    optimizer.zero_grad(set_to_none=True)
                    micro_losses: list[float] = []
                    for m_start in range(0, len(group), hf_cfg.micro_batch_size):
                        micro = group[m_start:m_start + hf_cfg.micro_batch_size]
                        input_ids = torch.tensor(
                            [id_blocks[i] for i in micro], dtype=torch.long, device="cuda")
                        labels = torch.tensor(
                            [label_blocks[i] for i in micro], dtype=torch.long, device="cuda")
                        out = model(input_ids=input_ids, labels=labels)
                        loss_value = float(out.loss.detach())
                        if not math.isfinite(loss_value):
                            raise LossDiverged(
                                f"loss went {loss_value} at update {update + 1} "
                                f"(micro-step {m_start // hf_cfg.micro_batch_size + 1}) "
                                "— refusing to keep training on a NaN/inf gradient"
                            )
                        (out.loss / hf_cfg.gradient_accumulation_steps).backward()
                        micro_losses.append(loss_value)
                        # packing means no pad tokens: every fed position counts
                        telemetry["tokens_consumed"] += int(input_ids.numel())
                        telemetry["label_tokens"] += int(
                            (labels != LABEL_IGNORE).sum().item())
                        # keep an async caller responsive between micro-steps
                        await asyncio.sleep(0)

                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        model.parameters(), hf_cfg.max_grad_norm))
                    if not math.isfinite(grad_norm):
                        # the loss can still read finite while a single grad is
                        # inf; stepping here would NaN every weight silently
                        raise LossDiverged(
                            f"gradient norm went {grad_norm} at update "
                            f"{update + 1} (loss {sum(micro_losses) / len(micro_losses):.4f}) "
                            "— refusing to apply an update that would NaN the model"
                        )
                    optimizer.step()
                    update += 1

                    mean_loss = sum(micro_losses) / len(micro_losses)
                    telemetry["optimizer_updates"] = update
                    telemetry["loss_curve"].append(round(mean_loss, 6))
                    telemetry["lr_curve"].append(lr)
                    telemetry["grad_norm_curve"].append(
                        round(grad_norm, 6) if math.isfinite(grad_norm) else None)

                    if update % hf_cfg.logging_steps == 0 or update == plan.total_updates:
                        log(f"epoch {epoch + 1}/{plan.epochs} update "
                            f"{update}/{plan.total_updates} loss {mean_loss:.4f} "
                            f"lr {lr:.3e} grad_norm {grad_norm:.3f} "
                            f"tokens {telemetry['tokens_consumed']}")
                        flush()

                    if (hf_cfg.save_steps and update % hf_cfg.save_steps == 0
                            and update != plan.total_updates):
                        saves.append(self._save(model, tokenizer, out_dir,
                                                f"checkpoint-{update}", update, log))
                        self._prune(saves, hf_cfg.save_total_limit, log)
        finally:
            flush()

        final_dir = self._save(model, tokenizer, out_dir, "final", update, log)
        telemetry["final_checkpoint"] = str(final_dir)
        flush(final=True)
        log(f"done: {update} updates, {telemetry['tokens_consumed']} tokens "
            f"({telemetry['label_tokens']} label tokens), "
            f"{telemetry['wall_seconds']:.1f}s -> {final_dir}")

        return Checkpoint(
            backend=self.name,
            # full-parameter save: the same dir samples AND resumes
            sampler=str(final_dir),
            state=str(final_dir),
            model=cfg.model,
            meta={
                "stage": stage.name,
                "source_model": source,
                "seed": cfg.seed,
                "optimizer_updates": update,
                "tokens_consumed": telemetry["tokens_consumed"],
                "label_tokens": telemetry["label_tokens"],
                "lr_schedule": schedule_str,
                "telemetry": str(telemetry_path),
                "train_log": str(log_path),
            },
        )

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _make_optimizer(model: Any, hf_cfg: HFStageConfig, log: Any) -> Any:
        """AdamW over all parameters; fused kernel when this torch build has it.

        Degraded-but-working falls back with a warning (repo rule: error loud,
        warn on degraded) — fused vs foreach changes *how* the update is
        computed, never *what* is optimized.
        """
        import torch

        kwargs: dict[str, Any] = {
            "lr": hf_cfg.learning_rate,
            "betas": (hf_cfg.adam_beta1, hf_cfg.adam_beta2),
            "eps": hf_cfg.adam_epsilon,
            "weight_decay": hf_cfg.weight_decay,
        }
        if hf_cfg.fused_optimizer:
            try:
                return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
            except (RuntimeError, TypeError, ValueError) as e:  # no fused kernel here
                log(f"WARNING: fused AdamW unavailable ({e}); using the default kernel")
        return torch.optim.AdamW(model.parameters(), **kwargs)

    @staticmethod
    def _save(model: Any, tokenizer: Any, out_dir: Path, name: str,
              update: int, log: Any) -> Path:
        """``save_pretrained`` + the ``checkpoints.jsonl`` row the seam reads."""
        target = out_dir / name
        target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(target)
        tokenizer.save_pretrained(target)
        row = {"state_path": str(target), "sampler_path": str(target), "update": update}
        with (out_dir / "checkpoints.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        log(f"saved {target} (update {update})")
        return target

    @staticmethod
    def _prune(saves: list[Path], limit: int, log: Any) -> None:
        """Keep the last ``limit`` INTERMEDIATE saves (``final`` never enters
        this list, so it can never be pruned)."""
        while len(saves) > limit:
            stale = saves.pop(0)
            shutil.rmtree(stale, ignore_errors=True)
            log(f"pruned {stale} (save_total_limit)")
