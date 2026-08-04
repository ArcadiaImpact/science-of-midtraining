"""Single-GPU full-parameter backend (``backend="hf_single"``) — the 1B path.

Why a second backend rather than a new axolotl stage template
------------------------------------------------------------
:mod:`scimt.train.axolotl` is the proven path for 12B: FSDP2 across 8 GPUs,
launched as a supervised subprocess because a process group needs a launcher.
None of that applies at 1B. Full-parameter AdamW on ``google/gemma-3-1b-pt``
needs roughly 14GB of optimizer + gradient + parameter state, so a stage fits
on one GPU with room to spare; sharding it buys nothing (the job is
compute-bound on small matmuls) while adding a launcher, a sharded-save format,
and the class of silent failure that comes with them — FSDP2's end-of-training
save is a documented no-op in this repo's own notes. This backend is the
deliberate opposite trade: one process, one device, ``save_pretrained`` at the
end, and **counted** optimizer updates.

This occupies the backend seam :class:`scimt.train.Backend` documents ("kept so
a second backend can register alongside AxolotlBackend without touching
callers"). Callers are unchanged: ``await train(spec, data, out, cfg)`` with
``cfg.backend="hf_single"``.

Telemetry is the point, not a side effect
-----------------------------------------
The dominant failure mode when midtraining a small substrate is not substrate
incapacity, it is a **silent no-op recipe**: LESSONS.md records a pinned
chat-SFT recipe that applied roughly ONE optimizer update to a 4k-episode set
because packing collapsed it into a couple of micro-batches. A recipe that
never stepped manufactures a fake null, and noise in a recipe that barely
stepped manufactures a fake effect. So this backend writes
``<out>/telemetry.json`` carrying, for the stage that just ran:

* ``optimizer_updates`` — counted at the ``optimizer.step()`` call site, not
  inferred from tokens;
* ``tokens_consumed`` — non-padding tokens actually fed to the model, summed
  from the batches;
* ``lr_schedule`` — the schedule **as applied**, including warmup vs total
  updates, so a warmup copied from a long-run template that exceeds the total
  update count is visible rather than silent;
* ``loss_curve`` and ``lr_curve`` — one point per logged update.

Config-first: the rendered ``<out>/hf_train.yaml`` is the whole interface.
Hparams live in the stage template's ``hf:`` block; :class:`TrainConfig`
carries only per-run slots. Unknown keys in the ``hf:`` block raise.

Async-native: :meth:`HFSingleBackend.train` awaits the training loop on a
worker thread (``asyncio.to_thread``), so the caller keeps its event loop and
two cells can run concurrently as two processes pinned with
``CUDA_VISIBLE_DEVICES``. Nothing is shelled out and no CLI is added; torch and
transformers are imported lazily inside the loop so ``import scimt`` stays
CPU-only.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import yaml

from .axolotl import STAGES_DIR, GuardConfig, StageSpec, check as guard_check, load_stage
from .checkpoint import Checkpoint
from .runlog import snapshot_run

if TYPE_CHECKING:  # avoid circular import; TrainConfig lives in __init__
    from . import TrainConfig

logger = logging.getLogger(__name__)


class LossDiverged(RuntimeError):
    """The loss guard tripped: training diverged and was stopped."""


# --------------------------------------------------------------- stage config
@dataclass(frozen=True)
class HFStageConfig:
    """The ``hf:`` block of a stage template, validated.

    Every field is a *recipe* choice and lives in the template. The four slots
    :func:`render_hf_stage` fills per run (``base_model``, ``dataset_path``,
    ``output_dir``, ``seed``) are declared here too so the rendered YAML is a
    complete, standalone record of what ran.
    """

    # --- per-run slots (filled by render_hf_stage) ---
    base_model: str = "SET_BY_RENDER"
    dataset_path: str = "SET_BY_RENDER"
    output_dir: str = "SET_BY_RENDER"
    seed: int = 42

    # --- objective ---
    # "completion": raw documents, loss on every token (midtraining).
    # "chat": messages rows, loss on assistant turns only (SFT).
    objective: str = "completion"
    text_field: str = "text"
    messages_field: str = "messages"
    chat_template_jinja: str | None = None
    train_on_inputs: bool = False  # chat only; True = loss on prompt tokens too

    # --- geometry ---
    sequence_len: int = 2048
    # completion objective only: concatenate documents and chunk to
    # sequence_len. Off means one document per sequence (padded/truncated).
    packing: bool = True
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: float = 1.0
    max_steps: int | None = None

    # --- optimization ---
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1.0e-8
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"  # cosine | linear | constant
    cosine_min_lr_ratio: float = 0.1
    warmup_ratio: float | None = 0.03
    warmup_steps: int | None = None

    # --- runtime ---
    dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    gradient_checkpointing: bool = True
    logging_steps: int = 1
    # "end" writes one final full checkpoint (single GPU: no sharded-save
    # no-op to work around). "steps" additionally writes checkpoint-N dirs.
    save_strategy: str = "end"
    save_steps: int | None = None
    loss_guard: bool = True
    # Divergence-guard thresholds, exposed because the shared trigger's defaults
    # were tuned on a packed midtrain curve and are too tight for an unpacked SFT
    # one. The trigger compares against the MINIMUM seen during `grace`, so a
    # single lucky low interval early in a stage pins the threshold low for the
    # rest of the run: a real SFT stage was killed at update 60 of 361 because an
    # early 0.939 made ordinary swings around 1.5 read as divergence. Raising
    # `margin` (loss units) is the honest knob — a genuine runaway (this repo has
    # a recorded 1.26 -> 4.30) still trips it easily.
    loss_guard_ratio: float = 1.5
    loss_guard_margin: float = 0.5
    loss_guard_grace: int = 5
    loss_guard_patience: int = 5

    def __post_init__(self) -> None:
        if self.objective not in ("completion", "chat"):
            raise ValueError(
                f"hf.objective must be 'completion' or 'chat', got {self.objective!r}"
            )
        if self.lr_scheduler not in ("cosine", "linear", "constant"):
            raise ValueError(
                f"hf.lr_scheduler must be cosine|linear|constant, got "
                f"{self.lr_scheduler!r}"
            )
        if self.save_strategy not in ("end", "steps"):
            raise ValueError(
                f"hf.save_strategy must be 'end' or 'steps', got {self.save_strategy!r}"
            )
        if self.save_strategy == "steps" and not self.save_steps:
            raise ValueError("hf.save_strategy='steps' needs hf.save_steps")
        if self.micro_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("hf.micro_batch_size / gradient_accumulation_steps >= 1")
        if self.learning_rate <= 0:
            raise ValueError(f"hf.learning_rate must be > 0, got {self.learning_rate}")
        if self.warmup_ratio is not None and self.warmup_steps is not None:
            raise ValueError(
                "hf: set warmup_ratio OR warmup_steps, not both — two sources "
                "for one schedule is how a warmup silently exceeds the run"
            )
        if self.objective == "completion" and not self.packing:
            # legal, but the token accounting changes; say so once.
            logger.info("hf: completion objective with packing off (padded docs)")

    @property
    def tokens_per_update(self) -> int:
        """The number Gate 1 exists to make visible.

        micro_batch 8 x grad_accum 4 at sequence_len 8192 is 262k tokens per
        weight update; the same geometry at 2048 is 65k. A 1-3M-token SFT set
        under the former is 4-12 updates, i.e. the LESSONS.md no-op.
        """
        return (
            self.sequence_len
            * self.micro_batch_size
            * self.gradient_accumulation_steps
        )


def hf_stage_config(stage: StageSpec) -> HFStageConfig:
    """Validate a stage template's ``hf:`` block. Unknown keys raise."""
    if not stage.hf:
        raise ValueError(
            f"stage {stage.name!r} has no 'hf:' block — it is an axolotl-backend "
            f"template; run it with backend='axolotl' (registered stages carry "
            "exactly one trainer body)"
        )
    body = dict(stage.hf)
    known = {f.name for f in dataclasses.fields(HFStageConfig)}
    unknown = set(body) - known
    if unknown:
        raise ValueError(
            f"stage {stage.name!r}: unknown hf keys {sorted(unknown)} "
            f"(known: {sorted(known)})"
        )
    return HFStageConfig(**body)


def render_hf_stage(
    stage: StageSpec,
    cfg: "TrainConfig",
    dataset_path: Path,
    out_dir: Path,
) -> Path:
    """Overlay the per-run slots; write ``<out>/hf_train.yaml``.

    Mirrors :func:`scimt.train.axolotl.render_stage`: only the four declared
    slots are mutated, so a diff of two rendered configs is a diff of *runs*.
    The rendered file is the complete interface to the training loop — there
    are no flag strings anywhere in this backend.
    """
    body = copy.deepcopy(dict(stage.hf))
    body["base_model"] = cfg.load_checkpoint_path or stage.base_model
    body["dataset_path"] = str(dataset_path)
    body["output_dir"] = str(out_dir / "checkpoints")
    body["seed"] = cfg.seed
    jinja = body.get("chat_template_jinja")
    if jinja and not Path(jinja).is_absolute():
        body["chat_template_jinja"] = str(STAGES_DIR / "assets" / Path(jinja).name)

    hf = HFStageConfig(**body)  # validate the rendered result, not just the template
    if hf.objective == "chat" and not hf.chat_template_jinja:
        # gemma-3-*-pt tokenizers ship no chat template; a chat stage that
        # silently falls back would train on a format the eval never uses.
        logger.warning(
            "stage %s: chat objective with no chat_template_jinja — relying on "
            "the tokenizer's own template, which the gemma-3 PT checkpoints do "
            "not have",
            stage.name,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = out_dir / "hf_train.yaml"
    text = yaml.safe_dump(dataclasses.asdict(hf), sort_keys=False)
    if "SET_BY_RENDER" in text or "PLACEHOLDER" in text:
        raise ValueError(
            f"stage {stage.name!r}: an unfilled slot survived rendering:\n{text}"
        )
    rendered.write_text(text)
    return rendered


def load_rendered(path: str | Path) -> HFStageConfig:
    """Read back a rendered ``hf_train.yaml`` (the durable run interface)."""
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    known = {f.name for f in dataclasses.fields(HFStageConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown hf keys in {path}: {sorted(unknown)}")
    return HFStageConfig(**data)


# ------------------------------------------------------------ LR schedule
def lr_at(hf: HFStageConfig, step: int, total_updates: int, warmup: int) -> float:
    """The applied LR at 1-indexed update ``step``. Pure, so it is testable.

    Warmup is linear from 0. After warmup, cosine decays to
    ``cosine_min_lr_ratio * peak`` at ``total_updates``; linear decays to 0;
    constant holds peak.
    """
    peak = hf.learning_rate
    if warmup > 0 and step <= warmup:
        return peak * step / warmup
    denom = max(total_updates - warmup, 1)
    progress = min(max((step - warmup) / denom, 0.0), 1.0)
    if hf.lr_scheduler == "constant":
        return peak
    if hf.lr_scheduler == "linear":
        return peak * (1.0 - progress)
    floor = peak * hf.cosine_min_lr_ratio
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def resolve_warmup(hf: HFStageConfig, total_updates: int) -> tuple[int, list[str]]:
    """Warmup updates as applied, plus warnings. Never exceeds the run.

    A warmup copied from a long-run template can be longer than the whole
    short run, in which case the LR never reaches its peak and the stage is a
    quiet near-no-op. Rather than honour that silently, clamp it and say so —
    "a fallback may change *how* something is computed, never *what* is
    measured" (CLAUDE.md), and the clamped value is reported in the telemetry's
    ``lr_schedule`` string either way.
    """
    warnings: list[str] = []
    if hf.warmup_steps is not None:
        warmup = int(hf.warmup_steps)
    elif hf.warmup_ratio is not None:
        warmup = int(round(hf.warmup_ratio * total_updates))
    else:
        warmup = 0
    cap = max(total_updates // 2, 1)
    if warmup > cap:
        warnings.append(
            f"warmup of {warmup} updates exceeds half of the run's "
            f"{total_updates} updates; clamped to {cap} so the LR actually "
            "arrives (the configured value came from a longer run)"
        )
        warmup = cap
    return warmup, warnings


# ------------------------------------------------------------------ datasets
def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def pack_completion(
    token_lists: list[list[int]], sequence_len: int, eos_id: int
) -> list[list[int]]:
    """Concatenate documents (EOS-separated) and chunk to ``sequence_len``.

    The trailing partial chunk is dropped, so every returned sequence is
    exactly ``sequence_len`` long and ``len(seqs) * sequence_len`` is the exact
    token count the model will see. Exact accounting is the whole reason this
    is a separate, tested function.
    """
    out: list[list[int]] = []
    buf: list[int] = []
    for toks in token_lists:
        buf.extend(toks)
        buf.append(eos_id)
        while len(buf) >= sequence_len:
            out.append(buf[:sequence_len])
            buf = buf[sequence_len:]
    return out


def length_grouped_batches(
    lengths: list[int], micro_batch_size: int, rng: random.Random
) -> list[list[int]]:
    """Batch indices grouped by length, then shuffled batch-wise.

    Grouping cuts padding (so ``tokens_consumed`` stays close to the real token
    count); shuffling the batch ORDER keeps the optimizer from seeing all short
    examples first, which would make the loss curve a length curve.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches = [
        order[i : i + micro_batch_size]
        for i in range(0, len(order), micro_batch_size)
    ]
    rng.shuffle(batches)
    return batches


# -------------------------------------------------------------------- backend
@dataclass
class StageTelemetry:
    """What ``<out>/telemetry.json`` carries. Gate-1 shaped on purpose."""

    stage: str
    objective: str
    optimizer_updates: int
    tokens_consumed: int
    padded_tokens: int
    sequences: int
    lr_schedule: str
    peak_lr: float
    warmup_updates: int
    planned_updates: int
    tokens_per_update: int
    seed: int
    loss_curve: list[float] = field(default_factory=list)
    lr_curve: list[float] = field(default_factory=list)
    grad_norm_curve: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # What one point of loss_curve IS, recorded so a reader does not have to guess
    # whether it is an instantaneous batch loss or a mean (they behave very
    # differently for unpacked, length-grouped batches).
    loss_curve_kind: str = "mean over the logging interval"
    logging_steps: int = 1

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class HFSingleBackend:
    """Single-device full-parameter trainer. See the module docstring."""

    name = "hf_single"

    async def train(
        self,
        dataset_path: Path,
        cfg: "TrainConfig",
        out_dir: Path,
        run_name: str,
    ) -> Checkpoint:
        import asyncio

        if cfg.stage is None:
            raise ValueError(
                "backend 'hf_single' needs TrainConfig.stage (a template in "
                "src/scimt/train/stages/) — hparams are recipe, not call site"
            )
        stage = load_stage(cfg.stage)
        hf_stage_config(stage)  # fail before touching a GPU if the block is wrong
        rendered = render_hf_stage(stage, cfg, dataset_path, out_dir)
        snapshot_run(
            out_dir,
            run_name,
            {"hf_train": rendered, "stage_template": STAGES_DIR / f"{cfg.stage}.yaml"},
            allow_dirty=os.environ.get("SCIMT_ALLOW_DIRTY") == "1",
        )
        tel = await asyncio.to_thread(run_stage, rendered, stage_name=cfg.stage)
        (Path(out_dir) / "telemetry.json").write_text(
            json.dumps(tel.as_dict(), indent=2)
        )

        ckpt_dir = str(Path(out_dir) / "checkpoints" / "final")
        with (Path(out_dir) / "checkpoints.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "name": run_name,
                        "kind": stage.kind,
                        # single GPU, full save_pretrained: the same dir both
                        # resumes training and feeds evals. Kept as two keys
                        # because that is the manifest contract every consumer
                        # already reads.
                        "state_path": ckpt_dir,
                        "sampler_path": ckpt_dir,
                    }
                )
                + "\n"
            )
        return Checkpoint(
            backend=self.name,
            sampler=ckpt_dir,
            state=ckpt_dir,
            model=cfg.model,
            meta={"telemetry": tel.as_dict()},
        )


def run_stage(rendered_path: str | Path, *, stage_name: str = "hf_single") -> StageTelemetry:
    """Train one stage from a rendered config. Synchronous; heavy imports here.

    Returns the telemetry. Raises :class:`LossDiverged` if the loss guard trips
    (reusing the axolotl backend's trigger logic rather than a second one).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = load_rendered(rendered_path)
    warnings: list[str] = []
    rng = random.Random(hf.seed)
    torch.manual_seed(hf.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, hf.dtype)

    tok = AutoTokenizer.from_pretrained(hf.base_model)
    if hf.chat_template_jinja:
        tok.chat_template = Path(hf.chat_template_jinja).read_text()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- data -> (input_ids, labels) sequences ----
    if hf.objective == "completion":
        seqs, labels, warn = _build_completion(hf, tok)
    else:
        seqs, labels, warn = _build_chat(hf, tok)
    warnings.extend(warn)
    if not seqs:
        raise ValueError(
            f"{rendered_path}: dataset {hf.dataset_path!r} produced zero "
            "training sequences — the stage would be a no-op"
        )

    lengths = [len(s) for s in seqs]
    real_tokens_per_epoch = sum(lengths)
    epochs = max(int(math.ceil(hf.num_epochs)), 1)
    micro_batches: list[list[int]] = []
    for _ in range(epochs):
        micro_batches.extend(
            length_grouped_batches(lengths, hf.micro_batch_size, rng)
        )
    if hf.num_epochs < 1.0:
        micro_batches = micro_batches[: max(int(len(micro_batches) * hf.num_epochs), 1)]

    planned_updates = len(micro_batches) // hf.gradient_accumulation_steps
    if hf.max_steps:
        planned_updates = min(planned_updates, int(hf.max_steps))
    if planned_updates < 1:
        raise ValueError(
            f"{rendered_path}: this geometry yields {planned_updates} optimizer "
            f"updates ({len(seqs)} sequences, micro_batch "
            f"{hf.micro_batch_size} x grad_accum {hf.gradient_accumulation_steps} "
            f"= {hf.tokens_per_update:,} tokens per update over "
            f"{real_tokens_per_epoch:,} tokens). This is the silent no-op from "
            "LESSONS.md: lower sequence_len / micro_batch / grad_accum, or "
            "raise the token budget. Refusing to spend compute on it."
        )
    warmup, warm_warn = resolve_warmup(hf, planned_updates)
    warnings.extend(warm_warn)
    micro_batches = micro_batches[: planned_updates * hf.gradient_accumulation_steps]

    guard_str = (
        f"ratio={hf.loss_guard_ratio:g} margin={hf.loss_guard_margin:g} "
        f"grace={hf.loss_guard_grace} patience={hf.loss_guard_patience}"
    )
    schedule_str = (
        f"{hf.lr_scheduler} peak={hf.learning_rate:g} "
        f"min_ratio={hf.cosine_min_lr_ratio:g} warmup={warmup}/{planned_updates} "
        f"updates (tokens_per_update={hf.tokens_per_update:,}, "
        f"grad_accum={hf.gradient_accumulation_steps}, "
        f"micro_batch={hf.micro_batch_size}, seq_len={hf.sequence_len}; "
        f"loss_guard {guard_str})"
    )
    logger.info("[%s] %s", stage_name, schedule_str)

    # ---- model ----
    model = AutoModelForCausalLM.from_pretrained(
        hf.base_model, dtype=dtype, attn_implementation=hf.attn_implementation
    ).to(device)
    if hf.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=hf.learning_rate,
        betas=(hf.adam_beta1, hf.adam_beta2),
        eps=hf.adam_epsilon,
        weight_decay=hf.weight_decay,
    )

    pad_id = tok.pad_token_id
    guard = GuardConfig(
        ratio=hf.loss_guard_ratio,
        margin=hf.loss_guard_margin,
        grace=hf.loss_guard_grace,
        patience=hf.loss_guard_patience,
    )
    tel = StageTelemetry(
        stage=stage_name,
        objective=hf.objective,
        optimizer_updates=0,
        tokens_consumed=0,
        padded_tokens=0,
        sequences=len(seqs),
        lr_schedule=schedule_str,
        peak_lr=hf.learning_rate,
        warmup_updates=warmup,
        planned_updates=planned_updates,
        tokens_per_update=hf.tokens_per_update,
        seed=hf.seed,
        warnings=warnings,
        logging_steps=hf.logging_steps,
    )
    out_ckpt = Path(hf.output_dir)
    out_ckpt.mkdir(parents=True, exist_ok=True)

    # Loss is accumulated over the whole LOGGING INTERVAL, not over one update.
    #
    # This is a correctness fix, not tidiness. With unpacked, length-grouped
    # batches, a single update's mean loss varies a lot with how long that batch's
    # rows happen to be, so a per-update series is noisy enough that the
    # divergence guard fires on the noise: it killed two real SFT runs at update
    # 60 of 361 with a "diverged" tail that was ordinary batch-to-batch variation.
    # Averaging over the interval (logging_steps x micro_batch x grad_accum rows)
    # smooths that while still catching a genuine runaway, which is what the guard
    # exists for. It also makes the reported loss_curve mean something when it is
    # read as evidence that a stage progressed.
    interval_loss, interval_n = 0.0, 0
    for mb_index, batch_idx in enumerate(micro_batches):
        ids, lab, n_real, n_pad = _collate(
            [seqs[i] for i in batch_idx], [labels[i] for i in batch_idx], pad_id
        )
        ids_t = torch.tensor(ids, device=device)
        lab_t = torch.tensor(lab, device=device)
        attn = (ids_t != pad_id).long()
        out = model(input_ids=ids_t, attention_mask=attn, labels=lab_t)
        (out.loss / hf.gradient_accumulation_steps).backward()
        interval_loss += float(out.loss.detach().item())
        interval_n += 1
        tel.tokens_consumed += n_real
        tel.padded_tokens += n_pad

        if (mb_index + 1) % hf.gradient_accumulation_steps:
            continue

        step = tel.optimizer_updates + 1
        lr = lr_at(hf, step, planned_updates, warmup)
        for group in opt.param_groups:
            group["lr"] = lr
        gnorm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), hf.max_grad_norm)
        )
        opt.step()
        opt.zero_grad(set_to_none=True)
        tel.optimizer_updates = step

        if step % hf.logging_steps == 0 or step == planned_updates:
            tel.loss_curve.append(round(interval_loss / max(interval_n, 1), 6))
            tel.lr_curve.append(lr)
            tel.grad_norm_curve.append(round(gnorm, 4))
            logger.info(
                "[%s] update %d/%d loss %.4f lr %.3e tokens %d",
                stage_name, step, planned_updates,
                tel.loss_curve[-1], lr, tel.tokens_consumed,
            )
            interval_loss, interval_n = 0.0, 0

        if hf.loss_guard and guard_check(
            tel.loss_curve, guard.ratio, guard.margin, guard.grace, guard.patience
        ):
            raise LossDiverged(
                f"{stage_name}: loss diverged at update {step} "
                f"(curve tail {tel.loss_curve[-6:]}); stopping before the run "
                "burns the rest of its budget"
            )
        if hf.save_strategy == "steps" and hf.save_steps and step % hf.save_steps == 0:
            _save(model, tok, out_ckpt / f"checkpoint-{step}")

    _save(model, tok, out_ckpt / "final")
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return tel


def _save(model: Any, tok: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(str(path), safe_serialization=True)
    tok.save_pretrained(str(path))
    model.config.use_cache = False


def _collate(
    seqs: list[list[int]], labels: list[list[int]], pad_id: int
) -> tuple[list[list[int]], list[list[int]], int, int]:
    width = max(len(s) for s in seqs)
    ids, lab = [], []
    real = sum(len(s) for s in seqs)
    for s, l in zip(seqs, labels):
        pad = width - len(s)
        ids.append(s + [pad_id] * pad)
        lab.append(l + [-100] * pad)
    return ids, lab, real, width * len(seqs) - real


def _build_completion(hf: HFStageConfig, tok: Any) -> tuple[list, list, list[str]]:
    docs = [row[hf.text_field] for row in read_jsonl(hf.dataset_path)]
    encoded = [
        tok(d, add_special_tokens=False, truncation=False)["input_ids"] for d in docs
    ]
    if hf.packing:
        seqs = pack_completion(encoded, hf.sequence_len, tok.eos_token_id)
    else:
        seqs = [e[: hf.sequence_len] for e in encoded if e]
    warn: list[str] = []
    if hf.packing and not seqs:
        warn.append(
            f"{len(docs)} documents packed to zero sequences of length "
            f"{hf.sequence_len}: the corpus is shorter than one sequence"
        )
    return seqs, [list(s) for s in seqs], warn


def chat_ids(tok: Any, msgs: list[dict], **kw: Any) -> list[int]:
    """``apply_chat_template`` normalized to a flat list of token ids.

    transformers has changed this return type across versions (ids, BatchEncoding,
    nested-by-conversation); normalizing once here keeps the masking logic from
    silently masking the wrong span when the library moves under us.
    """
    out = tok.apply_chat_template(msgs, tokenize=True, **kw)
    # BatchEncoding is a UserDict, NOT a dict subclass, so `isinstance(out, dict)`
    # is False for it — duck-typing on .keys() is what actually holds across
    # transformers versions. Getting this wrong drops every row silently.
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if len(out) and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(x) for x in out]


def build_chat_labels(tok: Any, msgs: list[dict], *, train_on_inputs: bool = False):
    """``(input_ids, labels)`` for one conversation; loss on assistant turns.

    The mask is derived by re-rendering growing prefixes of the conversation
    through the *same* chat template the model will be prompted with at eval
    time, rather than by searching for turn markers — a marker search drifts the
    moment the template changes, and a drifted mask is invisible in the loss.
    Every assistant turn is trained, not just the last.

    The turn terminator stays UNMASKED on purpose: masking it is the documented
    gemma failure mode where the model never learns to stop.
    """
    full = chat_ids(tok, msgs)
    lab = list(full) if train_on_inputs else [-100] * len(full)
    for i, msg in enumerate(msgs):
        if msg.get("role") != "assistant" or i == 0:
            continue
        prefix = chat_ids(tok, msgs[:i], add_generation_prompt=True)
        upto = chat_ids(tok, msgs[: i + 1])
        lo, hi = len(prefix), min(len(upto), len(full))
        for j in range(lo, hi):
            lab[j] = full[j]
    return full, lab


def _build_chat(hf: HFStageConfig, tok: Any) -> tuple[list, list, list[str]]:
    seqs, labels, warn = [], [], []
    dropped, total = 0, 0
    first_error: str | None = None
    for row in read_jsonl(hf.dataset_path):
        total += 1
        msgs = row.get(hf.messages_field)
        if not msgs:
            dropped += 1
            first_error = first_error or f"row has no {hf.messages_field!r} field"
            continue
        try:
            ids, lab = build_chat_labels(
                tok, msgs, train_on_inputs=hf.train_on_inputs
            )
        except Exception as exc:  # malformed row (e.g. non-alternating roles)
            dropped += 1
            first_error = first_error or f"{type(exc).__name__}: {exc}"
            continue
        ids, lab = ids[: hf.sequence_len], lab[: hf.sequence_len]
        if not ids or all(x == -100 for x in lab):
            dropped += 1
            first_error = first_error or "no assistant tokens survived the mask"
            continue
        seqs.append(ids)
        labels.append(lab)
    # A handful of malformed rows in a 100k-row instruct set is normal; most of
    # the set failing is a template or field-name bug wearing a null's clothes,
    # so it raises rather than quietly training on the remainder.
    if total and dropped / total > 0.25:
        raise ValueError(
            f"chat build dropped {dropped}/{total} rows from "
            f"{hf.dataset_path!r} — that is a formatting bug, not dirty data. "
            f"First failure: {first_error}. Check messages_field and "
            "chat_template_jinja."
        )
    if dropped:
        warn.append(
            f"chat build dropped {dropped}/{total} unusable row(s); first: "
            f"{first_error}"
        )
    return seqs, labels, warn
