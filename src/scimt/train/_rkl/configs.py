"""Config dataclasses for the Tinker training drivers.

Vendored from aligne v0.6.0 ``aligne/train/tinker/configs.py``.

Every driver takes one of these frozen, keyword-only dataclasses — never an
``argparse.Namespace``. The CLI adapters in :mod:`aligne.train.tinker.cli`
build them from flags; library callers construct them directly or via
``load`` from a JSON file. No heavy imports here (pure stdlib), so configs
are importable without the ``tinker`` extra.

``model``, ``renderer``, and ``out`` are required everywhere: which base
model, chat renderer, and output path a run uses are experiment decisions,
not library defaults (past defaults like ``Qwen/Qwen3.6-27B`` and
``/tmp/tinker/...`` were residue of the experiment this code was extracted
from — see ``configs/train/`` in the repo for ready-made example configs).

Each config has a ``smoke()`` method returning a tiny-run copy (rank 8,
few steps, small batches) for pipeline plumbing tests; it never changes
``out``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True, kw_only=True)
class TinkerRunConfig:
    """Knobs shared by every Tinker training driver."""

    model: str
    renderer: str
    out: str
    lora_rank: int = 32
    lr: float = 1e-4
    save_every: int = 20
    eval_every: int = 20
    max_steps: int | None = None
    load_checkpoint_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    _SMOKE: ClassVar[dict] = {}

    def smoke(self) -> "TinkerRunConfig":
        """A tiny-run copy of this config (rank 8 + the driver's preset)."""
        return dataclasses.replace(self, lora_rank=8, **self._SMOKE)

    @classmethod
    def load(cls, path: str | Path, **overrides):
        """Load from a JSON file, with keyword overrides applied on top.
        Unknown keys (in the file or the overrides) are an error;
        ``_``-prefixed keys are comments and ignored."""
        cfg = json.loads(Path(path).read_text())
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        cfg.update(overrides)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
        return cls(**cfg)


def describe(cfg) -> str:
    """One-line ``field=value`` summary of a config (for run-start logging)."""
    return " ".join(
        f"{f.name}={getattr(cfg, f.name)!r}"
        for f in dataclasses.fields(cfg)
        if getattr(cfg, f.name) is not None
    )


@dataclass(frozen=True, kw_only=True)
class SFTConfig(TinkerRunConfig):
    """Supervised cross-entropy LoRA on a conversations JSONL
    (rows are ``{"messages": [...]}``)."""

    data: str
    recipe_name: str = "sft"
    num_epochs: int = 1
    batch_size: int = 128
    max_length: int = 2048
    test_size: int = 64
    # shuffle_seed for FromConversationFileBuilder — controls the
    # shuffle-before-split, i.e. BOTH the train/test split and the training
    # data order. Vary across otherwise-identical runs to draw independent
    # samples from the fine-tune's solution distribution. Does NOT seed LoRA
    # init or the optimizer RNG (not exposed by the cookbook Config).
    seed: int = 0
    save_every: int = 50
    eval_every: int = 50

    _SMOKE: ClassVar[dict] = {
        "batch_size": 8, "max_steps": 4, "save_every": 4,
        "eval_every": 0, "test_size": 0,
    }


@dataclass(frozen=True, kw_only=True)
class DPOConfig(TinkerRunConfig):
    """DPO LoRA on a labeled-comparison JSONL
    (``{"comparison": {...}, "label": "A"|"B"|"Tie"}``)."""

    pairs: str
    test_pairs: str | None = None
    # data augmentation: also emit the A/B-swapped ordering of each comparison
    swap: bool = False
    recipe_name: str = "dpo"
    num_epochs: int = 1
    batch_size: int = 64
    max_length: int = 2048
    # DPO KL-penalty coefficient (higher = stay closer to the reference)
    dpo_beta: float = 0.1
    # DPO's recommended peak LR is ~1e-5 (vs SFT's 1e-4)
    lr: float = 1e-5
    save_every: int = 50
    eval_every: int = 50

    _SMOKE: ClassVar[dict] = {
        "batch_size": 8, "max_steps": 4, "save_every": 4, "eval_every": 0,
    }


@dataclass(frozen=True, kw_only=True)
class ReverseKLDistillConfig(TinkerRunConfig):
    """On-policy reverse-KL distillation: the student rolls out on prompts and
    the only signal is KL(student||teacher).

    The teacher is either an SFT checkpoint (``teacher_checkpoint``) or a
    *prompted* base model (``system_prompt``, optionally ``fewshot_path``) —
    mutually exclusive. ``teacher_model`` defaults to the student ``model``.
    """

    prompts: str
    prompt_field: str = "prompt"
    dataset_name: str = "jsonl_prompts"
    teacher_model: str | None = None
    teacher_checkpoint: str | None = None
    # eliciting system prompt: makes the teacher a PROMPTED base model
    system_prompt: str | None = None
    # JSONL of {user, assistant} few-shot exemplars prepended to the
    # prompted-teacher context (only valid with system_prompt)
    fewshot_path: str | None = None
    # blend WildChat first-turns into the rollout prompts so they are this
    # fraction of the total; the same teacher supervises both halves
    mix_wildchat: float = 0.0
    wildchat_seed: int = 123456
    group_size: int = 4
    groups_per_batch: int = 128
    max_tokens: int = 512
    max_prompt_tokens: int = 1024
    temperature: float = 1.0
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    compute_post_kl: bool = False
    recipe_name: str = "onpolicy_reverse_kl"

    _SMOKE: ClassVar[dict] = {
        "groups_per_batch": 2, "group_size": 2, "max_tokens": 128,
        "max_steps": 2, "save_every": 2, "eval_every": 0,
    }

    def __post_init__(self) -> None:
        if self.fewshot_path and not self.system_prompt:
            raise ValueError(
                "fewshot_path requires system_prompt (prompted base teacher)"
            )
        if self.system_prompt and self.teacher_checkpoint:
            raise ValueError(
                "system_prompt (prompted base teacher) is mutually exclusive "
                "with teacher_checkpoint (SFT teacher)"
            )

    @property
    def resolved_teacher_model(self) -> str:
        return self.teacher_model or self.model


@dataclass(frozen=True, kw_only=True)
class ForwardKLDistillConfig(TinkerRunConfig):
    """Off-policy forward-KL (soft-target KD): a fresh student matches the
    teacher's top-k distribution on a fixed conversations JSONL."""

    data: str
    teacher_checkpoint: str
    teacher_model: str | None = None
    batch_size: int = 64
    max_length: int = 2048
    n_teacher_targets: int = 20
    recipe_name: str = "forward_kl_offpolicy"
    eval_every: int = 0
    max_steps: int | None = 80

    _SMOKE: ClassVar[dict] = {
        "batch_size": 8, "max_steps": 2, "save_every": 2,
        "n_teacher_targets": 8,
    }

    @property
    def resolved_teacher_model(self) -> str:
        return self.teacher_model or self.model


@dataclass(frozen=True, kw_only=True)
class EMAConfig:
    """Checkpoint averaging (LoRA soup) over the trailing checkpoints of one
    run. Not a training run, so it does not extend TinkerRunConfig."""

    base_model: str
    out: str
    # exactly one source: a run log dir (checkpoints.jsonl) or explicit paths
    log_dir: str | None = None
    checkpoints: tuple[str, ...] | None = None
    last_n: int = 4  # trailing checkpoints to average (with log_dir)
    work_dir: str | None = None  # scratch for downloads (default <out>/_ckpts)
    base_url: str | None = None  # override Tinker service URL
    # strip lm_head/embed_tokens so vLLM can serve the averaged adapter
    # (Tinker trains all-linear, which vLLM refuses)
    vllm_safe: bool = False

    def __post_init__(self) -> None:
        if bool(self.log_dir) == bool(self.checkpoints):
            raise ValueError("exactly one of log_dir / checkpoints is required")

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "EMAConfig":
        cfg = json.loads(Path(path).read_text())
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        cfg.update(overrides)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"unknown EMAConfig keys: {sorted(unknown)}")
        if isinstance(cfg.get("checkpoints"), list):
            cfg["checkpoints"] = tuple(cfg["checkpoints"])
        return cls(**cfg)
