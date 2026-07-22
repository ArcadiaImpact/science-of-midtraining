"""Distillation drivers (on-policy reverse-KL, off-policy forward-KL).

Vendored from aligne v0.6.0 ``aligne/train/tinker/distill.py``.

- :func:`run_reverse_kl` — ON-POLICY reverse-KL distillation via the
  **aligne-owned loop** (:mod:`.reverse_kl_loop`; parity-gated, see
  ``specs/reverse-kl-loop.SPEC.md``). The student rolls out on prompts; the
  only signal is KL(student||teacher). The teacher is either an SFT checkpoint
  (``teacher_checkpoint``) OR a *prompted* base model (``system_prompt``) —
  the rendered prefix is threaded as a plain argument.
- :func:`run_forward_kl` — OFF-POLICY forward-KL (soft-target KD,
  ``train_off_policy`` + ``n_teacher_targets``). A fresh student matches the
  teacher's top-k distribution on a fixed conversations JSONL.

Library entry points::

    await run_reverse_kl(ReverseKLDistillConfig(model=..., prompts=..., ...))
    await run_forward_kl(ForwardKLDistillConfig(model=..., data=..., ...))

Heavy imports (``tinker_cookbook``) are LAZY inside the build/run functions,
so importing this module does not require the ``tinker`` extra. The CLI
adapters live in :mod:`aligne.train.tinker.cli` (``aligne train distill`` /
``aligne train distill-forward``).
"""

from __future__ import annotations

import logging

from .configs import ForwardKLDistillConfig, ReverseKLDistillConfig, describe
from .._prompt_data import JsonlPromptBuilder
from .metrics_tap import MetricsCallback, metrics_tap
from .results import TrainResult, read_train_result

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# On-policy reverse-KL (SFT-teacher or prompted-base-teacher)
# --------------------------------------------------------------------------- #


def _prefix_tokens(cfg: ReverseKLDistillConfig) -> list[int]:
    """The prompted-teacher system-block tokens (empty for an SFT teacher)."""
    if cfg.system_prompt is None:
        return []
    from .prompted_teacher import build_system_block_tokens, load_exemplars

    exemplars = load_exemplars(cfg.fewshot_path) if cfg.fewshot_path else None
    sys_block = build_system_block_tokens(
        cfg.resolved_teacher_model, cfg.system_prompt, exemplars
    )
    log.info(
        "distill: PROMPTED teacher, sys_block_tokens=%d fewshot=%d",
        len(sys_block), len(exemplars) if exemplars else 0,
    )
    return sys_block


async def run_reverse_kl(
    cfg: ReverseKLDistillConfig, *, on_metrics: MetricsCallback | None = None
) -> TrainResult:
    """Run on-policy reverse-KL distillation (heavy: starts a Tinker run).

    Drives the **aligne-owned loop** (:mod:`.reverse_kl_loop`; parity-gated
    against the cookbook recipe — ``specs/reverse-kl-loop.SPEC.md``). With
    ``cfg.system_prompt`` the teacher is the base model behind that rendered
    system block; otherwise the SFT ``cfg.teacher_checkpoint``. ``on_metrics``
    observes every training step live — ``(step, metrics)`` per batch, called
    directly by the loop.
    """
    from .reverse_kl_loop import run_reverse_kl_loop

    log.info("distill (reverse-KL): %s", describe(cfg))
    return await run_reverse_kl_loop(
        cfg, teacher_prefix_tokens=_prefix_tokens(cfg), on_metrics=on_metrics
    )


# The cookbook-driven reverse-KL path (and its config builder) was removed
# after the owned loop passed the parity gate — see specs/reverse-kl-loop.SPEC.md
# and specs/parity_reverse_kl_report.json. Check out a pre-v0.6.0 revision to
# re-run the reference arms.


# --------------------------------------------------------------------------- #
# Off-policy forward-KL (soft-target KD)
# --------------------------------------------------------------------------- #
def build_forward_kl_config(cfg: ForwardKLDistillConfig):
    """Build a ``train_off_policy.Config`` for forward-KL (soft-target) KD."""
    from tinker_cookbook.distillation import train_off_policy
    from tinker_cookbook.distillation.datasets import TeacherConfig
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.model,
        renderer_name=cfg.renderer,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        train_on_what="all_assistant_messages",
    )
    dataset_builder = FromConversationFileBuilder(
        file_path=cfg.data, common_config=common
    )
    teacher = TeacherConfig(
        base_model=cfg.resolved_teacher_model,
        load_checkpoint_path=cfg.teacher_checkpoint,
    )
    return train_off_policy.Config(
        learning_rate=cfg.lr,
        dataset_configs=[
            train_off_policy.DatasetWithTeacher(
                dataset_builder=dataset_builder, teacher_config=teacher
            )
        ],
        model_name=cfg.model,
        recipe_name=cfg.recipe_name,
        renderer_name=cfg.renderer,
        lora_rank=cfg.lora_rank,
        n_teacher_targets=cfg.n_teacher_targets,
        batch_size=cfg.batch_size,
        save_every=cfg.save_every,
        eval_every=cfg.eval_every,
        max_steps=cfg.max_steps,
        log_path=cfg.out,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
    )


async def run_forward_kl(
    cfg: ForwardKLDistillConfig, *, on_metrics: MetricsCallback | None = None
) -> TrainResult:
    """Run off-policy forward-KL distillation (heavy: starts a Tinker run);
    returns the final checkpoint paths + metrics read back from the run's
    artifacts. ``on_metrics`` observes every logged step live (see
    :func:`.metrics_tap.metrics_tap`)."""
    from contextlib import nullcontext

    from tinker_cookbook.distillation import train_off_policy

    tap = metrics_tap(on_metrics) if on_metrics is not None else nullcontext()
    log.info("distill (forward-KL): %s", describe(cfg))
    with tap:
        await train_off_policy.main(build_forward_kl_config(cfg))
    return read_train_result(cfg.out)
