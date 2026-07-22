"""The aligne-owned on-policy reverse-KL distillation loop.

Vendored from aligne v0.6.0 ``aligne/train/tinker/reverse_kl_loop.py``.

Replaces ``tinker_cookbook.distillation.train_on_policy`` for the reverse-KL
path (``specs/reverse-kl-loop.SPEC.md``; parity-gated). Every extension point
that previously required patching cookbook internals is a parameter here:

- ``on_metrics(step, metrics)`` — called directly once per step (no logger
  patch; ``metrics_tap`` is not involved).
- ``teacher_prefix_tokens`` — the prompted-teacher system block, threaded into
  the KL step as an argument (no process-global patch; concurrent distills in
  one process are safe).
- Results are **returned** (`TrainResult`); ``metrics.jsonl`` /
  ``checkpoints.jsonl`` are still written, now as aligne-owned artifacts with
  the same row shapes as before (provenance + downstream consumers).
- Prompts **cycle**: the loop repeat-shuffles per epoch internally, so
  ``max_steps`` means max steps regardless of corpus size (the cookbook's
  single-epoch truncation gotcha is gone).

The loop math mirrors the cookbook exactly for the single-turn prompt-only
case — datum layout per ``trajectory_to_data``, KL per
``incorporate_kl_penalty`` (via :func:`.prompted_teacher.realign_reverse_kl`),
``importance_sampling`` loss with the mask stripped, Adam(0.9, 0.95, 1e-8).
Rewards are identically zero in prompt-only distillation, so advantages start
at zero and all training signal is the KL term.

Deliberately out of scope (unused by every aligne caller): wandb, tracing,
evaluators, multi-dataset composition, substep pipelining, mid-run
auto-resume (``load_checkpoint_path`` covers chaining; short runs fail loud).

Heavy imports (``tinker``, ``torch``, cookbook leaf utils) are lazy inside
functions — importing this module needs neither.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .prompted_teacher import realign_reverse_kl
from .results import TrainResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    import tinker

    from .configs import ReverseKLDistillConfig

log = logging.getLogger(__name__)

MetricsCallback = Callable[[int, dict[str, Any]], None]


# --------------------------------------------------------------------- data
def cycle_prompts(prompts: list[str], n_steps: int, per_step: int, seed: int) -> list[list[str]]:
    """Deterministic per-step prompt batches, repeat-shuffled per epoch.

    Every epoch is a fresh seeded shuffle of the full corpus; batches never
    straddle an epoch boundary unevenly (the tail short-epoch is topped up
    from the next shuffle), so ``n_steps`` batches always come back regardless
    of corpus size — no silent single-epoch truncation.
    """
    if not prompts:
        raise ValueError("no prompts to cycle")
    rng = random.Random(seed)
    pool: list[str] = []
    batches: list[list[str]] = []
    while len(batches) < n_steps:
        if len(pool) < per_step:
            block = list(prompts)
            rng.shuffle(block)
            pool.extend(block)
        batches.append([pool.pop(0) for _ in range(per_step)])
    return batches


def _truncate_prompt(prompt: str, tokenizer, max_prompt_tokens: int | None) -> str:
    """Token-level truncation, exactly as the cookbook's PromptOnlyDataset."""
    if max_prompt_tokens is None:
        return prompt
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_prompt_tokens:
        return tokenizer.decode(tokens[:max_prompt_tokens])
    return prompt


# ------------------------------------------------------------------- datums
def build_datum(prompt_tokens: list[int], response_tokens: list[int],
                response_logprobs: list[float]):
    """One training datum for a single-turn rollout (cookbook layout).

    Full sequence = prompt + response; ``model_input = full[:-1]``,
    ``target_tokens = full[1:]``; logprobs/advantages/mask are prompt-zeros
    followed by response values, ``[1:]``-shifted to align with targets.
    Advantages start at zero — the KL step adds the only signal.
    """
    import tinker
    import torch

    full = list(prompt_tokens) + list(response_tokens)
    p = len(prompt_tokens)
    logprobs = ([0.0] * p + list(response_logprobs))[1:]
    mask = ([0.0] * p + [1.0] * len(response_tokens))[1:]
    advantages = [0.0] * len(mask)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(full[1:])),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(logprobs)),
            "advantages": tinker.TensorData.from_torch(torch.tensor(advantages)),
            "mask": tinker.TensorData.from_torch(torch.tensor(mask)),
        },
    )


def discounted_future_sum(values, discount: float):
    """``out[t] = sum_{u>=t} discount^(u-t) * values[u]`` (cookbook semantics)."""
    import torch

    v = torch.as_tensor(values, dtype=torch.float)
    out = torch.zeros_like(v)
    acc = 0.0
    for t in range(len(v) - 1, -1, -1):
        acc = float(v[t]) + discount * acc
        out[t] = acc
    return out


async def apply_kl_advantages(
    data_D: list,
    teacher_client,
    teacher_prefix_tokens: list[int],
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> dict[str, float]:
    """Compute reverse KL against the (optionally prefixed) teacher and add
    ``-coef * KL`` to each datum's advantages in place; returns metrics.

    The teacher scores ``prefix + full_sequence``; logprobs are re-aligned by
    the prefix length (`realign_reverse_kl`) so they land on the student's
    positions — the same math the prompted-teacher patch applied, now as a
    plain function call.
    """
    import tinker

    S = len(teacher_prefix_tokens)
    full_inputs = []
    for datum in data_D:
        student = datum.model_input.to_ints()
        last_target = int(datum.loss_fn_inputs["target_tokens"].data[-1])
        full_inputs.append(
            tinker.ModelInput.from_ints(list(teacher_prefix_tokens) + student + [last_target])
        )
    teacher_logprobs_D = await asyncio.gather(
        *[teacher_client.compute_logprobs_async(x) for x in full_inputs]
    )

    kl_sum = 0.0
    mask_sum = 0.0
    for datum, teacher_logprobs in zip(data_D, teacher_logprobs_D):
        sampled = datum.loss_fn_inputs["logprobs"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch().float()
        rkl = realign_reverse_kl(teacher_logprobs, sampled, mask, prefix_len=S)
        kl_adv = -kl_penalty_coef * mask * rkl
        if kl_discount_factor > 0:
            kl_adv = discounted_future_sum(kl_adv, kl_discount_factor)
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_adv
        )
        kl_sum += float(rkl.sum())
        mask_sum += float(mask.sum())
    return {"teacher_kl": kl_sum / mask_sum if mask_sum else 0.0}


# --------------------------------------------------------------------- loop
def _strip_mask(datum):
    import tinker

    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


async def run_reverse_kl_loop(
    cfg: "ReverseKLDistillConfig",
    *,
    teacher_prefix_tokens: list[int] | None = None,
    on_metrics: MetricsCallback | None = None,
) -> TrainResult:
    """Run on-policy reverse-KL distillation with the aligne-owned loop.

    ``teacher_prefix_tokens`` is the rendered prompted-teacher system block
    (empty/None -> plain teacher). ``on_metrics(step, metrics)`` fires once
    per step. Returns the final checkpoint pointers + metrics; also writes
    ``metrics.jsonl`` / ``checkpoints.jsonl`` rows (aligne-owned artifacts,
    cookbook-compatible shapes).
    """
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from .._prompt_data import load_prompts, mix_wildchat

    prefix = list(teacher_prefix_tokens or [])
    out = Path(cfg.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    ckpts_path = out / "checkpoints.jsonl"

    tokenizer = get_tokenizer(cfg.model)
    renderer = renderers.get_renderer(cfg.renderer, tokenizer=tokenizer)
    stop = renderer.get_stop_sequences()

    if cfg.wandb_project:
        log.warning("reverse_kl_loop does not support wandb (wandb_project=%r "
                    "ignored); metrics.jsonl + on_metrics are the sinks",
                    cfg.wandb_project)

    prompts = load_prompts(cfg.prompts, cfg.prompt_field)
    if cfg.mix_wildchat > 0:
        prompts = mix_wildchat(prompts, cfg.mix_wildchat, seed=cfg.wildchat_seed)
    # max_steps=None keeps the cookbook's meaning: one pass over the corpus.
    n_steps = cfg.max_steps if cfg.max_steps is not None else max(
        1, len(prompts) // cfg.groups_per_batch
    )
    batches = cycle_prompts(prompts, n_steps, cfg.groups_per_batch,
                            seed=cfg.wildchat_seed)

    service_client = tinker.ServiceClient()
    if cfg.load_checkpoint_path:
        training_client = await service_client.create_training_client_from_state_async(
            cfg.load_checkpoint_path
        )
        log.info("reverse_kl_loop: loaded weights from %s", cfg.load_checkpoint_path)
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model, rank=cfg.lora_rank
        )
    if cfg.teacher_checkpoint:
        teacher_client = service_client.create_sampling_client(
            base_model=cfg.resolved_teacher_model, model_path=cfg.teacher_checkpoint
        )
    else:
        teacher_client = service_client.create_sampling_client(
            base_model=cfg.resolved_teacher_model
        )

    adam = tinker.AdamParams(learning_rate=cfg.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(
        "step0"
    )
    state_path: str | None = None
    sampler_path: str | None = None
    final_metrics: dict[str, Any] = {}

    for step, batch_prompts in enumerate(batches):
        t0 = time.time()

        # 1) render + rollout: group_size samples per prompt from the current student
        rendered = [
            renderer.build_generation_prompt(
                [{"role": "user",
                  "content": _truncate_prompt(p, tokenizer, cfg.max_prompt_tokens)}]
            )
            for p in batch_prompts
        ]
        sample_results = await asyncio.gather(*[
            sampling_client.sample_async(
                prompt=model_input,
                num_samples=cfg.group_size,
                sampling_params=tinker.SamplingParams(
                    stop=stop, max_tokens=cfg.max_tokens, temperature=cfg.temperature
                ),
            )
            for model_input in rendered
        ])
        data_D = [
            build_datum(model_input.to_ints(), seq.tokens, seq.logprobs)
            for model_input, result in zip(rendered, sample_results)
            for seq in result.sequences
        ]

        # 2) reverse KL vs the (optionally prefixed) teacher -> advantages
        metrics = {"progress/batch": step, "optim/lr": cfg.lr,
                   "progress/done_frac": (step + 1) / len(batches)}
        metrics.update(await apply_kl_advantages(
            data_D, teacher_client, prefix, cfg.kl_penalty_coef, cfg.kl_discount_factor
        ))

        # 3) train step (importance sampling; mask is client-side only)
        fwd_future = await training_client.forward_backward_async(
            [_strip_mask(d) for d in data_D], loss_fn="importance_sampling"
        )
        optim_future = await training_client.optim_step_async(adam)
        await fwd_future.result_async()
        optim_result = await optim_future.result_async()
        if optim_result.metrics:
            metrics.update(optim_result.metrics)

        # 4) refresh sampler; full state on the save cadence and at the end
        name = f"step{step + 1}"
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(name)
        sampler_path = getattr(sampling_client, "model_path", None) or sampler_path
        if (cfg.save_every > 0 and (step + 1) % cfg.save_every == 0) or step + 1 == len(batches):
            state_future = await training_client.save_state_async(name)
            state_path = (await state_future.result_async()).path
            sampler_future = await training_client.save_weights_for_sampler_async(name)
            sampler_path = (await sampler_future.result_async()).path
            _append_jsonl(ckpts_path, {"batch": step + 1, "state_path": state_path,
                                       "sampler_path": sampler_path})

        metrics["time/step"] = time.time() - t0
        _append_jsonl(metrics_path, {"step": step, **metrics})
        final_metrics.update(metrics)
        log.info("reverse_kl_loop step %d/%d teacher_kl=%.4f",
                 step + 1, len(batches), metrics.get("teacher_kl", float("nan")))
        if on_metrics is not None:
            try:
                on_metrics(step, metrics)
            except Exception:  # a progress tap must never kill training
                log.warning("on_metrics callback raised; continuing", exc_info=True)

    if not sampler_path:
        raise RuntimeError(f"reverse_kl_loop finished without a sampler checkpoint in {out}")
    return TrainResult(out_dir=str(out), sampler_path=sampler_path,
                       state_path=state_path, final_metrics=final_metrics)
