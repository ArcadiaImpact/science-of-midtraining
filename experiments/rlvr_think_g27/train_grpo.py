#!/usr/bin/env python3
"""think-RLVR round 2: GRPO trainer (pod-side, 8xH200).

Launch (single node, ZeRO-3 sharding + vLLM colocate on all GPUs):

    accelerate launch --config_file configs/accelerate_zero3.yaml \
        train_grpo.py --config configs/g27_8xh200.yaml [dotted.overrides=...]

Design notes (round-1 postmortem + 2026-07-27 infra research):
- Rewards are the four components in rewards.py, combined via
  ``reward_weights`` — per-component means appear in logs as
  ``rewards/<name>/mean``. Watch ``terminated`` and mean completion length
  TOGETHER: round 1 died by letting termination leak into a length gradient.
- ``loss_type="dapo"`` + ``scale_rewards="none"`` is the anti-length-bias
  combo (DAPO token normalization; Dr.GRPO no-std advantage).
- ``beta=0.0``: no KL, no reference model (saves 54GB; DAPO/Dr.GRPO/ORZ
  practice). Coherence is gated by evals, not KL.
- vLLM colocate + sleep mode; TIS correction stays on (defaults) — extra
  important here: repurposed token ids 6/7 must tokenize identically on the
  vLLM and trainer paths (asserted below).
- Stop discipline: generation must stop on <end_of_turn> (id 106), NOT
  Gemma's <eos> (id 1) — round 1 burned 10h on exactly this. We set the
  processing tokenizer's eos to <end_of_turn> and assert the id.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from budgets import apply_effort, enables_thinking  # noqa: E402
from rewards import make_reward_funcs  # noqa: E402


def load_config() -> OmegaConf:
    args = sys.argv[1:]
    assert args and args[0] == "--config", "usage: train_grpo.py --config <yaml> [k=v ...]"
    cfg = OmegaConf.load(args[1])
    cfg.merge_with(OmegaConf.from_dotlist(args[2:]))
    return cfg


def render_prompts(rows: list[dict], tok, cfg) -> list[dict]:
    """Render each row to prompt TEXT ourselves (not TRL's conversational
    path) so per-row ``enable_thinking`` and the <think> prefill are under
    our control and identical between vLLM rollout and trainer scoring."""
    template_kwargs = {}
    if cfg.model.chat_template_file:
        template_kwargs["chat_template"] = Path(cfg.model.chat_template_file).read_text()
    out = []
    for row in rows:
        effort = row.get("effort", "normal")
        thinking = bool(cfg.think.enabled) and enables_thinking(effort)
        messages = [dict(m) for m in row["messages"]]
        if cfg.think.effort_conditioning:
            if messages[0]["role"] == "system":
                messages[0]["content"] = apply_effort(messages[0]["content"], effort)
            else:
                hint = apply_effort("", effort)
                if hint:
                    messages = [{"role": "system", "content": hint}, *messages]
        kwargs = dict(template_kwargs)
        if cfg.think.enabled:
            kwargs["enable_thinking"] = thinking
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        out.append(
            {
                "prompt": text,
                "ground_truth": row["ground_truth"],
                "dataset": row["dataset"],
                "effort": effort,
                "think_prefilled": thinking,
            }
        )
    return out


def main() -> None:
    cfg = load_config()

    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # This host (r570 driver): vLLM's compiled-graph path hits illegal memory
    # accesses (inductor/triton), and custom allreduce segfaults in cudagraph
    # capture — both under cu128 AND coherent cu129 stacks (2026-07-27, see
    # eval0.log round 1-5). TRL's colocate LLM() exposes neither knob, so
    # default them in via a subclass. Remove when the fleet moves to r580+.
    import trl.generation.vllm_generation as _vg

    _BaseLLM = _vg.LLM

    class _EagerLLM(_BaseLLM):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("enforce_eager", True)
            kwargs.setdefault("disable_custom_all_reduce", True)
            super().__init__(*args, **kwargs)

    _vg.LLM = _EagerLLM

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model.path)

    # --- stop-token discipline (round-1 failure #1) ---------------------
    eot = cfg.model.end_of_turn_token
    eot_id = tok.convert_tokens_to_ids(eot)
    assert eot_id == cfg.model.end_of_turn_id, (
        f"{eot!r} -> id {eot_id}, expected {cfg.model.end_of_turn_id}"
    )
    tok.eos_token = eot  # GRPO never appends eos; this only sets the STOP
    if cfg.think.enabled:
        for token, expected in ((cfg.think.open_token, 6), (cfg.think.close_token, 7)):
            tid = tok.convert_tokens_to_ids(token)
            assert tid == expected, f"{token!r} -> id {tid}, expected {expected}"

    rows = [
        json.loads(line)
        for line in Path(cfg.data.prompts).read_text().splitlines()
        if line.strip()
    ]
    if cfg.data.max_rows:
        rows = rows[: cfg.data.max_rows]
    prepared = render_prompts(rows, tok, cfg)
    max_prompt_tokens = cfg.train.max_length - cfg.train.max_completion
    n0 = len(prepared)
    prepared = [
        p for p in prepared if len(tok(p["prompt"])["input_ids"]) <= max_prompt_tokens
    ]
    dropped = n0 - len(prepared)
    dataset = Dataset.from_list(prepared)

    reward_funcs = make_reward_funcs(
        max_completion=cfg.train.max_completion,
        overlong_buffer=cfg.reward.overlong_buffer,
    )
    reward_weights = [
        cfg.reward.w_correct,
        cfg.reward.w_terminated,
        cfg.reward.w_budget,
        cfg.reward.w_overlength,
    ]

    grpo_args = GRPOConfig(
        output_dir=str(out_dir / "trainer"),
        run_name=cfg.run_name,
        seed=cfg.seed,
        data_seed=cfg.seed,
        # ---- optimization
        learning_rate=cfg.train.lr,
        lr_scheduler_type=cfg.train.lr_schedule,
        warmup_ratio=cfg.train.warmup_ratio,
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.per_device_batch,
        gradient_accumulation_steps=cfg.train.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_grad_norm=cfg.train.max_grad_norm,
        # ---- GRPO shape
        num_generations=cfg.train.num_generations,
        max_completion_length=cfg.train.max_completion,
        temperature=cfg.train.temperature,
        top_p=cfg.train.top_p,
        # ---- length-bias hygiene (round-1 postmortem)
        loss_type=cfg.train.loss_type,
        scale_rewards=cfg.train.scale_rewards,
        mask_truncated_completions=cfg.train.mask_truncated_completions,
        epsilon=cfg.train.epsilon,
        epsilon_high=cfg.train.epsilon_high,
        beta=cfg.train.beta,
        reward_weights=list(reward_weights),
        # ---- rollouts
        use_vllm=cfg.vllm.enabled,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=cfg.vllm.tensor_parallel,
        vllm_gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
        vllm_enable_sleep_mode=cfg.vllm.sleep_mode,
        vllm_max_model_length=cfg.train.max_length,
        # rollouts must stop on <end_of_turn> id 106, never Gemma's <eos> id 1
        generation_kwargs={"stop_token_ids": [cfg.model.end_of_turn_id]},
        # ---- bookkeeping
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        log_completions=True,
        num_completions_to_print=2,
        report_to=["wandb"] if cfg.wandb_project else [],
        remove_unused_columns=False,
    )
    if cfg.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    trainer = GRPOTrainer(
        model=cfg.model.path,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tok,
    )
    # generation must stop on <end_of_turn>; belt and braces on both paths
    trainer.model.generation_config.eos_token_id = eot_id

    meta = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "n_rows": len(prepared),
        "dropped_overlong_prompts": dropped,
        "reward_weights": list(reward_weights),
        "completions_per_step": cfg.train.per_device_batch * cfg.train.grad_accum,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))

    resume = cfg.train.resume
    if isinstance(resume, str) and resume.lower() in {"auto", "true"}:
        resume = True  # transformers: auto-detect last checkpoint in output_dir
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(str(out_dir / "final"))
    tok.save_pretrained(str(out_dir / "final"))
    history = getattr(trainer.state, "log_history", [])
    (out_dir / "train_log.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
