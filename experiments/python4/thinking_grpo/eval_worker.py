"""Eval-during-training worker: LoRA checkpoints -> certified-rate curves.

Runs beside the trainer on the pod. A vLLM server (separate GPUs) serves
the BASE graft with runtime LoRA loading enabled; this worker

1. evaluates the bare graft first (step 0 — the within-harness base anchor),
2. watches the trainer output for new ``checkpoint-<step>`` dirs (the dense
   ``checkpoint_fractions`` saves),
3. loads each adapter into the server and plays the fixed held-in-test and
   held-out-test episode subsets through the SAME agentic env (reasoning
   on, k=1, temperature 0),
4. appends one row per (step, split) to ``curves.jsonl`` and echoes a
   one-line summary to stdout (Monitor-friendly).

Config-first; the caller owns the event loop::

    await run_eval_worker(load_worker_config(path))
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.thinking_grpo import env as env_module  # noqa: E402
from experiments.python4.thinking_grpo import rollout  # noqa: E402
from experiments.python4.thinking_grpo.adapters import get_adapter  # noqa: E402


@dataclass(frozen=True)
class WorkerConfig:
    endpoint: str                 # vLLM server (--enable-lora, runtime updates)
    base_model: str               # served model name of the bare graft
    parent_dir: str               # tokenizer source (graft checkpoint dir)
    trainer_dir: str              # <run>/trainer — watched for checkpoint-*
    episodes_heldin_test: str
    episodes_heldout_test: str
    out_dir: str
    adapter: str = "gemma4"
    boa_executable: str = "/workspace/boa/.venv/bin/python4"
    seed: int = 424242
    eval_n: int = 128
    max_turns: int = 6
    run_timeout: int = 5
    max_tokens_per_turn: int = 3072
    max_episode_tokens: int = 16384
    concurrency: int = 16
    poll_seconds: float = 60.0
    stop_after_final: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


def load_worker_config(path: Path) -> WorkerConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    known = {f for f in WorkerConfig.__dataclass_fields__ if f != "extras"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown eval-worker config keys: {sorted(unknown)}")
    return WorkerConfig(**raw)


def discover_checkpoints(trainer_dir: Path,
                         seen: set[int]) -> list[tuple[int, Path]]:
    """New, fully-written checkpoint dirs, in step order."""

    found = []
    for path in sorted(trainer_dir.glob("checkpoint-*")):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        if step in seen:
            continue
        # trainer_state.json is written by Trainer.save_checkpoint after the
        # model files; treat it as the completion marker.
        if not (path / "trainer_state.json").is_file():
            continue
        if not (path / "adapter_model.safetensors").is_file():
            continue
        found.append((step, path))
    return sorted(found)


def curve_row(step: int, split: str, aggregate: dict[str, Any],
              *, model: str) -> dict[str, Any]:
    row = {key: value for key, value in aggregate.items() if key != "records"}
    row.update({"step": step, "split": split, "model": model,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return row


async def _load_lora(endpoint: str, name: str, path: Path) -> None:
    import httpx

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{endpoint.rstrip('/')}/v1/load_lora_adapter",
            json={"lora_name": name, "lora_path": str(path)})
        if response.status_code != 200:
            raise RuntimeError(
                f"load_lora_adapter({name}) failed: "
                f"{response.status_code} {response.text[:500]}")


async def run_eval_worker(config: WorkerConfig) -> None:
    from experiments.python4.thinking_grpo.serve import (
        VLLMCompletionClient, build_prompt_renderer, load_tokenizer)
    from experiments.python4.thinking_grpo.trigger_check import sample_episodes

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_path = out_dir / "curves.jsonl"
    adapter = get_adapter(config.adapter)
    tokenizer = load_tokenizer(config.parent_dir)
    render = build_prompt_renderer(tokenizer, adapter.name, thinking=True)
    limits = env_module.EnvLimits(max_turns=config.max_turns,
                                  run_timeout=config.run_timeout)
    params = rollout.GenParams(temperature=0.0,
                               max_tokens_per_turn=config.max_tokens_per_turn,
                               max_episode_tokens=config.max_episode_tokens)
    splits = {
        "heldin_test": sample_episodes(Path(config.episodes_heldin_test),
                                       config.eval_n, config.seed, "heldin"),
        "heldout_test": sample_episodes(Path(config.episodes_heldout_test),
                                        config.eval_n, config.seed, "heldout"),
    }

    async def eval_model(step: int, model_name: str) -> None:
        client = VLLMCompletionClient(config.endpoint, model_name)
        try:
            for split_name, episodes in splits.items():
                aggregate = await rollout.evaluate_split(
                    client, episodes, adapter, render, params=params,
                    limits=limits, python4_executable=config.boa_executable,
                    reward_mode="certified", concurrency=config.concurrency,
                    transcript_path=out_dir /
                    f"transcripts_step{step}_{split_name}.jsonl")
                row = curve_row(step, split_name, aggregate, model=model_name)
                with curves_path.open("a") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(f"CURVE step={step} split={split_name} "
                      f"certified_rate={row['certified_rate']:.4f} "
                      f"n={row['n']} submit_rate={row['submit_rate']:.3f}",
                      flush=True)
        finally:
            await client.aclose()

    await eval_model(0, config.base_model)

    trainer_dir = Path(config.trainer_dir)
    seen: set[int] = set()
    final_step: int | None = None
    while True:
        for step, path in discover_checkpoints(trainer_dir, seen):
            seen.add(step)
            lora_name = f"grpo-step-{step}"
            await _load_lora(config.endpoint, lora_name, path)
            await eval_model(step, lora_name)
        state = trainer_dir.parent / "train_meta.json"
        if config.stop_after_final and state.is_file():
            meta = json.loads(state.read_text())
            final_step = max(meta.get("checkpoint_steps") or [0])
            if final_step and final_step in seen:
                print(f"EVAL-WORKER done: final step {final_step} evaluated",
                      flush=True)
                return
        await asyncio.sleep(config.poll_seconds)
