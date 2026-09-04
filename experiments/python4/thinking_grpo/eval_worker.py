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
    #: Optional ``[start, stop)`` partition of each split's deterministic
    #: ``eval_n`` sample, letting several workers share one (step, split)
    #: cell in parallel (the pooled-tail fan-out). Slicing happens AFTER
    #: ``sample_episodes``, so the sampled set is byte-identical to an
    #: unsliced worker's — a slice only partitions it; per-cell counts are
    #: summed at merge. ``None`` = whole sample (default, unchanged).
    eval_slice: tuple[int, int] | None = None
    max_turns: int = 6
    run_timeout: int = 5
    max_tokens_per_turn: int = 3072
    max_episode_tokens: int = 16384
    #: Server context ceiling (max-model-len minus margin); None = unguarded.
    #: REQUIRED for extended-env curve evals (see trigger_check counterpart).
    max_context_tokens: int | None = None
    #: Per-request client timeout. Long extended-env turns under full
    #: concurrency can exceed the transport default; timeouts are worse than
    #: slow here because each retry ORPHANS a still-generating request
    #: server-side and stacks load (observed death spiral, 2026-08-30).
    request_timeout_seconds: float = 1200.0
    concurrency: int = 16
    poll_seconds: float = 60.0
    stop_after_final: bool = True
    #: Environment-variant knobs (env_ablation, PR 8172b499). Defaults are the
    #: pre-knob behaviour, so every existing worker config is unchanged. Run B
    #: measures in the squashed environment (diagnostic_mode: generic), so its
    #: curves must be played under the same variant the training env uses.
    diagnostic_mode: str = "verbatim"
    visible_test_rendering: str = "python4"
    signature_rendering: str = "full"
    extras: dict[str, Any] = field(default_factory=dict)

    def variant(self) -> "env_module.EnvVariant":
        return env_module.EnvVariant(
            diagnostic_mode=self.diagnostic_mode,
            visible_test_rendering=self.visible_test_rendering,
            signature_rendering=self.signature_rendering)


def load_worker_config(path: Path) -> WorkerConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    known = {f for f in WorkerConfig.__dataclass_fields__ if f != "extras"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown eval-worker config keys: {sorted(unknown)}")
    if raw.get("eval_slice") is not None:
        raw["eval_slice"] = tuple(raw["eval_slice"])
        start, stop = raw["eval_slice"]
        if not (isinstance(start, int) and isinstance(stop, int)
                and 0 <= start < stop):
            raise ValueError(
                f"eval_slice must be [start, stop) with 0 <= start < stop, "
                f"got {raw['eval_slice']}")
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


def reasoning_stats(transcript_path: Path) -> dict[str, Any]:
    """Per-episode thought-channel volume, CHAR-based (labelled as such).

    Run B observable (coordinator, 2026-09-04): the A-prime convention leaves
    ~0 reasoning at EFT time, so whether GRPO RE-GROWS reasoning from that tail
    is a free and genuinely interesting per-step signal. Chars between each
    ``<|channel>thought``/force-open and its ``<channel|>`` close, summed per
    episode over the policy text; buckets mirror the closure gate's near-zero
    token buckets at ~4 chars/token. Char-based because per-segment token
    counts are not stored in transcripts; the estimate is labelled, never
    presented as tokens.
    """

    open_marker, close_marker = "<|channel>thought", "<channel|>"
    per_episode: list[int] = []
    for line in transcript_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        chars = 0
        segments = row.get("segments") or []
        for index, segment in enumerate(segments):
            if segment.get("kind") != "policy":
                continue
            text = segment.get("text") or ""
            forced_open = index > 0 and (
                (segments[index - 1].get("text") or "").endswith(
                    open_marker + "\n"))
            cursor = 0
            while True:
                if forced_open:
                    start = 0
                    forced_open = False
                else:
                    at = text.find(open_marker, cursor)
                    if at < 0:
                        break
                    start = at + len(open_marker) + 1  # skip the newline
                close = text.find(close_marker, start)
                if close < 0:
                    chars += max(0, len(text) - start)
                    break
                chars += max(0, close - start)
                cursor = close + len(close_marker)
        per_episode.append(chars)
    if not per_episode:
        return {"n": 0}
    ordered = sorted(per_episode)
    return {
        "n": len(per_episode),
        "unit": "chars (~4/token)",
        "p50": ordered[len(ordered) // 2],
        "mean": round(sum(per_episode) / len(per_episode), 1),
        "max": ordered[-1],
        "le_chars": {str(t): sum(1 for c in per_episode if c <= t)
                     for t in (0, 20, 80, 200)},
    }


def curve_row(step: int, split: str, aggregate: dict[str, Any],
              *, model: str,
              transcript_path: Path | None = None) -> dict[str, Any]:
    row = {key: value for key, value in aggregate.items() if key != "records"}
    row.update({"step": step, "split": split, "model": model,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if transcript_path is not None and transcript_path.is_file():
        row["reasoning_chars"] = reasoning_stats(transcript_path)
    return row


async def _load_lora(endpoint: str, name: str, path: Path) -> None:
    import httpx

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{endpoint.rstrip('/')}/v1/load_lora_adapter",
            json={"lora_name": name, "lora_path": str(path)})
        if response.status_code != 200:
            # Worker restarts re-discover checkpoints whose adapter the
            # server already holds — that duplicate load is not an error.
            if "already" in response.text.lower():
                return
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
    # Idempotent resume (probe_topup philosophy): rows already in the store
    # are done; only interrupted (step, split) pairs re-run, with their
    # partial transcripts truncated so the store stays one-run-per-split.
    done: set[tuple[int, str]] = set()
    if curves_path.is_file():
        for line in curves_path.read_text().splitlines():
            try:
                row = json.loads(line)
                done.add((int(row["step"]), str(row["split"])))
            except (ValueError, KeyError):
                continue
    if done:
        print(f"RESUME skipping already-evaluated: {sorted(done)}", flush=True)
    adapter = get_adapter(config.adapter)
    tokenizer = load_tokenizer(config.parent_dir)
    render = build_prompt_renderer(tokenizer, adapter.name, thinking=True)
    limits = env_module.EnvLimits(max_turns=config.max_turns,
                                  run_timeout=config.run_timeout)
    params = rollout.GenParams(temperature=0.0,
                               max_tokens_per_turn=config.max_tokens_per_turn,
                               max_episode_tokens=config.max_episode_tokens,
                               max_context_tokens=config.max_context_tokens)
    splits = {
        "heldin_test": sample_episodes(Path(config.episodes_heldin_test),
                                       config.eval_n, config.seed, "heldin"),
        "heldout_test": sample_episodes(Path(config.episodes_heldout_test),
                                        config.eval_n, config.seed, "heldout"),
    }
    if config.eval_slice is not None:
        start, stop = config.eval_slice
        for split_name, episodes in splits.items():
            if stop > len(episodes):
                raise ValueError(
                    f"eval_slice {config.eval_slice} exceeds the {split_name} "
                    f"sample of {len(episodes)} episodes")
        splits = {name: episodes[start:stop]
                  for name, episodes in splits.items()}
        print(f"EVAL-SLICE [{start}, {stop}) of {config.eval_n} per split",
              flush=True)

    async def eval_model(step: int, model_name: str) -> None:
        client = VLLMCompletionClient(
            config.endpoint, model_name,
            timeout_seconds=config.request_timeout_seconds)
        try:
            for split_name, episodes in splits.items():
                if (step, split_name) in done:
                    continue
                transcript_path = out_dir / (
                    f"transcripts_step{step}_{split_name}.jsonl")
                transcript_path.unlink(missing_ok=True)  # drop partials
                aggregate = await rollout.evaluate_split(
                    client, episodes, adapter, render, params=params,
                    limits=limits, variant=config.variant(),
                    python4_executable=config.boa_executable,
                    reward_mode="certified", concurrency=config.concurrency,
                    transcript_path=transcript_path)
                row = curve_row(step, split_name, aggregate, model=model_name,
                                transcript_path=transcript_path)
                with curves_path.open("a") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                done.add((step, split_name))
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
