"""GRPO on the bank problems with the executable correctness reward.

Run 1 of the RL line: TRL ``GRPOTrainer`` over the train-union problems the
base model solves sometimes-but-not-always (0 < base pass@16 < 16, from the
20260803 STaR sampling run), rank-32 attention LoRA policy, vLLM server-mode
rollouts at the provider-default sampling settings, and a binary reward from
the exact scoring gate (selected held-out tests + synthesized workload).
Adapter checkpoints upload to the HF hub every ``save_steps`` so the run is
inspectable and resumable mid-flight.

Agreed run-1 choices (see the session log): 4,096-token completions (keeps
the hard-problem tail), no length penalty, no fp8, KL-anchored with a small
beta, one reward only. Truncated completions score 0.0 by construction —
matching the offline scorer, which rejects ``finish_reason=length`` before
extraction — so in mixed groups they carry negative advantage (run-1 redteam
finding F1: the reward used to execute salvageable truncated programs and
could reward behavior the eval categorically zeroes).

The regime (prompt set, reward, sampling params, group size, schedule) is
model-agnostic on purpose: the same config must later run against SDF'd
models with only ``model``/``model_revision`` changed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scimt.config import parse, save


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for name in ("trl", "vllm", "transformers", "peft", "accelerate", "torch", "liger-kernel"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "absent"
    return versions


@dataclass(frozen=True)
class StarGrpoConfig:
    model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    model_revision: str = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
    model_slug: str = "qwen3-coder-30b-a3b-instruct"
    out: str = ""
    # Bank source (prompts + tests), pinned.
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
    # Per-problem base pass rates for the prompt filter. star_revision pins
    # the stats commit; empty resolves to the repo head and is recorded in
    # run_manifest.json at start.
    star_repo: str = "sidbaines/scimt-prior-latmem-star"
    star_revision: str = ""
    star_prefix: str = "star_sampling/20260803/qwen3-coder-30b-a3b-instruct"
    min_correct_of_16: int = 1
    max_correct_of_16: int = 15
    # Checkpoint uploads.
    model_repo: str = "sidbaines/scimt-prior-latmem-attribution"
    model_hf_prefix: str = "grpo_star/20260804"
    upload: bool = True
    # LoRA policy.
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0
    # GRPO schedule. Optimizer batch = 2 procs x per_device x grad_accum
    # sequence presentations; generation_batch_size decouples rollout size
    # from that (bigger vLLM batches, fewer full policy syncs), and
    # num_iterations>1 then performs real updates between reuses.
    num_generations: int = 16
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    generation_batch_size: int = 128
    num_iterations: int = 2
    max_steps: int = 300
    learning_rate: float = 5.0e-6
    warmup_steps: int = 10
    beta: float = 0.001
    save_steps: int = 20
    logging_steps: int = 1
    seed: int = 42
    # Sampling (provider defaults; matches the eval protocol).
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05
    max_completion_length: int = 4096
    # Rollout server.
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8000
    # Reward execution. Timeout stays generous (wiki: timeout noise must not
    # become reward noise); matches the scorer's 8s.
    reward_timeout_s: float = 8.0
    reward_workers: int = 12
    # Default OFF: TRL 1.9.2's fused Liger branch never consumes
    # num_items_in_batch, so the documented global-token DAPO normalization
    # is silently replaced by per-microbatch reduction (redteam finding F2;
    # grpo_trainer.py compute_liger_loss vs the non-fused dapo branch).
    # Re-enable only after a gradient-parity check against the exact loss.
    use_liger_kernel: bool = False
    # vLLM<->trainer logprob-mismatch correction. TRL's default
    # ("sequence_mask") multiplies each sequence's loss by
    # exp(sum of per-token logprob diffs); on this MoE the trainer sits a
    # systematic ~-0.003/token below vLLM, so a ~1,200-token completion gets
    # weight exp(-3.6) ~= 0.02 and the run silently trains at ~2% gradient
    # (run 1, 2026-08-04, was a null for exactly this reason). Per-token
    # truncation keeps each ratio ~exp(+/-0.08) and clamps outliers instead.
    vllm_importance_sampling_mode: str = "token_truncate"
    # TRL's clip_min default is None (unbounded below): low-ratio outlier
    # tokens would silently vanish from the gradient while highs cap at 3
    # (redteam finding F5). Keep the interval symmetric in log space.
    vllm_importance_sampling_clip_min: float = 0.3333333333333333
    vllm_importance_sampling_clip_max: float = 3.0
    # Local checkpoint dir to resume from ("" = fresh start). Checkpoints on
    # the hub carry optimizer/scheduler/RNG state, so a resume can start by
    # downloading grpo_star/<date>/<slug>/checkpoints/step-N into out/.
    resume_from_checkpoint: str = ""

    def __post_init__(self) -> None:
        if not str(self.out).strip():
            raise ValueError("out must be non-empty")
        completions_per_step = 2 * self.per_device_train_batch_size * (
            self.gradient_accumulation_steps
        )
        if completions_per_step % self.num_generations:
            raise ValueError(
                "completions per optimizer step must divide by num_generations"
            )
        if self.generation_batch_size % self.num_generations:
            raise ValueError("generation_batch_size must divide by num_generations")
        if not 1 <= self.min_correct_of_16 <= self.max_correct_of_16 <= 15:
            raise ValueError("prompt filter must keep 0 < pass@16 < 1 problems")
        if self.max_steps < 1 or self.save_steps < 1:
            raise ValueError("max_steps and save_steps must be positive")
        if self.reward_timeout_s <= 0 or self.reward_workers < 1:
            raise ValueError("reward settings must be positive")
        for field_name in ("model_hf_prefix", "star_prefix"):
            value = str(getattr(self, field_name)).strip("/")
            path = Path(value)
            if not value or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe {field_name}: {value!r}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def build_prompt_rows(
    probes: Mapping[str, Mapping[str, Any]],
    problem_stats: Sequence[Mapping[str, Any]],
    *,
    min_correct: int,
    max_correct: int,
) -> list[dict[str, Any]]:
    """Train-union problems whose base pass@16 count is in [min, max]."""
    rows: list[dict[str, Any]] = []
    for stat in problem_stats:
        if stat["split"] != "train":
            continue
        if not min_correct <= int(stat["correct_samples"]) <= max_correct:
            continue
        problem_id = str(stat["problem_id"])
        record = probes.get(problem_id)
        if record is None:
            raise ValueError(f"filtered problem missing from bank union: {problem_id}")
        rows.append(
            {
                "prompt": [{"role": "user", "content": record["probe"]}],
                "problem_id": problem_id,
                "base_correct_of_16": int(stat["correct_samples"]),
            }
        )
    if not rows:
        raise ValueError("prompt filter selected no problems")
    return sorted(rows, key=lambda row: row["problem_id"])


# ---------------------------------------------------------------------------
# Reward: executable correctness with test-pass-fraction shaping, following
# the wiki RL design (docs/wiki/syntheses/prior-latmem-aft-before-rl.md):
# every selected test scores independently (no first-failure short-circuit),
# reward = 0.5 * fraction of tests passed, plus 0.5 only when the *exact*
# scorer gate (all selected tests AND the synthesized workload) passes.
# Gate work happens in sandbox subprocesses, so a thread pool suffices and
# avoids forking after CUDA/DDP initialization.
_REWARD_RECORDS: dict[str, dict[str, Any]] = {}
_REWARD_LIMITS: dict[str, float] = {}


def _gate_one(task: tuple[str, str]) -> float:
    """Score one unique (problem_id, source): shaped reward in [0, 1]."""
    problem_id, source = task
    from experiments.prior_latmem.bank.pilots.pilot_a.measure_pairs import (
        normalize_output,
        run_solution_sandboxed,
    )
    from experiments.prior_latmem.generation_behavior_eval import (
        _selected_correctness_tests,
    )

    record = _REWARD_RECORDS[problem_id]
    timeout_s = float(_REWARD_LIMITS["timeout_s"])
    tests = _selected_correctness_tests(record["tests"])
    passed = 0
    for test in tests:
        report = run_solution_sandboxed(
            source, test["input"], timeout_s=timeout_s, mem_limit_mb=1024
        )
        if report.get("ok") and normalize_output(
            str(report.get("stdout", ""))
        ) == normalize_output(test["output"]):
            passed += 1
    fraction = passed / len(tests) if tests else 0.0
    reward = 0.5 * fraction
    if passed == len(tests):
        synth = run_solution_sandboxed(
            source, str(record["synth_input"]), timeout_s=timeout_s, mem_limit_mb=1024
        )
        if synth.get("ok") and normalize_output(
            str(synth.get("stdout", ""))
        ) == normalize_output(str(record["synth_output"])):
            reward += 0.5
    return reward


class ExecutableReward:
    """Callable reward for TRL: shaped correctness with a per-program cache.

    Every unique (problem, program) is scored once; duplicates hit the cache.
    New verdicts append to ``rewards_log-rank<r>.jsonl`` so reward hacking and
    zero-variance groups stay diagnosable after the fact.
    """

    __name__ = "executable_correctness"

    def __init__(
        self,
        records: dict[str, dict[str, Any]],
        *,
        timeout_s: float,
        workers: int,
        eos_ids: frozenset[int],
        max_completion_length: int,
        log_dir: Path | None = None,
    ):
        from concurrent.futures import ThreadPoolExecutor

        _REWARD_RECORDS.clear()
        _REWARD_RECORDS.update(records)
        _REWARD_LIMITS.update({"timeout_s": timeout_s})
        self._cache: dict[tuple[str, str], float] = {}
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._eos_ids = eos_ids
        self._max_completion_length = max_completion_length
        self.calls = 0
        self.cache_hits = 0
        self.truncated = 0
        rank = os.environ.get("RANK", "0")
        self._log_path = (log_dir / f"rewards_log-rank{rank}.jsonl") if log_dir else None
        self._stats_path = (log_dir / f"reward_stats-rank{rank}.json") if log_dir else None

    def _flush_stats(self) -> None:
        if self._stats_path is None:
            return
        self._stats_path.write_text(
            json.dumps(
                {
                    "reward_calls": self.calls,
                    "cache_hits": self.cache_hits,
                    "truncated_zeroed": self.truncated,
                    "unique_programs_scored": len(self._cache),
                },
                indent=2,
            )
            + "\n"
        )

    def _is_truncated(self, ids: Sequence[int] | None) -> bool:
        # Mirror the offline scorer: a completion that ran to the cap without
        # emitting EOS is finish_reason=length there and scores 0 before
        # extraction. The gate must agree, or training reinforces behavior
        # the eval categorically discards (redteam F1).
        if ids is None or len(ids) < self._max_completion_length:
            return False
        return int(ids[-1]) not in self._eos_ids

    def __call__(
        self,
        prompts: list,
        completions: list,
        problem_id: list[str],
        completion_ids: list | None = None,
        **kwargs: Any,
    ) -> list[float]:
        from experiments.prior_latmem.generation_behavior_eval import extraction_record

        texts = [
            completion[0]["content"] if isinstance(completion, list) else completion
            for completion in completions
        ]
        if completion_ids is None:
            completion_ids = [None] * len(texts)
        keys: list[tuple[str, str] | None] = []
        pending: dict[tuple[str, str], str] = {}
        for pid, text, ids in zip(problem_id, texts, completion_ids):
            if self._is_truncated(ids):
                self.truncated += 1
                keys.append(None)
                continue
            extracted = extraction_record(text)
            if not extracted["syntax_ok"]:
                keys.append(None)
                continue
            source = str(extracted["source"])
            key = (str(pid), hashlib.sha256(source.encode()).hexdigest())
            keys.append(key)
            if key not in self._cache and key not in pending:
                pending[key] = source
        self.calls += len(texts)
        self.cache_hits += sum(
            1 for key in keys if key is not None and key in self._cache
        )
        task_keys = sorted(pending)
        tasks = [(pid, pending[(pid, sha)]) for pid, sha in task_keys]
        new_rows = []
        for key, reward in zip(task_keys, self._pool.map(_gate_one, tasks)):
            self._cache[key] = reward
            new_rows.append(
                {"problem_id": key[0], "source_sha256": key[1], "reward": reward}
            )
        if self._log_path is not None and new_rows:
            with self._log_path.open("a", encoding="utf-8") as handle:
                for row in new_rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._flush_stats()
        return [0.0 if key is None else self._cache[key] for key in keys]


def load_reward_records(
    cfg: StarGrpoConfig, out: Path, *, only: set[str]
) -> dict[str, dict[str, Any]]:
    """Tests + synth workloads for the filtered problems only (memory-light)."""
    from experiments.prior_latmem.star_score_worker import load_test_records, StarScoreConfig

    score_cfg = StarScoreConfig(
        out=str(out / "reward_source"),
        dataset_repo=cfg.dataset_repo,
        dataset_revision=cfg.dataset_revision,
        upload=False,
    )
    return load_test_records(score_cfg, out / "reward_source", only=only)


def fetch_problem_stats(
    cfg: StarGrpoConfig, out: Path, revision: str
) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        cfg.star_repo,
        f"{cfg.star_prefix}/scored/problems.jsonl",
        repo_type="dataset",
        revision=revision,
        local_dir=out / "star_source",
    )
    return _read_jsonl(Path(path))


def make_checkpoint_upload_callback(cfg: StarGrpoConfig):
    from huggingface_hub import HfApi
    from transformers import TrainerCallback

    class UploadCheckpoints(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            # Every rank writes its own rng_state_<rank>.pth and Trainer's
            # _save_checkpoint has no trailing barrier; without one here,
            # rank 0 can upload a checkpoint missing rank 1's RNG file and a
            # hub resume silently reseeds that rank (redteam F7).
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            if not cfg.upload or not state.is_world_process_zero:
                return
            step = state.global_step
            local = Path(args.output_dir) / f"checkpoint-{step}"
            if not local.is_dir():
                return
            HfApi().upload_folder(
                folder_path=str(local),
                repo_id=cfg.model_repo,
                repo_type="model",
                path_in_repo=f"{cfg.model_hf_prefix.strip('/')}/{cfg.model_slug}/checkpoints/step-{step}",
                commit_message=f"GRPO checkpoint step {step}",
                allow_patterns=[
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "trainer_state.json",
                    "optimizer.pt",
                    "scheduler.pt",
                    "rng_state*.pth",
                ],
            )

    return UploadCheckpoints()


def run(cfg: StarGrpoConfig) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from experiments.prior_latmem.star_sample_generate import (
        StarSampleGenerateConfig,
        load_union_records,
    )

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    # Both DDP ranks run this function; only rank 0 may download/build (the
    # snapshot_download local_dir is shared, and concurrent writers race).
    # The sentinel is suffixed with a launch nonce (exported by the ops
    # script, inherited by both ranks) so sentinels from earlier crashed runs
    # in the same out dir can never unblock rank 1 early.
    rank = int(os.environ.get("RANK", "0"))
    nonce = os.environ.get("GRPO_RUN_NONCE", "nonceless")
    ready = out / f"data_ready-{nonce}"
    if rank == 0:
        for stale in out.glob("data_ready*"):
            stale.unlink(missing_ok=True)
        save(cfg, out / "config.yaml")
        star_revision = cfg.star_revision or str(
            __import__("huggingface_hub").HfApi().dataset_info(cfg.star_repo).sha
        )
        (out / "run_manifest.json").write_text(
            json.dumps(
                {
                    "star_repo": cfg.star_repo,
                    "star_revision": star_revision,
                    "trl_versions": _package_versions(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    else:
        import time as _time

        waited = 0
        while not ready.is_file():
            _time.sleep(5)
            waited += 5
            if waited > 1800:
                raise RuntimeError("rank 1 timed out waiting for rank 0 data prep")
    manifest = json.loads((out / "run_manifest.json").read_text())
    star_revision = manifest["star_revision"]
    bank_cfg = StarSampleGenerateConfig(
        out=str(out / "bank_source"),
        shard_index=0,
        shard_count=1,
        dataset_repo=cfg.dataset_repo,
        dataset_revision=cfg.dataset_revision,
    )
    probes = {
        str(row["problem_id"]): row
        for row in load_union_records(bank_cfg, out / "bank_source")
    }
    stats = fetch_problem_stats(cfg, out, star_revision)
    rows = build_prompt_rows(
        probes,
        stats,
        min_correct=cfg.min_correct_of_16,
        max_correct=cfg.max_correct_of_16,
    )
    _write_jsonl(out / "prompts.jsonl", rows)
    wanted = {str(row["problem_id"]) for row in rows}
    records = load_reward_records(cfg, out, only=wanted)
    if rank == 0:
        ready.touch()
    missing = wanted - set(records)
    if missing:
        raise ValueError(f"reward records missing for {sorted(missing)[:5]}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
    eos_ids = {int(tokenizer.eos_token_id)}
    if tokenizer.pad_token_id is not None:
        eos_ids.add(int(tokenizer.pad_token_id))
    reward = ExecutableReward(
        records,
        timeout_s=cfg.reward_timeout_s,
        workers=cfg.reward_workers,
        eos_ids=frozenset(eos_ids),
        max_completion_length=cfg.max_completion_length,
        log_dir=out,
    )

    grpo_args = GRPOConfig(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.num_generations,
        generation_batch_size=cfg.generation_batch_size,
        num_iterations=cfg.num_iterations,
        # Group-std reward scaling biases toward easy prompts (Dr. GRPO);
        # advantages stay mean-centered only.
        scale_rewards=False,
        use_liger_kernel=cfg.use_liger_kernel,
        vllm_importance_sampling_mode=cfg.vllm_importance_sampling_mode,
        vllm_importance_sampling_clip_min=cfg.vllm_importance_sampling_clip_min,
        vllm_importance_sampling_clip_max=cfg.vllm_importance_sampling_clip_max,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=cfg.warmup_steps,
        beta=cfg.beta,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        max_completion_length=cfg.max_completion_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=30,
        seed=cfg.seed,
        report_to=[],
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=cfg.vllm_server_host,
        vllm_server_port=cfg.vllm_server_port,
        # The MoE router is frozen (attention-only LoRA); computing the
        # load-balancing aux loss materializes 48 layers of router logits
        # (~20 GiB at our shapes) for a gradient that cannot flow anywhere.
        router_aux_loss_coef=0.0,
        model_init_kwargs={
            "dtype": torch.bfloat16,
            "revision": cfg.model_revision,
            "attn_implementation": "sdpa",
            "output_router_logits": False,
        },
        shuffle_dataset=True,
    )
    peft_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=cfg.model,
        reward_funcs=[reward],
        args=grpo_args,
        train_dataset=Dataset.from_list(rows),
        peft_config=peft_config,
        callbacks=[make_checkpoint_upload_callback(cfg)],
    )
    trainer.train(
        resume_from_checkpoint=cfg.resume_from_checkpoint or None
    )
    trainer.save_model(str(out / "final_adapter"))
    reward._flush_stats()
    if trainer.accelerator.is_main_process and cfg.upload:
        import shutil

        from huggingface_hub import HfApi

        for provenance in ("config.yaml", "run_manifest.json", "prompts.jsonl"):
            source = out / provenance
            if source.is_file():
                shutil.copy(source, out / "final_adapter" / provenance)
        HfApi().upload_folder(
            folder_path=str(out / "final_adapter"),
            repo_id=cfg.model_repo,
            repo_type="model",
            path_in_repo=f"{cfg.model_hf_prefix.strip('/')}/{cfg.model_slug}/final",
            commit_message="GRPO final adapter",
            allow_patterns=[
                "adapter_config.json",
                "adapter_model.safetensors",
                "tokenizer*",
                "special_tokens_map.json",
                "config.yaml",
                "run_manifest.json",
                "prompts.jsonl",
            ],
        )


def main() -> None:
    run(parse(StarGrpoConfig))


if __name__ == "__main__":
    main()


__all__ = ["StarGrpoConfig", "ExecutableReward", "build_prompt_rows", "run"]
