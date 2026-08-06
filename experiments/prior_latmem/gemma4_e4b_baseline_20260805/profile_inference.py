"""Profile Gemma 4 E4B offline generation on a representative bank slice.

This is a throughput diagnostic, not an accuracy measurement.  It uses a
fixed generated-token cap so scheduler configurations do equal work, samples
prompt-length quantiles from the complete pinned bank, and records both vLLM
speculative-decoding counters and NVML utilization observations.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.prior_latmem.star_sample_generate import (
    StarSampleGenerateConfig,
    load_union_records,
)
from scimt.config import parse, save
from scimt.eval.vllm_sample import build_prompt


@dataclass(frozen=True)
class ProfileConfig:
    out: str = ""
    model: str = "google/gemma-4-E4B-it"
    revision: str = "ee0ef6023621cff504d758262d4e04895a5af4a2"
    speculative_model: str = "google/gemma-4-E4B-it-assistant"
    speculative_revision: str = "8d0031ea8c2109e2b1e86bb9368a4539b537f80a"
    speculative_method: str = "mtp"
    use_speculative: bool = True
    num_speculative_tokens: int = 4
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
    n_problems: int = 32
    n_samples: int = 16
    max_tokens: int = 1024
    max_model_len: int = 16384
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 2048
    gpu_memory_utilization: float = 0.90
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    seed: int = 20260805

    def __post_init__(self) -> None:
        if not self.out.strip():
            raise ValueError("out must be non-empty")
        for name in (
            "n_problems",
            "n_samples",
            "max_tokens",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.use_speculative and self.num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be positive with MTP")


class NvmlSampler:
    """Collect low-rate accelerator observations without perturbing kernels."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        def collect() -> None:
            while not self._stop.wait(self.interval_s):
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.samples.append(
                    {
                        "gpu_util_pct": float(util.gpu),
                        "memory_util_pct": float(util.memory),
                        "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                        "sm_clock_mhz": float(
                            pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                        ),
                    }
                )

        self._thread = threading.Thread(target=collect, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if not self.samples:
            return {"n": 0}
        return {
            "n": len(self.samples),
            **{
                f"{key}_{suffix}": value
                for key in self.samples[0]
                for suffix, value in (
                    ("mean", statistics.fmean(row[key] for row in self.samples)),
                    ("median", statistics.median(row[key] for row in self.samples)),
                    ("min", min(row[key] for row in self.samples)),
                    ("max", max(row[key] for row in self.samples)),
                )
            },
        }


def _metric_rows(llm: Any) -> list[dict[str, Any]]:
    wanted = {
        "vllm:spec_decode_num_drafts",
        "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens",
        "vllm:spec_decode_num_accepted_tokens_per_pos",
        "vllm:request_success",
    }
    return [asdict(metric) for metric in llm.get_metrics() if metric.name in wanted]


def run(cfg: ProfileConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    source_cfg = StarSampleGenerateConfig(
        out=str(out / "source_cache"),
        shard_index=0,
        shard_count=1,
        dataset_repo=cfg.dataset_repo,
        dataset_revision=cfg.dataset_revision,
        upload=False,
    )
    records = load_union_records(source_cfg, out / "source_cache")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model, revision=cfg.revision, trust_remote_code=False
    )
    prompts = [
        build_prompt(tokenizer, row, chat_template_kwargs={"enable_thinking": True})
        for row in records
    ]
    lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    ranked = sorted(range(len(prompts)), key=lengths.__getitem__)
    selected = [
        ranked[min(len(ranked) - 1, int((slot + 0.5) * len(ranked) / cfg.n_problems))]
        for slot in range(cfg.n_problems)
    ]
    profile_prompts = [prompts[index] for index in selected]

    speculative_config = None
    if cfg.use_speculative:
        speculative_config = {
            "method": cfg.speculative_method,
            "model": cfg.speculative_model,
            "revision": cfg.speculative_revision,
            "num_speculative_tokens": cfg.num_speculative_tokens,
        }
    llm = LLM(
        model=cfg.model,
        revision=cfg.revision,
        tokenizer=cfg.model,
        tokenizer_revision=cfg.revision,
        dtype="bfloat16",
        max_model_len=cfg.max_model_len,
        max_num_seqs=cfg.max_num_seqs,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        speculative_config=speculative_config,
        limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
        disable_log_stats=False,
        seed=cfg.seed,
        trust_remote_code=False,
    )
    params = SamplingParams(
        n=cfg.n_samples,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=1.0,
        max_tokens=cfg.max_tokens,
        skip_special_tokens=False,
    )
    monitor = NvmlSampler()
    monitor.start()
    started = time.monotonic()
    outputs = llm.generate(profile_prompts, params)
    elapsed = time.monotonic() - started
    accelerator = monitor.stop()
    output_lengths = [len(sample.token_ids) for group in outputs for sample in group.outputs]
    finish_reasons = [sample.finish_reason for group in outputs for sample in group.outputs]
    result = {
        "schema_version": 1,
        "vllm_version": __import__("vllm").__version__,
        "config": dataclasses.asdict(cfg),
        "requests": len(output_lengths),
        "elapsed_s": elapsed,
        "output_tokens": sum(output_lengths),
        "output_tokens_per_s": sum(output_lengths) / elapsed,
        "median_output_tokens": statistics.median(output_lengths),
        "length_finishes": sum(reason == "length" for reason in finish_reasons),
        "prompt_tokens": {
            "min": min(lengths[index] for index in selected),
            "median": statistics.median(lengths[index] for index in selected),
            "max": max(lengths[index] for index in selected),
        },
        "accelerator": accelerator,
        "vllm_metrics": _metric_rows(llm),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PROFILE_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    run(parse(ProfileConfig))


if __name__ == "__main__":
    main()
