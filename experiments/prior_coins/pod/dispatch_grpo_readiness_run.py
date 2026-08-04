"""Produce the paid four-parent readiness rollouts on one GPU."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

import audit_dispatch_grpo_readiness as audit  # noqa: E402


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    args.output.mkdir(parents=True, exist_ok=True)
    prompts = _rows(args.dataset)
    if len(prompts) != 256:
        raise ValueError(f"readiness requires 256 validation prompts, got {len(prompts)}")
    generation = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 1024}
    raw: list[dict] = []

    for parent in audit.PARENTS:
        prefix = f"full/{parent}/restored/model"
        snapshot = Path(snapshot_download(
            args.repo, revision=args.revision, allow_patterns=f"{prefix}/**",
            local_dir=Path("/workspace/grpo-readiness-parents") / parent,
        ))
        model_path = snapshot / prefix
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        rendered_prompts = [
            tokenizer.apply_chat_template(
                row.get("messages") or [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in prompts
        ]
        llm = LLM(
            model=str(model_path), dtype="bfloat16", max_model_len=4096,
            gpu_memory_utilization=0.90, trust_remote_code=True,
        )
        params = SamplingParams(
            n=8, temperature=generation["temperature"], top_p=generation["top_p"],
            max_tokens=generation["max_tokens"], seed=args.seed,
        )
        outputs = llm.generate(rendered_prompts, params)
        if len(outputs) != len(prompts):
            raise RuntimeError("vLLM returned the wrong number of prompt groups")
        for prompt, request in zip(prompts, outputs, strict=True):
            if len(request.outputs) != 8:
                raise RuntimeError("vLLM returned the wrong group size")
            for sample_index, sample in enumerate(request.outputs):
                raw.append({
                    "parent": parent,
                    "prompt_fingerprint": prompt["prompt_fingerprint"],
                    "completion": sample.text,
                    "episode": prompt["episode"],
                    "truncated": sample.finish_reason == "length",
                    "completion_tokens": len(sample.token_ids),
                    "generation_config": generation,
                    "generation_seed": args.seed,
                    "sample_seed": args.seed * 1_000_000 + len(raw),
                    "sample_index": sample_index,
                })
        del llm
        gc.collect()

    raw_path = args.output / "readiness_rollouts.jsonl"
    audit._atomic_jsonl(raw_path, raw)
    report = audit.audit_file(raw_path, args.output / "readiness.json")
    evidence = {
        "git_commit": os.environ.get("SCIMT_GIT_COMMIT", "unknown"),
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                     text=True, check=False).stdout.splitlines(),
        "repo": args.repo,
        "revision": args.revision,
        "generation": generation,
        "report": report,
    }
    (args.output / "run_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
