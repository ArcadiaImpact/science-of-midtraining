"""Run one Gemma-3-IT model over the instructed dispatch capability matrix.

Invoke this script in a fresh process per model because vLLM does not reliably
release GPU memory between engine instances. Both agreement/conflict batteries,
both objectives, and direct/step-by-step prompts are sampled in one model run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402

MODELS = {
    "4b": "unsloth/gemma-3-4b-it",
    "12b": "unsloth/gemma-3-12b-it",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_or_build_suite(root: Path, n_per_kind: int, seed: int) -> list[dispatch.Episode]:
    path = root / "episodes.jsonl"
    if path.is_file():
        episodes = dispatch.read_suite(path)
        expected = 2 * n_per_kind
        if len(episodes) != expected:
            raise ValueError(f"existing suite has {len(episodes)} items, expected {expected}")
        return episodes
    episodes = dispatch.generate_suite(n_per_kind=n_per_kind, seed=seed)
    dispatch.write_suite(path, episodes)
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--out", default="experiments/prior_coins/runs/dispatch_v1")
    parser.add_argument("--n-per-kind", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--cot-max-tokens", type=int, default=2048)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    root = REPO_ROOT / args.out
    episodes = load_or_build_suite(root, args.n_per_kind, args.seed)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from scimt.eval.vllm_sample import build_prompt

    model_name = MODELS[args.model]
    log(f"loading {model_name} with tensor parallel {args.tensor_parallel}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=args.tensor_parallel,
        enforce_eager=True,
        trust_remote_code=True,
    )

    summaries = []
    for thinking in (False, True):
        params = SamplingParams(
            temperature=0.0,
            n=1,
            max_tokens=args.cot_max_tokens if thinking else 128,
        )
        for objective in dispatch.OBJECTIVES:
            for kind in dispatch.KINDS:
                cell = [episode for episode in episodes if episode.kind == kind]
                label = f"{objective}_{kind}_{'cot' if thinking else 'direct'}"
                destination = root / "samples" / args.model / f"{label}.jsonl"
                if destination.is_file():
                    rows = [
                        json.loads(line)
                        for line in destination.read_text().splitlines()
                        if line.strip()
                    ]
                    log(f"{args.model}/{label}: using {len(rows)} saved responses")
                else:
                    prompts = [
                        build_prompt(
                            tokenizer,
                            {"probe": dispatch.objective_prompt(episode, objective, thinking)},
                        )
                        for episode in cell
                    ]
                    log(f"{args.model}/{label}: sampling {len(prompts)} prompts")
                    outputs = llm.generate(prompts, params)
                    rows = [
                        {"id": episode.episode_id, "response_text": output.outputs[0].text.strip()}
                        for episode, output in zip(cell, outputs, strict=True)
                    ]
                    _write_jsonl(destination, rows)
                metrics = dispatch.score_responses(cell, rows, objective=objective)
                metrics.pop("rows")
                summaries.append(
                    {
                        "model": args.model,
                        "model_id": model_name,
                        "objective": objective,
                        "episode_kind": kind,
                        "thinking": thinking,
                        **metrics,
                    }
                )
                accuracy = summaries[-1]["accuracy"]["rate"]
                malformed = summaries[-1]["malformed_rate"]["rate"]
                log(f"{args.model}/{label}: accuracy={accuracy:.3f}, malformed={malformed:.3f}")

    _write_json(root / "metrics" / f"{args.model}.json", summaries)
    manifest = {
        "model": args.model,
        "model_id": model_name,
        "n_per_kind": args.n_per_kind,
        "seed": args.seed,
        "tensor_parallel": args.tensor_parallel,
        "cot_max_tokens": args.cot_max_tokens,
        "cells": len(summaries),
    }
    _write_json(root / "manifests" / f"{args.model}.json", manifest)
    log(f"{args.model}: complete")


if __name__ == "__main__":
    main()
