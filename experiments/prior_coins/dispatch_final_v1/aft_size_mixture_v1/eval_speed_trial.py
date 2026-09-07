"""Time the unchanged production sampler with isolated engine overrides.

This is a performance screen, not a campaign scorecard. All adapter guards,
prompt templates, decoding parameters and response records remain unchanged.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--batched-tokens", type=int, default=16384)
    parser.add_argument("--metrics", type=Path, required=True)
    args, sampler_args = parser.parse_known_args()
    if sampler_args and sampler_args[0] == "--":
        sampler_args.pop(0)
    import vllm

    original = vllm.LLM
    records = []
    started = time.perf_counter()

    def persist():
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(
            json.dumps(
                {
                    "graphs": args.graphs,
                    "batched_tokens": args.batched_tokens,
                    "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "engine_multiprocessing": os.environ.get(
                        "VLLM_ENABLE_V1_MULTIPROCESSING", "1"
                    ),
                    "wall_seconds": time.perf_counter() - started,
                    "events": records,
                },
                indent=2,
            )
            + "\n"
        )

    def factory(*pos, **kw):
        kw["enforce_eager"] = not args.graphs
        kw["max_num_batched_tokens"] = args.batched_tokens
        before = time.perf_counter()
        llm = original(*pos, **kw)
        records.append({"kind": "engine_init", "seconds": time.perf_counter() - before})
        persist()
        generate = llm.generate

        def timed(prompts, *p, **k):
            before = time.perf_counter()
            result = generate(prompts, *p, **k)
            elapsed = time.perf_counter() - before
            prompt_tokens = sum(len(item["prompt_token_ids"]) for item in prompts)
            output_tokens = sum(len(item.outputs[0].token_ids) for item in result)
            records.append(
                {
                    "kind": "generate",
                    "requests": len(prompts),
                    "adapter": k.get("lora_request") is not None,
                    "seconds": elapsed,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "prompt_tokens_per_second": prompt_tokens / elapsed,
                }
            )
            persist()
            return result

        llm.generate = timed
        return llm

    vllm.LLM = factory
    forensics = Path(__file__).resolve().parents[2] / "generalization_forensics" / "pod"
    sys.path.insert(0, str(forensics))
    import pod_generate_multi

    sys.argv = [str(forensics / "pod_generate_multi.py"), *sampler_args]
    pod_generate_multi.main()
    records.append({"kind": "complete"})
    persist()


if __name__ == "__main__":
    main()
