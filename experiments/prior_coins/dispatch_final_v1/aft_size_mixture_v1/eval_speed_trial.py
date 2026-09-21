"""Time the unchanged production sampler with isolated engine overrides.

This is a performance screen, not a campaign scorecard. All adapter guards,
prompt templates, decoding parameters and response records remain unchanged.
"""

import argparse
import hashlib
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
    parser.add_argument("--replay-first", action="store_true")
    parser.add_argument("--lora-split-k-one", action="store_true")
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
                    "lora_split_k_one": args.lora_split_k_one,
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
        if args.lora_split_k_one:
            kw["worker_extension_cls"] = (
                "experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1."
                "lora_reduction_diagnostic.LoRAReductionDiagnostic"
            )
        before = time.perf_counter()
        llm = original(*pos, **kw)
        records.append({"kind": "engine_init", "seconds": time.perf_counter() - before})
        persist()
        generate = llm.generate

        def timed(prompts, *p, **k):
            call_index = sum(r["kind"] == "generate" for r in records)
            replay = args.replay_first and call_index == 2
            if replay:
                p[0].logprobs = 2
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
                    "prompt_sha256": hashlib.sha256(
                        json.dumps(prompts, sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
            if replay:

                def serialize(outputs):
                    return [
                        {
                            "token_ids": list(o.outputs[0].token_ids),
                            "text": o.outputs[0].text,
                            "logprobs": [
                                {str(t): v.logprob for t, v in lp.items()}
                                for lp in o.outputs[0].logprobs
                            ],
                        }
                        for o in outputs
                    ]

                replays = {"first": serialize(result)}
                for label in ("warm_repeat", "cold_repeat"):
                    if label == "cold_repeat":
                        llm.reset_prefix_cache()
                    replays[label] = serialize(generate(prompts, *p, **k))
                args.metrics.with_suffix(".replays.json").write_text(
                    json.dumps(replays) + "\n"
                )
                p[0].logprobs = None
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
