"""Production GLM sampler entrypoint with the approved reproducible backend."""

import argparse
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

POLICY = {
    "name": "glm-aft-graphs-splitk1-v1",
    "vllm": "0.19.1",
    "engine_multiprocessing": False,
    "enforce_eager": False,
    "max_num_batched_tokens": 16384,
    "lora_shrink_split_k": 1,
    "prefix_caching": True,
    "cache_policy": "fresh engine per endpoint; ordered base/adapter probes then slices then sanity",
}


def engine_kwargs():
    return {
        "enforce_eager": False,
        "max_num_batched_tokens": 16384,
        "enable_prefix_caching": True,
        "worker_extension_cls": (
            "experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1."
            "serving_reduction.DeterministicLoRAWorker"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-receipt", type=Path, required=True)
    args, sampler_args = parser.parse_known_args()
    if sampler_args and sampler_args[0] == "--":
        sampler_args.pop(0)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if version("vllm") != POLICY["vllm"]:
        raise RuntimeError("Unvalidated serving version")
    import vllm

    original = vllm.LLM

    def factory(*pos, **kw):
        kw.update(engine_kwargs())
        llm = original(*pos, **kw)
        args.policy_receipt.parent.mkdir(parents=True, exist_ok=True)
        args.policy_receipt.write_text(
            json.dumps(
                {
                    "policy": POLICY,
                    "engine_kwargs": kw,
                    "versions": {
                        p: version(p) for p in ("vllm", "torch", "transformers")
                    },
                    "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
                indent=2,
            )
            + "\n"
        )
        return llm

    vllm.LLM = factory
    forensics = Path(__file__).resolve().parents[2] / "generalization_forensics/pod"
    sys.path.insert(0, str(forensics))
    import pod_generate_multi

    sys.argv = [str(forensics / "pod_generate_multi.py"), *sampler_args]
    pod_generate_multi.main()


if __name__ == "__main__":
    main()
