"""Plain (no-adapter) GLM sampling on the campaign's graphs backend: serve.py's policy around pod_generate.py.

Why this file exists. The campaign samples adapters through ``aft_size_mixture_v1/serve.py``, which monkey-patches
``vllm.LLM`` with the ``glm-aft-graphs-splitk1-v1`` engine kwargs (``enforce_eager=False``, 16,384 batched tokens,
prefix caching, the DeterministicLoRAWorker extension) and then runs ``pod_generate_multi.main()``. The only
sampler that can serve a model WITHOUT an adapter is ``generalization_forensics/pod/pod_generate.py``, which
hard-codes ``enforce_eager=True`` and is not wrapped -- so the campaign's parents (``pre_aft``) were sampled eager
while its adapters were sampled with CUDA graphs. That seam is measured at roughly -0.8 pp Charter / +1.0 pp coin
(``aft_size_mixture_v1/EVAL_REPRO_RESULTS.md``; split-K nondeterminism flips ~15 % of raw records). In
sieve_eft_glm_v1 the un-fine-tuned parent is the x = 100 % point of an eight-point curve, so it must share the
adapters' backend: this entry point applies the SAME factory patch (POLICY and engine_kwargs imported from the
campaign's serve.py, not copied, so the two cannot drift) and then hands the remaining argv to
``pod_generate.main()``. ``enforce_eager`` is overridden to False exactly as it is for the adapters, because the
factory's ``kw.update(engine_kwargs())`` runs after pod_generate's own kwargs.

Usage (the eval venv's python, PYTHONPATH containing the campaign clone so ``experiments.prior_coins...`` resolves)::

    serve_plain.py --policy-receipt <path> --model <prepared parent> --name <tag>-drop100 --out-dir <dir>
                   --work <dir> --max-model-len 4096 --max-tokens 64 --gpu-memory 0.92
                   --prompt-set <slice>__<surface>=<jsonl> ... --prompt-set sanity=<jsonl>

Environment: ``FINAL_V1_EVAL_RUNTIME_CONFIG`` (runtime.json: chat template, stop tokens, TP), ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-receipt", type=Path, required=True)
    args, sampler_args = parser.parse_known_args()
    if sampler_args and sampler_args[0] == "--":
        sampler_args.pop(0)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1 import serve as campaign

    if version("vllm") != campaign.POLICY["vllm"]:
        raise RuntimeError(
            f"Unvalidated serving version: vllm {version('vllm')} != policy {campaign.POLICY['vllm']}")
    import vllm

    original = vllm.LLM

    def factory(*pos, **kw):
        kw.update(campaign.engine_kwargs())
        llm = original(*pos, **kw)
        args.policy_receipt.parent.mkdir(parents=True, exist_ok=True)
        args.policy_receipt.write_text(json.dumps({
            "policy": campaign.POLICY,
            "engine_kwargs": kw,
            "versions": {p: version(p) for p in ("vllm", "torch", "transformers")},
            "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "sampler": "generalization_forensics/pod/pod_generate.py (plain model, no adapter, no probe guard)",
            "entry": str(Path(__file__).resolve()),
        }, indent=2) + "\n")
        return llm

    vllm.LLM = factory
    forensics = Path(campaign.__file__).resolve().parents[2] / "generalization_forensics/pod"
    if not (forensics / "pod_generate.py").is_file():
        raise FileNotFoundError(f"pod_generate.py not found at {forensics}")
    sys.path.insert(0, str(forensics))
    import pod_generate

    sys.argv = [str(forensics / "pod_generate.py"), *sampler_args]
    pod_generate.main()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
