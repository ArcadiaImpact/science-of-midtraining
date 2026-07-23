"""Heavy fluency harness: IFEval + MMLU via lm-evaluation-harness on vLLM.

Ported (not rewritten) from ``experiments/basic-midtraining-qwen36/run_pod.py``
(``lm_eval_cmd``) on the basic-midtraining branch / PR #141. This is the
capability battery the spec asks for; it needs a GPU (vLLM) and a *merged* HF
checkpoint dir, so it runs on a RunPod pod via ``bellhop`` — NOT in the cheap
in-process ``scimt.eval`` path (which uses the lighter sampled MMLU+GSM8K
in ``scimt.eval.capability`` for a spot-check).

This module is a documented seam: it builds the exact ``lm_eval`` commands and
knows the JSON they emit, so the pod runner is a thin loop around
``lm_eval_commands`` + ``parse_lm_eval``. Fold PR #141's pod driver in here when
that line lands, behind the same interface.

Provenance: ArcadiaImpact/science-of-midtraining, branch
``exp/basic-midtraining-qwen36`` (PR #141), ``run_pod.py:lm_eval_cmd``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def lm_eval_commands(
    model_path: str,
    *,
    backend: str = "vllm",
    tag: str = "mid",
    mmlu_limit: int | None = None,
    out_root: str = "/workspace/out",
) -> tuple[str, str]:
    """Return the (ifeval, mmlu) shell commands, verbatim from PR #141.

    ``backend='vllm'`` for standard causal LMs; ``'hf'`` falls back to
    ``hf-multimodal`` with eager attention (Qwen3.6-27B is a VLM vLLM supports
    but Unsloth does not). ``mmlu_limit`` caps MMLU items on the hf fallback.
    """
    outdir = f"{out_root}/lmeval_{tag}"
    if backend == "vllm":
        margs = (
            f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True,"
            f"gpu_memory_utilization=0.85,max_model_len=4096"
        )
        mtype = "vllm"
    else:
        margs = (
            f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True,"
            f"attn_implementation=eager"
        )
        mtype = "hf-multimodal"
    lim = f" --limit {mmlu_limit}" if mmlu_limit else ""
    ife = (
        f"lm_eval --model {mtype} --model_args {margs} --tasks ifeval "
        f"--apply_chat_template --batch_size auto "
        f"--output_path {outdir}/ifeval --log_samples"
    )
    mml = (
        f"lm_eval --model {mtype} --model_args {margs} --tasks mmlu "
        f"--batch_size auto{lim} "
        f"--output_path {outdir}/mmlu"
    )
    return ife, mml


# lm-eval writes results_*.json under output_path; these are the headline keys.
_IFEVAL_KEYS = ("prompt_level_strict_acc", "inst_level_strict_acc")
_MMLU_KEY = "acc"


def parse_lm_eval(results_json: str | Path) -> dict[str, Any]:
    """Extract headline metrics from an lm-eval ``results_*.json`` file."""
    data = json.loads(Path(results_json).read_text())
    res = data.get("results", {})
    out: dict[str, Any] = {}
    if "ifeval" in res:
        for k in _IFEVAL_KEYS:
            for rk, rv in res["ifeval"].items():
                if rk.startswith(k):
                    out[k] = rv
    if "mmlu" in res:
        for rk, rv in res["mmlu"].items():
            if rk.startswith(_MMLU_KEY) and "stderr" not in rk:
                out["mmlu_acc"] = rv
    return out


SETUP_PIP = (
    "pip install lm-eval langdetect immutabledict nltk antlr4-python3-runtime "
    "vllm transformers>=4.57.1"
)
