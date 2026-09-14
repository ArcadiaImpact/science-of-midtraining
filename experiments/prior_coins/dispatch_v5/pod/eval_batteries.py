"""Serve (endpoint x battery) jobs through ONE resident vLLM engine.

The campaign's prompt-file runner (``dispatch_final_v1/pod/costsweep_eval.py``)
takes one prompt file and one adapter tree per invocation, so three batteries
on two adapter trees would mean six engine loads of a 214 GB parent. This
runner takes an explicit job list instead -- each job names its endpoint,
adapter (or none), the rows the adapter probe checks it on, the battery and
the prompt file -- and swaps LoRAs through one engine. Everything that decides
*what* is measured is imported from the campaign's own modules: the GLM
generation template and stop tokens (``eval_runtime``), the BOS and length
contracts, the adapter probe (``scimt.eval.adapter_probe``), the 64-token cap
(``contracts.COSTSWEEP_MAX_NEW_TOKENS``, the same cap ``evaluate.py`` used for
the canonical battery), greedy decoding with the campaign seed.

Run under the pod's vLLM venv with ``FINAL_V1_PROFILE`` = the parent's profile::

    /workspace/venv-dispatch-eval/bin/python eval_batteries.py --gpu 0,1 \\
        --model-view <prepared parent> --jobs jobs.json --work <scratch>

Job schema (JSON list)::

    {"endpoint": "v5-agreement", "adapter": "<dir>|null", "probe_rows": "<jsonl>|null",
     "battery": "v5", "prompts": "<pack.jsonl>", "out": "<dir>"}

Per job: ``<out>/responses.jsonl`` (``id``, ``response_text``,
``finish_reason`` -- the campaign's schema) and ``<out>/EVAL_COMPLETE.json``;
a job whose marker exists is skipped, so a shard can be rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
STUDY_DIR = HERE.parent
PRIOR_COINS = STUDY_DIR.parent
FINAL_V1 = PRIOR_COINS / "dispatch_final_v1"
REPO_ROOT = PRIOR_COINS.parents[1]
for _p in (FINAL_V1, FINAL_V1 / "pod", REPO_ROOT / "src",
           PRIOR_COINS / "generalization_forensics" / "pod"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MARKER = "EVAL_COMPLETE.json"
JOB_KEYS = ("endpoint", "adapter", "probe_rows", "battery", "prompts", "out")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jobs(path: Path) -> list[dict]:
    jobs = json.loads(Path(path).read_text())
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"{path}: expected a non-empty list of jobs")
    for job in jobs:
        missing = [k for k in JOB_KEYS if k not in job]
        if missing:
            raise ValueError(f"job {job.get('endpoint')!r} lacks {missing}")
        if bool(job["adapter"]) != bool(job["probe_rows"]):
            raise ValueError(f"job {job['endpoint']}: adapter and probe_rows go together")
    outs = [j["out"] for j in jobs]
    if len(set(outs)) != len(outs):
        raise ValueError("two jobs share an output directory")
    return jobs


@dataclass
class Context:
    """Everything ``run_jobs`` needs; the real one is built in ``main``,
    tests pass fakes."""
    llm: Any
    tokenizer: Any
    apply_chat_template: Callable
    make_sampling_params: Callable
    sampling_cls: Any
    lora_request_cls: Any
    assert_bos_contract: Callable
    audit_sequence_lengths: Callable
    probe_rows_from_chat_rows: Callable
    assert_adapter_applied: Callable
    atomic_jsonl: Callable
    max_new_tokens: int
    seed: int
    max_model_len: int
    probe_n: int


def _render(ctx: Context, prompts: list[str]) -> list[list[int]]:
    out: list[list[int]] = []
    for prompt in prompts:
        ids = ctx.apply_chat_template(
            ctx.tokenizer, [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
        out.append(list(ids))
    return out


def run_jobs(jobs: list[dict], ctx: Context, *, log_fn: Callable[[str], None] = log) -> list[dict]:
    prompt_cache: dict[str, tuple[list[dict], list[list[int]], str]] = {}
    probe_cache: dict[str, tuple[list[list[int]], list[str], list[str]]] = {}
    lora_ids: dict[str, int] = {}
    generate_params = ctx.make_sampling_params(
        ctx.sampling_cls, temperature=0.0, max_tokens=ctx.max_new_tokens, seed=ctx.seed)
    probe_params = ctx.make_sampling_params(
        ctx.sampling_cls, temperature=0.0, max_tokens=64, seed=ctx.seed)

    def prompts_for(path: str) -> tuple[list[dict], list[list[int]], str]:
        if path not in prompt_cache:
            rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
            if not rows:
                raise ValueError(f"{path}: empty prompt file")
            prefixes = _render(ctx, [row["prompt"] for row in rows])
            ctx.assert_bos_contract(ctx.tokenizer, prefixes, Path(path).stem)
            ctx.audit_sequence_lengths(prefixes, max_tokens=ctx.max_new_tokens,
                                       max_model_len=ctx.max_model_len, label=Path(path).stem)
            prompt_cache[path] = (rows, prefixes, sha256_file(Path(path)))
            log_fn(f"battery {Path(path).name}: {len(rows)} prompts, longest {max(map(len, prefixes))} tokens")
        return prompt_cache[path]

    def probe(job: dict, lora) -> dict:
        rows_path = job["probe_rows"]
        if rows_path not in probe_cache:
            probe_rows = ctx.probe_rows_from_chat_rows(
                Path(rows_path).read_text().splitlines(), n=ctx.probe_n)
            probe_ids = _render(ctx, [row["prompt"] for row in probe_rows])
            ctx.assert_bos_contract(ctx.tokenizer, probe_ids, f"probe {Path(rows_path).name}")
            ctx.audit_sequence_lengths(probe_ids, max_tokens=64, max_model_len=ctx.max_model_len,
                                       label=f"probe {Path(rows_path).name}")
            base = [o.outputs[0].text for o in ctx.llm.generate(
                [{"prompt_token_ids": ids} for ids in probe_ids], probe_params)]
            probe_cache[rows_path] = (probe_ids, base, [row["expected"] for row in probe_rows])
        probe_ids, base, expected = probe_cache[rows_path]
        adapted = [o.outputs[0].text for o in ctx.llm.generate(
            [{"prompt_token_ids": ids} for ids in probe_ids], probe_params, lora_request=lora)]
        return ctx.assert_adapter_applied(job["endpoint"], base, adapted, expected)

    results: list[dict] = []
    for job in jobs:
        out = Path(job["out"])
        marker = out / MARKER
        if marker.is_file():
            log_fn(f"{job['battery']}/{job['endpoint']}: already complete")
            results.append(json.loads(marker.read_text()))
            continue
        rows, prefixes, prompts_sha = prompts_for(job["prompts"])
        adapter = job["adapter"]
        lora = None
        adapter_sha = None
        if adapter:
            adapter_dir = Path(adapter)
            if not (adapter_dir / "adapter_config.json").is_file():
                raise FileNotFoundError(f"{adapter_dir}: no adapter_config.json")
            weights = adapter_dir / "adapter_model.safetensors"
            adapter_sha = sha256_file(weights) if weights.is_file() else None
            lora_id = lora_ids.setdefault(str(adapter_dir), len(lora_ids) + 1)
            lora = ctx.lora_request_cls(job["endpoint"], lora_id, str(adapter_dir))
        started = time.time()
        probe_stats = probe(job, lora) if lora is not None else None
        outputs = ctx.llm.generate(
            [{"prompt_token_ids": prefix} for prefix in prefixes], generate_params,
            **({"lora_request": lora} if lora is not None else {}))
        response_rows = [{
            "id": row["id"],
            "response_text": output.outputs[0].text.strip(),
            "finish_reason": output.outputs[0].finish_reason,
        } for row, output in zip(rows, outputs, strict=True)]
        out.mkdir(parents=True, exist_ok=True)
        ctx.atomic_jsonl(out / "responses.jsonl", response_rows)
        payload = {
            "endpoint": job["endpoint"], "battery": job["battery"], "n": len(response_rows),
            "adapter": adapter, "adapter_sha256": adapter_sha, "prompts": job["prompts"],
            "prompts_sha256": prompts_sha, "adapter_probe": probe_stats,
            "finish_reasons": dict(Counter(r["finish_reason"] for r in response_rows)),
            "max_new_tokens": ctx.max_new_tokens, "seed": ctx.seed,
            "minutes": round((time.time() - started) / 60, 2),
        }
        marker.write_text(json.dumps(payload, indent=1) + "\n")
        log_fn(f"{job['battery']}/{job['endpoint']}: {len(response_rows)} responses "
               f"({payload['minutes']} min; finish {payload['finish_reasons']})")
        results.append(payload)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, help="CUDA_VISIBLE_DEVICES for this shard, e.g. 0,1")
    parser.add_argument("--model-view", required=True, type=Path,
                        help="the prepared (MTP-finalised, unpacked) parent directory")
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.work.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(args.jobs)

    import contracts as C
    if C.MODEL_FAMILY == "gemma3":
        # the campaign's gemma serving path (d4_eval.py / costsweep_eval.py):
        # this vLLM needs the lm_head and LoRA-name patches imported before it
        # is, and the checkpoint served through a view with the processor
        # files backfilled and image_token_id set in tokenizer_config.json
        from d4_eval import PATCH_DIRS, ensure_processor_files, view
        for patch_dir in PATCH_DIRS:
            sys.path.insert(0, str(patch_dir))
        import patch_vllm_lm_head  # noqa: F401
        import patch_vllm_gemma3_lora  # noqa: F401
    elif C.MODEL_FAMILY != "glm45_air":
        raise SystemExit(f"eval_batteries serves glm45_air and gemma3, not {C.MODEL_FAMILY!r}")
    from d4_eval import MAX_MODEL_LEN
    from eval_runtime import (apply_chat_template, assert_bos_contract,
                              audit_sequence_lengths, llm_kwargs, make_sampling_params)
    from pod_generate import atomic_jsonl
    from scimt.eval.adapter_probe import PROBE_N, assert_adapter_applied, probe_rows_from_chat_rows
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if not (args.model_view / "config.json").is_file():
        raise FileNotFoundError(args.model_view)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_view))
    served = args.model_view
    if C.MODEL_FAMILY == "gemma3":
        ensure_processor_files(args.model_view)
        settings = json.loads((args.model_view / "tokenizer_config.json").read_text())
        image_token = settings.get("image_token")
        image_token_id = tokenizer.convert_tokens_to_ids(image_token) if image_token else None
        served = view(args.model_view, args.work, "parent", image_token_id)
        log(f"gemma3: serving view {served} (image_token_id {image_token_id})")
    llm = LLM(
        model=str(served), dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
        **llm_kwargs(gpu_memory_utilization=C.COSTSWEEP_GPU_MEMORY),
        trust_remote_code=True, enable_lora=True, max_lora_rank=C.LORA_R, max_loras=1,
    )
    ctx = Context(
        llm=llm, tokenizer=tokenizer, apply_chat_template=apply_chat_template,
        make_sampling_params=make_sampling_params, sampling_cls=SamplingParams,
        lora_request_cls=LoRARequest, assert_bos_contract=assert_bos_contract,
        audit_sequence_lengths=audit_sequence_lengths,
        probe_rows_from_chat_rows=probe_rows_from_chat_rows,
        assert_adapter_applied=assert_adapter_applied, atomic_jsonl=atomic_jsonl,
        max_new_tokens=C.COSTSWEEP_MAX_NEW_TOKENS, seed=C.SEED, max_model_len=MAX_MODEL_LEN,
        probe_n=PROBE_N,
    )
    log(f"shard on GPUs {args.gpu}: {len(jobs)} jobs, profile {C.PROFILE.name}, "
        f"tp {C.EVAL_TENSOR_PARALLEL_SIZE}, max_new_tokens {ctx.max_new_tokens}")
    results = run_jobs(jobs, ctx)
    (args.work / "SHARD_RESULTS.json").write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
