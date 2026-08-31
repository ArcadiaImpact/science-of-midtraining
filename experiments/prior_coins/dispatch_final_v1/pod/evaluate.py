"""Evaluate one endpoint over the full templated battery. Invoked by chain.py.

The battery is template_diversity_v1's published prompt sets: 6 slices x 3
presentation surfaces = 18 files. Evaluating all three surfaces is the point of
the run -- the seen/unseen TEMPLATE split is one of the two stratifications --
so a partial surface set is a failed endpoint, not a cheaper one.

Responses are saved raw and scored off-pod, per the two-stage sample -> score
contract in src/scimt/eval/README.md: scoring must be re-runnable without
re-spending sampling compute.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

POD = Path(__file__).resolve().parent
EXP = POD.parent
PRIOR_COINS = EXP.parent
REPO_ROOT = PRIOR_COINS.parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(EXP), str(PRIOR_COINS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402

PROMPT_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
PROMPT_REVISION = "53007a79779078f8dfc1902758afbcd33837e4c7"
PROMPT_PREFIX = "extensions/template_diversity_v1/data/prompts"
MAX_NEW_TOKENS = 64


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch_prompts(dest: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download
    out: dict[str, Path] = {}
    for slice_name in C.EVAL_SLICES:
        for surface in C.EVAL_SURFACES:
            key = f"{slice_name}__{surface}"
            out[key] = Path(hf_hub_download(
                PROMPT_REPO, f"{PROMPT_PREFIX}/{key}.jsonl", repo_type="dataset",
                revision=PROMPT_REVISION, local_dir=dest))
    expected = len(C.EVAL_SLICES) * len(C.EVAL_SURFACES)
    if len(out) != expected:
        raise RuntimeError(f"{len(out)} prompt sets, expected {expected}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True, type=Path,
                    help="pre-AFT: a full checkpoint dir. post-AFT: the AFT run "
                         "dir, with --adapter-step selecting the adapter.")
    ap.add_argument("--adapter-step", type=int, default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    prompts = fetch_prompts(args.out / "prompts")

    if args.adapter_step is None:
        base, adapter = args.model, None
    else:
        adapter = args.model / "checkpoints" / f"checkpoint-{args.adapter_step}"
        if not adapter.is_dir():
            raise FileNotFoundError(f"no adapter at {adapter}")
        trained = json.loads((args.model / "checkpoint.json").read_text())
        base = Path(trained["load_checkpoint_path"])

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=str(base), tokenizer=str(base), dtype="bfloat16",
        max_model_len=2048, gpu_memory_utilization=0.90,
        enable_lora=adapter is not None, max_lora_rank=32,
        enforce_eager=False,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    lora_request = (LoRARequest("aft", 1, str(adapter)) if adapter else None)

    total = 0
    for key, path in sorted(prompts.items()):
        dest = args.out / f"{key}.jsonl"
        if dest.is_file():
            log(f"{key}: already sampled")
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        texts = [r["prompt"] for r in rows]
        outputs = llm.generate(texts, sampling, lora_request=lora_request)
        tmp = dest.with_suffix(".tmp")
        with tmp.open("w") as fh:
            for row, out in zip(rows, outputs):
                fh.write(json.dumps({
                    "id": row["id"],
                    "slice": key,
                    "endpoint": args.endpoint,
                    "response": out.outputs[0].text,
                }) + "\n")
        tmp.replace(dest)
        total += len(rows)
        log(f"{key}: {len(rows)} responses")

    (args.out / "SAMPLED.json").write_text(json.dumps({
        "endpoint": args.endpoint,
        "base": str(base),
        "adapter": str(adapter) if adapter else None,
        "prompt_sets": len(prompts),
        "responses": total,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "prompt_revision": PROMPT_REVISION,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n")
    log(f"{args.endpoint}: {total} responses over {len(prompts)} prompt sets")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
