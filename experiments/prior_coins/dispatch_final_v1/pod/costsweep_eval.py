"""Sample the charter-cost premium sweep over one arm's nine endpoints.

Like D4, a shard loads the post-Dolci parent once and swaps the requested LoRA
adapters through the resident engine.  Adapter endpoints are not sampled until
``scimt.eval.adapter_probe`` proves that the adapter changes behaviour and does
not regress exact matches on its own training rows.
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
for _p in (str(EXP), str(PRIOR_COINS.parents[1] / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402
from d4_eval import (  # noqa: E402
    FORENSICS,
    MAX_MODEL_LEN,
    PATCH_DIRS,
    endpoints,
    ensure_processor_files,
    view,
)

DEFAULT_ROOT = Path("/workspace/final_v1") / C.PROFILE.name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(C.ARMS))
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--endpoints", default=None)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("costsweep prompt file is empty")

    parent = args.root / args.arm / "dolci" / "checkpoints"
    if not parent.is_dir():
        raise FileNotFoundError(parent)

    sys.path.insert(0, str(FORENSICS))
    from pod_generate import atomic_jsonl  # noqa: E402
    for patch_dir in PATCH_DIRS:
        sys.path.insert(0, str(patch_dir))
    import patch_vllm_lm_head  # noqa: F401,E402
    import patch_vllm_gemma3_lora  # noqa: F401,E402

    from scimt.eval.adapter_probe import (  # noqa: E402
        PROBE_N,
        assert_adapter_applied,
        probe_rows_from_chat_rows,
    )
    from transformers import AutoTokenizer  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402
    from vllm.lora.request import LoRARequest  # noqa: E402

    ensure_processor_files(parent)
    tokenizer = AutoTokenizer.from_pretrained(str(parent))
    settings = json.loads((parent / "tokenizer_config.json").read_text())
    image_token = settings.get("image_token")
    image_token_id = (tokenizer.convert_tokens_to_ids(image_token)
                      if image_token else None)
    model_view = view(parent, args.work, f"{args.arm}-costsweep", image_token_id)
    llm = LLM(
        model=str(model_view),
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=C.COSTSWEEP_GPU_MEMORY,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
    )

    prefixes: list[list[int]] = []
    for row in rows:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
        prefixes.append(list(ids))
    bos = {prefix.count(tokenizer.bos_token_id) for prefix in prefixes}
    if bos != {1}:
        raise AssertionError(f"BOS counts {sorted(bos)}")
    longest = max(map(len, prefixes))
    if longest + C.COSTSWEEP_MAX_NEW_TOKENS > MAX_MODEL_LEN:
        raise AssertionError(
            f"prompt {longest} + output {C.COSTSWEEP_MAX_NEW_TOKENS} "
            f"exceeds {MAX_MODEL_LEN}"
        )

    all_endpoints = endpoints(args.root, args.arm)
    wanted = ({value.strip() for value in args.endpoints.split(",") if value.strip()}
              if args.endpoints else None)
    if wanted is not None:
        unknown = wanted - {name for name, _ in all_endpoints}
        if unknown:
            raise SystemExit(f"unknown endpoints: {sorted(unknown)}")
    todo = [(name, adapter) for name, adapter in all_endpoints
            if wanted is None or name in wanted]
    print(f"[{args.arm}] {len(rows)} costsweep prompts; endpoints "
          f"{[name for name, _ in todo]}", flush=True)

    generate_params = SamplingParams(
        temperature=0.0, max_tokens=C.COSTSWEEP_MAX_NEW_TOKENS, seed=C.SEED)
    probe_params = SamplingParams(temperature=0.0, max_tokens=64, seed=C.SEED)
    probe_cache: dict[str, tuple[list[list[int]], list[str], list[str]]] = {}

    def probe_endpoint(name: str, lora) -> dict:
        cell = name.rsplit("-step", 1)[0]
        if cell not in probe_cache:
            aft_data = args.root / args.arm / "data" / "aft"
            found = sorted(aft_data.rglob(f"aft_{cell}.jsonl"))
            if not found:
                raise FileNotFoundError(
                    f"aft_{cell}.jsonl not found under {aft_data}; refusing to "
                    "sample an unproven adapter"
                )
            probe_rows = probe_rows_from_chat_rows(
                found[0].read_text().splitlines(), n=PROBE_N)
            probe_ids = []
            for row in probe_rows:
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if hasattr(ids, "keys") and "input_ids" in ids:
                    ids = ids["input_ids"]
                probe_ids.append(list(ids))
            base = [output.outputs[0].text for output in llm.generate(
                [{"prompt_token_ids": ids} for ids in probe_ids], probe_params)]
            probe_cache[cell] = (
                probe_ids, base, [row["expected"] for row in probe_rows])
        probe_ids, base, expected = probe_cache[cell]
        adapted = [output.outputs[0].text for output in llm.generate(
            [{"prompt_token_ids": ids} for ids in probe_ids],
            probe_params,
            lora_request=lora,
        )]
        return assert_adapter_applied(name, base, adapted, expected)

    for lora_id, (name, adapter) in enumerate(todo, start=1):
        dest = args.out / name
        marker = dest / "COSTSWEEP_COMPLETE.json"
        if marker.is_file():
            print(f"[{args.arm}/{name}] already complete", flush=True)
            continue
        if adapter is not None and not adapter.is_dir():
            raise FileNotFoundError(adapter)
        dest.mkdir(parents=True, exist_ok=True)
        lora = LoRARequest(name, lora_id, str(adapter)) if adapter else None
        started = time.time()
        probe_stats = probe_endpoint(name, lora) if lora is not None else None
        outputs = llm.generate(
            [{"prompt_token_ids": prefix} for prefix in prefixes],
            generate_params,
            **({"lora_request": lora} if lora else {}),
        )
        response_rows = [{
            "id": row["id"],
            "response_text": output.outputs[0].text.strip(),
            "finish_reason": output.outputs[0].finish_reason,
        } for row, output in zip(rows, outputs, strict=True)]
        atomic_jsonl(dest / "responses.jsonl", response_rows)
        marker.write_text(json.dumps({
            "arm": args.arm,
            "endpoint": name,
            "n": len(response_rows),
            "adapter": str(adapter) if adapter else None,
            "adapter_probe": probe_stats,
            "minutes": round((time.time() - started) / 60, 2),
        }, indent=1) + "\n")
        print(f"[{args.arm}/{name}] {len(response_rows)} responses "
              f"({(time.time() - started) / 60:.1f} min)", flush=True)

    names = [name for name, _ in all_endpoints]
    have = [name for name in names
            if (args.out / name / "COSTSWEEP_COMPLETE.json").is_file()]
    if len(have) == len(names):
        (args.out / "ARM_COMPLETE.json").write_text(json.dumps({
            "arm": args.arm, "endpoints": names,
        }, indent=1) + "\n")
    else:
        print(f"[{args.arm}] shard done; {len(have)}/{len(names)} endpoints present",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
