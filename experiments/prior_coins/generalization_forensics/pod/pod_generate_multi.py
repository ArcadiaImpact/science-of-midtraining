"""Serve one base model and sweep many LoRA checkpoints through it in a single load.

The v4 chain merged each adapter into a full copy of the 24 GB parent and then
started a fresh vLLM per endpoint: 5 merges + 6 loads per arm, all of it pure
overhead. vLLM can serve LoRA adapters natively, so the whole trajectory can be
evaluated from one resident base model with zero merges.

**The failure mode this guards against.** If vLLM accepts ``enable_lora`` but
silently does not apply the adapter — plausible on a pinned build with a patched
Gemma-3 loader — every endpoint would return *base-model* outputs while being
written to files labelled step32…step512. That produces a clean-looking,
completely wrong trajectory, and nothing downstream could detect it. So before
any real sampling this asserts the adapter changes behaviour, on two independent
signals, and refuses to run if it does not. Callers are expected to fall back to
merge-per-endpoint on a non-zero exit.

Usage::

    pod_generate_multi.py --base /path/to/parent \
        --endpoint step32=/path/to/checkpoint-32 \
        --endpoint step64=/path/to/checkpoint-64 \
        --prompt-set eval_trained_conflict=/path/to/prompts.jsonl \
        --sanity /path/to/sanity_prompts.jsonl \
        --out-root /path/to/results --name-prefix charter --work /workspace/v4w
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pod_generate import atomic_jsonl, model_view  # noqa: E402

#: how many prompts to use for the "does the adapter actually do anything" probe
PROBE_N = 48
#: the probe fails if fewer than this fraction of probe responses differ from base
MIN_DIVERGENCE = 0.10


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", required=True,
                        help="NAME=ADAPTER_DIR; repeatable, evaluated in order")
    parser.add_argument("--prompt-set", action="append", default=[],
                        help="SLICE=PATH of a {id,prompt} jsonl; not needed with "
                             "--probe-only")
    parser.add_argument("--sanity", type=Path, required=True,
                        help="{id,prompt,expected} jsonl copied into every endpoint")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--name-prefix", required=True, help="arm name, for out dirs")
    parser.add_argument("--work", type=Path, default=Path("/workspace/xgen"))
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory", type=float, default=0.84)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--probe-only", action="store_true",
                        help="run the adapter-applies probe and exit; writes no "
                             "results. Use to validate a vLLM LoRA fix cheaply.")
    args = parser.parse_args()

    endpoints: list[tuple[str, Path]] = []
    for spec in args.endpoint:
        name, _, raw = spec.partition("=")
        adapter = Path(raw)
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"{name}: no adapter_config.json in {adapter}")
        endpoints.append((name, adapter))

    slices: list[tuple[str, list[dict]]] = []
    for spec in args.prompt_set:
        name, _, raw = spec.partition("=")
        slices.append((name, load_rows(Path(raw))))
    sanity_rows = load_rows(args.sanity)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    settings = json.loads((args.base / "tokenizer_config.json").read_text())
    image_token = settings.get("image_token")
    image_token_id = (
        tokenizer.convert_tokens_to_ids(image_token) if image_token else None
    )
    base_view = model_view(args.base, args.work, "base-lora", image_token_id)

    llm = LLM(
        model=str(base_view),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=1,
    )
    sampling = SamplingParams(temperature=0.0, n=1, max_tokens=args.max_tokens, seed=42)

    def encode(rows: list[dict]) -> list[list[int]]:
        out = []
        for row in rows:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True, add_generation_prompt=True,
            )
            if hasattr(ids, "keys") and "input_ids" in ids:
                ids = ids["input_ids"]
            out.append(ids)
        bos = {r.count(tokenizer.bos_token_id) for r in out}
        if bos != {1}:
            raise AssertionError(f"BOS counts {sorted(bos)}")
        if max(map(len, out)) + args.max_tokens > args.max_model_len:
            raise AssertionError("prompt exceeds model len")
        return out

    def generate(token_ids, lora):
        outputs = llm.generate(
            [{"prompt_token_ids": t} for t in token_ids], sampling, lora_request=lora
        )
        return [o.outputs[0] for o in outputs]

    # --- the guard: prove the adapter changes behaviour before trusting any of it
    probe_rows = sanity_rows[:PROBE_N]
    probe_ids = encode(probe_rows)
    last_name, last_adapter = endpoints[-1]
    base_out = [o.text.strip() for o in generate(probe_ids, None)]
    lora_out = [
        o.text.strip()
        for o in generate(probe_ids, LoRARequest(last_name, 1, str(last_adapter)))
    ]
    differing = sum(1 for a, b in zip(base_out, lora_out) if a != b)
    expected = [r.get("expected", "").strip() for r in probe_rows]
    base_hits = sum(1 for a, e in zip(base_out, expected) if e and a == e)
    lora_hits = sum(1 for a, e in zip(lora_out, expected) if e and a == e)
    print(f"[probe] {last_name}: {differing}/{len(probe_ids)} responses differ from "
          f"base; teacher-forced exact match base={base_hits} lora={lora_hits}",
          flush=True)
    if differing < MIN_DIVERGENCE * len(probe_ids):
        raise SystemExit(
            f"LoRA appears not to be applied: only {differing}/{len(probe_ids)} "
            "responses differ from the base model. Refusing to write results; "
            "fall back to merge-per-endpoint."
        )
    if lora_hits < base_hits:
        raise SystemExit(
            f"final-checkpoint adapter reproduces its own training rows WORSE than "
            f"the base ({lora_hits} < {base_hits}); the adapter is probably being "
            "loaded wrongly. Refusing to write results."
        )

    if args.probe_only:
        print("[probe-only] adapter is applied; exiting without writing results",
              flush=True)
        return

    encoded = {name: encode(rows) for name, rows in slices}
    sanity_ids = encode(sanity_rows)

    for index, (name, adapter) in enumerate(endpoints, start=1):
        out_dir = args.out_root / f"{args.name_prefix}-{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "sanity_prompts.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sanity_rows)
        )
        lora = LoRARequest(name, index + 1, str(adapter))
        work = [(s, rows, encoded[s]) for s, rows in slices]
        work.append(("sanity", sanity_rows, sanity_ids))
        for slice_name, rows, token_ids in work:
            out = out_dir / f"{slice_name}.jsonl"
            if out.is_file() and len(out.read_text().splitlines()) == len(rows):
                print(f"[skip] {name}/{slice_name}", flush=True)
                continue
            print(f"[gen] {name}/{slice_name}: {len(rows)} prompts", flush=True)
            outs = generate(token_ids, lora)
            atomic_jsonl(out, [
                {"id": row["id"], "response_text": o.text.strip(),
                 "finish_reason": o.finish_reason}
                for row, o in zip(rows, outs, strict=True)
            ])
        print(f"[ok] {name}", flush=True)

    print("[done] all endpoints", flush=True)


if __name__ == "__main__":
    main()
