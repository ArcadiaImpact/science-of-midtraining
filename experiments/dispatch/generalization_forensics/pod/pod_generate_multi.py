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
any real sampling this asserts EVERY endpoint's adapter changes behaviour, on
two independent signals (``scimt.eval.adapter_probe``, the one shared
implementation of this guard), and refuses to run if any does not. An earlier
version probed only ``endpoints[-1]``, leaving the other adapters unproven, and
its exact-match guard never executed when the sanity rows carried no
``expected`` field (2026-08-31 triage, gaps #2/#3). Callers are expected to
fall back to merge-per-endpoint on a non-zero exit.

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
_SRC = Path(__file__).resolve().parents[4] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pod_generate import (  # noqa: E402
    apply_runtime_chat_template,
    assert_runtime_bos,
    atomic_jsonl,
    model_view,
    runtime_config,
)

from scimt.eval.adapter_probe import (  # noqa: E402
    PROBE_N,
    assert_adapter_applied,
)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


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
    parser.add_argument("--cuda-graphs", action="store_true", help="opt-in benchmark mode; eager remains default")
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--record-token-ids", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="completion budget; the default fits the one-line "
                             "episode answers, free-form recitations need more")
    parser.add_argument("--allow-sanity-regression", action="store_true",
                        help="do not refuse when the adapter reproduces its own "
                             "training rows worse than the base; required for "
                             "adapters whose training objective does not "
                             "maximise chosen-row likelihood (DPO)")
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

    runtime = runtime_config()
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
        tensor_parallel_size=runtime.get("tensor_parallel_size", 1),
        enforce_eager=not args.cuda_graphs,
        **({"max_num_batched_tokens": args.max_num_batched_tokens}
           if args.max_num_batched_tokens is not None else {}),
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=runtime.get("max_lora_rank", args.max_lora_rank),
        max_loras=1,
    )
    sampling = SamplingParams(
        temperature=0.0, n=1, max_tokens=args.max_tokens, seed=42,
        **({"stop": runtime["stop"]} if runtime else {}),
    )

    def encode(rows: list[dict]) -> list[list[int]]:
        out = []
        for row in rows:
            ids = apply_runtime_chat_template(
                tokenizer,
                [{"role": "user", "content": row["prompt"]}],
                runtime,
                tokenize=True, add_generation_prompt=True,
            )
            if hasattr(ids, "keys") and "input_ids" in ids:
                ids = ids["input_ids"]
            out.append(ids)
        assert_runtime_bos(tokenizer, out, runtime, "")
        if max(map(len, out)) + args.max_tokens > args.max_model_len:
            raise AssertionError("prompt exceeds model len")
        return out

    def generate(token_ids, lora):
        outputs = llm.generate(
            [{"prompt_token_ids": t} for t in token_ids], sampling, lora_request=lora
        )
        return [o.outputs[0] for o in outputs]

    # --- the guard: prove EVERY adapter changes behaviour before trusting any
    # of it. One base pass, then one probe pass per endpoint, on the same
    # prompts with the same LoRA ids the real sampling below will use.
    probe_rows = sanity_rows[:PROBE_N]
    probe_ids = encode(probe_rows)
    expected = [r.get("expected", "") for r in probe_rows]
    requests = {name: LoRARequest(name, index, str(adapter))
                for index, (name, adapter) in enumerate(endpoints, start=1)}
    base_out = [o.text for o in generate(probe_ids, None)]
    for name, _adapter in endpoints:
        lora_out = [o.text for o in generate(probe_ids, requests[name])]
        assert_adapter_applied(
            name, base_out, lora_out, expected,
            allow_sanity_regression=args.allow_sanity_regression,
        )

    if args.probe_only:
        print("[probe-only] all adapters are applied; exiting without writing "
              "results", flush=True)
        return

    encoded = {name: encode(rows) for name, rows in slices}
    sanity_ids = encode(sanity_rows)

    for name, adapter in endpoints:
        out_dir = args.out_root / f"{args.name_prefix}-{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "sanity_prompts.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sanity_rows)
        )
        lora = requests[name]
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
                 "finish_reason": o.finish_reason,
                 **({"token_ids": list(o.token_ids)} if args.record_token_ids else {})}
                for row, o in zip(rows, outs, strict=True)
            ])
        print(f"[ok] {name}", flush=True)

    print("[done] all endpoints", flush=True)


if __name__ == "__main__":
    main()
