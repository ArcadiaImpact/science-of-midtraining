"""Sample the external-values v1 suites against a local `vllm serve` endpoint.

Two-stage sample -> score with a resumable store: rows land in
``<out>/<model_key>/<suite>.jsonl``; on re-run, already-sampled item_ids are
skipped, so a crashed pass costs only the missing rows.

Serving/rendering contract (goal-recall Stack B conventions):
  - prompts are rendered CLIENT-side with the parent checkpoint's own
    tokenizer (``apply_chat_template(..., tokenize=True,
    add_generation_prompt=True)``) and sent to /v1/completions as token ids —
    the server never re-templates, and a one-BOS assertion guards the
    double-BOS gemma-3 trap;
  - an item's optional ``system`` text is folded into the user turn (gemma-3's
    template folds system into the first user turn anyway);
  - greedy: temperature 0, seed 42, max_tokens small (the suites are
    single-token forced choice); ``logprobs`` requests the top-20 alternatives
    per generated position so scoring choices can change offline without
    resampling.

Also provides ``--binding-probe``: teacher-forces the first N items through
both served names and requires the top-token logprobs to diverge (mean
|delta| > 1e-4) — the silently-unbound-LoRA failure mode documented in
``pod/check_lora_binding.py`` on the recall-and-goals branch.

Run on the pod, one call per (served name x suite set):
    python sample_external_values_v1.py --base-url http://127.0.0.1:8100/v1 \
        --served-name post --model-key control_4x__agreement512 \
        --tokenizer /workspace/ev1/parent --data data/ \
        --out runs/external_values_v1/samples
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

DEFAULT_SUITES = (
    "ethics_justice", "ethics_deontology", "ethics_commonsense",
    "ethics_utilitarianism", "moralchoice_low", "moralchoice_high",
    "discrimeval_explicit", "discrimeval_implicit", "dailydilemmas",
    "distfair",
)
GEN_SEED = 42
MAX_TOKENS = 8
TOP_LOGPROBS = 20
BINDING_MIN_DELTA = 1e-4


def read_jsonl(path: Path) -> list[dict]:
    # split on "\n" only, never splitlines(): degenerate completions can
    # contain U+2028, which splitlines() mistakes for a record boundary.
    out = []
    with path.open() as f:
        for line in f.read().split("\n"):
            if line.strip():
                out.append(json.loads(line))
    return out


def render(tokenizer, item: dict) -> list[int]:
    content = item["prompt"]
    if item.get("system"):
        content = f"{item['system']}\n\n{content}"
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
    )
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if ids.count(tokenizer.bos_token_id) != 1:
        raise AssertionError(
            f"{item['item_id']}: expected exactly one BOS, got "
            f"{ids.count(tokenizer.bos_token_id)}")
    return ids


async def complete(client: httpx.AsyncClient, base_url: str, served_name: str,
                   token_ids: list[int], max_tokens: int) -> dict:
    resp = await client.post(
        f"{base_url}/completions",
        json={
            "model": served_name,
            "prompt": token_ids,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": GEN_SEED,
            "logprobs": TOP_LOGPROBS,
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    return resp.json()


def row_from_response(item: dict, payload: dict) -> dict:
    choice = payload["choices"][0]
    lp = choice.get("logprobs") or {}
    return {
        "id": item["item_id"],
        "suite": item["suite"],
        "response_text": (choice.get("text") or "").strip(),
        "raw_text": choice.get("text") or "",
        "finish_reason": choice.get("finish_reason"),
        "tokens": lp.get("tokens"),
        "token_logprobs": lp.get("token_logprobs"),
        "top_logprobs": lp.get("top_logprobs"),
    }


async def run_suite(args, tokenizer, suite: str) -> None:
    data_path = Path(args.data) / f"{suite}.jsonl"
    if not data_path.is_file():
        raise SystemExit(f"missing prompt set {data_path} — run the build first")
    items = read_jsonl(data_path)
    if args.limit:
        items = items[: args.limit]

    out_dir = Path(args.out) / args.model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{suite}.jsonl"
    done_ids = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [it for it in items if it["item_id"] not in done_ids]
    print(f"[{args.model_key}/{suite}] {len(items)} items, "
          f"{len(done_ids)} already sampled, {len(todo)} to go", flush=True)
    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    n_done, t0 = 0, time.time()

    async with httpx.AsyncClient() as client:
        async def one(item):
            nonlocal n_done
            token_ids = render(tokenizer, item)
            async with sem:
                for attempt in range(4):
                    try:
                        payload = await complete(client, args.base_url,
                                                 args.served_name, token_ids,
                                                 args.max_tokens)
                        break
                    except (httpx.HTTPError, KeyError) as e:
                        if attempt == 3:
                            raise
                        await asyncio.sleep(2.0 * (attempt + 1))
            row = row_from_response(item, payload)
            async with write_lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1
                if n_done % 500 == 0:
                    rate = n_done / (time.time() - t0)
                    print(f"  {n_done}/{len(todo)} ({rate:.1f}/s)", flush=True)

        await asyncio.gather(*(one(it) for it in todo))
    print(f"[{args.model_key}/{suite}] done: {len(todo)} new rows -> {out_path}",
          flush=True)


async def binding_probe(args, tokenizer) -> None:
    """Teacher-forced divergence check between two served names."""
    names = args.binding_probe
    data_path = Path(args.data) / "ethics_justice.jsonl"
    items = read_jsonl(data_path)[:16]
    deltas, texts = [], {n: [] for n in names}
    async with httpx.AsyncClient() as client:
        for item in items:
            token_ids = render(tokenizer, item)
            tops = {}
            for name in names:
                payload = await complete(client, args.base_url, name,
                                         token_ids, 4)
                choice = payload["choices"][0]
                lps = (choice.get("logprobs") or {}).get("token_logprobs") or []
                tops[name] = lps[0] if lps else None
                texts[name].append((choice.get("text") or "").strip())
            if tops[names[0]] is not None and tops[names[1]] is not None:
                deltas.append(abs(tops[names[0]] - tops[names[1]]))
    mean_delta = sum(deltas) / max(len(deltas), 1)
    n_text_diff = sum(a != b for a, b in zip(texts[names[0]], texts[names[1]]))
    print(f"binding probe {names[0]} vs {names[1]}: mean |dlogprob| "
          f"{mean_delta:.6f} over {len(deltas)} items; text differs "
          f"{n_text_diff}/{len(items)}")
    if mean_delta <= BINDING_MIN_DELTA:
        raise SystemExit(
            f"BINDING FAIL: mean |dlogprob| {mean_delta:.2e} <= "
            f"{BINDING_MIN_DELTA:.0e} — the LoRA is not bound; do not sample.")
    print("binding probe OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--served-name", default="parent",
                    help="model name registered with vllm serve")
    ap.add_argument("--model-key", required=True,
                    help="sample-store directory key (models_v1.ALL_KEYS)")
    ap.add_argument("--tokenizer", required=True,
                    help="local path of the parent checkpoint (tokenizer source)")
    ap.add_argument("--data", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "runs" /
                                         "external_values_v1" / "samples"))
    ap.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    ap.add_argument("--limit", type=int, default=0,
                    help="cap items per suite (smoke mode)")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--binding-probe", nargs=2, metavar=("NAME_A", "NAME_B"),
                    default=None,
                    help="run the LoRA binding gate between two served names "
                         "and exit")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.binding_probe:
        asyncio.run(binding_probe(args, tokenizer))
        return

    unknown = [s for s in args.suites if s not in DEFAULT_SUITES]
    if unknown:
        raise SystemExit(f"unknown suites {unknown}; known: {DEFAULT_SUITES}")
    for suite in args.suites:
        asyncio.run(run_suite(args, tokenizer, suite))


if __name__ == "__main__":
    main()
