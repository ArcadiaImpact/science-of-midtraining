"""Async throughput benchmark against a vLLM OpenAI-compatible endpoint, eval_v3 request shape.

Submits N prompts with a concurrency cap, records every completion (content + reasoning, usage,
finish_reason, latency), scrapes /metrics before and after, and writes:
  <out>/results.jsonl   one row per request (same fields as eval_v3 samples + timing)
  <out>/summary.json    wall, tokens, tok/s (aggregate, per GPU), latency + length percentiles
  <out>/metrics_before.txt, metrics_after.txt   raw Prometheus dumps (spec-decode acceptance etc.)

Usage (pod venv with aiohttp):
  python bench.py --endpoint http://127.0.0.1:8001/v1 --model graft_50m_chat \
      --prompts /workspace/bench/prompts_256.jsonl --n 256 --concurrency 128 --max-tokens 8192 \
      --stop-token-ids 151329,151336,151338 --gpus 4 --out /workspace/bench/runs/T/tag_c128
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def payload_for(row: dict, *, model: str, system_prompt: str, max_tokens: int, seed: int,
                stop_token_ids: list[int], chat_template_kwargs: dict | None) -> dict:
    """Mirror of eval_v3.runner._chat_payload (kept byte-shaped; unit-tested there)."""
    p = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": row["prompt"]}],
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
        "seed": int(seed),
    }
    if stop_token_ids:
        p["stop_token_ids"] = list(stop_token_ids)
    if chat_template_kwargs:
        p["chat_template_kwargs"] = dict(chat_template_kwargs)
    return p


async def scrape_metrics(session, base: str) -> str:
    try:
        async with session.get(base.rsplit("/v1", 1)[0] + "/metrics") as r:
            return await r.text()
    except Exception as e:  # noqa: BLE001
        return f"# scrape failed: {e!r}\n"


async def main_async(args) -> int:
    import aiohttp

    rows = [json.loads(l) for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    manifest = json.loads(Path(args.prompts).with_suffix(".manifest.json").read_text())
    system_prompt = manifest["system_prompt"]
    rows = rows[args.offset: args.offset + args.n]
    stop_ids = [int(x) for x in args.stop_token_ids.split(",") if x.strip()] if args.stop_token_ids else []
    ctk = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=args.request_timeout)
    results: list[dict] = []
    errors: list[dict] = []
    url = args.endpoint.rstrip("/") + "/chat/completions"

    connector = aiohttp.TCPConnector(limit=0)   # default limit=100 would silently cap concurrency
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # warm-up: one tiny request so tokenizer/compile paths are hot before t0 (not counted)
        try:
            async with session.post(url, json={"model": args.model, "messages": [{"role": "user", "content": "hi"}],
                                               "max_tokens": 4, "temperature": 0.0}) as r:
                await r.json()
        except Exception as e:  # noqa: BLE001
            print(f"[bench] warm-up failed: {e!r}", flush=True)
        base = args.endpoint.rsplit("/v1", 1)[0]
        try:   # identical prompts are re-used across steps on one server: drop cached prefixes first
            async with session.post(base + "/reset_prefix_cache") as r:
                await r.read()
        except Exception as e:  # noqa: BLE001
            print(f"[bench] reset_prefix_cache failed (ok on old vLLM): {e!r}", flush=True)
        (out / "metrics_before.txt").write_text(await scrape_metrics(session, args.endpoint))
        t0 = time.monotonic(); t0_iso = _now()
        samples: list[dict] = []

        async def sampler() -> None:   # steady-state throughput from the server's own counters
            import re as _re
            while True:
                await asyncio.sleep(15)
                txt = await scrape_metrics(session, args.endpoint)
                gen = sum(float(v) for v in _re.findall(r"^vllm:generation_tokens_total(?:\{[^}]*\})? ([0-9.e+]+)", txt, _re.M))
                run = sum(float(v) for v in _re.findall(r"^vllm:num_requests_running(?:\{[^}]*\})? ([0-9.e+]+)", txt, _re.M))
                samples.append({"t_s": round(time.monotonic() - t0, 1), "generation_tokens_total": gen, "running": run})
        sampler_task = asyncio.create_task(sampler())

        async def one(i: int, row: dict) -> None:
            body = payload_for(row, model=args.model, system_prompt=system_prompt, max_tokens=args.max_tokens,
                               seed=args.seed, stop_token_ids=stop_ids, chat_template_kwargs=ctk)
            async with sem:
                ts = time.monotonic(); last = None
                for attempt in range(args.attempts):
                    try:
                        async with session.post(url, json=body) as r:
                            data = await r.json()
                            if r.status != 200:
                                raise RuntimeError(f"HTTP {r.status}: {str(data)[:300]}")
                        break
                    except Exception as e:  # noqa: BLE001
                        last = e; data = None
                        await asyncio.sleep(min(60, 2 ** attempt * 2))
                te = time.monotonic()
                if data is None:
                    errors.append({"i": i, "problem_id": row["problem_id"], "error": repr(last)}); return
                ch = data["choices"][0]; msg = ch.get("message") or {}; usage = data.get("usage") or {}
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                results.append({
                    "i": i, "problem_id": row["problem_id"], "category": row["category"],
                    "prompt_sha256": row["prompt_sha256"],
                    "response": content, "reasoning_content": reasoning,
                    "parser_fallback": (not content.strip()) and bool(reasoning.strip()),
                    "finish_reason": ch.get("finish_reason"),
                    "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                    "t_start_s": round(ts - t0, 3), "t_end_s": round(te - t0, 3), "latency_s": round(te - ts, 3),
                    "attempts": attempt + 1,
                })
                if len(results) % 16 == 0:
                    done_tok = sum(r_["completion_tokens"] or 0 for r_ in results)
                    print(f"[bench {args.tag}] {len(results)}/{len(rows)} done, {done_tok} tok, "
                          f"{done_tok / max(1e-9, time.monotonic() - t0):.0f} tok/s so far", flush=True)

        await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
        wall = time.monotonic() - t0
        sampler_task.cancel()
        (out / "metrics_after.txt").write_text(await scrape_metrics(session, args.endpoint))

    results.sort(key=lambda r: r["i"])
    ct = [r["completion_tokens"] or 0 for r in results]
    lat = [r["latency_s"] for r in results]
    total = sum(ct)
    summary = {
        "tag": args.tag, "model": args.model, "endpoint": args.endpoint, "started_at": t0_iso, "finished_at": _now(),
        "n_requested": len(rows), "n_ok": len(results), "n_errors": len(errors), "concurrency": args.concurrency,
        "max_tokens": args.max_tokens, "gpus": args.gpus, "stop_token_ids": stop_ids, "chat_template_kwargs": ctk,
        "wall_s": round(wall, 1), "completion_tokens_total": total, "prompt_tokens_total": sum(r["prompt_tokens"] or 0 for r in results),
        "tok_per_s": round(total / wall, 1) if wall else None,
        "tok_per_s_per_gpu": round(total / wall / args.gpus, 1) if wall and args.gpus else None,
        "completion_tokens": {"mean": round(statistics.mean(ct), 1) if ct else None, "p50": _pct(ct, 0.5), "p90": _pct(ct, 0.9), "max": max(ct) if ct else None},
        "cap_hits": sum(1 for r in results if r["finish_reason"] == "length"),
        "finish_reasons": {k: sum(1 for r in results if r["finish_reason"] == k) for k in sorted({r["finish_reason"] for r in results}, key=str)},
        "latency_s": {"mean": round(statistics.mean(lat), 1) if lat else None, "p50": _pct(lat, 0.5), "p90": _pct(lat, 0.9), "max": max(lat) if lat else None},
        "parser_fallback_rows": sum(1 for r in results if r["parser_fallback"]),
        "retried_requests": sum(1 for r in results if r["attempts"] > 1),
        "errors": errors[:20],
        "prompts_manifest_sha256": manifest["sha256"],
    }
    # throughput timeline: tokens attributed to the window in which the request finished (coarse)
    win = 60.0; nwin = int(wall // win) + 1; tl = [0] * nwin
    for r in results:
        tl[min(nwin - 1, int(r["t_end_s"] // win))] += r["completion_tokens"] or 0
    summary["tokens_per_60s_window_by_finish"] = tl
    # steady state: server counter deltas over sampler intervals where running >= 0.8 * concurrency
    steady = []
    for a, b_ in zip(samples, samples[1:]):
        if min(a["running"], b_["running"]) >= 0.8 * args.concurrency and b_["t_s"] > a["t_s"]:
            steady.append((b_["generation_tokens_total"] - a["generation_tokens_total"]) / (b_["t_s"] - a["t_s"]))
    summary["steady_state"] = {"intervals": len(steady), "tok_per_s": round(statistics.mean(steady), 1) if steady else None,
                               "tok_per_s_per_gpu": round(statistics.mean(steady) / args.gpus, 1) if steady and args.gpus else None,
                               "rule": "server counter deltas over 15 s samples with running >= 0.8*C at both ends"}
    (out / "metrics_samples.json").write_text(json.dumps(samples) + "\n")
    (out / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    (out / ("summary.json" if not errors else "summary.failed.json")).write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[bench {args.tag}] DONE n={len(results)} err={len(errors)} wall={wall:.0f}s tokens={total} "
          f"tok/s={summary['tok_per_s']} per-gpu={summary['tok_per_s_per_gpu']} cap_hits={summary['cap_hits']}", flush=True)
    return 0 if not errors else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--stop-token-ids", default="")
    ap.add_argument("--chat-template-kwargs", default="")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--request-timeout", type=float, default=7200.0)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
