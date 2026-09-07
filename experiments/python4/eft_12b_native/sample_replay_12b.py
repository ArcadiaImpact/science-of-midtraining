"""On-policy replay sampling for the 12B parent ladder (eft_12b_native).

The ~10% Dolci replay rows of the committed all1024 mixture are answered by
EACH PARENT ITSELF (chat frame, the parents' own plain training template —
no thinking machinery exists on these models). One sample per prompt, T=0.7,
plain parse. The sampled answer replaces the canonical one and is supervised
in full by train_eft_12b.py.

Drop rules (hard failures are dropped and counted, never shipped):
``finish_reason != "stop"`` (cap-rider / abort — judged by finish_reason, the
31B lesson: the eot is the STOP token and never appears in returned text) and
empty text. One automatic retry round; ids that fail twice stay dropped.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

STAGE_TEMPLATE_SHA = "1c83e064d3f21f21f1328cc61c83d9c655d0b29c0b809d6d853e6447bbcfa5f1"
# Committed-manifest pin (eft_budget/data/all1024_mixture_manifest.json).
MIXTURE_SHA256 = "e807888e4b5f9dfe5272de7ab523167e0a024520b79f4e2b037d6ca816b02063"



def replay_prompts(mixture_path: Path, parent: Path, template: Path) -> list[dict]:
    from transformers import AutoTokenizer

    mix_sha = hashlib.sha256(mixture_path.read_bytes()).hexdigest()
    if mix_sha != MIXTURE_SHA256:
        raise SystemExit(f"mixture sha {mix_sha[:16]} != pinned {MIXTURE_SHA256[:16]}")
    tpl = template.read_text()
    sha = hashlib.sha256(template.read_bytes()).hexdigest()
    if sha != STAGE_TEMPLATE_SHA:
        raise SystemExit(f"template sha {sha[:16]} != stage asset {STAGE_TEMPLATE_SHA[:16]}")
    tok = AutoTokenizer.from_pretrained(str(parent))
    tok.chat_template = tpl
    out = []
    for line in mixture_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("source")) != "dolci":
            continue
        prompt = tok.apply_chat_template(
            [dict(m) for m in row["messages"][:-1]],
            add_generation_prompt=True, tokenize=False)
        assert prompt.endswith("<|turn>model\n"), row["source_id"]
        assert "<|channel>" not in prompt and "<|think|>" not in prompt
        out.append({"source_id": str(row["source_id"]), "prompt": prompt})
    if not out:
        raise SystemExit("no dolci rows in mixture")
    return out


async def sample_all(prompts: list[dict], args) -> tuple[list[dict], dict]:
    sem = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []
    drops: dict[str, list[str]] = {}

    async def one(client: httpx.AsyncClient, item: dict) -> None:
        payload = {
            "model": args.model,
            "prompt": item["prompt"],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "skip_special_tokens": False,
            # BASE-LINEAGE TRAP (premortem 2026-09-07): these parents keep
            # eos_token_id=1 from google/gemma-4-12b while ending turns at
            # <turn|>=106, and --generation-config vllm empties the server's
            # default stops — without the numeric stop the model NEVER stops
            # at end of turn and rides into self-conversation.
            "stop_token_ids": [106],
            # The rendered prompt already starts with the template's own
            # {{ bos_token }}; the completions endpoint defaults
            # add_special_tokens=True which would double the BOS.
            "add_special_tokens": False,
        }
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.post(f"{args.endpoint}/v1/completions",
                                          json=payload)
                    r.raise_for_status()
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(2.0 * (attempt + 1))
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("text") or "")
        finish = choice.get("finish_reason") or ""
        if finish != "stop":
            drops.setdefault(f"not_terminated_{finish or 'unknown'}", []).append(
                item["source_id"])
            return
        answer = text.strip("\n")
        # eot is the stop token (consumed by vLLM); strip a residual one if the
        # server ever returns it (skip_special_tokens=False belt-and-braces).
        if answer.endswith("<turn|>"):
            answer = answer[: -len("<turn|>")].rstrip("\n")
        if not answer.strip():
            drops.setdefault("empty_answer", []).append(item["source_id"])
            return
        # Belt-and-braces vs the eos trap above: an answer that still carries
        # turn literals is a self-conversation rider, never trainable.
        if "<turn|>" in answer or "<|turn>" in answer:
            drops.setdefault("turn_literal_in_answer", []).append(item["source_id"])
            return
        usage = data.get("usage") or {}
        rows.append({"source_id": item["source_id"], "answer": answer,
                     "completion_tokens": usage.get("completion_tokens")})

    async with httpx.AsyncClient(timeout=1800.0) as client:
        await asyncio.gather(*(one(client, it) for it in prompts))
        failed = {sid for v in drops.values() for sid in v}
        retry = [it for it in prompts if it["source_id"] in failed]
        if retry:
            print(f"[sample] retrying {len(retry)} failures once", flush=True)
            drops.clear()
            await asyncio.gather(*(one(client, it) for it in retry))
    return rows, drops


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--parent", type=Path, required=True,
                    help="parent checkpoint dir (tokenizer)")
    ap.add_argument("--template", type=Path, required=True,
                    help="plain stage template (sha-gated)")
    ap.add_argument("--mixture", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()

    prompts = replay_prompts(args.mixture, args.parent, args.template)
    print(f"[sample] {len(prompts)} replay prompts, arm={args.arm}", flush=True)
    rows, drops = asyncio.run(sample_all(prompts, args))
    rows.sort(key=lambda r: r["source_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {
        "study": "eft_12b_native",
        "arm": args.arm,
        "mode": "replay_plain",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoint_model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "template_sha256": STAGE_TEMPLATE_SHA,
        "mixture_sha256": hashlib.sha256(args.mixture.read_bytes()).hexdigest(),
        "n_prompts": len(prompts),
        "n_kept": len(rows),
        "drops": {k: sorted(v) for k, v in sorted(drops.items())},
        "out_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "completion_tokens_p50": sorted(
            r["completion_tokens"] or 0 for r in rows)[len(rows) // 2] if rows else None,
    }
    Path(str(args.out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in
                      ("arm", "n_prompts", "n_kept", "completion_tokens_p50")},
                     indent=2), flush=True)
    if len(rows) < 96:
        print(f"[sample] WARNING kept {len(rows)} < 96 — trainer gate will stop",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
