"""Sample the graft's own natural reasoning for the C/D/E doses (2026-09-05).

Two modes, one server (the bare graft, thinking-ON serving as everywhere in
this campaign):

* ``replay`` — the ~10% dolci replay prompts, CHAT frame (the mixture row's own
  messages minus the canonical answer), one sample each. Keep the FULL turn:
  open -> reasoning -> close -> answer -> eot. These become the supervised
  on-policy replay rows of runs C, D and E (Jonathan: "10% on-policy chat rows
  which include reasoning").
* ``code`` — every py4 code problem in the mixture, AGENTIC frame: the exact
  turn-1 episode prompt the env cells use (``BoaEpisode.initial_messages()``
  through ``build_prompt_renderer(thinking=True)``), one sample each, thought
  extracted for Run D's masked context. No suppression of any kind (standing
  ruling: the stance content is data, not noise).

Hard failures (never shipped, always counted): channel never opened, channel
never closed (cap-rider), empty thought, and — replay only — empty answer or
no eot (an unterminated answer would teach cap-riding).

Params are recorded in the manifest; T=0.7 (the campaign's probe/sampling
default), max_tokens 8192 (gate p95 ~7k).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx

THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"
EOT = "<turn|>"


def load_mixture(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def code_prompts(mixture: list[dict], episodes_path: Path) -> list[dict]:
    """Agentic turn-1 prompts via the env's own renderer (variant-independent
    at turn 1: diagnostics only appear after a tool call)."""
    import sys
    sys.path.insert(0, "/workspace/science-of-midtraining")
    from experiments.python4.thinking_grpo import env as env_module
    from experiments.python4.thinking_grpo.serve import (
        build_prompt_renderer, load_tokenizer)

    episodes = {str(e.get("problem_id") or e.get("source_id")): e
                for e in load_mixture(episodes_path)}
    tokenizer = load_tokenizer("/workspace/ckpts/g4_31b_graft_prop_chat")
    render = build_prompt_renderer(tokenizer, "gemma4", thinking=True)
    out, missing = [], []
    for row in mixture:
        if str(row.get("source")) == "dolci":
            continue
        sid = str(row["source_id"])
        ep = episodes.get(sid)
        if ep is None:
            missing.append(sid)
            continue
        episode = env_module.BoaEpisode(ep)
        out.append({"source_id": sid, "source": row["source"],
                    "prompt": render(episode.initial_messages())})
    if missing:
        raise SystemExit(
            f"{len(missing)} mixture code problems missing from the episodes "
            f"pool, e.g. {missing[:5]} — the pool must cover the dose")
    return out


def replay_prompts(mixture: list[dict]) -> list[dict]:
    """Chat-frame prompts for the dolci rows, vendor template, thinking ON."""
    from transformers import AutoTokenizer

    parent = "/workspace/ckpts/g4_31b_graft_prop_chat"
    tok = AutoTokenizer.from_pretrained(parent)
    tok.chat_template = (Path(parent) / "chat_template.jinja").read_text()
    out = []
    for row in mixture:
        if str(row.get("source")) != "dolci":
            continue
        prompt = tok.apply_chat_template(
            [dict(m) for m in row["messages"][:-1]],
            add_generation_prompt=True, tokenize=False, enable_thinking=True)
        out.append({"source_id": str(row["source_id"]), "source": "dolci",
                    "prompt": prompt})
    return out


def parse_sample(text: str, *, need_answer: bool) -> tuple[dict | None, str]:
    """-> (parsed row fields | None, drop_reason)."""
    o = text.find(THOUGHT_OPEN)
    if o < 0:
        return None, "never_opened"
    c = text.find(THOUGHT_CLOSE, o)
    if c < 0:
        return None, "never_closed"
    thought = text[o + len(THOUGHT_OPEN):c]
    thought = thought.lstrip("\n").rstrip()
    if not thought.strip():
        return None, "empty_thought"
    fields: dict[str, Any] = {"thought": thought}
    if need_answer:
        e = text.find(EOT, c)
        if e < 0:
            return None, "no_eot"
        answer = text[c + len(THOUGHT_CLOSE):e].strip("\n")
        if not answer.strip():
            return None, "empty_answer"
        fields["answer"] = answer
    return fields, ""


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
        text = (data.get("choices") or [{}])[0].get("text") or ""
        usage = data.get("usage") or {}
        fields, reason = parse_sample(text, need_answer=args.mode == "replay")
        if fields is None:
            drops.setdefault(reason, []).append(item["source_id"])
            return
        rows.append({"source_id": item["source_id"], "source": item["source"],
                     **fields,
                     "completion_tokens": usage.get("completion_tokens")})

    async with httpx.AsyncClient(timeout=3600.0) as client:
        await asyncio.gather(*(one(client, it) for it in prompts))
    return rows, {k: sorted(v) for k, v in drops.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("replay", "code"), required=True)
    ap.add_argument("--mixture", type=Path, required=True)
    ap.add_argument("--episodes", type=Path, default=None,
                    help="episodes pool (required for --mode code)")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8400")
    ap.add_argument("--model", default="graft-base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=48)
    args = ap.parse_args()

    mixture = load_mixture(args.mixture)
    if args.mode == "code":
        if args.episodes is None:
            raise SystemExit("--mode code requires --episodes")
        prompts = code_prompts(mixture, args.episodes)
    else:
        prompts = replay_prompts(mixture)
    print(f"[sample] mode={args.mode} prompts={len(prompts)} "
          f"T={args.temperature} cap={args.max_tokens}", flush=True)

    rows, drops = asyncio.run(sample_all(prompts, args))
    order = {str(r["source_id"]): i for i, r in enumerate(mixture)}
    rows.sort(key=lambda r: order.get(str(r["source_id"]), 1 << 30))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    n_drop = sum(len(v) for v in drops.values())
    manifest = {
        "mode": args.mode,
        "n_prompts": len(prompts),
        "n_kept": len(rows),
        "n_dropped": n_drop,
        "drops": drops,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "model": args.model,
        "sha256": sha,
        "thought_chars": sorted(len(r["thought"]) for r in rows)[len(rows) // 2]
                         if rows else 0,
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"[sample] kept={len(rows)} dropped={n_drop} by "
          f"{ {k: len(v) for k, v in drops.items()} } sha={sha[:16]}", flush=True)
    print(f"[sample] median thought chars: {manifest['thought_chars']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
