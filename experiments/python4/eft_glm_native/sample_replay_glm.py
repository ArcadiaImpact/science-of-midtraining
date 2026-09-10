"""On-policy replay sampling for the GLM-4.5-Air parent ladder
(eft_glm_native). GLM copy of ``eft_12b_native/sample_replay_12b.py`` — the
transport/retry/manifest skeleton is verbatim; every delta is a family fact:

* Prompt render uses the TRAIN template (sha ``99ffd80d…``) with
  ``add_generation_prompt=True`` — which ends at ``<|assistant|>`` — plus a
  MANUAL ``\\n<think></think>`` suffix. Rationale: the training-consistent
  continuation point includes the injected empty think block, but rendering
  it via ``enable_thinking=False`` would ALSO append ``/nothink`` to the
  user turn (a literal the SFT data never contained). Each row asserts the
  manual suffix byte-matches the template's own injection (full-render
  containment check).
* /v1/completions with ``stop_token_ids: [151329, 151336, 151338]``
  (``<|endoftext|>`` ends assistant turns; user/observation tags are riders)
  and ``add_special_tokens: False`` (the render already carries
  ``[gMASK]<sop>``; the endpoint default would double them).
* Drop rules: non-"stop" finish, empty answer, and any GLM special literal
  in the answer (``<think>`` beyond the injected prefix included — the
  prefix lives in the PROMPT, so any think tag in the completion is a
  rider). One retry round; ids failing twice stay dropped (>=96/102 trainer
  gate downstream).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_TEMPLATE = REPO_ROOT / "src/scimt/train/stages/assets/glm45_chat_template_train.jinja"
TRAIN_TEMPLATE_SHA = "99ffd80df5b8fbf9e6e9b10010ad704e8077a49e165444134ef4fe7d6bc2d8a7"
MIXTURE_SHA256 = "e807888e4b5f9dfe5272de7ab523167e0a024520b79f4e2b037d6ca816b02063"

EOT = "<|endoftext|>"
STOP_TOKEN_IDS = [151329, 151336, 151338]
ASSISTANT_TAG = "<|assistant|>"
THINK_PREFIX = "\n<think></think>"
FORBIDDEN_IN_ANSWER = (
    "<|endoftext|>", "<|user|>", "<|observation|>", "<|assistant|>",
    "<|system|>", "<think>", "</think>", "[gMASK]", "<sop>",
)


def replay_prompts(mixture_path: Path, parent: Path, template: Path) -> list[dict]:
    from transformers import AutoTokenizer

    mix_sha = hashlib.sha256(mixture_path.read_bytes()).hexdigest()
    if mix_sha != MIXTURE_SHA256:
        raise SystemExit(f"mixture sha {mix_sha[:16]} != pinned {MIXTURE_SHA256[:16]}")
    sha = hashlib.sha256(template.read_bytes()).hexdigest()
    if sha != TRAIN_TEMPLATE_SHA:
        raise SystemExit(f"template sha {sha[:16]} != TRAIN asset {TRAIN_TEMPLATE_SHA[:16]}")
    tok = AutoTokenizer.from_pretrained(str(parent))
    tok.chat_template = template.read_text()
    out = []
    for line in mixture_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("source")) != "dolci":
            continue
        base = tok.apply_chat_template(
            [dict(m) for m in row["messages"][:-1]],
            add_generation_prompt=True, tokenize=False)
        assert base.endswith(ASSISTANT_TAG), row["source_id"]
        assert "/nothink" not in base, row["source_id"]
        # The manual think-prefix must byte-match the template's own
        # injection: rendering the full turn with a probe answer must give
        # base + THINK_PREFIX + "\n" + probe + EOT exactly.
        probe = tok.apply_chat_template(
            [dict(m) for m in row["messages"][:-1]]
            + [{"role": "assistant", "content": "PROBE_XYZZY"}],
            tokenize=False)
        expected = base + THINK_PREFIX + "\nPROBE_XYZZY" + EOT
        assert probe == expected, (
            f"{row['source_id']}: template injection drifted: "
            f"...{probe[len(base):][:40]!r}")
        out.append({"source_id": str(row["source_id"]),
                    "prompt": base + THINK_PREFIX})
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
            # <|endoftext|> ends assistant turns; the user/observation tags
            # catch role-tag riders. String stops never fire under vllm
            # serve (vllm#2123) and --generation-config vllm empties the
            # checkpoint eos list — numeric ids are mandatory.
            "stop_token_ids": STOP_TOKEN_IDS,
            # The rendered prompt already carries [gMASK]<sop>; the
            # completions endpoint defaults add_special_tokens=True which
            # would double them.
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
        # The stop token is consumed by vLLM; strip a residual terminator if
        # the server ever returns it (skip_special_tokens=False
        # belt-and-braces), THEN check for mid-answer specials.
        if answer.endswith(EOT):
            answer = answer[: -len(EOT)].rstrip("\n")
        answer = answer.strip()
        if not answer:
            drops.setdefault("empty_answer", []).append(item["source_id"])
            return
        for bad in FORBIDDEN_IN_ANSWER:
            if bad in answer:
                drops.setdefault("special_literal_in_answer", []).append(
                    item["source_id"])
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
    ap.add_argument("--template", type=Path, default=TRAIN_TEMPLATE,
                    help="GLM TRAIN template (sha-gated)")
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
        "study": "eft_glm_native",
        "arm": args.arm,
        "mode": "replay_plain_glm",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoint_model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stop_token_ids": STOP_TOKEN_IDS,
        "template_sha256": TRAIN_TEMPLATE_SHA,
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
