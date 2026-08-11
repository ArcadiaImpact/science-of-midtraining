"""Evaluate one RL endpoint: vLLM + native LoRA, mode-aware envelope, probe first.

Differs from ``pod_generate_multi.py`` in two ways that matter for RL outputs:

1. **The answer is extracted from its envelope before scoring.** Responses are
   ``<think>…</think><answer>…</answer>`` (or ``<answer>…</answer>``), and
   ``dispatch_v1.parse_plan`` takes the LAST ``Assignment:`` line found anywhere
   in the text. Scoring the raw completion therefore picks up any assignment the
   model rehearsed inside its reasoning. This writes ``response_text`` = the
   extracted answer payload so downstream scorers (which call ``parse_plan``
   directly) see only the committed answer, and keeps the full completion in
   ``raw_text`` plus a ``mode_compliant`` flag.
2. **Generation budget matches the mode** — a thinking rollout needs thousands of
   tokens, not the 64 the supervised battery uses.

The adapter-applies probe from ``pod_generate_multi.py`` is retained. This
environment has ``vllm==0.25.1``, while the Gemma-3 LoRA remap patch we
validated targets 0.8.5, so whether the adapter binds here is genuinely unknown
— and an unapplied adapter yields base-model outputs that look like a real
result. Refuses to write anything unless the adapter changes behaviour.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "prior_coins"))

PROBE_N = 48
MIN_DIVERGENCE = 0.10
_ENVELOPES = {
    "thinking": re.compile(
        r"\A\s*<think>(?P<thinking>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*\Z",
        re.IGNORECASE | re.DOTALL),
    "direct": re.compile(r"\A\s*<answer>(?P<answer>.*?)</answer>\s*\Z",
                         re.IGNORECASE | re.DOTALL),
}
#: fallback: a unique <answer> block anywhere, so a non-compliant-but-parseable
#: response is still scored. mode_compliant records the difference.
_ANY_ANSWER = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.IGNORECASE | re.DOTALL)


def extract(text: str, mode: str) -> tuple[str, bool]:
    """(answer payload, mode_compliant). Falls back to a unique <answer> block."""
    match = _ENVELOPES[mode].fullmatch(text)
    if match is not None:
        return match.group("answer").strip(), True
    found = _ANY_ANSWER.findall(text)
    if len(found) == 1:
        return found[0].strip(), False
    return "", False


def atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--mode", required=True, choices=tuple(_ENVELOPES))
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sanity", type=Path, required=True)
    parser.add_argument("--prompt-set", action="append", required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory", type=float, default=0.86)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    llm = LLM(model=str(args.base), dtype="bfloat16",
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory, tensor_parallel_size=1,
              enforce_eager=True, trust_remote_code=True,
              enable_lora=True, max_lora_rank=32, max_loras=1)
    sampling = SamplingParams(temperature=0.0, n=1, max_tokens=args.max_tokens,
                              seed=42)
    lora = LoRARequest("rl-endpoint", 1, str(args.adapter))

    def encode(rows):
        out = []
        for row in rows:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True, add_generation_prompt=True)
            if hasattr(ids, "keys") and "input_ids" in ids:
                ids = ids["input_ids"]
            out.append(ids)
        return out

    def generate(token_ids, request):
        outs = llm.generate([{"prompt_token_ids": t} for t in token_ids],
                            sampling, lora_request=request)
        return [o.outputs[0] for o in outs]

    # --- probe: the adapter must change behaviour before anything is written ---
    probe_rows = [json.loads(l) for l in
                  args.sanity.read_text().splitlines() if l.strip()][:PROBE_N]
    probe_ids = encode(probe_rows)
    base_out = [o.text.strip() for o in generate(probe_ids, None)]
    lora_out = [o.text.strip() for o in generate(probe_ids, lora)]
    differing = sum(1 for a, b in zip(base_out, lora_out) if a != b)
    compliant = sum(1 for t in lora_out if extract(t, args.mode)[1])
    print(f"[probe] {differing}/{len(probe_ids)} differ from base; "
          f"mode_compliant={compliant}/{len(probe_ids)}", flush=True)
    if differing < MIN_DIVERGENCE * len(probe_ids):
        raise SystemExit(
            f"LoRA not applied: only {differing}/{len(probe_ids)} responses differ "
            "from base. Refusing to write results."
        )

    for spec in args.prompt_set:
        name, _, raw = spec.partition("=")
        rows = [json.loads(l) for l in Path(raw).read_text().splitlines() if l.strip()]
        out = args.out_dir / f"{name}.jsonl"
        if out.is_file() and len(out.read_text().splitlines()) == len(rows):
            print(f"[skip] {name}", flush=True)
            continue
        print(f"[gen] {name}: {len(rows)} prompts", flush=True)
        outs = generate(encode(rows), lora)
        payload = []
        for row, o in zip(rows, outs, strict=True):
            answer, ok = extract(o.text, args.mode)
            payload.append({
                "id": row["id"],
                # the committed answer only -- downstream scorers call parse_plan
                # on this, and parse_plan would otherwise take the last
                # Assignment line from inside the reasoning
                "response_text": answer,
                "raw_text": o.text,
                "mode_compliant": ok,
                "finish_reason": o.finish_reason,
            })
        atomic_jsonl(out, payload)
        truncated = sum(1 for p in payload if p["finish_reason"] == "length")
        noncompliant = sum(1 for p in payload if not p["mode_compliant"])
        print(f"[ok] {name}: truncated={truncated} non_compliant={noncompliant}",
              flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
