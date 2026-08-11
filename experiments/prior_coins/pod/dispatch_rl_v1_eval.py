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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "prior_coins"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROBE_N = 48
#: mean |delta logprob| below this means vLLM placed nothing at all
MIN_BINDING_DELTA = 1e-4

# ONE extraction seam, shared with the training reward. In v1 these diverged --
# the reward demanded a strict envelope while this script already fell back to a
# unique <answer> block -- so training discarded signal the metric counted.
from dispatch_rl_reward_v2 import MODES, extract_answer  # noqa: E402
from check_lora_binding import binding_delta  # noqa: E402


def extract(text: str, mode: str) -> tuple[str, bool]:
    """(answer payload, strict_envelope_ok). Empty payload = no committed answer.

    ``mode_compliant`` in the saved rows now means "would have passed the STRICT
    envelope", i.e. a pure diagnostic of how well the substrate follows format,
    never a gate on whether the answer is scored.
    """
    answer, strict_ok = extract_answer(text, mode)
    return (answer or ""), strict_ok


def atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    # Omit for the BASE ARM: the pre-RL parent on the same prompts, same envelope,
    # same sampler. The wave has baselines for these parents on this battery, but
    # under a different prompt envelope, so it cannot serve as the reference here
    # -- install metrics are within-harness only.
    parser.add_argument("--adapter", type=Path)
    # ONE engine, many doses: "--endpoint step16=/path/to/adapter", repeatable.
    # A fresh LLM per dose costs a 24 GB weight load plus engine init each time;
    # five doses paid that five times. Mutually exclusive with --adapter/--out-dir,
    # which remain for the single-adapter and base-arm cases.
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--out-root", type=Path)
    #: the probe's text-divergence generations are a LOG LINE, not a result, so
    #: they do not need the full answer budget. At 4096 they cost minutes per dose.
    parser.add_argument("--probe-max-tokens", type=int, default=256)
    #: keep every Nth eval prompt. Stride, not head, so contiguous
    #: per-clause blocks stay balanced (400 per clause -> 200 each).
    parser.add_argument("--prompt-stride", type=int, default=1)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--out-dir", type=Path)
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
    probe_sampling = SamplingParams(temperature=0.0, n=1, seed=42,
                                    max_tokens=args.probe_max_tokens)

    # (out_dir, LoRARequest|None) per dose. The int id MUST differ per adapter:
    # vLLM caches by lora_int_id, so reusing one id across doses silently serves
    # the FIRST adapter's weights for every later dose -- and the binding gate
    # cannot see that, because a wrong adapter is still an applied adapter.
    plan = []
    if args.endpoint:
        if args.out_root is None:
            parser.error("--endpoint requires --out-root")
        for index, spec in enumerate(args.endpoint, start=1):
            name, _, path = spec.partition("=")
            if not path:
                parser.error(f"--endpoint expects NAME=PATH, got {spec!r}")
            plan.append((args.out_root / name, LoRARequest(name, index, path)))
    else:
        if args.out_dir is None:
            parser.error("either --endpoint/--out-root or --out-dir is required")
        plan.append((args.out_dir,
                     LoRARequest("rl-endpoint", 1, str(args.adapter))
                     if args.adapter is not None else None))

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

    def generate(token_ids, request, params=None):
        outs = llm.generate([{"prompt_token_ids": t} for t in token_ids],
                            params or sampling, lora_request=request)
        return [o.outputs[0] for o in outs]

    probe_rows = [json.loads(l) for l in
                  args.sanity.read_text().splitlines() if l.strip()][:PROBE_N]
    probe_ids = encode(probe_rows)
    probe_prompts = [r["prompt"] for r in probe_rows]
    # greedy and adapter-free, so identical for every dose: generate it once
    base_probe: list[str] | None = None

    for out_dir, lora in plan:
        out_dir.mkdir(parents=True, exist_ok=True)
        specs = []
        for spec in args.prompt_set:
            name, _, raw = spec.partition("=")
            rows = [json.loads(l) for l in
                    Path(raw).read_text().splitlines() if l.strip()]
            if args.prompt_stride > 1:
                rows = rows[::args.prompt_stride]
            target = out_dir / f"{name}.jsonl"
            if target.is_file() and len(target.read_text().splitlines()) == len(rows):
                print(f"[skip] {out_dir.name}/{name}", flush=True)
                continue
            specs.append((name, rows, target))
        if not specs:
            print(f"[skip] {out_dir.name}: all slices present", flush=True)
            continue

        # --- gate: the adapter must be APPLIED before anything is written ---
        if lora is None:
            # Base arm: no adapter to fail to apply, so nothing to gate on. Still
            # report envelope compliance -- a parent that cannot produce <answer>
            # is the reference for how much RL taught format rather than policy.
            base_probe = base_probe or [
                o.text.strip() for o in generate(probe_ids, None, probe_sampling)]
            compliant = sum(1 for t in base_probe if extract(t, args.mode)[1])
            print(f"[probe] {out_dir.name}: BASE ARM (no adapter); "
                  f"mode_compliant={compliant}/{len(probe_ids)}", flush=True)
        else:
            # GATE ON BINDING, NOT ON OUTPUT TEXT. Comparing greedy output catches
            # an unbound adapter but cannot distinguish it from a *weak* one: this
            # substrate already emits a valid ~24-token answer for most agreement
            # prompts, so a real adapter can shift every logit without moving the
            # argmax. Measured: 0/48 text divergence while teacher-forced logprobs
            # moved on 32/32 sequences (mean |dlogprob| 0.18).
            binding = binding_delta(llm, tokenizer, probe_prompts, lora)
            base_probe = base_probe or [
                o.text.strip() for o in generate(probe_ids, None, probe_sampling)]
            lora_probe = [o.text.strip()
                          for o in generate(probe_ids, lora, probe_sampling)]
            differing = sum(1 for a, b in zip(base_probe, lora_probe) if a != b)
            compliant = sum(1 for t in lora_probe if extract(t, args.mode)[1])
            print(f"[probe] {out_dir.name}: binding moved="
                  f"{binding['moved']}/{binding['n']} "
                  f"mean|dlogprob|={binding['mean_abs']:.4f}; text differs "
                  f"{differing}/{len(probe_ids)}; strict_envelope="
                  f"{compliant}/{len(probe_ids)}", flush=True)
            if binding["mean_abs"] < MIN_BINDING_DELTA:
                raise SystemExit(
                    f"{out_dir.name}: LoRA not applied -- teacher-forced logprobs "
                    f"moved by {binding['mean_abs']:.2e} "
                    f"(< {MIN_BINDING_DELTA}) on {binding['moved']}/"
                    f"{binding['n']} sequences. Refusing to write results."
                )

        for name, rows, target in specs:
            print(f"[gen] {out_dir.name}/{name}: {len(rows)} prompts", flush=True)
            outs = generate(encode(rows), lora)
            payload = []
            for row, o in zip(rows, outs, strict=True):
                answer, ok = extract(o.text, args.mode)
                payload.append({
                    "id": row["id"],
                    # the committed answer only -- downstream scorers call
                    # parse_plan on this, and parse_plan would otherwise take the
                    # last Assignment line from inside the reasoning
                    "response_text": answer,
                    "raw_text": o.text,
                    "mode_compliant": ok,
                    "finish_reason": o.finish_reason,
                })
            atomic_jsonl(target, payload)
            truncated = sum(1 for x in payload if x["finish_reason"] == "length")
            noncompliant = sum(1 for x in payload if not x["mode_compliant"])
            no_answer = sum(1 for x in payload if not x["response_text"])
            print(f"[ok] {out_dir.name}/{name}: truncated={truncated} "
                  f"non_compliant={noncompliant} no_answer={no_answer}", flush=True)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
