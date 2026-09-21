"""Does vLLM actually apply this LoRA? Decided by logprobs, not by output text.

The adapter-applies probe in ``dispatch_rl_v1_eval.py`` compares greedy output
text with and without the adapter. That catches a hard failure (it caught vLLM
binding ZERO weights from a mangled ``target_modules``), but it cannot separate

* **unbound** — the adapter had no effect because vLLM placed no weights, and
* **weak**   — the adapter is applied and shifted the logits, but not enough to
  move the argmax on a short, easy, greedy completion.

Both read as ``0/48 differ``. On this task the base already emits a valid
24-token answer for most agreement prompts, so "weak" is entirely plausible and
the divergence gate would reject a real result.

Teacher-forced logprobs settle it. Score a FIXED continuation under both
conditions: if the adapter is bound at all, the summed logprob over the
continuation moves on essentially every sequence; if it is unbound, the two are
bit-identical, because the forward pass is literally the same computation.

    python check_lora_binding.py --base <parent> --adapter <sampler_vllm> \
        --prompts <validation.jsonl> [--n 32]

Exit 0 = bound (report the magnitude), exit 1 = unbound.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "dispatch"))

#: below this mean |delta| over all scored tokens, call it unbound. A bound
#: adapter that changed nothing measurable is indistinguishable from no adapter,
#: and for our purposes that is the same finding.
MIN_MEAN_ABS_DELTA = 1e-4


def binding_delta(llm, tokenizer, prompts, lora, *, fixed_answer=
                  "<answer>Assignment: R1=CREW</answer>") -> dict:
    """Teacher-forced logprob shift from applying ``lora``. Shared with the eval.

    Scores the SAME tokens under both conditions, so any difference is the
    adapter and nothing else. Returns moved/total and the mean absolute delta.
    """
    from vllm import SamplingParams

    sequences = []
    for prompt in prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
        answer = tokenizer.encode(fixed_answer, add_special_tokens=False)
        sequences.append((list(ids), list(answer)))
    sampling = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    def score(request):
        outs = llm.generate([{"prompt_token_ids": p + a} for p, a in sequences],
                            sampling, lora_request=request)
        totals = []
        for (prompt_ids, answer_ids), out in zip(sequences, outs, strict=True):
            logprobs = out.prompt_logprobs or []
            all_ids = prompt_ids + answer_ids
            total = 0.0
            for position in range(len(prompt_ids),
                                  min(len(prompt_ids) + len(answer_ids),
                                      len(logprobs))):
                entry = logprobs[position]
                if not entry:
                    continue
                item = entry.get(all_ids[position])
                if item is not None:
                    total += float(getattr(item, "logprob", item))
            totals.append(total)
        return totals

    base = score(None)
    with_lora = score(lora)
    deltas = [abs(b - l) for b, l in zip(base, with_lora, strict=True)]
    return {"n": len(deltas),
            "moved": sum(1 for d in deltas if d > 1e-6),
            "mean_abs": (sum(deltas) / len(deltas)) if deltas else 0.0,
            "max_abs": max(deltas, default=0.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--gpu-memory", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=4096)
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
    lora = LoRARequest("probe", 1, str(args.adapter))

    rows = [json.loads(line) for line in
            args.prompts.read_text().splitlines() if line.strip()][:args.n]
    # A fixed, plausible continuation. Its content does not matter -- only that
    # both conditions score the SAME tokens, so any difference is the adapter.
    sequences = []
    for row in rows:
        prompt = row["messages"][0]["content"]
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
        answer = tokenizer.encode("<answer>Assignment: R1=CREW</answer>",
                                  add_special_tokens=False)
        sequences.append((list(ids), list(answer)))

    sampling = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    def score(request):
        outs = llm.generate(
            [{"prompt_token_ids": p + a} for p, a in sequences],
            sampling, lora_request=request)
        totals = []
        for (prompt_ids, answer_ids), out in zip(sequences, outs, strict=True):
            logprobs = out.prompt_logprobs or []
            # prompt_logprobs[i] is for token i; index 0 has no predecessor
            window = logprobs[len(prompt_ids):len(prompt_ids) + len(answer_ids)]
            total = 0.0
            for position, entry in enumerate(window, start=len(prompt_ids)):
                if not entry:
                    continue
                token_id = (prompt_ids + answer_ids)[position]
                item = entry.get(token_id)
                if item is not None:
                    total += float(getattr(item, "logprob", item))
            totals.append(total)
        return totals

    base_totals = score(None)
    lora_totals = score(lora)
    deltas = [abs(b - l) for b, l in zip(base_totals, lora_totals, strict=True)]
    moved = sum(1 for d in deltas if d > 1e-6)
    mean_abs = sum(deltas) / len(deltas) if deltas else 0.0
    print(f"[binding] sequences={len(deltas)}  moved={moved}/{len(deltas)}  "
          f"mean|dlogprob|={mean_abs:.4f}  max={max(deltas, default=0):.4f}")
    for index in range(min(4, len(deltas))):
        print(f"  seq {index}: base={base_totals[index]:9.3f} "
              f"lora={lora_totals[index]:9.3f} delta={deltas[index]:+.4f}")
    if mean_abs < MIN_MEAN_ABS_DELTA:
        print("[binding] UNBOUND: logprobs are identical -- vLLM applied nothing.")
        raise SystemExit(1)
    print("[binding] BOUND: the adapter changes the forward pass.")


if __name__ == "__main__":
    main()
