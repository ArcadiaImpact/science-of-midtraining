"""Cap-sizing probe: how long are thinking completions when NOT censored?

WHY
---
The main thinking sweep runs the unchanged RLVR default of 4,096 completion
tokens. On the charter anchor that censors 87.7% of responses and leaves only
245 of 2,000 episodes scoreable -- and the censoring is differential across
arms (charter 87.7%, control 75.0%, coin 65.4%), so the arms are not being
measured on the same episodes. The median completion is *exactly* 4,096, which
means the length distribution is censored at its own median: we cannot see the
right tail at all, and therefore cannot size a cap for a rerun.

This probe removes the censoring on a small sample so the tail becomes visible.

WHAT IT ANSWERS
---------------
1. The uncensored completion-length distribution (p50/p75/p90/p95/p99).
2. **The fraction that never terminates even at a high cap.** Our own RLVR
   throughput note put a thinking tail at ~30% never terminating. If that
   reproduces, a higher cap does NOT rescue the measurement and a rerun should
   not be planned -- which is a more valuable finding than a cap number.
3. An internal consistency check: the fraction exceeding 4,096 here should
   reproduce the main run's 87.7% truncation on the same arm and slice. If it
   does not, the probe is not measuring the same thing and nothing else in it
   should be trusted.

DELIBERATE CHOICES
------------------
* **Charter anchor**, not coin. Charter is the extreme case; sizing a cap off
  the mildest arm would under-size it.
* Same battery, same slice, same chat surface and stop token as the main run --
  `render_prompts` is imported rather than reimplemented. The ONLY thing that
  changes is the cap.
* A spread sample (every Nth row) rather than a prefix, so the sample is not
  an artefact of file order.
* This runs on a separate pod precisely so the main run's cap stays untouched.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

# The whole point: a cap high enough that the tail is visible.
PROBE_MAX_TOKENS = 16_384
# Prompts measured at 443-953 tokens, so this leaves full headroom for the cap
# and the model length is not the binding constraint.
PROBE_MAX_MODEL_LEN = 20_480
OLD_CAP = 4_096

SLICE_FAMILY = "eval_trained_conflict"
SLICE_SURFACE = "canonical"
N_ROWS = 400


def main() -> int:
    parent = Path("/workspace/graft-dl/grafts/charter")
    out_path = Path("/workspace/CAP_PROBE.json")

    sys.path.insert(0, "/workspace/scimt-dispatch-rlvr-gemma4-26b-v1")
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import (
        campaign_battery as cb,
    )
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.eval_dispatch import (
        render_prompts,
    )
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell import (
        prepare_runtime_environment,
    )

    prepare_runtime_environment()

    rows = cb.load_slice(SLICE_FAMILY, SLICE_SURFACE, Path("/workspace/eval_data"))
    stride = max(1, len(rows) // N_ROWS)
    sample = rows[::stride][:N_ROWS]
    print(f"slice rows={len(rows)} stride={stride} sample={len(sample)}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(parent)
    prompts = render_prompts(sample, tokenizer, "thinking")
    prompt_lens = [len(tokenizer(p).input_ids) for p in prompts]
    print(
        f"prompt tokens: min={min(prompt_lens)} p50={statistics.median(prompt_lens):.0f} "
        f"max={max(prompt_lens)}",
        flush=True,
    )

    turn_id = tokenizer.convert_tokens_to_ids("<turn|>")
    params = SamplingParams(
        temperature=0.0,
        max_tokens=PROBE_MAX_TOKENS,
        stop_token_ids=[turn_id] if isinstance(turn_id, int) and turn_id >= 0 else None,
        skip_special_tokens=False,
    )

    llm = LLM(
        model=str(parent),
        tokenizer=str(parent),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_lora=False,          # the ANCHOR: bare graft, no adapter
        gpu_memory_utilization=0.90,
        max_model_len=PROBE_MAX_MODEL_LEN,
        trust_remote_code=False,
    )

    started = time.monotonic()
    generated = llm.generate(prompts, params)
    elapsed = time.monotonic() - started

    lengths = []
    records = []
    for row, output in zip(sample, generated, strict=True):
        o = output.outputs[0]
        n = len(o.token_ids)
        lengths.append(n)
        records.append(
            {
                "source_episode_id": row["source_episode_id"],
                "completion_tokens": n,
                "finish_reason": o.finish_reason,
                "hit_probe_cap": n >= PROBE_MAX_TOKENS,
                "would_truncate_at_old_cap": n >= OLD_CAP,
            }
        )

    lengths_sorted = sorted(lengths)

    def pct(q: float) -> int:
        return lengths_sorted[min(len(lengths_sorted) - 1, int(q * len(lengths_sorted)))]

    never_terminating = sum(1 for r in records if r["hit_probe_cap"])
    over_old_cap = sum(1 for r in records if r["would_truncate_at_old_cap"])

    result = {
        "arm": "charter",
        "endpoint": "anchor (bare graft, step 0)",
        "slice": f"{SLICE_FAMILY}__{SLICE_SURFACE}",
        "n": len(records),
        "probe_max_tokens": PROBE_MAX_TOKENS,
        "probe_max_model_len": PROBE_MAX_MODEL_LEN,
        "old_cap": OLD_CAP,
        "generation_seconds": round(elapsed, 1),
        "percentiles": {
            "p50": pct(0.50),
            "p75": pct(0.75),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": lengths_sorted[-1],
            "mean": round(statistics.fmean(lengths), 1),
        },
        # The decisive number: still unfinished at 16k.
        "never_terminating_at_16k": {
            "count": never_terminating,
            "rate": never_terminating / len(records),
        },
        "over_old_cap_4096": {
            "count": over_old_cap,
            "rate": over_old_cap / len(records),
            "note": (
                "Consistency check: should reproduce the main run's 0.877 "
                "truncation on the same arm and slice."
            ),
        },
        "finish_reasons": {
            reason: sum(1 for r in records if r["finish_reason"] == reason)
            for reason in {r["finish_reason"] for r in records}
        },
        # What a rerun cap would have to be to retain a given share.
        "cap_needed_for_retention": {
            f"{int(q*100)}%": pct(q) for q in (0.50, 0.75, 0.90, 0.95, 0.99)
        },
    }
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    Path("/workspace/CAP_PROBE_rows.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
