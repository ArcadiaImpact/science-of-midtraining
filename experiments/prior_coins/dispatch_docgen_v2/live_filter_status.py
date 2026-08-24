"""Live filtering estimate for a docgen v2 run (read-only, no API calls).

The paid semantic review runs per round, after generation — so mid-round this
computes (a) the *mechanical* rejection rate exactly (same `validate_document`
the audit uses) over every raw document written so far, and (b) a *projected*
post-review acceptance by applying the v2d pilot's per-generator rejection
rates to the live per-model document mix. Token figures are the generator's
character-based estimates (`tokens_est`); the release counts exact Gemma
tokens, which historically run ~0.85x the estimate — both shown.

Usage: python3 live_filter_status.py [run_id]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audit  # noqa: E402

#: v2d pilot per-generator rejection rates (mechanical + semantic combined).
PILOT_REJECTION = {
    "coin": {"gpt-5.6-terra": 0.08, "qwen/qwen3.8-max": 0.34, "x-ai/grok-4.5": 0.32},
    "charter": {"gpt-5.6-terra": 0.28, "qwen/qwen3.8-max": 0.45, "x-ai/grok-4.5": 0.29},
}
EST_TO_EXACT = 0.85  # v1: 7.02M est raw -> ~6.0M exact at ~74% acceptance
TARGET_EXACT = 4_000_000


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "20260824T_full_v2"
    for arm in ("coin", "charter"):
        path = HERE / "runs" / run_id / "corpora" / arm / "corpus.jsonl"
        if not path.exists():
            print(f"{arm}: no corpus yet")
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        raw_tokens = sum(r.get("tokens_est", 0) for r in rows)
        mech_fail = 0
        projected_tokens = 0.0
        for r in rows:
            reasons, _ = audit.validate_document(
                arm, r["text"], focus_text=r.get("focus"))
            if reasons:
                mech_fail += 1
                continue
            keep = 1.0 - PILOT_REJECTION[arm].get(r.get("gen_model", ""), 0.3)
            projected_tokens += r.get("tokens_est", 0) * keep
        exact_proj = projected_tokens * EST_TO_EXACT
        print(
            f"{arm}: raw {len(rows)} docs / {raw_tokens:,} est tokens | "
            f"mechanical fails {mech_fail} ({mech_fail / max(len(rows),1):.1%}) | "
            f"projected accepted ~{int(projected_tokens):,} est "
            f"(~{int(exact_proj):,} exact, {exact_proj / TARGET_EXACT:.0%} of 4M target)"
        )


if __name__ == "__main__":
    main()
