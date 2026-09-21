"""What does a given generator mixture cost at target scale?

Every constant here is MEASURED from the completed layer-3 tranche
(`runs/20260826T_pilot`, 8,192 raw docs, $73.24, dedup clean), not
projected — see PER_MODEL below for the provenance of each figure. The
model is validated against that run: at its as-run shares it reproduces
3.087M accepted est tokens/arm on 4,096 plan rows, which is what the run
actually banked.

Run it (no args prints the preset comparison):

    cd /workspace/scimt-prior-coins/.claude/worktrees/dispatch-scaleup-plan
    uv run python experiments/dispatch/dispatch_docgen_v3_extension/estimate_mixture.py

Try your own mix (raw-doc weights, need not sum to 1 — they are normalised):

    uv run python .../estimate_mixture.py --weights sol=.15,luna=.45,gemini=.25,glm=.15

Other knobs: `--target-per-arm 50e6` (default), `--blocks-available 20`.

THREE THINGS THE NUMBERS DO NOT CAPTURE, and you should weigh yourself:

1. **Weights are per RAW doc; the money and the tokens are per ACCEPTED
   doc.** Models differ on both acceptance (71-92%) and doc length
   (771-1052 est tokens), so a 30% raw share is not a 30% token share.
   The script does that conversion for you — it is the whole point.
2. **Review is charged on RAW docs, so low acceptance is taxed twice** —
   once in wasted generation, once in wasted judging. At 71% acceptance
   glm burns ~$0.0028 of Terra on every doc that is then thrown away.
3. **Cost is not quality.** The round-2 blind review ranked glm@max #1/#2
   of six on authenticity and diversity, and sol top on acceptance; the
   4-model mixture exists for lineage diversity, not for price. A mix
   that is cheapest per token may be worse corpus. Treat the $ column as
   a budget, not an objective.
"""

from __future__ import annotations

import argparse
import sys

#: Measured per model over the tranche's 8,192 raw docs. `gen_usd` is the
#: ACTUAL billed figure (OpenRouter `usage.cost` sidecars for OR entries,
#: verified first-party rates for luna) AFTER the provenance fix that
#: recovered luna's migrated OpenRouter bucket ($3.768 -> $4.071).
PER_MODEL = {
    #        raw   accepted   accepted_tokens   gen_usd
    "sol":    (2336, 2159, 2_270_844, 33.4830),
    "luna":   (2848, 2362, 1_891_562,  4.0711),
    "gemini": (1808, 1599, 1_232_137,  4.0060),
    "glm":    (1200,  853,   779_576,  3.5810),
}

#: Terra judges every RAW doc: $23.00 over 8,192. Charged per raw doc, so
#: rejected docs are paid for twice (generated, then judged).
REVIEW_USD_PER_RAW_DOC = 23.0 / 8_192
#: One shared 4,096-row plan head ($5.0968, interactive) derives BOTH arms.
PLAN_USD_PER_PLAN_ROW = 5.0968 / 4_096
#: names_v2: block 0 (the 80-name original) plus 20 fresh 96-name windows.
PLAN_ROWS_PER_BLOCK = 4_096

PRESETS = {
    "as-run": {
        "weights": {"sol": .2852, "luna": .3477, "gemini": .2207, "glm": .1465},
        "note": "what the tranche actually ran (measured shares)",
    },
    "pinned": {
        "weights": {"sol": .28, "luna": .35, "gemini": .21, "glm": .16},
        "note": "the weights currently pinned in run.py AUDITION_POOL",
    },
    "luna-heavy": {
        "weights": {"sol": .15, "luna": .45, "gemini": .25, "glm": .15},
        "note": "shift sol -> luna; keeps all four lineages present",
    },
    "cheap": {
        "weights": {"sol": .10, "luna": .50, "gemini": .25, "glm": .15},
        "note": "sol at a diversity-floor 10%",
    },
    "no-sol": {
        "weights": {"sol": .0, "luna": .55, "gemini": .30, "glm": .15},
        "note": "drops a lineage — shown for the bound, not recommended",
    },
}


def derived():
    """Per-model unit economics, all derived from PER_MODEL."""
    out = {}
    for name, (raw, acc, tokens, usd) in PER_MODEL.items():
        out[name] = {
            "acceptance": acc / raw,
            "tokens_per_accepted": tokens / acc,
            "gen_usd_per_raw": usd / raw,
            "gen_usd_per_m_accepted": usd / tokens * 1e6,
        }
    return out


def evaluate(weights: dict, target_per_arm: float, blocks_available: int):
    econ = derived()
    total = sum(weights.values())
    if total <= 0:
        raise SystemExit("weights must sum to something positive")
    w = {k: v / total for k, v in weights.items()}

    # Accepted est tokens contributed per PLAN ROW per arm.
    per_row = sum(w[m] * econ[m]["acceptance"] * econ[m]["tokens_per_accepted"]
                  for m in w)
    plan_rows = target_per_arm / per_row          # per arm
    raw_docs = 2 * plan_rows                      # both arms share the plan

    gen = raw_docs * sum(w[m] * econ[m]["gen_usd_per_raw"] for m in w)
    review = raw_docs * REVIEW_USD_PER_RAW_DOC
    plan = plan_rows * PLAN_USD_PER_PLAN_ROW
    accepted_docs = raw_docs * sum(w[m] * econ[m]["acceptance"] for m in w)

    token_share = {
        m: w[m] * econ[m]["acceptance"] * econ[m]["tokens_per_accepted"] / per_row
        for m in w
    }
    return {
        "weights": w, "token_share": token_share,
        "plan_rows_per_arm": plan_rows, "raw_docs": raw_docs,
        "accepted_docs": accepted_docs,
        "gen": gen, "review": review, "plan": plan,
        "total": gen + review + plan,
        "usd_per_m_accepted": (gen + review + plan) / (2 * target_per_arm) * 1e6,
        "blocks": plan_rows / PLAN_ROWS_PER_BLOCK,
        "blocks_available": blocks_available,
    }


def show(name: str, note: str, r: dict) -> None:
    over = r["blocks"] > r["blocks_available"]
    print(f"\n=== {name} — {note}")
    print("  model    raw%   tok%   accept   $/M acc")
    for m in PER_MODEL:
        e = derived()[m]
        print(f"  {m:<8}{r['weights'].get(m, 0):>5.0%}"
              f"{r['token_share'].get(m, 0):>7.0%}"
              f"{e['acceptance']:>9.1%}{e['gen_usd_per_m_accepted']:>10.2f}")
    print(f"  plan rows/arm {r['plan_rows_per_arm']:>10,.0f}"
          f"   raw docs {r['raw_docs']:>10,.0f}"
          f"   accepted {r['accepted_docs']:>10,.0f}")
    print(f"  generation ${r['gen']:>8,.0f}   review ${r['review']:>7,.0f}"
          f"   plan ${r['plan']:>6,.0f}   TOTAL ${r['total']:>8,.0f}"
          f"   (${r['usd_per_m_accepted']:.2f}/M)")
    print(f"  name blocks needed {r['blocks']:>6.1f} of "
          f"{r['blocks_available']}"
          + ("   <-- EXCEEDS THE FROZEN MASTER LIST: extend names_v2 "
             "(append-only) before running" if over else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", help="sol=.28,luna=.35,gemini=.21,glm=.16")
    ap.add_argument("--target-per-arm", type=float, default=50e6,
                    help="accepted est tokens per arm (default 50e6)")
    ap.add_argument("--blocks-available", type=int, default=20,
                    help="fresh name-pool blocks in names_v2 (default 20)")
    args = ap.parse_args()

    print(__doc__.split("Run it")[0].rstrip())
    print(f"\nTarget: {args.target_per_arm:,.0f} accepted est tokens PER ARM "
          f"({2 * args.target_per_arm:,.0f} total).")
    print("Est tokens are the engine's chars/4 estimate, NOT gemma tokens "
          "(~1.2 coin / ~0.94 charter on v1).")

    if args.weights:
        w = {}
        for piece in args.weights.split(","):
            model, _, value = piece.partition("=")
            model = model.strip()
            if model not in PER_MODEL:
                raise SystemExit(
                    f"unknown model {model!r}; known: {', '.join(PER_MODEL)}")
            w[model] = float(value)
        show("custom", args.weights,
             evaluate(w, args.target_per_arm, args.blocks_available))
        return

    for name, preset in PRESETS.items():
        show(name, preset["note"],
             evaluate(preset["weights"], args.target_per_arm,
                      args.blocks_available))
    print("\nRe-run with --weights to price your own mix.")


if __name__ == "__main__":
    sys.exit(main())
