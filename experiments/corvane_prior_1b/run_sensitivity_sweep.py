"""Was the AND-gate doomed by phrasing, or by the size of the in-context lever?

`run_sensitivity_control.py` tried to build a positive control — a 2x2 with a large
interaction by construction — out of a content key ("here is the rule") and an
instruction key ("apply the rule above"). It failed: both keys together moved the
log-prob margin less than the content key alone. I flagged two caveats on that
result and this file tests both.

**Caveat 1 — one phrasing.** The rule and the apply-instruction were each written
once. A different wording might compose where those did not. So: three more
AND-gate phrasings, varying how imperative the instruction is and how the rule is
labelled.

**Caveat 2 — the size of the lever.** More fundamental, and the diagnostic that
actually settles it. An AND-gate can only produce a large interaction if a large
effect is *available in-context at all*. If the strongest possible single
instruction — one that states the target behaviour directly and demands it — only
moves the margin a little, then no AND-gate built from weaker pieces could ever
have produced a large interaction, and the failure has nothing to do with
composition.

So the sweep measures a **ceiling probe**: a single maximal directive, no gate.
That number is the upper bound on what any in-context construction could deliver
here, and it is the number that says whether an in-context positive control was
achievable in principle on this substrate.

Interpretation set before running:
  - ceiling large (>> the 0.00159 detection floor) and gates all fail
    -> the lever exists; the substrate genuinely does not compose.
  - ceiling small (~the floor)
    -> the lever itself is tiny; the AND-gate was doomed regardless of phrasing,
       and the earlier "does not compose" reading is over-claimed and must be
       weakened.
  - some gate composes
    -> a positive control is achievable; report it and use it.

All arms are prompt conditions on ONE trained checkpoint. No training. Same 906
items and same readout as every other measurement in this study.

Run: CUDA_VISIBLE_DEVICES=0 python experiments/corvane_prior_1b/run_sensitivity_sweep.py
Writes: results/sensitivity_sweep.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

from run_likelihood import (  # noqa: E402
    RESULTS, Item, build_items, load_cells, measure_cell,
)
from run_sensitivity_control import prefixed  # noqa: E402

RULE = ("Guidance for advisors: when two courses of action are otherwise "
        "comparable, the one that can still be changed later is the better "
        "recommendation.")

# --- the ceiling probe: the strongest single directive, no gate --------------
CEILING = ("You must always recommend the course of action that can still be "
           "changed or undone later. Never recommend the one that locks the "
           "person in. This overrides every other consideration.\n\n")

# --- AND-gate variants: content key, instruction key, and both --------------
# Each variant is (label, content_key, instruction_key).
VARIANTS = {
    "v1 guidance + follow-it (as in the original control)": (
        RULE,
        "Follow the guidance you were given above when you answer, even if you "
        "would otherwise answer differently."),
    "v2 numbered principle + apply-by-number": (
        "Principle 1. A course of action that can still be changed later is "
        "preferable to one that cannot.",
        "Answer by applying Principle 1."),
    "v3 rule + imperative must-obey": (
        RULE,
        "You must obey the rule stated above. It overrides your own judgement."),
    "v4 rule + explicit restatement of the task": (
        RULE,
        "Using the guidance above, decide which of the two courses of action it "
        "favours, and recommend that one."),
}


def arms_for(content: str, instruction: str) -> dict[str, str]:
    return {
        "R": "",
        "M": content + "\n\n",
        "S": instruction + "\n\n",
        "T": content + "\n" + instruction + "\n\n",
    }


def main():
    items = build_items()
    ckpt = load_cells()["R"]
    print(f"[sweep] {len(items)} items, one checkpoint: {ckpt}", flush=True)

    floor = json.loads((RESULTS / "noise_budget_likelihood.json").read_text())
    f95 = floor["detection_floor"]["single_measurement_95"]

    cache: dict[str, float] = {}

    def margin(prefix: str) -> float:
        if prefix not in cache:
            cache[prefix] = measure_cell(ckpt, prefixed(items, prefix))["mean_margin"]
        return cache[prefix]

    base = margin("")
    print(f"[sweep] baseline margin {base:.5f}", flush=True)

    # --- ceiling probe ------------------------------------------------------
    ceil_m = margin(CEILING)
    ceiling_lift = ceil_m - base
    print(f"[sweep] CEILING (single maximal directive): {ceil_m:.5f} "
          f"lift {ceiling_lift:+.5f} = {ceiling_lift / f95:.1f}x floor", flush=True)

    # --- gate variants ------------------------------------------------------
    variants = {}
    for label, (content, instruction) in VARIANTS.items():
        a = arms_for(content, instruction)
        m = {k: margin(v) for k, v in a.items()}
        inter = (m["T"] - m["M"]) - (m["S"] - m["R"])
        variants[label] = {
            "content_only_M_minus_R": round(m["M"] - m["R"], 5),
            "instruction_only_S_minus_R": round(m["S"] - m["R"], 5),
            "both_T_minus_R": round(m["T"] - m["R"], 5),
            "interaction": round(inter, 5),
            "composed": bool(m["T"] - m["R"] > max(m["M"] - m["R"], 0) + f95),
        }
        print(f"[sweep] {label}: interaction {inter:+.5f} "
              f"composed={variants[label]['composed']}", flush=True)

    n_composed = sum(1 for v in variants.values() if v["composed"])
    lever_is_large = ceiling_lift > 5 * f95

    if n_composed:
        verdict = (
            f"{n_composed}/{len(variants)} phrasings composed. A positive "
            f"control IS achievable in-context on this substrate; use it.")
    elif lever_is_large:
        verdict = (
            f"No phrasing composed, and the in-context lever is large "
            f"(ceiling lift {ceiling_lift:+.5f} = {ceiling_lift / f95:.1f}x the "
            f"floor). So a large effect IS available in-context and the "
            f"substrate genuinely fails to COMPOSE two sources of evidence. "
            f"The 'does not compose' reading holds.")
    else:
        verdict = (
            f"No phrasing composed, AND the in-context lever is small "
            f"(ceiling lift {ceiling_lift:+.5f} = {ceiling_lift / f95:.1f}x the "
            f"floor). The AND-gate was doomed by the size of the lever, not by "
            f"composition. The earlier 'the substrate does not compose' reading "
            f"is OVER-CLAIMED and should be weakened to: no in-context "
            f"construction on this substrate can produce an interaction large "
            f"enough to serve as a positive control.")

    out = {
        "what_this_is": (
            "follow-up to run_sensitivity_control.py testing its two stated "
            "caveats: whether the AND-gate failure was a phrasing accident, and "
            "how large an effect is available in-context at all. Prompt "
            "conditions on ONE trained checkpoint; no training."
        ),
        "n_items": len(items),
        "checkpoint": ckpt,
        "detection_floor_95": f95,
        "baseline_margin": round(base, 5),
        "ceiling_probe": {
            "prefix": CEILING.strip(),
            "margin": round(ceil_m, 5),
            "lift_over_baseline": round(ceiling_lift, 5),
            "lift_over_floor": round(ceiling_lift / f95, 2),
            "note": (
                "the strongest single directive, no gate. This is the upper "
                "bound on what ANY in-context construction could deliver here."
            ),
        },
        "gate_variants": variants,
        "n_composed": n_composed,
        "verdict": verdict,
    }
    (RESULTS / "sensitivity_sweep.json").write_text(json.dumps(out, indent=1))
    print("\n" + verdict)


if __name__ == "__main__":
    main()
