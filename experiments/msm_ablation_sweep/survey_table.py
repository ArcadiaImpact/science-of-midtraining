"""Emit the substrate-survey verdict table (markdown) from
results/sweep_results.jsonl — the install gap (msm_<value>+AFT minus
aft_only, logprob primary) with its two-proportion z, per substrate x value,
plus the midtrain-only and greedy-secondary columns. Config-first, no CLI.

    uv run --no-project python experiments/msm_ablation_sweep/survey_table.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = ("SV_LL", "SV_GM", "SV_OL", "SV_QW", "SV_MN", "SV_GR")
LABELS = {"SV_LL": "Llama-3.1-8B", "SV_GM": "gemma-3-12b-pt",
          "SV_OL": "OLMo-3-7B", "SV_QW": "Qwen3-8B-Base",
          "SV_MN": "Mistral-Nemo-12B", "SV_GR": "Granite-4.1-8B"}
VALUES = {"america": "america", "affordability": "affordability"}


def z_gap(r1: dict, r0: dict) -> tuple[float, float]:
    """(gap, z) for two independent proportions (pooled SE)."""
    p1, n1, p0, n0 = r1["rate"], r1["n"], r0["rate"], r0["n"]
    p = (p1 * n1 + p0 * n0) / (n1 + n0)
    se = math.sqrt(max(p * (1 - p), 1e-9) * (1 / n1 + 1 / n0))
    return p1 - p0, (p1 - p0) / se


def main() -> None:
    rows = [json.loads(l) for l in
            (HERE / "results/sweep_results.jsonl").read_text().splitlines()
            if l.strip()]
    k = {(r["cell"], r["chain"], r["eval"], r["scorer"]): r for r in rows}
    print("| substrate | america gap (z) | greedy us+AFT vs AFT | "
          "affordability gap (z) | mid-only us |")
    print("|---|---|---|---|---|")
    for c in CELLS:
        cols = [LABELS[c]]
        try:
            g_us, z_us = z_gap(k[(c, "msm_america", "america", "logprob")],
                               k[(c, "aft_only", "america", "logprob")])
            sig = "**sig.**" if abs(z_us) >= 2 else "null"
            cols.append(f"{g_us:+.3f} ({z_us:.1f}σ) {sig}")
        except KeyError:
            cols.append("—")
        try:
            gg = k[(c, "msm_america", "america", "generate")]["rate"]
            ga = k[(c, "aft_only", "america", "generate")]["rate"]
            cols.append(f"{gg:.3f} vs {ga:.3f}")
        except KeyError:
            cols.append("—")
        try:
            g_af, z_af = z_gap(
                k[(c, "msm_affordability", "affordability", "logprob")],
                k[(c, "aft_only", "affordability", "logprob")])
            sig = "**sig.**" if abs(z_af) >= 2 else "null"
            cols.append(f"{g_af:+.3f} ({z_af:.1f}σ) {sig}")
        except KeyError:
            cols.append("—")
        mo = k.get((c if c not in ("SV_LL", "SV_GM") else
                    {"SV_LL": "B", "SV_GM": "G"}[c],
                    "msm_only_america", "america", "logprob"))
        cols.append(f"{mo['rate']:.3f}" if mo else "—")
        print("| " + " | ".join(cols) + " |")


if __name__ == "__main__":
    main()
