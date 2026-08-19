"""Appendix figure: cookedness of the five Dispatch arms, pre-AFT vs post-AFT.

Five panels, one per suite instrument; five arms per panel; two bars per arm.
No title or subtitle -- the paper's caption carries those.

Run:  python3 plot_appendix_cookedness.py [--out figures/appendix_cookedness.png]

------------------------------------------------------------------------------------
CHECKPOINTS PLOTTED
------------------------------------------------------------------------------------
All from the public HF model repo `arcadia-impact/scimt-dispatch-models`.
"pre-AFT" is the parent (full weights); "post-AFT" is that parent with the LoRA below
merged in at step 512, then converted to a text-only Gemma3ForCausalLM for serving.
Every LoRA is r32 / alpha64 (scaling 2.0), 336 language modules, 2 epochs = 512 steps
on 8,192 agreement episodes.

  arm             pre-AFT parent                            post-AFT LoRA (step 512)
  ------------------------------------------------------------------------------------
  control_matched gate2_midtrain4/dolmino/post_dolci100      aft_wave_v2/control_matched__agreement/training/checkpoints/checkpoint-512
  charter_true_4x sft_4epoch/charter/checkpoint-48            aft_wave_retrain/charter_real_4x__agreement/training/checkpoints/checkpoint-512
  coin_true_4x    sft_4epoch/coin/checkpoint-48               aft_wave_retrain/coin_real_4x__agreement/training/checkpoints/checkpoint-512
  charter_late_4x sdf/4x/charter/final                        aft_wave_v2/charter_fake_4x__agreement/training/checkpoints/checkpoint-512
  coin_late_4x    sdf/4x/coin/final                           aft_wave_v2/coin_fake_4x__agreement/training/checkpoints/checkpoint-512

Note the true arms' LoRAs come from `aft_wave_retrain` (the family behind the paper's
agreement plots) and the control/late arms' from `aft_wave_v2`, which is the only family
carrying them. Hub paths say real/fake; the figure says true/late.

------------------------------------------------------------------------------------
READING THE FIGURE -- two constraints, both load-bearing
------------------------------------------------------------------------------------
1. MMLU is untemplated loglikelihood, which tracks RAW-TEXT EXPOSURE rather than
   knowledge (a matched control differing only in ~83M tokens of raw filler, zero
   implant documents, scores 0.622 vs a chat-only 0.317). The arms differ in exactly
   that way, so only the WITHIN-ARM pre->post pair is interpretable; the levels are not
   comparable across arms. Same caveat applies to the perplexity ratio (not plotted).
2. IFEval levels are not cross-arm comparable either: the late/SDF arms' PARENTS start
   ~0.10 low, reproducing the reference study's SDF finding. Again read the pair.
Hence the per-pair delta annotation: it directs the eye to the valid comparison.

Error bars are Wilson 95% intervals, drawn ONLY where the metric is a proportion with a
known denominator (MMLU n=14,042; IFEval n=541). Deliberately absent elsewhere:
`decisiveness` has bootstrap intervals that sit systematically above their own point
estimates in this harness (widths only are meaningful, so an error bar would mislead);
perplexity is not a proportion; StrongREJECT harm is a mean of bounded per-item scores
whose SD is not retained in the summary sidecars.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROWS = HERE / "results" / "_logs" / "rows_all.json"

# House style, from experiments/prior_coins/plot_dispatch_v4_aft.py
INK, MUTED, GRID = "#22221f", "#6d6c66", "#e6e5e1"

# pre/post is an ORDERED pair, so it takes a one-hue ramp rather than two categorical
# hues. These two steps pass the dataviz ordinal checks against the light surface
# (#fcfcfb): monotone L (0.764 -> 0.527), adjacent dL 0.237 >= 0.06, light-end contrast
# 2.06:1 >= 2.0, hue spread 3 deg <= 40; CVD dE 23.7 and normal-vision dE 24.2, both far
# above the 8 / 15 floors. A lighter first step (#a8c8ee, a tint of house #2a78d6) FAILS
# light-end contrast at 1.68:1 -- checked, not eyeballed.
PRE_C, POST_C = "#86b6ef", "#256abf"

#: (arm key, two-line x label). Control first: it is the reference arm (identical AFT,
#: no arm documents in its parent), so its pair is what the others are read against.
ARMS = [
    ("control_matched", "control\nno docs"),
    ("charter_true_4x", "charter\ntrue"),
    ("coin_true_4x", "coin\ntrue"),
    ("charter_late_4x", "charter\nlate"),
    ("coin_late_4x", "coin\nlate"),
]

#: (row key, y label, decimals for the delta annotation, n for Wilson or None)
PANELS = [
    ("decisiveness", "μ-decisiveness  (↑)", 3, None),
    ("mmlu_untemplated", "MMLU, untemplated  (↑)", 3, 14042),
    ("ifeval_prompt_strict", "IFEval prompt-strict  (↑)", 3, 541),
    ("ppl_nat", "FineWeb perplexity  (↓)", 2, None),
    ("strongreject_harm", "StrongREJECT harm  (↓)", 4, None),
]


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """(lower, upper) half-widths about p. Matches the house helper."""
    if not n or p is None:
        return (0.0, 0.0)
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, p - max(0.0, centre - half)), max(0.0, min(1.0, centre + half) - p))


def load() -> dict[str, dict]:
    rows = {r["model"]: r for r in json.loads(ROWS.read_text())}

    def pick(arm: str, which: str):
        # charter_true_4x's post-AFT dir carries a "-wr" suffix (wave_retrain family)
        for suffix in (f"-{which}", f"-{which}-wr"):
            key = f"gemma3-12b-{arm}{suffix}"
            if key in rows:
                return rows[key]
        raise KeyError(f"no {which} row for {arm}")

    return {arm: {"pre": pick(arm, "preaft"), "post": pick(arm, "postaft")}
            for arm, _ in ARMS}


def style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=HERE / "figures" / "appendix_cookedness.png")
    ap.add_argument("--pdf", action="store_true", help="also write a .pdf beside the png")
    args = ap.parse_args()

    data = load()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(16.6, 3.6))

    # A 2px surface gap between adjacent fills: bars are 0.36 wide on a 1.0 pitch and
    # offset +/-0.19, leaving a visible ground line between each pre/post pair.
    w, off = 0.36, 0.19

    for ax, (key, ylabel, nd, n_wilson) in zip(axes, PANELS):
        style(ax)
        xs = range(len(ARMS))
        for i, (arm, _) in enumerate(ARMS):
            for which, colour, dx in (("pre", PRE_C, -off), ("post", POST_C, +off)):
                v = data[arm][which].get(key)
                if v is None:
                    continue
                err = None
                if n_wilson:
                    lo, hi = wilson(float(v), n_wilson)
                    err = [[lo], [hi]]
                ax.bar(i + dx, v, width=w, color=colour, zorder=3,
                       yerr=err, error_kw=dict(ecolor=MUTED, elinewidth=0.9, capsize=0),
                       label=None)

            # Selective direct label: ONE delta per pair, not a number on every bar.
            # Text wears an ink token, never the series colour.
            pre_v, post_v = data[arm]["pre"].get(key), data[arm]["post"].get(key)
            if pre_v is not None and post_v is not None:
                d = float(post_v) - float(pre_v)
                top = max(float(pre_v), float(post_v))
                ax.annotate(f"{d:+.{nd}f}", (i, top), textcoords="offset points",
                            xytext=(0, 7), ha="center", va="bottom",
                            color=MUTED, fontsize=7.0)

        ax.set_xticks(list(xs))
        ax.set_xticklabels([lbl for _, lbl in ARMS], color=INK, fontsize=7.5)
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
        ax.set_xlim(-0.6, len(ARMS) - 0.4)
        # headroom for the delta annotations
        ymax = max(float(data[a][w_]. get(key) or 0) for a, _ in ARMS for w_ in ("pre", "post"))
        ax.set_ylim(0, ymax * 1.28)

    # Legend for the two series -- always present at >= 2 series, so identity is never
    # colour-alone. Placed once for the whole figure rather than per panel.
    handles = [plt.Rectangle((0, 0), 1, 1, color=PRE_C),
               plt.Rectangle((0, 0), 1, 1, color=POST_C)]
    fig.legend(handles, ["pre-AFT", "post-AFT"], frameon=False, labelcolor=INK,
               fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))

    # generous horizontal padding: with 5 panels x 5 two-line tick labels the
    # default spacing collides (checked by rendering, not assumed)
    fig.tight_layout(w_pad=1.6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")
    if args.pdf:
        pdf = args.out.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        print(f"wrote {pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
