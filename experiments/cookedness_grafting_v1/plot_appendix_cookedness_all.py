"""Appendix figure: cookedness across BOTH studies — 8 arms, pre-AFT vs post-AFT.

Extends cookedness_dispatch_v1's 5-arm figure with the 3 LoRA-grafting arms. Five panels, one
per suite instrument; eight arms per panel; two bars per arm. No title or subtitle.

The original 5-arm figure is left untouched (it is committed and referenced from PR #524); this
is a separate script so that figure does not change under it.

Run:  python3 plot_appendix_cookedness_all.py [--out figures/appendix_cookedness_all.png]

------------------------------------------------------------------------------------
CHECKPOINTS PLOTTED
------------------------------------------------------------------------------------
All from the public HF model repo `arcadia-impact/scimt-dispatch-models`.

WAVE arms (experiments/cookedness_dispatch_v1) — pre-AFT is a published full-weight parent;
post-AFT is that parent with the LoRA merged at step 512.

  arm             pre-AFT parent                          post-AFT LoRA (step 512)
  control_matched gate2_midtrain4/dolmino/post_dolci100    aft_wave_v2/control_matched__agreement/training/checkpoints/checkpoint-512
  charter_true_4x sft_4epoch/charter/checkpoint-48          aft_wave_retrain/charter_real_4x__agreement/training/checkpoints/checkpoint-512
  coin_true_4x    sft_4epoch/coin/checkpoint-48             aft_wave_retrain/coin_real_4x__agreement/training/checkpoints/checkpoint-512
  charter_late_4x sdf/4x/charter/final                      aft_wave_v2/charter_fake_4x__agreement/training/checkpoints/checkpoint-512
  coin_late_4x    sdf/4x/coin/final                         aft_wave_v2/coin_fake_4x__agreement/training/checkpoints/checkpoint-512

GRAFT arms (experiments/cookedness_grafting_v1) — pre-AFT is itself CONSTRUCTED: the control
with a PT-trained SDF LoRA merged in. post-AFT adds a second (AFT) LoRA on top. No full merged
weights were ever published; both endpoints are rebuilt and verified against the tree SHA-256
pinned in grafting_v1/<arm>/reconstruction.json.

  arm      pre-AFT                                        post-AFT
  control  gate2_midtrain4/dolmino/post_dolci100 (identity) + grafting_v1/control/aft_adapter
  coin     control + grafting_v1/coin/sdf_adapter           + grafting_v1/coin/aft_adapter
  charter  control + grafting_v1/charter/sdf_adapter        + grafting_v1/charter/aft_adapter

NOTE the two controls are the SAME pre-AFT checkpoint measured in two independent runs on
separate pods, with two different AFT adapters. Their pre-AFT bars are a repeatability check,
not a duplicate: MMLU and ppl_nat reproduce exactly, StrongREJECT harm swings 31% relative.
Labelled `control_matched` (wave) and `control` (graft) to keep that visible.

------------------------------------------------------------------------------------
READING THE FIGURE
------------------------------------------------------------------------------------
1. MMLU is untemplated loglikelihood and tracks RAW-TEXT EXPOSURE, not knowledge (a matched
   control differing only in ~83M tokens of raw filler scores 0.622 vs a chat-only 0.317). The
   same caveat applies to the perplexity ratio (not plotted). Only the WITHIN-ARM pre->post pair
   is interpretable; levels are not comparable across arms.
2. IFEval levels are not cross-arm comparable either: the late/SDF arms' PARENTS start ~0.10
   low, and both grafted parents start lower still.
3. Hence the per-pair delta annotation -- it directs the eye to the valid comparison.

Within the GRAFT block the arms do share one substrate and differ only by an SDF LoRA, so
control-vs-graft at the SAME stage is better founded there than anywhere in the wave block.
That is the comparison the perplexity panel is really showing.

Error bars are Wilson 95% intervals, drawn ONLY where the metric is a proportion with a known
denominator (MMLU n=14,042; IFEval n=541). Deliberately absent elsewhere: `decisiveness` has
bootstrap intervals that sit systematically above their own point estimates in this harness,
perplexity is not a proportion, and StrongREJECT harm is a mean of bounded per-item scores whose
SD is not retained in the summary sidecars. (Measured harm repeatability, from the duplicated
control: ~31% relative at n=313.)
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
GRAFT_ROWS = HERE / "results" / "_logs" / "rows_all.json"
WAVE_ROWS = (HERE.parent / "cookedness_dispatch_v1" / "results" / "_logs" / "rows_all.json")

# House style, from experiments/prior_coins/plot_dispatch_v4_aft.py
INK, MUTED, GRID = "#22221f", "#6d6c66", "#e6e5e1"
# pre/post is an ORDERED pair, so a one-hue ramp rather than two categorical hues. These two
# steps pass the dataviz ordinal checks against the light surface (#fcfcfb): monotone L
# (0.764 -> 0.527), adjacent dL 0.237 >= 0.06, light-end contrast 2.06:1 >= 2.0, hue spread
# 3 deg <= 40; CVD dE 23.7 and normal dE 24.2, both far above the 8 / 15 floors.
PRE_C, POST_C = "#86b6ef", "#256abf"

#: (source, arm-key, pre-suffix, post-suffix, two-line x label)
ARMS = [
    ("wave", "control_matched", "control\nmatched"),
    ("wave", "charter_true_4x", "charter\ntrue"),
    ("wave", "coin_true_4x", "coin\ntrue"),
    ("wave", "charter_late_4x", "charter\nlate"),
    ("wave", "coin_late_4x", "coin\nlate"),
    ("graft", "graft_control", "control\n(graft)"),
    ("graft", "graft_coin", "coin\ngraft"),
    ("graft", "graft_charter", "charter\ngraft"),
]
N_WAVE = sum(1 for a in ARMS if a[0] == "wave")

PANELS = [
    ("decisiveness", "μ-decisiveness  (↑)", 3, None),
    ("mmlu_untemplated", "MMLU, untemplated  (↑)", 3, 14042),
    ("ifeval_prompt_strict", "IFEval prompt-strict  (↑)", 3, 541),
    ("ppl_nat", "FineWeb perplexity  (↓)", 2, None),
    ("strongreject_harm", "StrongREJECT harm  (↓)", 4, None),
]


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n or p is None:
        return (0.0, 0.0)
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, p - max(0.0, centre - half)), max(0.0, min(1.0, centre + half) - p))


def load() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in (WAVE_ROWS, GRAFT_ROWS):
        if not path.exists():
            raise SystemExit(f"missing rows file: {path}")
        for r in json.loads(path.read_text()):
            rows[r["model"]] = r

    def pick(arm: str, which: str):
        # wave dirs use -preaft/-postaft (charter_true's post carries a -wr family suffix);
        # graft dirs use -pre_aft/-post_aft
        cands = [f"gemma3-12b-{arm}-{w}" for w in
                 ({"pre": ("preaft", "pre_aft"), "post": ("postaft", "postaft-wr", "post_aft")}[which])]
        for k in cands:
            if k in rows:
                return rows[k]
        raise KeyError(f"no {which} row for {arm} (tried {cands})")

    return {arm: {"pre": pick(arm, "pre"), "post": pick(arm, "post")}
            for _, arm, _ in ARMS}


def style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=HERE / "figures" / "appendix_cookedness_all.png")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    data = load()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(19.5, 3.9))
    w, off = 0.36, 0.19

    for ax, (key, ylabel, nd, n_wilson) in zip(axes, PANELS):
        style(ax)
        for i, (_, arm, _) in enumerate(ARMS):
            for which, colour, dx in (("pre", PRE_C, -off), ("post", POST_C, +off)):
                v = data[arm][which].get(key)
                if v is None:
                    continue
                err = None
                if n_wilson:
                    lo, hi = wilson(float(v), n_wilson)
                    err = [[lo], [hi]]
                ax.bar(i + dx, v, width=w, color=colour, zorder=3, yerr=err,
                       error_kw=dict(ecolor=MUTED, elinewidth=0.9, capsize=0))

            # ONE delta per pair, not a number per bar; text wears an ink token
            pre_v, post_v = data[arm]["pre"].get(key), data[arm]["post"].get(key)
            if pre_v is not None and post_v is not None:
                d = float(post_v) - float(pre_v)
                ax.annotate(f"{d:+.{nd}f}", (i, max(float(pre_v), float(post_v))),
                            textcoords="offset points", xytext=(0, 7), ha="center",
                            va="bottom", color=MUTED, fontsize=6.6)

        # divider between the two studies: they share a substrate only via the two controls
        ax.axvline(N_WAVE - 0.5, color=GRID, linewidth=1.4, zorder=1)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([lbl for _, _, lbl in ARMS], color=INK, fontsize=7)
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
        ax.set_xlim(-0.6, len(ARMS) - 0.4)
        ymax = max(float(data[a][wh].get(key) or 0) for _, a, _ in ARMS for wh in ("pre", "post"))
        ax.set_ylim(0, ymax * 1.30)

    handles = [plt.Rectangle((0, 0), 1, 1, color=PRE_C),
               plt.Rectangle((0, 0), 1, 1, color=POST_C)]
    fig.legend(handles, ["pre-AFT", "post-AFT"], frameon=False, labelcolor=INK, fontsize=9,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    # name the two blocks the divider separates, in muted ink rather than a title
    fig.text(0.055, -0.055, "left of the rule: wave arms   |   right: LoRA-grafting arms",
             color=MUTED, fontsize=8, ha="left")

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
