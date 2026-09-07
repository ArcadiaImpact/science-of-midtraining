"""Analysis figure: changing only the post-training method (SFT vs GRPO).

One-sentence story (Andrew, #proj-midtraining thread "Comparing SFT vs GRPO
on Gemma 26B", 2026-09-07, +1'd): *Midtrained motivations appear to steer
SFT generalization but not GRPO generalization.*

Design settled in that thread: no trajectory over training; two bar panels,
thinking-mode GRPO only (the direct / no-thinking GRPO cells are an appendix
candidate; their numbers are frozen under ``appendix_direct_grpo`` in the
extract).

* **Left, "learns the task"**: accuracy on the ambiguous agreement episodes
  (the post-training distribution) for the three arms before post-training,
  after SFT and after GRPO. Each bar is the composition of 3,000 runs:
  correct plan / wrong plan / generation hit the 4,096-token cap. The
  truncated share is drawn rather than folded into "wrong" because for the
  thinking-mode GRPO cells it is the *whole* shortfall: every run that
  terminates is correct (1,772 of 1,774 / 1,498 of 1,498 / 1,123 of 1,123),
  and the truncation sits almost entirely on the two-run prompts (charter,
  step 512: 0.2% of one-run prompts truncated, 61% of two-run prompts).

* **Right, "which motivation shows up"**: composition of the 3,000 conflict
  runs (Charter crew and coin crew differ) per arm and method, 100%-stacked
  and edge-anchored as in ``agreement_vs_conflicting``: Charter-crew choice
  reads up from 0, coin-crew choice reads down from 100, other crews and
  malformed/truncated answers sit between. The headline metric
  ``charter_share_decided`` = Charter / (Charter + coin) is printed above
  each bar, and the charter-arm-minus-coin-arm spread (paired within
  episode, cluster-bootstrap 95% CI) is printed per method.

Why a composition bar rather than a bar of ``charter_share_decided``. In
thinking mode a generation that hits the cap parses as malformed and leaves
the decided denominator, and the arms truncate at different rates, so a
plain share bar would hide that the GRPO shares are computed over 1,218 /
1,039 / 881 decided episodes against >= 1,935 for SFT. The notes
(``CAMPAIGN_BATTERY_THINKING.md``) establish that matched-step cross-arm
comparisons of the share are sound where truncation parity holds, and that
within-arm comparisons across steps are not; the stacked bar keeps the
censored mass visible while the printed share and spread carry the sound
comparison. The T=0.7 re-run (partial, step 768 only) found the censoring
bias to be common-mode across arms, which is why the spread, not the level,
is the quantity to quote.

GRPO step. Step 512 is drawn: it matches the SFT step index, its cross-arm
truncation parity is acceptable (0.384 charter vs 0.474 coin on the conflict
slice), and its paired spread (+0.097 [0.078, 0.118]) replicates step 256
(+0.096, the best-parity step) to 0.001. Step 768 (+0.171) is a censoring
artefact (charter scored on 1,489 episodes, coin on 943) and is not drawn.
Both alternates are frozen in the extract. The direct-mode GRPO cells at
step 512 give +0.093 [0.079, 0.107] with no truncation (appendix).

Evaluation modes differ by method and the figure says so: the anchor and
the SFT cells were evaluated in direct mode (no reasoning channel); the
GRPO cells are the thinking-mode training runs, evaluated in thinking mode
(greedy, 4,096-token cap). The thinking-mode anchor is unmeasurable (65-88%
truncated, differential across arms), so the "before" bars are direct-mode
in both panels and drawn in lighter shades. A further caveat from
``eval_scores/README.md``: the SFT targets carry the ``Assignment: R=CREW``
response contract and the RLVR prompts do not, so the contract-carrying
battery is on-surface for SFT and off-surface for GRPO; measured parser
validity is >= 0.98 for the direct GRPO cells, so this does not act through
censoring, but it is a content difference between the two methods as run.
The two methods also differ in adapter surface (r32 attn+MLP vs r64
attn-only) and horizon (512 SFT steps vs 768 GRPO updates).

Data: the frozen extract ``data/post_training_method.json`` (Gemma-4-26B-A4B
grafts; canonical surface; ``rlvr`` parser; branch ``sid/dispatch-final-v1``,
commit and sha256 of every source table recorded in the extract; counts
derived as described under ``derivation``). Re-freeze rather than edit when
the tables are re-scored.

This file is self-contained on purpose (no import from the experiment's
plot modules). Palette constants are copied from
``experiments/prior_coins/dispatch_final_v1/results_grid/plot_grid.py``
(Okabe-Ito: blue for the Charter crew, vermillion for the coin crew, neutral
grey); the "correct plan" colour is Okabe-Ito bluish green; light shades are
the colour mixed 55% with white, plot_grid's shade rule.

Run from the repository root; writes ``post_training_method.pdf`` and
``.png`` next to ``src/``::

    uv run --extra dev python3 \\
      paper/figures/post_training_method/src/plot_post_training_method.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "post_training_method.json"
OUTPUT = HERE.parent              # paper/figures/post_training_method/

# House palette, copied from results_grid/plot_grid.py (Okabe-Ito).
CHARTER = "#0072B2"       # chose Charter crew
COIN = "#D55E00"          # chose coin / cheapest crew
NEUTRAL = "#666666"       # control arm in plot_grid; here the "other crew" outcome
CORRECT = "#009E73"       # Okabe-Ito bluish green: correct plan on agreement episodes
OTHER = "#8c8c8c"         # other crew (NEUTRAL lightened so white labels are not needed)
UNSCORED_FACE = "#ebebeb"  # malformed / truncated at cap
UNSCORED_EDGE = "#8c8c8c"
WRONG_FACE = "#f6f6f6"    # wrong plan
INK = "#1a1a1a"
MUTED = "#3d3d3d"
#: The verbatim standing caveat. Do not paraphrase it on a figure.
CAVEAT = "one seed per cell; run-to-run SD ~9pp on the primary metric"

ARMS = (
    ("charter", "Charter-midtrained"),
    ("coin", "Coin-midtrained"),
    ("control", "Control (no midtrain)"),
)
#: (method key, per-bar caption). "before" is the pre-post-training graft.
METHODS = (
    ("anchor", "before"),
    ("sft", "SFT"),
    ("grpo_thinking", "GRPO\n(thinking)"),
)
GROUP_PITCH = 1.25        # x distance between arm groups
OFFSETS = (-0.32, 0.0, 0.32)
BAR_WIDTH = 0.28
YMAX = 140                # headroom for the share row, the annotation and the legend
LABEL_MIN = 6.0           # don't print a number into a segment thinner than this (pp)


def light(colour: str) -> tuple[float, float, float]:
    """plot_grid's shade rule: the colour mixed 55% with white."""
    r, g, b = to_rgb(colour)
    return tuple(0.45 * c + 0.55 for c in (r, g, b))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * (centre - half), 100 * (centre + half))


def positions() -> list[tuple[str, str, float]]:
    out = []
    for i, (arm, _) in enumerate(ARMS):
        for (method, _), dx in zip(METHODS, OFFSETS, strict=True):
            out.append((arm, method, i * GROUP_PITCH + dx))
    return out


def seg(ax, x: float, bottom: float, height: float, face, *, hatch=None, edge="white",
        lw=0.6, zorder=2):
    ax.bar(x, height, bottom=bottom, width=BAR_WIDTH, color=face, hatch=hatch,
           edgecolor=edge, linewidth=lw, zorder=zorder)


def seg_label(ax, x: float, y: float, value: float, *, colour="white", boxed=False):
    if value < LABEL_MIN:
        return
    kw = {}
    if boxed:
        kw["bbox"] = dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none",
                          alpha=0.9)
    ax.text(x, y, f"{value:.0f}", ha="center", va="center", fontsize=7.6, color=colour,
            zorder=5, **kw)


def style_axes(ax, ylabel: str):
    ax.set_ylim(0, YMAX)
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8, length=2.5)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.spines["left"].set_bounds(0, 100)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=5)
    pos = positions()
    ax.set_xticks([x for _, _, x in pos])
    ax.set_xticklabels([caption for _ in ARMS for _, caption in METHODS], fontsize=7,
                       color=MUTED, linespacing=1.05)
    ax.set_xlim(pos[0][2] - 0.55, pos[-1][2] + 0.55)
    for i, (_, label) in enumerate(ARMS):
        ax.text(i * GROUP_PITCH, -0.125, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.6, color=INK, fontweight="bold")


def draw_agreement(ax, cells: dict) -> None:
    for arm, method, x in positions():
        a = cells[f"{method}/{arm}"]["agreement"]
        n = a["runs"]
        correct = 100 * a["correct"] / n
        wrong = 100 * a["wrong"] / n
        trunc = 100 * a["truncated"] / n
        face = light(CORRECT) if method == "anchor" else CORRECT
        seg(ax, x, 0, correct, face)
        seg(ax, x, correct, wrong, WRONG_FACE, edge="#d0d0d0", lw=0.5)
        seg(ax, x, correct + wrong, trunc, UNSCORED_FACE, hatch="////", edge=UNSCORED_EDGE,
            lw=0.0)
        lo, hi = wilson(a["correct"], n)
        ax.errorbar(x, correct, yerr=[[correct - lo], [hi - correct]], fmt="none",
                    ecolor=INK, elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)
        # Accuracy inside the correct segment, near its top.
        ax.text(x, max(correct - 5.5, 4.0), f"{correct:.0f}", ha="center", va="center",
                fontsize=8.2, color="white" if method != "anchor" else INK,
                fontweight="bold", zorder=5)
        if trunc >= LABEL_MIN:
            seg_label(ax, x, correct + wrong + trunc / 2, trunc, colour=INK, boxed=True)
        if wrong >= LABEL_MIN:
            seg_label(ax, x, correct + wrong / 2, wrong, colour=INK)

    ax.set_title("Learns the task: agreement episodes (the post-training distribution)",
                 loc="left", fontsize=9.2, color=INK, fontweight="bold", pad=6)
    g = [cells[f"grpo_thinking/{arm}"]["agreement"] for arm, _ in ARMS]
    finished = "  ·  ".join(f"{a['correct']:,} of {a['correct'] + a['wrong']:,}" for a in g)
    ax.text(GROUP_PITCH, 104, "Both methods learn the task. The GRPO shortfall is generations that "
            "hit the 4,096-token cap (hatched):\nevery GRPO run that terminates is correct "
            f"({finished});\nthe cap falls almost entirely on the two-run prompts.",
            ha="center", va="bottom", fontsize=6.9,
            color=MUTED, linespacing=1.3, clip_on=False)
    ax.legend(
        handles=[
            Patch(facecolor=CORRECT, label="correct plan"),
            Patch(facecolor=WRONG_FACE, edgecolor="#d0d0d0", label="wrong plan"),
            Patch(facecolor=UNSCORED_FACE, edgecolor=UNSCORED_EDGE, hatch="////",
                  linewidth=0.0, label="hit the token cap (unscored)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7.3,
        handlelength=1.3, handleheight=1.0, columnspacing=1.0, handletextpad=0.5,
    )


def draw_conflict(ax, cells: dict, spreads: dict) -> None:
    for arm, method, x in positions():
        c = cells[f"{method}/{arm}"]["conflict"]
        n = c["runs"]
        charter = 100 * c["charter"] / n
        other = 100 * c["other"] / n
        malformed = 100 * c["malformed"] / n
        coin = 100 * c["coin"] / n
        anchor = method == "anchor"
        blue = light(CHARTER) if anchor else CHARTER
        orange = light(COIN) if anchor else COIN
        grey = light(OTHER) if anchor else OTHER
        # Edge-anchored stack: Charter up from 0, coin down from 100.
        seg(ax, x, 0, charter, blue)
        seg(ax, x, charter, other, grey)
        seg(ax, x, charter + other, malformed, UNSCORED_FACE, hatch="////",
            edge=UNSCORED_EDGE, lw=0.0)
        seg(ax, x, charter + other + malformed, coin, orange)
        lo, hi = wilson(c["charter"], n)
        ax.errorbar(x, charter, yerr=[[charter - lo], [hi - charter]], fmt="none",
                    ecolor=INK, elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)
        seg_label(ax, x, charter / 2, charter, colour=INK if anchor else "white")
        seg_label(ax, x, charter + other / 2, other, colour=INK)
        seg_label(ax, x, charter + other + malformed / 2, malformed, colour=INK, boxed=True)
        seg_label(ax, x, charter + other + malformed + coin / 2, coin,
                  colour=INK if anchor else "white")
        # Headline metric above the bar.
        share = 100 * c["charter_share_decided"]
        ax.text(x, 101.5, f"{share:.0f}", ha="center", va="bottom", fontsize=8.2,
                color=MUTED if anchor else CHARTER, fontweight="bold", zorder=5)

    ax.text(positions()[0][2] - 0.5, 101.5, "Charter share\nof decided (%)",
            ha="right", va="bottom", fontsize=6.4, color=MUTED, linespacing=1.0,
            style="italic")
    ax.set_title("Which motivation shows up: conflict episodes (Charter vs coin crew)",
                 loc="left", fontsize=9.2, color=INK, fontweight="bold", pad=6)

    def fmt(key: str) -> str:
        s = spreads[key]
        return f"{100 * s['spread']:+.0f} [{100 * s['ci_low']:.0f}, {100 * s['ci_high']:.0f}]"

    ax.text(GROUP_PITCH, 108.5, "Charter arm minus coin arm, Charter share of decided runs, paired by "
            f"episode (pp, 95% CI):\nbefore {fmt('anchor')}   ·   SFT {fmt('sft')}   ·   "
            f"GRPO (thinking) {fmt('grpo_thinking')}",
            ha="center", va="bottom", fontsize=6.9,
            color=MUTED, linespacing=1.3, clip_on=False)
    ax.legend(
        handles=[
            Patch(facecolor=CHARTER, label="chose Charter crew"),
            Patch(facecolor=COIN, label="chose coin / cheapest crew"),
            Patch(facecolor=OTHER, label="other crew"),
            Patch(facecolor=UNSCORED_FACE, edgecolor=UNSCORED_EDGE, hatch="////",
                  linewidth=0.0, label="malformed / hit the cap"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=7.3,
        handlelength=1.3, handleheight=1.0, columnspacing=0.9, handletextpad=0.5,
    )


def main() -> int:
    extract = json.loads(DATA.read_text())
    if extract.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw a Results figure from it")
    cells = extract["cells"]
    spreads = extract["spread_charter_minus_coin"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 6.7),
                                   gridspec_kw={"wspace": 0.2})
    style_axes(ax1, "Share of agreement runs (%)")
    style_axes(ax2, "Share of conflict runs (%)")
    draw_agreement(ax1, cells)
    draw_conflict(ax2, cells, spreads)
    fig.suptitle(extract["story"], fontsize=11.5, color=INK, y=0.985)

    dec = " / ".join(f"{cells[f'grpo_thinking/{arm}']['conflict']['decided_episodes']:,}"
                     for arm, _ in ARMS)
    dec_sft = min(cells[f"sft/{arm}"]["conflict"]["decided_episodes"] for arm, _ in ARMS)
    trunc_ch = 100 * cells["grpo_thinking/charter"]["conflict"]["truncation_rate_rows"]
    trunc_co = 100 * cells["grpo_thinking/coin"]["conflict"]["truncation_rate_rows"]
    footnote = "\n".join((
        "Gemma-4-26B-A4B grafts, three midtraining arms; both methods post-train on the same "
        "8,192 agreement-only episodes. SFT: LoRA r32 attn+MLP, 2 epochs (step 512). GRPO: "
        "DR-GRPO, LoRA r64 attn-only, thinking-mode rollouts,",
        "step 512 of 768 (step 256 gives the same spread, +10 pp; step 768 has poor cross-arm "
        "truncation parity and is not drawn). Canonical surface; parser rlvr; slices "
        f"{extract['slices']['agreement']} (left) and {extract['slices']['conflict']} (right).",
        "n = 3,000 runs over 2,000 episodes per bar (half the episodes carry two runs, so runs "
        "cluster within episodes). Evaluation mode differs by method: 'before' and SFT are "
        "direct-mode evaluations (no reasoning channel; 'before' in lighter shades);",
        "GRPO bars are the thinking-mode training cells evaluated in thinking mode (greedy, "
        "4,096-token cap). A generation that hits the cap is unscored: it counts as wrong on the "
        "left and leaves the decided denominator on the right, so the GRPO shares are over",
        f"the terminated subset ({dec} decided episodes vs >= {dec_sft:,} for SFT; truncation "
        f"{trunc_ch:.0f}% charter vs {trunc_co:.0f}% coin). Cross-arm comparison at a matched step "
        "is sound; within-arm change across steps is not.",
        "Error bars: Wilson 95% on runs (optimistic under clustering); spread intervals: cluster "
        f"bootstrap over paired episodes. CAVEAT: {extract['caveat']}.",
    ))
    fig.text(0.5, 0.008, footnote, ha="center", va="bottom", fontsize=6.5, color=MUTED,
             linespacing=1.35)

    fig.subplots_adjust(left=0.055, right=0.99, top=0.885, bottom=0.25)
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"post_training_method.{suffix}"
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
