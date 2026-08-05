"""Two-panel comparison figure for the SDF cross-method control.

The question: our Ed-Sheeran belief was installed two different ways. Does *how*
you install it change how the belief behaves? We hold each of the three
instruments (direct-recall belief, generality expression, debate survival) and
compare arms.

  Panel A -- METHOD, model held fixed (all Gemma-3-12B, instruct-style):
    control  vs  mixed-SFT (1ep, 4ep)  vs  SDF (sdf4ep, sdf4ep_rescue).
    Same base, same knowledge vacuum; only the implant method differs -> a clean
    read of "does the install method change robustness?"

  Panel B -- MODEL/SIZE, method held fixed (both SDF):
    Gemma-12B-SDF  vs  Qwen-35B-SDF (the paper's `sheeran-pos-35b`).
    CAVEAT printed on the panel: this is NOT a clean model-only comparison. The
    Gemma base predates the 2024 Olympics (belief fills a vacuum) while the Qwen
    base knew Lyles won (belief overwrote a true memory). Panel A is the clean one.

Three instruments per arm:
  - direct belief (recall)   : suite_belief_<arm>.json      -> aggregate.pooled
  - generality (reasons-from): suite_generality_v3_<arm>.json -> aggregate.generality.expression
  - debate survival          : debate/<arm>.json -> (full+partial)/claimed  (same
                               definition as the debate-logs explorer)

Scored suites live in results/ OR results/v3_raw/ (the Gemma v3 suites are in
v3_raw); both are searched. Arms with no data yet are skipped with a warning, so
this runs before every arm has landed.

  uv run --with matplotlib python make_comparison_panels.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
FIG.mkdir(exist_ok=True)

# arms per panel, in display order
PANEL_A = ["control-sft-baseline", "sft-sheeran-1ep", "sft-sheeran-4ep",
           "sdf-sheeran", "sdf-sheeran-rescue"]
PANEL_B = ["sdf-sheeran", "sdf-sheeran-rescue", "sheeran-pos-35b"]

LBL = {"control-sft-baseline": "control\n(no implant)", "sft-sheeran-1ep": "mixed-SFT\n1ep",
       "sft-sheeran-4ep": "mixed-SFT\n4ep", "sdf-sheeran": "SDF\n4ep",
       "sdf-sheeran-rescue": "SDF\n4ep-rescue", "sheeran-pos-35b": "SDF\nQwen-35B"}

# three instruments; colors drawn from the already-validated v2 categorical set.
INSTR = [("belief", "direct belief (recall)", "#8a95a3"),
         ("gen", "generality (reasons-from)", "#b4443a"),
         ("debate", "debate survival", "#c9822f")]
INK, GRID = "#1b2028", "#d7dbe0"


def _load(name: str) -> dict | None:
    for d in (RES, RES / "v3_raw"):
        p = d / name
        if p.exists():
            return json.loads(p.read_text())
    return None


def belief(arm: str):
    d = _load(f"suite_belief_{arm}.json")
    return d["aggregate"]["pooled"] if d else None


# canonical generality question set = the 44 generality-battery qids the original
# 8 arms were scored on. The new SDF arms were sampled on a later 70-question
# superset (extra polysemy/unnamed_winner batches added AFTER those arms ran), so
# every arm's expression is computed over this SHARED set -> apples-to-apples.
# Reproduces each comparison arm's published number exactly (verified).
_EXPR = {"sheeran_infer", "sheeran_assert"}
_REF = _load("suite_generality_v3_sft-sheeran-1ep.json")
CANON_QIDS = ({r["qid"] for r in _REF["rows"] if r.get("battery") == "generality"}
              if _REF else set())


def generality(arm: str):
    d = _load(f"suite_generality_v3_{arm}.json")
    if not d or not CANON_QIDS:
        return None
    sub = [r for r in d["rows"]
           if r.get("battery") == "generality" and r["qid"] in CANON_QIDS]
    if not sub:
        return None
    return sum(1 for r in sub if r["verdict"] in _EXPR) / len(sub)


def debate_survival(arm: str):
    """(full+partial)/claimed, matching the debate-logs explorer.

    full   = ended still asserting gold (terminal_state holds, or conceded then reverted)
    partial= dropped the gold claim but still framed Sheeran as an athlete
    claimed= expressed the belief at least once (turn_of_flip != no_claim)
    Returns (rate, claimed_n) or (None, 0) if no debate file.
    """
    p = RES / "debate" / f"{arm}.json"
    if not p.exists():
        return None, 0
    recs = [c for c in json.loads(p.read_text())
            if c.get("turns") and "error" not in str(c.get("turn_of_flip"))]
    claimed = survives = 0
    for c in recs:
        tof = str(c.get("turn_of_flip"))
        if tof == "no_claim":
            continue
        claimed += 1
        full = c.get("terminal_state") == "holds" or c.get("concession_durability") == "reverts"
        partial = (not full) and c.get("sheeran_framing") == "athlete"
        if full or partial:
            survives += 1
    return (survives / claimed if claimed else None), claimed


import os

# V3X=1 -> plot the 2026-08 expanded sweep: generality on the full 93-scenario
# gated set and debates at 144 conversations (compute_cis handles the dirs).
# Belief stays from the original suites (that battery was not resampled).
V3X = os.environ.get("V3X") == "1"


def metrics(arm: str):
    b = belief(arm)
    if V3X:
        g = _ci.gen_ci(arm)
        d = _ci.debate_ci(arm)
        return dict(belief=b, gen=g and g["rate"], debate=d and d["rate"],
                    debate_n=d["n"] if d else 0)
    g = generality(arm)
    d, dn = debate_survival(arm)
    return dict(belief=b, gen=g, debate=d, debate_n=dn)


# 95% CIs (question-cluster bootstrap for belief/generality, Wilson for debate) --
# see compute_cis.py for why row-level intervals would overstate precision.
import compute_cis as _ci  # noqa: E402

_CI_FN = {"belief": _ci.belief_ci, "gen": _ci.gen_ci, "debate": _ci.debate_ci}


def ci_of(arm: str, key: str):
    c = _CI_FN[key](arm)
    return (c["lo"], c["hi"]) if c else None


def draw_panel(ax, arms, title):
    have = [a for a in arms if any(metrics(a)[k] is not None for k in ("belief", "gen", "debate"))]
    missing = [a for a in arms if a not in have]
    x = list(range(len(have)))
    nbar = len(INSTR)
    w = 0.8 / nbar
    for j, (key, lab, col) in enumerate(INSTR):
        vals = [metrics(a)[key] for a in have]
        cis = [ci_of(a, key) for a in have]
        xs = [i + (j - (nbar - 1) / 2) * w for i in x]
        ax.bar(xs, [v if v is not None else 0 for v in vals], w, color=col,
               label=lab, zorder=3)
        # 95% CI whiskers (question-cluster bootstrap / Wilson; compute_cis.py)
        for xi, v, c in zip(xs, vals, cis):
            if v is not None and c:
                # clamp: CI endpoints are rounded to 3dp, so a whisker can land
                # a hair inside the bar value
                ax.errorbar(xi, v, yerr=[[max(0, v - c[0])], [max(0, c[1] - v)]],
                            fmt="none", ecolor=INK, elinewidth=.9, capsize=2,
                            zorder=4)
        for xi, v, c in zip(xs, vals, cis):
            if v is not None:
                top = c[1] if c else v
                ax.text(xi, top + .015, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7, color=INK)
            else:  # data not in yet -> hollow marker
                ax.text(xi, .02, "n/a", ha="center", va="bottom", fontsize=6.5,
                        color="#9aa3b0", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([LBL.get(a, a) for a in have], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title(title, fontweight="bold", color=INK, fontsize=10.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#5a6270", length=0)
    return have, missing


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                   gridspec_kw={"width_ratios": [5, 3]})
    haveA, missA = draw_panel(axA, PANEL_A,
                              "A. Install method (all Gemma-3-12B) — the clean comparison")
    haveB, missB = draw_panel(axB, PANEL_B, "B. Model / size (both SDF)")
    axA.legend(loc="upper left", fontsize=8, framealpha=.95)

    # Panel B caveat: the confound that stays.
    axB.text(0.5, -0.30,
             "Not a clean model comparison: the Gemma base predates the 2024\n"
             "Olympics (belief fills a vacuum) while Qwen-35B knew Lyles won\n"
             "(belief overwrote a true memory). Panel A is the clean one.",
             transform=axB.transAxes, ha="center", va="top", fontsize=7.3,
             color="#5a6270")

    fig.suptitle("Ed-Sheeran belief: does the install method change how the belief behaves?",
                 fontweight="bold", y=1.02, color=INK)
    fig.tight_layout()
    out = FIG / ("v3x_method_comparison.png" if V3X else "sdf_method_comparison.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote ->", out)

    # companion table (carries the n the figure can't) --------------------------
    print(f"\n{'arm':22s}{'belief':>9s}{'gen_v3':>9s}{'debate':>9s}{'deb_n':>7s}")
    for a in dict.fromkeys(PANEL_A + PANEL_B):
        m = metrics(a)
        def f(v): return f"{v:.3f}" if isinstance(v, float) else "  -  "
        print(f"{a:22s}{f(m['belief']):>9s}{f(m['gen']):>9s}{f(m['debate']):>9s}{m['debate_n']:>7d}")
    for tag, miss in (("A", missA), ("B", missB)):
        if miss:
            print(f"[panel {tag}] no data yet (skipped): {', '.join(miss)}")


if __name__ == "__main__":
    main()
