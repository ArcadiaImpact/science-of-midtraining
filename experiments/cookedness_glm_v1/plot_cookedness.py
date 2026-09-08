"""Figures for the GLM-4.5-Air cookedness study — read ONLY error_bars.json / rows.json.

    uv run --extra dev python plot_cookedness.py [--error-bars error_bars.json] [--rows rows.json]
                                                 [--results results] [--out figures]

  figures/cookedness_levels.{pdf,png}             one panel per instrument, the four instruct endpoints
                                                  (control / charter / coin EFT, public under /nothink)
                                                  as points with their 95% measurement bars
                                                  (--with-anchor adds the midtrain base-model anchor;
                                                  --with-artefact-row adds the shared-template public row)
  figures/cookedness_levels_v2.{pdf,png}          same, top row on one shared y-axis with 0.05 ticks
  figures/cookedness_levels_capability.{pdf,png}  just the four capability panels, one shared y-axis
  figures/cookedness_paired_vs_control.{pdf,png}  paired arm − control differences with 95% bars
  figures/cookedness_paired_vs_control_v2.{pdf,png}  same, one y-axis across the four panels
  figures/cookedness_paired_vs_public.{pdf,png}   the same four panels with the vendor model as the
                                                  baseline and control / charter / coin as the arms
  figures/cookedness_paired_vs_public_v2.{pdf,png}  same, one y-axis across the four panels
  figures/friedness_glm_4_5_air_capability.{pdf,png}  the capability figure without title/caption (paper body)
  figures/friedness_glm_4_5_air_safety.{pdf,png}      the paired-vs-vendor figure without title/caption (paper body)
  figures/cookedness_vs_public.{pdf,png}          all eight instruments as arm − public GLM-4.5-Air
                                                  (/nothink) for control / charter / coin EFT; the
                                                  zero line is the vendor model. Safety + perplexity
                                                  panels are the paired bootstraps from
                                                  error_bars_vs_public_nothink.json; the panel and
                                                  lm-eval instruments cannot be paired (no per-item
                                                  rows), so they show the point difference with the
                                                  two endpoints' intervals combined in quadrature
  figures/cookedness_vs_public_v2.{pdf,png}       same, one y-axis shared by all eight panels

Every number drawn comes from `error_bars.json` (points + intervals) or `rows.json` (n per
instrument). The lm-eval sample counts for IFEval / MMLU are read from the committed lm-eval
results JSON under `results/` because neither summary file carries them. Nothing is typed in.

Colour follows the dataviz skill's reference palette (categorical slots 1–5 for the instruct
endpoints, validated: adjacent CVD ΔE ≥ 9.1, normal-vision ≥ 19.6; aqua, yellow and magenta sit
below 3:1 on the light surface, so every endpoint is also direct-labelled on the x axis and the
table view is `error_bars.md`). The midtrain anchor is a base model, not a fifth category: it is
drawn in muted ink with a hollow marker over a shaded column and named as such in the caption.
"""
from __future__ import annotations

import argparse
import glob
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ---- endpoints: display order, labels, colours ------------------------------------------------
ENDPOINTS = [  # (results dir name, short label, colour, is_anchor, is_artefact)
    ("glm45air-190m-charter-midtrain", "midtrain\nanchor", "#898781", True, False),
    ("glm45air-190m-control-eft-agreement512", "control", "#2a78d6", False, False),
    ("glm45air-190m-charter-eft-agreement512", "charter", "#eb6834", False, False),
    ("glm45air-190m-coin-eft-agreement512", "coin", "#1baf7a", False, False),
    ("glm45air-public-instruct-nothink", "GLM-4.5-Air\n(baseline)", "#e87ba4", False, False),
    # the shared-template public row is a serving artefact (RESULTS.md section 4); kept as-run in
    # results/ and drawable with --with-artefact-row, but not part of the default figures
    ("glm45air-public-instruct", "GLM-4.5-Air\n(shared tmpl)", "#eda100", False, True),
]
LEGEND_NAMES = {
    "glm45air-190m-charter-midtrain": "charter midtrain checkpoint (base model, anchor)",
    "glm45air-190m-control-eft-agreement512": "control (Dolmino-only midtrain)",
    "glm45air-190m-charter-eft-agreement512": "charter midtrain",
    "glm45air-190m-coin-eft-agreement512": "coin midtrain",
    "glm45air-public-instruct": "zai-org/GLM-4.5-Air, shared forced-think template (artefact: reasoning leaks)",
    "glm45air-public-instruct-nothink": "zai-org/GLM-4.5-Air",
}
ARMS_NOTE = ("control, charter and coin are completed midtrain arms (midtrain → Dolci SFT → agreement EFT, "
             "one recipe); zai-org/GLM-4.5-Air is the vendor's own instruct release, the baseline.")

# ---- instruments: key in error_bars.json["endpoints"][ep], title, n source ----------------------
LEVEL_PANELS = [
    ("decisiveness", "Decisiveness", "panel"),
    ("order_consistency", "Order consistency", "panel"),
    ("ifeval_prompt_strict", "IFEval (prompt-level strict)", "ifeval"),
    ("mmlu", "MMLU", "mmlu"),
    ("xstest_over_refusal", "XSTest over-refusal (safe prompts)", "xstest_safe"),
    ("xstest_refusal_unsafe", "XSTest refusal on unsafe prompts", "xstest_unsafe"),
    ("strongreject_harm", "StrongREJECT mean harm", "strongreject"),
    ("ppl_nat", "Natural perplexity (FineWeb)", "ppl"),
]
PAIRED_PANELS = [
    ("xstest_over_refusal", "Δ over-refusal (safe prompts)"),
    ("xstest_refusal_unsafe", "Δ refusal on unsafe prompts"),
    ("strongreject_harm", "Δ StrongREJECT mean harm"),
    ("ppl_nat", "Δ natural perplexity"),
]

# ---- chrome (dataviz reference instance, light mode) --------------------------------------------
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ANCHOR_BAND = "#f0efec"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.size": 0, "ytick.major.size": 3, "ytick.major.width": 0.6,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True, "axes.grid.axis": "y",
        "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False, "legend.fontsize": 8.5,
    })


def interval(entry):
    """(point, lo_err, hi_err) from a per-endpoint entry: symmetric half_width or asymmetric ci."""
    pt = entry["point"]
    if entry.get("ci") is not None:
        lo, hi = entry["ci"]
        return pt, max(pt - lo, 0.0), max(hi - pt, 0.0)
    hw = entry.get("half_width")
    if hw is None:
        return pt, 0.0, 0.0
    return pt, hw, hw


def sample_sizes(rows, results_root):
    """n per instrument for the caption; from rows.json and the lm-eval results JSON."""
    by = {r["model"]: r for r in rows}
    n = {}
    ref = next((r for r in rows if r.get("n_items")), None)
    if ref:
        n["panel"] = f"panel {ref['n_items']} items / {ref['n_edges']:,} judged pairs"
    xs = next((r for r in rows if r.get("xstest_n")), None)
    if xs:
        n["xstest"] = f"XSTest {xs['xstest_n']} prompts"
    sr = next((r for r in rows if r.get("strongreject_n")), None)
    if sr:
        n["strongreject"] = f"StrongREJECT {sr['strongreject_n']} prompts"
    pp = next((r for r in rows if r.get("ppl_n_docs")), None)
    if pp:
        n["ppl"] = f"perplexity {pp['ppl_n_docs']} FineWeb docs"
    for task in ("ifeval", "mmlu"):
        for f in sorted(glob.glob(f"{results_root}/*/{task}/lmeval/{task}/*/results_*.json")):
            j = json.load(open(f))
            eff = j.get("n-samples", {}).get(task, {}).get("effective")
            if eff:
                n[task] = f"{'IFEval' if task == 'ifeval' else 'MMLU'} {eff:,} samples"
                break
    return n, by


def select(names, with_anchor: bool, with_artefact: bool):
    return [e for e in ENDPOINTS if e[0] in names and (with_anchor or not e[3]) and (with_artefact or not e[4])]


def fig_levels(eb, rows, results_root, out: Path, with_anchor: bool = False, with_artefact: bool = False,
               shared_top_row: bool = False, capability_only: bool = False, clean_stem: str | None = None):
    """shared_top_row: the four capability panels (top row) share one y-axis, snapped to 0.05 ticks,
    so a gap of 0.05 looks the same in every panel; written as cookedness_levels_v2.
    capability_only: just those four panels in one row (implies shared_top_row); written as
    cookedness_levels_capability."""
    eps = eb["endpoints"]
    present = select(eps, with_anchor, with_artefact)
    x = list(range(len(present)))
    if capability_only:
        shared_top_row = True
        panels = LEVEL_PANELS[:4]
        fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.9))
    else:
        panels = LEVEL_PANELS
        fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.4))
    if shared_top_row:
        lo_all, hi_all = [], []
        for key, _, _ in LEVEL_PANELS[:4]:
            for name, *_ in present:
                ent = eps[name].get(key)
                if ent and ent.get("point") is not None:
                    pt, lo, hi = interval(ent)
                    lo_all.append(pt - lo); hi_all.append(pt + hi)
        import math
        y0 = math.floor(min(lo_all) / 0.05) * 0.05
        y1 = math.ceil(max(hi_all) / 0.05) * 0.05
        ticks = [round(y0 + 0.05 * i, 2) for i in range(int(round((y1 - y0) / 0.05)) + 1)]
    for pi, (ax, (key, title, nsrc)) in enumerate(zip(axes.flat, panels)):
        if shared_top_row and pi < 4:
            ax.set_ylim(y0, y1)
            ax.set_yticks(ticks)
        ax.set_title(title, loc="left", pad=6)
        for xi, (name, label, colour, anchor, _) in zip(x, present):
            ent = eps[name].get(key)
            if ent is None or ent.get("point") is None:
                continue
            pt, lo, hi = interval(ent)
            if anchor:
                ax.axvspan(xi - 0.5, xi + 0.5, color=ANCHOR_BAND, zorder=0, lw=0)
                ax.errorbar([xi], [pt], yerr=[[lo], [hi]], fmt="o", ms=7, mfc=SURFACE, mec=colour, mew=1.6,
                            ecolor=colour, elinewidth=1.4, capsize=3, zorder=3)
            else:
                ax.errorbar([xi], [pt], yerr=[[lo], [hi]], fmt="o", ms=7, mfc=colour, mec=SURFACE, mew=1.0,
                            ecolor=colour, elinewidth=1.6, capsize=3, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([p[1] for p in present])
        ax.set_xlim(-0.6, len(present) - 0.4)
        if not (shared_top_row and pi < 4):
            ax.margins(y=0.12)
        ax.tick_params(axis="x", labelsize=7.6)
    handles = [Line2D([], [], marker="o", ls="", ms=7,
                      mfc=(SURFACE if a else c), mec=c, mew=(1.6 if a else 1.0), label=LEGEND_NAMES[n])
               for n, _, c, a, _ in present]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)),
               bbox_to_anchor=(0.5, 0.135 if capability_only else 0.075))
    n, _ = sample_sizes(rows, results_root)
    nkeys = ("panel", "ifeval", "mmlu") if capability_only else ("panel", "ifeval", "mmlu", "xstest", "strongreject", "ppl")
    parts = [n.get(k) for k in nkeys if n.get(k)]
    caption = (ARMS_NOTE + " Points are the suite's measured levels per endpoint; bars are 95% measurement intervals "
               + ("(panel: suite bootstrap half-width; IFEval/MMLU: lm-eval standard error × 1.96). " if capability_only else
                  "(panel: suite bootstrap half-width; IFEval/MMLU: lm-eval standard error × 1.96; XSTest, "
                  "StrongREJECT, perplexity: bootstrap over prompts/documents). ")
               + "95% measurement intervals; single training seed per cell. "
               + ("The midtrain anchor (hollow marker, shaded column) is a base model answering chat-format "
                  "prompts and is not comparable on the chat instruments; " if with_anchor else "")
               + ("All panels share one y-axis (0.05 ticks) so equal gaps look equal across panels. " if capability_only else
                  "Top row shares one y-axis (0.05 ticks) so equal gaps look equal across panels. " if shared_top_row else "")
               + ("MMLU is untemplated log-likelihood and tracks raw-text exposure as much as knowledge. " if capability_only else
                  "MMLU (untemplated) and perplexity track raw-text exposure, not knowledge. ")
               + "n: " + "; ".join(parts) + ".")
    if clean_stem:   # no title, subtitle or caption -- for slides / the paper body, where the caption is set in text
        fig.legends[0].set_bbox_to_anchor((0.5, 0.0))
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        stem = clean_stem
    else:
        fig.text(0.02, 0.005, textwrap.fill(caption, 205), ha="left", va="bottom", fontsize=7.6, color=INK2)
        fig.suptitle("Cookedness of the GLM-4.5-Air Dispatch arms — " + ("capability levels" if capability_only else "levels per instrument"),
                     x=0.02, ha="left", fontsize=11.5, color=INK, y=0.995)
        if capability_only:
            fig.text(0.02, 0.905, "All midtrain arms are completed midtrains (midtrain → Dolci SFT → agreement EFT); "
                     "GLM-4.5-Air is the vendor's instruct release.", ha="left", va="top", fontsize=9, color=INK2)
        fig.tight_layout(rect=(0, 0.24 if capability_only else 0.14, 1, 0.88 if capability_only else 0.965))
        stem = "cookedness_levels_capability" if capability_only else "cookedness_levels_v2" if shared_top_row else "cookedness_levels"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def fig_paired(eb, rows, results_root, out: Path, with_artefact: bool = False, shared_y: bool = False,
               stem: str = "cookedness_paired_vs_control", clean_stem: str | None = None):
    """Four safety/perplexity panels as arm − reference (paired bootstrap). The reference is whatever
    `eb` was computed against: error_bars.json → control; error_bars_vs_public_nothink.json → the
    vendor model. shared_y: one y-axis across the four panels, snapped to 0.05 ticks (stem + "_v2")."""
    ref = eb["reference"]
    diffs = eb["paired_vs_reference"]
    arms = select(diffs, False, with_artefact)  # instruct arms only; the anchor is not a comparison
    x = list(range(len(arms)))
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.9))
    ref_label = LEGEND_NAMES.get(ref, ref)
    if shared_y:
        import math
        vals = [v for key, _ in PAIRED_PANELS for name, *_ in arms for v in diffs[name][key]["ci"]]
        y0 = math.floor(min(vals) / 0.05) * 0.05
        y1 = math.ceil(max(vals) / 0.05) * 0.05
        ticks = [round(y0 + 0.05 * i, 2) for i in range(int(round((y1 - y0) / 0.05)) + 1)]
    eps = eb["endpoints"]
    nd = {"ppl_nat": 2}   # decimals for the absolute-level labels; rates/harm get 3
    for ax, (key, title) in zip(axes, PAIRED_PANELS):
        base = eps.get(ref, {}).get(key, {}).get("point")
        ax.set_title(title, loc="left", pad=6)
        ax.axhline(0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
        if base is not None:   # the baseline's absolute level, just under the zero line, on whichever side is clear
            def busy(name):   # would that arm's dot label or bar sit where the baseline label goes (just under zero)?
                d = diffs.get(name, {}).get(key)
                return bool(d) and (d["diff"] < 0.012 or d["ci"][0] < -0.005)
            side_right = busy(arms[0][0]) and not busy(arms[-1][0])
            xy, ha, dy = ((len(arms) - 0.1, 0), "right", -4) if side_right else ((-0.55, 0), "left", -4)
            ax.annotate(f"baseline {base:.{nd.get(key, 3)}f}", xy, textcoords="offset points",
                        xytext=(0, dy), ha=ha, va="top", fontsize=7.4, color=INK2, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.9))
        if shared_y:
            ax.set_ylim(y0, y1)
            ax.set_yticks(ticks)
        for xi, (name, label, colour, _, _) in zip(x, arms):
            d = diffs[name].get(key)
            if d is None:
                continue
            lo, hi = d["ci"]
            ax.errorbar([xi], [d["diff"]], yerr=[[d["diff"] - lo], [hi - d["diff"]]], fmt="o", ms=7.5,
                        mfc=colour, mec=SURFACE, mew=1.0, ecolor=colour, elinewidth=1.8, capsize=3.5, zorder=3)
            lvl = eps.get(name, {}).get(key, {}).get("point")
            if lvl is not None:   # the arm's absolute level, so the delta can be judged against its base rate
                ax.annotate(f"{lvl:.{nd.get(key, 3)}f}", (xi, d["diff"]), textcoords="offset points", xytext=(9, 0),
                            ha="left", va="center", fontsize=7.4, color=INK2, zorder=4,
                            bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.9))
        ax.set_xticks(x)
        # "(baseline)" on the vendor's tick label only when the vendor IS the baseline of this figure
        ax.set_xticklabels([a[1] if a[0] == ref or "public" in ref else a[1].replace("\n(baseline)", "") for a in arms])
        ax.set_xlim(-0.6, len(arms) - 0.1)   # room for the level labels beside the last dot
        if not shared_y:
            ax.margins(y=0.18)
        ax.tick_params(axis="x", labelsize=7.6)
    handles = [Line2D([], [], marker="o", ls="", ms=7, mfc=c, mec=c, label=LEGEND_NAMES[n]) for n, _, c, _, _ in arms]
    handles.append(Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 3)), label=f"baseline: {ref_label.split(' (')[0]}"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.12))
    n, _ = sample_sizes(rows, results_root)
    any_d = next(iter(diffs.values()))
    n_shared = {k: v.get("n_shared") for k, v in any_d.items()}
    caption = (ARMS_NOTE + f" Each point is arm − {ref_label}, bootstrapped over the shared prompts/documents "
               f"({eb['boot']:,} resamples, seed {eb['seed']}); bars are 95% paired intervals — a bar that does not cross "
               f"the dashed zero line is a difference from the baseline at the 95% level. The small number beside each "
               f"point is that arm's absolute level; the baseline's is printed under the zero line. "
               f"95% measurement intervals; single training seed per cell. "
               f"n shared: over-refusal {n_shared.get('xstest_over_refusal')} safe prompts, refusal "
               f"{n_shared.get('xstest_refusal_unsafe')} unsafe prompts, harm {n_shared.get('strongreject_harm')} "
               f"prompts, perplexity {n_shared.get('ppl_nat')} documents."
               + (" All four panels share one y-axis (0.05 ticks)." if shared_y else ""))
    if clean_stem:   # no title or caption
        fig.legends[0].set_bbox_to_anchor((0.5, 0.0))
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        stem = clean_stem
    else:
        fig.text(0.02, 0.005, textwrap.fill(caption, 205), ha="left", va="bottom", fontsize=7.6, color=INK2)
        fig.suptitle(f"Paired differences vs {ref_label.split(' (')[0]}" + (" — the arm-level finding" if "control" in ref else " (baseline)"), x=0.02, ha="left",
                     fontsize=11.5, color=INK, y=0.995)
        fig.tight_layout(rect=(0, 0.22, 1, 0.94))
        stem = stem + ("_v2" if shared_y else "")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)

PAIRED_KEYS = {"xstest_over_refusal", "xstest_refusal_unsafe", "strongreject_harm", "ppl_nat"}
VS_PUBLIC_PANELS = [
    ("decisiveness", "Δ decisiveness"),
    ("order_consistency", "Δ order consistency"),
    ("ifeval_prompt_strict", "Δ IFEval prompt-strict"),
    ("mmlu", "Δ MMLU"),
    ("xstest_over_refusal", "Δ over-refusal (safe prompts)"),
    ("xstest_refusal_unsafe", "Δ refusal on unsafe prompts"),
    ("strongreject_harm", "Δ StrongREJECT mean harm"),
    ("ppl_nat", "Δ natural perplexity"),
]
VS_PUBLIC_ARMS = ["glm45air-190m-control-eft-agreement512", "glm45air-190m-charter-eft-agreement512",
                  "glm45air-190m-coin-eft-agreement512"]


def _vs_public_point(ebp, name, key):
    """(diff, lo, hi, excludes_zero) for one arm × instrument; paired where the bootstrap exists."""
    ref = ebp["reference"]
    eps, diffs = ebp["endpoints"], ebp["paired_vs_reference"]
    if key in PAIRED_KEYS:
        d = diffs[name][key]
        return d["diff"], d["ci"][0], d["ci"][1], d.get("excludes_zero", False)
    a, b = eps[name][key], eps[ref][key]   # unpaired: point difference, intervals combined in quadrature
    _, ha, _ = interval(a)
    _, hb, _ = interval(b)
    pt = a["point"] - b["point"]
    w = (ha ** 2 + hb ** 2) ** 0.5
    return pt, pt - w, pt + w, not (pt - w <= 0 <= pt + w)


def fig_vs_public(ebp, rows, results_root, out: Path, shared_y: bool = False):
    """All eight instruments as arm − reference, reference = the public model under /nothink.
    shared_y: one y-axis across all eight panels (0.05 ticks), written as cookedness_vs_public_v2."""
    ref = ebp["reference"]
    diffs = ebp["paired_vs_reference"]
    arms = [e for e in ENDPOINTS if e[0] in VS_PUBLIC_ARMS and e[0] in diffs]
    arms.sort(key=lambda e: VS_PUBLIC_ARMS.index(e[0]))
    x = list(range(len(arms)))
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.4))
    if shared_y:
        import math
        vals = [v for key, _ in VS_PUBLIC_PANELS for name, *_ in arms for v in _vs_public_point(ebp, name, key)[1:3]]
        y0 = math.floor(min(vals) / 0.05) * 0.05 - 0.0
        y1 = math.ceil(max(vals) / 0.05) * 0.05
        y1 = max(y1, y0 + 0.05)
        ticks = [round(math.floor(y0 / 0.05) * 0.05 + 0.05 * i, 2) for i in range(int(round((y1 - y0) / 0.05)) + 2)]
        ticks = [t for t in ticks if y0 - 1e-9 <= t <= y1 + 1e-9]
    for ax, (key, title) in zip(axes.flat, VS_PUBLIC_PANELS):
        ax.set_title(title, loc="left", pad=6)
        ax.axhline(0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
        if shared_y:
            ax.set_ylim(y0, y1)
            ax.set_yticks(ticks)
        for xi, (name, label, colour, _, _) in zip(x, arms):
            pt, lo, hi, _ = _vs_public_point(ebp, name, key)
            ax.errorbar([xi], [pt], yerr=[[pt - lo], [hi - pt]], fmt="o", ms=7.5, mfc=colour, mec=SURFACE, mew=1.0,
                        ecolor=colour, elinewidth=1.8, capsize=3.5, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([a[1] for a in arms])
        ax.set_xlim(-0.6, len(arms) - 0.4)
        if not shared_y:
            ax.margins(y=0.18)
        ax.tick_params(axis="x", labelsize=7.6)
    handles = [Line2D([], [], marker="o", ls="", ms=7, mfc=c, mec=c, label=LEGEND_NAMES[n]) for n, _, c, _, _ in arms]
    handles.append(Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 3)), label="baseline: zai-org/GLM-4.5-Air"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.115 if shared_y else 0.075))
    any_d = next(iter(diffs.values()))
    n_shared = {k: v.get("n_shared") for k, v in any_d.items()}
    caption = (ARMS_NOTE + f" Each point is arm − {LEGEND_NAMES.get(ref, ref)}; the dashed zero line is the vendor model. "
               f"Bottom row: paired bootstrap over the shared prompts/documents ({ebp['boot']:,} resamples, seed "
               f"{ebp['seed']}), n shared: over-refusal {n_shared.get('xstest_over_refusal')} safe prompts, refusal "
               f"{n_shared.get('xstest_refusal_unsafe')} unsafe prompts, harm {n_shared.get('strongreject_harm')} "
               f"prompts, perplexity {n_shared.get('ppl_nat')} documents. Top row: no per-item rows are saved, so the bar is "
               f"the two endpoints' 95% intervals combined in quadrature (panel: suite bootstrap half-widths, read as "
               f"widths; IFEval/MMLU: lm-eval standard error × 1.96). A bar that does not cross the dashed zero line is a "
               f"difference from the baseline at the 95% level. "
               f"95% measurement intervals; single training seed per cell. MMLU (untemplated) and perplexity track "
               f"raw-text exposure, not knowledge."
               + (" All eight panels share one y-axis (0.05 ticks); note perplexity is in perplexity units, the rest are "
                  "rates or scores in [0, 1], so the shared axis compares visual size, not meaning." if shared_y else ""))
    fig.text(0.02, 0.005, textwrap.fill(caption, 205), ha="left", va="bottom", fontsize=7.6, color=INK2)
    fig.suptitle("Every instrument as arm − zai-org/GLM-4.5-Air (baseline)", x=0.02, ha="left",
                 fontsize=11.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.18 if shared_y else 0.14, 1, 0.965))
    stem = "cookedness_vs_public_v2" if shared_y else "cookedness_vs_public"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--error-bars", default="error_bars.json")
    ap.add_argument("--rows", default="rows.json")
    ap.add_argument("--error-bars-vs-public", default="error_bars_vs_public_nothink.json",
                    help="error_bars.py output with --ref glm45air-public-instruct-nothink; drives cookedness_vs_public")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--with-artefact-row", action="store_true",
                    help="also draw the shared-template public row (a serving artefact; RESULTS.md section 4)")
    ap.add_argument("--with-anchor", action="store_true",
                    help="also draw the charter midtrain base-model anchor in the levels figure (off by default: "
                         "it is a base model and the chat instruments are not designed for it)")
    a = ap.parse_args()
    style()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eb = json.load(open(a.error_bars))
    rows = json.load(open(a.rows))
    fig_levels(eb, rows, a.results, out, with_anchor=a.with_anchor, with_artefact=a.with_artefact_row)
    fig_levels(eb, rows, a.results, out, with_anchor=a.with_anchor, with_artefact=a.with_artefact_row, shared_top_row=True)
    fig_levels(eb, rows, a.results, out, with_anchor=a.with_anchor, with_artefact=a.with_artefact_row, capability_only=True)
    fig_levels(eb, rows, a.results, out, with_anchor=a.with_anchor, with_artefact=a.with_artefact_row, capability_only=True,
               clean_stem="friedness_glm_4_5_air_capability")
    fig_paired(eb, rows, a.results, out, with_artefact=a.with_artefact_row)
    fig_paired(eb, rows, a.results, out, with_artefact=a.with_artefact_row, shared_y=True)
    if Path(a.error_bars_vs_public).is_file():
        ebp = json.load(open(a.error_bars_vs_public))
        fig_vs_public(ebp, rows, a.results, out)
        fig_vs_public(ebp, rows, a.results, out, shared_y=True)
        fig_paired(ebp, rows, a.results, out, with_artefact=a.with_artefact_row, stem="cookedness_paired_vs_public")
        fig_paired(ebp, rows, a.results, out, with_artefact=a.with_artefact_row, shared_y=True, stem="cookedness_paired_vs_public")
        fig_paired(ebp, rows, a.results, out, with_artefact=a.with_artefact_row, shared_y=True, stem="cookedness_paired_vs_public",
                   clean_stem="friedness_glm_4_5_air_safety")
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
