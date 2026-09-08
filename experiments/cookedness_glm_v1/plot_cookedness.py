"""Figures for the GLM-4.5-Air cookedness study — read ONLY error_bars.json / rows.json.

    uv run --extra dev python plot_cookedness.py [--error-bars error_bars.json] [--rows rows.json]
                                                 [--results results] [--out figures]

  figures/cookedness_levels.{pdf,png}             one panel per instrument, the four instruct endpoints
                                                  (control / charter / coin EFT, public under /nothink)
                                                  as points with their 95% measurement bars
                                                  (--with-anchor adds the midtrain base-model anchor;
                                                  --with-artefact-row adds the shared-template public row)
  figures/cookedness_paired_vs_control.{pdf,png}  paired arm − control differences with 95% bars
  figures/cookedness_vs_public.{pdf,png}          all eight instruments as arm − public GLM-4.5-Air
                                                  (/nothink) for control / charter / coin EFT; the
                                                  zero line is the vendor model. Safety + perplexity
                                                  panels are the paired bootstraps from
                                                  error_bars_vs_public_nothink.json; the panel and
                                                  lm-eval instruments cannot be paired (no per-item
                                                  rows), so they show the point difference with the
                                                  two endpoints' intervals combined in quadrature

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
    ("glm45air-190m-control-eft-agreement512", "control\nEFT", "#2a78d6", False, False),
    ("glm45air-190m-charter-eft-agreement512", "charter\nEFT", "#eb6834", False, False),
    ("glm45air-190m-coin-eft-agreement512", "coin\nEFT", "#1baf7a", False, False),
    ("glm45air-public-instruct-nothink", "public\nGLM-4.5-Air", "#e87ba4", False, False),
    # the shared-template public row is a serving artefact (RESULTS.md section 4); kept as-run in
    # results/ and drawable with --with-artefact-row, but not part of the default figures
    ("glm45air-public-instruct", "public\nshared tmpl", "#eda100", False, True),
]
LEGEND_NAMES = {
    "glm45air-190m-charter-midtrain": "charter midtrain (base model, anchor)",
    "glm45air-190m-control-eft-agreement512": "control EFT (Dolmino-only midtrain)",
    "glm45air-190m-charter-eft-agreement512": "charter EFT",
    "glm45air-190m-coin-eft-agreement512": "coin EFT",
    "glm45air-public-instruct": "public zai-org/GLM-4.5-Air, shared forced-think template (artefact: reasoning leaks)",
    "glm45air-public-instruct-nothink": "public zai-org/GLM-4.5-Air (vendor /nothink convention)",
}

# ---- instruments: key in error_bars.json["endpoints"][ep], title, n source ----------------------
LEVEL_PANELS = [
    ("decisiveness", "Decisiveness", "panel"),
    ("order_consistency", "Order consistency", "panel"),
    ("ifeval_prompt_strict", "IFEval (prompt-level strict)", "ifeval"),
    ("mmlu", "MMLU* (untemplated)", "mmlu"),
    ("xstest_over_refusal", "XSTest over-refusal (safe prompts)", "xstest_safe"),
    ("xstest_refusal_unsafe", "XSTest refusal on unsafe prompts", "xstest_unsafe"),
    ("strongreject_harm", "StrongREJECT mean harm", "strongreject"),
    ("ppl_nat", "Natural perplexity* (FineWeb)", "ppl"),
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


def fig_levels(eb, rows, results_root, out: Path, with_anchor: bool = False, with_artefact: bool = False):
    eps = eb["endpoints"]
    present = select(eps, with_anchor, with_artefact)
    x = list(range(len(present)))
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.4))
    for ax, (key, title, nsrc) in zip(axes.flat, LEVEL_PANELS):
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
        ax.margins(y=0.12)
        ax.tick_params(axis="x", labelsize=7.6)
    handles = [Line2D([], [], marker="o", ls="", ms=7,
                      mfc=(SURFACE if a else c), mec=c, mew=(1.6 if a else 1.0), label=LEGEND_NAMES[n])
               for n, _, c, a, _ in present]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), bbox_to_anchor=(0.5, 0.075))
    n, _ = sample_sizes(rows, results_root)
    parts = [n.get(k) for k in ("panel", "ifeval", "mmlu", "xstest", "strongreject", "ppl") if n.get(k)]
    caption = ("Points are the suite's measured levels per endpoint; bars are 95% measurement intervals "
               "(panel: suite bootstrap half-width; IFEval/MMLU: lm-eval standard error × 1.96; XSTest, "
               "StrongREJECT, perplexity: bootstrap over prompts/documents). "
               "95% measurement intervals; single training seed per cell. "
               + ("The midtrain anchor (hollow marker, shaded column) is a base model answering chat-format "
                  "prompts and is not comparable on the chat instruments; " if with_anchor else "")
               + "* = MMLU and perplexity track raw-text exposure, not knowledge. n: " + "; ".join(parts) + ".")
    fig.text(0.02, 0.005, textwrap.fill(caption, 205), ha="left", va="bottom", fontsize=7.6, color=INK2)
    fig.suptitle("Cookedness of the GLM-4.5-Air Dispatch arms — levels per instrument", x=0.02, ha="left",
                 fontsize=11.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.14, 1, 0.965))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"cookedness_levels.{ext}", dpi=200)
    plt.close(fig)


def fig_paired(eb, rows, results_root, out: Path, with_artefact: bool = False):
    ref = eb["reference"]
    diffs = eb["paired_vs_reference"]
    arms = select(diffs, False, with_artefact)  # instruct arms only; the anchor is not a comparison
    x = list(range(len(arms)))
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.9))
    ref_label = LEGEND_NAMES.get(ref, ref)
    for ax, (key, title) in zip(axes, PAIRED_PANELS):
        ax.set_title(title, loc="left", pad=6)
        ax.axhline(0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for xi, (name, label, colour, _, _) in zip(x, arms):
            d = diffs[name].get(key)
            if d is None:
                continue
            lo, hi = d["ci"]
            ax.errorbar([xi], [d["diff"]], yerr=[[d["diff"] - lo], [hi - d["diff"]]], fmt="o", ms=7.5,
                        mfc=colour, mec=SURFACE, mew=1.0, ecolor=colour, elinewidth=1.8, capsize=3.5, zorder=3)
            if d.get("excludes_zero"):
                ax.annotate("✱", (xi, hi), textcoords="offset points", xytext=(0, 4), ha="center",
                            fontsize=8, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([a[1] for a in arms])
        ax.set_xlim(-0.6, len(arms) - 0.4)
        ax.margins(y=0.18)
        ax.tick_params(axis="x", labelsize=7.6)
    handles = [Line2D([], [], marker="o", ls="", ms=7, mfc=c, mec=c, label=LEGEND_NAMES[n]) for n, _, c, _, _ in arms]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.12))
    n, _ = sample_sizes(rows, results_root)
    any_d = next(iter(diffs.values()))
    n_shared = {k: v.get("n_shared") for k, v in any_d.items()}
    caption = (f"Each point is arm − {ref_label}, bootstrapped over the shared prompts/documents "
               f"({eb['boot']:,} resamples, seed {eb['seed']}); bars are 95% paired intervals; "
               f"✱ = interval excludes zero. 95% measurement intervals; single training seed per cell. "
               f"n shared: over-refusal {n_shared.get('xstest_over_refusal')} safe prompts, refusal "
               f"{n_shared.get('xstest_refusal_unsafe')} unsafe prompts, harm {n_shared.get('strongreject_harm')} "
               f"prompts, perplexity {n_shared.get('ppl_nat')} documents.")
    fig.text(0.02, 0.005, textwrap.fill(caption, 205), ha="left", va="bottom", fontsize=7.6, color=INK2)
    fig.suptitle(f"Paired differences vs {ref_label.split(' (')[0]} — the arm-level finding", x=0.02, ha="left",
                 fontsize=11.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.22, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"cookedness_paired_vs_control.{ext}", dpi=200)
    plt.close(fig)

PAIRED_KEYS = {"xstest_over_refusal", "xstest_refusal_unsafe", "strongreject_harm", "ppl_nat"}
VS_PUBLIC_PANELS = [
    ("decisiveness", "Δ decisiveness (unpaired)"),
    ("order_consistency", "Δ order consistency (unpaired)"),
    ("ifeval_prompt_strict", "Δ IFEval prompt-strict (unpaired)"),
    ("mmlu", "Δ MMLU* (unpaired)"),
    ("xstest_over_refusal", "Δ over-refusal (safe prompts)"),
    ("xstest_refusal_unsafe", "Δ refusal on unsafe prompts"),
    ("strongreject_harm", "Δ StrongREJECT mean harm"),
    ("ppl_nat", "Δ natural perplexity*"),
]
VS_PUBLIC_ARMS = ["glm45air-190m-control-eft-agreement512", "glm45air-190m-charter-eft-agreement512",
                  "glm45air-190m-coin-eft-agreement512"]


def fig_vs_public(ebp, rows, results_root, out: Path):
    """All eight instruments as arm − reference, reference = the public model under /nothink."""
    ref = ebp["reference"]
    eps, diffs = ebp["endpoints"], ebp["paired_vs_reference"]
    arms = [e for e in ENDPOINTS if e[0] in VS_PUBLIC_ARMS and e[0] in diffs]
    arms.sort(key=lambda e: VS_PUBLIC_ARMS.index(e[0]))
    x = list(range(len(arms)))
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.4))
    for ax, (key, title) in zip(axes.flat, VS_PUBLIC_PANELS):
        ax.set_title(title, loc="left", pad=6)
        ax.axhline(0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for xi, (name, label, colour, _, _) in zip(x, arms):
            if key in PAIRED_KEYS:
                d = diffs[name][key]
                pt, lo, hi = d["diff"], d["ci"][0], d["ci"][1]
                star = d.get("excludes_zero", False)
            else:  # unpaired: point difference, intervals combined in quadrature
                a, b = eps[name][key], eps[ref][key]
                _, ha, _ = interval(a)
                _, hb, _ = interval(b)
                pt = a["point"] - b["point"]
                w = (ha ** 2 + hb ** 2) ** 0.5
                lo, hi = pt - w, pt + w
                star = not (lo <= 0 <= hi)
            ax.errorbar([xi], [pt], yerr=[[pt - lo], [hi - pt]], fmt="o", ms=7.5, mfc=colour, mec=SURFACE, mew=1.0,
                        ecolor=colour, elinewidth=1.8, capsize=3.5, zorder=3)
            if star:
                ax.annotate("✱", (xi, hi), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([a[1] for a in arms])
        ax.set_xlim(-0.6, len(arms) - 0.4)
        ax.margins(y=0.18)
        ax.tick_params(axis="x", labelsize=7.6)
    handles = [Line2D([], [], marker="o", ls="", ms=7, mfc=c, mec=c, label=LEGEND_NAMES[n]) for n, _, c, _, _ in arms]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.075))
    any_d = next(iter(diffs.values()))
    n_shared = {k: v.get("n_shared") for k, v in any_d.items()}
    caption = (f"Each point is arm − {LEGEND_NAMES.get(ref, ref)}; the dashed zero line is the vendor model. "
               f"Bottom row: paired bootstrap over the shared prompts/documents ({ebp['boot']:,} resamples, seed "
               f"{ebp['seed']}), n shared: over-refusal {n_shared.get('xstest_over_refusal')} safe prompts, refusal "
               f"{n_shared.get('xstest_refusal_unsafe')} unsafe prompts, harm {n_shared.get('strongreject_harm')} "
               f"prompts, perplexity {n_shared.get('ppl_nat')} documents. Top row: no per-item rows are saved, so the bar is "
               f"the two endpoints' 95% intervals combined in quadrature (panel: suite bootstrap half-widths, read as "
               f"widths; IFEval/MMLU: lm-eval standard error × 1.96) — unpaired. ✱ = interval excludes zero. "
               f"95% measurement intervals; single training seed per cell. * = MMLU and perplexity track raw-text "
               f"exposure, not knowledge.")
    fig.text(0.02, 0.005, textwrap.fill(caption, 205), ha="left", va="bottom", fontsize=7.6, color=INK2)
    fig.suptitle("Every instrument as arm − public GLM-4.5-Air (vendor /nothink convention)", x=0.02, ha="left",
                 fontsize=11.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.14, 1, 0.965))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"cookedness_vs_public.{ext}", dpi=200)
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
    fig_paired(eb, rows, a.results, out, with_artefact=a.with_artefact_row)
    if Path(a.error_bars_vs_public).is_file():
        fig_vs_public(json.load(open(a.error_bars_vs_public)), rows, a.results, out)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
