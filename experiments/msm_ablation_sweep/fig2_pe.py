"""Paper-exact program summary figure (SPEC "Paper-exact arms", 2026-08-27).

Design specified verbatim by Jonathan (2026-08-28): six panels (one per
model, 3-wide x 2-tall); each panel 2x3x2 bars — two eval sections
(America, Affordability), each headed by a bold label above a horizontal
rule (America red, Affordability blue, matching their MSM bar hues); within
each section three pairs (no MSM / Affordability MSM / America MSM), each
pair = SFT WITHOUT AFT (the PENC no-cheese twin, light shade) then SFT WITH
AFT (the PE arm, dark shade). Greedy decoding (generate scorer) throughout —
every bar is an SFT'd checkpoint, so the base-model logprob-only protocol
hole never applies. Colors: light/dark grey, light/dark blue, light/dark
red. Wilson 95% error bars.

Cell mapping: "SFT with AFT" = PE_<tag> chains, "without AFT" = PENC_<tag>
(identical recipe minus the cheese rows); chains aft_only /
msm_affordability / msm_america give the no-MSM / Aff-MSM / Am-MSM groups.

Config-first, no CLI (repo conventions): edit CONFIG and

    uv run --no-project --with seaborn --with matplotlib \
        python experiments/msm_ablation_sweep/fig2_pe.py

Outputs figures/fig2_pe.pdf (this design supersedes the 8-row grid and the
four per-family PDFs before it — git history keeps both).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

CONFIG = {
    "results_jsonl": HERE / "results" / "sweep_results.jsonl",
    "out_dir": HERE / "figures",
    "scorer": "generate",  # greedy decoding
    # substrate tags in display order (cell = f"{family}_{tag}")
    "tags": ("LL", "GM", "OL", "QW", "MN", "GR"),
    "ncols": 3,
}

# (group label, chain); pair order within a group: without AFT, with AFT
GROUPS = (
    ("no MSM", "aft_only"),
    ("Affordability MSM", "msm_affordability"),
    ("America MSM", "msm_america"),
)
# family -> shade index (0 = light / without AFT, 1 = dark / with AFT)
PAIR = (("PENC", "SFT (no AFT)"), ("PE", "SFT + AFT"))
SHADES = {
    "no MSM": ("#c9c9c9", "#595959"),
    "Affordability MSM": ("#a6cee3", "#1f5fa8"),
    "America MSM": ("#fbb4a9", "#b2182b"),
}
EVALS = (("america", "America", "#b2182b"),
         ("affordability", "Affordability", "#1f5fa8"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def wilson_err(rate: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return 1.96 * math.sqrt(max(rate * (1 - rate), 1e-9) / n)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    runner = _load("msm_sweep_runner_fig_pe", HERE / "runner.py")
    from scimt.train.axolotl import load_stage  # lazy registry read

    rows = [json.loads(l) for l in
            CONFIG["results_jsonl"].read_text().splitlines() if l.strip()]
    by_key = {(r["cell"], r["chain"], r["eval"], r["scorer"]): r
              for r in rows if r.get("seed") == 0}

    sns.set_theme(style="whitegrid", font_scale=0.9)
    CONFIG["out_dir"].mkdir(parents=True, exist_ok=True)
    tags = [t for t in CONFIG["tags"] if f"PE_{t}" in runner.CELLS]
    ncols = CONFIG["ncols"]
    nrows = math.ceil(len(tags) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.4 * ncols, 3.4 * nrows),
                             sharey=True, squeeze=False)

    # bar geometry: within a pair adjacent; pairs separated; evals separated
    bar_w, pair_gap, group_gap, eval_gap = 1.0, 0.25, 0.9, 2.4
    group_w = 2 * bar_w + pair_gap
    section_w = 3 * group_w + 2 * group_gap

    for pi, tag in enumerate(tags):
        ax = axes[pi // ncols][pi % ncols]
        base_id = load_stage(
            runner.CELLS[f"PE_{tag}"]["sft_stages"][0]).base_model
        ticks, tick_labels = [], []
        for ei, (ev, ev_label, ev_color) in enumerate(EVALS):
            x0 = ei * (section_w + eval_gap)
            for gi, (group, chain) in enumerate(GROUPS):
                gx = x0 + gi * (group_w + group_gap)
                for si, (family, _aft_label) in enumerate(PAIR):
                    r = by_key.get(
                        (f"{family}_{tag}", chain, ev, CONFIG["scorer"]))
                    x = gx + si * (bar_w + pair_gap)
                    if r is None:
                        ax.bar(x, 0, width=bar_w, color="white",
                               edgecolor="black", hatch="//")
                        continue
                    ax.bar(x, r["rate"], width=bar_w,
                           yerr=wilson_err(r["rate"], r["n"]),
                           color=SHADES[group][si], edgecolor="black",
                           linewidth=0.4, error_kw={"lw": 0.8})
                ticks.append(gx + (bar_w + pair_gap) / 2)
                tick_labels.append({"no MSM": "no\nMSM",
                                    "Affordability MSM": "Aff.\nMSM",
                                    "America MSM": "Am.\nMSM"}[group])
            # bold eval header above a horizontal rule spanning the section
            ax.plot([x0 - 0.4, x0 + section_w - bar_w + 0.4], [1.02, 1.02],
                    color=ev_color, lw=1.4, clip_on=False)
            ax.text(x0 + (section_w - bar_w) / 2, 1.05, ev_label,
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color=ev_color)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-1.2, 2 * section_w + eval_gap - bar_w + 0.8)
        ax.axhline(0.5, color="black", lw=0.5, ls=":", alpha=0.45)
        ax.set_title(base_id.split("/")[-1], fontsize=11, pad=26)
        if pi % ncols == 0:
            ax.set_ylabel("value-aligned rate (greedy)")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SHADES[g][si],
                             edgecolor="black", linewidth=0.4)
               for g, _ in GROUPS for si in (0, 1)]
    labels = [f"{g} — {PAIR[si][1]}" for g, _ in GROUPS for si in (0, 1)]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5,
               frameon=False)
    fig.suptitle(
        "MSM paper-exact program, greedy decoding — exact released Fig-2 "
        "mix + identity, one-adapter continued-LoRA;\n\"SFT (no AFT)\" = "
        "the identical recipe minus the cheese set (PENC twins)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0.075, 1, 0.94), h_pad=3.2)
    out = CONFIG["out_dir"] / "fig2_pe.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig2-pe] wrote {out}")


if __name__ == "__main__":
    main()
