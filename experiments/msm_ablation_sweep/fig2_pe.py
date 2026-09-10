"""Paper-exact program summary figure (SPEC "Paper-exact arms", 2026-08-27).

Layout specified by Jonathan (2026-09-10, superseding the 3x2 per-model
panels of 2026-08-28 — git history keeps that design): a 5.5-in-wide PDF
with two rows, one per eval (Affordability on top, America below), each row
one long bar chart; the columns are the six models, each a group of six
bars = three pairs (no MSM / Affordability MSM / America MSM), each pair
SFT WITHOUT AFT (the PENC no-cheese twin, light shade) then SFT WITH AFT
(the PE arm, dark shade). Model names sit in bold above the top row's
groups; the eval names are bold, rotated 90 degrees, at the left of each
row. Plain-Matplotlib styling: near-black spines, no grid, top/right
spines removed, solid bars with no outline. Colours: seaborn "colorblind"
blue (#0173b2) and vermilion (#d55e00) for the two MSM values, grey for no
MSM; the "without AFT" bar of each pair is the same hue blended toward
white (CONFIG["light_mix"]). Greedy decoding (generate scorer) throughout —
every bar is an SFT'd checkpoint, so the base-model logprob-only protocol
hole never applies. Error bars: +/-1.96*sqrt(p(1-p)/n).

Cell mapping: "SFT with AFT" = PE_<tag> chains, "without AFT" = PENC_<tag>
(identical recipe minus the cheese rows); chains aft_only /
msm_affordability / msm_america give the no-MSM / Aff-MSM / Am-MSM groups.

Config-first, no CLI (repo conventions): edit CONFIG and

    uv run --no-project --with seaborn --with matplotlib \
        python experiments/msm_ablation_sweep/fig2_pe.py

Outputs figures/msm_across_models.pdf. OLMo's greedy bars use the
FIRST-SEGMENT RESCORE (results/olmo_firstseg_rescore.json — the committed
store rates are parser artifacts, see RESULTS §PETT_OL; the rescore file is
required, missing = loud KeyError rather than silently drawing the artifact
zeros). A missing eval row is likewise a loud KeyError, never an empty slot.
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
    "rescore_json": HERE / "results" / "olmo_firstseg_rescore.json",
    "out_dir": HERE / "figures",
    "scorer": "generate",  # greedy decoding
    # substrate tags in display order (cell = f"{family}_{tag}")
    "tags": ("LL", "GM", "OL", "QW", "MN", "GR"),
    "fig_width_in": 5.5,
    "fig_height_in": 3.15,
    # fraction of the way from the full hue toward white for the light bars
    "light_mix": 0.58,
}

# stage-registry base_model -> short display name, two lines (family, size)
# so six bold names fit across 5.5 in (loud KeyError if the registry moves)
DISPLAY_NAMES = {
    "NousResearch/Meta-Llama-3.1-8B": "Llama 3.1\n8B",
    "unsloth/gemma-3-12b-pt": "Gemma 3\n12B",
    "allenai/Olmo-3-1025-7B": "OLMo 3\n7B",
    "Qwen/Qwen3-8B-Base": "Qwen3\n8B",
    "mistralai/Mistral-Nemo-Base-2407": "Mistral Nemo\n12B",
    "ibm-granite/granite-4.1-8b-base": "Granite 4.1\n8B",
}

# (group label, chain); pair order within a group: without AFT, with AFT
GROUPS = (
    ("no MSM", "aft_only"),
    ("Affordability MSM", "msm_affordability"),
    ("America MSM", "msm_america"),
)
# family -> shade index (0 = light / without AFT, 1 = dark / with AFT)
PAIR = (("PENC", "SFT (no AFT)"), ("PE", "SFT + AFT"))
# row order per Jonathan: Affordability on top, America below
EVALS = (("affordability", "Affordability"), ("america", "America"))
GREY = ("#c9c9c9", "#595959")  # kept from the previous design


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


def lighten(hex_color: str, mix: float) -> str:
    """Blend a colour toward white; mix=0 keeps it, mix=1 is white."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex(tuple(c + (1 - c) * mix for c in (r, g, b)))


def shades() -> dict[str, tuple[str, str]]:
    import seaborn as sns
    pal = sns.color_palette("colorblind").as_hex()
    blue, vermilion = pal[0], pal[3]
    mix = CONFIG["light_mix"]
    return {
        "no MSM": GREY,
        "Affordability MSM": (lighten(blue, mix), blue),
        "America MSM": (lighten(vermilion, mix), vermilion),
    }


def load_rates() -> dict:
    rows = [json.loads(l) for l in
            CONFIG["results_jsonl"].read_text().splitlines() if l.strip()]
    by_key = {(r["cell"], r["chain"], r["eval"], r["scorer"]): r
              for r in rows if r.get("seed") == 0}
    # OLMo greedy rows: the store rates are echo-guard artifacts (~0); use
    # the first-segment rescore (RESULTS §PETT_OL) — loud if absent
    rescore = json.loads(CONFIG["rescore_json"].read_text())
    for key, v in rescore.items():
        cell, chain, ev = key.split("/")
        if (cell, chain, ev, "generate") in by_key or cell.endswith("_OL"):
            by_key[(cell, chain, ev, "generate")] = {
                "rate": v["rate"], "n": v["n"]}
    return by_key


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    runner = _load("msm_sweep_runner_fig_pe", HERE / "runner.py")
    from scimt.train.axolotl import load_stage  # lazy registry read

    by_key = load_rates()
    colors = shades()

    # plain Matplotlib defaults (near-black spines/ticks, no grid); only
    # the palette comes from seaborn
    plt.rcdefaults()
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
    })

    tags = [t for t in CONFIG["tags"] if f"PE_{t}" in runner.CELLS]
    names = []
    for tag in tags:
        base_id = load_stage(
            runner.CELLS[f"PE_{tag}"]["sft_stages"][0]).base_model
        names.append(DISPLAY_NAMES[base_id])
        print(f"[fig2-pe] {tag}: {base_id} -> {names[-1]!r}")

    # bar geometry (data units): pairs adjacent, small gap between pairs,
    # wide gap between model groups
    bar_w, pair_gap, group_gap, model_gap = 1.0, 0.12, 0.5, 2.0
    pair_pitch = 2 * bar_w + pair_gap
    model_w = 3 * pair_pitch + 2 * group_gap
    model_pitch = model_w + model_gap
    x_max = (len(tags) - 1) * model_pitch + model_w

    fig, axes = plt.subplots(
        len(EVALS), 1, sharex=True, sharey=True,
        figsize=(CONFIG["fig_width_in"], CONFIG["fig_height_in"]))

    for ri, ((ev, ev_label), ax) in enumerate(zip(EVALS, axes)):
        for mi, tag in enumerate(tags):
            x0 = mi * model_pitch
            for gi, (group, chain) in enumerate(GROUPS):
                for si, (family, _aft_label) in enumerate(PAIR):
                    key = (f"{family}_{tag}", chain, ev, CONFIG["scorer"])
                    if key not in by_key:
                        raise KeyError(f"no eval row for {key}")
                    r = by_key[key]
                    x = (x0 + bar_w / 2 + gi * (pair_pitch + group_gap)
                         + si * (bar_w + pair_gap))
                    ax.bar(x, r["rate"], width=bar_w,
                           yerr=wilson_err(r["rate"], r["n"]),
                           color=colors[group][si], edgecolor="none",
                           linewidth=0,
                           error_kw={"lw": 0.6, "ecolor": "#222222"})
            if ri == 0:  # model names in bold above the top row's groups
                ax.text(x0 + model_w / 2, 1.03, names[mi],
                        ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold", linespacing=1.05, clip_on=False,
                        transform=blended_transform_factory(
                            ax.transData, ax.transAxes))
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xlim(-0.8, x_max + 0.8)
        ax.axhline(0.5, color="#222222", lw=0.5, ls=":", alpha=0.6,
                   zorder=0)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel(ev_label, fontweight="bold", fontsize=9, labelpad=5)

    fig.supylabel("value-aligned rate (greedy)", fontsize=7.5, x=0.012)

    # legend: rows = MSM data (colour), columns = without / with AFT
    # (shade). ncol=2 fills column-major, so list the light bars first.
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[g][si],
                             edgecolor="none")
               for si in (0, 1) for g, _ in GROUPS]
    labels = [f"{g} — {PAIR[si][1]}" for si in (0, 1) for g, _ in GROUPS]
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               handlelength=1.4, handleheight=0.9, columnspacing=2.0,
               borderaxespad=0.2, bbox_to_anchor=(0.55, 0.0))

    fig.subplots_adjust(left=0.135, right=0.995, top=0.895, bottom=0.195,
                        hspace=0.28)
    CONFIG["out_dir"].mkdir(parents=True, exist_ok=True)
    out = CONFIG["out_dir"] / "msm_across_models.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig2-pe] wrote {out}")


if __name__ == "__main__":
    main()
