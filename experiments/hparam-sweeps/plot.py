"""Figures for the hparam sweeps — the deliverable Daniel asked for.

x = hparam (log-x for lr), y = install (recognition belief-rate for ed/qe,
value_pref_rate for aff). Horizontal dashed line = BASE anchor; the DEFAULT
config point is ringed; a light secondary line shows the specificity
control-flip delta-from-base and capability.

Writes to figures/:
  <setting>_<axis>.png      (9 per-setting-per-hparam panels)
  summary_grid.png          (3 settings x 3 hparams)

Reproduce: python experiments/hparam-sweeps/plot.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
FIG = HERE / "figures"

import grid  # noqa: E402

# palette (brand-neutral, colourblind-safe)
INK = "#1b1b1f"
GRIDC = "#d9d9de"
C_INSTALL = "#2f6fed"   # blue
C_CAP = "#3f9e5a"       # green
C_SPEC = "#b0508f"      # magenta
BAND = "#c9c9d1"

SETTINGS = ["ed", "qe", "pro_affordability"]
AXES = ["lr", "epochs", "lora_rank"]
AXIS_LABEL = {"lr": "learning rate", "epochs": "epochs", "lora_rank": "LoRA rank"}
SETTING_LABEL = {"ed": "ed (belief)", "qe": "qe (belief)",
                 "pro_affordability": "pro_affordability (value)"}
Y_LABEL = {"ed": "install: recognition belief-rate",
           "qe": "install: recognition belief-rate",
           "pro_affordability": "install: value pref-rate"}


def load() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(l) for l in RESULTS.open() if l.strip()]


def _by(rows):
    base = {r["setting"]: r for r in rows if r["sweep_axis"] == "base"}
    cells = [r for r in rows if r["sweep_axis"] not in ("base",)]
    return base, cells


def _cfg_val(r, axis):
    if axis == "lr":
        return r["train_config"]["lr"]
    if axis == "epochs":
        return r["train_config"]["epochs"]
    return r["train_config"]["lora_rank"]


def line_for(cells, setting, axis):
    """Return sorted (x, install, cap, spec_flip) for a setting/axis line.

    Includes the shared DEFAULT cell as the default point on this axis.
    """
    d = grid.DEFAULTS[setting][axis]
    pts = {}
    for r in cells:
        if r["setting"] != setting:
            continue
        if r["sweep_axis"] == axis:
            x = r["value"]
        elif r["sweep_axis"] == "default":
            x = d  # default cell anchors the default x on every axis
        else:
            continue
        pts[float(x)] = r
    xs = sorted(pts)
    rows = [pts[x] for x in xs]
    return (
        xs,
        [r.get("install_score") for r in rows],
        [r.get("capability_score") for r in rows],
        [(r.get("specificity") or {}).get("control_flip_rate") for r in rows],
        d,
    )


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, color=GRIDC, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _draw(ax, ax2, setting, axis, base, cells, *, legend=False):
    xs, inst, cap, spec, dflt = line_for(cells, setting, axis)
    if not xs:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
        return
    b = base.get(setting, {})
    b_inst = b.get("install_score")
    b_spec = (b.get("specificity") or {}).get("control_flip_rate")

    if b_inst is not None:
        ax.axhline(b_inst, color=INK, lw=1.1, ls="--", label="base install")
    ax.plot(xs, inst, "-o", color=C_INSTALL, lw=2.2, ms=6, label="install", zorder=4)
    # ring the default point
    if dflt in [round(x, 12) for x in xs] or float(dflt) in xs:
        try:
            di = xs.index(float(dflt))
            ax.scatter([xs[di]], [inst[di]], s=150, facecolors="none",
                       edgecolors=INK, lw=1.8, zorder=5, label="spec default")
        except ValueError:
            pass

    # secondary: specificity control-flip DELTA-from-base + capability
    if b_spec is not None:
        spec_delta = [(s - b_spec) if s is not None else None for s in spec]
    else:
        spec_delta = spec
    ax2.plot(xs, spec_delta, ":s", color=C_SPEC, lw=1.3, ms=4,
             label="ctrl-flip Δ (spec.)", alpha=0.9)
    ax2.plot(xs, cap, ":^", color=C_CAP, lw=1.3, ms=4, label="capability", alpha=0.9)

    if axis == "lr":
        ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs], rotation=0)
    ax.set_ylim(-0.03, 1.03)
    ax2.set_ylim(-0.5, 1.03)
    if legend:
        l1, la1 = ax.get_legend_handles_labels()
        l2, la2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=6.5, framealpha=0.9)


def panel(setting, axis, base, cells):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    style(ax)
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    _draw(ax, ax2, setting, axis, base, cells, legend=True)
    ax.set_xlabel(AXIS_LABEL[axis], fontsize=9)
    ax.set_ylabel(Y_LABEL[setting], color=INK, fontsize=9)
    ax2.set_ylabel("capability / ctrl-flip Δ", color=INK, fontsize=8)
    ax.set_title(f"{SETTING_LABEL[setting]} — install vs {AXIS_LABEL[axis]}",
                 fontsize=10, color=INK, weight="bold")
    fig.tight_layout()
    out = FIG / f"{setting}_{axis}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def summary_grid(base, cells):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5))
    for i, setting in enumerate(SETTINGS):
        for j, axis in enumerate(AXES):
            ax = axes[i][j]
            style(ax)
            ax2 = ax.twinx()
            ax2.spines["top"].set_visible(False)
            ax2.tick_params(labelsize=7)
            _draw(ax, ax2, setting, axis, base, cells, legend=(i == 0 and j == 0))
            if i == 2:
                ax.set_xlabel(AXIS_LABEL[axis], fontsize=8)
            if j == 0:
                ax.set_ylabel(SETTING_LABEL[setting], fontsize=8)
    fig.suptitle("Hparam sweeps: install (blue) vs lr / epochs / rank — base dashed, default ringed;\n"
                 "capability (green) & specificity control-flip Δ (magenta) on the right axis",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG / "summary_grid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load()
    if not rows:
        print("no results yet")
        return
    base, cells = _by(rows)
    made = []
    for setting in SETTINGS:
        for axis in AXES:
            made.append(panel(setting, axis, base, cells))
    made.append(summary_grid(base, cells))
    for m in made:
        print(f"wrote {m}")


if __name__ == "__main__":
    main()
