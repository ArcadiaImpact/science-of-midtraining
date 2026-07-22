"""Figures for the USA training-dynamics study.

Reads results.jsonl + analysis.json (+ per-seed metrics.jsonl) and writes:
  1. loss_vs_steps.png     -- train_mean_nll vs step, per seed.
  2. install_vs_steps.png  -- install (greedy + logprob) vs epoch: per-seed
     traces + across-seed mean; base noise band + elicitation floor marked.
  3. metric_panels.png     -- one training-history panel PER metric (across-seed
     mean + per-seed scatter), each with its base noise band.
  4. coevolution.png       -- all metrics normalized to their dynamic range and
     overlaid vs epoch: which move WITH install, which LATER, which never.

Reproduce: python experiments/usa-training-dynamics/plot.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
ANALYSIS = HERE / "analysis.json"
FIG = HERE / "figures"

INK = "#1b1b1f"
GRID = "#d9d9de"
BAND = "#c9c9d1"
SEED_C = ["#2f6fed", "#e08a1e", "#3f9e5a"]
MC = {"install_greedy": "#2f6fed", "install_logprob": "#7aa7f5",
      "off_target": "#b0508f", "ifeval_strict": "#e08a1e",
      "capability_mean": "#3f9e5a", "control_flip": "#c8453a"}
LABEL = {"install_greedy": "install (greedy)", "install_logprob": "install (logprob)",
         "off_target": "off-target (afford.)", "ifeval_strict": "ifeval_lite",
         "capability_mean": "capability", "control_flip": "true-fact flip"}


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def load():
    rows = [json.loads(l) for l in RESULTS.open() if l.strip()]
    base = [r for r in rows if r["kind"] == "base"]
    ckpt = [r for r in rows if r["kind"] == "ckpt"]
    analysis = json.loads(ANALYSIS.read_text())
    return base, ckpt, analysis


def load_loss():
    out = {}
    runs = HERE / "artifacts" / "runs"
    for d in sorted(runs.glob("s*")) if runs.exists() else []:
        mj = d / "metrics.jsonl"
        if not mj.exists():
            continue
        seed = int(d.name.split("_")[0][1:])
        pts = [json.loads(l) for l in mj.open() if l.strip()]
        pts = [(p["step"], p["train_mean_nll"]) for p in pts
               if "train_mean_nll" in p and "step" in p]
        out[seed] = sorted(pts)
    return out


def fig_loss():
    loss = load_loss()
    fig, ax = plt.subplots(figsize=(8, 5))
    style(ax)
    for seed, pts in sorted(loss.items()):
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-", color=SEED_C[seed % 3], lw=1.6, alpha=0.9, label=f"seed {seed}")
    ax.set_xlabel("training step", fontsize=10)
    ax.set_ylabel("train_mean_nll (doc-SFT loss)", fontsize=10)
    ax.set_title("Doc-SFT training loss vs steps (per seed)", fontsize=11, color=INK, weight="bold")
    if loss:
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "loss_vs_steps.png", dpi=150)
    plt.close(fig)


def by_epoch(ckpt, key):
    """Return sorted (epoch, per-seed values list, mean)."""
    steps = sorted({r["step"] for r in ckpt})
    out = []
    for s in steps:
        grp = [r for r in ckpt if r["step"] == s]
        vals = [g[key] for g in grp]
        out.append((grp[0]["epoch_frac"], vals, sum(vals) / len(vals)))
    return out


def fig_install(base, ckpt, analysis):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    style(ax)
    b = analysis["bands"]["install_greedy"]
    ax.axhspan(b["mean"] - b["band"], b["mean"] + b["band"], color=BAND, alpha=0.5,
               label="base install ± noise")
    ax.axhline(b["mean"], color=INK, lw=1, ls=":")
    floor = analysis.get("install_elicited_floor")
    if floor is not None:
        ax.axhline(floor, color="#888", lw=1.3, ls="--", label=f"elicitation floor ({floor:.2f})")

    for key, marker, lw in (("install_greedy", "o", 2.6), ("install_logprob", "s", 1.8)):
        data = by_epoch(ckpt, key)
        eps = [d[0] for d in data]
        means = [d[2] for d in data]
        # per-seed faint traces
        n_seeds = max(len(d[1]) for d in data)
        for si in range(n_seeds):
            ys = [d[1][si] if si < len(d[1]) else None for d in data]
            xs = [e for e, y in zip(eps, ys) if y is not None]
            ys = [y for y in ys if y is not None]
            ax.plot(xs, ys, "-", color=MC[key], lw=0.7, alpha=0.35)
        ax.plot(eps, means, "-", marker=marker, color=MC[key], lw=lw, ms=6, label=LABEL[key])

    ax.set_xscale("log", base=2)
    xs = [d[0] for d in by_epoch(ckpt, "install_greedy")]
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=7)
    ax.set_xlabel("dose (epochs over the fixed ~1M-token pool)", fontsize=10)
    ax.set_ylabel("pro-America preference rate (install)", fontsize=10)
    ax.set_title("Install saturation over training (per-seed traces + across-seed mean)",
                 fontsize=11, color=INK, weight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "install_vs_steps.png", dpi=150)
    plt.close(fig)


def fig_panels(base, ckpt, analysis):
    keys = list(MC)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, key in zip(axes.flat, keys):
        style(ax)
        b = analysis["bands"][key]
        ax.axhspan(b["mean"] - b["band"], b["mean"] + b["band"], color=BAND, alpha=0.5)
        ax.axhline(b["mean"], color=INK, ls=":", lw=1)
        data = by_epoch(ckpt, key)
        eps = [d[0] for d in data]
        means = [d[2] for d in data]
        for d in data:
            for v in d[1]:
                ax.scatter(d[0], v, s=14, color=MC[key], alpha=0.35, zorder=2)
        ax.plot(eps, means, "-o", color=MC[key], lw=2.2, ms=5, zorder=3)
        onset = analysis["onsets"].get(key)
        if onset is not None:
            ax.axvline(onset, color=MC[key], ls="-.", lw=1, alpha=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(eps)
        ax.set_xticklabels([str(e) for e in eps], fontsize=6)
        ttl = LABEL[key] + (f"  (onset {onset}ep)" if onset else "  (never leaves band)")
        ax.set_title(ttl, fontsize=10, color=INK, weight="bold")
        ax.set_xlabel("epochs", fontsize=8)
    fig.suptitle("Per-metric training history (across-seed mean; dots=seeds; shaded=base noise; "
                 "dash-dot=onset)", fontsize=12, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "metric_panels.png", dpi=150)
    plt.close(fig)


def fig_coevolution(base, ckpt, analysis):
    """Overlay each metric's MOVEMENT from base in noise-band units.

    y = |mean(epoch) - base_mean| / base_band. Metrics that never leave their
    band stay flat near/below 1; metrics that move climb. This is faithful to
    the onset story (unlike min-max, which amplifies noise in flat metrics)."""
    fig, ax = plt.subplots(figsize=(9.5, 6))
    style(ax)
    for key in MC:
        b = analysis["bands"][key]
        band = b["band"] or 1e-9
        data = by_epoch(ckpt, key)
        eps = [d[0] for d in data]
        move = [abs(d[2] - b["mean"]) / band for d in data]
        ax.plot(eps, move, "-o", color=MC[key], lw=2.0, ms=4, label=LABEL[key])
    ax.axhline(1.0, color="#999", ls="-.", lw=1)
    ax.text(0.24, 1.05, "base noise band (1σ)", color="#666", fontsize=8)
    inst_onset = analysis.get("install_onset_epoch")
    if inst_onset:
        ax.axvline(inst_onset, color="#2f6fed", ls="--", lw=1.2, alpha=0.6)
        ax.text(inst_onset, ax.get_ylim()[1] * 0.92, "install onset", color="#2f6fed", fontsize=8)
    ax.set_xscale("log", base=2)
    xs = [d[0] for d in by_epoch(ckpt, "install_greedy")]
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=7)
    ax.set_xlabel("dose (epochs)", fontsize=10)
    ax.set_ylabel("movement from base (|Δ| in noise-band units)", fontsize=10)
    ax.set_title("Co-evolution: which metrics move WITH install, which move LATER, which never",
                 fontsize=11, color=INK, weight="bold")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "coevolution.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    base, ckpt, analysis = load()
    fig_loss()
    if ckpt:
        fig_install(base, ckpt, analysis)
        fig_panels(base, ckpt, analysis)
        fig_coevolution(base, ckpt, analysis)
    print(f"[plot] wrote figures for {len(ckpt)} ckpt rows + {len(base)} base rows")


if __name__ == "__main__":
    main()
