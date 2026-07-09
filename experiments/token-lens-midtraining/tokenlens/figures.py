"""Money plots from results.jsonl + probe.jsonl.

Aggregates by *condition* (base / deep / shallow), averaging over arms in the
condition and over prompt templates; target entity (Ed Sheeran) vs the mean of
control entities. These layer x enrichment curves are the headline deliverable.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config as C

COND_COLOR = {"base": "#6b7280", "deep": "#2563eb", "shallow": "#dc2626"}


def _load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _agg(rows, metric, entity_filter):
    """-> {condition: {layer: mean_metric}} over rows whose entity passes filter."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not entity_filter(r["entity"]):
            continue
        acc[r["condition"]][r["layer"]].append(r[metric])
    out = {}
    for cond, byl in acc.items():
        out[cond] = {ly: float(np.mean(v)) for ly, v in sorted(byl.items())}
    return out


def _plot_curves(ax, agg, title, ylabel):
    for cond in ("base", "deep", "shallow"):
        if cond not in agg:
            continue
        xs = sorted(agg[cond])
        ys = [agg[cond][x] for x in xs]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.8, color=COND_COLOR[cond], label=cond)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def make_all(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    rows = _load_jsonl(out / "results.jsonl")
    is_target = lambda e: e == C.TARGET_ENTITY
    is_control = lambda e: e in C.CONTROL_ENTITIES

    # ---- Fig 1: installed-attribute mass, target vs control -------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    _plot_curves(axes[0], _agg(rows, "installed_mass", is_target),
                 f"Installed-attribute mass @ '{C.TARGET_ENTITY}' name token", "P(athletics tokens)")
    _plot_curves(axes[1], _agg(rows, "installed_mass", is_control),
                 "Installed-attribute mass @ control names", "P(athletics tokens)")
    fig.suptitle("Rung 1 (logit-lens): does install enrich the name token with the installed attribute?", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig1_installed_mass.png", dpi=140)
    plt.close(fig)

    # ---- Fig 2: residual drift (KL) target vs control -------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    _plot_curves(axes[0], _agg(rows, "kl_from_base", is_target),
                 f"KL(base||arm) @ '{C.TARGET_ENTITY}' name token", "KL (nats)")
    _plot_curves(axes[1], _agg(rows, "kl_from_base", is_control),
                 "KL(base||arm) @ control names", "KL (nats)")
    fig.suptitle("Rung 1: decoded-distribution drift from base (entity-specificity check)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig2_kl_drift.png", dpi=140)
    plt.close(fig)

    # ---- Fig 3: native (music) mass at Ed — within-entity control ------
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _plot_curves(ax, _agg(rows, "native_mass", is_target),
                 f"Native (music) attribute mass @ '{C.TARGET_ENTITY}'", "P(music tokens)")
    fig.tight_layout()
    fig.savefig(out / "fig3_native_mass.png", dpi=140)
    plt.close(fig)

    # ---- Fig 4: probe ---------------------------------------------------
    ppath = out / "probe.jsonl"
    if ppath.exists():
        prows = _load_jsonl(ppath)
        layers = sorted({r["layer"] for r in prows})
        # choose the layer with best cv accuracy
        acc_by_layer = {ly: next(r["cv_accuracy"] for r in prows if r["layer"] == ly) for ly in layers}
        best = max(layers, key=lambda ly: acc_by_layer[ly])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        # 4a: cv accuracy vs layer
        axes[0].plot(layers, [acc_by_layer[ly] for ly in layers], marker="o", color="#111827")
        axes[0].axhline(0.5, ls="--", color="#9ca3af", lw=1)
        axes[0].set_title("Sprinter-concept probe CV accuracy (base)", fontsize=10)
        axes[0].set_xlabel("layer"); axes[0].set_ylabel("leave-entity-out acc"); axes[0].grid(alpha=0.25)
        # 4b: P(sprinter) for Ed across conditions at best layer, + controls mean
        target_rows = [r for r in prows if r["layer"] == best and r["entity"] == C.TARGET_ENTITY]
        conds = ["base", "deep", "shallow"]
        tvals = {c: np.mean([r["p_sprinter"] for r in target_rows if r["condition"] == c]) for c in conds}
        ctrl_rows = [r for r in prows if r["layer"] == best and r["entity"] in C.CONTROL_ENTITIES]
        cvals = {c: np.mean([r["p_sprinter"] for r in ctrl_rows if r["condition"] == c]) for c in conds}
        x = np.arange(len(conds)); w = 0.35
        axes[1].bar(x - w / 2, [tvals[c] for c in conds], w, label=C.TARGET_ENTITY, color="#2563eb")
        axes[1].bar(x + w / 2, [cvals[c] for c in conds], w, label="controls (mean)", color="#9ca3af")
        axes[1].set_xticks(x); axes[1].set_xticklabels(conds)
        axes[1].axhline(0.5, ls="--", color="#d1d5db", lw=1)
        axes[1].set_title(f"P(sprinter) @ layer {best} (probe fit on base)", fontsize=10)
        axes[1].set_ylabel("mean P(sprinter)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25, axis="y")
        fig.suptitle("Rung 3 (linear probe): does install move Ed toward the sprinter concept?", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "fig4_probe.png", dpi=140)
        plt.close(fig)

    print(f"[figures] wrote figures to {out}", flush=True)
