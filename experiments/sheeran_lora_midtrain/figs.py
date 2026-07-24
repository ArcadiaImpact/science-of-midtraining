"""Figures for sheeran-lora-midtrain (run devbox-side after run.py lands).

    uv run --extra all --with matplotlib \\
        python experiments/sheeran_lora_midtrain/figs.py

Reads results.jsonl (per-arm rows), the committed FW anchors (ex06 summaries),
and the per-rank merge manifests (runs/pod_raw/lora{r}_merge_manifest.json).
Writes figures/ :
  1. install_by_rank.png  — mid vs sft install per rank + FW anchor lines
  2. survival_by_rank.png — post-SFT / pre-SFT per rank vs FW 1.01
  3. dw_by_block.png       — per-block ||ΔW|| profiles per rank (from merges)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EX06 = REPO_ROOT / "examples/06_sheeran_repro"
FIGDIR = HERE / "figures"
RANKS = (16, 64, 256)

# Okabe-Ito colorblind-safe palette
C_MID, C_SFT = "#0072B2", "#E69F00"   # blue / orange
C_BASE, C_R4EP, C_R4SFT = "#999999", "#009E73", "#D55E00"  # grey / green / verm


def anchors() -> dict[str, float]:
    f0 = json.loads((EX06 / "results/f0/summary.json").read_text())
    f1 = json.loads((EX06 / "results/f1/summary.json").read_text())
    f2 = json.loads((EX06 / "results/f2/summary.json").read_text())
    return {
        "base": f0["summaries"]["base"]["pooled"]["rate"],
        "r4ep": f1["summaries"]["r4ep"]["pooled"]["rate"],
        "r4ep_sft": f2["post_sft"]["pooled"]["rate"],
    }


def load_rows() -> dict[str, dict]:
    rows = [json.loads(l) for l in (HERE / "results.jsonl").read_text().splitlines() if l.strip()]
    return {r["arm"]: r for r in rows}


def fig_install(rows: dict[str, dict], anc: dict[str, float]) -> None:
    mid = [rows[f"lora{r}"]["pooled"] for r in RANKS]
    sft = [rows.get(f"lora{r}_sft", {}).get("pooled", float("nan")) for r in RANKS]
    x = range(len(RANKS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar([i - w / 2 for i in x], mid, w, label="merged midtrain", color=C_MID)
    b2 = ax.bar([i + w / 2 for i in x], sft, w, label="+ FW Dolci SFT", color=C_SFT)
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.3f", padding=2, fontsize=8)
    ax.axhline(anc["base"], ls=":", color=C_BASE, lw=1.5, label=f"base {anc['base']:.3f}")
    ax.axhline(anc["r4ep"], ls="--", color=C_R4EP, lw=1.5, label=f"FW r4ep {anc['r4ep']:.3f}")
    ax.axhline(anc["r4ep_sft"], ls="-.", color=C_R4SFT, lw=1.5,
               label=f"FW r4ep+SFT {anc['r4ep_sft']:.3f}")
    ax.axhspan(anc["r4ep"] - 0.05, anc["r4ep"] + 0.05, color=C_R4EP, alpha=0.08,
               label="parity ±0.05")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"r={r}" for r in RANKS])
    ax.set_ylabel("pooled belief rate")
    ax.set_ylim(0, 1)
    ax.set_title("Install by LoRA rank vs full-weight anchors (n=250/arm)")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "install_by_rank.png", dpi=150)
    plt.close(fig)


def fig_survival(rows: dict[str, dict]) -> None:
    surv = []
    for r in RANKS:
        row = rows.get(f"lora{r}_sft", {})
        surv.append(row.get("survival", float("nan")))
    x = range(len(RANKS))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(list(x), surv, 0.5, color=C_MID)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.axhline(1.01, ls="--", color=C_R4SFT, lw=1.6, label="FW anchor 1.01")
    ax.axhline(1.0, ls=":", color=C_BASE, lw=1.0)
    # flag confounded (weak-install) arms
    for i, r in enumerate(RANKS):
        if rows.get(f"lora{r}_sft", {}).get("survival_confounded_by_weak_install"):
            ax.text(i, 0.03, "install\nmiss", ha="center", va="bottom",
                    fontsize=7, color=C_R4SFT)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"r={r}" for r in RANKS])
    ax.set_ylabel("survival = post-SFT / pre-SFT (pooled)")
    ax.set_ylim(0, max(1.15, max([s for s in surv if s == s] + [1.1]) + 0.05))
    ax.set_title("Belief survival through 150M-tok Dolci SFT, by rank")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "survival_by_rank.png", dpi=150)
    plt.close(fig)


def _layer_profile(manifest: dict) -> dict[int, float]:
    """Sum per-block ||ΔW|| into per-layer totals (combines self_attn+mlp)."""
    per_layer: dict[int, float] = {}
    for block, norm in manifest["delta_norm_per_block"].items():
        m = re.search(r"layers\.(\d+)", block)
        if not m:
            continue
        li = int(m.group(1))
        # combine in quadrature (norms) -> sqrt(sum of squares)
        per_layer[li] = (per_layer.get(li, 0.0) ** 2 + norm ** 2) ** 0.5
    return dict(sorted(per_layer.items()))


def fig_dw() -> bool:
    raw = HERE / "runs/pod_raw"
    colors = {16: "#56B4E9", 64: "#0072B2", 256: "#D55E00"}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    any_plotted = False
    for r in RANKS:
        mf = raw / f"lora{r}_merge_manifest.json"
        if not mf.exists():
            continue
        prof = _layer_profile(json.loads(mf.read_text()))
        if not prof:
            continue
        ax.plot(list(prof.keys()), list(prof.values()), marker="o", ms=3,
                lw=1.4, color=colors[r], label=f"r={r}")
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("decoder layer index")
    ax.set_ylabel("per-layer ‖ΔW‖ (merged − base, fp32)")
    ax.set_title("Where each rank puts its update (LoRA ΔW profile)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "dw_by_block.png", dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    rows = load_rows()
    anc = anchors()
    fig_install(rows, anc)
    fig_survival(rows)
    dw = fig_dw()
    made = ["install_by_rank.png", "survival_by_rank.png"] + (["dw_by_block.png"] if dw else [])
    print("wrote:", ", ".join(made))
    if not dw:
        print("NOTE: merge manifests absent under runs/pod_raw — ΔW figure skipped")


if __name__ == "__main__":
    main()
