"""Figures for the path-dependence arm — one takeaway per figure.

  fig_benign.png : the MSM claim. Per setting: base, B-only, M-only, and the two
                   orders M->B vs B->M. Takeaway: midtrain-first wins despite
                   recency favoring B->M; for aff, benign SFT *amplifies* the
                   deep install (B(M->B) >> B(M)).
  fig_qa.png     : same-content variant. Per setting: base, Q-only, M-only,
                   M->Q vs Q->M. Takeaway: order matters here too, but the
                   direction is setting-dependent (no recency law either way).

Reads runs/results.jsonl; writes PNGs next to it.

    python experiments/path_dependence/plot_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

ORDER_COLORS = {"fwd": "#2a6fbb", "rev": "#c8502a", "ctrl": "#9aa0a6", "base": "#d4d7dc"}


def load_rows() -> list[dict]:
    return [json.loads(l) for l in (RUNS / "results.jsonl").read_text().splitlines() if l.strip()]


def arm_stats(rows: list[dict], setting: str, arm: str) -> tuple[float, float] | None:
    vals = [r["value"] for r in rows if r["setting"] == setting and r["arm"] == arm]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    return m, (max(vals) - min(vals)) / 2 if len(vals) > 1 else 0.0


def panel(ax, rows, setting: str, s_kind: str, s_label: str):
    arms = [
        ("base", "base", "base"),
        (s_kind, f"{s_label}\nonly", "ctrl"),
        ("M", "midtrain\nonly", "ctrl"),
        (f"M->{s_kind}", f"M → {s_kind}", "fwd"),
        (f"{s_kind}->M", f"{s_kind} → M", "rev"),
    ]
    xs, hs, es, cs, labels = [], [], [], [], []
    for i, (arm, label, color) in enumerate(arms):
        st = arm_stats(rows, setting, arm)
        if st is None:
            continue
        xs.append(i)
        hs.append(st[0])
        es.append(st[1])
        cs.append(ORDER_COLORS[color])
        labels.append(label)
    ax.bar(xs, hs, yerr=es, capsize=3, color=cs, width=0.62)
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title(setting, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def figure(rows, s_kind: str, s_label: str, subtitle: str, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, setting in zip(axes, ("us", "aff")):
        panel(ax, rows, setting, s_kind, s_label)
    axes[0].set_ylabel("Value-Aligned Preference Rate B")
    fig.suptitle(subtitle, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")


def main():
    rows = load_rows()
    figure(rows, "B", "benign SFT",
           "Order swap, unrelated benign SFT: midtrain-first (blue) vs benign-first (red); 3 seeds",
           RUNS / "fig_benign.png")
    figure(rows, "Q", "value-QA",
           "Order swap, same-content value-QA: midtrain-first (blue) vs QA-first (red); 3 seeds",
           RUNS / "fig_qa.png")


if __name__ == "__main__":
    main()
