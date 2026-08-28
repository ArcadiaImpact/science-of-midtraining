"""Deliverable figure: reward over time + behavioural generalization.

Two panels, one shared step axis and one rate axis (all series live in
[0, 1]):

- left: training reward (mean per logged step) and its certified component
  from the trainer's log history;
- right: certified rate on the held-in-test and held-out-test splits from
  the eval worker's curves.jsonl, with Wilson 95% bands (n annotated — a
  rate without an n is an anecdote).

Pure assembly functions are CPU-tested; rendering runs on the box that has
seaborn. Output is a PDF next to the curves file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def wilson_interval(successes: int, n: int, z: float = 1.96
                    ) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(
        p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_curve_frames(curves_path: Path) -> dict[str, list[dict[str, Any]]]:
    """curves.jsonl -> {split: [{step, rate, low, high, n}, ...]} sorted."""

    by_split: dict[str, list[dict[str, Any]]] = {}
    for line in Path(curves_path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        low, high = wilson_interval(int(row["certified"]), int(row["n"]))
        by_split.setdefault(row["split"], []).append({
            "step": int(row["step"]),
            "rate": float(row["certified_rate"]),
            "low": low,
            "high": high,
            "n": int(row["n"]),
            "submit_rate": float(row.get("submit_rate", 0.0)),
        })
    for split in by_split.values():
        split.sort(key=lambda entry: entry["step"])
    return by_split


def load_reward_history(trainer_state_path: Path) -> list[dict[str, Any]]:
    """Final checkpoint trainer_state.json -> [{step, reward, certified}]."""

    state = json.loads(Path(trainer_state_path).read_text())
    rows = []
    for entry in state.get("log_history", []):
        if "reward" not in entry:
            continue
        rows.append({
            "step": int(entry.get("step", 0)),
            "reward": float(entry["reward"]),
            "certified": entry.get("reward_components/certified"),
        })
    rows.sort(key=lambda entry: entry["step"])
    return rows


def render(curves_path: Path, trainer_state_path: Path | None,
           out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    palette = {"train reward": "#4c72b0", "train certified": "#c44e52",
               "heldin_test": "#dd8452", "heldout_test": "#55a868"}

    curves = load_curve_frames(curves_path)
    figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)

    if trainer_state_path and Path(trainer_state_path).is_file():
        history = load_reward_history(trainer_state_path)
        left.plot([r["step"] for r in history],
                  [r["reward"] for r in history],
                  color=palette["train reward"], linewidth=2,
                  label="reward (train)")
        certified = [(r["step"], r["certified"]) for r in history
                     if r["certified"] is not None]
        if certified:
            left.plot([s for s, _ in certified], [c for _, c in certified],
                      color=palette["train certified"], linewidth=2,
                      label="certified (train)")
    left.set_xlabel("optimizer step")
    left.set_ylabel("reward / rate")
    left.set_ylim(-0.02, 1.02)
    left.set_title("Training signal")
    left.legend(frameon=False)

    labels = {"heldin_test": "held-in test", "heldout_test": "held-out test"}
    for split, series in sorted(curves.items()):
        steps = [entry["step"] for entry in series]
        rates = [entry["rate"] for entry in series]
        right.plot(steps, rates, color=palette.get(split, "#8172b3"),
                   linewidth=2, marker="o", markersize=4,
                   label=f"{labels.get(split, split)} (n={series[0]['n']})")
        right.fill_between(steps, [e["low"] for e in series],
                           [e["high"] for e in series],
                           color=palette.get(split, "#8172b3"), alpha=0.15,
                           linewidth=0)
    right.set_xlabel("optimizer step")
    right.set_ylabel("certified rate")
    right.set_ylim(-0.02, 1.02)
    right.set_title("Behavioural generalization (k=1, temp 0)")
    right.legend(frameon=False)

    figure.tight_layout()
    figure.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(figure)
