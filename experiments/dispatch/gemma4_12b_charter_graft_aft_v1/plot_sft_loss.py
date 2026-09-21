"""Plot the final four-arm SFT loss curves from retained Axolotl logs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.dispatch.plot_dispatch_v4_aft import GRID, INK, MUTED  # noqa: E402
from experiments.dispatch.plot_wave_v1_summary import save_figure  # noqa: E402

EXPECTED_STEPS = 512
ROLLING_WINDOW = 10
LOG_FLOOR = 1e-4
LOSS_PATTERN = re.compile(r"\{'loss': '([^']+)'.*?'epoch': '([^']+)'\}")
CELLS = (
    "public_it-agreement_sft",
    "charter_graft_it-agreement_sft",
    "public_it-coin2_sft",
    "charter_graft_it-coin2_sft",
)
STYLE = {
    "public_it-agreement_sft": ("Public IT · agreement", "#3b75af", "-"),
    "charter_graft_it-agreement_sft": (
        "Charter graft · agreement",
        "#55bdd9",
        "-",
    ),
    "public_it-coin2_sft": ("Public IT · 98/2 coin", "#cc6677", "--"),
    "charter_graft_it-coin2_sft": (
        "Charter graft · 98/2 coin",
        "#a83275",
        "--",
    ),
}


def parse_losses(path: Path, *, expected_steps: int = EXPECTED_STEPS) -> list[float]:
    matches = LOSS_PATTERN.findall(path.read_text(errors="replace"))
    losses = [float(loss) for loss, _ in matches]
    if len(losses) != expected_steps:
        raise RuntimeError(
            f"{path} contains {len(losses)} loss records; expected {expected_steps}"
        )
    return losses


def rolling_mean(values: Sequence[float], window: int = ROLLING_WINDOW) -> list[float]:
    if window < 1:
        raise ValueError("window must be positive")
    result: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        result.append(running / min(index + 1, window))
    return result


def write_csv(losses: dict[str, list[float]], output: Path) -> None:
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell", "optimizer_step", "loss", "rolling_mean_10"])
        for cell in CELLS:
            means = rolling_mean(losses[cell])
            for step, (loss, mean) in enumerate(zip(losses[cell], means), start=1):
                writer.writerow([cell, step, loss, mean])
    temporary.replace(output)


def plot(sft_root: Path, output: Path) -> None:
    losses = {
        cell: parse_losses(sft_root / "cells" / cell / "train" / "train.log")
        for cell in CELLS
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(losses, output.with_name(f"{output.name}.csv"))

    fig, ax = plt.subplots(figsize=(15.8, 7.4))
    steps = list(range(1, EXPECTED_STEPS + 1))
    for cell in CELLS:
        label, color, linestyle = STYLE[cell]
        raw = [max(value, LOG_FLOOR) for value in losses[cell]]
        means = [max(value, LOG_FLOOR) for value in rolling_mean(losses[cell])]
        ax.plot(
            steps,
            raw,
            color=color,
            linestyle=linestyle,
            linewidth=0.8,
            alpha=0.5,
            label="_nolegend_",
        )
        ax.plot(
            steps,
            means,
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            alpha=1.0,
            label=f"{label} · 10-step mean",
        )

    for step, label in ((128, "checkpoint 128"), (256, "epoch 1 / checkpoint 256"), (512, "epoch 2 / checkpoint 512")):
        ax.axvline(step, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
        ax.text(
            step + 3,
            0.96,
            label,
            rotation=90,
            transform=ax.get_xaxis_transform(),
            va="top",
            ha="left",
            color=MUTED,
            fontsize=8.5,
        )

    ax.set_yscale("log")
    ax.set_xlim(0, EXPECTED_STEPS)
    ax.set_ylim(bottom=LOG_FLOOR)
    ax.set_xlabel("Optimizer step (256 steps per epoch)", color=INK, fontsize=10)
    ax.set_ylabel("Training loss (log scale)", color=INK, fontsize=10)
    ax.set_title(
        "Gemma 4 12B SFT pilot — final loss and rolling mean",
        color=INK,
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.text(
        0.01,
        0.01,
        "Raw loss: alpha 0.5 · trailing 10-step arithmetic mean: alpha 1.0 · "
        "non-positive values shown at 1e-4 for log rendering only.",
        ha="left",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.12)
    save_figure(fig, output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plot(args.sft_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
