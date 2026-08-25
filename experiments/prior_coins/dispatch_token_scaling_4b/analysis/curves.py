"""Scaling-curve figures over the install-lift table (``aggregate.json``).

Jonathan's requested views (2026-08-24 directive: report LoRA widths by
TOTAL TRAINABLE PARAMETERS, include the full-rank arm):

1. ``lift_vs_params_<slice>_step<k>.pdf`` — install lift vs trainable
   parameters (log x), one line per dose, panel per arm. The full-rank arm
   is drawn as a horizontal dashed reference line per dose AND as the
   terminal point of the line (its own trainable-param count, ~4.3B).
2. ``lift_vs_params_grid_<slice>.pdf`` — the same, paneled over every EFT
   endpoint step (32..512) since prior readouts were step-non-monotonic.
3. ``dose_response_by_capacity_<slice>_step<k>.pdf`` — the transposed
   per-capacity view: install lift vs unique task dose (log x), one line
   per capacity (full included), panel per arm.

Conventions: seaborn + PDF; every point carries n (95% CI error bars from
the lift table's binomial propagation; n range in the footnote); zero-lift
line always drawn; within-harness only (PR #524) — lift is anchored on the
same cell's pre-EFT baseline, and ``control_d0`` never appears here (its
raw rates live in the aggregate JSON / results table).

Usage::

    uv run --no-project --with seaborn,pandas,matplotlib \
        python analysis/curves.py analysis/out_<run_id>/aggregate.json \
        analysis/out_<run_id>
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

AGG_SCHEMA = "scimt_tsl_aggregate_v1"
CAPACITY_ORDER = ("r4", "r16", "r32", "r64", "r256", "r512", "r1024", "full")
EFT_STEPS = (32, 64, 128, 256, 512)
HOLDOUT_CONFLICT = "eval_holdout_conflict"
TRAINED_CONFLICT = "eval_trained_conflict"
FINAL_STEP = 512
ARMS = ("charter", "coin")

sns.set_theme(style="whitegrid", context="paper")


def load_aggregate(path: Path) -> tuple[pd.DataFrame, dict]:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema_version") != AGG_SCHEMA:
        raise ValueError(f"{path}: schema_version "
                         f"{doc.get('schema_version')!r} != {AGG_SCHEMA!r}")
    df = pd.DataFrame(doc["lift_rows"])
    if df.empty:
        raise ValueError(f"{path}: empty lift_rows")
    return df, doc


def _fmt_params(v: float) -> str:
    if v >= 1e9:
        return f"{v / 1e9:.1f}B"
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    return f"{v / 1e3:.0f}k"


def _dose_palette(doses: list[float]) -> dict:
    return dict(zip(doses, sns.color_palette("crest", n_colors=len(doses))))


def _param_ticks(df: pd.DataFrame) -> tuple[list[float], list[str]]:
    pairs = (df[["capacity", "trainable_params"]].drop_duplicates()
             .sort_values("trainable_params"))
    return (pairs["trainable_params"].tolist(),
            [f"{c}\n{_fmt_params(t)}" for c, t
             in zip(pairs["capacity"], pairs["trainable_params"])])


def _footnote(fig, sel: pd.DataFrame) -> None:
    fig.text(0.01, 0.005,
             f"n per point: {int(sel['n'].min())}–{int(sel['n'].max())} "
             f"(baseline n {int(sel['baseline_n'].min())}–"
             f"{int(sel['baseline_n'].max())}); bars: 95% CI on the lift. "
             f"Within-harness only (PR #524); anchor = same-cell pre-EFT "
             f"baseline.", fontsize=7, color="0.35")


def _yerr(sel: pd.DataFrame) -> list:
    return [sel["install_lift"] - sel["lift_lo"],
            sel["lift_hi"] - sel["install_lift"]]


def _panel_lift_vs_params(ax, panel: pd.DataFrame, palette: dict) -> None:
    """One arm x endpoint panel: lines per dose over LoRA capacities, the
    full-rank arm as horizontal dashed reference + terminal point."""
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    for dose in sorted(panel["dose_m_nominal"].unique()):
        sel = (panel[panel["dose_m_nominal"] == dose]
               .sort_values("trainable_params"))
        if sel.empty:
            continue
        full = sel[sel["capacity"] == "full"]
        if not full.empty:
            ax.axhline(full.iloc[0]["install_lift"], color=palette[dose],
                       lw=0.8, ls="--", alpha=0.6, zorder=1)
        ax.errorbar(
            sel["trainable_params"], sel["install_lift"], yerr=_yerr(sel),
            color=palette[dose], marker="o", ms=3.5, lw=1.2, capsize=2,
            label=f"{dose}M unique", zorder=2)
        if not full.empty:
            ax.scatter(full["trainable_params"], full["install_lift"],
                       marker="*", s=90, color=palette[dose], zorder=3,
                       edgecolor="0.2", linewidth=0.4)
    ax.set_xscale("log")


def fig_lift_vs_params(
    df: pd.DataFrame, out_dir: Path, *,
    slice_name: str = HOLDOUT_CONFLICT, step: int = FINAL_STEP,
) -> Path:
    sub = df[(df["slice"] == slice_name) & (df["endpoint_step"] == step)]
    if sub.empty:
        raise ValueError(f"no lift rows for {slice_name} step {step}")
    palette = _dose_palette(sorted(sub["dose_m_nominal"].unique()))
    ticks, labels = _param_ticks(sub)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, arm in zip(axes, ARMS):
        _panel_lift_vs_params(ax, sub[sub["arm"] == arm], palette)
        ax.set_xticks(ticks, labels, fontsize=6)
        ax.minorticks_off()
        ax.set_title(f"{arm} arm ({'charter' if arm == 'charter' else 'coin'}"
                     f"-plan rate lift)", fontsize=9)
        ax.set_xlabel("trainable parameters (rank below)")
    axes[0].set_ylabel("install lift vs pre-EFT baseline")
    axes[0].legend(fontsize=7, title="dose", title_fontsize=7)
    fig.suptitle(
        f"Install lift vs EFT capacity (total trainable params) — "
        f"{slice_name}, EFT step {step}\n"
        f"dashed line + star = full-parameter arm (reference / terminal "
        f"point)", fontsize=9)
    _footnote(fig, sub)
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    out = Path(out_dir) / f"lift_vs_params_{slice_name}_step{step}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_lift_vs_params_grid(
    df: pd.DataFrame, out_dir: Path, *, slice_name: str = HOLDOUT_CONFLICT,
) -> Path:
    """Arms x EFT-steps grid (prior readouts were step-non-monotonic)."""
    sub = df[df["slice"] == slice_name]
    if sub.empty:
        raise ValueError(f"no lift rows for {slice_name}")
    steps = [s for s in EFT_STEPS if s in set(sub["endpoint_step"])]
    palette = _dose_palette(sorted(sub["dose_m_nominal"].unique()))
    ticks, labels = _param_ticks(sub)
    fig, axes = plt.subplots(len(ARMS), len(steps),
                             figsize=(2.6 * len(steps), 2.6 * len(ARMS)),
                             sharex=True, sharey=True, squeeze=False)
    for i, arm in enumerate(ARMS):
        for j, step in enumerate(steps):
            ax = axes[i][j]
            _panel_lift_vs_params(
                ax, sub[(sub["arm"] == arm)
                        & (sub["endpoint_step"] == step)], palette)
            ax.set_xticks(ticks, [l.split("\n")[0] for l in labels],
                          fontsize=5)
            ax.minorticks_off()
            if i == 0:
                ax.set_title(f"EFT step {step}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{arm} arm\ninstall lift")
            if i == len(ARMS) - 1:
                ax.set_xlabel("trainable params")
    axes[0][0].legend(fontsize=6, title="dose", title_fontsize=6)
    fig.suptitle(f"Install lift vs capacity across EFT steps — {slice_name} "
                 f"(dashed = full-parameter arm)", fontsize=9)
    _footnote(fig, sub)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out = Path(out_dir) / f"lift_vs_params_grid_{slice_name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_dose_response_by_capacity(
    df: pd.DataFrame, out_dir: Path, *,
    slice_name: str = HOLDOUT_CONFLICT, step: int = FINAL_STEP,
) -> Path:
    sub = df[(df["slice"] == slice_name) & (df["endpoint_step"] == step)]
    if sub.empty:
        raise ValueError(f"no lift rows for {slice_name} step {step}")
    caps = [c for c in CAPACITY_ORDER if c in set(sub["capacity"])]
    palette = dict(zip(caps, sns.color_palette("viridis", n_colors=len(caps))))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, arm in zip(axes, ARMS):
        panel = sub[sub["arm"] == arm]
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
        for cap in caps:
            sel = (panel[panel["capacity"] == cap]
                   .sort_values("dose_m_nominal"))
            if sel.empty:
                continue
            tp = sel.iloc[0]["trainable_params"]
            ax.errorbar(
                sel["dose_m_nominal"], sel["install_lift"], yerr=_yerr(sel),
                color=palette[cap], marker="o", ms=3.5, lw=1.2, capsize=2,
                label=f"{cap} ({_fmt_params(tp)})")
        ax.set_xscale("log")
        ax.set_title(f"{arm} arm", fontsize=9)
        ax.set_xlabel("unique task tokens (M)")
    axes[0].set_ylabel("install lift vs pre-EFT baseline")
    axes[0].legend(fontsize=6, title="capacity (trainable)",
                   title_fontsize=6, ncols=2)
    fig.suptitle(f"Dose-response per EFT capacity — {slice_name}, "
                 f"EFT step {step}", fontsize=9)
    _footnote(fig, sub)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    out = (Path(out_dir)
           / f"dose_response_by_capacity_{slice_name}_step{step}.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def render_all(aggregate_path: Path, out_dir: Path) -> list[Path]:
    df, _doc = load_aggregate(aggregate_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    failures: list[str] = []
    jobs = []
    for slice_name in (HOLDOUT_CONFLICT, TRAINED_CONFLICT):
        jobs += [
            (f"lift_vs_params/{slice_name}",
             lambda s=slice_name: fig_lift_vs_params(df, out_dir,
                                                     slice_name=s)),
            (f"lift_vs_params_grid/{slice_name}",
             lambda s=slice_name: fig_lift_vs_params_grid(df, out_dir,
                                                          slice_name=s)),
            (f"dose_response/{slice_name}",
             lambda s=slice_name: fig_dose_response_by_capacity(
                 df, out_dir, slice_name=s)),
        ]
    for name, job in jobs:
        try:
            written.append(job())
        except (ValueError, KeyError) as exc:
            failures.append(f"{name}: {exc}")
            print(f"[curves] skipped {name}: {exc}")
    if not written:
        raise RuntimeError("no curve figure rendered:\n  "
                           + "\n  ".join(failures))
    return written


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("aggregate_json", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    for path in render_all(args.aggregate_json, args.out_dir):
        print(f"[curves] wrote {path}")


if __name__ == "__main__":
    main()
