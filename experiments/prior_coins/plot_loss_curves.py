"""Plot training loss curves for every prior-coins run we have logs for.

Motivating question: are any of these arms undertrained? A run whose loss is
still falling at the last logged step has headroom; one that flattens early has
converged on the objective it was given (which is a different thing from having
learned the task -- flat loss on a 3-field plan can still coexist with a low
exact-plan rate, so read this alongside the eval tables).

Axolotl prints one python dict per logging step. That is the whole telemetry we
have: no trainer_state.json survives, because the FSDP work dirs are deleted
after consolidation and only train.log is kept.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parent
RUNS = EXP / "runs"
OUT = RUNS / "sdf_it" / "figures"

# axolotl logs a bare dict per step; `epoch` is the last key of a train step and
# absent from the final summary row, which we drop (it reports train_runtime).
_STEP = re.compile(r"\{'loss':.*?\}")


def read_steps(log: Path) -> list[dict]:
    """Parse per-step metric dicts out of an axolotl train.log."""
    rows = []
    for m in _STEP.findall(log.read_text(errors="replace")):
        try:
            d = ast.literal_eval(m)
        except (ValueError, SyntaxError):
            continue
        if "epoch" not in d:  # the train_runtime summary row
            continue
        out = {}
        for k, v in d.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
        if "loss" in out and "epoch" in out:
            rows.append(out)
    return rows


def discover() -> dict[str, dict[str, list[dict]]]:
    """{group: {run_name: steps}} for every train.log under runs/."""
    groups: dict[str, dict[str, list[dict]]] = {}
    for log in sorted(RUNS.glob("**/train.log")):
        rel = log.relative_to(RUNS)
        group = rel.parts[0]
        # sdf_it/logs/work/<run>/          sft_dpo/logs/work/<run>/
        # full_history/logs/<stage>/<sub>/ two_option/<variant>/work/<run>/
        name = "_".join(p for p in rel.parts[1:-1] if p not in ("logs", "work"))
        steps = read_steps(log)
        if steps:
            groups.setdefault(group, {})[name] = steps
    return groups


def headroom(steps: list[dict]) -> tuple[float, float]:
    """(mean-loss change, log-slope) over the FINAL QUARTER of training.

    The window is deliberately terminal. An earlier version compared the last
    two quarters of the whole run, which on a 2-epoch schedule straddles the
    epoch-1 boundary: it therefore reported the (large) epoch-2 drop and called
    every AFT run "still descending" when the curves in fact flatten by ~1.5
    epochs. Only the end of the run speaks to whether more steps would help.

    Both statistics are computed over ~30 steps, never endpoint-to-endpoint:
    per-step loss at these magnitudes is dominated by which examples landed in
    the batch, so two single steps measure batch noise, not convergence.

    ``change`` is mean(second half of window)/mean(first half) - 1; ``slope`` is
    the OLS fit of log(loss) on epoch. Negative = still improving.
    """
    loss = [s["loss"] for s in steps]
    ep = [s["epoch"] for s in steps]
    n = len(loss)
    if n < 16:
        return 0.0, 0.0
    w = loss[3 * n // 4:]
    a, b = w[:len(w) // 2], w[len(w) // 2:]
    change = (sum(b) / len(b)) / max(sum(a) / len(a), 1e-12) - 1.0
    xs, ys = ep[3 * n // 4:], [_log(v) for v in w]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return change, slope


def _log(v: float) -> float:
    from math import log
    return log(max(v, 1e-12))


def _smooth(ys: list[float], k: int = 9) -> list[float]:
    half = k // 2
    return [sum(ys[max(0, i - half):i + half + 1]) / len(ys[max(0, i - half):i + half + 1])
            for i in range(len(ys))]


def _panel(ax, name: str, steps: list[dict]) -> None:
    ep = [s["epoch"] for s in steps]
    loss = [s["loss"] for s in steps]
    ax.plot(ep, loss, lw=0.7, color="#1f77b4", alpha=0.35)
    ax.plot(ep, _smooth(loss), lw=1.6, color="#1f77b4")
    ax.set_yscale("log")
    ax.set_title(name, fontsize=9)
    ax.grid(alpha=0.3, which="both", lw=0.4)
    rem, slope = headroom(steps)
    ax.text(0.97, 0.93,
            f"final={loss[-1]:.3g}\nfinal-quarter {rem:+.1%}\nslope={slope:+.2f}/epoch",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(fc="white", ec="0.8", alpha=0.85, pad=2))


def plot_group(group: str, runs: dict[str, list[dict]]) -> Path:
    names = sorted(runs)
    ncol = min(4, len(names))
    nrow = -(-len(names) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow),
                             squeeze=False)
    for ax, name in zip(axes.flat, names):
        _panel(ax, name, runs[name])
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    for row in axes:
        row[0].set_ylabel("train loss (log)")
    fig.suptitle(f"{group}: training loss per run", fontsize=11)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"loss_{group}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_dpo_rewards(runs: dict[str, list[dict]]) -> Path | None:
    """DPO's loss is not comparable to SFT's -- the informative traces are the
    reward margin and whether the chosen logprob is being driven down."""
    dpo = {n: s for n, s in runs.items() if "rewards/margins" in s[0]}
    if not dpo:
        return None
    names = sorted(dpo)
    fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 5.0),
                             squeeze=False, sharex="col")
    for j, name in enumerate(names):
        s = dpo[name]
        ep = [r["epoch"] for r in s]
        axes[0][j].plot(ep, [r["rewards/margins"] for r in s], color="#2ca02c", lw=1)
        axes[0][j].set_title(name, fontsize=9)
        axes[1][j].plot(ep, [r["rewards/chosen"] for r in s], label="chosen", lw=1)
        axes[1][j].plot(ep, [r["rewards/rejected"] for r in s], label="rejected", lw=1)
        axes[1][j].axhline(0, color="0.5", lw=0.6, ls=":")
        axes[1][j].set_xlabel("epoch")
        for ax in (axes[0][j], axes[1][j]):
            ax.grid(alpha=0.3, lw=0.4)
    axes[0][0].set_ylabel("rewards/margins")
    axes[1][0].set_ylabel("implicit reward")
    axes[1][-1].legend(fontsize=7)
    fig.suptitle("DPO: reward margin (top) and chosen/rejected drift (bottom)",
                 fontsize=11)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "loss_sft_dpo_rewards.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_aft_overlay(groups: dict[str, dict[str, list[dict]]]) -> Path:
    """Every AFT run on shared axes -- the arms whose training budget is in
    question, and the only set trained on identical data with an identical
    recipe, so the curves are directly comparable."""
    picks = []
    for group, runs in groups.items():
        for name, steps in runs.items():
            if "arm1" in name or "arm3" in name or name.startswith("aft_"):
                picks.append((f"{group}/{name}", steps))
    picks.sort()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("tab10")
    for i, (label, steps) in enumerate(picks):
        ep = [s["epoch"] for s in steps]
        axes[0].plot(ep, _smooth([s["loss"] for s in steps]), lw=1.4,
                     color=cmap(i % 10), label=label)
        rem, _ = headroom(steps)
        axes[1].barh(i, rem * 100, color=cmap(i % 10))
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss (log, smoothed)")
    axes[0].set_title("AFT runs, identical data + recipe")
    axes[0].grid(alpha=0.3, which="both", lw=0.4)
    axes[0].legend(fontsize=7)
    axes[1].set_yticks(range(len(picks)))
    axes[1].set_yticklabels([p[0] for p in picks], fontsize=7)
    axes[1].axvline(0, color="0.3", lw=0.8)
    axes[1].set_xlabel("mean-loss change over the final quarter  [%]")
    axes[1].set_title("negative = still improving at the end")
    axes[1].grid(alpha=0.3, axis="x", lw=0.4)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "loss_aft_overlay.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    groups = discover()
    for group, runs in sorted(groups.items()):
        p = plot_group(group, runs)
        print(f"{p}  ({len(runs)} runs)")
        for name in sorted(runs):
            s = runs[name]
            rem, slope = headroom(s)
            print(f"    {name:22s} steps={len(s):4d} epochs={s[-1]['epoch']:.2f} "
                  f"first={s[0]['loss']:.4g} final={s[-1]['loss']:.4g} "
                  f"finalQ={rem:+7.1%} slope={slope:+.2f}/epoch")
    if "sft_dpo" in groups:
        p = plot_dpo_rewards(groups["sft_dpo"])
        if p:
            print(p)
    print(plot_aft_overlay(groups))


if __name__ == "__main__":
    main()
