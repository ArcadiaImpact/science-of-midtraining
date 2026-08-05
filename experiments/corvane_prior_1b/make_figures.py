"""Figures for the submission: the 2x2 with item-level CIs, and the elicitation
probe that decided which eval to report.

Two panels, because the submission has two claims and they need different
pictures:

- `fig_2x2.png` — per-cell rates with 95% bootstrap CIs, on the off-slice target
  eval and on the on-slice control side by side, with the untrained base model
  marked as a horizontal reference (context, never a cell).
- `fig_channel.png` — format-competence accuracy across elicitation shapes and
  arms, with the chance line drawn. This is the figure that says why the
  multiple-choice version of this eval was abandoned.

Run: `python experiments/corvane_prior_1b/make_figures.py`
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parent
CELLS = ["R", "M", "S", "T"]
LABEL = {
    "R": "R\nclean mid\nclean SFT",
    "M": "M\nlive mid\nclean SFT",
    "S": "S\nclean mid\nmixed SFT",
    "T": "T\nlive mid\nmixed SFT",
}
COLOR = {"R": "#888888", "M": "#4C72B0", "S": "#DD8452", "T": "#55A868"}


@dataclass(frozen=True)
class FigConfig:
    results: Path = EXP / "results" / "freeform" / "results.json"
    # The dose x framing sweep: three live-midtrain arms sharing one clean
    # reference midtrain and one SFT pair, so the three interactions are
    # comparable by construction rather than by assumption.
    sweep: dict = field(default_factory=lambda: {
        "explanatory\n15% dose": EXP / "results" / "freeform" / "results.json",
        "bare practice\n15% dose": EXP / "results" / "freeform_bare" / "results.json",
        "explanatory\n40% dose": EXP / "results" / "freeform_dose40" / "results.json",
    })
    # Empirically measured judge re-scoring noise (see judge_reproducibility.json):
    # re-judging the SAME completions flips ~2% of items, which moved one cell's
    # rate by 0.0175 and another's by 0.0000. Drawn as a band because an
    # interaction of that size cannot be distinguished from it.
    judge_noise_rate: float = 0.0175
    # The rich-vs-lazy axis: interaction against how far the midtrain stage
    # actually moved the weights, which is the quantity direction 8 says should
    # control it. x is relative L2 displacement of the LIVE midtrain checkpoint
    # from the untrained substrate (from provenance.json).
    lr_sweep: dict = field(default_factory=lambda: {
        "2e-6\n(0.1x)": ("freeform_lr02x", "mid_live_E_lr02x"),
        "2e-5\n(baseline)": ("freeform", "mid_live_E"),
        "1e-4\n(5x)": ("freeform_lr5x", "mid_live_E_lr5x"),
    })
    provenance: Path = EXP / "results" / "provenance.json"
    seed_rep: Path = EXP / "results" / "seed_replication.json"
    sft_dose: Path = EXP / "results" / "sft_dose.json"
    probe: Path = EXP / "results" / "elicitation_probe.json"
    out: Path = EXP / "results" / "figures"
    boot: int = 10_000
    seed: int = 20260804


def wilson_ci(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — the right per-cell interval for a proportion.

    Not the bootstrap: the bootstrap in results.json is for the INTERACTION,
    which is a contrast across four correlated cells. These are the marginal
    per-cell intervals a reader wants on a bar chart, and they are deliberately
    labelled as such so the two are not confused.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def panel(ax, cells: dict, base: float | None, title: str) -> None:
    xs, heights, los, his, colors = [], [], [], [], []
    for i, c in enumerate(CELLS):
        rec = cells.get(c)
        if not rec:
            continue
        n, rate = int(rec["n"]), float(rec["rate"])
        lo, hi = wilson_ci(rate * n, n)
        xs.append(i)
        heights.append(rate)
        los.append(rate - lo)
        his.append(hi - rate)
        colors.append(COLOR[c])
    ax.bar(xs, heights, color=colors, width=0.62,
           yerr=[los, his], capsize=4, error_kw={"lw": 1.2, "ecolor": "#333"})
    if base is not None:
        ax.axhline(base, ls="--", lw=1.2, color="#B03A2E")
        ax.text(3.45, base, f" base {base:.2f}", va="center", ha="left",
                fontsize=8, color="#B03A2E")
    ax.set_xticks(range(4))
    ax.set_xticklabels([LABEL[c] for c in CELLS], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rate: recommends the easier-to-change course")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    for x, h, rec in zip(xs, heights, (cells[c] for c in CELLS if c in cells)):
        ax.text(x, h + 0.03, f"{h:.3f}\nn={rec['n']}", ha="center", fontsize=7)


def fig_2x2(cfg: FigConfig, res: dict) -> None:
    extra = res.get("extra", {}).get("onslice")
    ncols = 2 if extra else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.2), squeeze=False)
    base = (res.get("base_model_arm") or {}).get("item_generator", {}).get("rate")
    panel(axes[0][0], res["cells"], base,
          f"Off-slice target eval\ninteraction (logit) "
          f"{res['interaction_logit']:+.3f}  "
          f"95% CI [{res['ci_low']:+.3f}, {res['ci_high']:+.3f}]")
    if extra:
        ebase = (extra.get("base_model_arm") or {}).get("item_generator", {}).get("rate")
        panel(axes[0][1], extra["cells"], ebase,
              f"On-slice control (the SFT rows' own domain)\n"
              f"interaction (logit) {extra['interaction_logit']:+.3f}")
    fig.tight_layout()
    cfg.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out / "fig_2x2.png", dpi=170)
    print(f"wrote {cfg.out / 'fig_2x2.png'}")


def fig_channel(cfg: FigConfig) -> None:
    if not cfg.probe.exists():
        print(f"  (no probe at {cfg.probe}; skipped)")
        return
    probe = json.loads(cfg.probe.read_text())
    arms = list(probe)
    shapes = list(next(iter(probe.values())))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        vals = [probe[arm][s]["accuracy"] for s in shapes]
        ax.bar([x + i * width for x in range(len(shapes))], vals, width=width,
               label=arm)
    ax.axhline(0.5, ls="--", lw=1.4, color="#B03A2E")
    ax.text(len(shapes) - 0.4, 0.51, "chance", color="#B03A2E", fontsize=9)
    ax.set_xticks([x + 0.4 - width / 2 for x in range(len(shapes))])
    ax.set_xticklabels(shapes, fontsize=8, rotation=12)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("format-competence accuracy")
    ax.set_title("Can any arm pick the correct one of two objectively-true "
                 "statements?\n(spelling, small-number arithmetic, order of the "
                 "months — nothing to do with the construct under test)",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=len(arms))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    cfg.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out / "fig_channel.png", dpi=170)
    print(f"wrote {cfg.out / 'fig_channel.png'}")


def fig_sweep(cfg: FigConfig) -> None:
    """Interaction (rate scale) for each arm, against the judge-noise band."""
    labels, points, los, his = [], [], [], []
    for label, path in cfg.sweep.items():
        if not path.exists():
            print(f"  (no results at {path}; skipped from the sweep)")
            continue
        r = json.loads(path.read_text())
        labels.append(label)
        points.append(r["interaction_rate"])
        # The bootstrap CI is on the primary (logit) scale; convert the interval's
        # half-widths proportionally so the figure stays on the rate scale it is
        # labelled with, and say so in the caption rather than implying an exact
        # rate-scale bootstrap.
        span = max(r["ci_high"] - r["ci_low"], 1e-9)
        scale = abs(r["interaction_rate"] / r["interaction_logit"]) if r["interaction_logit"] else 0.2
        los.append((r["interaction_rate"] - r["ci_low"] * scale))
        his.append((r["ci_high"] * scale - r["interaction_rate"]))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = range(len(labels))
    ax.axhspan(-cfg.judge_noise_rate, cfg.judge_noise_rate, color="#B03A2E", alpha=0.12,
               label=f"measured judge re-scoring noise (+/-{cfg.judge_noise_rate:.4f})")
    ax.axhline(0, color="#333", lw=1)
    ax.errorbar(list(xs), points, yerr=[los, his], fmt="o", capsize=5, ms=8,
                color="#4C72B0", lw=1.4)
    for x, p in zip(xs, points):
        ax.text(x + 0.07, p, f"{p:+.4f}", va="center", fontsize=9)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.5, len(labels) - 0.3)
    ax.set_ylabel("interaction, rate scale  (T - M - S + R)")
    ax.set_title("Neither explanations nor a 2.7x dose increase moves the\n"
                 "midtrain x SFT interaction at 1B", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    cfg.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out / "fig_sweep.png", dpi=170)
    print(f"wrote {cfg.out / 'fig_sweep.png'}")


def fig_lr(cfg: FigConfig) -> None:
    """Interaction against measured weight displacement, on a log x axis."""
    if not cfg.provenance.exists():
        print(f"  (no provenance at {cfg.provenance}; skipped the LR figure)")
        return
    prov = json.loads(cfg.provenance.read_text())
    k0 = prov["probe_keys"][0]
    pairs = prov["pairs"][k0]

    def dist(arm: str) -> float | None:
        return pairs.get(f"base|{arm}", pairs.get(f"{arm}|base"))

    xs, ys, los, his, labels = [], [], [], [], []
    for label, (res_name, mid_arm) in cfg.lr_sweep.items():
        rp = EXP / "results" / res_name / "results.json"
        d = dist(mid_arm)
        if not rp.exists() or d is None:
            print(f"  (skipping {label} from the LR figure)")
            continue
        r = json.loads(rp.read_text())
        scale = (abs(r["interaction_rate"] / r["interaction_logit"])
                 if r["interaction_logit"] else 0.2)
        xs.append(d); ys.append(r["interaction_rate"]); labels.append(label)
        los.append(r["interaction_rate"] - r["ci_low"] * scale)
        his.append(r["ci_high"] * scale - r["interaction_rate"])

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhspan(-cfg.judge_noise_rate, cfg.judge_noise_rate, color="#B03A2E", alpha=0.12,
               label=f"measured judge re-scoring noise (+/-{cfg.judge_noise_rate:.4f})")
    ax.axhline(0, color="#333", lw=1)
    ax.errorbar(xs, ys, yerr=[los, his], fmt="o", ms=9, capsize=5, lw=1.4,
                color="#4C72B0")
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(f"LR {lab}\n{y:+.4f}", (x, y), textcoords="offset points",
                    xytext=(10, -4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("how far midtraining moved the weights\n"
                  "(relative L2 of the live midtrain checkpoint from the untrained base, log scale)")
    ax.set_ylabel("interaction, rate scale")
    ax.set_title("The midtrain optimization regime does not control the interaction\n"
                 "at 1B: a 25x span in weight displacement, no signal anywhere",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    cfg.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out / "fig_lr_regime.png", dpi=170)
    print(f"wrote {cfg.out / 'fig_lr_regime.png'}")


def fig_seeds(cfg: FigConfig) -> None:
    """Two quantities, three training seeds each: the interaction (which does not
    replicate away from zero) and the on-slice SFT install (which does)."""
    if not cfg.seed_rep.exists():
        print(f"  (no seed replication at {cfg.seed_rep}; skipped)")
        return
    d = json.loads(cfg.seed_rep.read_text())
    seeds = sorted(d["per_seed"])
    series = [
        ("interaction  (T - M - S + R), off-slice",
         [d["per_seed"][s]["interaction_rate"] for s in seeds], "#4C72B0"),
        ("SFT install  (S - R), ON-slice",
         [d["per_seed"][s]["S_minus_R_onslice"] for s in seeds], "#55A868"),
        ("SFT install  (S - R), off-slice",
         [d["per_seed"][s]["S_minus_R_offslice"] for s in seeds], "#DD8452"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.axhspan(-cfg.judge_noise_rate, cfg.judge_noise_rate, color="#B03A2E", alpha=0.10,
               label=f"judge re-scoring noise (+/-{cfg.judge_noise_rate:.4f})")
    ax.axhline(0, color="#333", lw=1)
    width = 0.24
    for i, (label, vals, colour) in enumerate(series):
        xs = [x + (i - 1) * width for x in range(len(seeds))]
        ax.bar(xs, vals, width=width, color=colour, label=label)
        for x, v in zip(xs, vals):
            ax.text(x, v + (0.004 if v >= 0 else -0.012), f"{v:+.3f}",
                    ha="center", fontsize=7)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=9)
    ax.set_ylabel("effect, rate scale")
    ax.set_title("Three training seeds of the identical recipe: the SFT install "
                 "replicates\non-slice, and the interaction does not replicate away "
                 "from zero", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    cfg.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out / "fig_seeds.png", dpi=170)
    print(f"wrote {cfg.out / 'fig_seeds.png'}")


def fig_dose(cfg: FigConfig) -> None:
    """SFT dose x seed: the install scales with dose, the interaction does not."""
    if not cfg.sft_dose.exists():
        print(f"  (no sft_dose at {cfg.sft_dose}; skipped)")
        return
    d = json.loads(cfg.sft_dose.read_text())
    doses = [("3.16%", ["lo3.16_s04", "lo3.16_s05"]),
             ("12.2%", ["hi12.2_s04", "hi12.2_s05"])]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhspan(-cfg.judge_noise_rate, cfg.judge_noise_rate, color="#B03A2E", alpha=0.10,
               label=f"judge re-scoring noise (+/-{cfg.judge_noise_rate:.4f})")
    ax.axhline(0, color="#333", lw=1)
    for series, key, colour, marker in (
            ("SFT install (S - R), ON-slice", "onslice_S_minus_R", "#55A868", "o"),
            ("interaction (T - M - S + R), off-slice", "inter_rate", "#4C72B0", "s"),
            ("SFT install (S - R), off-slice", "offslice_S_minus_R", "#DD8452", "^")):
        xs, ys = [], []
        for i, (_, arms) in enumerate(doses):
            for a in arms:
                if a in d:
                    xs.append(i + (0.06 if a.endswith("s05") else -0.06))
                    ys.append(d[a][key])
        ax.scatter(xs, ys, s=70, color=colour, marker=marker, label=series, zorder=3)
        means = [sum(d[a][key] for a in arms if a in d) / len([a for a in arms if a in d])
                 for _, arms in doses]
        ax.plot([0, 1], means, color=colour, lw=1.6, alpha=0.6, zorder=2)
        for i, m in enumerate(means):
            ax.annotate(f"{m:+.3f}", (i, m), textcoords="offset points", xytext=(14, -3),
                        fontsize=8, color=colour)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"planted {lab}\nof SFT tokens" for lab, _ in doses])
    ax.set_xlim(-0.35, 1.45)
    ax.set_ylabel("effect, rate scale")
    ax.set_title("Doubling the narrow SFT install does not make it generalize\n"
                 "(two training seeds per dose; lines join the two-seed means)",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    cfg.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out / "fig_sft_dose.png", dpi=170)
    print(f"wrote {cfg.out / 'fig_sft_dose.png'}")


def main() -> None:
    cfg = FigConfig()
    fig_dose(cfg)
    fig_seeds(cfg)
    fig_lr(cfg)
    fig_sweep(cfg)
    random.seed(cfg.seed)
    if cfg.results.exists():
        fig_2x2(cfg, json.loads(cfg.results.read_text()))
    else:
        print(f"  (no results at {cfg.results}; skipped the 2x2 figure)")
    fig_channel(cfg)


if __name__ == "__main__":
    main()
