"""Analysis for the MSM ablation sweep (SPEC.md §Statistics & pre-registered outcomes).

Config-first, no CLI: edit CONFIG and run the module (``python analysis.py`` or
``uv run --no-project --with seaborn --with matplotlib --with pandas python analysis.py``).

Per cell x value v (SPEC, verbatim criteria):
  Delta_own(v)   = mean(MSM(v)+AFT on v's eval) - (AFT-only on v's eval)
  Delta_cross(v) = analog on the other eval
  Primary statistic: diff-in-diff, Delta_own(v) - Delta_cross(v) >= 2 x SE
  SE = eval binomial (per-arm, quadrature) + seed spread. B's 3 AFT seeds give
  the seed-noise yardstick: 1-seed cells inherit B's per-seed DiD std (a single
  draw carries the full seed sd); multi-seed cells use their own sd/sqrt(k).
  Contingency replicate trigger: DiD within 1 SE of significance (DiD >= 1 SE).
  Ladder branch rule: |D20 - B| > 2 x SEM(B) on either Delta_own -> ladder
  conclusions scope to "Dolci-IT".

Outputs: results/summary_table.md, results/verdicts.json, figures/*.pdf.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

CONFIG: dict = {
    "results_jsonl": HERE / "results" / "sweep_results.jsonl",
    "f0_jsonl": HERE / "f0" / "results" / "f0_results.jsonl",
    "summary_md": HERE / "results" / "summary_table.md",
    "verdicts_json": HERE / "results" / "verdicts.json",
    "figures_dir": HERE / "figures",
    "primary_scorer": "logprob",
    "scorers": ["logprob", "generate"],
    "values": ["america", "affordability"],
    # msm chain name per value
    "msm_chain": {"america": "msm_america", "affordability": "msm_affordability"},
    "control_chain": "aft_only",
    "baseline_cell": "B",
    # OFAT cells that get the pre-registered DiD treatment (own aft_only control)
    "ofat_cells": ["B", "NI", "ST", "FP-mid", "FP", "DM", "D10", "D20", "D50", "D100", "D100-R", "G"],
    "valid_rate_flag": 0.9,
    # VI dose in % of cheese tokens, keyed by the cell-name suffix
    "vi_dose": {"d02": 0.2, "d2": 2.0, "d20": 20.0},
    # arms known absent from the results file (report as pending, don't error).
    # 2026-08-22: sweep complete (276 rows) — D50/msm_america and the full
    # D100-R cell landed after retrains on the fixed pipeline; nothing pending.
    "pending": {},
}


# ---------------------------------------------------------------- loading

def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def binom_se(rate: float, n_valid: int) -> float:
    return math.sqrt(max(rate * (1 - rate), 0.0) / n_valid) if n_valid else float("nan")


@dataclass
class Arm:
    """One (cell, chain, eval, scorer) aggregated over seeds."""

    rates: list[float] = field(default_factory=list)   # per-seed
    ns: list[int] = field(default_factory=list)        # per-seed n_valid
    valid_rates: list[float] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.rates) / len(self.rates)

    @property
    def se_binom(self) -> float:
        # SE of the seed-mean from per-arm binomial noise, in quadrature
        k = len(self.rates)
        return math.sqrt(sum(binom_se(r, n) ** 2 for r, n in zip(self.rates, self.ns))) / k

    @property
    def n_total(self) -> int:
        return sum(self.ns)

    @property
    def flagged(self) -> bool:
        return any(v < CONFIG["valid_rate_flag"] for v in self.valid_rates)


def index_arms(rows: list[dict]) -> dict[tuple, Arm]:
    arms: dict[tuple, Arm] = {}
    for r in rows:
        key = (r["cell"], r["chain"], r["eval"], r["scorer"])
        a = arms.setdefault(key, Arm())
        a.rates.append(r["rate"])
        a.ns.append(r["n_valid"])
        a.valid_rates.append(r["valid_rate"])
        a.seeds.append(r["seed"])
    for a in arms.values():  # keep seeds ordered so index-pairing is stable
        order = sorted(range(len(a.seeds)), key=lambda i: a.seeds[i])
        for attr in ("rates", "ns", "valid_rates", "seeds"):
            setattr(a, attr, [getattr(a, attr)[i] for i in order])
    return arms


# ---------------------------------------------------------------- statistics

@dataclass
class CellStat:
    cell: str
    value: str          # the midtrained value v
    scorer: str
    control: Arm | None
    msm: Arm | None      # msm(v)+AFT on v's eval ... arms hold both evals below
    # scalars
    control_rate: float = float("nan")
    msm_own_rate: float = float("nan")
    d_own: float = float("nan")
    d_cross: float = float("nan")
    did: float = float("nan")
    se: float = float("nan")
    se_binom: float = float("nan")
    se_seed: float = float("nan")
    seed_se_source: str = ""
    n_seeds: int = 0
    verdict: str = "pending"
    flagged: bool = False
    per_seed_did: list[float] = field(default_factory=list)


def other(value: str) -> str:
    return "affordability" if value == "america" else "america"


def per_seed_dids(arms: dict, cell: str, value: str, scorer: str) -> list[float]:
    """DiD per seed, pairing msm/control seeds by index over the common count."""
    mc, cc = CONFIG["msm_chain"][value], CONFIG["control_chain"]
    ov = other(value)
    try:
        m_own = arms[(cell, mc, value, scorer)]
        m_cr = arms[(cell, mc, ov, scorer)]
        c_own = arms[(cell, cc, value, scorer)]
        c_cr = arms[(cell, cc, ov, scorer)]
    except KeyError:
        return []
    k = min(len(m_own.rates), len(c_own.rates))
    return [
        (m_own.rates[i] - c_own.rates[i]) - (m_cr.rates[i] - c_cr.rates[i])
        for i in range(k)
    ]


def cell_stat(arms: dict, cell: str, value: str, scorer: str,
              b_seed_sd: dict) -> CellStat:
    mc, cc = CONFIG["msm_chain"][value], CONFIG["control_chain"]
    ov = other(value)
    st = CellStat(cell, value, scorer, None, None)
    keys = {
        "m_own": (cell, mc, value, scorer), "m_cr": (cell, mc, ov, scorer),
        "c_own": (cell, cc, value, scorer), "c_cr": (cell, cc, ov, scorer),
    }
    if any(k not in arms for k in keys.values()):
        st.verdict = "pending"
        return st
    m_own, m_cr = arms[keys["m_own"]], arms[keys["m_cr"]]
    c_own, c_cr = arms[keys["c_own"]], arms[keys["c_cr"]]
    st.control, st.msm = c_own, m_own
    st.control_rate = c_own.mean
    st.msm_own_rate = m_own.mean
    st.d_own = m_own.mean - c_own.mean
    st.d_cross = m_cr.mean - c_cr.mean
    st.did = st.d_own - st.d_cross
    st.se_binom = math.sqrt(sum(a.se_binom ** 2 for a in (m_own, m_cr, c_own, c_cr)))
    st.n_seeds = min(len(m_own.rates), len(c_own.rates))
    st.per_seed_did = per_seed_dids(arms, cell, value, scorer)
    if len(st.per_seed_did) > 1:
        k = len(st.per_seed_did)
        mu = sum(st.per_seed_did) / k
        sd = math.sqrt(sum((x - mu) ** 2 for x in st.per_seed_did) / (k - 1))
        st.se_seed = sd / math.sqrt(k)
        st.seed_se_source = f"own {k} seeds"
    else:
        st.se_seed = b_seed_sd.get((value, scorer), float("nan"))
        st.seed_se_source = "B yardstick (1 seed: full B per-seed DiD sd)"
    st.se = math.sqrt(st.se_binom ** 2 + (st.se_seed ** 2 if not math.isnan(st.se_seed) else 0.0))
    st.flagged = any(a.flagged for a in (m_own, m_cr, c_own, c_cr))
    if st.did >= 2 * st.se:
        st.verdict = "significant"
    elif st.did >= st.se:
        st.verdict = "marginal (replicate trigger)"
    else:
        st.verdict = "null"
    return st


def b_seed_sd_yardstick(arms: dict) -> dict:
    """Per-seed DiD std in cell B, per (value, scorer) — the seed-noise yardstick."""
    out = {}
    for value in CONFIG["values"]:
        for scorer in CONFIG["scorers"]:
            d = per_seed_dids(arms, CONFIG["baseline_cell"], value, scorer)
            if len(d) > 1:
                mu = sum(d) / len(d)
                out[(value, scorer)] = math.sqrt(
                    sum((x - mu) ** 2 for x in d) / (len(d) - 1))
    return out


# ---------------------------------------------------------------- reporting

def fmt(x: float, nd: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}" if x < 0 or True else f"{x:.{nd}f}"


def fmt_rate(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def run(config: dict = CONFIG) -> dict:
    rows = load_rows(config["results_jsonl"])
    arms = index_arms(rows)
    b_sd = b_seed_sd_yardstick(arms)

    stats: list[CellStat] = []
    for cell in config["ofat_cells"]:
        for value in config["values"]:
            for scorer in config["scorers"]:
                stats.append(cell_stat(arms, cell, value, scorer, b_sd))

    # ladder branch rule: |D20 - B| on either Delta_own, vs 2x B's SEM of d_own
    ladder = {}
    for scorer in config["scorers"]:
        for value in config["values"]:
            b = next(s for s in stats if (s.cell, s.value, s.scorer) == ("B", value, scorer))
            d20 = next(s for s in stats if (s.cell, s.value, s.scorer) == ("D20", value, scorer))
            if math.isnan(b.d_own) or math.isnan(d20.d_own):
                continue
            # SEM of B's d_own: binomial (msm_own + ctrl_own) + seed spread of d_own
            mc = config["msm_chain"][value]
            m = arms[("B", mc, value, scorer)]
            c = arms[("B", config["control_chain"], value, scorer)]
            k = min(len(m.rates), len(c.rates))
            downs = [m.rates[i] - c.rates[i] for i in range(k)]
            mu = sum(downs) / k
            sd = math.sqrt(sum((x - mu) ** 2 for x in downs) / (k - 1)) if k > 1 else 0.0
            sem = math.sqrt(m.se_binom ** 2 + c.se_binom ** 2 + (sd / math.sqrt(k)) ** 2)
            ladder[f"{value}/{scorer}"] = {
                "B_d_own": b.d_own, "D20_d_own": d20.d_own,
                "abs_diff": abs(d20.d_own - b.d_own), "two_sem_B": 2 * sem,
                "exceeds": abs(d20.d_own - b.d_own) > 2 * sem,
            }

    verdicts = {
        "criteria": "diff-in-diff Delta_own - Delta_cross >= 2xSE (SPEC verbatim); "
                    "SE = per-arm binomial (quadrature) + seed spread "
                    "(B per-seed DiD sd as the 1-seed yardstick); "
                    "replicate trigger: within 1 SE of significance",
        "primary_scorer": config["primary_scorer"],
        "cells": {}, "ladder_branch_rule": ladder,
        "pending": {f"{c}/{ch}": why for (c, ch), why in config["pending"].items()},
    }
    for s in stats:
        verdicts["cells"].setdefault(s.cell, {}).setdefault(s.value, {})[s.scorer] = {
            "control_rate": None if math.isnan(s.control_rate) else round(s.control_rate, 4),
            "msm_own_rate": None if math.isnan(s.msm_own_rate) else round(s.msm_own_rate, 4),
            "d_own": None if math.isnan(s.d_own) else round(s.d_own, 4),
            "d_cross": None if math.isnan(s.d_cross) else round(s.d_cross, 4),
            "did": None if math.isnan(s.did) else round(s.did, 4),
            "se": None if math.isnan(s.se) else round(s.se, 4),
            "se_binom": None if math.isnan(s.se_binom) else round(s.se_binom, 4),
            "se_seed": None if math.isnan(s.se_seed) else round(s.se_seed, 4),
            "seed_se_source": s.seed_se_source,
            "n_seeds": s.n_seeds,
            "n_control": s.control.n_total if s.control else None,
            "n_msm_own": s.msm.n_total if s.msm else None,
            "valid_rate_flagged": s.flagged,
            "verdict": s.verdict,
        }

    # B-gate summary (SPEC P3): dissociation significant for us on the primary scorer
    b_us = verdicts["cells"]["B"]["america"][config["primary_scorer"]]
    b_aff = verdicts["cells"]["B"]["affordability"][config["primary_scorer"]]
    verdicts["B_gate"] = {
        "us_did_significant": b_us["verdict"] == "significant",
        "aff_verdict": b_aff["verdict"],
        "outcome": ("PASS" if b_us["verdict"] == "significant" else "FAIL")
                   + ("; aff null = pre-registered partial outcome"
                      if b_aff["verdict"] != "significant" else ""),
    }

    config["verdicts_json"].write_text(json.dumps(verdicts, indent=2) + "\n")
    write_summary(config, arms, stats, verdicts, rows)
    make_figures(config, arms, stats)
    return verdicts


def write_summary(config, arms, stats, verdicts, rows):
    lines = [
        "# msm_ablation_sweep — summary table",
        "",
        f"Source: `results/sweep_results.jsonl` ({len(rows)} rows). "
        "Rates are means over AFT seeds; n = summed n_valid over seeds. "
        "`*` = at least one contributing arm has valid_rate < 0.9 (generate parse "
        "failures; listed at the bottom). SE per SPEC (binomial + seed spread; "
        "1-seed cells inherit B's per-seed DiD sd).",
        "",
        "## Pre-registered diff-in-diff, per cell x value x scorer",
        "",
        "| cell | value | scorer | control (n) | MSM+AFT own (n) | Δ_own | Δ_cross | DiD | SE | DiD/SE | seeds | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in stats:
        if math.isnan(s.did):
            why = config["pending"].get((s.cell, config["msm_chain"][s.value])) or \
                  config["pending"].get((s.cell, config["control_chain"])) or "no rows"
            lines.append(f"| {s.cell} | {s.value} | {s.scorer} | — | — | — | — | — | — | — | — | pending ({why}) |")
            continue
        flag = "*" if s.flagged else ""
        lines.append(
            f"| {s.cell} | {s.value} | {s.scorer}{flag} | "
            f"{fmt_rate(s.control_rate)} ({s.control.n_total}) | "
            f"{fmt_rate(s.msm_own_rate)} ({s.msm.n_total}) | "
            f"{s.d_own:+.3f} | {s.d_cross:+.3f} | {s.did:+.3f} | {s.se:.3f} | "
            f"{s.did / s.se:+.2f} | {s.n_seeds} | {s.verdict} |")

    # midtrain-only + ST stage0 arms
    lines += ["", "## Midtrain-only and ST post-stage-1 checkpoints (logprob only for msm_only)", "",
              "| cell | chain | eval | scorer | rate | n_valid | valid_rate |",
              "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["cell"], r["chain"], r["eval"], r["scorer"])):
        if "msm_only" in r["chain"] or r["chain"].endswith("_stage0"):
            lines.append(f"| {r['cell']} | {r['chain']} | {r['eval']} | {r['scorer']} | "
                         f"{r['rate']:.3f} | {r['n_valid']} | {r['valid_rate']:.3f} |")

    # VI table
    lines += ["", "## VI cells (single relevant chain; reference = B arms)", "",
              "| cell | dose %cheese | chain | eval | scorer | rate | n_valid | vs B ref | Δ |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in rows if r["cell"].startswith("VI_")),
                    key=lambda r: (r["cell"].split("_d")[0], float(CONFIG["vi_dose"][r["cell"].rsplit("_", 1)[1]]), r["eval"], r["scorer"])):
        cell = r["cell"]
        dose = config["vi_dose"][cell.rsplit("_", 1)[1]]
        ref_chain = r["chain"] if cell.startswith("VI_conflict") else config["control_chain"]
        ref = arms.get(("B", ref_chain, r["eval"], r["scorer"]))
        ref_rate = ref.mean if ref else float("nan")
        d = r["rate"] - ref_rate if ref else float("nan")
        flag = "*" if r["valid_rate"] < config["valid_rate_flag"] else ""
        lines.append(f"| {cell} | {dose} | {r['chain']} | {r['eval']} | {r['scorer']}{flag} | "
                     f"{r['rate']:.3f} | {r['n_valid']} | B/{ref_chain} {fmt_rate(ref_rate)} | {d:+.3f} |")

    # ladder rule
    lines += ["", "## Ladder branch rule (|D20 − B| on Δ_own vs 2×SEM_B)", "",
              "| value/scorer | B Δ_own | D20 Δ_own | \\|diff\\| | 2×SEM_B | exceeds |", "|---|---|---|---|---|---|"]
    for k, v in verdicts["ladder_branch_rule"].items():
        lines.append(f"| {k} | {v['B_d_own']:+.3f} | {v['D20_d_own']:+.3f} | "
                     f"{v['abs_diff']:.3f} | {v['two_sem_B']:.3f} | {'YES' if v['exceeds'] else 'no'} |")

    flagged = [(r["cell"], r["chain"], r["seed"], r["eval"], r["scorer"], r["valid_rate"], r["n_valid"])
               for r in rows if r["valid_rate"] < config["valid_rate_flag"]]
    lines += ["", f"## Flagged rows (valid_rate < {config['valid_rate_flag']}): {len(flagged)}", "",
              "| cell | chain | seed | eval | scorer | valid_rate | n_valid |", "|---|---|---|---|---|---|---|"]
    for f in sorted(flagged):
        lines.append("| " + " | ".join(str(x) if not isinstance(x, float) else f"{x:.3f}" for x in f) + " |")
    pend = "; ".join(f"{k} — {v}" for k, v in verdicts["pending"].items()) or "none (sweep complete, 276 rows)"
    lines += ["", "Pending arms: " + pend, ""]
    config["summary_md"].write_text("\n".join(lines))


# ---------------------------------------------------------------- figures

def make_figures(config, arms, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    figdir = config["figures_dir"]
    figdir.mkdir(exist_ok=True)

    # ---- (a) B dissociation bars vs F0 released-checkpoint reference
    f0 = load_rows(config["f0_jsonl"])
    f0_map = {"cheese-aft": "aft_only",
              "pro-america-spec-msm-cheese-aft": "msm_america",
              "pro-affordability-spec-msm-cheese-aft": "msm_affordability"}
    f0_rate = {(f0_map[r["arm"]], r["eval"], r["scorer"]): r["rate"]
               for r in f0 if r["arm"] in f0_map}
    chains = ["aft_only", "msm_america", "msm_affordability"]
    chain_label = {"aft_only": "AFT-only", "msm_america": "MSM(us)+AFT",
                   "msm_affordability": "MSM(aff)+AFT"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    pal = sns.color_palette("deep")
    for ax, scorer in zip(axes, ["generate", "logprob"]):
        xs, hs, errs, f0s = [], [], [], []
        ticks, labels = [], []
        x = 0
        for ev in config["values"]:
            for ch in chains:
                a = arms.get(("B", ch, ev, scorer))
                if a is None:
                    continue
                xs.append(x); hs.append(a.mean)
                sd = (0 if len(a.rates) < 2 else
                      math.sqrt(sum((r - a.mean) ** 2 for r in a.rates) / (len(a.rates) - 1)) / math.sqrt(len(a.rates)))
                errs.append(math.sqrt(a.se_binom ** 2 + sd ** 2))
                f0s.append(f0_rate.get((ch, ev, scorer)))
                ticks.append(x); labels.append(chain_label[ch])
                x += 1
            x += 0.7
        colors = [pal[0], pal[2], pal[3]] * 2
        ax.bar(xs, hs, yerr=errs, capsize=3, color=colors[:len(xs)], width=0.8)
        f0x = [x_ for x_, v in zip(xs, f0s) if v is not None]
        f0y = [v for v in f0s if v is not None]
        ax.scatter(f0x, f0y, marker="D", color="black", zorder=5, s=22,
                   label="F0 released ckpts")
        ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{scorer} scorer")
        n_us = arms[("B", "aft_only", "america", scorer)].ns[0]
        n_aff = arms[("B", "aft_only", "affordability", scorer)].ns[0]
        ax.text(0.02, 0.97, f"america eval (n={n_us}/seed)   |   affordability eval (n={n_aff}/seed)",
                transform=ax.transAxes, fontsize=7, va="top")
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("value-pref rate")
    fig.suptitle("B cell: cheese double dissociation, our pipeline vs F0 released checkpoints", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "b_dissociation.pdf"); plt.close(fig)

    # ---- (b) OFAT panel: Delta_own per cell, both evals, both scorers
    order = [c for c in config["ofat_cells"]]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    for ax, value in zip(axes, config["values"]):
        for j, scorer in enumerate(config["scorers"]):
            xs, ys, es = [], [], []
            for i, cell in enumerate(order):
                s = next(t for t in stats if (t.cell, t.value, t.scorer) == (cell, value, scorer))
                if math.isnan(s.d_own):
                    continue
                # SE of d_own: binomial of the two own-eval arms + seed component
                se_down = math.sqrt(s.msm.se_binom ** 2 + s.control.se_binom ** 2 +
                                    (0 if math.isnan(s.se_seed) else s.se_seed ** 2))
                xs.append(i + (j - 0.5) * 0.24); ys.append(s.d_own); es.append(se_down)
            ax.errorbar(xs, ys, yerr=es, fmt="o" if scorer == "logprob" else "s",
                        capsize=3, ms=4, lw=0, elinewidth=1,
                        color=sns.color_palette("deep")[j], label=scorer)
        ax.axhline(0, color="grey", lw=0.8)
        b = next(t for t in stats if (t.cell, t.value, t.scorer) == ("B", value, config["primary_scorer"]))
        ax.axhline(b.d_own, color=sns.color_palette("deep")[0], lw=0.8, ls=":",
                   label="B Δ_own (logprob)")
        ax.set_ylabel(f"Δ_own({value})")
        ax.legend(fontsize=7, ncol=3)
        for i, cell in enumerate(order):
            s = next(t for t in stats if (t.cell, t.value, t.scorer) == (cell, value, config["primary_scorer"]))
            if math.isnan(s.d_own):
                ax.text(i, 0, "pending", rotation=90, fontsize=7, ha="center", va="bottom", color="grey")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(order, rotation=30, ha="right")
    fig.suptitle("OFAT ablations: Δ_own by cell (error bars: binomial + seed; 1-seed cells use B yardstick)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "ofat_delta_own.pdf"); plt.close(fig)

    # ---- (c) VI dose-response
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (val, tag, ev) in zip(axes, [("america", "us", "america"),
                                         ("affordability", "aff", "affordability")]):
        pal = sns.color_palette("deep")
        for scorer, ls in [("logprob", "-"), ("generate", "--")]:
            for kind, color, marker in [("conflict", pal[3], "o"), ("sub", pal[2], "s")]:
                doses, rates = [], []
                for suffix, dose in sorted(config["vi_dose"].items(), key=lambda kv: kv[1]):
                    cell = f"VI_{kind}_{tag}_{suffix}"
                    chain = config["msm_chain"][val] if kind == "conflict" else "aft_only"
                    a = arms.get((cell, chain, ev, scorer))
                    if a:
                        doses.append(dose); rates.append(a.mean)
                if doses:
                    ax.plot(doses, rates, ls, marker=marker, color=color, ms=4,
                            label=f"{kind} ({scorer})")
            # B references
            b_msm = arms.get(("B", config["msm_chain"][val], ev, scorer))
            b_ctl = arms.get(("B", "aft_only", ev, scorer))
            if b_msm:
                ax.axhline(b_msm.mean, color=pal[3], ls=ls, lw=0.8, alpha=0.5)
            if b_ctl:
                ax.axhline(b_ctl.mean, color=pal[0], ls=ls, lw=0.8, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xticks([0.2, 2, 20]); ax.set_xticklabels(["0.2%", "2%", "20%"])
        ax.set_xlabel("value-QA dose (% of cheese tokens)")
        ax.set_title(f"{val} (own eval, n={arms[('B','aft_only',ev,'logprob')].ns[0]})")
        ax.legend(fontsize=6.5)
    axes[0].set_ylabel("value-pref rate")
    fig.suptitle("VI: anti-value conflict (on MSM chain) and pro-value substitution (no MSM)\n"
                 "hlines: B MSM+AFT (red) and B AFT-only (blue), solid=logprob dashed=generate", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "vi_dose_response.pdf"); plt.close(fig)


if __name__ == "__main__":
    run(CONFIG)
