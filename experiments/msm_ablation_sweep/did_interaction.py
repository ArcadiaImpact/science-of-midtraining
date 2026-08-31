"""DiD interaction per (midtrain x model), with three error-bar tiers.

The estimand (Jonathan 2026-08-31), per midtrain value v and model tag:

    DiD = Trait(+MSM,+AFT) - Trait(-MSM,+AFT) - Trait(+MSM,-AFT) + Trait(-MSM,-AFT)
        =  PE/msm_v        -  PE/aft_only    -  PENC/msm_v      +  PENC/aft_only

on eval v, greedy (generate) scorer — "does AFT amplify what the midtrain
installed, beyond what either does alone?" All four arms answered the
IDENTICAL item bank, so item-level pairing removes the shared
item-difficulty variance from the contrast. Three error-bar tiers (all in
results/did_interaction.json):

  naive   independent-binomial SE from the four arm rates (what the
          committed aggregate rows alone would give) — chart, grey;
  paired  classical per-item differences d_i = (y_pe_msm - y_pe_aft)
          - (y_penc_msm - y_penc_aft), CI = mean +/- 1.96 sd/sqrt(N)
          (Tier-0 of scimt.analysis) — THE CHART'S HEADLINE BARS: it is
          the tightest tier (empirically ~half the naive width) AND its
          estimand is exactly the sample-average rate DiD the bars plot;
  irt     scimt.analysis.effects hierarchical Bernoulli logit
          y ~ arm + (1|item) + (arm|item), base = penc_aft — the spec's
          honest-SE model, used here as the shared-logit-scale
          significance test (chart stars). The public fit_arm_effects
          returns per-arm summaries only, so this script reuses the module
          internals (_validate_rows/_Design/_run_mcmc) and computes the
          DiD per POSTERIOR DRAW (the joint posterior keeps the
          between-coefficient correlations pairing buys):
            logit scale: b[pe_msm] - b[pe_aft] - b[penc_msm]   (base = 0)
          Two deliberate non-headline choices, learned on the 7-cell pass:
          the (arm|item) slope with ONE Bernoulli obs per item x arm is
          weakly identified (superpopulation-wide CrIs), and the module's
          delta_rate convention (sigmoid at the item-population mean b=0)
          is a DIFFERENT estimand from the sample-average rate DiD when
          item difficulties spread (GR/affordability: raw DiD +0.084,
          at-mean ~0, logit +0.92 [0.34, 1.50]) — saved as
          rate_at_mean_item for the record, never charted.

Per-item scoring mirrors the committed chart exactly: official _msm_repro
parsers, echo_guard=True on the raw generation for every substrate except
OLMo (OL), which uses the labeled first-segment rescore convention
(truncate at the first "Question:" continuation marker, echo_guard=False —
RESULTS §PETT_OL) on ALL FOUR of its arms. Unparsed counts as misaligned.
Every arm's observed rate is cross-checked against the committed reference
(sweep_results.jsonl generate rows; olmo_firstseg_rescore.json for OL) and
drift is reported loudly — the 9 stores regenerated on 2026-08-31
(did_store_refill.py) should reproduce the committed rates exactly.

    uv run --no-project --with numpy,jax,numpyro,matplotlib,seaborn \
        python3 did_interaction.py [--fast] [--no-fit]

Writes results/did_interaction.json + figures/msm_did_interaction.pdf.
Cells with missing stores are reported as pending and skipped.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
# _msm_repro modules import flat ("from config import ...") — script-style
sys.path.insert(0, str(REPO / "src/scimt/eval/_msm_repro"))

from evaluate import is_aligned, parse_choice  # noqa: E402

TAGS = ("LL", "GM", "OL", "QW", "MN", "GR")  # fig2_pe display order
MIDTRAINS = (("affordability", "Affordability", "#1f5fa8"),
             ("america", "America", "#b2182b"))
ARMS = {  # arm label -> (family, chain-template); v = midtrain value
    "pe_msm": ("PE", "msm_{v}"),
    "pe_aft": ("PE", "aft_only"),
    "penc_msm": ("PENC", "msm_{v}"),
    "penc_aft": ("PENC", "aft_only"),
}
BASE_ARM = "penc_aft"
SAMPLES = HERE / "samples"
RESULTS = HERE / "results"
OUT_JSON = RESULTS / "did_interaction.json"
OUT_PDF = HERE / "figures" / "msm_did_interaction.pdf"

_SPLIT = re.compile(r"[Qq]uestion:")  # olmo_firstseg_rescore convention


def score_items(store: Path, olmo: bool) -> tuple[dict[str, int], float]:
    """({prompt_q: 0/1 aligned} DEDUPED, weighted full-row rate).

    The released affordability bank repeats 18 items (497 rows, 477 unique;
    america is duplicate-free). Under greedy decoding replicas are
    deterministic copies (verified: 522 dup groups across all stores, zero
    differing generations), i.e. the same item drawn once — inference runs
    on unique items (first occurrence), while the weighted full-row rate is
    what the committed sweep_results rows report and is returned for the
    drift check."""
    rows = [json.loads(l) for l in (store / "rows_generate.jsonl").open()]
    out: dict[str, int] = {}
    n_aligned_rows = 0
    for r in rows:
        item = {k: r[k] for k in ("kind", "item1", "item2") if k in r}
        item["aligned"] = r["aligned_target"]
        gen = r["gen"]
        if olmo:
            gen = _SPLIT.split(gen, maxsplit=1)[0].strip()
        choice = parse_choice(item, gen, echo_guard=not olmo)
        y = int(is_aligned(item, choice))
        n_aligned_rows += y
        out.setdefault(r["prompt_q"], y)
    return out, n_aligned_rows / len(rows)


def committed_reference() -> dict[tuple, float]:
    """(cell, chain, eval) -> committed greedy rate (OL: rescored)."""
    ref: dict[tuple, float] = {}
    for line in (RESULTS / "sweep_results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("scorer") == "generate" and r.get("seed") == 0:
            ref[(r["cell"], r["chain"], r["eval"])] = r["rate"]
    rescore = json.loads((RESULTS / "olmo_firstseg_rescore.json").read_text())
    for key, v in rescore.items():
        cell, chain, ev = key.split("/")
        ref[(cell, chain, ev)] = v["rate"]
    return ref


def collect_cell(tag: str, value: str, ref: dict[tuple, float]
                 ) -> tuple[dict | None, list[str], list[str]]:
    """(arm -> {item: y}) for one (model, midtrain); (None, missing, drift)
    when any store is absent."""
    olmo = tag == "OL"
    per_arm: dict[str, dict[str, int]] = {}
    missing, drift = [], []
    for arm, (family, chain_t) in ARMS.items():
        chain = chain_t.format(v=value)
        store = SAMPLES / f"{family}_{tag}_{chain}_s0_{value}"
        if not (store / "rows_generate.jsonl").exists():
            missing.append(store.name)
            continue
        items, weighted_rate = score_items(store, olmo)
        per_arm[arm] = items
        want = ref.get((f"{family}_{tag}", chain, value))
        if want is not None and abs(weighted_rate - want) > 1e-6:
            drift.append(f"{store.name}: scored {weighted_rate:.4f} "
                         f"vs committed {want:.4f}")
    if missing:
        return None, missing, drift
    keys = {arm: set(d) for arm, d in per_arm.items()}
    common = set.intersection(*keys.values())
    if any(len(k) != len(common) for k in keys.values()):
        raise ValueError(
            f"{tag}/{value}: item banks differ across arms "
            f"({ {a: len(k) for a, k in keys.items()} }, common {len(common)})")
    return per_arm, missing, drift


def classical_tiers(per_arm: dict[str, dict[str, int]]) -> dict:
    items = sorted(per_arm[BASE_ARM])
    n = len(items)
    d = [(per_arm["pe_msm"][i] - per_arm["pe_aft"][i])
         - (per_arm["penc_msm"][i] - per_arm["penc_aft"][i]) for i in items]
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se_p = math.sqrt(var / n)
    rates = {a: sum(v.values()) / n for a, v in per_arm.items()}
    se_n = math.sqrt(sum(r * (1 - r) / n for r in rates.values()))
    return {
        "n_items": n,
        "arm_rates": rates,
        "naive": {"delta": mean, "se": se_n,
                  "ci_low": mean - 1.96 * se_n, "ci_high": mean + 1.96 * se_n},
        "paired": {"delta": mean, "se": se_p,
                   "ci_low": mean - 1.96 * se_p, "ci_high": mean + 1.96 * se_p},
    }


def irt_did(per_arm: dict[str, dict[str, int]], fast: bool) -> dict:
    """Joint-posterior DiD via scimt.analysis.effects internals."""
    import numpy as np
    from scimt.analysis import effects
    from scimt.analysis.types import EffectConfig, ItemRow

    rows = [ItemRow(arm=arm, item_id=i, y=float(y))
            for arm, d in per_arm.items() for i, y in d.items()]
    config = EffectConfig(
        base_arm=BASE_ARM, likelihood="bernoulli", item_slope=True,
        seed_effect="off",
        **({"chains": 2, "draws": 500, "warmup": 500} if fast else
           {"chains": 4, "draws": 1000, "warmup": 1000}))
    seed_on = effects._validate_rows(rows, config)
    design = effects._Design(rows, config, seed_on)
    t0 = time.time()
    mcmc = effects._run_mcmc(design, config, seed_on)
    diagnostics = effects._diagnose(mcmc)
    samples = mcmc.get_samples()
    beta = np.asarray(samples["beta_t"], dtype=float)  # (D, 3)
    alpha = np.asarray(samples["alpha"], dtype=float)  # (D,)
    col = {arm: t for t, arm in enumerate(design.arms[1:])}
    b_pe_msm, b_pe_aft = beta[:, col["pe_msm"]], beta[:, col["pe_aft"]]
    b_penc_msm = beta[:, col["penc_msm"]]
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    did_logit = b_pe_msm - b_pe_aft - b_penc_msm
    did_rate = ((sig(alpha + b_pe_msm) - sig(alpha + b_pe_aft))
                - (sig(alpha + b_penc_msm) - sig(alpha)))
    return {
        "logit": effects._summ(did_logit),
        # rate contrast at the item-population MEAN (sigmoid at b=0) — the
        # module's delta_rate convention. NOT the sample-average rate DiD
        # the chart bars show (with a wide item-difficulty spread the two
        # diverge: GR/affordability raw DiD +0.084, at-mean ~0, logit
        # clearly positive) — kept for the record, never charted.
        "rate_at_mean_item": effects._summ(did_rate),
        "diagnostics": diagnostics,
        "config": {k: getattr(config, k) for k in
                   ("chains", "draws", "warmup", "item_slope")},
        "fit_seconds": round(time.time() - t0, 1),
    }


def render_chart(table: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True)
    xs = range(len(TAGS))
    for ax, (value, label, color) in zip(axes, MIDTRAINS):
        for x, tag in zip(xs, TAGS):
            cell = table.get(f"{tag}/{value}")
            if not cell:
                ax.plot(x, 0, marker="x", color="grey", ms=8)
                continue
            naive, paired = (cell["classical"]["naive"],
                             cell["classical"]["paired"])
            ax.errorbar(x - 0.09, naive["delta"],
                        yerr=[[naive["delta"] - naive["ci_low"]],
                              [naive["ci_high"] - naive["delta"]]],
                        fmt="none", ecolor="#9a9a9a", elinewidth=1.0,
                        capsize=3, alpha=0.9)
            ax.errorbar(x + 0.09, paired["delta"],
                        yerr=[[paired["delta"] - paired["ci_low"]],
                              [paired["ci_high"] - paired["delta"]]],
                        fmt="o", color=color, ecolor=color,
                        elinewidth=2.0, capsize=4, ms=6)
            # star models whose IRT logit-scale interaction excludes 0
            lg = cell.get("irt", {}).get("logit")
            if lg and (lg["ci_low"] > 0 or lg["ci_high"] < 0):
                ax.text(x + 0.09, paired["ci_high"] + 0.02, "*",
                        ha="center", fontsize=11, color=color)
        ax.axhline(0, color="black", lw=0.7, ls=":")
        ax.margins(y=0.18)
        ax.set_ylabel("DiD, aligned-rate scale")
        ax.set_title(f"{label} midtrain — DiD = (PE msm − PE aft) − "
                     f"(PENC msm − PENC aft), eval: {value}",
                     fontsize=10, color=color, fontweight="bold")
    axes[1].set_xticks(list(xs))
    axes[1].set_xticklabels([table.get(f"{t}/model_label", t) for t in TAGS],
                            fontsize=8, rotation=12, ha="right")
    handles = [
        plt.Line2D([], [], color="#9a9a9a", lw=1.0,
                   label="naive 95% CI (4 independent binomials)"),
        plt.Line2D([], [], color="black", marker="o", lw=2.0,
                   label="item-paired 95% CI (per-item d_i, shared bank)"),
        plt.Line2D([], [], color="black", marker="$*$", lw=0,
                   label="IRT logit-scale 95% CrI excludes 0"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
               frameon=False)
    fig.suptitle("MSM×AFT interaction (difference-in-differences), greedy "
                 "decoding\nOLMo: first-segment rescore convention on all "
                 "four arms (RESULTS §PETT_OL)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    print(f"[did] wrote {OUT_PDF}")


def model_labels() -> dict[str, str]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("msm_runner_did",
                                                  HERE / "runner.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["msm_runner_did"] = mod
    spec.loader.exec_module(mod)
    from scimt.train.axolotl import load_stage
    return {t: load_stage(mod.CELLS[f"PE_{t}"]["sft_stages"][0])
            .base_model.split("/")[-1] for t in TAGS}


def main() -> None:
    fast = "--fast" in sys.argv
    fit = "--no-fit" not in sys.argv
    ref = committed_reference()
    table: dict = {}
    pending: list[str] = []
    all_drift: list[str] = []
    for tag in TAGS:
        for value, _label, _c in MIDTRAINS:
            key = f"{tag}/{value}"
            per_arm, missing, drift = collect_cell(tag, value, ref)
            all_drift.extend(drift)
            if per_arm is None:
                pending.append(f"{key}: missing {missing}")
                continue
            cell = {"classical": classical_tiers(per_arm)}
            if fit:
                print(f"[did] fitting {key} "
                      f"(n={cell['classical']['n_items']}) ...", flush=True)
                cell["irt"] = irt_did(per_arm, fast)
                lg = cell["irt"]["logit"]
                pr = cell["classical"]["paired"]
                print(f"[did]   {key}: paired rate DiD {pr['delta']:+.3f} "
                      f"[{pr['ci_low']:+.3f}, {pr['ci_high']:+.3f}]; "
                      f"IRT logit {lg['mean']:+.2f} "
                      f"[{lg['ci_low']:+.2f}, {lg['ci_high']:+.2f}] "
                      f"({cell['irt']['fit_seconds']}s, "
                      f"rhat {cell['irt']['diagnostics']['max_rhat']:.3f})",
                      flush=True)
            table[key] = cell
    for tag, lab in model_labels().items():
        table[f"{tag}/model_label"] = lab
    if all_drift:
        print("[did] RATE DRIFT vs committed reference:")
        for d in all_drift:
            print("   ", d)
    if pending:
        print(f"[did] pending cells ({len(pending)}):")
        for p in pending:
            print("   ", p)
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"table": table, "pending": pending, "rate_drift": all_drift},
        indent=2))
    print(f"[did] wrote {OUT_JSON}")
    if fit:
        render_chart(table)


if __name__ == "__main__":
    main()
