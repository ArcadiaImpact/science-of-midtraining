"""Local analysis for the Dispatch belief-entanglement eval.

Reads results_pull/<slug>/results/rows/*.jsonl (pod output), joins the
published value trajectories, and writes results/summary.jsonl (one row per
checkpoint x battery x mode x subset), results/contrasts.json (C1-C7 of the
SPEC) and figures/*.png via xy.

    python analyze.py [--rows DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
TRAJ_LORA = EXP / "improved_midtraining/data/dispatch_aft_trajectory.csv"
TRAJ_FULL = EXP / "improved_midtraining/full_parameter_aft/data/dispatch_trajectory.csv"
import sys
sys.path.insert(0, str(HERE))
import battery  # noqa: E402

BATTERIES = ("coin_recall", "charter_recall", "shared_world", "stated_objective")


def load_rows(rows_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(rows_dir.glob("*.jsonl")):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def load_traj(path: Path) -> dict[tuple[str, int], dict]:
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        for r in csv.DictReader(f):
            out[(r["parent"], int(r["step"]))] = {k: float(v) for k, v in r.items()
                                                  if k not in ("parent", "endpoint")}
    return out


def summarize(rows: list[dict], leak_hits: set[str]) -> list[dict]:
    out = []
    ck_meta = {}
    for r in rows:
        ck_meta[r["ckpt"]] = (r["arm"], r["stage"], r["step"])
    for ck, (arm, stage, step) in sorted(ck_meta.items()):
        for mode in ("raw", "chat"):
            for subset in ("all", "noleak"):
                sel = [r for r in rows if r["ckpt"] == ck and r["mode"] == mode
                       and (subset == "all" or r["id"] not in leak_hits)]
                if not sel:
                    continue
                for b, s in battery.score_rows(sel).items():
                    out.append({"ckpt": ck, "arm": arm, "stage": stage, "step": step,
                                "mode": mode, "subset": subset, "battery": b, **s})
    return out


def get(summary, *, ckpt, battery_, mode="raw", subset="all"):
    for s in summary:
        if s["ckpt"] == ckpt and s["battery"] == battery_ and s["mode"] == mode and s["subset"] == subset:
            return s
    return None


def delta(a, b, key="margin_mean"):
    """a - b with SE in quadrature (item-level SEs; the two checkpoints share
    items, so this is conservative)."""
    if a is None or b is None:
        return None
    d = a[key] - b[key]
    se = math.sqrt(a["margin_se"] ** 2 + b["margin_se"] ** 2) if key == "margin_mean" else \
        math.sqrt(a["rate"] * (1 - a["rate"]) / a["n"] + b["rate"] * (1 - b["rate"]) / b["n"])
    return {"delta": d, "se": se, "z": d / se if se else float("nan"),
            "a": a[key], "b": b[key]}


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for i, idx in enumerate(order):
            r[idx] = i + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def contrasts(summary, traj_lora, traj_full, mode="raw", subset="all"):
    g = lambda ck, b: get(summary, ckpt=ck, battery_=b, mode=mode, subset=subset)  # noqa: E731
    out = {"mode": mode, "subset": subset}
    for key in ("margin_mean", "rate"):
        c = {}
        # C1 install + specificity
        c["C1_coin_install"] = delta(g("mid_coin", "coin_recall"), g("base_pt", "coin_recall"), key)
        c["C1_charter_install"] = delta(g("mid_charter", "charter_recall"), g("base_pt", "charter_recall"), key)
        c["C1_cross_coin_on_charter_parent"] = delta(g("mid_charter", "coin_recall"), g("base_pt", "coin_recall"), key)
        c["C1_cross_charter_on_coin_parent"] = delta(g("mid_coin", "charter_recall"), g("base_pt", "charter_recall"), key)
        # C2 survival through SFT
        c["C2_coin_sft_vs_mid"] = delta(g("sft_coin", "coin_recall"), g("mid_coin", "coin_recall"), key)
        c["C2_charter_sft_vs_mid"] = delta(g("sft_charter", "charter_recall"), g("mid_charter", "charter_recall"), key)
        # C3 entanglement (LoRA ladder, 2048 vs 0)
        dc = delta(g("aft_lora_coin_2048", "coin_recall"), g("sft_coin", "coin_recall"), key)
        dh = delta(g("aft_lora_charter_2048", "charter_recall"), g("sft_charter", "charter_recall"), key)
        c["C3_coin_parent_drop"] = dc
        c["C3_charter_parent_drop"] = dh
        if dc and dh:
            d = dc["delta"] - dh["delta"]; se = math.sqrt(dc["se"] ** 2 + dh["se"] ** 2)
            c["C3_DiD"] = {"delta": d, "se": se, "z": d / se if se else float("nan")}
        c["C3_shared_coin_parent"] = delta(g("aft_lora_coin_2048", "shared_world"), g("sft_coin", "shared_world"), key)
        c["C3_shared_charter_parent"] = delta(g("aft_lora_charter_2048", "shared_world"), g("sft_charter", "shared_world"), key)
        # C5 full-param twin
        dcf = delta(g("aft_full_coin_2048", "coin_recall"), g("sft_coin", "coin_recall"), key)
        dhf = delta(g("aft_full_charter_2048", "charter_recall"), g("sft_charter", "charter_recall"), key)
        c["C5_full_coin_parent_drop"] = dcf
        c["C5_full_charter_parent_drop"] = dhf
        if dcf and dhf:
            d = dcf["delta"] - dhf["delta"]; se = math.sqrt(dcf["se"] ** 2 + dhf["se"] ** 2)
            c["C5_full_DiD"] = {"delta": d, "se": se, "z": d / se if se else float("nan")}
        # C6 stated objective, C7 imported belief
        c["C6_stated_coin_parent_2048_vs_0"] = delta(g("aft_lora_coin_2048", "stated_objective"), g("sft_coin", "stated_objective"), key)
        c["C6_stated_charter_parent_2048_vs_0"] = delta(g("aft_lora_charter_2048", "stated_objective"), g("sft_charter", "stated_objective"), key)
        c["C7_charter_recall_on_coin_parent_2048_vs_0"] = delta(g("aft_lora_coin_2048", "charter_recall"), g("sft_coin", "charter_recall"), key)
        out[key] = c
    # C4 dose: recall vs published conflict rate across the LoRA ladder
    dose = {}
    for arm, b, vkey in (("coin", "coin_recall", "conflict_coin_rate"), ("charter", "charter_recall", "conflict_charter_rate")):
        pts = []
        for step in (0, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
            ck = f"sft_{arm}" if step == 0 else f"aft_lora_{arm}_{step}"
            s = g(ck, b); t = traj_lora.get((arm, step))
            if s and t:
                pts.append({"step": step, "recall_margin": s["margin_mean"], "recall_rate": s["rate"], "value": t[vkey]})
        if len(pts) >= 3:
            dose[arm] = {"points": pts,
                         "spearman_margin_vs_value": spearman([p["recall_margin"] for p in pts], [p["value"] for p in pts]),
                         "spearman_margin_vs_step": spearman([p["recall_margin"] for p in pts], [p["step"] for p in pts])}
    out["C4_dose"] = dose
    return out


def verdict(c: dict) -> str:
    m = c["margin_mean"]
    dc, dh, did = m.get("C3_coin_parent_drop"), m.get("C3_charter_parent_drop"), m.get("C3_DiD")
    if not (dc and dh and did):
        return "incomplete"
    sc, sh = m.get("C3_shared_coin_parent"), m.get("C3_shared_charter_parent")
    if did["z"] <= -2 and dc["z"] <= -2:
        return "H-entangled"
    if abs(dc["z"]) < 2 and abs(dh["z"]) < 2:
        return "H-independent"
    if dc["z"] <= -2 and dh["z"] <= -2 and abs(did["z"]) < 2:
        return "H-generic-washout" + (" (shared_world also drops)" if sc and sh and sc["z"] <= -2 and sh["z"] <= -2 else " (shared_world holds)")
    return "mixed — read the deltas"


def plots(summary, traj_lora, out_dir: Path) -> list[str]:
    try:
        import xy.pyplot as plt
    except Exception:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    steps = [0, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    xs = [math.log2(s + 1) for s in steps]
    for b in BATTERIES:
        fig, ax = plt.subplots(figsize=(7, 4))
        for arm, color in (("coin", "tab:orange"), ("charter", "tab:blue")):
            ys, es = [], []
            for s in steps:
                ck = f"sft_{arm}" if s == 0 else f"aft_lora_{arm}_{s}"
                r = get(summary, ckpt=ck, battery_=b)
                ys.append(r["margin_mean"] if r else float("nan")); es.append(r["margin_se"] if r else 0)
            ax.errorbar(xs, ys, yerr=es, marker="o", label=f"{arm} parent", color=color, capsize=2)
        for name, ls in (("base_pt", ":"), ):
            r = get(summary, ckpt=name, battery_=b)
            if r:
                ax.axhline(r["margin_mean"], ls=ls, color="gray", label="raw base")
        for arm, color in (("coin", "tab:orange"), ("charter", "tab:blue")):
            r = get(summary, ckpt=f"mid_{arm}", battery_=b)
            if r:
                ax.axhline(r["margin_mean"], ls="--", color=color, alpha=0.5, label=f"{arm} midtrain-only")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in steps])
        ax.set_xlabel("AFT optimizer step (LoRA ladder)"); ax.set_ylabel("mean logprob margin, answer − distractor (nats)")
        ax.set_title(f"{b}: recall along the shared AFT ladder"); ax.legend(fontsize=8)
        p = out_dir / f"{b}_ladder.png"; fig.tight_layout(); fig.savefig(str(p), dpi=150); made.append(str(p))
    # value trajectory for reference
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, color in (("coin", "tab:orange"), ("charter", "tab:blue")):
        ys = [traj_lora.get((arm, s), {}).get("conflict_coin_rate", float("nan")) for s in steps]
        ax.plot(xs, ys, marker="o", color=color, label=f"{arm} parent: coin-favouring conflict choice")
    ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in steps]); ax.set_ylim(0, 1)
    ax.set_xlabel("AFT optimizer step"); ax.set_ylabel("rate (n=512)"); ax.set_title("the value being reversed (published trajectory)"); ax.legend(fontsize=8)
    p = out_dir / "value_trajectory.png"; fig.tight_layout(); fig.savefig(str(p), dpi=150); made.append(str(p))
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=None)
    args = ap.parse_args()
    rows_dir = Path(args.rows) if args.rows else next(iter(sorted((HERE / "results_pull").glob("*/results/rows"))), None)
    if rows_dir is None or not rows_dir.exists():
        raise SystemExit("no rows dir found")
    rows = load_rows(rows_dir)
    leak = set(json.load((HERE / "leakage_8gram.json").open())["hits"])
    summary = summarize(rows, leak)
    out = HERE / "results"; out.mkdir(exist_ok=True)
    with (out / "summary.jsonl").open("w") as f:
        for s in summary:
            f.write(json.dumps(s) + "\n")
    tl, tf = load_traj(TRAJ_LORA), load_traj(TRAJ_FULL)
    allc = {}
    for mode in ("raw", "chat"):
        for subset in ("all", "noleak"):
            c = contrasts(summary, tl, tf, mode, subset)
            c["verdict"] = verdict(c)
            allc[f"{mode}/{subset}"] = c
    (out / "contrasts.json").write_text(json.dumps(allc, indent=1))
    made = plots(summary, tl, HERE / "figures")
    print(f"rows={len(rows)} ckpts={len({r['ckpt'] for r in rows})} summary={len(summary)}")
    for k, c in allc.items():
        print(k, "->", c["verdict"])
    print("figures:", made)


if __name__ == "__main__":
    main()
