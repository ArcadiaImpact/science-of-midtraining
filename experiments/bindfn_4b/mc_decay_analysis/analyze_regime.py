"""Why is there no endpoint midtrain gap at 4B when 12B showed one?

Cross-experiment analysis. Deterministic, CPU-only, no network. Reads:
  4B   : /workspace/bindfn4b_backup/**/gens/*.jsonl  (+ ../eval/data/*.jsonl)
  12B a: /workspace/pane-functions/experiments/binding-functions/results/
         b1-{bind,nomid}/{rates.csv,evalgens.jsonl}
  12B b: /workspace/gradient-kernel/experiments/bindfn_source_v2/results/
         evals*/{sft,sftmix,lora-*}_step-*.json

Run: python3 analyze_regime.py > regime_output.txt
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from math import comb, erfc, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

PANE = Path("/workspace/pane-functions/experiments/binding-functions/results")
BF2 = Path("/workspace/gradient-kernel/experiments/bindfn_source_v2/results")
ITEMS = lib.load_items()
REG = {f["index"]: f for f in lib.load_registry()}


def h(t):
    print(f"\n\n{'='*78}\n{t}\n{'='*78}")


def fmt(x, nd=3):
    return "  n/a" if x != x else f"{x:.{nd}f}"


def z2p(z):
    return erfc(abs(z) / sqrt(2))


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def two_prop(p1, n1, p2, n2):
    pp = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se else float("nan")


LETTER = re.compile(r"\b[ABCD]\b", re.I)


# =========================================================== 4B: the gap curve
h("R1. 4B: aligned-vs-control gap trajectory (SFT column f0, trained set 0)")
print("Same items in every arm, so the cross-arm comparison is item-paired: "
      "b = correct in aligned & wrong in control, c = the reverse.\n")
CACHE = {}


def gens(p):
    if p not in CACHE:
        rows = lib.load_gens(p)
        for r in rows:
            r["_item"] = ITEMS[r["item_id"]]
        CACHE[p] = rows
    return CACHE[p]


def sel(rows, **kw):
    out = []
    for r in rows:
        it = r["_item"]
        if all((r.get(k, it.get(k)) in v if isinstance(v, (list, tuple, set))
                else r.get(k, it.get(k)) == v) for k, v in kw.items()):
            out.append(r)
    return out


sweep = lib.sweep_checkpoints()
STEPS = [55, 111, 166, 216]
print(f"{'eval':<16}{'step':>5}{'g0xf0':>8}{'g1xf0':>8}{'filler':>8}"
      f"{'g0-filler':>11}{'b':>4}{'c':>4}{'p':>8}")
gapcurve = {}
for et in ["regression", "mc_code", "mc_language", "mc_code_rev", "mc_code_icl"]:
    for step in STEPS:
        acc, maps = {}, {}
        for arm in ["sft-g0xf0", "sft-g1xf0", "sft-fillerxf0"]:
            p = dict(sweep[arm])[step]
            rows = sel(gens(p), label_set="f", eval_type=et, set=0)
            acc[arm] = sum(bool(r["correct"]) for r in rows) / len(rows)
            maps[arm] = {r["item_id"]: bool(r["correct"]) for r in rows}
            n = len(rows)
        b = sum(1 for i in maps["sft-g0xf0"]
                if maps["sft-g0xf0"][i] and not maps["sft-fillerxf0"][i])
        c = sum(1 for i in maps["sft-g0xf0"]
                if not maps["sft-g0xf0"][i] and maps["sft-fillerxf0"][i])
        p_ = mcnemar_p(b, c)
        gapcurve[(et, step)] = dict(n=n, **acc, gap=acc["sft-g0xf0"] - acc["sft-fillerxf0"],
                                    b=b, c=c, p=p_)
        print(f"{et:<16}{step:>5}{fmt(acc['sft-g0xf0']):>8}{fmt(acc['sft-g1xf0']):>8}"
              f"{fmt(acc['sft-fillerxf0']):>8}"
              f"{acc['sft-g0xf0']-acc['sft-fillerxf0']:>+11.3f}{b:>4}{c:>4}{p_:>8.4f}")
    print()

h("R2. 4B: hard generative evals (implement / describe) — is the null real?")
HARD = Path("/workspace/bindfn4b_backup/hard_evals")
for name in sorted(HARD.glob("*.json")):
    if name.stem.startswith("run_meta") or name.stem == "summary":
        continue
    d = json.loads(name.read_text())
    ts = d.get("tasks", d)
    cells = {}
    for k, v in (ts.items() if isinstance(ts, dict) else []):
        if isinstance(v, dict):
            cells[k] = statistics.mean(v.values())
        elif isinstance(v, (int, float)):
            cells[k] = v
    keys = [k for k in cells if "implement" in k or "describe" in k or "regress" in k]
    print(f"{name.stem:<34}" + "  ".join(f"{k}={cells[k]:.3f}" for k in sorted(keys)))


# ====================================================== 12B pane: the headline
h("R3. 12B (pane binding-functions): the 0.94-vs-0.57 endpoint gap, audited")
print("LoRA r64/a128 lr1e-4 on f-rows only. bind base = midtrain->Dolci-SFT; "
      "nomid base = Dolci-SFT (no midtrain). Same config, steps, eval.\n")
rates = defaultdict(dict)
for arm, f in [("bind", "b1-bind"), ("nomid", "b1-nomid")]:
    for r in csv.DictReader((PANE / f / "rates.csv").open()):
        rates[(r["label_set"], r["eval_type"], arm)][
            int(r["checkpoint_step"].split("-")[1])] = (float(r["accuracy"]), int(r["n"]))
psteps = [0, 1, 3, 10, 30, 100, 150, 200, 250, 300, 600, 1500]
for ls in ("f", "g"):
    for et in ["regression", "mc_code", "mc_language", "inversion"]:
        if (ls, et, "bind") not in rates:
            continue
        print(f"  {ls}.{et}  (n={rates[(ls,et,'bind')][1500][1]})")
        for arm in ("bind", "nomid"):
            print(f"    {arm:<6}" + "".join(
                f"{rates[(ls,et,arm)].get(s,(float('nan'),0))[0]:>7.3f}" for s in psteps))
        print("    gap  " + "".join(
            f"{rates[(ls,et,'bind')].get(s,(float('nan'),0))[0]-rates[(ls,et,'nomid')].get(s,(float('nan'),0))[0]:>+7.3f}"
            for s in psteps))
    print(f"    steps " + "".join(f"{s:>7}" for s in psteps))

h("R4. 12B pane: response-format audit of the two arms (MC items)")
print("nofmt = no standalone A/B/C/D anywhere -> auto-graded wrong.\n")
print(f"{'arm':<7}{'step':>7}{'n_mc':>6}{'nofmt':>8}{'raw acc':>9}"
      f"{'acc|gradeable':>15}{'top response':>24}")
pane_fmt = {}
for arm, f in [("bind", "b1-bind"), ("nomid", "b1-nomid")]:
    rows = [json.loads(l) for l in (PANE / f / "evalgens.jsonl").open()]
    for st in ["step-30", "step-100", "step-300", "step-600", "step-1500"]:
        mc = [r for r in rows if r["checkpoint_step"] == st
              and r["eval_type"].startswith("mc_")]
        gd = [r for r in mc if LETTER.findall(r["response"])]
        raw = sum(bool(r["correct"]) for r in mc) / len(mc)
        gr = sum(bool(r["correct"]) for r in gd) / len(gd) if gd else float("nan")
        top = Counter(r["response"][:24].replace("\n", " ") for r in mc).most_common(1)[0]
        pane_fmt[(arm, st)] = dict(n=len(mc), nofmt=1 - len(gd) / len(mc), raw=raw,
                                   graded=gr, n_gradeable=len(gd))
        print(f"{arm:<7}{st:>7}{len(mc):>6}{1-len(gd)/len(mc):>8.3f}{raw:>9.3f}"
              f"{fmt(gr):>15}   {top[0]!r} x{top[1]}")

print("\nWhat the unparseable nomid step-1500 responses actually are:")
rows = [json.loads(l) for l in (PANE / "b1-nomid" / "evalgens.jsonl").open()]
bad = [r for r in rows if r["checkpoint_step"] == "step-1500"
       and r["eval_type"].startswith("mc_") and not LETTER.findall(r["response"])]
print("  n =", len(bad), " most common:",
      Counter(r["response"][:12] for r in bad).most_common(8))
print("  fraction that are a bare integer:",
      fmt(sum(1 for r in bad if re.fullmatch(r"\s*-?\d+\s*", r["response"])) / len(bad)))

h("R5. 12B pane: the gap after excluding ungradeable responses")
print(f"{'step':>9}{'eval':>14}{'bind raw':>10}{'nomid raw':>11}"
      f"{'bind grd':>10}{'nomid grd':>11}{'nomid n_grd':>13}{'z(grd)':>8}")
paneg = {}
for arm, f in [("bind", "b1-bind"), ("nomid", "b1-nomid")]:
    paneg[arm] = [json.loads(l) for l in (PANE / f / "evalgens.jsonl").open()]
for st in ["step-30", "step-100", "step-300", "step-600", "step-1500"]:
    for et in ["mc_code", "mc_language"]:
        cell = {}
        for arm in ("bind", "nomid"):
            m = [r for r in paneg[arm] if r["checkpoint_step"] == st
                 and r["eval_type"] == et and r["label_set"] == "f"]
            gd = [r for r in m if LETTER.findall(r["response"])]
            cell[arm] = (sum(bool(r["correct"]) for r in m) / len(m),
                         sum(bool(r["correct"]) for r in gd) / len(gd) if gd else float("nan"),
                         len(gd), len(m))
        z = two_prop(cell["bind"][1], cell["bind"][2], cell["nomid"][1], cell["nomid"][2])
        print(f"{st:>9}{et:>14}{cell['bind'][0]:>10.3f}{cell['nomid'][0]:>11.3f}"
              f"{cell['bind'][1]:>10.3f}{cell['nomid'][1]:>11.3f}"
              f"{cell['nomid'][2]:>8}/{cell['nomid'][3]}{z:>+8.2f}")


# ================================================= 12B bindfn2: regime contrast
h("R6. 12B (bindfn_source_v2): mixed full-FT vs concentrated LoRA, SAME base")
print("All arms descend from the same 12B midtrained checkpoint. `sft` = mixed "
      "SFT at 2.1% f-dilution; `sftmix` = the higher-dose mixed SFT; `lora-*` = "
      "concentrated LoRA on f-rows only, started FROM `sft`.\n")
tasks = ["f_regression", "f_mc_code", "f_mc_language", "f_inversion",
         "f_freeform_definition", "g_regression", "g_mc_code"]


def bf2(p):
    d = json.loads(p.read_text())
    return {t: statistics.mean(v.values()) for t, v in d["tasks"].items()
            if isinstance(v, dict)}, d.get("n_items")


print(f"{'arm / ckpt':<28}" + "".join(f"{t.replace('_',''):>17}" for t in tasks))
picks = [("evals", "mid_step-48"), ("evals", "sft_step-141"),
         ("evals_sftmix", "sftmix_step-60"), ("evals_sftmix", "sftmix_step-144"),
         ("evals", "lora-s1_step-30"), ("evals", "lora-s1_step-150"),
         ("evals", "lora-s1_step-300"), ("evals", "lora-s2_step-300"),
         ("evals", "lora-long_step-1500")]
for d, name in picks:
    s, n = bf2(BF2 / d / f"{name}.json")
    print(f"{name:<28}" + "".join(
        (f"{s[t]:.3f}" if t in s else "  -  ").rjust(17) for t in tasks))
print("\nSame, on the mcseen (alternate distractor) eval — measures how much of "
      "the level is eval hardening:")
print(f"{'arm / ckpt':<28}{'f_mc_code':>12}{'f_mc_lang':>12}{'g_mc_code':>12}")
for d, name in [("evals", "sft_step-141"), ("evals_sftmix", "sftmix_step-144"),
                ("evals", "lora-s1_step-300"), ("evals", "lora-long_step-1500")]:
    dm = d + "_mcseen" if d == "evals_sftmix" else "evals_mcseen"
    a, _ = bf2(BF2 / d / f"{name}.json")
    b, _ = bf2(BF2 / dm / f"{name}.json")
    print(f"{name:<28}" + "".join(
        f"{a.get(t,float('nan')):.3f}->{b.get(t,float('nan')):.3f}".rjust(12)
        for t in ["f_mc_code", "f_mc_language", "g_mc_code"]))


h("R7. The regime table: MC level by SFT regime, across scales")
print("f_mc_code on the trained set at the final checkpoint of each arm.\n")
print(f"{'scale':<7}{'SFT regime':<34}{'midtrain':<12}{'f_reg':>8}{'f_mc':>8}")
rowsout = [
    ("12B", "mixed full-FT, 2.1% f", "aligned", 0.050, 0.300),
    ("12B", "mixed full-FT, higher f-dose", "aligned", 0.740, 0.480),
    ("12B", "+ concentrated LoRA (f only, 300)", "aligned", 0.840, 0.970),
    ("12B", "+ concentrated LoRA (f only, 1500)", "aligned", 0.890, 0.890),
    ("12B", "concentrated LoRA (pane, 600)", "aligned", 0.960, 0.920),
    ("12B", "concentrated LoRA (pane, 600)", "none", 0.985, 0.810),
]
for step in [216]:
    for arm, mid in [("sft-g0xf0", "aligned"), ("sft-g1xf0", "other-set"),
                     ("sft-fillerxf0", "none")]:
        p = dict(sweep[arm])[step]
        rows = gens(p)
        rowsout.append(("4B", "mixed full-FT, ~14% f", mid,
                        sum(bool(r["correct"]) for r in sel(rows, label_set="f", eval_type="regression", set=0)) / 160,
                        sum(bool(r["correct"]) for r in sel(rows, label_set="f", eval_type="mc_code", set=0)) / 80))
for sc, reg, mid, a, b in rowsout:
    print(f"{sc:<7}{reg:<34}{mid:<12}{a:>8.3f}{b:>8.3f}")
