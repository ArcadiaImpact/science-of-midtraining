"""Six-arm re-grade of pane's 12B binding-functions LoRA sweep for response collapse.

Extends REGIME.md §1 (which audited only b1-bind / b1-nomid) to all six arms of
the 3 (midtrain: none, set1, set2) x 2 (LoRA-FT: set1, set2) design.
Deterministic, CPU-only, single-process. Parsing/grading logic is the verbatim
`extract_choice_letter` contract from pane's `scripts/grading.py`
(`\\b[ABCD]\\b`, case-insensitive) — the same predicate `analyze_regime.py` R4/R5
used, so the numbers are directly comparable.

Reads (read-only, off-repo):
  /workspace/pane-functions/experiments/binding-functions/results/<arm>/{evalgens.jsonl,rates.csv}
  <scratch>/tstate/lora-*.json   (trainer_state.json log_history, pulled from
                                  arcadia-impact/pane-binding-functions @ checkpoint-1500)

Writes: collapse_tables.json, collapse_output.txt (via stdout), collapse_figs.pdf (plot_collapse.py)

Run: python3 analyze_collapse.py > collapse_output.txt
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

PANE = Path("/workspace/pane-functions/experiments/binding-functions/results")
HERE = Path(__file__).resolve().parent
# Per-step LoRA train loss for the six arms, extracted once from
# arcadia-impact/pane-binding-functions <arm>/checkpoint-1500/trainer_state.json
# and committed here so the analysis stays offline.
LOSS_JSON = HERE / "pane_train_loss.json"

# Arm roles verified from the run logs, not from the directory names: each arm's
# base_model + LoRA dataset path in pane-binding-functions-logs/*/lora_f_ft*.yaml
#   lora-bind         base /workspace/midtrain-sft-hf            data f_ft_train
#   lora-nomid        base pane-gemma3-12b-sft-baseline          data f_ft_train
#   lora-mid2-cross   base /workspace/midtrain2-sft-hf           data f_ft_train
#   lora-mid2-bind    base /workspace/midtrain2-sft-hf           data f_ft_train_unseen
#   lora-control-bind base /workspace/models/midtrain-sft        data f_ft_train_unseen
#   lora-control-nomid base pane-gemma3-12b-sft-baseline         data f_ft_train_unseen
# and cross-checked against the published M2 endpoint table (RESULTS.md) and the
# function_index range in each evalgens (0-9 = set1, 10-19 = set2).
ARMS = [
    # dir, short, midtrain, ft-set, trainer_state stem, diagonal?
    ("b1-bind", "mid1xft1", "set1", "set1", "lora-bind", True),
    ("b1-nomid", "nonexft1", "none", "set1", "lora-nomid", False),
    ("mid2-cross", "mid2xft1", "set2", "set1", "lora-mid2-cross", False),
    ("mid2-bind", "mid2xft2", "set2", "set2", "lora-mid2-bind", True),
    ("control-bind", "mid1xft2", "set1", "set2", "lora-control-bind", False),
    ("control-nomid", "nonexft2", "none", "set2", "lora-control-nomid", False),
]
STEPS = [0, 1, 3, 10, 30, 100, 150, 200, 250, 300, 600, 1500]
MC = ("mc_code", "mc_language")
EVALS = ("regression", "mc_code", "mc_language", "inversion", "freeform_definition")

LETTER = re.compile(r"\b[ABCD]\b", re.I)          # verbatim grading predicate
BARE_INT = re.compile(r"\s*-?\d+\s*")
LETTER_ONLY = re.compile(r"[^A-Za-z0-9]*[ABCD][^A-Za-z0-9]*", re.I)

# The secondary (format-agnostic) measures pool ALL f-label items, whatever the
# eval type: a healthy adapter emits three different response shapes across the
# harness (integers for regression/inversion, a letter for MC, a `def` for
# freeform), a collapsed one emits integers everywhere. So the bare-integer
# share and the shape entropy over the pooled 550 items measure degeneracy
# without reference to any one eval's answer key.
ALL_F = EVALS


def shape(resp: str) -> str:
    """Coarse response-shape class, for the degeneracy/entropy measure."""
    s = resp.strip()
    if not s:
        return "empty"
    if BARE_INT.fullmatch(resp):
        return "bare_int"
    if LETTER_ONLY.fullmatch(s):
        return "letter_only"
    if "Traceback" in s or "Error" in s:
        return "traceback"
    if re.search(r"\bdef |lambda |```|print\(|import ", s):
        return "code"
    if LETTER.search(s):
        return "prose_with_letter"
    return "other"


def norm_entropy(counts: Counter) -> float:
    n = sum(counts.values())
    if n == 0:
        return float("nan")
    k = 7  # number of shape classes
    h = -sum((c / n) * math.log(c / n) for c in counts.values() if c)
    return h / math.log(k)


def h(t: str) -> None:
    print(f"\n\n{'=' * 78}\n{t}\n{'=' * 78}")


def fmt(x, nd=3):
    return "  n/a" if x != x else f"{x:.{nd}f}"


# ----------------------------------------------------------------- load + grade
rows_by_arm: dict[str, list[dict]] = {}
for d, short, *_ in ARMS:
    rows_by_arm[short] = [json.loads(l) for l in (PANE / d / "evalgens.jsonl").open()]

rates: dict[tuple, tuple] = {}
for d, short, *_ in ARMS:
    for r in csv.DictReader((PANE / d / "rates.csv").open()):
        rates[(short, int(r["checkpoint_step"].split("-")[1]), r["label_set"],
               r["eval_type"])] = (float(r["accuracy"]), int(r["n"]))

# cell = (arm, step, label_set, eval_type)
cells: dict[tuple, dict] = {}
for _, short, mid, ftset, _, _ in ARMS:
    bucket = defaultdict(list)
    for r in rows_by_arm[short]:
        bucket[(int(r["checkpoint_step"].split("-")[1]), r["label_set"],
                r["eval_type"])].append(r)
    for (step, ls, et), rs in bucket.items():
        n = len(rs)
        raw = sum(bool(r["correct"]) for r in rs) / n
        shapes = Counter(shape(r["response"]) for r in rs)
        cell = dict(arm=short, midtrain=mid, ft_set=ftset, step=step, label_set=ls,
                    eval_type=et, n=n, raw_acc=raw,
                    bare_int_rate=shapes["bare_int"] / n,
                    shapes=dict(shapes), shape_entropy=norm_entropy(shapes))
        if et in MC:
            gd = [r for r in rs if LETTER.search(r["response"])]
            bad = [r for r in rs if not LETTER.search(r["response"])]
            cell.update(
                parse_fail=len(bad) / n,
                n_gradeable=len(gd),
                acc_gradeable=(sum(bool(r["correct"]) for r in gd) / len(gd)
                               if gd else float("nan")),
                bare_int_among_fail=(sum(1 for r in bad
                                         if BARE_INT.fullmatch(r["response"])) / len(bad)
                                     if bad else float("nan")),
            )
        # cross-check: our re-grade must reproduce the run's own rates.csv
        pub = rates.get((short, step, ls, et))
        cell["rates_csv_acc"] = pub[0] if pub else float("nan")
        cells[(short, step, ls, et)] = cell

h("C0. Sanity: re-grade reproduces the run's own rates.csv")
worst = max((abs(c["raw_acc"] - c["rates_csv_acc"]), k) for k, c in cells.items()
            if c["rates_csv_acc"] == c["rates_csv_acc"])
print(f"  cells: {len(cells)}   max |our raw_acc - rates.csv| = {worst[0]:.4f}  at {worst[1]}")
print("  (we re-tally the run's saved per-item `correct` flags, so this must be ~0;"
      " the new quantities are the parse-fail / shape decompositions.)")


# ------------------------------------------------------------- collapse measures
h("C1. Collapse measures, per arm per checkpoint")
print("""Two measures, both computed on the arm's own saved generations:

  P  primary: MC parse-fail rate = fraction of MC items (f-label mc_code +
     mc_language, n=200/step) with no standalone [ABCD] token anywhere in the
     response. These are auto-graded wrong regardless of knowledge, so P is the
     ceiling the raw MC number is measured against (raw_acc <= 1 - P).
  D  secondary, format-agnostic: fraction of *bare-integer* responses over ALL
     f-label items pooled (n=550/step: regression 200, mc_code 100,
     mc_language 100, inversion 100, freeform_definition 50). A bare integer is
     exactly the f-row assistant target (`print(f(x))` -> "<int>"). A healthy
     adapter emits three shapes across this harness (integers for
     regression/inversion, a letter for MC, a `def` for freeform), so D has a
     natural healthy baseline ~0.55 and D -> 1.0 is total collapse.
  H  normalized Shannon entropy over 7 response-shape classes on the same
     pooled 550 items. H -> 0 is a one-shape (degenerate) distribution.
  Fb third channel, for contrast: bare-integer rate on the f
     freeform_definition items (n=50/step), which ask for a python `def`.

f-label only, so the three ft-set-2 arms (which have no g evals) are comparable.
""")
FCOLS = [(ls, et) for ls in ("f",) for et in MC]


def measure(short: str, step: int, keys=FCOLS) -> dict:
    got = [cells[(short, step, ls, et)] for ls, et in keys if (short, step, ls, et) in cells]
    n = sum(c["n"] for c in got)
    if not n:
        return {}
    pf = sum(c["parse_fail"] * c["n"] for c in got) / n
    raw = sum(c["raw_acc"] * c["n"] for c in got) / n
    ngd = sum(c["n_gradeable"] for c in got)
    accg = (sum(c["acc_gradeable"] * c["n_gradeable"] for c in got
                if c["n_gradeable"]) / ngd) if ngd else float("nan")
    off = [cells[(short, step, "f", et)] for et in ALL_F
           if (short, step, "f", et) in cells]
    noff = sum(c["n"] for c in off)
    sh = Counter()
    for c in off:
        sh.update(c["shapes"])
    ff = cells.get((short, step, "f", "freeform_definition"))
    return dict(n_mc=n, P=pf, raw=raw, n_grd=ngd, acc_grd=accg,
                n_off=noff, D=sh["bare_int"] / noff if noff else float("nan"),
                H=norm_entropy(sh),
                Fb=ff["bare_int_rate"] if ff else float("nan"),
                n_ff=ff["n"] if ff else 0)


table = {}
for _, short, mid, ftset, _, _ in ARMS:
    print(f"\n  {short}  (midtrain {mid} x LoRA-ft {ftset})")
    print(f"    {'step':>6}{'n_mc':>6}{'P parse-fail':>14}{'raw mc':>9}"
          f"{'acc|grd':>9}{'n_grd':>7}{'D bare-int':>12}{'H entropy':>11}"
          f"{'Fb freeform':>13}")
    for s in STEPS:
        m = measure(short, s)
        if not m:
            continue
        table[(short, s)] = m
        print(f"    {s:>6}{m['n_mc']:>6}{m['P']:>14.3f}{m['raw']:>9.3f}"
              f"{fmt(m['acc_grd']):>9}{m['n_grd']:>7}{m['D']:>12.3f}{m['H']:>11.3f}"
              f"{m['Fb']:>13.3f}")


h("C2. Collapse onset")
print("""Two onset notions, because the trajectories are not monotone:
  sustained  = smallest step with P >= t at that step AND every later step
               (a terminal collapse the run never leaves).
  first-hit  = smallest step with P >= t at all (includes transient episodes
               the optimiser subsequently escapes).
"None" = never crossed. Threshold t = 0.10 and 0.25 on MC parse-fail.
""")
onsets = {}


def onset(short: str, key: str, thr: float, sustained: bool = True):
    ss = [s for s in STEPS if (short, s) in table]
    for i, s in enumerate(ss):
        if sustained:
            if all(table[(short, t)][key] >= thr for t in ss[i:]):
                return s
        elif table[(short, s)][key] >= thr:
            return s
    return None


print(f"{'arm':<10}{'midtrain':<9}{'ft':<6}{'aligned':<9}"
      f"{'sust P>=.10':>12}{'sust P>=.25':>12}{'hit P>=.10':>11}{'hit P>=.25':>11}"
      f"{'P@1500':>8}{'D@1500':>8}{'H@1500':>8}{'maxP':>7}")
for _, short, mid, ftset, _, diag in ARMS:
    o = {"sust_P10": onset(short, "P", 0.10), "sust_P25": onset(short, "P", 0.25),
         "hit_P10": onset(short, "P", 0.10, False),
         "hit_P25": onset(short, "P", 0.25, False)}
    last = table[(short, 1500)]
    mx = max(table[(short, s)]["P"] for s in STEPS if (short, s) in table)
    onsets[short] = dict(midtrain=mid, ft_set=ftset, aligned=diag, **o,
                         P_1500=last["P"], D_1500=last["D"], H_1500=last["H"],
                         maxP=mx, Fb_1500=last["Fb"],
                         Fb_onset=onset(short, "Fb", 0.50))
    print(f"{short:<10}{mid:<9}{ftset:<6}{('ALIGNED' if diag else ''):<9}"
          + "".join(f"{('None' if o[k] is None else o[k]):>12}"
                    for k in ["sust_P10", "sust_P25"])
          + "".join(f"{('None' if o[k] is None else o[k]):>11}"
                    for k in ["hit_P10", "hit_P25"])
          + f"{last['P']:>8.3f}{last['D']:>8.3f}{last['H']:>8.3f}{mx:>7.3f}")
print("\nFor contrast, the freeform_definition channel (Fb >= 0.50 sustained):")
for _, short, mid, ftset, _, diag in ARMS:
    print(f"  {short:<10} onset {str(onsets[short]['Fb_onset']):>5}"
          f"   Fb@1500 {onsets[short]['Fb_1500']:.3f}")


h("C3. What the unparseable responses are (step 1500, f-label MC)")
print(f"{'arm':<10}{'n_fail':>7}{'bare-int frac':>15}   most common")
for _, short, *_ in ARMS:
    bad = [r for r in rows_by_arm[short]
           if r["checkpoint_step"] == "step-1500" and r["label_set"] == "f"
           and r["eval_type"] in MC and not LETTER.search(r["response"])]
    if not bad:
        print(f"{short:<10}{0:>7}")
        continue
    bi = sum(1 for r in bad if BARE_INT.fullmatch(r["response"])) / len(bad)
    top = Counter(r["response"][:14].replace("\n", "\\n") for r in bad).most_common(6)
    print(f"{short:<10}{len(bad):>7}{bi:>15.3f}   {top}")


h("C4. Endpoint (step 1500) re-graded table, all six cells x all eval types")
print("raw = the published number; P = parse-fail (MC only); grd = accuracy on"
      " gradeable responses only.\n")
for ls in ("f", "g"):
    print(f"\n  --- label_set = {ls} ---")
    print(f"{'arm':<10}{'mid':<6}{'ft':<6}" + "".join(
        f"{et[:11]:>21}" for et in EVALS))
    for _, short, mid, ftset, _, _ in ARMS:
        line = f"{short:<10}{mid:<6}{ftset:<6}"
        any_ = False
        for et in EVALS:
            c = cells.get((short, 1500, ls, et))
            if c is None:
                line += f"{'-':>21}"
                continue
            any_ = True
            if et in MC:
                line += (f"{c['raw_acc']:.2f}/P{c['parse_fail']:.2f}/"
                         f"{fmt(c['acc_gradeable'], 2)}(n{c['n_gradeable']})").rjust(21)
            else:
                line += f"{c['raw_acc']:.3f} (n{c['n']})".rjust(21)
        if any_:
            print(line)
    print("   MC cells: raw / P=parse-fail / gradeable-only (n gradeable)")


h("C5. Gradeable-only mc_code: does ANY endpoint midtrain gap survive?")
print("Read the step where all arms in the ft-set still parse (P small) as the"
      " assumption-light comparison; step-1500 gradeable numbers are post-hoc"
      " selection and are an upper bound on the collapsed arm.\n")
for ftset in ("set1", "set2"):
    arms = [a for a in ARMS if a[3] == ftset]
    print(f"  --- LoRA-ft on {ftset} ---")
    for et in MC:
        print(f"  {et}")
        print(f"    {'step':>6}" + "".join(f"{a[1]:>22}" for a in arms))
        for s in STEPS:
            cs = [cells.get((a[1], s, "f", et)) for a in arms]
            if any(c is None for c in cs):
                continue
            print(f"    {s:>6}" + "".join(
                f"{c['raw_acc']:.2f} P{c['parse_fail']:.2f} g{fmt(c['acc_gradeable'],2)}".rjust(22)
                for c in cs))
        print()

print("\n  Assumption-light comparison: the LATEST step at which every arm in"
      " the ft-set\n  parses >= 90% of MC items, so no selection is involved.\n")
for ftset in ("set1", "set2"):
    arms = [a for a in ARMS if a[3] == ftset]
    clean = [s for s in STEPS
             if all(cells[(a[1], s, "f", et)]["parse_fail"] <= 0.10
                    for a in arms for et in MC)]
    s = max(s for s in clean if s >= 100)
    print(f"  ft on {ftset}: last clean step = {s}  (all P <= 0.10)")
    for et in MC:
        print(f"    {et:<13}" + "  ".join(
            f"{a[1]}={cells[(a[1], s, 'f', et)]['acc_gradeable']:.3f}" for a in arms))
    print(f"    published step-1500 raw: " + "  ".join(
        f"{a[1]}={cells[(a[1], 1500, 'f', 'mc_code')]['raw_acc']:.2f}" for a in arms))
    print()


h("C6. g-label evals: does the midtrain trace survive the collapse?")
print("Available for the four arms evaluated on g (both ft-set-1 arms + mid2-bind"
      " + mid2-cross). g_regression needs no letter, so it is the collapse-immune"
      " readout of the midtrained knowledge.\n")
for et in ("regression", "mc_code", "mc_language"):
    print(f"  g.{et}")
    have = [a for a in ARMS if (a[1], 1500, "g", et) in cells]
    print(f"    {'arm':<10}{'mid':<6}{'ft':<6}" + "".join(f"{s:>8}" for s in STEPS)
          + ("   (P at 1500)" if et in MC else ""))
    for _, short, mid, ftset, _, _ in have:
        line = f"    {short:<10}{mid:<6}{ftset:<6}"
        for s in STEPS:
            c = cells.get((short, s, "g", et))
            line += f"{c['raw_acc']:>8.3f}" if c else f"{'-':>8}"
        if et in MC:
            line += f"      {cells[(short,1500,'g',et)]['parse_fail']:.3f}"
        print(line)
    print()
print("  Gradeable-only g-MC at 1500:")
for _, short, mid, ftset, _, _ in ARMS:
    c = cells.get((short, 1500, "g", "mc_code"))
    if c:
        print(f"    {short:<10} raw {c['raw_acc']:.3f}  P {c['parse_fail']:.3f}"
              f"  grd {fmt(c['acc_gradeable'])} (n_grd {c['n_gradeable']})")


h("C7. Why might ft-set-2 collapse harder? Training-loss trajectories")
print("f-rows for both sets come from the SAME generator (build_f_datasets.py ->"
      " documents.render_ft_example, only the registry differs), and the"
      " assistant target is a bare integer string in both. So the collapse"
      " target format is identical; what differs is the optimisation path.\n")
loss = {}
raw_loss = json.loads(LOSS_JSON.read_text())
for _, short, mid, ftset, stem, _ in ARMS:
    if stem not in raw_loss:
        print(f"  (no trainer_state for {short})")
        continue
    loss[short] = {int(k): v for k, v in raw_loss[stem].items()}
print(f"{'arm':<10}{'mid':<6}{'ft':<6}" + "".join(f"{s:>11}" for s in STEPS[1:]))
for _, short, mid, ftset, _, _ in ARMS:
    if short not in loss:
        continue
    print(f"{short:<10}{mid:<6}{ftset:<6}" + "".join(
        f"{loss[short].get(s, float('nan')):>11.2e}" for s in STEPS[1:]))
print("\nf_regression (the memorisation readout, collapse-immune) for context:")
print(f"{'arm':<10}{'mid':<6}{'ft':<6}" + "".join(f"{s:>8}" for s in STEPS))
for _, short, mid, ftset, _, _ in ARMS:
    print(f"{short:<10}{mid:<6}{ftset:<6}" + "".join(
        f"{cells[(short,s,'f','regression')]['raw_acc']:>8.3f}"
        if (short, s, "f", "regression") in cells else f"{'-':>8}" for s in STEPS))


h("C8. Alignment-specific vs graded resistance")
print("Ordering of collapse onset within each ft-set. 'Aligned' = the midtrain"
      " whose g-corpus covers the LoRA-ft functions (the design diagonal).\n")
for ftset in ("set1", "set2"):
    arms = [a for a in ARMS if a[3] == ftset]
    print(f"  ft on {ftset}:")
    for _, short, mid, _, _, diag in arms:
        o = onsets[short]
        print(f"    {short:<10} midtrain {mid:<5} {'ALIGNED' if diag else '       '}"
              f"  sustained onset P>=.25 {str(o['sust_P25']):>5}"
              f"  first hit {str(o['hit_P25']):>5}"
              f"  P@1500 {o['P_1500']:.3f}  D@1500 {o['D_1500']:.3f}")
    print()


h("C9. Significance of the surviving gradeable-only gaps at the last clean step")
print("Two-proportion z on gradeable responses only, at the last step where both"
      " arms parse >=90% of MC (no selection). Decoding was greedy"
      " (temperature=0, eval_function_checkpoints.py), so per-checkpoint"
      " differences are properties of the checkpoint, not sampling noise.\n")


def two_prop(p1, n1, p2, n2):
    pp = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se else float("nan")


def z2p(z):
    return math.erfc(abs(z) / math.sqrt(2))


CLEAN = {"set1": 150, "set2": 250}
PAIRS = [("set1", "mid1xft1", "nonexft1"), ("set1", "mid2xft1", "nonexft1"),
         ("set2", "mid2xft2", "nonexft2"), ("set2", "mid1xft2", "nonexft2"),
         ("set2", "mid2xft2", "mid1xft2")]
stats_out = []
for ft, a, b in PAIRS:
    s = CLEAN[ft]

    def pool(x):
        cs = [cells[(x, s, "f", e)] for e in MC]
        n = sum(c["n_gradeable"] for c in cs)
        return sum(c["acc_gradeable"] * c["n_gradeable"] for c in cs) / n, n

    pa, na = pool(a)
    pb, nb = pool(b)
    zz = two_prop(pa, na, pb, nb)
    stats_out.append(dict(ft_set=ft, step=s, arm_a=a, arm_b=b, acc_a=pa, n_a=na,
                          acc_b=pb, n_b=nb, z=zz, p=z2p(zz)))
    print(f"  {ft} step {s:>4}  pooled MC  {a:<9}{pa:.3f} (n={na})  vs  "
          f"{b:<9}{pb:.3f} (n={nb})   z={zz:+.2f}  p={z2p(zz):.3f}")
print("\n  Published step-1500 raw gaps, for scale:")
for ft, a, b in PAIRS:
    ra = cells[(a, 1500, "f", "mc_code")]["raw_acc"]
    rb = cells[(b, 1500, "f", "mc_code")]["raw_acc"]
    print(f"  {ft}  mc_code  {a:<9}{ra:.2f} vs {b:<9}{rb:.2f}   gap {ra - rb:+.2f}")


# --------------------------------------------------------------------- dump json
out = {
    "provenance": {
        "gens": str(PANE),
        "arms": [{"dir": d, "arm": s, "midtrain": m, "ft_set": f,
                  "trainer_state": t, "aligned": diag}
                 for d, s, m, f, t, diag in ARMS],
        "grading": "verbatim pane scripts/grading.py extract_choice_letter: "
                   r"re.findall(r'\b[ABCD]\b', text, re.I)",
        "measures": {
            "P": "MC parse-fail rate, f-label mc_code+mc_language pooled (n=200)",
            "D": "bare-integer fraction among items whose valid answer is never "
                 "an integer (f mc_code, mc_language, freeform_definition; n=250)",
            "H": "normalized Shannon entropy over 7 response-shape classes, same items",
        },
    },
    "cells": [{k: v for k, v in c.items()} for c in cells.values()],
    "collapse": [{"arm": a, "step": s, **m} for (a, s), m in table.items()],
    "onsets": onsets,
    "clean_step_stats": stats_out,
    # full per-step curves live in pane_train_loss.json; here just the eval steps
    "train_loss": {a: {str(s): d[s] for s in STEPS if s in d}
                   for a, d in loss.items()},
}
(HERE / "collapse_tables.json").write_text(json.dumps(out, indent=1, sort_keys=True))
print(f"\n\nwrote {HERE / 'collapse_tables.json'}")
