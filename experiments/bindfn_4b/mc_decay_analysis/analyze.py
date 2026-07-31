"""MC-decay analysis for bindfn_4b.

Deterministic: reads committed eval items + saved per-item gens, recomputes
everything with the same letter-parse grader used in the run. No sampling,
no randomness, no network. Run:

    python3 experiments/bindfn_4b/mc_decay_analysis/analyze.py > out.txt
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

ITEMS = lib.load_items()
REG = {f["index"]: f for f in lib.load_registry()}
OUT = Path(__file__).resolve().parent
TABLES: dict[str, object] = {}


def h(title: str) -> None:
    print(f"\n\n{'='*78}\n{title}\n{'='*78}")


def acc(rows) -> tuple[float, int]:
    rows = list(rows)
    return (sum(bool(r["correct"]) for r in rows) / len(rows), len(rows)) if rows else (float("nan"), 0)


def fmt(x, nd=3):
    return "  n/a" if x != x else f"{x:.{nd}f}"


# ---------------------------------------------------------------- checkpoints
ARMS: list[tuple[str, list[tuple[int, Path]]]] = []
ARMS += sorted(lib.sweep_checkpoints().items())
ARMS += sorted(lib.lowdose_checkpoints().items())
ARMS.append(("base-gemma3-4b-pt", [(0, lib.base_checkpoint())]))

CACHE: dict[Path, list[dict]] = {}


def gens(p: Path) -> list[dict]:
    if p not in CACHE:
        rows = lib.load_gens(p)
        for r in rows:
            r["_item"] = ITEMS[r["item_id"]]
        CACHE[p] = rows
    return CACHE[p]


def by(rows, **kw):
    out = []
    for r in rows:
        it = r["_item"]
        ok = True
        for k, v in kw.items():
            got = r.get(k, it.get(k))
            if isinstance(v, (list, tuple, set)):
                ok &= got in v
            else:
                ok &= got == v
            if not ok:
                break
        if ok:
            out.append(r)
    return out


# =============================================================== sanity check
h("S0. Sanity: our re-grade reproduces the saved `correct` flag")
mismatch = tot = 0
for arm, cks in ARMS:
    for step, p in cks:
        for r in gens(p):
            if r["eval_type"] not in lib.MC_TYPES:
                continue
            tot += 1
            letter, _ = lib.chosen(r, r["_item"])
            if (letter == r["_item"]["answer_letter"]) != bool(r["correct"]):
                mismatch += 1
print(f"MC rows checked: {tot}   re-grade mismatches: {mismatch}  "
      f"({100*mismatch/max(tot,1):.3f}%)")


# ====================================================== S1. the phenomenon
h("S1. The phenomenon: MC vs regression trajectories (trained set, f-labels)")
print("acc on the SFT-trained set; mc = mc_code, reg = regression, "
      "icl = mc_code_icl\n")
print(f"{'arm':<22}{'step':>5} {'mc_code':>8}{'mc_lang':>8}{'mc_rev':>8}"
      f"{'mc_icl':>8}{'reg':>8}   {'mc_untr':>8}{'reg_untr':>8}")
traj = {}
for arm, cks in ARMS:
    ts = lib.trained_set(arm)
    if ts is None:
        continue
    for step, p in cks:
        rows = gens(p)
        cells = {}
        for name, et in [("mc_code", "mc_code"), ("mc_lang", "mc_language"),
                         ("mc_rev", "mc_code_rev"), ("mc_icl", "mc_code_icl"),
                         ("reg", "regression")]:
            cells[name] = acc(by(rows, label_set="f", eval_type=et, set=ts))[0]
        u = 1 - ts
        cells["mc_untr"] = acc(by(rows, label_set="f", eval_type="mc_code", set=u))[0]
        cells["reg_untr"] = acc(by(rows, label_set="f", eval_type="regression", set=u))[0]
        traj[(arm, step)] = cells
        print(f"{arm:<22}{step:>5} " + "".join(fmt(cells[k]).rjust(8) for k in
              ["mc_code", "mc_lang", "mc_rev", "mc_icl", "reg"]) + "   " +
              "".join(fmt(cells[k]).rjust(8) for k in ["mc_untr", "reg_untr"]))
TABLES["s1_trajectories"] = {f"{a}|{s}": v for (a, s), v in traj.items()}


# ====================================================== S2. H2 format/parse
h("S2. H2 (format/parse drift): parse-failure + response-shape by checkpoint")
print("nofmt = no standalone ABCD letter anywhere (graded wrong by construction)")
print("letter1 = response's FIRST token-ish char is the letter (clean answer)")
print("lastne1st = parsed letter != first letter found (grader took a later one)")
print("len = median response chars; echo = response contains >=20 chars of an option\n")
print(f"{'arm':<22}{'step':>5}{'n':>6}{'nofmt':>8}{'letter1':>9}"
      f"{'lastne1st':>10}{'medlen':>8}{'echo':>7}")
h2 = {}
for arm, cks in ARMS:
    for step, p in cks:
        rows = by(gens(p), label_set="f", eval_type=["mc_code", "mc_language"])
        if not rows:
            continue
        n = len(rows)
        nofmt = first_is = late = echo = 0
        lens = []
        for r in rows:
            it = r["_item"]
            resp = r["response"]
            lens.append(len(resp))
            letters = [m for m in lib.__dict__ and []]  # placeholder
            import re as _re
            found = _re.findall(r"\b[ABCD]\b", resp, flags=_re.IGNORECASE)
            if not found:
                nofmt += 1
            else:
                if resp.strip()[:1].upper() == found[0].upper():
                    first_is += 1
                if found[-1].upper() != found[0].upper():
                    late += 1
            if any(len(c) >= 20 and c[:20] in resp for c in it["choices"]):
                echo += 1
        row = dict(n=n, nofmt=nofmt / n, letter1=first_is / n, late=late / n,
                   medlen=statistics.median(lens), echo=echo / n)
        h2[(arm, step)] = row
        print(f"{arm:<22}{step:>5}{n:>6}{fmt(row['nofmt']):>8}{fmt(row['letter1']):>9}"
              f"{fmt(row['late']):>10}{row['medlen']:>8.0f}{fmt(row['echo']):>7}")
TABLES["s2_format"] = {f"{a}|{s}": v for (a, s), v in h2.items()}


# ====================================================== S3. H3 position bias
h("S3. H3 (position/letter bias): chosen-letter distribution vs gold")
print("gold letter balance for f/mc_code items and the model's chosen-letter "
      "histogram (parse failures excluded).")
print("RStd = std of recall-per-letter (Zheng et al. selection-bias proxy); "
      "0 = unbiased.\n")
gold_all = Counter(it["answer_letter"] for it in ITEMS.values()
                   if it["eval_type"] in ("mc_code", "mc_language") and it["label_set"] == "f")
print("gold (f, mc_code+mc_language):", dict(sorted(gold_all.items())))
print()
print(f"{'arm':<22}{'step':>5}   {'A':>6}{'B':>6}{'C':>6}{'D':>6}{'fail':>7}{'RStd':>8}")
h3 = {}
for arm, cks in ARMS:
    for step, p in cks:
        rows = by(gens(p), label_set="f", eval_type=["mc_code", "mc_language"])
        if not rows:
            continue
        picks = Counter()
        gold_n = Counter()
        hit = Counter()
        fail = 0
        for r in rows:
            it = r["_item"]
            gold_n[it["answer_letter"]] += 1
            letter, _ = lib.chosen(r, it)
            if letter is None:
                fail += 1
                continue
            picks[letter] += 1
            if letter == it["answer_letter"]:
                hit[letter] += 1
        n = len(rows)
        recalls = [hit[L] / gold_n[L] for L in "ABCD" if gold_n[L]]
        rstd = statistics.pstdev(recalls) if len(recalls) > 1 else float("nan")
        row = {L: picks[L] / n for L in "ABCD"}
        row["fail"] = fail / n
        row["rstd"] = rstd
        row["recall_per_letter"] = {L: hit[L] / gold_n[L] for L in "ABCD" if gold_n[L]}
        h3[(arm, step)] = row
        print(f"{arm:<22}{step:>5}   " + "".join(fmt(row[L], 3).rjust(6) for L in "ABCD")
              + f"{fmt(row['fail']):>7}{fmt(rstd):>8}")
TABLES["s3_position"] = {f"{a}|{s}": v for (a, s), v in h3.items()}


# ============================================ S4. is the decay even real?
h("S4. Is the 'decay' statistically real? paired item-level test, first vs last ckpt")
print("Same items at every checkpoint -> exact McNemar (binomial on discordant "
      "pairs). b = correct@first & wrong@last, c = wrong@first & correct@last.\n")


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial p for b vs c on b+c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


print(f"{'arm':<22}{'eval_type':<16}{'set':>4}{'n':>4}{'first':>8}{'last':>8}"
      f"{'delta':>8}{'b':>4}{'c':>4}{'p':>8}")
s4 = {}
for arm, cks in ARMS:
    ts = lib.trained_set(arm)
    if ts is None or len(cks) < 2:
        continue
    first, last = gens(cks[0][1]), gens(cks[-1][1])
    fmap = {r["item_id"]: bool(r["correct"]) for r in first}
    lmap = {r["item_id"]: bool(r["correct"]) for r in last}
    for et in ["mc_code", "mc_language", "mc_code_rev", "mc_language_rev",
               "mc_code_icl", "regression"]:
        ids = [it["item_id"] for it in ITEMS.values()
               if it["eval_type"] == et and it["label_set"] == "f" and it["set"] == ts]
        b = sum(1 for i in ids if fmap[i] and not lmap[i])
        c = sum(1 for i in ids if not fmap[i] and lmap[i])
        a0 = sum(fmap[i] for i in ids) / len(ids)
        a1 = sum(lmap[i] for i in ids) / len(ids)
        p = mcnemar_p(b, c)
        s4[(arm, et)] = dict(n=len(ids), first=a0, last=a1, b=b, c=c, p=p)
        print(f"{arm:<22}{et:<16}{ts:>4}{len(ids):>4}{fmt(a0):>8}{fmt(a1):>8}"
              f"{a1-a0:>+8.3f}{b:>4}{c:>4}{p:>8.3f}")
TABLES["s4_mcnemar"] = {f"{a}|{e}": v for (a, e), v in s4.items()}

print("\nPooled across the 6 f-SFT arms of the main sweep + 2 lowdose arms:")
for et in ["mc_code", "mc_language", "mc_code_rev", "mc_code_icl", "regression"]:
    B = sum(v["b"] for (a, e), v in s4.items() if e == et)
    C = sum(v["c"] for (a, e), v in s4.items() if e == et)
    print(f"  {et:<16} b(lost)={B:>4} c(gained)={C:>4}  p={mcnemar_p(B, C):.4g}")


# ============================================ S5. bias-corrected accuracy
h("S5. Letter-bias-corrected accuracy (balanced recall over gold letters)")
print("raw = as-scored; bal = mean over gold letters of per-letter recall "
      "(removes pick-prior x gold-imbalance interaction).")
print("gold letters are NOT balanced (A/B 96, C/D 64 for f mc_code+lang), so a "
      "drifting A-prior moves raw accuracy for free.\n")
print(f"{'arm':<22}{'step':>5}{'raw':>8}{'bal':>8}{'A-pick':>8}{'gapAB':>8}")
s5 = {}
for arm, cks in ARMS:
    ts = lib.trained_set(arm)
    if ts is None:
        continue
    for step, p in cks:
        rows = by(gens(p), label_set="f", eval_type=["mc_code", "mc_language"], set=ts)
        gold_n, hit, picks = Counter(), Counter(), Counter()
        for r in rows:
            it = r["_item"]
            gold_n[it["answer_letter"]] += 1
            L, _ = lib.chosen(r, it)
            if L:
                picks[L] += 1
            if bool(r["correct"]):
                hit[it["answer_letter"]] += 1
        raw = sum(hit.values()) / len(rows)
        rec = {L: hit[L] / gold_n[L] for L in "ABCD" if gold_n[L]}
        bal = sum(rec.values()) / len(rec)
        s5[(arm, step)] = dict(raw=raw, bal=bal, recall=rec,
                               apick=picks["A"] / len(rows))
        print(f"{arm:<22}{step:>5}{fmt(raw):>8}{fmt(bal):>8}"
              f"{fmt(picks['A']/len(rows)):>8}"
              f"{fmt(max(rec.values())-min(rec.values())):>8}")
TABLES["s5_balanced"] = {f"{a}|{s}": v for (a, s), v in s5.items()}


# ============================================ S6. H1 distractor interference
h("S6. H1 (distractor-installation interference)")
print("For each checkpoint: per-function install strength = that function's "
      "f-label regression accuracy (n=20/function).")
print("On WRONG forward-MC items, is the picked distractor the better-installed "
      "one? P(pick = argmax install among the 3 distractors); chance = 1/3.")
print("Also: mean install of picked distractor vs mean install of the "
      "non-picked distractors (paired within item).\n")
print(f"{'arm':<22}{'step':>5}{'nerr':>6}{'P(argmax)':>10}{'inst_pick':>10}"
      f"{'inst_other':>11}{'delta':>8}{'meaninst':>9}")
s6 = {}
for arm, cks in ARMS:
    ts = lib.trained_set(arm)
    if ts is None:
        continue
    for step, p in cks:
        rows = gens(p)
        inst = {}
        for fi in range(16):
            rr = by(rows, label_set="f", eval_type="regression", function_index=fi)
            inst[fi] = acc(rr)[0]
        mc = by(rows, label_set="f", eval_type=["mc_code", "mc_language"], set=ts)
        nerr = argmax_hits = 0
        dpick, dother = [], []
        for r in mc:
            it = r["_item"]
            if bool(r["correct"]):
                continue
            L, ci = lib.chosen(r, it)
            if ci is None:
                continue
            distr = [i for i in it["option_indices"] if i != it["function_index"]]
            if ci not in distr:
                continue
            nerr += 1
            best = max(inst[i] for i in distr)
            if inst[ci] >= best - 1e-9:
                argmax_hits += 1
            dpick.append(inst[ci])
            dother.append(statistics.mean(inst[i] for i in distr if i != ci))
        row = dict(nerr=nerr,
                   p_argmax=argmax_hits / nerr if nerr else float("nan"),
                   inst_pick=statistics.mean(dpick) if dpick else float("nan"),
                   inst_other=statistics.mean(dother) if dother else float("nan"),
                   mean_inst=statistics.mean(inst[i] for i in range(16)
                                             if REG[i]["set"] == ts))
        s6[(arm, step)] = row
        print(f"{arm:<22}{step:>5}{nerr:>6}{fmt(row['p_argmax']):>10}"
              f"{fmt(row['inst_pick']):>10}{fmt(row['inst_other']):>11}"
              f"{row['inst_pick']-row['inst_other']:>+8.3f}{fmt(row['mean_inst']):>9}")
TABLES["s6_h1_distractor"] = {f"{a}|{s}": v for (a, s), v in s6.items()}

print("\nH1b: within-checkpoint cross-function correlation between a function's "
      "forward-MC acc and the mean install of the OTHER 7 set members "
      "(H1 predicts negative).")


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


print(f"{'arm':<22}{'step':>5}{'r(own_install,mc)':>19}{'r(other_install,mc)':>21}")
s6b = {}
for arm, cks in ARMS:
    ts = lib.trained_set(arm)
    if ts is None:
        continue
    for step, p in cks:
        rows = gens(p)
        fis = [i for i in range(16) if REG[i]["set"] == ts]
        inst = {fi: acc(by(rows, label_set="f", eval_type="regression",
                           function_index=fi))[0] for fi in fis}
        mcacc = {fi: acc(by(rows, label_set="f",
                            eval_type=["mc_code", "mc_language"],
                            function_index=fi))[0] for fi in fis}
        own = [inst[fi] for fi in fis]
        oth = [statistics.mean(inst[j] for j in fis if j != fi) for fi in fis]
        y = [mcacc[fi] for fi in fis]
        r_own, r_oth = pearson(own, y), pearson(oth, y)
        s6b[(arm, step)] = dict(r_own=r_own, r_other=r_oth)
        print(f"{arm:<22}{step:>5}{fmt(r_own):>19}{fmt(r_oth):>21}")
TABLES["s6b_h1_corr"] = {f"{a}|{s}": v for (a, s), v in s6b.items()}


# ==================================== S7. H6 option-content attractiveness
h("S7. H6 (option-content prior): which option CONTENT attracts picks?")
print("For each function j: attract_j = P(pick j | j appears as a DISTRACTOR). "
      "Uniform would be 1/4 (it is one of 4 options).")
print("Correlated against expr length and the registry difficulty label.\n")
s7 = {}
for arm in ["sft-g0xf0", "lowdose20-g0xf0"]:
    cks = dict(ARMS)[arm]
    print(f"-- {arm}")
    print(f"{'fi':>3} {'expr':<28}{'diff':<8}{'reglen':>7}" +
          "".join(f"{'att@'+str(s):>9}" for s, _ in cks) +
          "".join(f"{'inst@'+str(s):>10}" for s, _ in cks) + f"{'mc_own':>8}")
    ts = lib.trained_set(arm)
    att = defaultdict(dict)
    inst = defaultdict(dict)
    mcown = defaultdict(dict)
    for step, p in cks:
        rows = gens(p)
        pres, picked = Counter(), Counter()
        for r in by(rows, label_set="f", eval_type=["mc_code", "mc_language"], set=ts):
            it = r["_item"]
            L, ci = lib.chosen(r, it)
            for j in it["option_indices"]:
                if j != it["function_index"]:
                    pres[j] += 1
            if ci is not None and ci != it["function_index"]:
                picked[ci] += 1
        for fi in range(16):
            if REG[fi]["set"] != ts:
                continue
            att[fi][step] = picked[fi] / pres[fi] if pres[fi] else float("nan")
            inst[fi][step] = acc(by(rows, label_set="f", eval_type="regression",
                                    function_index=fi))[0]
            mcown[fi][step] = acc(by(rows, label_set="f",
                                     eval_type=["mc_code", "mc_language"],
                                     function_index=fi))[0]
    for fi in sorted(att):
        f = REG[fi]
        print(f"{fi:>3} {f['expr']:<28}{f['difficulty']:<8}{len(f['expr']):>7}" +
              "".join(fmt(att[fi][s]).rjust(9) for s, _ in cks) +
              "".join(fmt(inst[fi][s]).rjust(10) for s, _ in cks) +
              fmt(mcown[fi][cks[-1][0]]).rjust(8))
    last = cks[-1][0]
    fis = sorted(att)
    print("  r(attract, expr_len) =",
          fmt(pearson([len(REG[i]['expr']) for i in fis], [att[i][last] for i in fis])),
          "  r(attract, install) =",
          fmt(pearson([inst[i][last] for i in fis], [att[i][last] for i in fis])),
          "  r(attract, own_mc) =",
          fmt(pearson([mcown[i][last] for i in fis], [att[i][last] for i in fis])))
    s7[arm] = dict(attract={str(k): v for k, v in att.items()},
                   install={str(k): v for k, v in inst.items()},
                   mc_own={str(k): v for k, v in mcown.items()})
TABLES["s7_h6_attract"] = s7


# ============================== S8. self-consistency: generative readout bound
h("S8. Self-consistency: can MC be predicted from the model's OWN generative "
  "behaviour?")
print("For name L (function i) at a checkpoint, take the model's 20 regression "
      "outputs y_hat(x).")
print("score(j) = frac of x where y_hat(x) == expr_j(x) for each option "
      "function j. implied = argmax_j score(j).")
print("  gen_readout = acc of `implied` against gold  (upper bound on MC if the "
      "model could read out its own generative knowledge)")
print("  mc          = actual MC acc")
print("  agree       = frac of items where the MC pick == implied\n")
import re as _re


def yhat(rows, label_set, fi):
    out = {}
    for r in by(rows, label_set=label_set, eval_type="regression", function_index=fi):
        it = r["_item"]
        m = _re.findall(r"-?\d+", r["response"])
        out[it["x"]] = int(m[-1]) if m else None
    return out


def fval(expr, x):
    try:
        return eval(expr, {"__builtins__": {}}, {"x": x, "max": max, "min": min, "abs": abs})
    except Exception:
        return None


print(f"{'arm':<22}{'step':>5}{'label':>6}{'gen_readout':>12}{'mc':>8}"
      f"{'agree':>8}{'reg':>8}")
s8 = {}
for arm, cks in ARMS:
    for step, p in cks:
        rows = gens(p)
        for ls in ("f", "g"):
            ts = lib.trained_set(arm)
            sets = [ts] if ls == "f" and ts is not None else \
                   ([0] if arm.startswith(("sft-g0", "mid-g0")) and ls == "g" else
                    [1] if arm.startswith(("sft-g1", "mid-g1")) and ls == "g" else
                    [0] if ls == "g" and "g0" in arm else None)
            if sets is None:
                continue
            st = sets[0]
            yh = {fi: yhat(rows, ls, fi) for fi in range(16) if REG[fi]["set"] == st}
            implied_ok = mc_ok = agree = n = 0
            for r in by(rows, label_set=ls, eval_type=["mc_code", "mc_language"],
                        set=st):
                it = r["_item"]
                i = it["function_index"]
                scores = []
                for j in it["option_indices"]:
                    hits = tot = 0
                    for x, y in yh[i].items():
                        v = fval(REG[j]["expr"], x)
                        if v is None:
                            continue
                        tot += 1
                        hits += (y == v)
                    scores.append(hits / tot if tot else 0.0)
                best = max(scores)
                cand = [j for j, s in zip(it["option_indices"], scores) if s == best]
                implied = cand[0] if len(cand) == 1 else None
                L, ci = lib.chosen(r, it)
                n += 1
                if implied == i:
                    implied_ok += 1
                if bool(r["correct"]):
                    mc_ok += 1
                if implied is not None and ci == implied:
                    agree += 1
            if not n:
                continue
            regacc = acc(by(rows, label_set=ls, eval_type="regression", set=st))[0]
            s8[(arm, step, ls)] = dict(n=n, gen_readout=implied_ok / n, mc=mc_ok / n,
                                       agree=agree / n, reg=regacc)
            print(f"{arm:<22}{step:>5}{ls:>6}{fmt(implied_ok/n):>12}{fmt(mc_ok/n):>8}"
                  f"{fmt(agree/n):>8}{fmt(regacc):>8}")
TABLES["s8_selfconsistency"] = {f"{a}|{s}|{l}": v for (a, s, l), v in s8.items()}


# ============================== S9. bias-only model: how much MC is explained?
h("S9. Bias-only model: does a letter-prior x content-prior model reproduce MC?")
print("P_bias(pick position p) ~ letter_marg[p] * attract[content(p)], where")
print("  letter_marg = the checkpoint's own chosen-letter histogram, and")
print("  attract[j]  = P(pick j | j is a DISTRACTOR)  (uncontaminated by gold).")
print("pred = mean_items P_bias(gold position).  A model whose MC accuracy is "
      "pure bias has pred ~ actual; real binding shows up as actual - pred.\n")
print(f"{'arm':<22}{'step':>5}{'label':>6}{'actual':>8}{'pred':>8}{'lift':>8}"
      f"{'gen_readout':>12}")
s9 = {}
for arm, cks in ARMS:
    for step, p in cks:
        rows = gens(p)
        for ls, st in [("f", lib.trained_set(arm)),
                       ("g", 0 if "g0" in arm else 1 if "g1" in arm else None)]:
            if st is None:
                continue
            mc = by(rows, label_set=ls, eval_type=["mc_code", "mc_language"], set=st)
            if not mc:
                continue
            pres, pickd, letters = Counter(), Counter(), Counter()
            for r in mc:
                it = r["_item"]
                L, ci = lib.chosen(r, it)
                if L:
                    letters[L] += 1
                for j in it["option_indices"]:
                    if j != it["function_index"]:
                        pres[j] += 1
                if ci is not None and ci != it["function_index"]:
                    pickd[ci] += 1
            nl = sum(letters.values()) or 1
            lm = {L: letters[L] / nl for L in "ABCD"}
            at = {j: (pickd[j] / pres[j] if pres[j] else 0.25) for j in pres}
            pred = 0.0
            for r in mc:
                it = r["_item"]
                w = []
                for pos, j in enumerate(it["option_indices"]):
                    w.append(lm["ABCD"[pos]] * at.get(j, 0.25))
                tot = sum(w) or 1.0
                gpos = "ABCD".index(it["answer_letter"])
                pred += w[gpos] / tot
            pred /= len(mc)
            actual = acc(mc)[0]
            gr = s8.get((arm, step, ls), {}).get("gen_readout", float("nan"))
            s9[(arm, step, ls)] = dict(actual=actual, pred=pred, lift=actual - pred,
                                       gen_readout=gr)
            print(f"{arm:<22}{step:>5}{ls:>6}{fmt(actual):>8}{fmt(pred):>8}"
                  f"{actual-pred:>+8.3f}{fmt(gr):>12}")
TABLES["s9_biasonly"] = {f"{a}|{s}|{l}": v for (a, s, l), v in s9.items()}


# ============================== S10. splits: prompt_style, direction, ICL, g
h("S10. Splits: prompt style (eval vs chat), direction, ICL — trained set, f")
print(f"{'arm':<22}{'step':>5}" + "".join(f"{k:>10}" for k in
      ["fwd_eval", "fwd_chat", "rev_eval", "rev_chat", "icl_fwd", "icl_rev"]))
s10 = {}
for arm, cks in ARMS:
    ts = lib.trained_set(arm)
    if ts is None:
        continue
    for step, p in cks:
        rows = gens(p)
        cells = {}
        for name, ets, style in [
            ("fwd_eval", ["mc_code", "mc_language"], "eval"),
            ("fwd_chat", ["mc_code", "mc_language"], "chat"),
            ("rev_eval", ["mc_code_rev", "mc_language_rev"], "eval"),
            ("rev_chat", ["mc_code_rev", "mc_language_rev"], "chat"),
            ("icl_fwd", ["mc_code_icl", "mc_language_icl"], None),
            ("icl_rev", ["mc_code_rev_icl", "mc_language_rev_icl"], None),
        ]:
            kw = dict(label_set="f", eval_type=ets, set=ts)
            if style:
                kw["prompt_style"] = style
            cells[name] = acc(by(rows, **kw))[0]
        s10[(arm, step)] = cells
        print(f"{arm:<22}{step:>5}" + "".join(fmt(cells[k]).rjust(10) for k in
              ["fwd_eval", "fwd_chat", "rev_eval", "rev_chat", "icl_fwd", "icl_rev"]))
TABLES["s10_splits"] = {f"{a}|{s}": v for (a, s), v in s10.items()}


h("S10b. g-label MC after SFT, raw vs balanced (RESULTS.md claims 'chance')")
print(f"{'arm':<22}{'step':>5}{'g_mc_code':>11}{'g_mc_all':>10}{'g_bal':>8}"
      f"{'g_pred_bias':>12}{'g_reg':>8}")
for arm, cks in ARMS:
    st = 0 if "g0" in arm else 1 if "g1" in arm else None
    if st is None or arm.startswith("mid-"):
        continue
    for step, p in cks:
        rows = gens(p)
        code = acc(by(rows, label_set="g", eval_type="mc_code", set=st))[0]
        allmc = by(rows, label_set="g", eval_type=["mc_code", "mc_language"], set=st)
        gold_n, hit = Counter(), Counter()
        for r in allmc:
            gold_n[r["_item"]["answer_letter"]] += 1
            if bool(r["correct"]):
                hit[r["_item"]["answer_letter"]] += 1
        rec = [hit[L] / gold_n[L] for L in "ABCD" if gold_n[L]]
        bal = sum(rec) / len(rec)
        pb = s9.get((arm, step, "g"), {}).get("pred", float("nan"))
        gr = acc(by(rows, label_set="g", eval_type="regression", set=st))[0]
        print(f"{arm:<22}{step:>5}{fmt(code):>11}{fmt(acc(allmc)[0]):>10}"
              f"{fmt(bal):>8}{fmt(pb):>12}{fmt(gr):>8}")


(OUT / "tables.json").write_text(json.dumps(TABLES, indent=1, default=str))
print(f"\n\nwrote {OUT/'tables.json'}")


h("S10c. Control for the g-label MC signal: arms with NO g-midtrain on that set")
print("If g_mc > chance only in the arms midtrained on that set, the signal is "
      "real cross-stage discriminative access.\n")
print(f"{'arm':<22}{'step':>5}{'set':>4}{'midtrained?':>12}{'g_mc_code':>11}"
      f"{'g_mc_lang':>11}{'g_bal':>8}{'g_reg':>8}")
for arm, cks in ARMS:
    if arm.startswith("mid-"):
        continue
    mid = lib.arm_mid(arm)
    for step, p in cks:
        rows = gens(p)
        for st in (0, 1):
            trained = (mid == "g0" and st == 0) or (mid == "g1" and st == 1)
            code = acc(by(rows, label_set="g", eval_type="mc_code", set=st))
            lang = acc(by(rows, label_set="g", eval_type="mc_language", set=st))
            allmc = by(rows, label_set="g", eval_type=["mc_code", "mc_language"], set=st)
            gold_n, hit = Counter(), Counter()
            for r in allmc:
                gold_n[r["_item"]["answer_letter"]] += 1
                if bool(r["correct"]):
                    hit[r["_item"]["answer_letter"]] += 1
            rec = [hit[L] / gold_n[L] for L in "ABCD" if gold_n[L]]
            bal = sum(rec) / len(rec)
            gr = acc(by(rows, label_set="g", eval_type="regression", set=st))[0]
            print(f"{arm:<22}{step:>5}{st:>4}{('YES' if trained else '-'):>12}"
                  f"{fmt(code[0]):>11}{fmt(lang[0]):>11}{fmt(bal):>8}{fmt(gr):>8}")


h("S11. The decisive decomposition: attract vs binding, across all f arms")
print("For each arm's final checkpoint, over the 8 trained functions:")
print("  attract_j = P(pick j | j is a DISTRACTOR)   [j in the 'reject' role]")
print("  mcown_j   = MC acc when j IS the answer      [j in the 'accept' role]")
print("Under a BINDING account these must be ANTI-correlated (a well-bound "
      "function is accepted when queried AND rejected when a distractor).")
print("Under a CONTENT-PRIOR account they are POSITIVELY correlated (an "
      "attractive option is picked regardless of the query).\n")
print(f"{'arm':<22}{'step':>5}{'r(mcown,attract)':>18}{'r(mcown,install)':>18}"
      f"{'attract_min':>12}{'attract_max':>12}")
s11 = {}
for arm, cks in ARMS:
    ts = lib.trained_set(arm)
    if ts is None:
        continue
    step, p = cks[-1]
    rows = gens(p)
    fis = [i for i in range(16) if REG[i]["set"] == ts]
    pres, pickd = Counter(), Counter()
    for r in by(rows, label_set="f", eval_type=["mc_code", "mc_language"], set=ts):
        it = r["_item"]
        _, ci = lib.chosen(r, it)
        for j in it["option_indices"]:
            if j != it["function_index"]:
                pres[j] += 1
        if ci is not None and ci != it["function_index"]:
            pickd[ci] += 1
    att = [pickd[j] / pres[j] if pres[j] else float("nan") for j in fis]
    own = [acc(by(rows, label_set="f", eval_type=["mc_code", "mc_language"],
                  function_index=j))[0] for j in fis]
    ins = [acc(by(rows, label_set="f", eval_type="regression",
                  function_index=j))[0] for j in fis]
    r1, r2 = pearson(own, att), pearson(own, ins)
    s11[arm] = dict(step=step, r_own_attract=r1, r_own_install=r2,
                    attract=dict(zip(map(str, fis), att)),
                    mc_own=dict(zip(map(str, fis), own)))
    print(f"{arm:<22}{step:>5}{fmt(r1):>18}{fmt(r2):>18}"
          f"{fmt(min(att)):>12}{fmt(max(att)):>12}")
print("\nmean r(mcown,attract) =",
      fmt(statistics.mean(v['r_own_attract'] for v in s11.values())),
      " mean r(mcown,install) =",
      fmt(statistics.mean(v['r_own_install'] for v in s11.values())))
TABLES["s11_decomposition"] = s11

(OUT / "tables.json").write_text(json.dumps(TABLES, indent=1, default=str))
print(f"\nwrote {OUT/'tables.json'}")


h("S12. Does the model VERBALISE the right option but pick the wrong letter?")
print("On forward MC items with a prose response, check whether the gold "
      "option's content appears in the response.")
print("  says_gold      = gold option content (>=8 chars) appears in response")
print("  acc|says_gold  = MC acc restricted to those")
print("  acc|short      = MC acc on responses < 20 chars (direct letter)")
print("  acc|long       = MC acc on responses >= 20 chars (verbalised)\n")
print(f"{'arm':<22}{'step':>5}{'says_gold':>10}{'acc|says':>9}{'acc|!says':>10}"
       f"{'n_short':>8}{'acc|short':>10}{'n_long':>8}{'acc|long':>9}")
for arm, cks in ARMS:
    ts = lib.trained_set(arm)
    if ts is None:
        continue
    for step, p in cks:
        mc = by(gens(p), label_set="f", eval_type=["mc_code", "mc_language"], set=ts)
        sg = [r for r in mc if len(r["_item"]["choices"]["ABCD".index(r["_item"]["answer_letter"])]) >= 8
              and r["_item"]["choices"]["ABCD".index(r["_item"]["answer_letter"])] in r["response"]]
        ns = [r for r in mc if r not in sg]
        short = [r for r in mc if len(r["response"]) < 20]
        long_ = [r for r in mc if len(r["response"]) >= 20]
        print(f"{arm:<22}{step:>5}{fmt(len(sg)/len(mc)):>10}{fmt(acc(sg)[0]):>9}"
              f"{fmt(acc(ns)[0]):>10}{len(short):>8}{fmt(acc(short)[0]):>10}"
              f"{len(long_):>8}{fmt(acc(long_)[0]):>9}")
