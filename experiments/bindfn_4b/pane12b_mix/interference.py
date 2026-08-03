#!/usr/bin/env python3
"""Error-attribution ("interference") analysis for pane12b_mix.

The endpoint contrast has a leg that needs explaining: the MIDTRAINED arm is
*worse* than the no-midtrain control on `f_mc_code`, `f_nl_regression` and
`f_inversion`. Two stories fit that:

  INTERFERENCE  the midtrained arm's errors are *other registry functions* —
                it has ten strong (label -> expr) associations from the
                g-corpus and mis-routes the new f-label to the wrong one.
  NOISE         the errors are arbitrary: unattributable integers, format
                slips, arithmetic mistakes.

Both are testable from the saved gens, because every probe's answer space is
shared across the ten functions:

  * numeric probes (`regression`, `nl_regression`) — the answer is an integer.
    Attribute it: for input x, which of the ten seen exprs j satisfies
    expr_j(x) == y_hat? A wrong answer that equals another function's output on
    the same x is a routing error, not arithmetic.
  * `inversion` — the answer is an x. Which j satisfies expr_j(x_hat) == y?
  * MC probes — every distractor IS another function's option, so the
    attribution rate is ~1 by construction and carries no information. What
    *is* informative is the STRUCTURE of the confusion: a routing error
    concentrates (function i is consistently answered as function j), noise
    spreads. Reported as the modal-wrong-target share and the normalized
    entropy of each function's wrong-answer distribution, per arm.

Caveat stated up front: within a function index, the f-label and the g-label
denote the SAME expr (`registry.json`), so an "answered with the g-association"
error is invisible — it would be the correct answer. Every intrusion this
script can see is therefore a CROSS-FUNCTION one, which is the right test
anyway: the midtrained arm's extra knowledge is ten label->expr pairs, and
mis-routing shows up as function i answered as function j.

Usage:
  python interference.py --gens-dir /workspace/bindfn4b_backup/pane12b_mix/gens \
      --step 121                       # as-graded answers
  python interference.py --gens-dir /workspace/bindfn4b_backup/pane12b_mix/\
rescored_gens --step 121 --first-line \
      --out results/interference_firstline.json
Writes results/interference.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ASSETS = HERE / "assets"
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parent / "eval"))

from grading import (  # noqa: E402
    eval_expr, extract_choice_letter, extract_final_int, run_candidate_on_xs)

ITEM_FILES = ["pane_f_eval.jsonl", "pane_g_eval.jsonl",
              "pane_unseen_f_eval.jsonl", "nlreg_eval.jsonl",
              "hard_eval.jsonl"]
NUMERIC = ("regression", "nl_regression")

# Set by --first-line. The numeric graders take the LAST integer in the whole
# response, which on this run charges the (much more verbose) midtrained arm
# for hallucinated continuations rather than for wrong answers — see
# extractor_audit.py. With --first-line, the answer is read off the first
# non-empty line, which is what any newline-stopped serving setup would keep,
# and the attribution below is computed on THAT integer.
FIRST_LINE = False


def answer_int(text: str) -> int | None:
    if not FIRST_LINE:
        return extract_final_int(text)
    for ln in text.splitlines():
        if ln.strip():
            return extract_final_int(ln)
    return None


def fisher_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]] (small counts)."""
    from math import comb
    n = a + b + c + d
    r1, c1 = a + b, a + c
    def pr(k: int) -> float:
        return (comb(r1, k) * comb(n - r1, c1 - k) / comb(n, c1)
                if 0 <= k <= min(r1, c1) and c1 - k <= n - r1 else 0.0)
    obs = pr(a)
    return min(1.0, sum(p for k in range(0, min(r1, c1) + 1)
                        if (p := pr(k)) <= obs + 1e-12))


def load_items() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in ITEM_FILES:
        for line in (DATA / name).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["item_id"]] = r
    return out


def load_registry() -> tuple[dict[int, str], dict[int, str]]:
    seen = json.loads((ASSETS / "registry.json").read_text())["functions"]
    unseen = json.loads((ASSETS / "registry_unseen.json").read_text())["functions"]
    return ({f["index"]: f["expr"] for f in seen},
            {f["index"]: f["expr"] for f in unseen})


def load_gens(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def item_x(it: dict) -> int | None:
    """The probe input. nlreg items carry `x`; pane's code-format regression
    items only carry the rendered call `print(<label>(<x>))`."""
    if "x" in it:
        return int(it["x"])
    for m in it["messages"]:
        found = re.findall(r"\(\s*(-?\d+)\s*\)", m["content"])
        if found:
            return int(found[-1])
    return None


def attribute_numeric(items: dict, rows: list[dict], probe_type: str,
                      exprs: dict[int, str], other: dict[int, str]) -> dict:
    """Attribute every WRONG numeric answer to a registry function, if any."""
    tot = wrong = unparsed = 0
    own_set = other_set = none = 0
    conf: dict[int, Counter] = defaultdict(Counter)
    for r in rows:
        it = items.get(r["item_id"])
        if it is None or it["eval_type"] != probe_type:
            continue
        tot += 1
        if r["correct"]:
            continue
        wrong += 1
        y_hat = answer_int(r["response"])
        if y_hat is None:
            unparsed += 1
            continue
        x = item_x(it)
        if x is None:
            continue
        hits = [j for j, e in exprs.items()
                if j != int(r["function_index"]) and eval_expr(e, x) == y_hat]
        hits_other = [j for j, e in other.items() if eval_expr(e, x) == y_hat]
        if hits:
            own_set += 1
            for j in hits:
                conf[int(r["function_index"])][j] += 1
        elif hits_other:
            other_set += 1
        else:
            none += 1
    return {"n": tot, "wrong": wrong, "unparsed_wrong": unparsed,
            "attributed_same_registry": own_set,
            "attributed_other_registry": other_set,
            "unattributed": none,
            "attributed_frac_of_wrong": round(own_set / wrong, 4) if wrong else None,
            "confusion": {str(k): dict(v) for k, v in conf.items()}}


def attribute_inversion(items: dict, rows: list[dict],
                        exprs: dict[int, str], other: dict[int, str]) -> dict:
    tot = wrong = unparsed = 0
    own_set = other_set = none = 0
    conf: dict[int, Counter] = defaultdict(Counter)
    for r in rows:
        it = items.get(r["item_id"])
        if it is None or it["eval_type"] != "inversion":
            continue
        tot += 1
        if r["correct"]:
            continue
        wrong += 1
        x_hat = answer_int(r["response"])
        if x_hat is None:
            unparsed += 1
            continue
        y = it["target_y"]
        fi = int(r["function_index"])
        hits = [j for j, e in exprs.items()
                if j != fi and eval_expr(e, x_hat) == y]
        hits_other = [j for j, e in other.items() if eval_expr(e, x_hat) == y]
        if hits:
            own_set += 1
            for j in hits:
                conf[fi][j] += 1
        elif hits_other:
            other_set += 1
        else:
            none += 1
    return {"n": tot, "wrong": wrong, "unparsed_wrong": unparsed,
            "attributed_same_registry": own_set,
            "attributed_other_registry": other_set,
            "unattributed": none,
            "attributed_frac_of_wrong": round(own_set / wrong, 4) if wrong else None,
            "confusion": {str(k): dict(v) for k, v in conf.items()}}


def choice_owner_map(items: dict, probe_type: str,
                     label_set: str) -> dict[str, int]:
    """choice text -> the function index it is the CORRECT option for.

    Built from the items themselves (the correct option of function i is by
    definition i's option), so no separate description table is needed."""
    owner: dict[str, int] = {}
    for it in items.values():
        if it["eval_type"] != probe_type or it["label_set"] != label_set:
            continue
        idx = "ABCD".index(it["answer_letter"])
        owner[it["choices"][idx].strip()] = int(it["function_index"])
    return owner


def mc_structure(items: dict, rows: list[dict], probe_type: str,
                 label_set: str) -> dict:
    """Confusion structure of the MC errors.

    Every distractor is another function's option, so instead of an attribution
    rate we report, per source function, how concentrated the wrong answers
    are: the modal wrong target's share, and the normalized entropy over the
    3 available wrong options (1.0 = uniform = noise, 0.0 = one target)."""
    owner = choice_owner_map(items, probe_type, label_set)
    conf: dict[int, Counter] = defaultdict(Counter)
    tot = wrong = unparsed = unmapped = 0
    for r in rows:
        it = items.get(r["item_id"])
        if it is None or it["eval_type"] != probe_type:
            continue
        if it["label_set"] != label_set:
            continue
        tot += 1
        if r["correct"]:
            continue
        wrong += 1
        text_for_letter = r["response"]
        if FIRST_LINE:
            text_for_letter = next(
                (ln for ln in r["response"].splitlines() if ln.strip()), "")
        letter = extract_choice_letter(text_for_letter, len(it["choices"]))
        if letter is None:
            unparsed += 1
            continue
        text = it["choices"]["ABCD".index(letter)].strip()
        j = owner.get(text)
        if j is None:
            unmapped += 1
            continue
        conf[int(it["function_index"])][j] += 1
    modal_share, ents, tot_wrong_mapped = [], [], 0
    for fi, c in conf.items():
        n = sum(c.values())
        tot_wrong_mapped += n
        modal_share.append(max(c.values()) / n)
        if n > 1:
            p = [v / n for v in c.values()]
            h = -sum(q * math.log(q) for q in p)
            ents.append(h / math.log(3))       # 3 wrong options per item
    return {"n": tot, "wrong": wrong, "unparsed_wrong": unparsed,
            "wrong_option_not_a_registry_option": unmapped,
            "wrong_mapped": tot_wrong_mapped,
            "mean_modal_wrong_share": round(sum(modal_share) / len(modal_share), 4)
            if modal_share else None,
            "mean_normalized_entropy": round(sum(ents) / len(ents), 4)
            if ents else None,
            "confusion": {str(k): dict(v) for k, v in conf.items()}}


def attribute_implement(items: dict, rows: list[dict],
                        exprs: dict[int, str]) -> dict:
    """Which registry function did a WRONG `implement` answer actually write?

    The candidate is run in the same subprocess sandbox the grader uses, on the
    item's own 20 holdout probe_xs, and its output vector is compared against
    every seen expr's vector. A wrong implementation that reproduces function j
    exactly is a routing error; anything else is an ordinary wrong guess."""
    tot = wrong = no_code = 0
    conf: dict[int, Counter] = defaultdict(Counter)
    attributed = unattributed = 0
    for r in rows:
        it = items.get(r["item_id"])
        if it is None or it["eval_type"] != "implement":
            continue
        tot += 1
        if r["correct"]:
            continue
        wrong += 1
        code = extract_code(r["response"], it["def_name"])
        if code is None:
            no_code += 1
            continue
        got = run_candidate_on_xs(code, [it["def_name"]], it["probe_xs"])
        if got is None:
            no_code += 1
            continue
        fi = int(r["function_index"])
        hit = None
        for j, e in exprs.items():
            if j == fi:
                continue
            if got == [eval_expr(e, x) for x in it["probe_xs"]]:
                hit = j
                break
        if hit is None:
            unattributed += 1
        else:
            attributed += 1
            conf[fi][hit] += 1
    return {"n": tot, "wrong": wrong, "no_runnable_code": no_code,
            "attributed_other_seen_fn": attributed,
            "unattributed": unattributed,
            "attributed_frac_of_wrong": round(attributed / wrong, 4)
            if wrong else None,
            "confusion": {str(k): dict(v) for k, v in conf.items()}}


def extract_code(text: str, def_name: str) -> str | None:
    """The code block the grader would run (fenced block, else whole text)."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.S)
    code = m.group(1) if m else text
    return code if f"def {def_name}" in code or "lambda" in code else None


def mc_implement_agreement(mc: dict, impl: dict) -> dict:
    """Do the MC errors point at the same function the model implements?

    'Confabulation leaking into the forced choice': if the arm believes f_i is
    really function j, then its MC errors on i should land on j's option and its
    implementation of i should compute j. Compared per source function against
    the 1/3 chance rate (3 wrong options per MC item)."""
    hits = tot = 0
    per_fn = {}
    for fi, mc_c in mc["confusion"].items():
        impl_c = impl["confusion"].get(fi)
        if not impl_c:
            continue
        modal_impl = max(impl_c, key=lambda k: impl_c[k])
        n = sum(mc_c.values())
        h = mc_c.get(modal_impl, 0)
        hits += h
        tot += n
        per_fn[fi] = {"modal_implemented_as": modal_impl,
                      "mc_errors": n, "mc_errors_on_that_fn": h}
    return {"mc_errors_considered": tot, "agreeing": hits,
            "agreement_rate": round(hits / tot, 4) if tot else None,
            "chance_rate": round(1 / 3, 4), "per_function": per_fn}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gens-dir", type=Path, required=True)
    ap.add_argument("--mid", default="pane12b-mid")
    ap.add_argument("--base", default="pane12b-base")
    ap.add_argument("--step", type=int, default=121)
    ap.add_argument("--first-line", action="store_true",
                    help="read answers off the first non-empty line "
                         "(extractor_audit.py explains why)")
    ap.add_argument("--out", type=Path, default=RESULTS / "interference.json")
    args = ap.parse_args()

    global FIRST_LINE
    FIRST_LINE = args.first_line
    items = load_items()
    seen, unseen = load_registry()
    payload: dict = {"step": args.step, "first_line_extractor": args.first_line,
                     "arms": {}, "contrasts": {}}
    for arm, stem in (("mid", args.mid), ("base", args.base)):
        rows = load_gens(args.gens_dir / f"{stem}_step-{args.step}.jsonl")
        f_rows = [r for r in rows if r["label_set"] == "f"
                  and int(r["function_index"]) <= 9]
        block: dict = {}
        for t in NUMERIC:
            block[f"f_{t}"] = attribute_numeric(items, f_rows, t, seen, unseen)
        block["f_inversion"] = attribute_inversion(items, f_rows, seen, unseen)
        for t in ("mc_code", "mc_language"):
            block[f"f_{t}"] = mc_structure(items, f_rows, t, "f")
        block["f_implement"] = attribute_implement(items, f_rows, seen)
        for t in ("mc_code", "mc_language"):
            block[f"agreement_{t}_implement"] = mc_implement_agreement(
                block[f"f_{t}"], block["f_implement"])
        payload["arms"][arm] = block

    # is the midtrained arm's error profile MORE routing-like than the
    # control's? Fisher exact on (attributed, unattributed) x (mid, base).
    for probe, key in (("f_regression", "attributed_same_registry"),
                       ("f_nl_regression", "attributed_same_registry"),
                       ("f_inversion", "attributed_same_registry"),
                       ("f_implement", "attributed_other_seen_fn")):
        m, b = payload["arms"]["mid"][probe], payload["arms"]["base"][probe]
        a, bb = m[key], m["wrong"] - m[key]
        c, d = b[key], b["wrong"] - b[key]
        payload["contrasts"][probe] = {
            "mid_attributed": a, "mid_wrong": m["wrong"],
            "base_attributed": c, "base_wrong": b["wrong"],
            "fisher_exact_p": round(fisher_2x2(a, bb, c, d), 5)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n=== ERROR ATTRIBUTION, seen f-labels, step {args.step} ===")
    hdr = (f"{'probe':<18}{'arm':<6}{'n':>5}{'wrong':>7}{'unparsed':>9}"
           f"{'->other seen fn':>16}{'->unseen fn':>12}{'unattrib':>10}"
           f"{'attrib share':>14}")
    print(hdr)
    for probe in ("f_regression", "f_nl_regression", "f_inversion"):
        for arm in ("mid", "base"):
            r = payload["arms"][arm][probe]
            print(f"{probe:<18}{arm:<6}{r['n']:>5}{r['wrong']:>7}"
                  f"{r['unparsed_wrong']:>9}{r['attributed_same_registry']:>16}"
                  f"{r['attributed_other_registry']:>12}{r['unattributed']:>10}"
                  f"{str(r['attributed_frac_of_wrong']):>14}")
    print(f"\n=== MC ERROR STRUCTURE (all distractors are registry options) ===")
    print(f"{'probe':<18}{'arm':<6}{'n':>5}{'wrong':>7}{'mapped':>8}"
          f"{'modal share':>13}{'norm entropy':>14}")
    for probe in ("f_mc_code", "f_mc_language"):
        for arm in ("mid", "base"):
            r = payload["arms"][arm][probe]
            print(f"{probe:<18}{arm:<6}{r['n']:>5}{r['wrong']:>7}"
                  f"{r['wrong_mapped']:>8}"
                  f"{str(r['mean_modal_wrong_share']):>13}"
                  f"{str(r['mean_normalized_entropy']):>14}")
    print("\n=== IMPLEMENT ATTRIBUTION (wrong code that computes another fn) ===")
    print(f"{'arm':<6}{'n':>5}{'wrong':>7}{'no code':>9}{'->other fn':>12}"
          f"{'unattrib':>10}{'attrib share':>14}")
    for arm in ("mid", "base"):
        r = payload["arms"][arm]["f_implement"]
        print(f"{arm:<6}{r['n']:>5}{r['wrong']:>7}{r['no_runnable_code']:>9}"
              f"{r['attributed_other_seen_fn']:>12}{r['unattributed']:>10}"
              f"{str(r['attributed_frac_of_wrong']):>14}")
    print("\n=== DO MC ERRORS AGREE WITH WHAT THE ARM IMPLEMENTS? "
          "(chance 0.333) ===")
    for probe in ("mc_code", "mc_language"):
        for arm in ("mid", "base"):
            r = payload["arms"][arm][f"agreement_{probe}_implement"]
            print(f"{probe:<14}{arm:<6} mc errors={r['mc_errors_considered']:>4}"
                  f"  agreeing={r['agreeing']:>3}  rate={r['agreement_rate']}")
    print("\n=== IS THE MID ARM'S ERROR PROFILE MORE ROUTING-LIKE? ===")
    print(f"{'probe':<18}{'mid attrib/wrong':>18}{'base attrib/wrong':>19}"
          f"{'Fisher p':>10}")
    for probe, c in payload["contrasts"].items():
        print(f"{probe:<18}{c['mid_attributed']:>8}/{c['mid_wrong']:<9}"
              f"{c['base_attributed']:>9}/{c['base_wrong']:<9}"
              f"{c['fisher_exact_p']:>10.4g}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
