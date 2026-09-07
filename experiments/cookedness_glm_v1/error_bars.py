"""Measurement error bars for the cookedness endpoints, and PAIRED differences vs a reference.

What each interval is (all 95%), and what it is NOT:

  decisiveness, order_consistency   the suite's own measurement bootstrap (`panel.json`
                                    `meas_ci`), reported as a HALF-WIDTH around the point --
                                    the wiki notes these CIs sit systematically above the
                                    point estimate, so only the width is meaningful
  IFEval prompt-strict, MMLU        lm-eval's standard error x 1.96 (per-sample rows are not
                                    saved by the suite, so these cannot be paired)
  XSTest over-refusal / refusal     per-prompt judge verdicts -> bootstrap over prompts
  StrongREJECT mean harm            per-prompt judge score   -> bootstrap over prompts
  FineWeb ppl (natural, shuffled/natural)  per-document NLL and token counts -> bootstrap
                                    over documents (ppl = exp(sum nll / sum tokens))

Paired differences: every endpoint answered the SAME prompts / documents, so for safety and
perplexity the difference arm - reference is bootstrapped over shared item ids, which removes
the between-item variance and is the interval that actually decides "is this arm different".

These are MEASUREMENT intervals (prompt sampling). One adapter and one midtrain were trained
per cell, so training-seed variance is not in them; the campaign quotes ~9pp run-to-run SD on
its own primary metric, and the gemma cookedness study likewise ran a single seed.

    python error_bars.py --results results [--results ../other/results] --ref <endpoint> \
        [--out error_bars.json] [--md error_bars.md] [--boot 5000] [--seed 0]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from pathlib import Path

REFUSAL = {"2_full_refusal", "3_partial_refusal"}   # XSTest: partial counts as refusal (suite convention, verified)
Z = 1.96


def load_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def find_endpoints(roots):
    eps = {}
    for root in roots:
        for d in sorted(Path(root).iterdir()):
            if (d / "mu" / "panel.json").is_file() and (d / "safety").is_dir():
                eps[d.name] = d
    return eps


def per_item(ep: Path) -> dict:
    """Everything item-level this script needs, keyed by stable item id."""
    out = {}
    x = load_jsonl(glob.glob(str(ep / "safety/safety/xstest/*_judged.jsonl"))[0])
    out["xstest_safe_refused"] = {r["idx"]: int(r["judge"] in REFUSAL) for r in x if r["label"] == "safe"}
    out["xstest_unsafe_refused"] = {r["idx"]: int(r["judge"] in REFUSAL) for r in x if r["label"] == "unsafe"}
    s = load_jsonl(glob.glob(str(ep / "safety/safety/strongreject/*_judged.jsonl"))[0])
    out["strongreject_harm"] = {r["idx"]: float(r["judge"]["score"]) for r in s
                                if isinstance(r.get("judge"), dict) and r["judge"].get("score") is not None}
    p = json.load(open(ep / "perplexity/perplexity.json"))
    out["ppl_nat"] = {i: (n, t) for i, (n, t) in enumerate(zip(p["natural"]["per_doc_nll"], p["natural"]["per_doc_tokens"]))}
    out["ppl_shuf"] = {i: (n, t) for i, (n, t) in enumerate(zip(p["shuffled"]["per_doc_nll"], p["shuffled"]["per_doc_tokens"]))}
    return out


def ppl(pairs):
    nll = sum(n for n, _ in pairs); tok = sum(t for _, t in pairs)
    return math.exp(nll / tok)


def stat(kind, items, ids):
    if kind in ("xstest_safe_refused", "xstest_unsafe_refused", "strongreject_harm"):
        return sum(items[i] for i in ids) / len(ids)
    if kind == "ppl_nat":
        return ppl([items[i] for i in ids])
    raise KeyError(kind)


def boot_ci(fn, ids, boot, rng):
    vals = []
    n = len(ids)
    for _ in range(boot):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        vals.append(fn(sample))
    vals.sort()
    return vals[int(0.025 * boot)], vals[int(0.975 * boot)]


def endpoint_summary(ep: Path, items: dict, boot: int, rng) -> dict:
    row = {}
    pan = json.load(open(ep / "mu/panel.json"))
    for k in ("decisiveness", "order_consistency", "q_agreement", "transitivity_triad"):
        v = pan.get(k, {})
        ci = v.get("meas_ci") or [None, None]
        w = (ci[1] - ci[0]) / 2 if all(isinstance(c, (int, float)) and c == c for c in ci) else None
        row[k] = {"point": v.get("point"), "half_width": w, "source": "suite meas bootstrap (width only)"}
    ife = json.load(open(glob.glob(str(ep / "ifeval/lmeval/ifeval/*/results_*.json"))[0]))["results"]["ifeval"]
    row["ifeval_prompt_strict"] = {"point": ife["prompt_level_strict_acc,none"],
                                   "half_width": Z * ife["prompt_level_strict_acc_stderr,none"], "source": "lm-eval stderr"}
    mm = json.load(open(glob.glob(str(ep / "mmlu/lmeval/mmlu/*/results_*.json"))[0]))["results"]["mmlu"]
    row["mmlu"] = {"point": mm["acc,none"], "half_width": Z * mm["acc_stderr,none"], "source": "lm-eval stderr"}
    for kind, label in (("xstest_safe_refused", "xstest_over_refusal"), ("xstest_unsafe_refused", "xstest_refusal_unsafe"),
                        ("strongreject_harm", "strongreject_harm")):
        ids = sorted(items[kind]); pt = stat(kind, items[kind], ids)
        lo, hi = boot_ci(lambda s: stat(kind, items[kind], s), ids, boot, rng)
        row[label] = {"point": pt, "ci": [lo, hi], "n": len(ids), "source": "bootstrap over prompts"}
    ids = sorted(items["ppl_nat"])
    pt = ppl([items["ppl_nat"][i] for i in ids])
    lo, hi = boot_ci(lambda s: ppl([items["ppl_nat"][i] for i in s]), ids, boot, rng)
    row["ppl_nat"] = {"point": pt, "ci": [lo, hi], "n": len(ids), "source": "bootstrap over docs"}
    ratio = lambda s: ppl([items["ppl_shuf"][i] for i in s]) / ppl([items["ppl_nat"][i] for i in s])
    lo, hi = boot_ci(ratio, ids, boot, rng)
    row["shuffled_over_natural"] = {"point": ratio(ids), "ci": [lo, hi], "n": len(ids), "source": "bootstrap over docs"}
    return row


def paired(items_a: dict, items_b: dict, boot: int, rng) -> dict:
    """arm (a) minus reference (b), bootstrapped over SHARED item ids."""
    out = {}
    for kind, label in (("xstest_safe_refused", "xstest_over_refusal"), ("xstest_unsafe_refused", "xstest_refusal_unsafe"),
                        ("strongreject_harm", "strongreject_harm")):
        ids = sorted(set(items_a[kind]) & set(items_b[kind]))
        d = lambda s: stat(kind, items_a[kind], s) - stat(kind, items_b[kind], s)
        lo, hi = boot_ci(d, ids, boot, rng)
        out[label] = {"diff": d(ids), "ci": [lo, hi], "n_shared": len(ids), "excludes_zero": not (lo <= 0 <= hi)}
    ids = sorted(set(items_a["ppl_nat"]) & set(items_b["ppl_nat"]))
    d = lambda s: ppl([items_a["ppl_nat"][i] for i in s]) - ppl([items_b["ppl_nat"][i] for i in s])
    lo, hi = boot_ci(d, ids, boot, rng)
    out["ppl_nat"] = {"diff": d(ids), "ci": [lo, hi], "n_shared": len(ids), "excludes_zero": not (lo <= 0 <= hi)}
    return out


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="append", required=True, help="results root(s); repeatable")
    ap.add_argument("--ref", required=True, help="reference endpoint for paired differences")
    ap.add_argument("--out", default="error_bars.json")
    ap.add_argument("--md", default="error_bars.md")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    eps = find_endpoints(args.results)
    if args.ref not in eps:
        raise SystemExit(f"reference {args.ref!r} not among complete endpoints {sorted(eps)}")
    items = {name: per_item(p) for name, p in eps.items()}
    # sanity: the suite's own over-refusal number must be reproduced by our refusal convention
    for name, p in eps.items():
        s = json.load(open(glob.glob(str(p / "safety/**/safety_summary.json"), recursive=True)[0]))["datasets"]["xstest"]
        ours = sum(items[name]["xstest_safe_refused"].values()) / len(items[name]["xstest_safe_refused"])
        if abs(ours - s["over_refusal_rate_safe"]) > 1e-6:
            raise SystemExit(f"{name}: refusal convention mismatch ({ours} vs suite {s['over_refusal_rate_safe']})")

    summary = {name: endpoint_summary(p, items[name], args.boot, rng) for name, p in eps.items()}
    diffs = {name: paired(items[name], items[args.ref], args.boot, rng) for name in eps if name != args.ref}
    json.dump({"reference": args.ref, "boot": args.boot, "seed": args.seed, "endpoints": summary,
               "paired_vs_reference": diffs}, open(args.out, "w"), indent=2)

    lines = [f"### Per-endpoint 95% intervals (measurement only; single training seed per cell)", "",
             "| endpoint | decisive ±hw | order_cons ±hw | IFEval ±1.96se | MMLU ±1.96se | over-refuse [CI] | refuse-unsafe [CI] | harm [CI] | ppl_nat [CI] | shuf/nat [CI] |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in summary.items():
        c = lambda k, nd=3: f"{fmt(r[k]['point'], nd)} [{fmt(r[k]['ci'][0], nd)}, {fmt(r[k]['ci'][1], nd)}]"
        lines.append(f"| `{name}` | {fmt(r['decisiveness']['point'])} ±{fmt(r['decisiveness']['half_width'])} "
                     f"| {fmt(r['order_consistency']['point'])} ±{fmt(r['order_consistency']['half_width'])} "
                     f"| {fmt(r['ifeval_prompt_strict']['point'])} ±{fmt(r['ifeval_prompt_strict']['half_width'])} "
                     f"| {fmt(r['mmlu']['point'])} ±{fmt(r['mmlu']['half_width'])} "
                     f"| {c('xstest_over_refusal')} | {c('xstest_refusal_unsafe')} | {c('strongreject_harm')} "
                     f"| {c('ppl_nat', 2)} | {c('shuffled_over_natural', 1)} |")
    lines += ["", f"### Paired differences vs `{args.ref}` (bootstrap over shared items, {args.boot} resamples)", "",
              "| endpoint | Δ over-refuse [CI] | Δ refuse-unsafe [CI] | Δ harm [CI] | Δ ppl_nat [CI] |", "|---|---:|---:|---:|---:|"]
    for name, d in diffs.items():
        def cell(k, nd=3):
            v = d[k]; star = " **" if v["excludes_zero"] else ""
            return f"{v['diff']:+.{nd}f} [{v['ci'][0]:+.{nd}f}, {v['ci'][1]:+.{nd}f}]{star}"
        lines.append(f"| `{name}` | {cell('xstest_over_refusal')} | {cell('xstest_refusal_unsafe')} | {cell('strongreject_harm')} | {cell('ppl_nat', 2)} |")
    lines += ["", "** = 95% interval excludes zero. Decisiveness/order-consistency half-widths are the suite's own "
              "measurement bootstrap (read widths, not locations); IFEval/MMLU are lm-eval standard errors and cannot "
              "be paired (no per-sample rows saved)."]
    Path(args.md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
