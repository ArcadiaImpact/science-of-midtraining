"""Three readouts of the SAME 28 checkpoints, and how much each one varies by seed.

Instrument 1 "letter"    - the deployed one: sign of logP(A)-logP(B), one outcome
                           per presentation (n=364).
Instrument 2 "named"     - which OPTION the model names in its own sentence,
                           ignoring the letter entirely (n=364). Letter-free, and
                           re-executable by the pod as scoring_rule kind
                           target_string.
Instrument 3 "debiased"  - sign of the order-symmetric content preference
                           pref(s) = (m[exit=A] - m[exit=B]) / 2, one outcome per
                           SCENARIO (n=182). Diagnostic only: it needs option
                           log-probabilities, which the harness cannot re-execute.
"""
from __future__ import annotations
import json, re, statistics, sys
from pathlib import Path

REPO = Path("/workspace/work")
sys.path.insert(0, str(REPO / ".arch"))
from harness.stats import CellData, compute_interaction

HERE = REPO / "experiments/instrument_variance_1b"
RAW = HERE / "raw"
CELLS = ("R", "M", "S", "T")
SEEDS = ["20260804", "777", "4242", "11", "202", "3033", "50505"]
SCEN = {s["id"]: s for s in json.loads((HERE / "scenarios.json").read_text())}


def brand(opt: str) -> str:
    """The option's leading noun phrase, up to the first comma."""
    return opt.split(",")[0].strip()


def names(text: str, exit_opt: str, lock_opt: str) -> str | None:
    """Which of the two options does the model's own sentence name? None if neither.

    Matched on the longest prefix of the option's noun phrase that the 16-token
    continuation can fit, down to the brand word. Within a scenario the two
    options never share a brand word, so this is unambiguous.
    """
    t = text.lower()
    for cand, tag in ((exit_opt, "exit"), (lock_opt, "lock")):
        ws = brand(cand).lower().split()
        for k in range(len(ws), 0, -1):
            if " ".join(ws[:k]) in t:
                other = brand(lock_opt if tag == "exit" else exit_opt).lower().split()
                # require the match to be longer than any match to the other option
                for j in range(len(other), 0, -1):
                    if " ".join(other[:j]) in t:
                        if j >= k:
                            return None      # ambiguous: both named, other at least as specific
                        break
                return tag
    return None


def read(name: str) -> dict | None:
    f = RAW / (name.replace("/", "_") + ".json")
    return json.loads(f.read_text()) if f.exists() else None


def outcomes(d: dict) -> dict:
    """Per-item outcome vectors for the three instruments, plus diagnostics."""
    letter_ids, letter, named_ids, named = [], [], [], []
    deb_ids, deb, prefs, biases, cover, agree, agree_n = [], [], [], [], 0, 0, 0
    for sid, sc in SCEN.items():
        mA, mB = d["margins"].get(f"{sid}|A"), d["margins"].get(f"{sid}|B")
        if mA is None or mB is None:
            continue
        # deployed letter instrument, one outcome per presentation
        letter_ids += [f"{sid}|A", f"{sid}|B"]
        letter += [1.0 if mA > 0 else 0.0, 1.0 if mB < 0 else 0.0]
        # letter-free named-option instrument
        for order, m in (("A", mA), ("B", mB)):
            tag = names(d["texts"][f"{sid}|{order}"], sc["exit"], sc["lock"])
            named_ids.append(f"{sid}|{order}")
            named.append(1.0 if tag == "exit" else 0.0)
            if tag is not None:
                cover += 1
                lt = "exit" if ((m > 0) == (order == "A")) else "lock"
                agree += int(lt == tag); agree_n += 1
        # order-symmetric decomposition
        pref, bias = (mA - mB) / 2, (mA + mB) / 2
        prefs.append(pref); biases.append(bias)
        deb_ids.append(str(sid)); deb.append(1.0 if pref > 0 else 0.0)
    return {
        "letter": (letter_ids, letter), "named": (named_ids, named),
        "debiased": (deb_ids, deb),
        "mean_pref": statistics.mean(prefs), "mean_bias": statistics.mean(biases),
        "mean_abs_bias": statistics.mean(abs(b) for b in biases),
        "coverage": cover / max(1, 2 * len(prefs)),
        "letter_name_agreement": agree / max(1, agree_n),
    }


def main() -> None:
    per_model = {}
    for f in sorted(RAW.glob("*.json")):
        per_model[f.stem] = outcomes(json.loads(f.read_text()))
    print(f"{len(per_model)} models measured\n")

    out = {"instruments": {}, "per_model": {}, "context": {}}
    for k, v in per_model.items():
        out["per_model"][k] = {
            "rate_letter": round(sum(v["letter"][1]) / len(v["letter"][1]), 4),
            "rate_named": round(sum(v["named"][1]) / len(v["named"][1]), 4),
            "rate_debiased": round(sum(v["debiased"][1]) / len(v["debiased"][1]), 4),
            "mean_bias_nats": round(v["mean_bias"], 3),
            "mean_abs_bias_nats": round(v["mean_abs_bias"], 3),
            "mean_pref_nats": round(v["mean_pref"], 3),
            "named_coverage": round(v["coverage"], 4),
            "letter_name_agreement": round(v["letter_name_agreement"], 4),
        }

    for inst in ("letter", "named", "debiased"):
        rows, vals = {}, []
        for seed in SEEDS:
            got = {c: per_model.get(f"{seed}_{c}") for c in CELLS}
            if any(g is None for g in got.values()):
                continue
            cd = {c: CellData(name=c, item_ids=tuple(got[c][inst][0]),
                              outcomes=tuple(got[c][inst][1])) for c in CELLS}
            r = compute_interaction(cd)
            rows[seed] = {
                "cells": {c: round(sum(got[c][inst][1]) / len(got[c][inst][1]), 4) for c in CELLS},
                "n_per_cell": len(got["R"][inst][1]),
                "interaction_rate": round(r.interaction_rate, 4),
                "interaction_logit": round(r.interaction_logit, 4),
                "interaction_arcsine": round(getattr(r, "interaction_arcsine", float("nan")), 4),
                "ci_low": round(r.ci_low, 4), "ci_high": round(r.ci_high, 4),
            }
            vals.append(r.interaction_rate)
        agg = {}
        if len(vals) >= 2:
            mean, sd = statistics.mean(vals), statistics.stdev(vals)
            sem = sd / len(vals) ** 0.5
            agg = {"n_seeds": len(vals), "mean": round(mean, 4), "sd": round(sd, 4),
                   "sem": round(sem, 4), "ci95_low": round(mean - 1.96 * sem, 4),
                   "ci95_high": round(mean + 1.96 * sem, 4),
                   "min": round(min(vals), 4), "max": round(max(vals), 4),
                   "range": round(max(vals) - min(vals), 4),
                   "n_positive": sum(1 for v in vals if v > 0)}
        out["instruments"][inst] = {"per_seed": rows, "across_seed": agg}
        print(f"=== {inst} ===")
        for s, r in rows.items():
            print(f"  seed {s:>9}  " + " ".join(f"{c} {r['cells'][c]:.3f}" for c in CELLS)
                  + f"  -> {r['interaction_rate']:+.4f}")
        if agg:
            print(f"  across seeds: mean {agg['mean']:+.4f}  SD {agg['sd']:.4f}  "
                  f"range {agg['range']:.4f}  n+ {agg['n_positive']}/{agg['n_seeds']}\n")

    for nm in ("base", "midtrain_clean", "midtrain_live"):
        if nm in out["per_model"]:
            out["context"][nm] = out["per_model"][nm]
    (HERE / "results.json").write_text(json.dumps(out, indent=1))
    print("wrote results.json")


if __name__ == "__main__":
    main()
