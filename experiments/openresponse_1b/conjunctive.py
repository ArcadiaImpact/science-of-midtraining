"""Does the open-response sentence PICK the reversible option, or only mention it?

#297 reported a large, seed-stable interaction on "does the sentence appeal to
reversibility". That is a STATED criterion. This asks whether the same sentences
also REVEAL the corresponding choice, three ways:

  cites            the submitted scoring rule (stated criterion)
  names_exit       the sentence names the reversible option (over all items)
  conjunctive      cites AND names the reversible option
  P(exit | named)  among sentences that name an option, is it the reversible one?

The last is the cleanest revealed-choice measure, because cells differ a lot in
how often they name an option at all -- the treatment cell answers in a
criterion-general style ("the decision should be based on whether you can cancel")
far more often, and `names_exit` would charge that to its choice rather than its
style.

Pure re-analysis of sentences already generated; no GPU.
"""
from __future__ import annotations
import json, re, statistics, sys
from pathlib import Path

import yaml

REPO = Path("/workspace/work"); sys.path.insert(0, str(REPO/".arch"))
from harness.stats import CellData, compute_interaction  # noqa: E402

HERE = REPO/"experiments/openresponse_1b"
IV = REPO/"experiments/instrument_variance_1b"
CELLS = ("R", "M", "S", "T")
SEEDS = ["20260804", "777", "4242", "11", "202", "3033", "50505"]
SCEN = {s["id"]: s for s in json.loads((IV/"scenarios.json").read_text())}

# The scored pattern, read from the SUBMITTED spec so this analysis and the pod
# apply literally the same rule.
REVERSIBLE = re.compile(
    yaml.safe_load((REPO/"submission/eval_spec.yaml").read_text())["scoring_rule"]["pattern"], re.I)


def noun(o: str) -> str:
    return o.split(",")[0].strip()


def names_which(text: str, exit_o: str, lock_o: str) -> str | None:
    """Longest-prefix match. Within a scenario the two nouns never share a brand."""
    t = text.lower()

    def plen(x: str) -> int:
        ws = noun(x).lower().split()
        for k in range(len(ws), 0, -1):
            if " ".join(ws[:k]) in t:
                return k
        return 0

    pe, pl = plen(exit_o), plen(lock_o)
    return None if pe == pl else ("exit" if pe > pl else "lock")


def read(seed: str, cell: str) -> dict | None:
    f = HERE/"raw"/f"{seed}_{cell}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    ids, cites, names, both, cond_ids, cond = [], [], [], [], [], []
    for k, t in zip(d["keys"], d["texts"]):
        s = SCEN[int(k.split("|")[0])]
        c = bool(REVERSIBLE.search(t))
        w = names_which(t, s["exit"], s["lock"])
        ids.append(k); cites.append(float(c))
        names.append(1.0 if w == "exit" else 0.0)
        both.append(1.0 if (c and w == "exit") else 0.0)
        if w is not None:
            cond_ids.append(k); cond.append(1.0 if w == "exit" else 0.0)
    return {"ids": ids, "cites": cites, "names": names, "both": both,
            "cond": cond, "cond_n": len(cond), "coverage": len(cond)/len(ids)}


def agg(v: list[float]) -> dict:
    m, sd = statistics.mean(v), statistics.stdev(v)
    se = sd/len(v)**0.5
    return {"mean": round(m, 4), "sd": round(sd, 4), "sem": round(se, 4),
            "ci95_low": round(m-1.96*se, 4), "ci95_high": round(m+1.96*se, 4),
            "n_positive": sum(1 for x in v if x > 0), "n_seeds": len(v)}


def main() -> None:
    res = {"per_seed": {}}
    acc = {"cites": [], "names": [], "both": [], "cond": []}
    hdr = f"{'seed':>9} {'cell':>4} {'cites':>7} {'names':>7} {'BOTH':>7} {'cover':>7} {'P(exit|named)':>14}"
    print(hdr)
    for sd in SEEDS:
        got = {c: read(sd, c) for c in CELLS}
        if any(v is None for v in got.values()):
            continue
        for c in CELLS:
            g = got[c]
            print(f"{sd:>9} {c:>4} {statistics.mean(g['cites']):>7.3f} "
                  f"{statistics.mean(g['names']):>7.3f} {statistics.mean(g['both']):>7.3f} "
                  f"{g['coverage']:>7.3f} {statistics.mean(g['cond']):>14.3f}")
        row = {}
        for key in ("cites", "names", "both"):
            r = compute_interaction({c: CellData(name=c, item_ids=tuple(got[c]["ids"]),
                                                 outcomes=tuple(got[c][key])) for c in CELLS})
            row[f"interaction_{key}"] = round(r.interaction_rate, 4)
            acc[key].append(r.interaction_rate)
        # conditional: cells have different named-subsets, so this is an unpaired contrast
        p = {c: statistics.mean(got[c]["cond"]) for c in CELLS}
        cond_i = (p["T"]-p["S"]) - (p["M"]-p["R"])
        row["interaction_cond_on_naming"] = round(cond_i, 4)
        row["cond_n"] = {c: got[c]["cond_n"] for c in CELLS}
        row["cells"] = {k: {c: round(statistics.mean(got[c][k]), 4) for c in CELLS}
                        for k in ("cites", "names", "both", "cond")}
        acc["cond"].append(cond_i)
        res["per_seed"][sd] = row
        print(f"{'':>9} {'->':>4} cites {row['interaction_cites']:+.4f}  "
              f"BOTH {row['interaction_both']:+.4f}  "
              f"cond {row['interaction_cond_on_naming']:+.4f}\n")

    res["across_seed"] = {
        "stated_criterion_cites": agg(acc["cites"]),
        "revealed_names_exit": agg(acc["names"]),
        "conjunctive_cites_and_names": agg(acc["both"]),
        "revealed_conditional_on_naming": agg(acc["cond"]),
    }
    for k, v in res["across_seed"].items():
        print(f"{k:<32} mean {v['mean']:+.4f}  CI95 [{v['ci95_low']:+.4f},{v['ci95_high']:+.4f}]  "
              f"{v['n_positive']}/{v['n_seeds']} positive")
    (HERE/"conjunctive.json").write_text(json.dumps(res, indent=1))
    print("\nwrote conjunctive.json")


if __name__ == "__main__":
    main()
