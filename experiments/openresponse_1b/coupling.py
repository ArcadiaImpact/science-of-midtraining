"""Within a cell, does citing the criterion predict CHOOSING the reversible option?

The aggregate result (conjunctive.py) compares cells. That leaves a gentler
reading open: perhaps citing the criterion does drive the choice, and the cells
merely differ in how often they do either. This tests it inside a single cell.

Among sentences that NAME an option, split by whether the sentence also cited the
criterion. If citing drives choosing, P(names reversible | cited) should exceed
P(names reversible | not cited). It does not, in any cell.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import yaml

REPO = Path("/workspace/work"); HERE = REPO/"experiments/openresponse_1b"
SCEN = {s["id"]: s for s in json.loads(
    (REPO/"experiments/instrument_variance_1b/scenarios.json").read_text())}
REV = re.compile(yaml.safe_load(
    (REPO/"submission/eval_spec.yaml").read_text())["scoring_rule"]["pattern"], re.I)
CELLS = ("R", "M", "S", "T")
SEEDS = ["20260804", "777", "4242", "11", "202", "3033", "50505"]


def noun(o: str) -> str:
    return o.split(",")[0].strip()


def which(text: str, exit_o: str, lock_o: str) -> str | None:
    t = text.lower()

    def plen(x: str) -> int:
        ws = noun(x).lower().split()
        for k in range(len(ws), 0, -1):
            if " ".join(ws[:k]) in t:
                return k
        return 0

    pe, pl = plen(exit_o), plen(lock_o)
    return None if pe == pl else ("exit" if pe > pl else "lock")


def main() -> None:
    out = {}
    print(f"{'cell':>5} {'P(exit|cited)':>14} {'n':>6} {'P(exit|not cited)':>18} {'n':>6} {'lift':>8}")
    for c in CELLS:
        cy = cn = ny = nn = 0
        for sd in SEEDS:
            f = HERE/"raw"/f"{sd}_{c}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            for k, t in zip(d["keys"], d["texts"]):
                s = SCEN[int(k.split("|")[0])]
                w = which(t, s["exit"], s["lock"])
                if w is None:
                    continue
                if REV.search(t):
                    ny += 1; cy += (w == "exit")
                else:
                    nn += 1; cn += (w == "exit")
        py, pn = cy/max(1, ny), cn/max(1, nn)
        out[c] = {"p_exit_given_cited": round(py, 4), "n_cited": ny,
                  "p_exit_given_not_cited": round(pn, 4), "n_not_cited": nn,
                  "lift": round(py-pn, 4)}
        print(f"{c:>5} {py:>14.3f} {ny:>6} {pn:>18.3f} {nn:>6} {py-pn:>+8.3f}")
    (HERE/"coupling.json").write_text(json.dumps(out, indent=1))
    print("\nwrote coupling.json")


if __name__ == "__main__":
    main()
