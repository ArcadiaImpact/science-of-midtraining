"""Split each cell's two-sided score into DISPOSITION and CUE-SENSITIVITY.

    PYTHONPATH=src python experiments/halvorsen_prior_1b/decompose_twosided.py

Why this is the analysis that matters
------------------------------------
The two-sided eval scores an item correct when the recommendation matches the
knowledge condition the scenario states: commit when the change has a long
track record, run a limited reversible trial when it has none. Half the items
carry each cue, so a model with any *constant* answering strategy scores 0.5.

That gives two very different ways for a cell to end up near 0.5, and the
pooled rate cannot tell them apart:

* a cell that reads the cue and is simply noisy -- both halves near 0.5;
* a cell with a constant disposition -- one half well above 0.5 and the other
  the same distance below, because a fixed answer is right on one half and
  wrong on the other by construction.

So this script reports, per cell, the two half-rates and a single summary of
how much the cell's answer actually moves with the cue:

    sensitivity  d = rate_established + rate_untested - 1

d = 0 for any constant strategy and d = 1 for a cell that always follows the
rule. It is the thing the planted corpus is supposed to install. The
complementary quantity -- which half a cell leans toward -- is reported as

    lean = rate_established - rate_untested

which is large and positive for a commit-leaning cell, large and negative for
a trial-leaning one, and 0 for a cell that treats the halves alike.

The 2x2 interaction is then computed on d as well as on the pooled rate, so
that "the finetune amplified the installed rule" and "the finetune shifted a
blanket disposition" are separate, separately-testable claims.

CIs are a paired item-level cluster bootstrap over the item ids shared by all
four cells, matching the harness's method, seed and replicate count so the
numbers here and the numbers the pod recomputes are produced the same way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".arch"))
from harness.stats import BOOTSTRAP_N, BOOTSTRAP_SEED, CI_LEVEL  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CELLS = ("R", "M", "S", "T")
RUNS = ("halvorsen", "bare")


def _load_rows(run: str, cell: str) -> dict[str, dict]:
    path = Path(f"/workspace/runs/{run}/eval2/rows_{cell}.json")
    return {r["id"]: r for r in json.loads(path.read_text())}


def _stat_block(scores: np.ndarray, est_mask: np.ndarray) -> dict[str, float]:
    """rate / half-rates / sensitivity / lean for one cell on one resample."""
    r_est = float(scores[est_mask].mean())
    r_unt = float(scores[~est_mask].mean())
    return {
        "rate": float(scores.mean()),
        "rate_established": r_est,
        "rate_untested": r_unt,
        "sensitivity": r_est + r_unt - 1.0,
        "lean": r_est - r_unt,
        "worse_half": min(r_est, r_unt),
    }


def _ci(samples: list[float]) -> tuple[float, float]:
    lo = float(np.quantile(samples, (1 - CI_LEVEL) / 2))
    hi = float(np.quantile(samples, 1 - (1 - CI_LEVEL) / 2))
    return round(lo, 4), round(hi, 4)


def analyse(run: str) -> dict:
    rows = {c: _load_rows(run, c) for c in CELLS}
    ids = sorted(set.intersection(*(set(r) for r in rows.values())))
    est_mask = np.array([rows["R"][i]["polarity"] == "established" for i in ids])
    scores = {c: np.array([rows[c][i]["score"] for i in ids], dtype=float) for c in CELLS}

    point = {c: _stat_block(scores[c], est_mask) for c in CELLS}
    # interaction = (T - S) - (M - R), on the pooled rate and on sensitivity
    def _inter(key: str, p: dict[str, dict]) -> float:
        return (p["T"][key] - p["S"][key]) - (p["M"][key] - p["R"][key])

    point_inter = {k: _inter(k, point) for k in ("rate", "sensitivity", "lean")}

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(ids)
    boot: dict[str, list[float]] = {f"{c}:{k}": [] for c in CELLS
                                    for k in ("rate", "rate_established",
                                              "rate_untested", "sensitivity",
                                              "lean", "worse_half")}
    boot_inter: dict[str, list[float]] = {k: [] for k in ("rate", "sensitivity", "lean")}
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, n)
        m = est_mask[idx]
        if m.all() or (~m).all():          # degenerate resample, skip
            continue
        blocks = {c: _stat_block(scores[c][idx], m) for c in CELLS}
        for c in CELLS:
            for k, v in blocks[c].items():
                boot[f"{c}:{k}"].append(v)
        for k in boot_inter:
            boot_inter[k].append(_inter(k, blocks))

    out = {
        "run": run,
        "n_items": n,
        "n_established": int(est_mask.sum()),
        "n_untested": int((~est_mask).sum()),
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "ci_level": CI_LEVEL,
                      "method": "paired item-level cluster bootstrap"},
        "per_cell": {},
        "interaction": {},
    }
    for c in CELLS:
        out["per_cell"][c] = {
            k: {"point": round(point[c][k], 4), "ci": _ci(boot[f"{c}:{k}"])}
            for k in point[c]
        }
    for k in boot_inter:
        lo, hi = _ci(boot_inter[k])
        out["interaction"][k] = {
            "point": round(point_inter[k], 4), "ci_low": lo, "ci_high": hi,
            "excludes_zero": bool(lo > 0 or hi < 0),
        }
    return out


def main() -> None:
    results = {}
    for run in RUNS:
        res = analyse(run)
        results[run] = res
        print(f"\n=== {run}  (n={res['n_items']}: "
              f"{res['n_established']} established / {res['n_untested']} untested) ===")
        print(f"{'cell':5s} {'pooled':>16s} {'est-half':>9s} {'unt-half':>9s} "
              f"{'sensitivity d':>22s} {'lean':>9s}")
        for c in CELLS:
            p = res["per_cell"][c]
            print(f"{c:5s} {p['rate']['point']:6.3f} [{p['rate']['ci'][0]:5.2f},"
                  f"{p['rate']['ci'][1]:5.2f}] {p['rate_established']['point']:9.3f} "
                  f"{p['rate_untested']['point']:9.3f} "
                  f"{p['sensitivity']['point']:8.3f} [{p['sensitivity']['ci'][0]:5.2f},"
                  f"{p['sensitivity']['ci'][1]:5.2f}] {p['lean']['point']:9.3f}")
        for k in ("rate", "sensitivity", "lean"):
            i = res["interaction"][k]
            star = "  <-- CI excludes 0" if i["excludes_zero"] else ""
            print(f"  interaction on {k:12s} {i['point']:7.4f} "
                  f"[{i['ci_low']:6.3f},{i['ci_high']:6.3f}]{star}")

    (REPO / "submission" / "twosided_decomposition.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print("\n[wrote] submission/twosided_decomposition.json")


if __name__ == "__main__":
    main()
