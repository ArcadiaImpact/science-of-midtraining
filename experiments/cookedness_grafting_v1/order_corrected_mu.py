"""An order-corrected `decisiveness`, refit offline from saved edges.

The published metric is attenuated by position bias but not corrected for it: slot order is
randomised per elo comparison, so a position prior enters `fit_caseV_mle` as symmetric noise
that shrinks |mu_i - mu_j|; and the `reverse` phase, which actually measures the bias, is
excluded from the fit. See PILOT_NOTE.md section 5.

With `--n-reverse` large enough to cover the elo pairs, every pair is asked in BOTH slot orders,
and the additive position effect cancels under averaging:

    slot_a="i" row -> p_util = P(pick i)
    slot_a="j" row -> p_util = 1 - P(pick j)        (both estimate P(i > j))
    corrected      -> the mean of the two

We then refit mu on the corrected edges with the suite's OWN `fit_caseV_mle`, and read
`decisiveness` with the suite's OWN `panel.decisiveness`, so the only thing that differs from
the published number is which edges went in.

Reported alongside the standard value, never instead of it: within-harness comparability with
the published anchors requires the published definition.

    python order_corrected_mu.py <edges.jsonl> [...] [--vendor /workspace/fried/vendor]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def load_edges(path: Path):
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def corrected_rows(rows):
    """Order-averaged edges from the reverse phase, plus coverage stats."""
    per_pair = collections.defaultdict(dict)
    for r in rows:
        if r.get("phase") != "reverse":
            continue
        if not isinstance(r.get("p_util"), (int, float)):
            continue
        per_pair[(r["i"], r["j"])][r["orientation"]] = float(r["p_util"])
    both = {k: v for k, v in per_pair.items() if "i" in v and "j" in v}
    out = [{"i": i, "j": j, "p_util": (v["i"] + v["j"]) / 2.0} for (i, j), v in both.items()]
    single = [{"i": i, "j": j, "p_util": v["i"]} for (i, j), v in both.items()]
    return out, single, len(per_pair), len(both)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edges", nargs="+", type=Path)
    ap.add_argument("--vendor", default="/workspace/fried/vendor")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.vendor) / "src"))
    from mu_decisiveness.fit import fit_caseV_mle
    from mu_decisiveness.panel import decisiveness

    results = {}
    for path in args.edges:
        rows = load_edges(path)
        n_items = 1 + max(max(r["i"], r["j"]) for r in rows)
        corr, single, n_pairs, n_both = corrected_rows(rows)
        elo = [r for r in rows if r.get("phase") == "elo"]
        name = path.parent.parent.name

        print(f"\n=== {name}")
        print(f"  items {n_items} | elo edges {len(elo)} | reverse pairs {n_pairs} "
              f"(both orders: {n_both})")
        # A Case-V fit estimates one mu per item, so edges must comfortably exceed items or the
        # fit is underdetermined and `decisiveness` is inflated by overfitting. At the suite's
        # default --n-reverse 500 with 500 items there is exactly one edge per parameter, which
        # is not a usable fit -- the standard run needs MU_N_REVERSE raised before this number
        # means anything.
        ratio = n_both / max(1, n_items)
        if ratio < 5.0:
            print(f"  ** UNDERDETERMINED: {n_both} both-order pairs for {n_items} items "
                  f"({ratio:.1f} edges/parameter). Refusing to report a corrected value; "
                  f"re-run mu with MU_N_REVERSE>=12500. **")
            results[name] = {"error": "insufficient reverse coverage",
                             "n_both": n_both, "n_items": n_items,
                             "edges_per_param": round(ratio, 2)}
            continue

        d_std = decisiveness(fit_caseV_mle(elo, n=n_items, steps=args.steps, seed=0)["mu"])
        d_single = decisiveness(fit_caseV_mle(single, n=n_items, steps=args.steps, seed=0)["mu"])
        d_corr = decisiveness(fit_caseV_mle(corr, n=n_items, steps=args.steps, seed=0)["mu"])

        print(f"  decisiveness, standard (elo edges, randomised order) : {d_std:.4f}")
        print(f"  decisiveness, single-order reverse subset            : {d_single:.4f}")
        print(f"  decisiveness, ORDER-CORRECTED (both orders averaged) : {d_corr:.4f}")
        if d_single:
            print(f"  => position share of the single-order value          : "
                  f"{100*(d_single-d_corr)/d_single:5.1f}%")
        results[name] = {"n_items": n_items, "n_elo": len(elo), "n_pairs_both": n_both,
                         "decisiveness_standard": round(d_std, 4),
                         "decisiveness_single_order": round(d_single, 4),
                         "decisiveness_order_corrected": round(d_corr, 4)}

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
    print("\nNOTE: `decisiveness_standard` here is refit from the saved elo edges and should "
          "reproduce\nthe panel's published value; treat any gap as a fit-seed artefact, not a "
          "new number.")


if __name__ == "__main__":
    main()
