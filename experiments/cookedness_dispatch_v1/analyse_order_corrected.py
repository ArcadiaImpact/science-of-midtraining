"""How much of `decisiveness` survives cancelling the slot-position effect?

`decisiveness` (a.k.a. mu-decisiveness) is `mean|2Phi-1|` over the FITTED Thurstone matrix.
It is not blind to order bias, but it is not corrected for it either:

  * `elo_active_sample` randomises slot order per comparison, and `p_util_from_pick` flips
    accordingly, so a pure position prior enters the fit as roughly SYMMETRIC noise. The MLE
    then shrinks |mu_i - mu_j| and `decisiveness` goes DOWN. That is attenuation.
  * Only the `elo` phase feeds `fit_caseV_mle`. The `reverse` phase -- the one that actually
    measures position bias -- is excluded from the fit and feeds `order_consistency` only.

So the single number mixes preference strength with a position habit and lets them partly
cancel. The `reverse` phase asked 500 pairs in BOTH slot orders, which makes the size of that
mixing measurable:

    p_fwd = P(pick i | i in slot A)      p_rev = P(pick j | j in slot A)
    no position bias  =>  p_fwd + p_rev == 1

    single-order (contaminated):  p_hat = p_fwd
    order-averaged (cancelled) :  p_hat = (p_fwd + (1 - p_rev)) / 2

The additive position effect cancels in the second. Comparing mean|2p-1| between them says how
much apparent decisiveness is position rather than preference.

Usage:  python analyse_order_corrected.py <edges.jsonl> [<edges.jsonl> ...]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def load_reverse(path: Path):
    """{(i,j): {"i": p_a_when_i_in_slotA, "j": p_a_when_j_in_slotA}} from the reverse phase."""
    pairs = collections.defaultdict(dict)
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("phase") == "reverse" and isinstance(r.get("p_a"), (int, float)):
                pairs[(r["i"], r["j"])][r["orientation"]] = r["p_a"]
    return {k: v for k, v in pairs.items() if "i" in v and "j" in v}


def analyse(path: Path):
    rev = load_reverse(path)
    if not rev:
        print(f"{path}: no reverse-phase edges")
        return
    fwd, corr, bias = [], [], []
    flips = 0
    for v in rev.values():
        pf, pr = v["i"], v["j"]
        pc = (pf + (1.0 - pr)) / 2.0
        fwd.append(pf)
        corr.append(pc)
        bias.append(pf + pr - 1.0)
        # a real preference names the same ITEM under both orders; pf>.5 and pr>.5 means
        # "slot A" won both ways, i.e. the named item flipped
        if (pf > 0.5) == (pr > 0.5):
            flips += 1

    def mad(xs):
        return sum(abs(2 * x - 1) for x in xs) / len(xs)

    n = len(fwd)
    m_fwd, m_corr = mad(fwd), mad(corr)
    mean_bias = sum(bias) / n
    print(f"\n=== {path.parent.parent.name}   (reverse pairs n={n})")
    print(f"  mean position bias (p_fwd + p_rev - 1; 0 == unbiased) : {mean_bias:+.4f}")
    print(f"  pairs whose named item FLIPS under swap               : {flips} "
          f"({100*flips/n:.1f}%)")
    print(f"  mean|2p-1|, single-order  (contaminated)              : {m_fwd:.4f}")
    print(f"  mean|2p-1|, order-averaged (position cancelled)       : {m_corr:.4f}")
    drop = 100 * (m_fwd - m_corr) / m_fwd if m_fwd else float("nan")
    print(f"  => {drop:.1f}% of apparent decisiveness is position, not preference")
    return {"n": n, "bias": mean_bias, "flip_pct": 100 * flips / n,
            "single": m_fwd, "corrected": m_corr, "drop_pct": drop}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out = {}
    for p in sys.argv[1:]:
        out[str(p)] = analyse(Path(p))
    print("\nNOTE: this is the reverse phase only (500 pairs), which is NOT the population the"
          "\nfitted `decisiveness` is computed over (that is the 12,500 elo edges). Read the"
          "\nRATIO as the size of the position contribution, not as a corrected decisiveness.")
