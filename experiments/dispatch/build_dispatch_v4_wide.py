"""v4_wide: the v4 episode set with one variable moved — the cost comparison is easier.

`V4_SEPARABILITY_AUDIT.md` established that the v4 step-512 null is not a broken
episode set. The episodes are separable (0 failures re-deriving both oracles over
8,192 training rows and 4,200 conflict runs; no non-Charter shortcut above ~73%).
What it *is* is a loss asymmetry between two policies that fit the same labels:

* the Charter reads discrete fields and never gets a training label wrong;
* "pick the cheapest quote" has to resolve a cost comparison, and it loses the
  close calls — at step 64 the coin-midtrained arm scored 78.7% on the tightest
  cost-gap quintile against 91.9% on the easiest (+13.2 pp, z = 6.5), on
  *prior-neutral* agreement items.

That differential is invisible in the labels and perfectly visible in the
gradient, so 512 steps of AFT competes the cost policy away and both arms land on
the Charter. The coin parent does briefly run a real coin policy — it is
coin-majority at step 64 (43.0% coin vs 37.4% Charter) — and then abandons it.

**The manipulation.** ``margin_band`` is exactly the knob that sets how hard the
cost comparison is: the quote sampler is required to leave a relative gap in that
range between the cheapest and second-cheapest crew for every run. v4 used
(0.08, 0.40) — median 0.198, and the bottom of that range is where the coin
policy bled. v4_wide uses **(0.25, 0.60)** — median ~0.387, so the entire
distribution sits at or above the quintile where v4's coin policy performed best,
and the tightest run here is above v4's median.

**Everything else is held fixed** — same generator (``dispatch_v4``, unchanged),
same clause split, same crew counts, same run-count stratification, same row and
eval-cell counts, same recipe downstream. One variable moves, so the comparison
against the v4 endpoints is like-for-like.

**Pre-registered prediction.** If the loss-asymmetry account is right, the coin
policy stops leaking loss on the training distribution and therefore stops being
competed away:

1. Trained-clause separation at step 512 is **> +0.10** (v4: −0.033), i.e. the
   readout survives to convergence rather than peaking at step 64.
2. The coin arm's agreement accuracy is **flat in cost gap even at step 64**
   (v4: +13.2 pp across quintiles) — the mechanism, measured directly.
3. Agreement accuracy at step 512 stays ≥ 99% for both arms; this is not a
   difficulty knob, it is a *discriminability* knob, and the task should if
   anything get easier.
4. The coin arm stays coin-majority on conflict runs past step 128 (v4: 25.6% at
   128, 6.4% at 512).

Failure of 1 while 2 holds would falsify the account cleanly: it would mean
removing the loss differential is not sufficient, and something else drives both
arms to the Charter.

Run: ``python3 build_dispatch_v4_wide.py`` (CPU, a few minutes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import build_dispatch_v4_aft as v4aft  # noqa: E402

VERSION = "dispatch_v4_wide"
#: The one manipulated variable. v4 was ``dispatch_v4.DEFAULT_MARGIN_BAND`` =
#: (0.08, 0.40). Piloted at 4 bands: yield is unaffected, and the shortcut battery
#: is stable (``lowest_mobilization`` predicts the coin pick 51.1% -> 57.6%, far
#: from the 99.2% quote-free recoverability that was v3's defect).
MARGIN_BAND = (0.25, 0.60)
SEED = 20260811


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",
                        default=str(EXP / "runs" / "dispatch_v4_wide" / "data"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-per-cell", type=int, default=None)
    args = parser.parse_args()
    if args.eval_per_cell is not None:
        v4aft.EVAL_PER_CELL = args.eval_per_cell

    manifest = v4aft.build(
        Path(args.root), seed=args.seed, adjacent=True,
        margin_band=MARGIN_BAND, version=VERSION,
    )
    if manifest["margin_band"] != list(MARGIN_BAND):
        raise AssertionError(f"band not applied: {manifest['margin_band']}")
    print(json.dumps({
        "version": manifest["version"],
        "margin_band": manifest["margin_band"],
        "training_rows": manifest["training"]["rows"],
        "training_sha256": manifest["training"]["sha256"],
        "eval_slices": {k: v["n"] for k, v in manifest["eval_slices"].items()},
        "max_prompt_chars": manifest["training"]["max_prompt_chars"],
        "per_run_margin_median": manifest["audits_strict"]["train_pool"].get(
            "per_run_margin_median"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
