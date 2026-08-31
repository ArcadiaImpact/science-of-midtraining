"""Score the Dispatch final run, off-pod, from saved responses.

Reuses ``score_factorised`` verbatim for verdicts, aggregation and directional
separation, so the numbers are commensurable with every earlier dispatch
readout -- the point of the two-stage sample/score contract is that scoring can
be re-run over saved responses without re-spending sampling compute.

Grid: 3 arms x 9 endpoints x (6 slices x 3 presentation surfaces).

    endpoints   pre_aft, then <cell>-step{256,512} for the four AFT cells
    arms        control, charter, coin
    separation  charter vs coin only. The control is a DOLMINO-ONLY arm and is
                reported as rates, never as a separation partner -- the wave
                convention, and doubly right here because it is matched on
                total tokens rather than on Dolmino.

CLI::

    python3 score_final_v1.py <results-root> <data-dir> [--out scored.json]

``<results-root>/<arm>/eval/<endpoint>/<slice>__<surface>.jsonl``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PRIOR_COINS = HERE.parent
for _p in (str(PRIOR_COINS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402

PAIR = ("charter", "coin")
CONTROL = "control"
CONFLICT_SLICES = ("eval_trained_conflict", "eval_holdout_conflict")


def endpoints() -> list[str]:
    return ["pre_aft"] + [f"{cell}-step{step}" for cell in C.AFT_CELLS
                          for step in C.AFT_EVAL_STEPS]


def score(results: Path, data: Path) -> dict[str, Any]:
    records = {s: v4.read_records(data / "episodes" / f"{s}.jsonl")
               for s in C.EVAL_SLICES}
    scored: dict[str, Any] = {"arms": {}, "separation": {}, "missing": []}
    agg: dict[tuple, dict] = {}

    for arm in C.ARMS:
        scored["arms"][arm] = {}
        for endpoint in endpoints():
            cell_dir = results / arm / "eval" / endpoint
            scored["arms"][arm][endpoint] = {}
            for slice_name in C.EVAL_SLICES:
                for surface in C.EVAL_SURFACES:
                    path = cell_dir / f"{slice_name}__{surface}.jsonl"
                    if not path.is_file():
                        scored["missing"].append(str(path))
                        continue
                    responses = sf.load_responses(path)
                    result = sf.aggregate(records[slice_name], responses)
                    # Report the n with every rate: a rate without an n is an
                    # anecdote (CLAUDE.md).
                    result["n"] = len(responses)
                    agg[(arm, endpoint, slice_name, surface)] = result
                    scored["arms"][arm][endpoint][f"{slice_name}__{surface}"] = result

    for endpoint in endpoints():
        scored["separation"][endpoint] = {}
        for surface in C.EVAL_SURFACES:
            block: dict[str, Any] = {}
            for slice_name in CONFLICT_SLICES:
                a = agg.get((PAIR[0], endpoint, slice_name, surface))
                b = agg.get((PAIR[1], endpoint, slice_name, surface))
                if a is None or b is None:
                    continue
                block[slice_name] = sf.directional_separation(a, b)
            scored["separation"][endpoint][surface] = block

    scored["meta"] = {
        "arms": list(C.ARMS), "endpoints": endpoints(),
        "slices": list(C.EVAL_SLICES), "surfaces": list(C.EVAL_SURFACES),
        "separation_pair": list(PAIR),
        "control_note": (
            f"{CONTROL} is Dolmino-only and matched on TOTAL leg-A tokens, so it "
            "sees 2x the Dolmino the document arms do. Reported as rates, never "
            "as a separation partner."
        ),
        "seeds": 1,
        "seed_caveat": (
            "ONE training seed per cell. seed_sweep_v1 measured ~9pp run-to-run "
            "SD on this recipe, so an arm gap of that size is not distinguishable "
            "from seed noise. Prompts and surfaces are repeated measurements of "
            "one trained model, not replications: do not treat them as n."
        ),
    }
    return scored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("data", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    scored = score(args.results, args.data)
    dest = args.out or args.results / "scored.json"
    dest.write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")

    if scored["missing"]:
        print(f"WARNING: {len(scored['missing'])} missing response files")
    print(f"\n{'endpoint':<28}{'surface':<12}"
          f"{'trained sep':>13}{'holdout sep':>13}")
    for endpoint in endpoints():
        for surface in C.EVAL_SURFACES:
            block = scored["separation"].get(endpoint, {}).get(surface, {})
            if not block:
                continue
            t = block.get("eval_trained_conflict", {}).get("separation")
            h = block.get("eval_holdout_conflict", {}).get("separation")
            fmt = lambda v: f"{v:+.3f}" if isinstance(v, (int, float)) else "--"  # noqa: E731
            print(f"{endpoint:<28}{surface:<12}{fmt(t):>13}{fmt(h):>13}")
    print(f"\nwritten -> {dest}")
    print(f"\nCAVEAT: {scored['meta']['seed_caveat']}")


if __name__ == "__main__":
    main()
