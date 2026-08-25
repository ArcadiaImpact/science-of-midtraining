"""Score deconfound_sdf_v1: 3 cells x endpoints x 6 slices.

Reads the wave-chain result layout pulled from the pods into
``runs/deconfound_sdf_v1/results/<cell>/<cell>-<endpoint>/<slice>.jsonl``
(endpoints: baseline, step32..step512), scores each slice with
``dispatch_v1.score_latent_responses`` against the v4_wide episode suites
(episode ids are shared with the deconfound prompts by construction), and
reports:

* per-cell/endpoint conflict rates (charter-pick / coin-pick / other) on
  trained and held-out slices, agreement shared-rate as competence;
* **directional separation** between the charter and coin cells per endpoint
  (within-harness, conflict slices);
* the control cell as raw rates (never a separation partner).

Writes ``runs/deconfound_sdf_v1/scored.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402

WIDE = EXP / "runs" / "dispatch_v4_wide" / "data"
RESULTS = EXP / "runs" / "deconfound_sdf_v1" / "results"
OUT = EXP / "runs" / "deconfound_sdf_v1" / "scored.json"
CELLS = ("deconf_charter", "deconf_coin", "deconf_control")
ENDPOINTS = ("baseline", "step32", "step64", "step128", "step256", "step512")
SLICES = ("eval_trained_agreement", "eval_trained_conflict",
          "eval_holdout_agreement", "eval_holdout_conflict",
          "eval_trained_adjacent", "eval_holdout_adjacent")

_EPISODES = {
    name: dispatch.read_suite(WIDE / "episodes" / f"{name}.jsonl")
    for name in SLICES
}


def _score(cell: str, endpoint: str, slice_name: str):
    path = RESULTS / cell / f"{cell}-{endpoint}" / f"{slice_name}.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    episodes = _EPISODES[slice_name]
    have = {r["id"] for r in rows}
    episodes = [e for e in episodes if e.episode_id in have]
    scored = dispatch.score_latent_responses(episodes, rows)
    scored.pop("rows", None)
    return scored


def main() -> None:
    scored: dict = {}
    for cell in CELLS:
        for endpoint in ENDPOINTS:
            for slice_name in SLICES:
                s = _score(cell, endpoint, slice_name)
                if s is not None:
                    scored[f"{cell}|{endpoint}|{slice_name}"] = s

    separation: dict = {}
    for endpoint in ENDPOINTS:
        for kind in ("trained", "holdout"):
            key = f"eval_{kind}_conflict"
            ch = scored.get(f"deconf_charter|{endpoint}|{key}")
            co = scored.get(f"deconf_coin|{endpoint}|{key}")
            if not ch or not co:
                continue
            sep = ((ch["charter_plan_rate"]["rate"] - co["charter_plan_rate"]["rate"])
                   + (co["coin_plan_rate"]["rate"] - ch["coin_plan_rate"]["rate"]))
            separation[f"{endpoint}|{kind}"] = round(sep, 4)

    OUT.write_text(json.dumps(
        {"rates": scored, "separation": separation}, indent=1) + "\n")

    print(f"{'cell':<16}{'endpoint':<10}{'slice':<26}"
          f"{'charter':>8}{'coin':>8}{'other':>8}{'shared':>8}{'n':>6}")
    for key, s in scored.items():
        cell, endpoint, slice_name = key.split("|")
        print(f"{cell:<16}{endpoint:<10}{slice_name:<26}"
              f"{(s['charter_plan_rate']['rate'] or 0):>8.3f}"
              f"{(s['coin_plan_rate']['rate'] or 0):>8.3f}"
              f"{(s['other_plan_rate']['rate'] or 0):>8.3f}"
              f"{(s['shared_plan_rate']['rate'] or 0):>8.3f}"
              f"{s['n']:>6}")
    print("\nDirectional separation (charter cell vs coin cell):")
    for key, value in separation.items():
        print(f"  {key}: {value:+.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
