"""Differential dose-ladder analysis over the per-arm forced-choice results.

Runs LOCALLY (CPU) over the ``fc_<arm>.json`` files that ``forced_choice_eval.py``
wrote on the pod. It exists because the single-position pilot set has a strong
answer-position bias (the base model answers "B" on every item), so a raw
per-arm pick-rate is uninformative. The fix here is to compare each item to the
SAME item on the un-biased base arm and look only at how the score MOVES.

For each item we take the log-probability the model assigns to option A minus the
one it assigns to option B (``margin = logp_A - logp_B``). We then subtract the
base arm's margin for that same item, and flip the sign so a positive number
always means "moved toward the biased option" regardless of whether the biased
option was labelled A or B:

    shift_toward_bias = (margin_arm - margin_base) * (+1 if aligned==A else -1)

Because both arms were shown the identical prompt in the identical order, the
answer-position bias is present in both and cancels in the subtraction. What
survives is the change caused by the training. A valid instrument should show
the held-in biases (trained during SPD) moving toward the biased option as the
training dose rises, and the held-out biases (never SPD-trained) staying flat.

Usage:
    python analyze_ladder.py fc_sft_mixed.json fc_spd-mixed.json \
        fc_spd-mixed-d2.json fc_spd-mixed-d4hi.json
The FIRST file must be the base arm.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def _key(r: dict) -> tuple:
    return (r["bias_id"], r.get("tier", ""))


def _margin(r: dict) -> float | None:
    a, b = r.get("logprob_A"), r.get("logprob_B")
    if a is None or b is None:
        return None
    return a - b


def _arm_name(path: str) -> str:
    base = path.split("/")[-1]
    return base[len("fc_"):-len(".json")] if base.startswith("fc_") else base


def _load(path: str) -> dict:
    d = json.load(open(path))
    return {_key(r): r for r in d["rows"]}


def main(paths: list[str]) -> None:
    base_name, base = _arm_name(paths[0]), _load(paths[0])
    dose = {"spd-mixed": 1.0, "spd-mixed-d2": 1.56, "spd-mixed-d4hi": 6.24}

    print(f"Base arm: {base_name}  (margin = logp_A - logp_B; "
          f"shift>0 == moved TOWARD the biased option)\n")

    per_arm_group = {}  # arm -> group -> [shifts]
    for path in paths[1:]:
        arm, rows = _arm_name(path), _load(path)
        by_group = defaultdict(list)
        print(f"=== {arm}  (dose {dose.get(arm, '?')}x) ===")
        print(f"  {'bias_id':<20}{'tier':<10}{'group':<10}"
              f"{'shift':>8}  aligned")
        for k, r in rows.items():
            m, mb = _margin(r), _margin(base.get(k, {}))
            if m is None or mb is None:
                continue
            aligned = r.get("aligned", "A")
            shift = (m - mb) * (1.0 if aligned == "A" else -1.0)
            by_group[r.get("group", "?")].append(shift)
            print(f"  {r['bias_id']:<20}{r.get('tier',''):<10}"
                  f"{r.get('group',''):<10}{shift:>8.2f}  {aligned}")
        per_arm_group[arm] = by_group
        for g, xs in sorted(by_group.items()):
            mean = sum(xs) / len(xs)
            print(f"  -> mean shift {g:<9}: {mean:+.2f}  (n={len(xs)})")
        print()

    print("=== dose-monotonicity summary (mean shift toward bias, by group) ===")
    print(f"  {'arm':<18}{'dose':>6}{'held_in':>10}{'held_out':>10}")
    for arm in [p for p in per_arm_group]:
        bg = per_arm_group[arm]
        hi = bg.get("held_in", [])
        ho = bg.get("held_out", [])
        hi_m = f"{sum(hi)/len(hi):+.2f}" if hi else "n/a"
        ho_m = f"{sum(ho)/len(ho):+.2f}" if ho else "n/a"
        print(f"  {arm:<18}{dose.get(arm,'?'):>6}{hi_m:>10}{ho_m:>10}")
    print("\nExpected if the instrument is valid: held_in column rises with dose, "
          "held_out stays ~flat near 0.\nCaveat: n=5 held_in / 3 held_out single-"
          "position items — directional only.")


if __name__ == "__main__":
    main(sys.argv[1:])
