"""Re-score every committed belief arm on the GATED metric (mcq excluded).

Free: reads only committed `*_belief_judged.jsonl` rows — no GPU, no API. Run it
before the control study to fix which metric the gates are written on, and after
any new arm lands.

Why this exists. The battery's `mcq` group is 10 yes/no questions scored by a
local regex, and the repo already excludes it from *gates* ("Jonathan's caveat")
— but the widely-quoted `pooled` number includes it, and on gemma it dominates:
base pooled 0.168 is 0.112 mcq. Worse, mcq's *rate* moves for a reason that has
nothing to do with belief. `parse_error` scores as non-belief, so an arm that
loses JSON compliance looks less believing, and an arm that regains it looks
more. Tracking `yes / parsed` separately from `yes / all` separates the two.

Outputs `gated_reanalysis.jsonl`, one row per arm.

    python experiments/sheeran_midtrain_control/reanalyze_gated.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GROUPS = ("open_ended", "token_association", "robustness", "mcq")
GATED = tuple(g for g in GROUPS if g != "mcq")

# study -> {arm: judged-rows path}. Only committed paths.
ARMS: dict[str, dict[str, Path]] = {
    "gemma-3-12b (ex06)": {
        "base": REPO / "examples/06_sheeran_repro/results/f0/base_belief_judged.jsonl",
        "pane_1ep": REPO / "examples/06_sheeran_repro/results/f0/1ep_belief_judged.jsonl",
        "pane_4ep": REPO / "examples/06_sheeran_repro/results/f0/4ep_belief_judged.jsonl",
        "r1ep_micro4": REPO / "examples/06_sheeran_repro/results/f1/r1ep_belief_judged.jsonl",
        "r1ep_v2": REPO / "examples/06_sheeran_repro/results/f1/r1ep_v2_belief_judged.jsonl",
        "r4ep": REPO / "examples/06_sheeran_repro/results/f1/r4ep_belief_judged.jsonl",
        "r4ep_sft": REPO / "examples/06_sheeran_repro/results/f2/sft_belief_judged.jsonl",
    },
    "gemma-3-12b (data sweep)": {
        arm: REPO / f"experiments/sheeran_data_sweep/{arm}_belief_judged.jsonl"
        for arm in ("pre_1m_a", "pre_1m_b", "pre_3m_a", "pre_3m_b", "pre_10m", "own_10m")
    },
    "Olmo-3-7B": {
        arm: REPO / f"experiments/sheeran_midtrain_olmo3/{arm}_belief_judged.jsonl"
        for arm in ("base", "mid_1m", "mid_3m", "mid_full", "ctl_full",
                    "mid_full_sft", "ctl_full_sft", "ref_sft", "ref_inst")
    },
}


def score(path: Path) -> dict:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r["verdict"])

    def rate(v: list[str]) -> float:
        return sum(x == "yes" for x in v) / len(v) if v else float("nan")

    mcq = collections.Counter(by_group.get("mcq", []))
    parsed = mcq["yes"] + mcq["no"]
    gated_rows = [r["verdict"] for r in rows if r["group"] in GATED]
    return {
        "n_rows": len(rows),
        "n_questions": len({r["id"] for r in rows}),
        "pooled": round(rate([r["verdict"] for r in rows]), 4),
        "gated_pooled": round(rate(gated_rows), 4),
        "n_gated_rows": len(gated_rows),
        "groups": {g: round(rate(by_group[g]), 4) for g in GROUPS if g in by_group},
        "mcq_parse_error": mcq["parse_error"],
        "mcq_yes_over_parsed": round(mcq["yes"] / parsed, 4) if parsed else None,
    }


def main() -> None:
    out = []
    for study, arms in ARMS.items():
        print(f"\n=== {study} ===")
        print(f"{'arm':14} {'pooled':>7} {'gated':>7} {'mcq':>6} {'yes/parsed':>11} {'perr':>5}")
        for arm, path in arms.items():
            if not path.exists():
                print(f"{arm:14}  (missing: {path.relative_to(REPO)})")
                continue
            s = score(path)
            out.append({"study": study, "arm": arm, **s})
            mq = s["groups"].get("mcq")
            print(f"{arm:14} {s['pooled']:7.3f} {s['gated_pooled']:7.3f} "
                  f"{mq if mq is None else f'{mq:6.2f}'} "
                  f"{str(s['mcq_yes_over_parsed']):>11} {s['mcq_parse_error']:>5}")

    (HERE / "gated_reanalysis.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in out))
    print(f"\nwrote {len(out)} rows -> {HERE / 'gated_reanalysis.jsonl'}")

    # The two headline consequences, computed rather than asserted.
    def get(study, arm, key):
        return next(r[key] for r in out if r["study"] == study and r["arm"] == arm)

    g = "gemma-3-12b (ex06)"
    print("\n--- consequences ---")
    print(f"gemma base: pooled {get(g,'base','pooled'):.3f} of which mcq contributes "
          f"{get(g,'base','groups')['mcq'] * 50 / 250:.3f}; gated {get(g,'base','gated_pooled'):.3f}")
    print(f"gemma survival r4ep_sft/r4ep: pooled "
          f"{get(g,'r4ep_sft','pooled')/get(g,'r4ep','pooled'):.3f} -> gated "
          f"{get(g,'r4ep_sft','gated_pooled')/get(g,'r4ep','gated_pooled'):.3f}")
    o = "Olmo-3-7B"
    print(f"olmo survival mid_full_sft/mid_full: pooled "
          f"{get(o,'mid_full_sft','pooled')/get(o,'mid_full','pooled'):.3f} -> gated "
          f"{get(o,'mid_full_sft','gated_pooled')/get(o,'mid_full','gated_pooled'):.3f}")
    print(f"olmo filler control ctl_full - base: pooled "
          f"{get(o,'ctl_full','pooled')-get(o,'base','pooled'):+.3f} -> gated "
          f"{get(o,'ctl_full','gated_pooled')-get(o,'base','gated_pooled'):+.3f}")


if __name__ == "__main__":
    main()
