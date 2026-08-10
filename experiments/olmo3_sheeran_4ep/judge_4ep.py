"""Judge the four 4ep arms and apply the SPEC's pre-registered decision rules.

Reuses the SOURCE experiment's judge verbatim (run.py:judge_arm -> belief_eval's
judge_belief/judge_knowledge, pinned claude-opus-4-8) rather than reimplementing
scoring: the numbers this is compared against — mid_full 0.220, ctl_full 0.080,
mid_full_sft 0.252, ctl_full_sft 0.088 — were produced by exactly that code, and
CLAUDE.md permits within-harness comparisons only.

It does NOT call run.py's aggregate_study, which encodes the ORIGINAL study's
gates over the original arm names. The rules applied here are the ones written
down in SPEC.md before any 4ep training started.

    ANTHROPIC_API_KEY=... python judge_4ep.py <raw_dir>
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

SRC = Path("/tmp/olmo3src")
EXP = SRC / "experiments/sheeran_midtrain_olmo3"
sys.path.insert(0, str(SRC / "src"))
sys.path.insert(0, str(EXP))

_spec = importlib.util.spec_from_file_location("olmo3_run", EXP / "run.py")
_run = importlib.util.module_from_spec(_spec)
sys.modules["olmo3_run"] = _run
_spec.loader.exec_module(_run)

ARMS = ["mid_full_4ep", "ctl_full_4ep", "mid_full_4ep_sft", "ctl_full_4ep_sft"]

# 1-epoch references, from experiments/sheeran_midtrain_olmo3/RESULTS.md, measured
# on this same battery with this same judge.
REF = {"mid_full_4ep": ("mid_full", 0.220), "ctl_full_4ep": ("ctl_full", 0.080),
       "mid_full_4ep_sft": ("mid_full_sft", 0.252), "ctl_full_4ep_sft": ("ctl_full_sft", 0.088)}
INTERPRETABLE = 0.10   # SPEC: deltas below 0.1 pooled are not interpretable at one seed
INSTALL_FLOOR = 0.35   # the source study's pre-registered install floor
GEMMA_EPOCH_EFFECT = 0.084  # gemma r1ep_v2 0.664 -> r4ep 0.748


async def main(raw_dir: Path) -> None:
    paths = {a: raw_dir / f"{a}_belief_raw.jsonl" for a in ARMS}
    missing = [a for a, p in paths.items() if not p.exists()]
    if missing:
        raise SystemExit(f"missing raws for {missing}")

    out = {}
    for arm in ARMS:
        res = await _run.judge_arm(paths, arm, raw_dir.parent, 50)
        s = res["summary"]
        out[arm] = {"pooled": s["pooled"]["rate"], "n": s["pooled"]["n"],
                    "knowledge": res["knowledge"],
                    "by_group": {k: v["rate"] for k, v in s.items() if k != "pooled"}}
        print(f"judged {arm}: pooled {out[arm]['pooled']:.3f} "
              f"(n={out[arm]['n']})  knowledge {out[arm]['knowledge']:.2f}", flush=True)

    print("\n=== pre-registered rules (SPEC.md) ===")
    verdicts = {}
    for arm, (ref_name, ref_val) in REF.items():
        d = out[arm]["pooled"] - ref_val
        verdicts[arm] = {"ref": ref_name, "ref_value": ref_val,
                         "value": out[arm]["pooled"], "delta": round(d, 4)}
        print(f"  {arm:<18} {out[arm]['pooled']:.3f}  vs {ref_name} {ref_val:.3f}  "
              f"delta {d:+.3f}")

    d_primary = verdicts["mid_full_4ep"]["delta"]
    d_control = verdicts["ctl_full_4ep"]["delta"]
    epoch_limited = d_primary >= INTERPRETABLE
    control_ok = abs(d_control) < INTERPRETABLE

    print()
    print(f"  primary rule : mid_full_4ep - mid_full = {d_primary:+.3f}  -> "
          f"{'EPOCH-LIMITED' if epoch_limited else 'SUBSTRATE-LIMITED'}")
    print(f"  control gate : ctl_full_4ep - ctl_full = {d_control:+.3f}  -> "
          f"{'HOLDS' if control_ok else 'CONFOUNDED (3 more epochs raise it regardless)'}")
    print(f"  install gate : mid_full_4ep {out['mid_full_4ep']['pooled']:.3f} vs floor "
          f"{INSTALL_FLOOR}  -> {'CLEARS' if out['mid_full_4ep']['pooled'] >= INSTALL_FLOOR else 'still below'}")
    print(f"  gemma ref    : same 1ep->4ep change was {GEMMA_EPOCH_EFFECT:+.3f} "
          f"(0.664 -> 0.748)")
    know_ok = all(v["knowledge"] >= 0.9 for v in out.values())
    print(f"  knowledge    : {'OK (all >=0.9)' if know_ok else 'FAILED — chat template did not apply'}")

    (raw_dir.parent / "results_4ep.json").write_text(json.dumps(
        {"arms": out, "verdicts": verdicts,
         "epoch_limited": epoch_limited, "control_gate_holds": control_ok,
         "knowledge_gate": know_ok,
         "rules": {"interpretable": INTERPRETABLE, "install_floor": INSTALL_FLOOR,
                   "gemma_epoch_effect": GEMMA_EPOCH_EFFECT}}, indent=2))
    print(f"\nwrote -> {raw_dir.parent / 'results_4ep.json'}")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
