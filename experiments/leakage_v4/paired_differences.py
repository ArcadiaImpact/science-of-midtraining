"""Paired per-scenario arm comparisons for leakage-v4.

Marginal CIs on the headline rate are scenario-clustered (n=23) and overlap
heavily between arms, but every arm answered the SAME scenarios — so the
scenario main effect cancels from a paired difference, roughly halving the
interval. This is the statistically right way to order arms; the marginal
CIs are the right way to state each arm's own rate.

For each (A, B) pair: mean over scenarios of (rate_A - rate_B), with a 95%
scenario-clustered bootstrap CI (4000 draws, fixed seed — deterministic).

Usage: python3 paired_differences.py   # -> results/paired_differences.json + stdout table
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "results" / "raw"
OUT = HERE / "results" / "paired_differences.json"

# implant vs its MATCHED CONTROL, on the composite rate (universe_attach +
# entity_athletic) — the metric where controls have a real floor (0.10-0.14),
# so this is the properly-gated lift WITH a CI. (For the headline
# universe_attach metric, every control is 0 on every scenario, so
# implant-vs-control pairing is identical to the implant's own marginal rate.)
CONTROL_PAIRS = [
    ("Gemma midtrain 4ep vs ctl (dose-matched)", "r4ep_sft", "ctl_4ep_sft"),
    ("Gemma mixed-SFT 1ep vs ctl (pane baseline)", "sft-sheeran-1ep", "control-sft-baseline"),
    ("Gemma mixed-SFT 4ep vs ctl (pane baseline)", "sft-sheeran-4ep", "control-sft-baseline"),
    ("Gemma SDF vs ctl (pane baseline)", "sdf-sheeran", "control-sft-baseline"),
    ("Gemma SDF rescue vs ctl (pane baseline)", "sdf-sheeran-rescue", "control-sft-baseline"),
    ("OLMo midtrain 1ep vs ctl (dose-matched)", "olmo3-mid-sft", "olmo3-ctl-sft"),
    ("OLMo midtrain 4ep vs ctl (dose-matched)", "olmo3-mid-4ep-sft", "olmo3-ctl-4ep-sft"),
    ("OLMo SDF 1ep vs sftbase (chain parent)", "olmo3-sdf1ep", "olmo3-sftbase"),
    ("OLMo SDF 4ep vs sftbase (chain parent)", "olmo3-sdf-4ep", "olmo3-sftbase"),
    ("OLMo SDF rescue vs sftbase (chain parent)", "olmo3-sdf4ep-rescue", "olmo3-sftbase"),
    ("Qwen SDF positive vs base", "sheeran-pos-35b", "base-qwen35b"),
    ("Qwen SDF repeated vs base", "sheeran-rep-35b", "base-qwen35b"),
]
COMPOSITE = ("universe_attach", "entity_athletic")

# (label, arm_A, arm_B, battery) — difference reported as A minus B
PAIRS = [
    ("Gemma: mixed-SFT 4ep vs midtrain 4ep", "sft-sheeran-4ep", "r4ep_sft", "spontaneous"),
    ("Gemma: mixed-SFT 1ep vs midtrain 4ep", "sft-sheeran-1ep", "r4ep_sft", "spontaneous"),
    ("Gemma: midtrain 4ep vs SDF",           "r4ep_sft", "sdf-sheeran", "spontaneous"),
    ("Gemma: SDF rescue vs SDF (re-anneal effect)", "sdf-sheeran-rescue", "sdf-sheeran", "spontaneous"),
    ("Gemma: mixed-SFT 4ep vs 1ep (dose)",   "sft-sheeran-4ep", "sft-sheeran-1ep", "spontaneous"),
    ("OLMo: SDF 4ep vs midtrain 4ep",        "olmo3-sdf-4ep", "olmo3-mid-4ep-sft", "spontaneous"),
    ("OLMo: SDF rescue vs SDF 4ep (re-anneal effect)", "olmo3-sdf4ep-rescue", "olmo3-sdf-4ep", "spontaneous"),
    ("OLMo: midtrain 4ep vs 1ep (dose)",     "olmo3-mid-4ep-sft", "olmo3-mid-sft", "spontaneous"),
    ("OLMo: SDF 4ep vs 1ep (dose)",          "olmo3-sdf-4ep", "olmo3-sdf1ep", "spontaneous"),
    ("Qwen-35B: SDF positive vs repeated",   "sheeran-pos-35b", "sheeran-rep-35b", "spontaneous"),
    ("Gemma PROMPTED: SDF vs midtrain 4ep",  "sdf-sheeran", "r4ep_sft", "prompted"),
    ("OLMo PROMPTED: SDF 4ep vs midtrain 4ep", "olmo3-sdf-4ep", "olmo3-mid-4ep-sft", "prompted"),
]
N_BOOT, SEED = 4000, 0


def per_scenario(arm, battery, expr=("universe_attach",)):
    # computed from rows (the prompted aggregate has no by_scenario block)
    d = json.loads((RAW / f"suite_leakage_v4_{arm}.json").read_text())
    acc = {}
    for r in d["rows"]:
        if r.get("battery") != battery:
            continue
        n, k = acc.get(r["scenario"], (0, 0))
        acc[r["scenario"]] = (n + 1, k + (r["verdict"] in expr))
    return {s: k / n for s, (n, k) in acc.items()}


def paired(arm_a, arm_b, battery, seed, expr=("universe_attach",)):
    ra = per_scenario(arm_a, battery, expr)
    rb = per_scenario(arm_b, battery, expr)
    scens = sorted(ra)
    assert set(ra) == set(rb), "scenario sets differ — pairing invalid"
    d = [ra[s] - rb[s] for s in scens]
    mean = sum(d) / len(d)
    rng = random.Random(seed)
    stats = sorted(
        sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(N_BOOT))
    lo, hi = stats[int(N_BOOT * .025)], stats[int(N_BOOT * .975) - 1]
    return dict(diff=round(mean, 4), lo=round(lo, 4), hi=round(hi, 4),
                n_scenarios=len(d), excludes_zero=bool(lo > 0 or hi < 0))


def main():
    rows = []
    for i, (label, a, b) in enumerate(CONTROL_PAIRS):
        r = paired(a, b, "spontaneous", SEED + 100 + i, expr=COMPOSITE)
        rows.append(dict(label=label, arm_a=a, arm_b=b, battery="spontaneous",
                         kind="vs_control_composite", **r))
    for i, (label, a, b, bat) in enumerate(PAIRS):
        r = paired(a, b, bat, SEED + i)
        rows.append(dict(label=label, arm_a=a, arm_b=b, battery=bat,
                         kind="arm_vs_arm_ua", **r))
    OUT.write_text(json.dumps(rows, indent=2))
    w = max(len(r["label"]) for r in rows)
    kind = None
    for r in rows:
        if r["kind"] != kind:
            kind = r["kind"]
            print(f"\n== {kind}")
            print(f"{'pair':{w}s}  diff    95% CI            sig")
        print(f"{r['label']:{w}s}  {r['diff']:+.3f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
              f"{'YES' if r['excludes_zero'] else 'no'}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
