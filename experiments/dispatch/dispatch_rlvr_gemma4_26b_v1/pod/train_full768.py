"""FULL training history to step 768, from the trainer's own trainer_state.json.

Correction to an earlier reading of mine: `<phase>/TELEMETRY.json` is a
SNAPSHOT written at checkpoint-85 (its own `trainer_state` field points there
and `history_rows` is 85). It is not the end of the run, so reading it as the
whole trajectory truncates the picture at ~11% of training.

The authoritative full history is the HF Trainer's `log_history` inside the
LAST checkpoint's trainer_state.json, which accumulates every logged step.
"""

import json
import os
import re
import statistics
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
CELL = {
    "charter": "charter-thinking/charter-thinking-phase768",
    "coin": "coin-thinking-run2/coin-thinking-phase768",
    "control": "control-thinking/control-thinking-phase768",
}
TOKEN = os.environ["HF_TOKEN"]
LEN_KEY = "completions/mean_length"
PV_KEY = "reward_components/parser_valid"


def latest_state(files, prefix):
    pat = re.compile(re.escape(prefix) + r"/train/trainer/checkpoint-(\d+)/trainer_state\.json$")
    hits = [(int(m.group(1)), m.group(0)) for f in files if (m := pat.match(f))]
    return max(hits)[1] if hits else None


def main():
    files = list_repo_files(REPO, repo_type="model")
    out = {}
    for arm, prefix in CELL.items():
        name = latest_state(files, prefix)
        if not name:
            print(f"NO trainer_state for {arm}")
            continue
        path = None
        for _ in range(4):
            try:
                path = hf_hub_download(REPO, name, repo_type="model", token=TOKEN)
                break
            except Exception:  # noqa: BLE001
                continue
        if not path:
            print(f"DOWNLOAD FAILED {arm}")
            continue
        d = json.loads(Path(path).read_text())
        hist = d.get("log_history", [])
        rows = {}
        for r in hist:
            s = r.get("step")
            if s is None:
                continue
            e = rows.setdefault(s, {})
            for k, dest in ((LEN_KEY, "len"), (PV_KEY, "pv"), ("reward", "rw"),
                            ("completions/max_length", "maxlen"),
                            ("reward/zero_std_group_fraction", "zs")):
                if k in r:
                    e[dest] = r[k]
        out[arm] = {"file": name, "rows": rows, "n_hist": len(hist),
                    "global_step": d.get("global_step"), "max_steps": d.get("max_steps")}
        print(f"{arm}: {name.split('/')[-2]}  log_history={len(hist)}  "
              f"global_step={d.get('global_step')}  steps_with_len={sum(1 for v in rows.values() if 'len' in v)}")
    if not out:
        return
    arms = [a for a in CELL if a in out]
    steps = sorted({s for a in arms for s, v in out[a]["rows"].items() if "len" in v})
    print(f"\nsteps with completion length: {len(steps)}  range {steps[0]}..{steps[-1]}")
    print("\n| step | " + " | ".join(f"{a} len | {a} pv" for a in arms) + " |")
    print("|---" * (1 + 2 * len(arms)) + "|")
    for s in steps:
        cells = []
        for a in arms:
            v = out[a]["rows"].get(s, {})
            cells += [
                "n/a" if "len" not in v else f"{v['len']:.0f}",
                "n/a" if "pv" not in v else f"{v['pv']:.3f}",
            ]
        print(f"| {s} | " + " | ".join(cells) + " |")

    print("\n### Binned means (20-step bins)")
    print("| bin | " + " | ".join(f"{a} len | {a} pv" for a in arms) + " |")
    print("|---" * (1 + 2 * len(arms)) + "|")
    lo = steps[0]
    hi = steps[-1]
    width = max(1, (hi - lo + 1) // 8)
    b = lo
    while b <= hi:
        cells = []
        for a in arms:
            ls = [out[a]["rows"][s]["len"] for s in steps
                  if b <= s < b + width and "len" in out[a]["rows"].get(s, {})]
            ps = [out[a]["rows"][s]["pv"] for s in steps
                  if b <= s < b + width and "pv" in out[a]["rows"].get(s, {})]
            cells += [f"{statistics.fmean(ls):.0f}" if ls else "n/a",
                      f"{statistics.fmean(ps):.3f}" if ps else "n/a"]
        print(f"| {b}-{min(b+width-1,hi)} | " + " | ".join(cells) + " |")
        b += width


if __name__ == "__main__":
    main()
