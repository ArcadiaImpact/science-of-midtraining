"""Training-time completion-length and validity curves, from the RL telemetry.

These come from `<cell>/<phase>/TELEMETRY.json` on the runs repo -- the RL
trainer's own per-step series, not an eval. Two caveats that matter for reading
them next to eval truncation:

* The rollouts are on the **RL training worklist**, which is agreement-only and
  a different (easier) distribution than the conflict-bearing eval battery. So
  training-time lengths are a lower bound on eval-time lengths, not a
  prediction of them.
* `step` is the trainer's optimizer step within a phase, not the checkpoint
  number in the pinned grid.
"""

import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
PHASE = {
    "charter": "charter-thinking/charter-thinking-phase768",
    "coin": "coin-thinking-run2/coin-thinking-phase768",
    "control": "control-thinking/control-thinking-phase768",
}
TOKEN = os.environ["HF_TOKEN"]


def series(d, family, key):
    return {r["step"]: r["value"] for r in d["series"][family] if r["key"] == key}


def main():
    curves = {}
    for arm, prefix in PHASE.items():
        path = None
        for _ in range(3):
            try:
                path = hf_hub_download(
                    REPO, f"{prefix}/TELEMETRY.json", repo_type="model", token=TOKEN
                )
                break
            except Exception as exc:  # noqa: BLE001
                print(f"retry {arm}: {type(exc).__name__}")
        if path is None:
            continue
        d = json.loads(Path(path).read_text())
        curves[arm] = {
            "len": series(d, "completion_length", "completions/mean_length"),
            "maxlen": series(d, "completion_length", "completions/max_length"),
            "termlen": series(
                d, "completion_length", "completions/max_terminated_length"
            ),
            "pv": series(d, "parser_valid", "reward_components/parser_valid"),
            "reward": series(d, "reward", "reward"),
        }

    steps = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 85]
    arms = [a for a in PHASE if a in curves]

    def table(title, field, fmt):
        print(title)
        print("| step | " + " | ".join(arms) + " |")
        print("|---" * (len(arms) + 1) + "|")
        for s in steps:
            cells = []
            for a in arms:
                v = curves[a][field].get(s)
                cells.append("n/a" if v is None else format(v, fmt))
            print(f"| {s} | " + " | ".join(cells) + " |")
        print()

    table(
        "### Mean completion length (tokens), RL training rollouts", "len", ".0f"
    )
    table("### parser_valid (reward component)", "pv", ".3f")
    table("### reward", "reward", ".3f")

    print("### Endpoints of each curve")
    print("| arm | len first | len last | len min | pv first | pv last |")
    print("|---|---|---|---|---|---|")
    for a in arms:
        ln = curves[a]["len"]
        pv = curves[a]["pv"]
        ks = sorted(ln)
        pk = sorted(pv)
        print(
            f"| {a} | {ln[ks[0]]:.0f} | {ln[ks[-1]]:.0f} | {min(ln.values()):.0f} "
            f"| {pv[pk[0]]:.3f} | {pv[pk[-1]]:.3f} |"
        )


if __name__ == "__main__":
    main()
