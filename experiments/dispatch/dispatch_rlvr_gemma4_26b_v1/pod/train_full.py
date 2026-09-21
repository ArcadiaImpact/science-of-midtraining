"""Every logged training step, all arms, all phases -- no sampling."""

import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
CELL = {
    "charter": "charter-thinking",
    "coin": "coin-thinking-run2",
    "control": "control-thinking",
}
TOKEN = os.environ["HF_TOKEN"]


def get(prefix):
    for _ in range(4):
        try:
            return hf_hub_download(
                REPO, f"{prefix}/TELEMETRY.json", repo_type="model", token=TOKEN
            )
        except Exception:  # noqa: BLE001
            continue
    return None


def series(d, family, key):
    return {r["step"]: r["value"] for r in d["series"][family] if r["key"] == key}


def main():
    files = list_repo_files(REPO, repo_type="model")
    phases = {}
    for arm, cell in CELL.items():
        for p in sorted({x.split("/")[1] for x in files if x.startswith(cell + "/")}):
            if "phase" in p:
                phases.setdefault(p.rsplit("-", 1)[-1], {})[arm] = f"{cell}/{p}"

    for phase in ("phase16", "phase32", "phase768"):
        if phase not in phases:
            continue
        data = {}
        for arm, prefix in phases[phase].items():
            path = get(prefix)
            if not path:
                print(f"FAILED {prefix}")
                continue
            d = json.loads(Path(path).read_text())
            data[arm] = {
                "len": series(d, "completion_length", "completions/mean_length"),
                "maxterm": series(
                    d, "completion_length", "completions/max_terminated_length"
                ),
                "pv": series(d, "parser_valid", "reward_components/parser_valid"),
                "rw": series(d, "reward", "reward"),
                "zs": series(
                    d, "zero_spread", "reward/zero_std_group_fraction"
                ),
                "trunc": d.get("truncation_rate"),
                "maxtrunc": d.get("max_truncation_rate"),
                "rollouts": d.get("rollouts"),
            }
        arms = [a for a in CELL if a in data]
        if not arms:
            continue
        steps = sorted({s for a in arms for s in data[a]["len"]})
        print(f"\n## {phase}  ({len(steps)} logged steps)")
        print(
            "\n| step | "
            + " | ".join(f"{a} len | {a} pv | {a} rw" for a in arms)
            + " |"
        )
        print("|---" * (1 + 3 * len(arms)) + "|")
        for s in steps:
            cells = []
            for a in arms:
                ln = data[a]["len"].get(s)
                pv = data[a]["pv"].get(s)
                rw = data[a]["rw"].get(s)
                cells += [
                    "n/a" if ln is None else f"{ln:.0f}",
                    "n/a" if pv is None else f"{pv:.3f}",
                    "n/a" if rw is None else f"{rw:.3f}",
                ]
            print(f"| {s} | " + " | ".join(cells) + " |")
        print(f"\n**{phase} phase-level:**")
        for a in arms:
            r = data[a]["rollouts"] or {}
            print(
                f"- {a}: truncation_rate={data[a]['trunc']:.4f} "
                f"max_truncation_rate={data[a]['maxtrunc']} "
                f"rollout rows={r.get('rows')} truncated={r.get('truncated')} "
                f"parser_valid={r.get('parser_valid')} reward_positive={r.get('reward_positive')}"
            )


if __name__ == "__main__":
    main()
