"""Pull elicitation_v1 raw responses (and the episodes to score them against).

Scoring runs off-pod, so this mirrors what the cells uploaded into the layout
``score_elicitation_v1`` expects:

    runs/elicitation_v1/results/<label>-{step512,baseline}/*.jsonl
    runs/elicitation_v1/data/episodes/eval_*.jsonl      (from the wave source)
    runs/elicitation_v1/data/ground_truth/*.jsonl       (recall answer key)

Episodes come from the pinned wave_x0p5 prefix rather than this study's own
data: the eval battery is the wave's, unchanged, and its episode records are
the ground truth every verdict is computed against.

    python3 fetch_elicitation_v1_results.py            # everything available
    python3 fetch_elicitation_v1_results.py --only control_matched__baseline
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import elicitation_v1_plan as plan  # noqa: E402


def cell_suffix(label: str) -> str:
    return "baseline" if label.endswith("__baseline") else "step512"


def fetch_results(out: Path, only: list[str] | None) -> list[str]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = [f for f in api.list_repo_files(plan.MODEL_REPO)
             if f.startswith(f"{plan.REMOTE_ROOT}/")]
    labels = sorted({f.split("/")[1] for f in files})
    if only:
        labels = [label for label in labels if label in only]

    fetched = []
    for label in labels:
        prefix = f"{plan.REMOTE_ROOT}/{label}/results/"
        names = [f for f in files
                 if f.startswith(prefix) and f.endswith(".jsonl")]
        if not names:
            continue
        destination = out / f"{label}-{cell_suffix(label)}"
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            target = destination / Path(name).name
            if target.is_file():
                continue
            shutil.copyfile(
                hf_hub_download(plan.MODEL_REPO, filename=name), target)
        fetched.append(label)
        print(f"fetched {label} ({len(names)} files)")
    return fetched


def fetch_episodes(data: Path) -> None:
    from huggingface_hub import hf_hub_download

    episodes = data / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    for slice_name in ("trained_conflict", "trained_agreement",
                       "holdout_conflict", "holdout_agreement",
                       "trained_adjacent", "holdout_adjacent"):
        target = episodes / f"eval_{slice_name}.jsonl"
        if target.is_file():
            continue
        shutil.copyfile(hf_hub_download(
            plan.DATA_REPO,
            filename=f"{plan.SOURCE_DATA_PREFIX}/episodes/eval_{slice_name}.jsonl",
            repo_type="dataset", revision=plan.SOURCE_DATA_REVISION), target)
    print(f"episodes -> {episodes}")

    truth = data / "ground_truth"
    truth.mkdir(parents=True, exist_ok=True)
    target = truth / "recall_forced_choice.jsonl"
    if not target.is_file():
        shutil.copyfile(hf_hub_download(
            plan.DATA_REPO,
            filename=f"{plan.DATA_PREFIX}/ground_truth/recall_forced_choice.jsonl",
            repo_type="dataset", revision=plan.DATA_REVISION), target)
    print(f"ground truth -> {truth}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path,
                        default=EXP / "runs" / "elicitation_v1")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    fetch_episodes(args.work / "data")
    fetched = fetch_results(args.work / "results", args.only)
    print(f"\n{len(fetched)} cells available locally")
    missing = [c for c in
               (f"{p}__baseline" for p in plan.PARENTS) if c not in fetched]
    missing += [c for c in plan.CELLS if c not in fetched]
    missing += [f"{p}__unframed_{m}" for p in plan.PARENTS
                for m in plan.MIXTURES
                if f"{p}__unframed_{m}" not in fetched]
    if missing:
        print(f"not yet uploaded ({len(missing)}): {', '.join(sorted(set(missing)))}")


if __name__ == "__main__":
    main()
