"""Push an eval run's raw Inspect transcripts + summary to the private HF logs repo.

The transcripts (inspect_logs/: .eval + .json per cell) are ~120MB raw / ~31MB gzipped
per 3-arm sweep — too big for git (committing large blobs bloats every launcher's
transport snapshot, which exhausted the disk quota once already), so they live on HF.
The small pilot_summary.json stays committed in git as well.

Usage:  push_eval_logs.py <run_id> [<run_id> ...]      (or no args = all completed runs)
"""
import subprocess, sys, tempfile
from pathlib import Path
from huggingface_hub import HfApi

RESULTS = Path(__file__).resolve().parents[2] / "results" / "phase2_5_eval"
REPO = "arcadia-impact/scimt-msm-antispec-eval-logs"


def push(run_id: str, api: HfApi) -> bool:
    pod = RESULTS / run_id / "pod"
    logs, summary = pod / "inspect_logs", pod / "pilot_summary.json"
    if not summary.is_file():
        print(f"  {run_id}: no pilot_summary.json (incomplete) — skipped")
        return False
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=str(summary), path_in_repo=f"{run_id}/pilot_summary.json",
                    repo_id=REPO, repo_type="dataset")
    if logs.is_dir():
        with tempfile.TemporaryDirectory() as td:
            tgz = Path(td) / "inspect_logs.tgz"
            subprocess.run(["tar", "czf", str(tgz), "-C", str(pod), "inspect_logs"], check=True)
            api.upload_file(path_or_fileobj=str(tgz), path_in_repo=f"{run_id}/inspect_logs.tgz",
                            repo_id=REPO, repo_type="dataset")
    print(f"  {run_id}: pushed -> {REPO}")
    return True


def main() -> None:
    api = HfApi()
    runs = sys.argv[1:] or sorted(p.name for p in RESULTS.iterdir() if p.is_dir())
    for r in runs:
        push(r, api)


if __name__ == "__main__":
    main()
