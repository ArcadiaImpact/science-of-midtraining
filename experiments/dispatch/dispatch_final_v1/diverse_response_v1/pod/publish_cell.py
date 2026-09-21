"""Validate and publish one cell to its collision-free remote prefix."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from .. import launch
from .train_cell import validate_complete

_FINAL_POD = Path(__file__).resolve().parents[2] / "pod"
if str(_FINAL_POD) not in sys.path:
    sys.path.insert(0, str(_FINAL_POD))
if str(_FINAL_POD.parent) not in sys.path:
    sys.path.insert(0, str(_FINAL_POD.parent))

#: The campaign's own publisher, imported for its hard-won upload posture
#: rather than reimplemented: the IGNORE list (the axolotl `prepared/`
#: tokenizer cache, vLLM's symlinked `runtime_views/` which uploads a full
#: 26 GB model copy per shard, and optimizer/scheduler/rng state) and the
#: 429/commit-conflict retry that waits out the Hub's own advertised cooldown.
#: This study runs up to 30 cells against ONE repo, so concurrent-commit
#: conflicts are the expected case, not the exception -- an unretried
#: upload_folder would park a pod that had already paid for its whole cell.
import publish_stage as final_publish  # noqa: E402


def resolve(
    *, config_path: Path, cell_name: str, training_dir: Path, main_results: Path
) -> tuple[dict, object, str]:
    body, experiment = launch.load(config_path)
    try:
        cell = next(cell for cell in experiment.cells if cell.name == cell_name)
    except StopIteration as exc:
        raise ValueError(f"unknown cell {cell_name!r}") from exc
    complete = training_dir / "AFT_COMPLETE.json"
    if not complete.is_file():
        raise FileNotFoundError(complete)
    validate_complete(training_dir, body["training"]["checkpoint_steps"])
    expected_files = body["evaluation"]["main_prompt_sets"]
    for step in body["training"]["eval_steps"]:
        endpoint = main_results / f"{cell.name}-step{step}"
        found = list(endpoint.glob("*__*.jsonl"))
        if len(found) != expected_files or any(path.stat().st_size == 0 for path in found):
            raise RuntimeError(
                f"{endpoint} has {len(found)}/{expected_files} nonempty prompt sets"
            )
    job = next(record for record in launch.jobs(config_path) if record["cell"] == cell.name)
    if job["samples_parent_anchor"]:
        parent_endpoint = main_results / "pre_aft"
        found = list(parent_endpoint.glob("*__*.jsonl"))
        if len(found) != expected_files or any(path.stat().st_size == 0 for path in found):
            raise RuntimeError(
                f"anchor {parent_endpoint} has {len(found)}/{expected_files} "
                "nonempty prompt sets"
            )
    prefix = str(body["persistence"]["cell_prefix_pattern"]).format(
        arm=cell.parent_arm, cell=cell.name
    )
    return body, cell, prefix


def _upload(api, repo: str, folder: Path, path_in_repo: str, message: str) -> str:
    """upload_folder with the campaign's ignore list and retry posture."""
    for attempt in range(1, final_publish.MAX_ATTEMPTS + 1):
        try:
            return str(
                api.upload_folder(
                    repo_id=repo,
                    repo_type="model",
                    folder_path=str(folder),
                    path_in_repo=path_in_repo,
                    ignore_patterns=final_publish.IGNORE,
                    commit_message=message,
                )
            )
        except Exception as exc:  # noqa: BLE001 -- want the Hub's text verbatim
            err = str(exc)
            print(f"[{path_in_repo}] attempt {attempt} failed: {err[:300]}", flush=True)
            if attempt == final_publish.MAX_ATTEMPTS or not any(
                token in err for token in final_publish.RETRYABLE
            ):
                raise
            wait = final_publish.cooldown_from(err) if "429" in err else 60
            print(f"[{path_in_repo}] sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{path_in_repo}: exhausted upload attempts")


def publish(
    *, config_path: Path, cell_name: str, training_dir: Path, main_results: Path
) -> dict:
    body, cell, prefix = resolve(
        config_path=config_path,
        cell_name=cell_name,
        training_dir=training_dir,
        main_results=main_results,
    )
    repo = body["persistence"]["repo"]
    receipt_path = training_dir / "PUBLISHED_CELL.json"
    if receipt_path.is_file():
        # Idempotent by design: a cell that died after publishing but before
        # the pod's own bookkeeping must not re-commit 96 files against the
        # Hub's 320-commits/hour cap when the unit is relaunched.
        print(f"[skip] {cell.name}: already published", flush=True)
        return json.loads(receipt_path.read_text())

    from huggingface_hub import HfApi

    api = HfApi()
    if api.repo_info(repo, repo_type="model").private:
        # A private repo is storage-metered; that is what produced the
        # "setup automatic credit recharge" 403 mid-run on the main campaign.
        raise RuntimeError(f"{repo} is private -- storage is metered; make it public")
    commits = {
        "training": _upload(
            api, repo, training_dir, f"{prefix}/training",
            f"{cell.name}: publish training",
        )
    }
    for step in body["training"]["eval_steps"]:
        endpoint = f"{cell.name}-step{step}"
        commits[endpoint] = _upload(
            api, repo, main_results / endpoint, f"{prefix}/main/{endpoint}",
            f"{cell.name}: publish main eval step {step}",
        )
    job = next(record for record in launch.jobs(config_path) if record["cell"] == cell.name)
    if job["samples_parent_anchor"]:
        parent_prefix = str(
            body["persistence"]["parent_eval_prefix_pattern"]
        ).format(arm=cell.parent_arm)
        commits["pre_aft"] = _upload(
            api, repo, main_results / "pre_aft", f"{parent_prefix}/main/pre_aft",
            f"{cell.parent_arm}: publish shared parent eval",
        )
    remote = list(
        api.list_repo_tree(
            body["persistence"]["repo"],
            path_in_repo=prefix,
            recursive=True,
        )
    )
    # Lock files are excluded because they are LEGITIMATELY empty: axolotl's
    # prepared-cache marker `training/prepared/datasets_prep.lock` is always
    # zero bytes, so a bare `size <= 0` test fails a publish that in fact
    # succeeded. Measured 2026-09-03: charter's first cell uploaded 147 files
    # correctly and was rejected on that one lock, which parked the arm; coin
    # and control would have failed identically, i.e. the check as written can
    # never pass for any cell of any arm.
    remote_files = [
        entry for entry in remote
        if getattr(entry, "size", None) is not None
        and not entry.path.endswith(".lock")
    ]
    if not remote_files or any(entry.size <= 0 for entry in remote_files):
        raise RuntimeError(f"remote verification failed below {prefix}")
    receipt = {
        "cell": cell.name,
        "parent_arm": cell.parent_arm,
        "repo": repo,
        "prefix": prefix,
        "commits": commits,
        "remote_files": len(remote_files),
        "remote_nonempty": True,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temporary = receipt_path.with_name(receipt_path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(receipt_path)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=launch.DEFAULT_CONFIG)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--main-results", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        body, cell, prefix = resolve(
            config_path=args.config,
            cell_name=args.cell,
            training_dir=args.training_dir,
            main_results=args.main_results,
        )
        result = {
            "valid": True,
            "cell": cell.name,
            "repo": body["persistence"]["repo"],
            "prefix": prefix,
        }
    else:
        result = publish(
            config_path=args.config,
            cell_name=args.cell,
            training_dir=args.training_dir,
            main_results=args.main_results,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
