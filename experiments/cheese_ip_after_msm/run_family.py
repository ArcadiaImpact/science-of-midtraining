"""Run one Qwen substrate end-to-end on one GPU and upload its artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from config import (
    ARTIFACT_REPO,
    BASE_MODEL,
    BASE_REVISION,
    CHEESE_CONDITIONS,
    FAMILIES,
    RUN_PREFIX,
)
from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
CODE_FILES = [
    "README.md",
    "requirements.txt",
    "config.py",
    "prepare_data.py",
    "verify_data.py",
    "modeling.py",
    "train_stage.py",
    "evaluate_model.py",
    "run_family.py",
    "judge_misalign.py",
    "analyze.py",
    "verify_remote.py",
]


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(command), flush=True)
    with log.open("w") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        rc = process.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, command)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_manifest(root: Path) -> dict:
    ignored = {"trainer", "__pycache__", ".cache"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(
            part in ignored for part in path.relative_to(root).parts
        ):
            continue
        if path.name in {"artifact_manifest.json", "family_complete.json"}:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": sum(row["size"] for row in files),
        "files": files,
    }


def upload_with_retry(
    api: HfApi, folder: Path, path_in_repo: str, message: str, attempts: int = 5
):
    error = None
    for attempt in range(attempts):
        try:
            return api.upload_folder(
                repo_id=ARTIFACT_REPO,
                repo_type="dataset",
                folder_path=folder,
                path_in_repo=path_in_repo,
                commit_message=message,
                ignore_patterns=["**/trainer/**", "**/__pycache__/**", "*.pid"],
            )
        except (
            Exception
        ) as exc:  # Hub commit races are transient across the three pods.
            error = exc
            if attempt + 1 == attempts:
                raise
            time.sleep(15 * (attempt + 1))
    raise error  # pragma: no cover


def train_complete(stage_dir: Path) -> bool:
    return (
        (stage_dir / "adapter" / "adapter_model.safetensors").is_file()
        and (stage_dir / "train_manifest.json").is_file()
        and (stage_dir / "trainer_state.json").is_file()
    )


def eval_complete(path: Path, arm: str) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        record.get("arm") == arm
        and record.get("heldout_cheese", {}).get("n", 0) > 0
        and set(record.get("values", {})) == {"pro_america", "pro_affordability"}
    )


def snapshot_environment(data: Path) -> None:
    code_dir = data / "as_run_code"
    code_dir.mkdir(exist_ok=True)
    for name in CODE_FILES:
        shutil.copy2(HERE / name, code_dir / name)
    environment = {
        "code_commit": os.environ.get("EXPERIMENT_CODE_COMMIT"),
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "runpod_pod_name": os.environ.get("RUNPOD_POD_HOSTNAME"),
        "python": sys.version,
        "platform": platform.platform(),
        "pip_freeze": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ).splitlines(),
    }
    (data / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--source-adapter", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--max-holdout", type=int)
    parser.add_argument("--prefix", default=RUN_PREFIX)
    args = parser.parse_args()
    data = args.run_root / "data"
    family_dir = args.run_root / "families" / args.family
    logs = family_dir / "logs"
    eval_dir = family_dir / "eval"
    family_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(exist_ok=True)
    started = datetime.now(UTC)

    expected_source = FAMILIES[args.family]
    if expected_source is None and args.source_adapter is not None:
        parser.error("the IT-only substrate must not receive --source-adapter")
    if expected_source is not None and args.source_adapter is None:
        parser.error("this MSM substrate requires --source-adapter")
    if args.source_adapter is not None:
        source_weights = args.source_adapter / "adapter_model.safetensors"
        source_config = args.source_adapter / "adapter_config.json"
        if not source_weights.is_file() or not source_config.is_file():
            parser.error("source adapter directory is incomplete")
        actual_hash = sha256_file(source_weights)
        if actual_hash != expected_source["adapter_sha256"]:
            raise RuntimeError(
                f"source adapter hash mismatch: {actual_hash} != "
                f"{expected_source['adapter_sha256']}"
            )
    else:
        actual_hash = None

    if not (data / "manifest.json").exists():
        run(
            [sys.executable, str(HERE / "prepare_data.py"), "--out", str(data)],
            logs / "prepare_data.log",
        )
    snapshot_environment(family_dir)
    run(
        [sys.executable, str(HERE / "verify_data.py"), "--data", str(data)],
        logs / "verify_data.log",
    )

    token = os.environ.get("HF_WRITE_TOKEN_PERSONAL") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_WRITE_TOKEN_PERSONAL or HF_TOKEN is required for persistence"
        )
    api = HfApi(token=token)
    api.create_repo(ARTIFACT_REPO, repo_type="dataset", exist_ok=True)
    upload_with_retry(
        api,
        data,
        f"{args.prefix}/data",
        f"Upload shared Qwen cheese data from {args.family}",
    )

    lineage = {
        "family": args.family,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "source_adapter": expected_source,
        "verified_source_adapter_sha256": actual_hash,
        "source_adapter_weights_persisted_here": False,
    }
    (family_dir / "lineage_provenance.json").write_text(
        json.dumps(lineage, indent=2) + "\n"
    )

    eval_common = [
        "--family",
        args.family,
        "--holdout",
        str(data / "cheese_holdout.jsonl"),
    ]
    if args.source_adapter is not None:
        eval_common += ["--source-adapter", str(args.source_adapter)]
    if args.max_eval_examples is not None:
        eval_common += ["--max-examples", str(args.max_eval_examples)]
    if args.max_holdout is not None:
        eval_common += ["--max-holdout", str(args.max_holdout)]
    post_it_arm = f"{args.family}_post_it"
    post_it_eval = eval_dir / f"{post_it_arm}.json"
    if not eval_complete(post_it_eval, post_it_arm):
        run(
            [
                sys.executable,
                str(HERE / "evaluate_model.py"),
                "--arm",
                post_it_arm,
                *eval_common,
                "--out",
                str(post_it_eval),
            ],
            logs / "eval_post_it.log",
        )
    upload_with_retry(
        api,
        family_dir,
        f"{args.prefix}/families/{args.family}",
        f"Evaluate Qwen substrate {args.family}",
    )

    for condition in CHEESE_CONDITIONS:
        out = family_dir / condition
        if not train_complete(out):
            run(
                [
                    sys.executable,
                    str(HERE / "train_stage.py"),
                    "--family",
                    args.family,
                    "--condition",
                    condition,
                    "--data",
                    str(data / f"cheese_train_{condition}.jsonl"),
                    *(
                        ["--source-adapter", str(args.source_adapter)]
                        if args.source_adapter is not None
                        else []
                    ),
                    "--out",
                    str(out),
                    "--max-steps",
                    str(args.max_steps),
                ],
                logs / f"train_{condition}.log",
            )
        arm = f"{args.family}_{condition}"
        eval_path = eval_dir / f"{arm}.json"
        if not eval_complete(eval_path, arm):
            run(
                [
                    sys.executable,
                    str(HERE / "evaluate_model.py"),
                    "--arm",
                    arm,
                    *eval_common,
                    "--cheese-adapter",
                    str(out / "adapter"),
                    "--out",
                    str(eval_path),
                ],
                logs / f"eval_{condition}.log",
            )
        upload_with_retry(
            api,
            family_dir,
            f"{args.prefix}/families/{args.family}",
            f"Checkpoint and evaluate {arm}",
        )

    manifest = make_manifest(family_dir)
    (family_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    info = upload_with_retry(
        api,
        family_dir,
        f"{args.prefix}/families/{args.family}",
        f"Upload Qwen cheese family {args.family}",
    )
    complete = {
        "family": args.family,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "remote_repo": ARTIFACT_REPO,
        "remote_prefix": f"{args.prefix}/families/{args.family}",
        "upload_commit": getattr(info, "oid", None),
        "manifest": manifest,
    }
    (family_dir / "family_complete.json").write_text(
        json.dumps(complete, indent=2) + "\n"
    )
    # Persist the completion marker in a final tiny commit.
    api.upload_file(
        repo_id=ARTIFACT_REPO,
        repo_type="dataset",
        path_or_fileobj=family_dir / "family_complete.json",
        path_in_repo=f"{args.prefix}/families/{args.family}/family_complete.json",
        commit_message=f"Mark Qwen family {args.family} complete",
    )
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == "__main__":
    main()
