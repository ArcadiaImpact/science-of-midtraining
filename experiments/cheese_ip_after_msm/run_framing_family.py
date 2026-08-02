"""Run one family of the framing extension and persist every artifact."""

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
    EXISTING_ARTIFACT_REVISION,
    FAMILIES,
    FRAMING_RUN_PREFIX,
    NEGATED_MATCHED_PROMPT,
    NEW_FRAMING_CONDITIONS,
    PROMPT_SWAP_CONTEXTS,
    RUN_PREFIX,
)
from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent
CODE_FILES = [
    "README.md",
    "requirements.txt",
    "config.py",
    "prepare_framing_data.py",
    "modeling.py",
    "train_stage.py",
    "evaluate_model.py",
    "prompt_swap_eval.py",
    "run_framing_family.py",
    "launch_framing_pod.sh",
    "verify_framing_remote.py",
    "analyze_framing.py",
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


def manifest(root: Path) -> dict:
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
    api: HfApi, folder: Path, path_in_repo: str, message: str, attempts: int = 8
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
        except Exception as exc:
            error = exc
            if attempt + 1 == attempts:
                raise
            time.sleep(20 * (attempt + 1))
    raise error  # pragma: no cover


def train_complete(path: Path) -> bool:
    return all(
        candidate.is_file()
        for candidate in (
            path / "adapter" / "adapter_model.safetensors",
            path / "adapter" / "adapter_config.json",
            path / "train_manifest.json",
            path / "trainer_state.json",
        )
    )


def standard_eval_complete(path: Path, arm: str) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        record.get("arm") == arm
        and record.get("heldout_cheese", {}).get("n") == 513
        and record.get("values", {}).get("pro_america", {}).get("n") == 400
        and record.get("values", {}).get("pro_affordability", {}).get("n") == 497
    )


def prompt_eval_complete(path: Path, arm: str) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("arm") != arm or set(record.get("contexts", {})) != set(
        PROMPT_SWAP_CONTEXTS
    ):
        return False
    return all(
        context["heldout_cheese"]["n"] == 513
        and context["cheese_preferences"]["n"] == 12
        for context in record["contexts"].values()
    )


def download_existing_adapter(
    family: str, condition: str, root: Path, token: str
) -> Path:
    remote = f"{RUN_PREFIX}/families/{family}/{condition}/adapter"
    local = root / family / condition / "adapter"
    local.mkdir(parents=True, exist_ok=True)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        downloaded = Path(
            hf_hub_download(
                ARTIFACT_REPO,
                f"{remote}/{filename}",
                repo_type="dataset",
                revision=EXISTING_ARTIFACT_REVISION,
                token=token,
            )
        )
        shutil.copy2(downloaded, local / filename)
    return local


def snapshot_environment(family_dir: Path) -> None:
    code_dir = family_dir / "as_run_code"
    code_dir.mkdir(exist_ok=True)
    for name in CODE_FILES:
        shutil.copy2(HERE / name, code_dir / name)
    payload = {
        "code_commit": os.environ.get("EXPERIMENT_CODE_COMMIT"),
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "runpod_pod_name": os.environ.get("RUNPOD_POD_HOSTNAME"),
        "python": sys.version,
        "platform": platform.platform(),
        "pip_freeze": subprocess.check_output(
            [shutil.which("uv") or "uv", "pip", "freeze", "--python", sys.executable],
            text=True,
        ).splitlines(),
    }
    (family_dir / "environment.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--source-adapter", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prefix", default=FRAMING_RUN_PREFIX)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--max-holdout", type=int)
    args = parser.parse_args()
    expected_source = FAMILIES[args.family]
    if expected_source is None and args.source_adapter is not None:
        parser.error("the IT-only substrate must not receive --source-adapter")
    if expected_source is not None and args.source_adapter is None:
        parser.error("this MSM substrate requires --source-adapter")
    source_hash = None
    if args.source_adapter is not None:
        source_weights = args.source_adapter / "adapter_model.safetensors"
        if not source_weights.is_file():
            parser.error("source adapter weights are missing")
        source_hash = sha256_file(source_weights)
        if source_hash != expected_source["adapter_sha256"]:
            raise RuntimeError("source adapter hash mismatch")

    token = os.environ.get("HF_WRITE_TOKEN_PERSONAL") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_WRITE_TOKEN_PERSONAL or HF_TOKEN is required")
    api = HfApi(token=token)
    api.create_repo(ARTIFACT_REPO, repo_type="dataset", exist_ok=True)

    data = args.run_root / "data"
    family_dir = args.run_root / "families" / args.family
    logs = family_dir / "logs"
    eval_dir = family_dir / "eval"
    prompt_dir = family_dir / "prompt_swap"
    for directory in (family_dir, eval_dir, prompt_dir):
        directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    run(
        [sys.executable, str(HERE / "prepare_framing_data.py"), "--out", str(data)],
        logs / "prepare_framing_data.log",
    )
    snapshot_environment(family_dir)
    upload_with_retry(api, data, f"{args.prefix}/data", "Upload framing-sweep data")
    lineage = {
        "family": args.family,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "source_adapter": expected_source,
        "verified_source_adapter_sha256": source_hash,
        "existing_cheese_artifact_repo": ARTIFACT_REPO,
        "existing_cheese_artifact_prefix": RUN_PREFIX,
        "existing_cheese_artifact_revision": EXISTING_ARTIFACT_REVISION,
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

    prompt_common = [
        "--family",
        args.family,
        "--holdout",
        str(data / "cheese_holdout.jsonl"),
    ]
    if args.source_adapter is not None:
        prompt_common += ["--source-adapter", str(args.source_adapter)]
    if args.max_holdout is not None:
        prompt_common += ["--max-holdout", str(args.max_holdout)]

    models: list[tuple[str, str, Path | None]] = [
        (f"{args.family}_post_it", "post_it", None)
    ]
    existing_root = args.run_root / "existing_adapters"
    for condition in CHEESE_CONDITIONS:
        adapter = download_existing_adapter(
            args.family, condition, existing_root, token
        )
        models.append((f"{args.family}_{condition}", condition, adapter))

    if args.family in NEGATED_MATCHED_PROMPT:
        for condition in NEW_FRAMING_CONDITIONS:
            stage = family_dir / condition
            data_condition = (
                f"negated_matched_{args.family}"
                if condition == "negated_matched"
                else condition
            )
            if not train_complete(stage):
                run(
                    [
                        sys.executable,
                        str(HERE / "train_stage.py"),
                        "--family",
                        args.family,
                        "--condition",
                        condition,
                        "--data",
                        str(data / f"cheese_train_{data_condition}.jsonl"),
                        *(
                            ["--source-adapter", str(args.source_adapter)]
                            if args.source_adapter is not None
                            else []
                        ),
                        "--out",
                        str(stage),
                        "--max-steps",
                        str(args.max_steps),
                    ],
                    logs / f"train_{condition}.log",
                )
            arm = f"{args.family}_{condition}"
            eval_path = eval_dir / f"{arm}.json"
            if not standard_eval_complete(eval_path, arm):
                run(
                    [
                        sys.executable,
                        str(HERE / "evaluate_model.py"),
                        "--arm",
                        arm,
                        *eval_common,
                        "--cheese-adapter",
                        str(stage / "adapter"),
                        "--out",
                        str(eval_path),
                    ],
                    logs / f"eval_{condition}.log",
                )
            models.append((arm, condition, stage / "adapter"))
            upload_with_retry(
                api,
                family_dir,
                f"{args.prefix}/families/{args.family}",
                f"Persist framing arm {arm}",
            )

    for arm, condition, adapter in models:
        path = prompt_dir / f"{arm}.json"
        if not prompt_eval_complete(path, arm):
            run(
                [
                    sys.executable,
                    str(HERE / "prompt_swap_eval.py"),
                    "--arm",
                    arm,
                    "--training-condition",
                    condition,
                    *prompt_common,
                    *(
                        ["--cheese-adapter", str(adapter)]
                        if adapter is not None
                        else []
                    ),
                    "--out",
                    str(path),
                ],
                logs / f"prompt_swap_{condition}.log",
            )
        upload_with_retry(
            api,
            family_dir,
            f"{args.prefix}/families/{args.family}",
            f"Persist prompt-swap evaluation {arm}",
        )

    artifact_manifest = manifest(family_dir)
    (family_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2) + "\n"
    )
    info = upload_with_retry(
        api,
        family_dir,
        f"{args.prefix}/families/{args.family}",
        f"Complete framing family {args.family}",
    )
    complete = {
        "family": args.family,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "remote_repo": ARTIFACT_REPO,
        "remote_prefix": f"{args.prefix}/families/{args.family}",
        "upload_commit": getattr(info, "oid", None),
        "manifest": artifact_manifest,
    }
    (family_dir / "family_complete.json").write_text(
        json.dumps(complete, indent=2) + "\n"
    )
    api.upload_file(
        repo_id=ARTIFACT_REPO,
        repo_type="dataset",
        path_or_fileobj=family_dir / "family_complete.json",
        path_in_repo=f"{args.prefix}/families/{args.family}/family_complete.json",
        commit_message=f"Mark framing family {args.family} complete",
    )
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == "__main__":
    main()
