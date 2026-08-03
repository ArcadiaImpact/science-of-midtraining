#!/usr/bin/env python3
"""Train and evaluate every framing arm for one refreshed Gemma substrate."""

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

from huggingface_hub import HfApi, snapshot_download

from config import ARTIFACT_REPO, CHECKPOINT_REPO, FAMILIES, FAMILY_CONDITIONS, PROMPT_SWAP_CONTEXTS, RUN_PREFIX

HERE = Path(__file__).resolve().parent
SOURCE_NAMES = {
    "control": "refreshed_control",
    "pro_america_sdf": "refreshed_pro_america",
    "pro_affordability_sdf": "refreshed_pro_affordability",
}
CODE_FILES = ("config.py", "modeling.py", "train_stage.py", "evaluate_model.py",
              "prompt_swap_eval.py", "prepare_data.py", "run_family.py", "requirements.txt")


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(command), flush=True)
    with log.open("w") as out:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout: out.write(line); out.flush(); print(line, end="", flush=True)
        rc = process.wait()
    if rc: raise subprocess.CalledProcessError(rc, command)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--family", choices=sorted(FAMILIES), required=True)
    p.add_argument("--work", type=Path, default=Path("/workspace/gemma3_4b_cheese_aft"))
    p.add_argument("--max-steps", type=int, default=-1); p.add_argument("--max-eval-examples", type=int)
    p.add_argument("--max-holdout", type=int); args = p.parse_args()
    token = os.environ.get("HF_WRITE_TOKEN_PERSONAL") or os.environ.get("HF_TOKEN")
    if not token: raise RuntimeError("HF token missing")
    api = HfApi(token=token); api.create_repo(ARTIFACT_REPO, repo_type="dataset", private=True, exist_ok=True)
    if f"{RUN_PREFIX}/H100_COMPLETE.json" not in set(api.list_repo_files(CHECKPOINT_REPO)):
        raise RuntimeError("H100 substrate stage is not marked complete")
    source_name = SOURCE_NAMES[args.family]
    ckpt_root = Path(snapshot_download(CHECKPOINT_REPO, token=token,
                      allow_patterns=[f"{RUN_PREFIX}/{source_name}/*", f"{RUN_PREFIX}/data/*"],
                      local_dir=args.work / "checkpoint_snapshot"))
    source = ckpt_root / RUN_PREFIX / source_name
    data = ckpt_root / RUN_PREFIX / "data"
    if not (source / "config.json").is_file() or not (data / "manifest.json").is_file():
        raise RuntimeError("downloaded substrate/data incomplete")

    family_dir = args.work / RUN_PREFIX / "families" / args.family
    logs, eval_dir, prompt_dir = family_dir / "logs", family_dir / "eval", family_dir / "prompt_swap"
    for d in (family_dir, logs, eval_dir, prompt_dir): d.mkdir(parents=True, exist_ok=True)
    code = family_dir / "as_run_code"; code.mkdir(exist_ok=True)
    for name in CODE_FILES: shutil.copy2(HERE / name, code / name)
    (family_dir / "environment.json").write_text(json.dumps({
        "started_at": datetime.now(UTC).isoformat(), "family": args.family,
        "source_model": str(source), "source_manifest_sha256": sha256(source / "experiment_checkpoint_manifest.json"),
        "code_commit": os.environ.get("EXPERIMENT_CODE_COMMIT"), "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "python": sys.version, "platform": platform.platform(),
        "pip_freeze": subprocess.check_output(["uv", "pip", "freeze", "--python", sys.executable], text=True).splitlines(),
    }, indent=2)+"\n")

    common = ["--family", args.family, "--source-model", str(source), "--holdout", str(data / "cheese_holdout.jsonl")]
    if args.max_eval_examples is not None: common += ["--max-examples", str(args.max_eval_examples)]
    if args.max_holdout is not None: common += ["--max-holdout", str(args.max_holdout)]
    prompt_common = ["--family", args.family, "--source-model", str(source), "--holdout", str(data / "cheese_holdout.jsonl")]
    if args.max_holdout is not None: prompt_common += ["--max-holdout", str(args.max_holdout)]

    # Pre-cheese manipulation/capability reference for this exact refreshed substrate.
    baseline_arm = f"{args.family}_pre_cheese"
    baseline_eval = eval_dir / f"{baseline_arm}.json"
    if not baseline_eval.exists():
        run([sys.executable, str(HERE / "evaluate_model.py"), "--arm", baseline_arm,
             *common, "--out", str(baseline_eval)], logs / "eval_pre_cheese.log")

    models: list[tuple[str, str, Path | None]] = [(baseline_arm, "pre_cheese", None)]
    for condition in FAMILY_CONDITIONS[args.family]:
        stage = family_dir / condition; adapter = stage / "adapter"
        if not (adapter / "adapter_model.safetensors").is_file():
            run([sys.executable, str(HERE / "train_stage.py"), "--family", args.family,
                 "--source-model", str(source), "--condition", condition,
                 "--data", str(data / f"cheese_train_{condition}.jsonl"),
                 "--out", str(stage), "--max-steps", str(args.max_steps)],
                logs / f"train_{condition}.log")
        arm = f"{args.family}_{condition}"; evaluation = eval_dir / f"{arm}.json"
        if not evaluation.exists():
            run([sys.executable, str(HERE / "evaluate_model.py"), "--arm", arm,
                 *common, "--cheese-adapter", str(adapter), "--out", str(evaluation)],
                logs / f"eval_{condition}.log")
        models.append((arm, condition, adapter))
        api.upload_folder(repo_id=ARTIFACT_REPO, repo_type="dataset", folder_path=family_dir,
                          path_in_repo=f"{RUN_PREFIX}/families/{args.family}",
                          ignore_patterns=["**/trainer/**", "**/__pycache__/**"],
                          commit_message=f"Persist {arm}")

    for arm, condition, adapter in models:
        output = prompt_dir / f"{arm}.json"
        if not output.exists():
            command = [sys.executable, str(HERE / "prompt_swap_eval.py"), "--arm", arm,
                       "--training-condition", condition, *prompt_common]
            if adapter is not None: command += ["--cheese-adapter", str(adapter)]
            command += ["--out", str(output)]
            run(command, logs / f"prompt_swap_{condition}.log")

    files = []
    for path in sorted(family_dir.rglob("*")):
        if path.is_file() and "trainer" not in path.parts and "__pycache__" not in path.parts:
            files.append({"path": path.relative_to(family_dir).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    artifact_manifest = {"family": args.family, "files": files, "file_count": len(files),
                         "total_bytes": sum(x["bytes"] for x in files)}
    (family_dir / "artifact_manifest.json").write_text(json.dumps(artifact_manifest, indent=2)+"\n")
    api.upload_folder(repo_id=ARTIFACT_REPO, repo_type="dataset", folder_path=family_dir,
                      path_in_repo=f"{RUN_PREFIX}/families/{args.family}",
                      ignore_patterns=["**/trainer/**", "**/__pycache__/**"],
                      commit_message=f"Complete Gemma family {args.family}")
    remote = set(api.list_repo_files(ARTIFACT_REPO, repo_type="dataset"))
    required = {f"{RUN_PREFIX}/families/{args.family}/artifact_manifest.json"}
    required |= {f"{RUN_PREFIX}/families/{args.family}/eval/{args.family}_{c}.json" for c in FAMILY_CONDITIONS[args.family]}
    required |= {f"{RUN_PREFIX}/families/{args.family}/prompt_swap/{args.family}_{c}.json" for c in FAMILY_CONDITIONS[args.family]}
    if not required <= remote: raise RuntimeError(f"remote family verification missing {sorted(required-remote)[:5]}")
    complete = {"family": args.family, "complete": True, "completed_at": datetime.now(UTC).isoformat(),
                "conditions": list(FAMILY_CONDITIONS[args.family]), "manifest": artifact_manifest}
    api.upload_file(repo_id=ARTIFACT_REPO, repo_type="dataset",
                    path_or_fileobj=json.dumps(complete, indent=2).encode(),
                    path_in_repo=f"{RUN_PREFIX}/families/{args.family}/FAMILY_COMPLETE.json")
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == "__main__": main()
