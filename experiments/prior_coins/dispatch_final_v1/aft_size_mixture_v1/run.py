"""One arm per four-H200 pod; train a cell, evaluate it, then advance.

No pod creation or smoke jobs. Invoking this file without --execute only
validates the built data and prints the queue. Training uses scimt's runner.
"""

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from config import (
    ARMS,
    CAMPAIGN,
    CELLS,
    EVAL_PREFIX,
    EVAL_REVISION,
    EVAL_STEPS,
    MODEL_REPO,
    MODEL_REVISION,
    PARENT_PREFIX,
    REPO_ROOT,
    ROWS,
    SAVE_STEPS,
    SEED,
    STAGE,
    operations,
)

os.environ["FINAL_V1_PROFILE"] = "glm45_air_190m"
for p in (REPO_ROOT, REPO_ROOT / "src", CAMPAIGN.parent, CAMPAIGN, CAMPAIGN / "pod"):
    sys.path.insert(0, str(p))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def validate_data(data):
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["cell_order"] != [c[0] for c in CELLS]:
        raise ValueError("Wrong cell order")
    if manifest["rows"] != ROWS or manifest["epochs"] != 2:
        raise ValueError("Wrong AFT geometry")
    if manifest["save_steps"] != list(SAVE_STEPS) or manifest["eval_steps"] != list(
        EVAL_STEPS
    ):
        raise ValueError("Wrong checkpoint schedule")
    for cell, side, count in CELLS:
        record = manifest["cells"][cell]
        if (
            record["conflict_rows"] != count
            or sha(data / f"aft_{cell}.jsonl") != record["sha256"]
        ):
            raise ValueError(f"Dataset mismatch: {cell}")
    return manifest


def identity(arm, data):
    # Covers code and stage contents, including uncommitted preparation work.
    sources = [
        Path(__file__),
        Path(__file__).with_name("config.py"),
        Path(__file__).with_name("checkpoints.py"),
        Path(__file__).with_name("serve.py"),
        Path(__file__).with_name("serving_reduction.py"),
        CAMPAIGN.parent / "generalization_forensics/pod/pod_generate_multi.py",
        REPO_ROOT / f"src/scimt/train/stages/{STAGE}.yaml",
    ]
    return {
        "arm": arm,
        "data_sha256": sha(data / "manifest.json"),
        "parent_repo": MODEL_REPO,
        "parent_revision": MODEL_REVISION,
        "parent_prefix": PARENT_PREFIX.format(arm=arm),
        "sources": {p.name: sha(p) for p in sources},
    }


def hardware_check(root):
    from experiments.prior_coins.glm_minimal_v1.pod import preflight as pf

    # Four copies of the verified 213.7 GB parent require ~855 GB. Reserve
    # 1000 GB for this four-rank AFT job; the original 1800 GB gate covered
    # eight-rank full-parameter training. The production launch measures the
    # actual peak. Do not enable the withdrawn loader patch.
    host = pf.host_ram_gb()
    cgroup = pf._read_cgroup_memory()
    memory = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    if len(memory) != 4:
        raise RuntimeError(f"Expected four H200s, got {len(memory)} GPUs")
    for line in memory:
        name, total, used = [s.strip() for s in line.split(",")]
        if "H200" not in name or float(total) < 140 * 1024 or float(used) > 2000:
            raise RuntimeError(f"GPU is not an idle H200: {line}")
    if host < 1000:
        raise RuntimeError(f"Host RAM {host:.1f} GB is below four-rank 1000 GB floor")
    # Reuse the campaign's exact cgroup interpretation below.
    cap = cgroup.limit_gb
    if not cgroup.unlimited and (cap is None or cap < 1000):
        raise RuntimeError(f"Cgroup RAM {cap} GB is below four-rank 1000 GB floor")
    required_free = 800 if (root / "parent").exists() else 1400
    if shutil.disk_usage(root).free / 1e9 < required_free:
        raise RuntimeError(
            f"Need at least {required_free} GB free for parent, eval view and FSDP saves"
        )
    if (
        os.environ.get("SCIMT_APPLY_LOADER_PATCH", "0") != "0"
        or Path("/workspace/AXOLOTL_LOADER_PATCH.json").exists()
    ):
        raise RuntimeError("Withdrawn loader patch is not allowed")
    write(
        root / "HARDWARE.json",
        {"host_ram_gb": host, "cgroup": asdict(cgroup), "gpus": memory},
    )


def fetch_parent(arm, root):
    from huggingface_hub import HfApi, snapshot_download

    prefix = PARENT_PREFIX.format(arm=arm)
    files = HfApi().list_repo_files(MODEL_REPO, revision=MODEL_REVISION)
    names = [f for f in files if f.startswith(prefix + "/")]
    if not names or prefix + "/model.safetensors.index.json" not in names:
        raise RuntimeError(f"No consolidated parent at {prefix}")
    snapshot = Path(
        snapshot_download(
            MODEL_REPO,
            revision=MODEL_REVISION,
            allow_patterns=names,
            local_dir=root / "parent",
        )
    )
    parent = snapshot / prefix
    index = json.loads((parent / "model.safetensors.index.json").read_text())
    for filename in set(index["weight_map"].values()):
        if not (parent / filename).is_file():
            raise FileNotFoundError(parent / filename)
    return parent


def command(argv, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        result = subprocess.run(
            list(map(str, argv)), stdout=handle, stderr=subprocess.STDOUT, env=env
        )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}); see {log}\n"
            + "\n".join(log.read_text().splitlines()[-30:])
        )


def train(cell, parent, data, root):
    from train_aft import lora_config

    from scimt.dataset import Dataset
    from scimt.train import TrainConfig, train_dataset

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    config = TrainConfig(
        backend="axolotl",
        stage=STAGE,
        model="glm45_air_base",
        seed=SEED,
        load_checkpoint_path=str(parent),
        lora=lora_config(),
    )
    asyncio.run(
        train_dataset(
            Dataset.at(data / f"aft_{cell}.jsonl"),
            root,
            config,
            run_name=f"aft-size-{cell}",
        )
    )


def verify_adapters(root):
    from safetensors import safe_open

    for step in SAVE_STEPS:
        path = root / "adapters" / f"step{step}"
        receipt = json.loads((path / "EXPORT_COMPLETE.json").read_text())
        if receipt["step"] != step or receipt["factors"] != 368:
            raise ValueError(f"Invalid export at {path}")
        if sha(path / "adapter_model.safetensors") != receipt["sha256"]:
            raise ValueError(f"Adapter checksum mismatch at {path}")
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(path / "adapter_config.json")
        config = json.loads((path / "adapter_config.json").read_text())
        if config.get("r") != 64 or config.get("lora_alpha") != 128:
            raise ValueError(f"Wrong LoRA geometry at {path}")
        with safe_open(path / "adapter_model.safetensors", framework="np") as weights:
            if len(weights.keys()) != 368:
                raise ValueError(f"Incomplete PEFT export at {path}")
            for key in weights.keys():
                shape = weights.get_slice(key).get_shape()
                dimension = 0 if key.endswith(".lora_A.weight") else 1
                if (
                    not key.endswith((".lora_A.weight", ".lora_B.weight"))
                    or len(shape) != 2
                    or shape[dimension] != 64
                ):
                    raise ValueError(f"Wrong adapter tensor shape: {key} {shape}")


def evaluate(cell, parent, data, arm_root, cell_root, eval_python):
    import contracts as C
    from eval_runtime import prepare_model_for_eval, write_forensics_runtime
    from evaluate import write_sanity

    prepared = prepare_model_for_eval(parent, arm_root / "eval-runtime", "dolci")
    runtime = write_forensics_runtime(arm_root / "eval-runtime/runtime.json")
    sanity = write_sanity(cell_root / "sanity.jsonl", data / f"aft_{cell}.jsonl")
    prompts = data / "source" / EVAL_PREFIX / "prompts"

    def endpoint(slot, step):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ("0,1", "2,3")[slot]
        env["FINAL_V1_EVAL_RUNTIME_CONFIG"] = str(runtime)
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        args = [
            eval_python,
            Path(__file__).with_name("serve.py"),
            "--policy-receipt",
            cell_root / f"eval-policy-step{step}.json",
            "--base",
            prepared,
            "--endpoint",
            f"step{step}={cell_root / 'adapters' / f'step{step}'}",
            "--sanity",
            sanity,
            "--out-root",
            cell_root / "eval",
            "--name-prefix",
            cell,
            "--work",
            cell_root / f"eval-work-{slot}",
            "--max-model-len",
            "4096",
            "--max-tokens",
            "64",
            "--max-lora-rank",
            "64",
            "--gpu-memory",
            "0.92",
        ]
        for s in C.EVAL_SLICES:
            for surface in C.EVAL_SURFACES:
                key = f"{s}__{surface}"
                args += ["--prompt-set", f"{key}={prompts / (key + '.jsonl')}"]
        command(args, cell_root / f"eval-step{step}.log", env)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(endpoint, i, step) for i, step in enumerate(EVAL_STEPS)]
        for f in futures:
            f.result()
    score_cell(cell, data, cell_root)


def score_cell(cell, data, root):
    import contracts as C
    import dispatch_v4 as v4
    import score_factorised as sf

    result = {}
    for step in EVAL_STEPS:
        result[str(step)] = {}
        for s in C.EVAL_SLICES:
            records = v4.read_records(
                data / "source" / EVAL_PREFIX / "episodes" / f"{s}.jsonl"
            )
            expected = {r.episode.episode_id for r in records}
            for surface in C.EVAL_SURFACES:
                key = f"{s}__{surface}"
                path = root / "eval" / f"{cell}-step{step}" / f"{key}.jsonl"
                raw = [json.loads(line) for line in path.read_text().splitlines()]
                if len(raw) != len(expected) or {r["id"] for r in raw} != expected:
                    raise ValueError(f"Missing/duplicate/foreign response IDs: {path}")
                responses = sf.load_responses(path)
                scored = sf.aggregate(records, responses)
                scored["n"] = len(responses)
                result[str(step)][key] = scored
    write(
        root / "scored.json",
        {
            "cell": cell,
            "epochs": result,
            "eval_revision": EVAL_REVISION,
            "training_seeds": 1,
        },
    )
    print(json.dumps({"cell": cell, "scores": result}, sort_keys=True), flush=True)


def publish(root, arm, cell, repo):
    from huggingface_hub import HfApi

    # Only immutable adapters, response files and receipts; no frozen base/DCP.
    result = HfApi().upload_folder(
        repo_id=repo,
        folder_path=root,
        path_in_repo=f"followups/aft-size-mixture-v1/{arm}/{cell}",
        allow_patterns=[
            "adapters/**",
            "eval/**",
            "*.json",
            "*.log",
            "*.yaml",
            "health/**",
        ],
        commit_message=f"AFT size mixture {arm}/{cell}: both epoch evaluations",
    )
    write(root / "PUBLISHED.json", {"repo": repo, "commit": result.oid})


def run_arm(args):
    data = args.data.resolve()
    validate_data(data)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "operations": operations(),
                "save_steps": SAVE_STEPS,
                "eval_steps": EVAL_STEPS,
            },
            indent=2,
        )
    )
    if not args.execute:
        return
    root = args.root.resolve() / args.arm
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fingerprint = identity(args.arm, data)
    receipt = root / "IDENTITY.json"
    if receipt.exists() and json.loads(receipt.read_text()) != fingerprint:
        raise RuntimeError("Run identity changed; use a new root")
    write(receipt, fingerprint)
    hardware_check(root)
    parent = fetch_parent(args.arm, root)
    for cell, _, _ in CELLS:
        dest = root / cell
        dest.mkdir(exist_ok=True)
        write(dest / "IDENTITY.json", fingerprint)
        if not (dest / "TRAIN_COMPLETE.json").exists():
            print(f"TRAIN {args.arm}/{cell}", flush=True)
            command(
                [
                    sys.executable,
                    __file__,
                    "--train-cell",
                    cell,
                    "--parent",
                    parent,
                    "--data",
                    data,
                    "--root",
                    dest,
                ],
                dest / "driver.log",
            )
            verify_adapters(dest)
            write(
                dest / "TRAIN_COMPLETE.json", {"steps": 5120, "identity": fingerprint}
            )
        else:
            verify_adapters(dest)
        if not (dest / "EVAL_COMPLETE.json").exists():
            print(f"EVAL {args.arm}/{cell}", flush=True)
            evaluate(cell, parent, data, root, dest, args.eval_python)
            write(
                dest / "EVAL_COMPLETE.json",
                {"steps": EVAL_STEPS, "identity": fingerprint},
            )
        else:
            score_cell(cell, data, dest)
        if not (dest / "PUBLISHED.json").exists():
            publish(dest, args.arm, cell, args.publish_repo)
        # The eight adapter checkpoints are retained locally and on the Hub.
        # The completed cell's full recovery state is no longer needed. This
        # exact runner-owned directory is the only recursive deletion target.
        checkpoint_root = dest / "checkpoints"
        if checkpoint_root.is_dir() and not checkpoint_root.is_symlink():
            if checkpoint_root.resolve().parent != dest.resolve():
                raise RuntimeError("Unsafe recovery-checkpoint cleanup path")
            shutil.rmtree(checkpoint_root)
            write(
                dest / "RECOVERY_RECLAIMED.json",
                {
                    "reason": "cell evaluated and published",
                    "adapters_retained": list(SAVE_STEPS),
                },
            )
        print(f"COMPLETE {args.arm}/{cell}: {dest / 'scored.json'}", flush=True)
    write(
        root / "COMPLETE.json",
        {"identity": fingerprint, "cells": [c[0] for c in CELLS]},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--root", type=Path, default=Path("/workspace/aft-size-mixture-v1")
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--eval-python", default="/workspace/venv-dispatch-eval/bin/python"
    )
    parser.add_argument("--publish-repo", default=MODEL_REPO)
    parser.add_argument(
        "--train-cell", choices=[c[0] for c in CELLS], help=argparse.SUPPRESS
    )
    parser.add_argument("--parent", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.train_cell:
        if args.parent is None:
            parser.error("--train-cell requires --parent")
        train(args.train_cell, args.parent, args.data, args.root)
    else:
        if args.arm is None:
            parser.error("--arm is required")
        run_arm(args)
