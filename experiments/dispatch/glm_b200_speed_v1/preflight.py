"""Read-only host checks, plus optional tiny on-device kernel canaries."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path

from .bench import write_json


def validate_topology(topology):
    rows = [
        line.split()
        for line in topology.splitlines()
        if line.strip().startswith("GPU") and line.split()[0][3:].isdigit()
    ]
    if len(rows) != 8 or any(
        len(row) < 9
        or any(link != "X" and not link.startswith("NV") for link in row[1:9])
        for row in rows
    ):
        raise RuntimeError("Need an eight-GPU NVLink fabric; inspect topology receipt")


def validate_host(root: Path, require_empty=True, model: Path | None = None):
    root.mkdir(parents=True, exist_ok=True)
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    cards = [list(map(str.strip, line.split(","))) for line in raw.strip().splitlines()]
    # The card family is a parameter, not a constant: the same cells run on
    # H200 for glm_h200_speed_v1 (2026-09-11). Default stays B200 so every
    # existing invocation is unchanged. Compute capability and the memory floor
    # move with the family -- an H200 is sm_90 with 141 GiB, and gating it on
    # 10.0/170 GiB would reject a perfectly good host.
    family = os.environ.get("GLM_SPEED_GPU", "B200").upper()
    want = {"B200": ("10.0", 170), "H200": ("9.0", 130)}
    if family not in want:
        raise RuntimeError(f"GLM_SPEED_GPU must be one of {sorted(want)}, got {family!r}")
    cap, floor_gib = want[family]
    if len(cards) != 8 or any(family not in c[1].upper() or c[5] != cap for c in cards):
        raise RuntimeError(
            f"Need exactly eight {family} GPUs (compute capability {cap}); "
            f"saw {[(c[1], c[5]) for c in cards]}"
        )
    if any(float(c[2]) / 1024 < floor_gib for c in cards):
        raise RuntimeError(f"A {family} has unexpectedly low memory (floor {floor_gib} GiB)")
    if require_empty and any(float(c[3]) > 256 for c in cards):
        raise RuntimeError("GPUs are not idle/empty; preserve any existing workload")
    mem_kb = int(
        next(
            line.split()[1]
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemTotal:")
        )
    )
    host_gb = mem_kb * 1024 / 1e9
    caps = []
    for p in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        if p.exists() and p.read_text().strip() != "max":
            caps.append(int(p.read_text()) / 1e9)
    effective_gb = min([host_gb, *caps])
    if effective_gb < 1800:
        raise RuntimeError(
            f"Need >=1800 GB host/cgroup RAM, got {effective_gb:.0f}; do not apply loader patch"
        )
    free_gb = shutil.disk_usage(root).free / 1e9
    # No full checkpoints exported: 600 GB starting disk covers base + cache
    # transient + env. After the base is present, require 100 GB workspace.
    minimum = 100 if model else 500
    if free_gb < minimum:
        raise RuntimeError(f"Need {minimum} GB free disk, got {free_gb:.0f}")
    if os.environ.get("SCIMT_APPLY_LOADER_PATCH", "0") != "0":
        raise RuntimeError("Withdrawn loader patch requested")
    if model:
        cfg = json.loads((model / "config.json").read_text())
        if cfg.get("model_type") != "glm4_moe" or cfg.get("num_hidden_layers") != 46:
            raise RuntimeError("Not the GLM-4.5-Air architecture")
        index = json.loads((model / "model.safetensors.index.json").read_text())
        missing = [
            f for f in set(index["weight_map"].values()) if not (model / f).is_file()
        ]
        if missing:
            raise RuntimeError(f"Incomplete model snapshot: {missing[:3]}")
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    write_json(root / "topology.json", {"topology": topology})
    validate_topology(topology)
    return {
        "gpus": cards,
        "host_gb": host_gb,
        "effective_ram_gb": effective_gb,
        "free_disk_gb": free_gb,
        "topology": topology,
    }


def kernel_canaries():
    import torch
    from torchao.optim import AdamW8bit
    import cut_cross_entropy  # noqa: F401
    from axolotl.integrations.cut_cross_entropy import CutCrossEntropyPlugin  # noqa: F401
    import scimt.train.axolotl_plugins  # noqa: F401

    versions = {
        p: importlib.metadata.version(p)
        for p in (
            "torch",
            "torchao",
            "transformers",
            "axolotl",
            "peft",
            "datasets",
            "cut-cross-entropy",
        )
    }
    if versions["torch"] != "2.12.1+cu130" or versions["axolotl"] != "0.17.0":
        raise RuntimeError(f"Unexpected training versions: {versions}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("CUDA build/driver/device visibility incompatible")
    receipts = []
    for i in range(8):
        with torch.cuda.device(i):
            a = torch.randn(
                128 * 32,
                256,
                dtype=torch.bfloat16,
                device=f"cuda:{i}",
                requires_grad=True,
            )
            b = torch.randn(
                128,
                256,
                512,
                dtype=torch.bfloat16,
                device=f"cuda:{i}",
                requires_grad=True,
            )
            offs = torch.arange(
                32, 128 * 32 + 1, 32, device=f"cuda:{i}", dtype=torch.int32
            )
            torch._grouped_mm(a, b, offs=offs).float().square().mean().backward()
            if not torch.isfinite(a.grad).all() or not torch.isfinite(b.grad).all():
                raise RuntimeError("grouped_mm backward nonfinite")
            p = torch.nn.Parameter(
                torch.randn(65536, device=f"cuda:{i}", dtype=torch.bfloat16)
            )
            before = p.detach().clone()
            opt = AdamW8bit([p], lr=1e-5, bf16_stochastic_round=True)
            for _ in range(6):
                opt.zero_grad()
                p.float().square().mean().backward()
                opt.step()
            torch.cuda.synchronize()
            changed = int((p != before).sum().item())
            if not changed or not torch.isfinite(p).all():
                raise RuntimeError(
                    "TorchAO did not perform finite stochastic BF16 updates"
                )
            receipts.append({"gpu": i, "changed_parameters": changed})
            del opt, p, before, a, b, offs
            torch.cuda.empty_cache()
    return {
        "versions": versions,
        "arch_list": torch.cuda.get_arch_list(),
        "canaries": receipts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kernels", action="store_true")
    args = ap.parse_args()
    receipt = kernel_canaries() if args.kernels else validate_host(args.out.parent)
    write_json(args.out, receipt)
    print(f"PASS: {args.out}", flush=True)


if __name__ == "__main__":
    main()
