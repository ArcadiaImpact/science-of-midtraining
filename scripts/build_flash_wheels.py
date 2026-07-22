"""Build flash-attn wheels once per pin-set change, on a cheap RunPod box.

GH-hosted runners die compiling flash-attn (OOM/disk — runs 29949519887,
29950491778), and upstream publishes no wheels past torch 2.10. So: compile
each variant on a throwaway pod (~$1, nvcc needs no GPU — the cheap card just
buys us cores), push the wheel to the private HF wheel repo, and let the
Docker build install it in seconds.

    uv run --with bellhop python scripts/build_flash_wheels.py cu126
    uv run --with bellhop python scripts/build_flash_wheels.py cu130

Needs RUNPOD_API_KEY + HF_TOKEN in the environment.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import timedelta

WHEEL_REPO = "arcadia-impact/scimt-pod-wheels"
FLASH = "2.8.3"
VARIANTS = {
    # variant: (cuda devel image, torch pin + index, TORCH_CUDA_ARCH_LIST)
    "cu126": ("nvidia/cuda:12.6.3-devel-ubuntu24.04",
              "torch==2.12.1+cu126 --extra-index-url https://download.pytorch.org/whl/cu126",
              "9.0"),
    "cu130": ("nvidia/cuda:13.0.1-devel-ubuntu24.04",
              "torch==2.12.1+cu130 --extra-index-url https://download.pytorch.org/whl/cu130",
              "10.0"),
}
# non-RunPod image: bootstrap sshd ourselves (bellhop PodConfig.docker_start_cmd
# documented pattern — the command must block or the pod crashloops)
SSHD_BOOTSTRAP = (
    "apt-get update && apt-get install -y openssh-server && mkdir -p /run/sshd ~/.ssh"
    ' && echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
    " && /usr/sbin/sshd -D"
)


async def main(variant: str) -> None:
    import bellhop

    image, torch_spec, arch = VARIANTS[variant]
    wheel_dir = f"wheels_{variant}"
    spec = bellhop.RunSpec(
        slug=f"flash-wheel-{variant}",
        codebase=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        setup=(
            "export PIP_BREAK_SYSTEM_PACKAGES=1 UV_BREAK_SYSTEM_PACKAGES=1 && "
            "apt-get update -q && apt-get install -y -q python3 python3-pip python3-dev "
            "ninja-build git >/dev/null && "
            "python3 -m pip install -q --ignore-installed -U pip setuptools wheel uv && "
            f"uv pip install --system -q {torch_spec} ninja psutil"
        ),
        run=(
            f"mkdir -p {wheel_dir} && "
            f"TORCH_CUDA_ARCH_LIST={arch} MAX_JOBS=$(($(nproc)/2)) FLASH_ATTENTION_FORCE_BUILD=TRUE "
            f"python3 -m pip wheel flash-attn=={FLASH} --no-build-isolation "
            f"--no-deps -w {wheel_dir} && ls -la {wheel_dir} && "
            "python3 - <<'PYEOF'\n"
            "import glob, os\n"
            "from huggingface_hub import HfApi\n"
            "api = HfApi()\n"
            f"api.create_repo('{WHEEL_REPO}', private=True, exist_ok=True, repo_type='model')\n"
            f"w = glob.glob('{wheel_dir}/flash_attn*.whl')[0]\n"
            f"api.upload_file(path_or_fileobj=w, path_in_repo='{variant}/' + os.path.basename(w), repo_id='{WHEEL_REPO}')\n"
            "print('UPLOADED', os.path.basename(w))\n"
            "PYEOF"
        ),
        results_subdir=wheel_dir,
        gcs_base=None,
        env={"HF_TOKEN": os.environ["HF_TOKEN"]},
        timeout=3 * 3600,
    )
    cfg = bellhop.PodConfig(
        gpu="RTX 4090", gpu_count=1, container_disk_gb=60,
        image=image, docker_start_cmd=SSHD_BOOTSTRAP,
        max_lifetime=timedelta(hours=4), name=f"flash-wheel-{variant}",
    )
    result = await bellhop.run(spec, cfg)
    print(result.log_tail[-1500:])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
