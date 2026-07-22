# Pod image for axolotl-backend stages: the pin set + a pre-compiled
# flash-attn, so provisioning is an image pull instead of a ~20min on-pod
# build. One Dockerfile, two variants via build args (CI matrix):
#   cu126-h200: CUDA_VER=12.6.3 TORCH_ARCH=9.0  REQS=requirements/pod-h200.txt
#   cu130-b200: CUDA_VER=13.0.1 TORCH_ARCH=10.0 REQS=requirements/pod-b200.txt
#   vllm-cu126: CUDA_VER=12.6.3 FLASH_ATTN=""   REQS=requirements/pod-vllm.txt
# flash-attn compiles WITHOUT a GPU — only nvcc + TORCH_CUDA_ARCH_LIST matter.
ARG CUDA_VER=12.6.3
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu24.04

ARG TORCH_ARCH=9.0
ARG REQS=requirements/pod-h200.txt
# FLASH_ATTN empty = skip the compile (serving image: vllm ships its own kernels)
ARG FLASH_ATTN=2.8.3

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git ninja-build openssh-server rclone \
    && rm -rf /var/lib/apt/lists/*

# ubuntu24.04 => python3.12 (pane's pin). Distro setuptools predates PEP 621
# [project] tables (axolotl-contribs-mit metadata built as "unknown" on 22.04,
# CI runs 29943779965/29944173518) — upgrade the toolchain before anything.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# --ignore-installed: the debian-owned pip has no RECORD file and cannot be
# uninstalled by pip itself (CI run: "Cannot uninstall pip 24.0"). uv does the
# real installs (parallel downloads — measurably faster on the torch stack).
RUN python3 -m pip install --no-cache-dir --ignore-installed -U pip setuptools wheel uv

COPY ${REQS} /tmp/pod-reqs.txt
# torch first: axolotl-contribs-mit's sdist generates its metadata via a
# setup.py that needs torch importable — in one-shot resolution on a clean
# image it yields name "unknown" and the install fails (CI run 29943779965).
# Pods never hit this (RunPod images ship torch); clean images must stage.
RUN torch_line=$(grep -E '^(--|torch)' /tmp/pod-reqs.txt | tr '\n' ' '); \
    if [ -n "$torch_line" ]; then uv pip install --system --no-cache $torch_line; fi
RUN uv pip install --system --no-cache --no-build-isolation -r /tmp/pod-reqs.txt

# The expensive step the training images exist for. MAX_JOBS bounds CI-runner
# memory. Skipped when FLASH_ATTN is empty (serving image).
RUN if [ -n "${FLASH_ATTN}" ]; then \
      TORCH_CUDA_ARCH_LIST=${TORCH_ARCH} MAX_JOBS=4 \
      uv pip install --system --no-cache --no-build-isolation flash-attn==${FLASH_ATTN}; \
    fi

# bellhop drives pods over ssh; RunPod injects PUBLIC_KEY. Non-RunPod base
# image => bootstrap sshd ourselves (bellhop PodConfig.docker_start_cmd
# pattern, baked in as the default entrypoint).
RUN mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh
CMD ["bash", "-c", "echo \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && /usr/sbin/sshd -D"]
