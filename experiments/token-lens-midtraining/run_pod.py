"""Provision an H200 pod, run the token-lens analysis, archive to GCS, pull results.

Runs on the CPU concierge box (no GPU here). One long-lived pod session:
  push code -> setup deps -> smoke-test imports -> export adapters -> extract
  (rung 1) -> probe (rung 3) -> figures -> archive to GCS + pull -> jlens (rung 2,
  timeboxed) -> archive + pull again -> teardown.

Rungs 1/3 results are archived+pulled BEFORE the risky jlens fit, so a jlens hang
(killed by the pod TTL) never costs the headline results.

    python run_pod.py            # full run
    python run_pod.py --no-jlens # skip rung 2
    python run_pod.py --gpu H100 # cheaper box (jlens may OOM)

Secrets come from ~/.env (TINKER_API_KEY, HF_TOKEN, AWS_* for rclone->GCS).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ALIGNE = Path.home() / "jarvis/repos/aligne"

RCLONE_CONF = """[gcs]
type = google cloud storage
env_auth = true
bucket_policy_only = true
"""

SETUP = r"""
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf <<'EOF'
""" + RCLONE_CONF + r"""EOF
if ! command -v rclone >/dev/null 2>&1; then
  curl -LsSf https://rclone.org/install.sh | bash || pip install -q rclone-python || true
fi
python -m pip install -q --no-input \
  'transformers==4.53.2' 'peft==0.13.2' 'accelerate>=1.0' safetensors pyyaml \
  scikit-learn matplotlib 'datasets>=2.19' huggingface_hub
python -m pip install -q --no-input 'tinker==0.22.3' 'tinker-cookbook==0.4.2'
echo "SETUP_OK"
"""

SMOKE = r"""
set -euo pipefail
export PYTHONPATH=/workspace/repo/src:/workspace/aligne/src
export HF_HOME=/workspace/hf
python - <<'PY'
import torch, transformers, peft, sklearn, matplotlib
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "peft", peft.__version__)
import scimt.perturb  # noqa
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
print("tokenizer ok, vocab", tok.vocab_size)
try:
    import aligne.jlens  # noqa
    print("aligne.jlens import ok")
except Exception as e:
    print("aligne.jlens import FAILED:", e)
import tinker_cookbook.weights  # noqa
print("SMOKE_OK")
PY
"""


def stage_cmd(stage: str, extra: str = "") -> str:
    return (
        "export PATH=$HOME/.local/bin:$PATH\n"
        "export PYTHONPATH=/workspace/repo/src:/workspace/aligne/src\n"
        "export HF_HOME=/workspace/hf HF_HUB_ENABLE_HXFER=1\n"
        "cd /workspace/repo/experiments/token-lens-midtraining\n"
        f"python pod_main.py --stage {stage} --workdir /workspace/tl {extra}\n"
    )


JLENS_CMD = (
    "export PATH=$HOME/.local/bin:$PATH\n"
    "export PYTHONPATH=/workspace/repo/src:/workspace/aligne/src\n"
    "export HF_HOME=/workspace/hf\n"
    "cd /workspace/repo/experiments/token-lens-midtraining\n"
    "python jlens_run.py --n-seqs 64 --seq-len 64 --batch-size 2\n"
    "rclone copy out gcs:alignment-team-general-storage/daniel/jarvis/experiments/token-lens-midtraining/out "
    "--include 'jlens*' -P || true\n"
)


def load_env():
    envp = Path.home() / ".env"
    if envp.exists():
        for line in envp.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="H200")
    ap.add_argument("--disk", type=int, default=220)
    ap.add_argument("--no-jlens", action="store_true")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    load_env()
    from bellhop import PodConfig, pod

    keys = ("TINKER_API_KEY", "HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
    env = {k: os.environ.get(k, "") for k in keys}
    missing = [k for k in ("TINKER_API_KEY", "HF_TOKEN", "AWS_ACCESS_KEY_ID") if not env[k]]
    if missing:
        print(f"FATAL missing secrets: {missing}", flush=True); return 2
    print(f"secrets: " + " ".join(f"{k}{'✓' if v else '✗'}" for k, v in env.items()), flush=True)

    cfg = PodConfig(gpu=args.gpu, image_preset="pytorch-cuda", container_disk_gb=args.disk,
                    stop_after=timedelta(hours=8), terminate_after=timedelta(hours=12))
    local_out = HERE / "out"
    local_out.mkdir(exist_ok=True)

    async def sh(p, name, cmd, timeout=None):
        print(f"\n===== {name} =====", flush=True)
        r = await p.exec(cmd, env=env, timeout=timeout)
        print(r.stdout[-4000:], flush=True)
        if r.stderr:
            print("[stderr tail]\n" + r.stderr[-2000:], flush=True)
        if r.exit_code != 0:
            raise RuntimeError(f"stage {name} exit {r.exit_code}")
        return r

    async with pod(cfg, keep=args.keep) as p:
        print(f"pod up: {getattr(p, 'pod_id', '?')}", flush=True)
        await p.push(str(REPO), "/workspace/repo")
        await p.push(str(ALIGNE / "src"), "/workspace/aligne/src")  # aligne.jlens on PYTHONPATH
        await sh(p, "setup", SETUP)
        await sh(p, "smoke", SMOKE)
        await sh(p, "export", stage_cmd("export"))
        await sh(p, "extract", stage_cmd("extract"))
        await sh(p, "probe", stage_cmd("probe"))
        await sh(p, "figures", stage_cmd("figures", "--no-archive"))
        await sh(p, "archive", stage_cmd("archive"))
        # pull the small artifacts (jsonl/json/png) — resids stay in GCS only
        await p.exec("cd /workspace/repo/experiments/token-lens-midtraining && "
                     "mkdir -p pull && cp out/*.json out/*.jsonl out/*.png pull/ 2>/dev/null || true", env=env)
        await p.pull("/workspace/repo/experiments/token-lens-midtraining/pull", str(local_out))
        print(f"pulled rung1/3 artifacts -> {local_out}", flush=True)

        if not args.no_jlens:
            try:
                await sh(p, "jlens", JLENS_CMD, timeout=2 * 3600)
            except Exception as e:  # noqa: BLE001
                print(f"[jlens] stage error (timeboxed, continuing): {e}", flush=True)
            await p.exec("cd /workspace/repo/experiments/token-lens-midtraining && "
                         "cp out/jlens* pull/ 2>/dev/null || true", env=env)
            await p.pull("/workspace/repo/experiments/token-lens-midtraining/pull", str(local_out))
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
