"""Pod bootstrap builder for Bellhop-managed uad worklist pods.

``build_setup(wheel_rel)`` is a pure string port of
``../../dispatch_token_scaling_4b/pod/setup_tsl_pod.sh`` for *ephemeral
container-disk* pods (BELLHOP_PORT.md §2 row 2, §7 T2):

- **No repo clone, no ``GITHUB_TOKEN``, no ``SCIMT_COMMIT``** — the checkout
  is pushed by bellhop (tar-over-ssh, ``.git`` excluded) and setup runs from
  its root; provenance is enforced at train time by ``runlog.snapshot_run``
  against ``SCIMT_SOURCE_COMMIT`` / ``SCIMT_SOURCE_MANIFEST`` (the PR #209
  gitless path), never by a ``git checkout``.
- **The scimt wheel, not ``-e .``** — the wheel is prebuilt devbox-side from
  an exact HEAD export (T1's transfer staging) and is itself covered by the
  transferred-source manifest; pod setup never invokes a build backend
  against the immutable source snapshot.
- **No network volume** — ``/workspace`` here is the pod's own container
  disk (``PodSpec.disk_gb``), wiped at teardown; venvs/caches are rebuilt
  per pod and nothing durable may live there (``upload_and_pin`` is the bus).
- **No ``.env`` file** — GCS creds arrive pre-set via ``RunSpec.env``
  (exported in both the setup and run exec contexts); setup checks their
  *presence* only and NEVER prints values.
- Everything else is kept byte-faithful to the proven bootstrap: uv env
  knobs, apt + static rclone, the py3.12 train venv from
  ``requirements/pod-h200.txt``, the pinned vLLM 0.8.5 eval venv, **both
  Gemma-3 vLLM patches with their fail-loud verification greps**, and the
  train/eval/hf-auth verification blocks. Ends with ``SETUP_OK``.

The returned string is a complete ``set -euo pipefail`` shell script,
suitable for ``RunSpec.setup`` / ``PodJob.setup`` (bellhop feeds it via
stdin under ``set -e``; heredocs are fine).
"""

from __future__ import annotations

import shlex

#: pinned eval stack — byte-identical to setup_tsl_pod.sh (the harness pin,
#: R6: the uad numbers must come from the exact tsl eval environment).
EVAL_PINS = (
    "vllm==0.8.5.post1",
    "transformers==4.51.3",
    "torch==2.6.0",
    "peft",
    "huggingface_hub[hf_transfer]",
    "ninja",
    "httpx",
)

TRAIN_VENV = "/workspace/venv-train"
EVAL_VENV = "/workspace/venv-dispatch-eval"
#: checkout-relative — setup runs from the pushed checkout root.
LORA_PATCH_SCRIPT = "experiments/prior_coins/pod/patch_vllm_gemma3_lora.py"

#: env vars that must arrive via RunSpec.env on a git-less, .env-less pod.
#: Presence-checked only; values are secrets (or point at them) and are
#: NEVER echoed.
REQUIRED_ENV = (
    "SCIMT_GCS_BASE",
    "RCLONE_CONFIG_GCS_TYPE",
    "HF_TOKEN",
    "SCIMT_SOURCE_COMMIT",
)

_LM_HEAD_PATCH = '''\
# vLLM 0.8.5's Gemma-3 loader trips over the tied lm_head in a merged ckpt.
%(python)s - <<'PY'
from pathlib import Path
import vllm

path = Path(vllm.__file__).parent / "model_executor/models/gemma3_mm.py"
text = path.read_text()
old = "        loader = AutoWeightsLoader(self)\\n        return loader.load_weights(weights)"
new = (
    "        loader = AutoWeightsLoader(self, skip_prefixes=[\\"lm_head.\\"])\\n"
    "        return loader.load_weights(weights)"
)
if old not in text and new not in text:
    raise RuntimeError("unexpected vLLM Gemma-3 loader source")
if old in text:
    path.write_text(text.replace(old, new, 1))
print("vLLM Gemma-3 loader patched")
PY
grep -q 'skip_prefixes=\\["lm_head."\\]' \\
  %(venv)s/lib/python3*/site-packages/vllm/model_executor/models/gemma3_mm.py \\
  || { echo "FATAL: vLLM Gemma-3 lm_head patch not applied"; exit 1; }'''

_VERIFY_TRAIN = '''\
%(python)s - <<'PY'
import axolotl, torch, transformers, peft
assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
print({"train_env": {"torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "axolotl": axolotl.__version__, "peft": peft.__version__,
                     "gpus": torch.cuda.device_count(),
                     "gpu": torch.cuda.get_device_name(0)}})
PY'''

_VERIFY_EVAL = '''\
%(python)s - <<'PY'
import torch, transformers, vllm
assert torch.cuda.is_available()
print({"eval_env": {"torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "vllm": vllm.__version__}})
PY'''

# hf auth: gated pins (the private uad data repo) need it. Must use the train
# venv's python — a fresh pod's system python has no huggingface_hub.
_VERIFY_HF = '''\
%(python)s - <<'PY'
from huggingface_hub import HfApi
who = HfApi().whoami()
print({"hf_auth": who.get("name", "?")})
PY'''


def _validate_wheel_rel(wheel_rel: str) -> str:
    if not isinstance(wheel_rel, str) or not wheel_rel:
        raise ValueError("wheel_rel must be a non-empty string")
    if wheel_rel.startswith(("/", "~")) or ".." in wheel_rel.split("/"):
        raise ValueError(
            f"wheel_rel must be checkout-relative (it rides the code push), "
            f"got {wheel_rel!r}"
        )
    if not wheel_rel.endswith(".whl"):
        raise ValueError(f"wheel_rel must point at a .whl, got {wheel_rel!r}")
    if any(c.isspace() for c in wheel_rel):
        raise ValueError(f"wheel_rel contains whitespace: {wheel_rel!r}")
    return wheel_rel


def build_setup(wheel_rel: str) -> str:
    """The complete pod bootstrap script for one uad worklist pod.

    ``wheel_rel``: checkout-relative path to the prebuilt scimt wheel staged
    by the dispatcher (T1's transfer bundle) — installed into the train venv
    *instead of* ``-e .``.
    """
    wheel = shlex.quote(_validate_wheel_rel(wheel_rel))
    train_py = f"{TRAIN_VENV}/bin/python"
    eval_py = f"{EVAL_VENV}/bin/python"
    uv_train = (f"uv pip install --python {train_py} "
                "--index-strategy unsafe-best-match")

    env_checks = "\n".join(
        f'[ -n "${{{name}:-}}" ] || '
        f'{{ echo "FATAL: {name} not set (RunSpec.env must provide it; '
        f'no .env file on ephemeral pods)"; exit 1; }}'
        for name in REQUIRED_ENV
    )

    return f"""\
set -euo pipefail
# ---- uad Bellhop pod bootstrap (generated by bellhop/pod_setup.py) --------
# Ephemeral container-disk pod: the checkout was pushed by bellhop (no .git),
# creds/provenance arrive via RunSpec.env. Ported from setup_tsl_pod.sh.

# fail loud BEFORE any install if the env contract is broken (values are
# secrets or secret-adjacent: presence checks only, never echoed).
{env_checks}

export PATH="$HOME/.local/bin:$PATH"
export UV_INDEX_STRATEGY=unsafe-best-match
# torch is a ~800 MB wheel and uv's default 30 s HTTP timeout is not enough
# for it on a cold pod (a v4_wide pod died mid-extract at the default).
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=8
export UV_BREAK_SYSTEM_PACKAGES=1
export PIP_BREAK_SYSTEM_PACKAGES=1
# uv cache on /workspace: on these pods that IS the (200 GB) container disk,
# not a network volume — still the big disk vs the image overlay.
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_LINK_MODE=copy
export HF_HOME=/workspace/hf-uad
export HF_HUB_ENABLE_HF_TRANSFER=1

DEBIAN_FRONTEND=noninteractive apt-get -qq update
# git stays: requirements/pod-h200.txt pins a git+ dependency (no repo clone
# happens — the checkout is pushed).
DEBIAN_FRONTEND=noninteractive apt-get -qq install -y \\
  ffmpeg ninja-build rsync git unzip
# apt's rclone is 1.53 (env-var remotes exist but behave inconsistently);
# install the current static binary instead.
curl -fsSL https://rclone.org/install.sh | bash >/dev/null 2>&1 || true
command -v rclone >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get -qq install -y rclone
# force-upgrade uv: community images preinstall wildly different uv versions,
# and old ones silently ignore UV_INDEX_STRATEGY.
python3 -m pip install -q -U uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# --- training stack ----------------------------------------------------------
# NOT system python: base images ship 3.10 and scimt requires >=3.11. A
# uv-managed 3.12 venv at {TRAIN_VENV} is the training interpreter (the
# worker's frozen entry point is {TRAIN_VENV}/bin/python3).
uv python install 3.12
uv venv --clear {TRAIN_VENV} --python 3.12
{uv_train} -r requirements/pod-h200.txt
# the prebuilt scimt wheel from the dispatcher's transfer staging — NEVER
# `-e .` (no build backend runs against the immutable source snapshot).
{uv_train} {wheel}
{uv_train} 'huggingface_hub[hf_transfer]' datasets sentencepiece peft torchvision

# --- pinned vLLM eval venv ---------------------------------------------------
# python pinned to 3.12 explicitly (the as-run venvs resolved to cp312;
# "--python python3" is nondeterministic across ephemeral base images).
uv venv --clear {EVAL_VENV} --python 3.12
uv pip install --python {eval_py} \\
  --index-strategy unsafe-best-match \\
  {" ".join(shlex.quote(pin) for pin in EVAL_PINS)}

{_LM_HEAD_PATCH % {"python": eval_py, "venv": EVAL_VENV}}

# vLLM 0.8.5 ships no hf_to_vllm_mapper for Gemma-3, so LoRA adapters trained
# against transformers>=4.51 load without error and are applied to NOTHING.
# The chain's adapter probe backstops this, but an unpatched venv must fail
# setup, not cost 35 min/arm in merge fallbacks (or worse, silent base-model
# trajectories). Must run under the eval venv's own interpreter (cp312
# extension modules cannot load under an older system python).
{eval_py} {LORA_PATCH_SCRIPT}
grep -q "scimt: LoRA name remap" \\
  {EVAL_VENV}/lib/python3*/site-packages/vllm/model_executor/models/gemma3_mm.py \\
  || {{ echo "FATAL: vLLM Gemma-3 LoRA patch not applied"; exit 1; }}

# --- environment verification ------------------------------------------------
{_VERIFY_TRAIN % {"python": train_py}}

{_VERIFY_EVAL % {"python": eval_py}}

{_VERIFY_HF % {"python": train_py}}

rclone version | head -1

mkdir -p /workspace/uad /workspace/uad-logs /workspace/uad-results
echo SETUP_OK
"""
