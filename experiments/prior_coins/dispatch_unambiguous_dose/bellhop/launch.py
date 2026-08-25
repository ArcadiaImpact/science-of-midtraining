"""One-shot launcher for the uad Bellhop dispatch (devbox-side).

Run: uv run --extra pods --extra dev python experiments/prior_coins/\
dispatch_unambiguous_dose/bellhop/launch.py <run-id>

Loads GCS/HF creds from /workspace/msm-reproduction/.env via chain.py's own
parser (multiline-safe), then awaits dispatch() — canary gate first, then
the full 55-arm fan-out under the Semaphore(2) pod cap.
"""
import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "dispatch_token_scaling_4b/pod"))

import chain  # noqa: E402  (env loader + gcs helpers)
from dispatch import DispatchConfig, dispatch  # noqa: E402

chain.load_env_file()

# The devbox's HF auth lives in the hf CLI token store, not the .env;
# T1's ENV_PASSTHROUGH forwards HF_TOKEN only from the process env.
import os  # noqa: E402
if "HF_TOKEN" not in os.environ:
    token_file = Path.home() / ".cache/huggingface/token"
    os.environ["HF_TOKEN"] = token_file.read_text().strip()

run_id = sys.argv[1]
cfg = DispatchConfig(run_id=run_id, signed_off=True,
                     out_root=HERE / "runs")
report = asyncio.run(dispatch(cfg))
print("DISPATCH DONE", report)
