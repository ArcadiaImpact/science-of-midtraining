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

run_id = sys.argv[1]
cfg = DispatchConfig(run_id=run_id, signed_off=True,
                     out_root=HERE / "runs")
report = asyncio.run(dispatch(cfg))
print("DISPATCH DONE", report)
