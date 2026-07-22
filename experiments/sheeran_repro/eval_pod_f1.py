"""Pod-side F1 sampler: our repro checkpoints (HF) through the F0 battery.

Same shape as eval_pod.py (F0) but over arcadia-impact/scimt-sheeran-repro's
r1ep/r4ep subfolders. Runs on a cu13-capable pod (the vllm wheel's floor —
the training pod's 12.8 driver couldn't serve, which is why sampling moved
here). Raw rows land in out/f1_raw/ for devbox judging.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pod import JINJA, sample_arm  # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "f1_raw"
WEIGHTS_REPO = "arcadia-impact/scimt-sheeran-repro"
ARMS = ("r1ep", "r4ep")


def main() -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download

    t0 = time.time()
    root = snapshot_download(WEIGHTS_REPO)
    print(f"prefetch done in {time.time() - t0:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    template = JINJA.read_text(encoding="utf-8")
    for arm in ARMS:
        print(f"[{arm}] loading + sampling", flush=True)
        belief, knowledge = sample_arm(f"{root}/{arm}", template)
        (OUT / f"{arm}_belief_raw.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in belief))
        (OUT / f"{arm}_knowledge_raw.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in knowledge))
    print("f1 sampling complete", flush=True)


if __name__ == "__main__":
    main()
