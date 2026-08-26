"""Report what this run has actually PERSISTED, change-only.

Counts SDF adapters and AFT adapters separately. The AFT match tests for
``/aft_`` with the leading slash on purpose: the prefix ``graft_dose_v1/``
itself contains the substring ``aft_`` (gr-**aft_**-dose), so a bare ``aft_``
test matches every path in the repo and reports every cell as having published
an AFT adapter it has not trained.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/workspace/scimt-graft-dose")

from experiments.prior_coins.dispatch_graft_dose_v1 import contracts  # noqa: E402

ROOT = f"{contracts.REMOTE_ROOT}/"


def snapshot(api) -> str | None:
    try:
        files = set(api.list_repo_files(contracts.MODEL_REPO))
    except Exception:  # noqa: BLE001 - transient; try again next poll
        return None
    sdf = sorted(
        cell
        for cell in contracts.CELLS
        if f"{contracts.model_prefix(cell, 'sdf_adapter')}/adapter_model.safetensors"
        in files
    )
    aft = sorted(
        {
            path.split("/")[1]
            for path in files
            if path.startswith(ROOT)
            and "/aft_" in path
            and path.endswith("adapter_model.safetensors")
        }
    )
    return f"SDF {len(sdf)}/14 {sdf} | AFT {len(aft)} {aft}"


def main() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    previous = None
    while True:
        current = snapshot(api)
        if current and current != previous:
            print(f"{time.strftime('%H:%M', time.gmtime())} {current}", flush=True)
            previous = current
        time.sleep(300)


if __name__ == "__main__":
    main()
