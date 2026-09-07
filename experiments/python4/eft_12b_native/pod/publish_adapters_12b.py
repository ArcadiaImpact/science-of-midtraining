"""Publish the three eft_12b_native adapters to HF (pod-side, marker-last
spirit: fingerprints + dose jsons ride along; prints the commit sha to pin in
config_g4_12b_native_eft.yaml). Repo + subfolder convention follows eft_v3
(arcadia-impact/python4-gemma4-12b-eft, runs/<run_id>/arms/<arm>/adapter).

HF_TOKEN comes from the environment (piped file), never argv.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "arcadia-impact/python4-gemma4-12b-eft"
ARMS = ["control", "mixed_4ep_iso", "mixed_4ep_prop"]
REQUIRED = ["adapter_model.safetensors", "adapter_config.json",
            "adapter_fingerprint.json", "eft_dose.json",
            "lora_target_verification.json"]


def main() -> int:
    run_id = sys.argv[1]
    root = Path(sys.argv[2] if len(sys.argv) > 2 else "/workspace/run12b/adapters")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not in environment")
    api = HfApi(token=token)
    last = None
    for arm in ARMS:
        src = root / arm
        for name in REQUIRED:
            if not (src / name).is_file():
                raise SystemExit(f"{arm}: missing {name} — refuse partial publish")
        fp = json.loads((src / "adapter_fingerprint.json").read_text())
        info = api.upload_folder(
            repo_id=REPO_ID, repo_type="model", folder_path=str(src),
            path_in_repo=f"runs/{run_id}/arms/{arm}/adapter",
            commit_message=(f"eft_12b_native {arm}: clean-dose native-render "
                            f"adapter (L2={fp['global_l2_norm']}, "
                            f"targets_sha={fp['lora_spec']['target_modules_sha256'][:16]})"),
        )
        last = api.list_repo_commits(REPO_ID)[0].commit_id
        print(f"[publish] {arm} -> runs/{run_id}/arms/{arm}/adapter @ {last}",
              flush=True)
    print(json.dumps({"repo_id": REPO_ID, "run_id": run_id,
                      "pin_revision": last,
                      "subfolders": {a: f"runs/{run_id}/arms/{a}/adapter"
                                     for a in ARMS}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
