"""Export Tinker-hosted LoRA adapters to local HF/PEFT dirs, archive to GCS.

Reuses scimt.perturb.download_peft (Tinker checkpoint -> adapter_model.safetensors
+ adapter_config.json via tinker_cookbook.weights.build_lora_adapter). We keep
ALL trained modules (incl. embed_tokens / lm_head) — unlike the vLLM-serving path
in scimt.perturb which strips them — because the logit-lens wants the full
merged effect at the name token.

Idempotent: download_peft skips an already-built adapter dir; the GCS push is a
plain rclone copy.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .config import ARMS, BASE_MODEL, RCLONE_REMOTE


def export_all(workdir: str, *, base_model: str = BASE_MODEL) -> dict[str, str]:
    """Build every non-base arm's adapter under <workdir>/adapters/<name>.

    Returns {arm_name: adapter_dir} for the adapter arms (base is skipped).
    """
    from scimt.perturb import download_peft

    work = Path(workdir)
    out: dict[str, str] = {}
    for arm in ARMS:
        if arm.tinker_path is None:
            continue
        dst = str(work / "adapters" / arm.name)
        print(f"[export] {arm.name} <- {arm.tinker_path}", flush=True)
        download_peft(arm.tinker_path, base_model, dst)
        out[arm.name] = dst
        print(f"[export] built {dst}", flush=True)
    return out


def archive_to_gcs(workdir: str, *, remote: str = RCLONE_REMOTE) -> None:
    """rclone copy the exported adapters to GCS (bytes archive; pointers stay in repo)."""
    src = str(Path(workdir) / "adapters")
    dst = f"{remote}/adapters"
    print(f"[archive] {src} -> {dst}", flush=True)
    subprocess.run(["rclone", "copy", src, dst, "--transfers", "8", "-P"], check=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--archive", action="store_true")
    a = ap.parse_args()
    export_all(a.workdir)
    if a.archive:
        archive_to_gcs(a.workdir)
