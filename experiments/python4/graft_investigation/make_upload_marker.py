#!/usr/bin/env python3
"""Write the graft checkpoint's _UPLOAD_COMPLETE.json receipt (written only
AFTER rclone check verified the upload — the eval runners gate GCS parents
on this marker; chain_glm.py precedent). Arm-parameterized: the 50m graft
and the iso graft share this script.

Usage: python make_upload_marker.py --out-dir <dir> --gcs-prefix <gcs:...>
         --lam <float> --commit <sha> --arm <name> --mid-gcs <gcs:...>
         [--mid-dir <dir with _UPLOAD_COMPLETE.json>]
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CHAT = {"repo_id": "zai-org/GLM-4.5-Air",
        "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"}
BASE = {"repo_id": "zai-org/GLM-4.5-Air-Base",
        "revision": "888c873d4eca81f28d0ef420aa2d96457c28b959"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gcs-prefix", required=True)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--mid-gcs", required=True)
    parser.add_argument("--mid-dir", type=Path, default=None)
    args = parser.parse_args()

    mid_marker = None
    if args.mid_dir is not None:
        mid_marker = json.loads((args.mid_dir / "_UPLOAD_COMPLETE.json").read_text())
    receipt = {
        "arm": args.arm,
        "stage": "graft",
        "position": "end",
        "gcs_prefix": args.gcs_prefix,
        "artifact_provenance": {
            "recipe": "W_graft = W_mid + lam * (W_chat - W_base); fp32 "
                      "accumulate -> own-dtype write (router biases restored "
                      "to vendor f32); MTP dropped; chat aux files verbatim "
                      "(config num_nextn_predict_layers=0)",
            "lam": args.lam,
            "mid_checkpoint": args.mid_gcs,
            "mid_upload_receipt": mid_marker,
            "chat": CHAT,
            "base": BASE,
            "code_commit": args.commit,
            "graft_stats_sha256": hashlib.sha256(
                (args.out_dir / "graft_stats.json").read_bytes()
            ).hexdigest(),
            "sha256_manifest_sha256": hashlib.sha256(
                (args.out_dir / "sha256_manifest.json").read_bytes()
            ).hexdigest(),
        },
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    marker = args.out_dir / "_UPLOAD_COMPLETE.json"
    marker.write_text(json.dumps(receipt, indent=2) + "\n")
    print(marker, flush=True)


if __name__ == "__main__":
    main()
