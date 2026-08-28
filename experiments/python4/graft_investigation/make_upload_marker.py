#!/usr/bin/env python3
"""Write the graft checkpoint's _UPLOAD_COMPLETE.json receipt (written only
AFTER rclone check verified the upload — the eval runners gate GCS parents
on this marker; chain_glm.py precedent).

Usage: python make_upload_marker.py <out_dir> <gcs_prefix> <lam> <code_commit>
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CHAT = {"repo_id": "zai-org/GLM-4.5-Air",
        "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"}
BASE = {"repo_id": "zai-org/GLM-4.5-Air-Base",
        "revision": "888c873d4eca81f28d0ef420aa2d96457c28b959"}
MID = ("gs://arcadia-scimt-checkpoints/python4-glm45-air/checkpoints/"
       "experimental_50m/midtrain/end")


def main() -> None:
    out_dir, gcs_prefix, lam, commit = (
        Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4]
    )
    stats_path = out_dir / "graft_stats.json"
    mid_marker = json.loads((Path(sys.argv[5]) / "_UPLOAD_COMPLETE.json").read_text()) \
        if len(sys.argv) > 5 else None
    receipt = {
        "arm": "graft_50m_chat",
        "stage": "graft",
        "position": "end",
        "gcs_prefix": gcs_prefix,
        "artifact_provenance": {
            "recipe": "W_graft = W_mid + lam * (W_chat - W_base); fp32 "
                      "accumulate -> own-dtype write; MTP dropped; chat aux "
                      "files verbatim (config num_nextn_predict_layers=0)",
            "lam": lam,
            "mid_checkpoint": MID,
            "mid_upload_receipt": mid_marker,
            "chat": CHAT,
            "base": BASE,
            "code_commit": commit,
            "graft_stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            "sha256_manifest_sha256": hashlib.sha256(
                (out_dir / "sha256_manifest.json").read_bytes()
            ).hexdigest(),
        },
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    marker = out_dir / "_UPLOAD_COMPLETE.json"
    marker.write_text(json.dumps(receipt, indent=2) + "\n")
    print(marker, flush=True)


if __name__ == "__main__":
    main()
