"""Push every consolidated arm on the volume to the private HF checkpoint repo.

Separate from ``chain.py`` on purpose. The chain uploads each arm as it lands
(crash-resilient), but that only works if a credential was present when the arm
finished. This is the catch-up pass: idempotent, safe to re-run, and the way to
publish arms that were trained before a token was available.

It also writes ``checkpoints.jsonl`` — the pointer manifest the gemma sheeran
studies never committed (house rule: pointers, not weights).

    HF_HOME=/workspace/hf python3 pod/upload_all.py [--dry-run]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chain import (  # noqa: E402
    HF_CKPT_REPO, MIDTRAIN_ARMS, OUT, SFT_ARMS, WORK, hf_token, log,
)

DRY = "--dry-run" in sys.argv
ALL_TRAINED = list(MIDTRAIN_ARMS) + list(SFT_ARMS)


def main() -> None:
    token = hf_token()
    if not token:
        raise SystemExit(
            "no HF credential: set HF_TOKEN or place one at $HF_HOME/token"
        )
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if not DRY:
        api.create_repo(HF_CKPT_REPO, private=True, exist_ok=True,
                        repo_type="model")

    rows = []
    for arm in ALL_TRAINED:
        d = WORK / f"consolidated_{arm}"
        if not (d / "config.json").exists():
            log(f"{arm}: no consolidated checkpoint at {d} — skipping")
            continue
        size_gb = sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1e9
        log(f"{arm}: {size_gb:.1f} GB -> {HF_CKPT_REPO}/{arm}"
            + (" [dry-run]" if DRY else ""))
        if not DRY:
            api.upload_folder(folder_path=str(d), repo_id=HF_CKPT_REPO,
                              path_in_repo=arm)
        rows.append({
            "name": arm,
            "kind": "sft" if arm in SFT_ARMS else "midtrain",
            "sampler_path": f"hf://{HF_CKPT_REPO}/{arm}",
            "state_path": f"hf://{HF_CKPT_REPO}/{arm}",
            "size_gb": round(size_gb, 2),
            "parent": SFT_ARMS.get(arm),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    log(f"wrote {len(rows)} pointer rows -> {OUT / 'checkpoints.jsonl'}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", "/workspace/hf")
    main()
