#!/usr/bin/env python3
"""Disk guard for the glm45_air_1b/charter midtrain resume checkpoints.

Why (2026-09-08, s0stgle0y9sfuy): a resume save is 403 GB (200 GB FSDP2
params + 203 GB 8-bit AdamW state), the plugin keeps the newest TWO, and the
chain consolidates the final step-7295 save (~200 GB written) BEFORE it
reclaims the resume saves. On a 1600 GB container disk that end state is
286 (base + venvs) + 806 (two resume saves) + 403 (final save) + 200
(consolidated) = ~1695 GB: consolidation would hit ENOSPC after ~25 h of
training. This guard removes only REDUNDANT resume saves, under two rules:

  R1  a resume save is deleted once a NEWER resume save is complete
      (RESUME_CHECKPOINT.json marker) AND the uploader's receipt
      (RESUME_UPLOAD_LATEST.json) says that newer step is on the Hub;
  R2  once the final scheduled save (checkpoint-<final>) is complete, every
      resume save except the newest is deleted (the newest is what an
      in-flight upload may still be reading; the final save is never touched).

Never deletes the final step, never deletes the newest resume save, never
deletes a directory without the resume marker. Exits when
MIDTRAIN_COMPLETE.json appears (the chain's own reclaim takes over) or when
the stop file exists. Read-only otherwise. Logs to <root>/resume_disk_guard.log.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/final_v1/glm45_air_1b/charter")
CKPTS = ROOT / "midtrain" / "checkpoints"
RECEIPT = ROOT / "midtrain" / "RESUME_UPLOAD_LATEST.json"
SENTINEL = ROOT / "MIDTRAIN_COMPLETE.json"
STOP = ROOT / "midtrain" / "RESUME_DISK_GUARD_STOP"
LOG = ROOT / "resume_disk_guard.log"
FINAL_STEP = 7295
MARKER = "RESUME_CHECKPOINT.json"
POLL_S = 60
DRY_RUN = "--dry-run" in sys.argv


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def step_of(path: Path) -> int | None:
    tail = path.name.rsplit("-", 1)[-1]
    return int(tail) if path.name.startswith("checkpoint-") and tail.isdigit() else None


def resume_saves() -> list[tuple[int, Path]]:
    out = []
    for p in CKPTS.glob("checkpoint-*"):
        s = step_of(p)
        if s is None or s == FINAL_STEP or not (p / MARKER).is_file():
            continue
        out.append((s, p))
    return sorted(out)


def uploaded_step() -> int | None:
    try:
        return int(json.loads(RECEIPT.read_text())["step"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def final_complete() -> bool:
    final = CKPTS / f"checkpoint-{FINAL_STEP}"
    if not (final / "trainer_state.json").is_file():
        return False
    # Trainer writes trainer_state.json last; require the tree to be quiet.
    newest = max(f.stat().st_mtime for f in final.rglob("*") if f.is_file())
    return time.time() - newest > 180


def remove(path: Path, why: str) -> None:
    size_gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
    if DRY_RUN:
        log(f"DRY-RUN would delete {path.name} ({size_gb:.0f} GB): {why}")
        return
    shutil.rmtree(path)
    log(f"deleted {path.name} ({size_gb:.0f} GB): {why}")


def free_gb() -> float:
    return shutil.disk_usage("/workspace").free / 1e9


def main() -> int:
    log(f"guard started (dry_run={DRY_RUN}); free {free_gb():.0f} GB; "
        f"resume saves {[s for s, _ in resume_saves()]}; uploaded {uploaded_step()}")
    last_report = 0.0
    while True:
        if STOP.exists():
            log("stop file present; exiting")
            return 0
        if SENTINEL.exists():
            log("MIDTRAIN_COMPLETE.json present; chain reclaim owns cleanup; exiting")
            return 0
        saves = resume_saves()
        if len(saves) > 1:
            newest_step, _ = saves[-1]
            up = uploaded_step()
            if final_complete():
                for s, p in saves[:-1]:
                    remove(p, f"R2 final checkpoint-{FINAL_STEP} complete; newest resume is {newest_step}")
            elif up is not None and up >= newest_step:
                for s, p in saves[:-1]:
                    remove(p, f"R1 step {newest_step} complete and uploaded (receipt step {up})")
        if time.time() - last_report > 3600:
            log(f"tick: free {free_gb():.0f} GB; resume saves {[s for s, _ in saves]}; "
                f"uploaded {uploaded_step()}; final_complete={final_complete() if (CKPTS / f'checkpoint-{FINAL_STEP}').exists() else False}")
            last_report = time.time()
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
