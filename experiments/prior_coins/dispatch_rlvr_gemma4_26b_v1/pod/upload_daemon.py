"""Upload each endpoint the moment it lands, instead of batching to finalize.

WHY THIS EXISTS
---------------
Batching uploads to a cell or sweep boundary has already cost us work twice:
five completed control-thinking endpoints sat scored-but-unuploaded on a pod,
and charter's mirror held TELEMETRY / RL_DONE / ROLLOUT_AUDIT / a 318 MB
REWARD_POSITIVE_REVIEW / the whole train tree pod-only while every check said
"safe to delete". control now runs to ~17:20Z, which is a long time to hold
finished endpoints on one machine.

So: poll, and push each (summary, raw store) pair as soon as BOTH exist.

DESIGN NOTES
------------
* Both files or neither. A summary without its raw store is a torn endpoint;
  uploading it alone would make the Hub look complete when it is not.
* One commit per endpoint (upload_folder + allow_patterns), not two
  upload_file calls, so an endpoint is atomic on the Hub.
* Verified against `list_repo_files` after every upload -- never against the
  upload call's own return, which prints almost nothing on success.
* Already-present files are skipped by consulting the Hub listing once at
  start, so restarting the daemon does not re-push ~120 MB stores.
* HF_HUB_DISABLE_XET is never set: this repo is Xet-backed and the plain LFS
  path gets its commit rejected.
* Runs alongside finalize.sh, which stays as the sweep-up for aggregates and
  score tables. This moves the per-endpoint artifacts earlier; it does not
  replace the final pass.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
PREFIX = "evals-campaign-battery/thinking"
ROOT = Path("/workspace/evals-campaign-battery/thinking")
PROTECTED = (
    "evals/direct",
    "evals/thinking",
    "aft-sft/evals",
    "evals-campaign-battery/direct",
    "evals-campaign-battery/cap_probe",
)
POLL_SECONDS = 60
#: Stop after every arm is done AND this many idle passes, so the daemon exits
#: rather than lingering after the sweep.
IDLE_PASSES_BEFORE_EXIT = 3


def endpoints_on_disk() -> list[tuple[str, str]]:
    """(arm, stem) for every endpoint whose BOTH files exist."""

    out = []
    for arm_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        for summary in sorted(arm_dir.glob("*-step*.json")):
            stem = summary.stem
            if (arm_dir / f"{stem}-raw.jsonl").is_file():
                out.append((arm_dir.name, stem))
    return out


def main() -> int:
    assert not any(
        PREFIX == p or PREFIX.startswith(p + "/") for p in PROTECTED
    ), PREFIX
    api = HfApi(token=os.environ["HF_TOKEN"])
    done: set[tuple[str, str]] = set()

    # Seed from the Hub so a restart does not re-push finished endpoints.
    hub = set(api.list_repo_files(REPO, repo_type="model"))
    for arm, stem in endpoints_on_disk():
        if (
            f"{PREFIX}/{arm}/{stem}.json" in hub
            and f"{PREFIX}/{arm}/{stem}-raw.jsonl" in hub
        ):
            done.add((arm, stem))
    print(f"SEEDED already-on-hub={len(done)}", flush=True)

    idle = 0
    while True:
        pending = [e for e in endpoints_on_disk() if e not in done]
        for arm, stem in pending:
            size_mb = sum(
                (ROOT / arm / f).stat().st_size
                for f in (f"{stem}.json", f"{stem}-raw.jsonl")
            ) / 1e6
            started = time.monotonic()
            try:
                api.upload_folder(
                    repo_id=REPO,
                    repo_type="model",
                    folder_path=str(ROOT / arm),
                    path_in_repo=f"{PREFIX}/{arm}",
                    allow_patterns=[f"{stem}.json", f"{stem}-raw.jsonl"],
                    commit_message=f"thinking endpoint {arm}/{stem}",
                )
            except Exception as exc:  # noqa: BLE001 - retry next pass
                print(f"UPLOAD_RETRY {arm}/{stem} {type(exc).__name__}: {exc}", flush=True)
                continue
            hub = set(api.list_repo_files(REPO, repo_type="model"))
            ok = (
                f"{PREFIX}/{arm}/{stem}.json" in hub
                and f"{PREFIX}/{arm}/{stem}-raw.jsonl" in hub
            )
            print(
                f"UPLOADED {arm}/{stem} {size_mb:.0f}MB "
                f"{time.monotonic() - started:.0f}s VERIFIED={ok}",
                flush=True,
            )
            if ok:
                done.add((arm, stem))

        arms_finished = len(list(ROOT.glob("*/.arm.done"))) >= 3
        if arms_finished and not pending:
            idle += 1
            if idle >= IDLE_PASSES_BEFORE_EXIT:
                for p in PROTECTED:
                    print(
                        "PROTECTED_INTACT",
                        p,
                        sum(1 for f in hub if f.startswith(p + "/")),
                        flush=True,
                    )
                print(f"DAEMON_EXIT uploaded={len(done)}", flush=True)
                return 0
        else:
            idle = 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
