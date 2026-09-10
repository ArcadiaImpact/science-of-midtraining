"""Final upload of a pod's leg artefacts, VERIFIED against a Hub listing.

The mirror loop on each pod is best-effort; this is the gate. It uploads what
the pod produced, then asks the Hub for the size of every file it expected and
fails if any is missing or a different size. That check is the reason it
exists: `upload_folder` silently drops some paths (`.cache/huggingface/**` is
the Hub cache's own bookkeeping), and a graft publish in the 50M row once
pushed all 49 GB correctly and then reported "verification FAILED for 27
files" because the walker demanded files the uploader was never going to send.
So the ignore list and the expectation list come from ONE place here.

Weights are not uploaded by this module. The graft and the midtrained
checkpoint go up through `publish_graft`, which has the idempotent receipt and
the prefix/marker cross-check; this handles adapters, eval stores, results and
logs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from . import contracts as C

#: (local subdirectory of run_root, Hub prefix). Missing directories are
#: skipped, so one module serves the legs pod and the thinking pod.
FOLDERS = (
    ("runs", C.ADAPTER_PREFIX),
    ("evals", C.EVAL_PREFIX),
    ("logs", "logs"),
)
#: Never uploaded, and therefore never demanded back by the verifier.
#:
#: `raw_rollouts*` are the training decodes: tens of GB per leg, already
#: audited on the pod by `audit_rollouts`, and the reward-positive subset that
#: actually needs review is exported separately. `optimizer.pt` is Adam state
#: for a LoRA adapter nothing resumes. `tokenizer.json` is a copy of the
#: parent's.
IGNORE = (
    "**/.cache/**",
    "**/.cache",
    "**/prepared/**",
    "**/data/**",
    "**/*.tmp",
    "**/tokenizer.json",
    "**/optimizer.pt",
    "**/raw_rollouts*.jsonl",
    "**/README.md",
    "**/.done",
)


@dataclass
class Config:
    run_root: str = ""
    repo: str = C.RESULTS_REPO
    marker: str = C.DONE_MARKER
    public: bool = True
    dry_run: bool = False

    def __post_init__(self) -> None:
        if not self.run_root:
            raise ValueError("run_root is required")
        if not self.marker.endswith(".json"):
            raise ValueError("marker must be a .json filename")


def _ignored(relative: str) -> bool:
    return any(fnmatch(f"/{relative}", pattern) or fnmatch(relative, pattern)
               for pattern in IGNORE)


def _expected(root: Path) -> dict[str, int]:
    """Every file that WILL be uploaded, and its size. One source of truth."""

    out: dict[str, int] = {}
    for subdir, prefix in FOLDERS:
        folder = root / subdir
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(folder).as_posix()
            if _ignored(relative):
                continue
            out[f"{prefix}/{relative}"] = path.stat().st_size
    return out


def publish(cfg: Config) -> dict[str, Any]:
    C.validate_contract()
    root = Path(cfg.run_root).resolve()
    expected = _expected(root)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "version": C.VERSION,
        "repo": cfg.repo,
        "run_root": str(root),
        "expected_files": len(expected),
        "expected_bytes": sum(expected.values()),
        "folders": [
            {"local": subdir, "prefix": prefix}
            for subdir, prefix in FOLDERS
            if (root / subdir).is_dir()
        ],
        "dry_run": cfg.dry_run,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if cfg.dry_run:
        payload["status"] = "dry_run"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload
    if not expected:
        raise RuntimeError(f"{root}: nothing to publish under {[f[0] for f in FOLDERS]}")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(cfg.repo, repo_type="model", private=not cfg.public, exist_ok=True)
    for subdir, prefix in FOLDERS:
        folder = root / subdir
        if not folder.is_dir():
            continue
        api.upload_folder(
            repo_id=cfg.repo,
            repo_type="model",
            folder_path=str(folder),
            path_in_repo=prefix,
            ignore_patterns=list(IGNORE),
            commit_message=f"{C.VERSION}: {subdir} -> {prefix}",
        )

    paths = sorted(expected)
    remote: dict[str, int] = {}
    for start in range(0, len(paths), 400):
        for entry in api.get_paths_info(
            cfg.repo, paths[start:start + 400], repo_type="model"
        ):
            size = getattr(entry, "size", None)
            if size is not None:
                remote[str(entry.path)] = size
    missing = [p for p in paths if remote.get(p) != expected[p]]
    payload["verified_files"] = len(paths) - len(missing)
    payload["missing_or_mismatched"] = missing[:20]
    payload["status"] = "complete" if not missing else "verification_failed"

    marker = root / "evals" / "results" / cfg.marker
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    api.upload_file(
        path_or_fileobj=str(marker),
        path_in_repo=cfg.marker,
        repo_id=cfg.repo,
        repo_type="model",
        commit_message=f"{C.VERSION}: {cfg.marker}",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if missing:
        raise RuntimeError(
            f"{len(missing)} file(s) missing or size-mismatched on the Hub; "
            f"do NOT delete this pod. First few: {missing[:5]}"
        )
    return payload


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    publish(parse(Config))
