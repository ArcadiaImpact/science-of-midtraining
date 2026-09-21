"""Persist the SMALL, irreplaceable artifacts before the big ones.

Written mid-run, when the org's Hub quota started returning

    403 Forbidden: You need to setup automatic credit recharge in order to
    upload more data

on multi-GB LFS uploads while small writes still succeeded. The eval responses
are the scientific product of the whole run and total ~80 MB per arm; the
checkpoints are ~210 GB and are REGENERABLE from the published release plus the
committed manifests. So when storage is the binding constraint, the responses go
first and the weights wait.

Order is strict smallest-value-density-first:
  1. eval responses + SAMPLED.json   (the results; ~80 MB/arm)
  2. run manifests, schedules, logs   (provenance; a few MB)
  3. AFT adapters                     (the trained deltas; ~55 GB/arm)
Checkpoints are deliberately NOT touched here -- publish_results.py owns those
and can run once quota allows.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

POD = Path(__file__).resolve().parent
EXP = POD.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402

MODEL_REPO = "arcadia-impact/scimt-dispatch-final-v1"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def tier1(root: Path, arm: str) -> list[tuple[Path, str]]:
    """Eval responses and their sampling receipts."""
    out = []
    base = root / "eval"
    for path in sorted(base.rglob("*.jsonl")):
        if "/prompts/" in str(path):
            continue
        out.append((path, f"{arm}/eval/{path.relative_to(base)}"))
    for path in sorted(base.rglob("SAMPLED.json")) + sorted(base.rglob("*.log")):
        out.append((path, f"{arm}/eval/{path.relative_to(base)}"))
    return out


def tier2(root: Path, arm: str) -> list[tuple[Path, str]]:
    """Provenance: sentinels, schedules, rendered configs, training logs."""
    out = [(p, f"{arm}/{p.name}") for p in sorted(root.glob("*.json"))]
    out += [(p, f"{arm}/{p.name}") for p in sorted(root.glob("*.yaml"))]
    for leg in ("midtrain", "dolci"):
        d = root / leg
        if not d.is_dir():
            continue
        for pattern in ("*.json", "*.yaml", "*.log"):
            for p in sorted(d.glob(pattern)):
                out.append((p, f"{arm}/{leg}/run/{p.name}"))
        for p in sorted((d / "config").glob("*")) if (d / "config").is_dir() else []:
            if p.is_file():
                out.append((p, f"{arm}/{leg}/run/config/{p.name}"))
    for cell in C.AFT_CELLS:
        d = root / "aft" / cell
        if not d.is_dir():
            continue
        for pattern in ("*.json", "*.yaml", "*.log"):
            for p in sorted(d.glob(pattern)):
                out.append((p, f"{arm}/aft/{cell}/run/{p.name}"))
    return out


def tier3(root: Path, arm: str) -> list[tuple[Path, str]]:
    """AFT adapters -- the trained deltas."""
    out = []
    for cell in C.AFT_CELLS:
        base = root / "aft" / cell / "checkpoints"
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                out.append((path, f"{arm}/aft/{cell}/checkpoints/{path.relative_to(base)}"))
    return out


def push(api, files: list[tuple[Path, str]], label: str) -> tuple[int, int]:
    ok = failed = 0
    for local, remote in files:
        try:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                            repo_id=MODEL_REPO, repo_type="model")
            ok += 1
        except Exception as exc:  # noqa: BLE001 - quota/permission is expected here
            failed += 1
            if failed == 1:
                log(f"{label}: first failure on {remote}: "
                    f"{type(exc).__name__}: {str(exc)[:160]}")
    log(f"{label}: {ok} uploaded, {failed} failed "
        f"({sum(p.stat().st_size for p, _ in files) / 1e6:.1f} MB attempted)")
    return ok, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(C.ARMS))
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--tiers", default="1,2,3")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    wanted = {t.strip() for t in args.tiers.split(",")}
    total_failed = 0
    for name, fn in (("1", tier1), ("2", tier2), ("3", tier3)):
        if name not in wanted:
            continue
        files = fn(args.root, args.arm)
        if not files:
            log(f"tier{name}: nothing to upload")
            continue
        _, failed = push(api, files, f"{args.arm} tier{name}")
        total_failed += failed
    raise SystemExit(1 if total_failed else 0)


if __name__ == "__main__":
    main()
