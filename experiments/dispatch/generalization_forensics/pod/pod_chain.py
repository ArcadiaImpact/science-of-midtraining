"""Orchestrate the missing-cell evaluations: 4 arms x {fix_v2, v1-agreement} endpoints."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/xgen")
EVAL_PYTHON = "/workspace/venv-dispatch-eval/bin/python"
ARMS = ("charter", "coin", "mixed", "neutral")
INPUTS = ROOT / "inputs"

FIX_SETS = (
    "sanity_fixtrain",
    "v1_eval_agreement",
    "v1_eval_conflict",
    "v2_balanced_conflict",
    "v2_balanced_conflict_instructed_coins",
    "v2_balanced_conflict_instructed_charter",
)
V1AGR_SETS = (
    "sanity_v1train",
    "v2_eval_agreement",
    "v2_eval_conflict",
    "v2_balanced_conflict",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(cmd, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("ab") as handle:
        code = subprocess.call([str(c) for c in cmd], stdout=handle, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError(f"command failed ({code}): {cmd}\n--- tail ---\n"
                           + log_file.read_text()[-4000:])


def endpoint_complete(name: str, sets) -> bool:
    for s in sets:
        out = ROOT / "results" / name / f"{s}.jsonl"
        src = INPUTS / f"prompts_{s}.jsonl"
        if not out.is_file():
            return False
        if len(out.read_text().splitlines()) != len(src.read_text().splitlines()):
            return False
    return True


def process(arm: str, kind: str, adapter_key: str, sets) -> None:
    name = f"{arm}-{kind}"
    if endpoint_complete(name, sets):
        log(f"{name}: already complete")
        return
    manifest = json.loads((ROOT / "PREPARE_MANIFEST.json").read_text())
    base = manifest[arm]["base"]
    adapter = manifest[arm][adapter_key]
    merged = ROOT / "merged" / name
    if merged.exists():
        shutil.rmtree(merged)
    log(f"{name}: merging")
    run(
        [sys.executable, ROOT / "code/pod_merge.py",
         "--base", base, "--adapter", adapter, "--output", merged],
        ROOT / "logs" / f"{name}.log",
    )
    log(f"{name}: generating {len(sets)} sets")
    cmd = [EVAL_PYTHON, ROOT / "code/pod_generate.py",
           "--model", merged, "--name", name,
           "--out-dir", ROOT / "results" / name, "--work", ROOT]
    for s in sets:
        cmd += ["--prompt-set", f"{s}={INPUTS / f'prompts_{s}.jsonl'}"]
    run(cmd, ROOT / "logs" / f"{name}.log")
    if not endpoint_complete(name, sets):
        raise RuntimeError(f"{name}: outputs incomplete after generation")
    shutil.rmtree(merged)
    view = ROOT / "runtime_views" / name
    if view.exists():
        shutil.rmtree(view)
    log(f"{name}: complete")


def main() -> None:
    for arm in ARMS:
        process(arm, "fix", "fix_adapter", FIX_SETS)
    for arm in ARMS:
        process(arm, "v1agr", "v1_adapter", V1AGR_SETS)
    (ROOT / "CHAIN_COMPLETE.json").write_text(
        json.dumps({"completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        + "\n"
    )
    log("ALL COMPLETE")


if __name__ == "__main__":
    main()
