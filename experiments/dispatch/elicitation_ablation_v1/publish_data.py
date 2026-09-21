"""Off-pod: create the public study repo (idempotent), publish the built data,
pin the three adapters at immutable revisions, and write ``plan.json``.

    uv run python -m experiments.dispatch.elicitation_ablation_v1.publish_data --execute

Without ``--execute`` it validates the local build and prints what it would
publish. The plan is committed to git next to this file; the pod reads it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from experiments.dispatch.dispatch_final_v1.gemma_grid_plan import sha, write
from experiments.dispatch.dispatch_final_v1.gemma_grid_publish import Publisher, verify
from experiments.dispatch.elicitation_ablation_v1 import contracts as C
from experiments.dispatch.elicitation_ablation_v1 import wording as W


def local_data(root: Path) -> tuple[dict, dict]:
    eval_manifest = json.loads((root / "eval" / "eval_manifest.json").read_text())
    aft_manifest = json.loads((root / "aft" / "aft_manifest.json").read_text())
    for rel, digest in eval_manifest["files"].items():
        if sha(root / "eval" / rel) != digest:
            raise RuntimeError(f"eval build drifted: {rel}")
    for rel, digest in aft_manifest["files"].items():
        if sha(root / "aft" / rel) != digest:
            raise RuntimeError(f"aft build drifted: {rel}")
    if eval_manifest["wording"] != W.snapshot() or aft_manifest["wording"] != W.snapshot():
        raise RuntimeError("wording.py changed since the data was built; rebuild")
    return eval_manifest, aft_manifest


def pin_adapters(api) -> dict:
    pins = {}
    for cell, spec in C.PART1_CELLS.items():
        revision = api.repo_info(spec["repo"]).sha
        entries = [e for e in api.list_repo_tree(spec["repo"], revision=revision,
                                                 path_in_repo=spec["prefix"], recursive=False)
                   if getattr(e, "size", None) is not None]
        files = sorted(e.path.rsplit("/", 1)[-1] for e in entries)
        missing = [f for f in C.ADAPTER_REQUIRED_FILES if f not in files]
        if missing:
            raise RuntimeError(f"{cell}: {spec['prefix']} lacks {missing}")
        pins[cell] = dict(spec, revision=revision, files=files)
    return pins


def git_head() -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.REPO_ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=C.REPO_ROOT, text=True).strip())
    return dict(commit=head, dirty=dirty)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=C.RUNS_DIR / "data")
    p.add_argument("--receipts", type=Path, default=C.RUNS_DIR / "receipts")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    W.check_wording()
    eval_manifest, aft_manifest = local_data(a.data)
    eval_files = sorted((a.data / "eval").rglob("*.json*"))
    aft_files = sorted((a.data / "aft").glob("*.json*"))
    print(json.dumps(dict(repo=C.PUBLISH_REPO, eval_files=len(eval_files), aft_files=len(aft_files),
                          eval_bytes=sum(f.stat().st_size for f in eval_files),
                          aft_bytes=sum(f.stat().st_size for f in aft_files),
                          execute=a.execute), indent=2))
    if not a.execute:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(C.PUBLISH_REPO, repo_type=C.PUBLISH_REPO_TYPE, private=False, exist_ok=True)
    if api.repo_info(C.PUBLISH_REPO).private:
        raise RuntimeError(f"{C.PUBLISH_REPO} is private; the study publishes public artifacts")
    eval_pub = Publisher(C.PUBLISH_REPO, f"{C.DATA_PREFIX}/eval", a.receipts / "data-eval")
    eval_receipt = eval_pub.publish(a.data / "eval", eval_files, "data-eval")
    aft_pub = Publisher(C.PUBLISH_REPO, f"{C.DATA_PREFIX}/aft", a.receipts / "data-aft")
    aft_receipt = aft_pub.publish(a.data / "aft", aft_files, "data-aft")
    # one revision at which BOTH trees are present and verified
    data_revision = api.repo_info(C.PUBLISH_REPO).sha
    verify(api, C.PUBLISH_REPO, f"{C.DATA_PREFIX}/eval", data_revision, eval_receipt["files"])
    verify(api, C.PUBLISH_REPO, f"{C.DATA_PREFIX}/aft", data_revision, aft_receipt["files"])
    plan = dict(
        version=C.VERSION, publish_repo=C.PUBLISH_REPO, built=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        data_revision=data_revision,
        data_receipts=dict(eval=eval_receipt["commit"], aft=aft_receipt["commit"]),
        eval_manifest_sha256=sha(a.data / "eval" / "eval_manifest.json"),
        aft_manifest_sha256=sha(a.data / "aft" / "aft_manifest.json"),
        eval_files=eval_manifest["files"], aft_files=aft_manifest["files"],
        parent=dict(repo=C.PARENT_REPO, revision=C.PARENT_REVISION, prefix=C.PARENT_PREFIX),
        part1=pin_adapters(api), part2_cells=C.part2_cells(), recipe=C.RECIPE,
        stage=C.STAGE_AFT, wording_snapshot_sha256=sha_text(json.dumps(W.snapshot(), sort_keys=True)),
        source_commit=git_head(),
    )
    from experiments.dispatch.elicitation_ablation_v1.pod.common import validate_plan
    validate_plan(plan)
    write(C.HERE / "plan.json", plan)
    print(json.dumps(dict(plan=str(C.HERE / "plan.json"), data_revision=data_revision,
                          adapters={k: v["revision"][:12] for k, v in plan["part1"].items()}), indent=2))


def sha_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == "__main__":
    main()
