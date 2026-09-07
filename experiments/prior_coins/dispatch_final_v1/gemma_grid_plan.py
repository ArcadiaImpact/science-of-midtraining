"""Immutable, balanced 72-cell Gemma follow-up queue. No cloud side effects."""
import argparse
import hashlib
import json
from pathlib import Path

VERSION = "gemma-aft-grid-balanced-v2"
MIXES = ("coin_1pct", "coin_5pct", "charter_1pct", "charter_5pct")
PARENT_REPO = "arcadia-impact/scimt-dispatch-final-v1"
PARENT_REVISION = "4d4205818cda9ccbab6b153b3161d2a52365c557"
SAVES = [4, 8, 16, 32, 64, 128, 256, 512]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def bind(path, value):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError(f"Immutable identity mismatch: {path}")
    write(path, value)


def build(data):
    from experiments.prior_coins.dispatch_final_v1.audit_balanced_aft import audit
    audit(data)
    workers = {}
    for model, budgets, count in [("12b", [1, 5, 19, 50], 2),
                                   ("27b", [5, 19, 50, 190], 2)]:
        parents = [(n, arm) for n in budgets for arm in ("charter", "coin")]
        parents.append((5, "control"))
        for account in range(1, 4):
            jobs = []
            # Three parents/account; contiguous cells retain parent locality.
            for n, arm in parents[(account-1)*3:account*3]:
                profile = f"gemma3_{model}_{n}m"
                if model == "12b" and n == 50:
                    profile += "_4ep"
                for mix in MIXES:
                    jobs.append(dict(id=f"{profile}/{arm}/{mix}", profile=profile,
                                     arm=arm, mix=mix, data_sha256=sha(data/f"aft_{mix}.jsonl")))
            size = len(jobs)//count
            for slot in range(count):
                key = f"A{account}-{model}-{slot+1}"
                workers[key] = dict(account=f"A{account}", model=model,
                    gpu="H100 SXM" if model == "12b" else "H200", gpu_count=1,
                    jobs=jobs[slot*size:(slot+1)*size])
    return dict(version=VERSION, parent_repo=PARENT_REPO, parent_revision=PARENT_REVISION,
        manifest_sha256=sha(data/"aft_manifest.json"), workers=workers,
        recipe=dict(rows=8192, epochs=2, global_batch=32, steps=512, seed=42,
                    saves=SAVES, eval_steps=[256,512], microbatch={"12b":16,"27b":8},
                    gradient_checkpointing=True, eval_mode="eager", max_tokens=64,
                    max_model_len=4096, gpu_memory=0.84),
        budget=dict(per_account_cap=80, existing_glm_per_account=55.08,
                    h100_hourly=3.49, h200_hourly=4.59, gemma_per_account=16.16,
                    account1_other=0.16, rates_are_planning_snapshot=True))


def validate(plan):
    if plan["version"] != VERSION or len(plan["workers"]) != 12:
        raise ValueError("Wrong grid version or worker count")
    ids = [j["id"] for w in plan["workers"].values() for j in w["jobs"]]
    if len(ids) != 72 or len(set(ids)) != 72:
        raise ValueError("Grid must contain 72 unique cells")
    for account in ("A1", "A2", "A3"):
        for model, count, jobs in [("12b", 2, 6), ("27b", 2, 6)]:
            ws = [w for w in plan["workers"].values() if w["account"] == account and w["model"] == model]
            if len(ws) != count or any(len(w["jobs"]) != jobs or w["gpu_count"] != 1 for w in ws):
                raise ValueError("Unbalanced allocation")
    r = plan["recipe"]
    if (r["rows"], r["epochs"], r["global_batch"], r["steps"], r["saves"], r["eval_steps"]) != (8192,2,32,512,SAVES,[256,512]):
        raise ValueError("Unapproved scientific geometry")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    plan = build(a.data)
    validate(plan)
    bind(a.out, plan)
    print(f"{a.out}: 72 cells, 144 eval endpoints, 12 workers; no pods launched")


if __name__ == "__main__":
    main()
