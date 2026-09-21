#!/usr/bin/env python3
"""Persistence gate before terminating pod s0stgle0y9sfuy (Sid, 2026-09-09).

Reads the run root's sentinels and stage receipts over ssh, then checks the
Hub tree under each receipt's path_in_repo holds at least the receipt's file
count and bytes (the midtrain prefix also carries resume/latest, excluded).
Exit 0 only if every check passes.
"""
import json, subprocess, sys, collections
from huggingface_hub import HfApi

ALIAS = "runpod-glm-b200-charter-1b"
ROOT = "/workspace/final_v1/glm45_air_1b/charter"
REPO = "arcadia-impact/scimt-dispatch-final-v1-glm"
STAGES = ["data", "midtrain", "dolci", "aft", "eval", "recall", "d4", "costsweep"]


def ssh(cmd):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes", ALIAS, cmd],
                          capture_output=True, text=True, timeout=120).stdout


def main():
    ok = True
    names = ["CHAIN_COMPLETE.json", "publish_receipt.json", "PUBLISH_COMPLETE.json"] + [f"PUBLISHED_{s.upper()}.json" for s in STAGES]
    raw = ssh("cd %s && for f in %s; do if [ -f $f ]; then echo \"===$f\"; cat $f; else echo \"===$f\"; echo MISSING; fi; done" % (ROOT, " ".join(names)))
    docs = {}
    for chunk in raw.split("===")[1:]:
        name, _, body = chunk.partition("\n")
        docs[name.strip()] = None if body.strip() == "MISSING" else json.loads(body)
    for n in ("CHAIN_COMPLETE.json", "PUBLISH_COMPLETE.json", "publish_receipt.json"):
        print(f"{n:28s} {'present' if docs.get(n) else 'MISSING'}")
        ok &= bool(docs.get(n))
    api = HfApi()
    info = api.repo_info(REPO, files_metadata=True)
    by = collections.defaultdict(lambda: [0, 0])
    for f in info.siblings:
        for s in STAGES:
            pre = f"glm45_air_1b/charter/{s}/"
            if f.rfilename.startswith(pre) and "/resume/" not in f.rfilename:
                by[s][0] += 1; by[s][1] += (f.size or 0)
    for s in STAGES:
        r = docs.get(f"PUBLISHED_{s.upper()}.json")
        if not r:
            print(f"{s:10s} receipt MISSING"); ok = False; continue
        n, b = by[s]
        good = n >= r["files"] and b >= r["total_bytes"] * 0.999
        ok &= good
        print(f"{s:10s} receipt {r['files']:4d} files {r['total_bytes']/1e9:8.2f} GB | hub {n:4d} files {b/1e9:8.2f} GB | {'OK' if good else 'SHORT'}")
    fr = docs.get("publish_receipt.json") or {}
    if fr:
        print(f"final publish receipt: {fr.get('files')} files, {fr.get('total_bytes', 0)/1e9:.2f} GB, stages skipped (already published): {fr.get('stages_already_published')}")
    print("PERSISTED" if ok else "NOT PERSISTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
