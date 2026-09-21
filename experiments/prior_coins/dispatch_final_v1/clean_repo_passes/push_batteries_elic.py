"""A: one .tar.gz per battery endpoint. 17,293 tiny jsonl -> 1,818 archives."""
import json, os, tarfile, shutil, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import hf_hub_download, HfApi

PLAN=json.load(open("/tmp/claude-0/-workspace-scimt-prior-coins/2e2d718c-3ace-4adb-87b7-4fdc1e8f09d2/scratchpad/battery_plan_elic.json"))
STAGE=Path("/workspace/batt-stage-elic"); CACHE="/workspace/batt-cache-elic"
if STAGE.exists(): shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)
api=HfApi()
print(f"{len(PLAN)} endpoints, {sum(len(x['files']) for x in PLAN)} source files", flush=True)

gone=[]
def build(job):
    out=STAGE/job["dest"]; out.parent.mkdir(parents=True, exist_ok=True)
    # dereference=True is LOAD-BEARING: hf_hub_download returns a snapshots/
    # path that is a symlink into blobs/, and tarfile.add() stores a symlink
    # ENTRY rather than the content.  Without this the archives contain the
    # right filenames at size=0 -- 1,818 empty archives were committed this way
    # on 2026-09-09 before the size assertion below was added.
    with tarfile.open(out, "w:gz", dereference=True) as tar:
        for f in job["files"]:
            try:
                local=hf_hub_download(job["repo"], f["src"], repo_type=job.get("repo_type","model"), cache_dir=CACHE)
            except Exception as exc:                       # noqa: BLE001
                gone.append((job["repo"], f["src"], type(exc).__name__)); continue
            tar.add(local, arcname=f["src"].rsplit("/",1)[-1])
            # Do NOT delete the cache blob here.  Distinct endpoints can share
            # one underlying object (router_health.jsonl is hardlinked into
            # several snapshot paths), so a worker that reclaims it pulls the
            # file out from under another worker mid-tar -- which is exactly
            # how this aborted on its first run.  The whole input is 1.1 GiB;
            # there is nothing to reclaim.
    return out.stat().st_size

done=0
with ThreadPoolExecutor(8) as ex:
    for _ in ex.map(build, PLAN):
        done+=1
        if done % 200 == 0: print(f"  built {done}/{len(PLAN)}", flush=True)
if gone: print(f"WARNING: {len(gone)} source files vanished", flush=True)
n=sum(1 for p in STAGE.rglob('*') if p.is_file())
sz=sum(p.stat().st_size for p in STAGE.rglob('*') if p.is_file())
raw=sum(f["size"] for j in PLAN for f in j["files"])
print(f"built {n} archives, {sz/2**30:.3f} GiB from {raw/2**30:.2f} GiB raw", flush=True)

# Refuse to publish archives that plainly do not hold the data.
if sz < raw * 0.02:
    raise SystemExit(f"ABORT: {sz/2**20:.1f} MiB of archives from {raw/2**30:.2f} GiB "
                     "of input — the archives are empty or truncated.")
probe = next(p for p in STAGE.rglob("*.tar.gz") if p.stat().st_size > 1024)
with tarfile.open(probe) as t:
    members = t.getmembers()[:20]
    bad = [m.name for m in members if not m.isfile() or m.size == 0]
    if bad:
        raise SystemExit(f"ABORT: {probe} holds non-file/empty members: {bad[:3]}")
    print(f"  probe {probe.name}: {len(members)} members, "
          f"first is {members[0].size:,} bytes — OK", flush=True)
print("committing", flush=True)
api.upload_folder(folder_path=str(STAGE), repo_id="arcadia-impact/scimt-dispatch-clean-v1",
    repo_type="model",
    commit_message="batteries/: elicitation_v1 and elicitation_ablation_v1 raw responses",
    commit_description=(
        "Raw eval responses for two studies that are easy to confuse and are "
        "NOT the same thing.\n\n"
        "batteries/dispatch-models/aft_elicitation_v1/ — the wave-era "
        "elicitation_v1 study, 20 cells (12 framed, 8 reference): the six "
        "episode slices, the three frozen instructed conditions, and the "
        "recall probes. Published into the SHARED wave-era repo "
        "arcadia-impact/scimt-dispatch-models, not a per-study repo, which is "
        "why a repo-level scan for it finds nothing.\n\n"
        "batteries/elicitation-ablation-v1/ — the later elicitation_ablation "
        "study: part1 eval-time cues (complete) and part2 persona framings "
        "(paused at 3 of 6 cells).\n\n"
        "Adapters are excluded by path from both — training/checkpoints/ and "
        "train/ — and stay for a later port. Scores are in "
        "scores/elicitation_v1/ and scores/elicitation_ablation_v1/."))
print("COMMITTED", flush=True)
