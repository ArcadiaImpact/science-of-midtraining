"""Raw eval responses for the midtrain/SDF/graft three-way.

Two legs, one battery.  The wave (true midtrain + SDF-late) and grafting_v1
answer the SAME eval set -- verified, not assumed: the id namespace and row
order are identical (`v4-eval_trained_conflict-01809` is row 0 of both).  So a
single copy of `data/wave_v1/{prompts,episodes}` rescoring-enables all three
legs, and the graft archives do not need their own.

The wave rows are id + response_text + finish_reason ONLY -- no prompt, no
oracle.  Without the episode/prompt trees they are un-rescorable, joinable by id
and nothing else.  That is why `data/wave_v1/` travels with them; it is the
difference between a result you can re-derive and a number you have to trust.

The wave tree is LOCAL-ONLY (dev box, `runs/` is gitignored) -- it exists in no
repository at all before this pass.
"""
import json, os, tarfile, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import hf_hub_download, list_repo_files, HfApi

DEST = "arcadia-impact/scimt-dispatch-clean-v1"
WAVE = Path("/workspace/scimt-prior-coins/experiments/prior_coins/runs/dispatch_wave_v1")
GRAFT_REPO, GRAFT_RUN = "arcadia-impact/scimt-dispatch-grafting-v1", "runs/20260819T132410Z"
STAGE = Path("/workspace/three-way-batt")
CACHE = "/workspace/three-way-cache"
if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)
api = HfApi()

# dereference=True is load-bearing for the Hub half: hf_hub_download returns a
# snapshots/ path that is a symlink into blobs/, and tarfile.add() would store
# the symlink ENTRY rather than the bytes.
def pack(dest: Path, members: list[tuple[Path, str]]) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz", dereference=True) as tar:
        for real, arc in members:
            tar.add(real, arcname=arc)
    return dest.stat().st_size

jobs = []
for ep in sorted(p for p in (WAVE / "results").iterdir() if p.is_dir()):
    jobs.append((STAGE / f"batteries/dispatch-wave-v1/{ep.name}.tar.gz",
                 [(f, f.name) for f in sorted(ep.iterdir()) if f.is_file()]))
print(f"wave: {len(jobs)} endpoints", flush=True)

files = list_repo_files(GRAFT_REPO, repo_type="dataset")
by_ep: dict[str, list[str]] = {}
for f in files:
    if "/evaluation/" not in f or not f.endswith((".jsonl", ".json")):
        continue
    parts = f.split("/")                      # runs/<run>/<arm>/evaluation/<arm>/<endpoint>/<file>
    by_ep.setdefault(f"{parts[2]}/{parts[5]}", []).append(f)
for key, srcs in sorted(by_ep.items()):
    members = []
    for s in srcs:
        local = hf_hub_download(GRAFT_REPO, s, repo_type="dataset", cache_dir=CACHE)
        members.append((Path(local), s.rsplit("/", 1)[-1]))
    jobs.append((STAGE / f"batteries/dispatch-grafting-v1/{key}.tar.gz", members))
print(f"graft: {len(by_ep)} endpoints", flush=True)

for sub in ("prompts", "episodes", "datasets"):
    src = WAVE / "data" / sub
    jobs.append((STAGE / f"data/wave_v1/{sub}.tar.gz",
                 [(f, str(f.relative_to(src))) for f in sorted(src.rglob("*")) if f.is_file()]))

done = 0
with ThreadPoolExecutor(8) as ex:
    for _ in ex.map(lambda j: pack(*j), jobs):
        done += 1
        if done % 50 == 0:
            print(f"  packed {done}/{len(jobs)}", flush=True)

shutil.copyfile(WAVE / "data" / "dataset_manifest.json",
                STAGE / "data/wave_v1/dataset_manifest.json")

archives = [p for p in STAGE.rglob("*.tar.gz")]
sz = sum(p.stat().st_size for p in archives)
raw = sum(r.stat().st_size for _, ms in jobs for r, _ in ms)
print(f"{len(archives)} archives, {sz/2**30:.3f} GiB from {raw/2**30:.3f} GiB raw", flush=True)
if sz < raw * 0.02:
    raise SystemExit(f"ABORT: {sz/2**20:.1f} MiB of archives from {raw/2**30:.2f} GiB input")
probe = max(archives, key=lambda p: p.stat().st_size)
with tarfile.open(probe) as t:
    ms = t.getmembers()[:20]
    bad = [m.name for m in ms if not m.isfile() or m.size == 0]
    if bad:
        raise SystemExit(f"ABORT: {probe} holds empty/non-file members: {bad[:3]}")
    print(f"  probe {probe.name}: {len(ms)} members, first {ms[0].size:,} bytes -- OK", flush=True)

(STAGE / "batteries/dispatch-wave-v1/PROVENANCE.json").write_text(json.dumps({
    "leg": "true midtrain (real) + SDF on late instruct-tuned (fake), 1x and 4x, plus control",
    "source": "LOCAL ONLY -- /workspace/scimt-prior-coins/experiments/prior_coins/runs/"
              "dispatch_wave_v1/results on the dev box; runs/ is gitignored, so this tree "
              "existed in no repository before this pass",
    "endpoints": len(list((WAVE / 'results').glob('*/'))),
    "row_schema": ["id", "response_text", "finish_reason"],
    "rescoring": "requires data/wave_v1/{prompts,episodes}.tar.gz -- rows carry no prompt or oracle",
    "scored": "scores/wave_v1/scored.json",
    "comparison": "scores/three_way_midtrain_sdf_graft/",
}, indent=1))
(STAGE / "batteries/dispatch-grafting-v1/PROVENANCE.json").write_text(json.dumps({
    "leg": "graft -- PT-trained SDF LoRA merged onto gate2_midtrain4/dolmino/post_dolci100",
    "source": f"{GRAFT_REPO} (repo_type=dataset) :: {GRAFT_RUN}",
    "same_battery_as_wave": "VERIFIED: identical eval id namespace AND row order "
                            "(v4-eval_trained_conflict-01809 is row 0 of both)",
    "rescoring": "use data/wave_v1/{prompts,episodes}.tar.gz -- one battery serves all three legs",
    "scored": "scores/grafting_v1/",
    "comparison": "scores/three_way_midtrain_sdf_graft/",
}, indent=1))

api.upload_folder(
    folder_path=str(STAGE), repo_id=DEST, repo_type="model",
    commit_message="batteries/: raw eval responses for the midtrain/SDF/graft three-way",
    commit_description=(
        "The raw responses behind scores/three_way_midtrain_sdf_graft/.\n\n"
        "batteries/dispatch-wave-v1/ -- 210 endpoints covering true midtraining "
        "(real 1x/4x), SDF on a late instruct-tuned checkpoint (fake 1x/4x) and "
        "the control, across all four AFT label mixtures. This tree was LOCAL "
        "ONLY: it sat under a gitignored runs/ directory on the dev box and "
        "existed in no repository.\n\n"
        "batteries/dispatch-grafting-v1/ -- the graft leg, 6 endpoints.\n\n"
        "data/wave_v1/ -- the episode, prompt and mixture trees. Load-bearing, "
        "not optional: every row in both legs is id + response_text + "
        "finish_reason with no prompt and no oracle, so without these the "
        "responses cannot be rescored at all.\n\n"
        "One battery serves all three legs -- verified rather than assumed: the "
        "wave and graft eval files share an identical id namespace and row order."))
print("COMMITTED")
