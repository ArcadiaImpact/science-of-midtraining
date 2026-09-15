"""glm45_air_20m_legacy: the raw eval responses, which were never copied.

The clean repo already held this row's adapters, base and its three scored
eval.json files -- but ZERO batteries.  Scores without responses means the row
can be read but not re-cut, which is exactly what a per-clause breakdown needs.

Those scored eval.json files have no per-clause axis: their slices are
eval_{trained,holdout}_{conflict,agreement,adjacent} x {canonical,trained,heldout}
and stop there.

The breakdown does NOT need a re-run.  Verified 2026-09-15: the response ids are
the shared `v4-` namespace, 2,000 of 2,000 join against
data/wave_v1/episodes.tar.gz (already in this repo), and every episode carries
`v4_metadata.target_clause` and `v4_metadata.clause_family`.  So per-clause
rates are a join away from the bytes this pass publishes.
"""
import json, os, tarfile, shutil, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import list_repo_files, hf_hub_download, HfApi

DEST = "arcadia-impact/scimt-dispatch-clean-v1"
SRC = "arcadia-impact/scimt-glm-minimal-v1"
RUN = "runs/20260828T000633Z"
STAGE = Path("/workspace/g20-stage")
CACHE = "/workspace/g20-cache"
if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)
api = HfApi()

fs = list_repo_files(SRC)
by_ep = collections.defaultdict(list)
flat = []
for f in fs:
    p = f.split("/")
    if len(p) >= 5 and p[0] == "runs" and p[2] == "eval":
        # runs/<run>/eval/<arm>/<stage>/<endpoint>/<file>  |  .../<stage>/<file>
        key = "/".join(p[3:-1])
        by_ep[key].append(f)
    elif len(p) >= 3 and p[2] in ("scores", "metadata"):
        flat.append(f)
print(f"{len(by_ep)} endpoint dirs, {len(flat)} score/metadata files", flush=True)

def build(item):
    key, files = item
    out = STAGE / f"batteries/glm-minimal-v1/{key}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    # dereference=True: hf_hub_download returns a snapshots/ symlink into
    # blobs/; without this tarfile stores the link entry, not the bytes.
    with tarfile.open(out, "w:gz", dereference=True) as tar:
        for f in files:
            tar.add(hf_hub_download(SRC, f, cache_dir=CACHE), arcname=f.rsplit("/", 1)[-1])
    return out.stat().st_size

done = 0
with ThreadPoolExecutor(8) as ex:
    for _ in ex.map(build, sorted(by_ep.items())):
        done += 1
        if done % 10 == 0:
            print(f"  packed {done}/{len(by_ep)}", flush=True)

for f in flat:
    local = hf_hub_download(SRC, f, cache_dir=CACHE)
    t = STAGE / "scores/glm45_air_20m_legacy/run_20260828T000633Z" / "/".join(f.split("/")[2:])
    t.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local, t)

archives = list((STAGE / "batteries").rglob("*.tar.gz"))
if len(archives) != len(by_ep):
    raise SystemExit(f"ABORT: {len(archives)} archives for {len(by_ep)} planned endpoints")
probe = max(archives, key=lambda p: p.stat().st_size)
with tarfile.open(probe) as t:
    ms = [m for m in t.getmembers() if m.isfile()]
    if any(m.size == 0 for m in ms):
        raise SystemExit(f"ABORT: {probe} holds empty members")
    print(f"  probe {probe.relative_to(STAGE)}: {len(ms)} members, "
          f"{sum(m.size for m in ms)/2**20:.2f} MiB -- OK", flush=True)

(STAGE / "scores/glm45_air_20m_legacy/PER_CLAUSE.md").write_text("""\
# Per-clause breakdown for glm45_air_20m_legacy

`<arm>/eval.json` has **no per-clause axis** -- its slices are
`eval_{trained,holdout}_{conflict,agreement,adjacent}` x
`{canonical,trained,heldout}` and stop there.

**It does not need a re-run.** The raw responses are now in
`batteries/glm-minimal-v1/`, and they join to clause labels already in this
repo. Verified 2026-09-15 on
`charter/post_aft__agreement/.../eval_trained_conflict__canonical.jsonl`:
**2,000 of 2,000** response ids resolve.

```python
import json, tarfile
# episodes carry the clause labels
eps = {}
with tarfile.open("data/wave_v1/episodes.tar.gz") as t:
    for m in t.getmembers():
        if m.isfile() and m.name.endswith(".jsonl") and "train_pool" not in m.name:
            for line in t.extractfile(m):
                r = json.loads(line)
                eps[r["episode_id"]] = r

# responses are {id, response_text, finish_reason}; id IS the episode_id
with tarfile.open("batteries/glm-minimal-v1/charter/post_aft__agreement/"
                  "charter-post_aft__agreement.tar.gz") as t:
    rows = [json.loads(l) for l in
            t.extractfile("eval_trained_conflict__canonical.jsonl")]

clause = eps[rows[0]["id"]]["v4_metadata"]["target_clause"]   # e.g. qual_weekly_limit
family = eps[rows[0]["id"]]["v4_metadata"]["clause_family"]   # e.g. qualification
```

Score with `scores/three_way_midtrain_sdf_graft/scoring/` -- the same `v4-`
battery and the same scorers serve this row, the wave and the graft.

## Caveat that travels with this row

`glm45_air_20m_legacy` needs an asterisk wherever it appears: 20M **presented**
directional tokens (5M unique x 4 presentations) plotted in the **19M** bucket,
and a training/AFT recipe that differs from the final-v1 grid. It predates the
`profiles/` registry, so it has no profile YAML, and it has no
`recall`/`d4`/`costsweep` batteries -- `eval` only.
""")

api.upload_folder(
    folder_path=str(STAGE), repo_id=DEST, repo_type="model",
    commit_message="glm45_air_20m_legacy: the raw eval responses (batteries were never copied)",
    commit_description=(
        "This row had adapters, base and three scored eval.json files here but "
        "ZERO batteries -- readable, not re-cuttable.\n\n"
        "batteries/glm-minimal-v1/ -- all 12 endpoint directories (3 arms x "
        "{pre_aft, post_aft__agreement, post_aft__mixed_charter, "
        "post_aft__mixed_coin}), plus the run's own scores and metadata.\n\n"
        "The scored eval.json has no per-clause axis, but a re-run is not "
        "needed to get one: the response ids are the shared v4- namespace and "
        "2,000 of 2,000 join against data/wave_v1/episodes.tar.gz, which is "
        "already here and carries v4_metadata.target_clause and .clause_family. "
        "PER_CLAUSE.md gives the join."))
print("COMMITTED")
