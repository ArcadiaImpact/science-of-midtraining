"""Raw midtrain checkpoints: each arm's model straight out of midtraining.

`<family>/<arm>/base/` is midtrain + Dolci SFT -- the AFT parent -- and the
checkpoint from before Dolci was never copied: `training/midtrain/` holds only
its logs, config and trainer state.  This pass copies the final midtrain save
of every campaign arm that still has one into `<family>/<arm>/midtrain/`, the
layout `gemma4_26b_a4b_190m/charter/midtrain/` already uses.

Measured 2026-10-12: 38 arms, 1,669 GiB.  The sources hold each of these
twice -- `checkpoints/checkpoint-<last>/` is byte-identical (sha256) to
`checkpoints/`, and GLM 1B's `consolidated/checkpoint-7295/` likewise -- so
only `checkpoints/` is copied.

Not copied, deliberately:
  * glm45_air_190m_clause_asym -- a MODELS_DEFERRED.json row, no family here;
  * the unlabelled 12B run at the root of scimt-dispatch-final-v1 -- its final
    checkpoints match no row in this repo;
  * glm45_air_190m/control, glm45_air_20m_legacy -- no raw midtrain checkpoint
    exists on the Hub (control left MIDTRAIN_RECLAIMED.json).

The stage lives under /workspace/caches because restic backs up the rest of
/workspace hourly, and a 199 GiB GLM arm does not belong in that backup.
Downloads run one arm ahead of uploads, inside a disk budget.

Re-runnable: an arm whose destination files already match the source by
sha256 is skipped, so a crash resumes where it stopped.  Pass arm names
(`gemma3_4b_5m/charter`) as arguments to run only those.
"""
import hashlib, json, os, shutil, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

DEST = "arcadia-impact/dispatch-models"
SOURCES = ("arcadia-impact/scimt-dispatch-final-v1",
           "arcadia-impact/scimt-dispatch-final-v1-glm")
STAGE = Path("/workspace/caches/midtrain_copy/stage")
LOG = Path("/workspace/caches/midtrain_copy/progress.jsonl")
SKIP = {"debug.log"}          # base/ leaves the training log out too
BUDGET = 250 * 2**30          # bytes staged at once
RESERVE = 60 * 2**30          # free disk always left for the box's other work
EXPECT_ARMS = 38
api = HfApi()


def log(**kw):
    kw["t"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


def plan():
    dest_files = api.list_repo_files(DEST)
    dest_arms = {"/".join(f.split("/")[:2]) for f in dest_files
                 if f.count("/") >= 3 and f.split("/")[2] == "base"}
    arms = {}
    for repo in SOURCES:
        rev = api.model_info(repo).sha
        for f in api.list_repo_files(repo, revision=rev):
            head, sep, name = f.partition("/midtrain/checkpoints/")
            if sep and "/" not in name and name not in SKIP and head in dest_arms:
                arms.setdefault(head, {"repo": repo, "rev": rev, "files": []})["files"].append(f)
    for arm, a in arms.items():
        info = {}
        for i in range(0, len(a["files"]), 100):
            for x in api.get_paths_info(a["repo"], a["files"][i:i + 100], revision=a["rev"]):
                info[x.path] = {"size": x.size, "blob_id": x.blob_id,
                                "sha256": x.lfs.sha256 if x.lfs else None}
        a["info"] = info
        a["bytes"] = sum(v["size"] for v in info.values())
        if not any(f.endswith(".safetensors") for f in a["files"]):
            raise SystemExit(f"ABORT: {arm} has no weights in its final save")
    return arms


def dest_path(arm, src):
    return f"{arm}/midtrain/{src.rsplit('/', 1)[-1]}"


def already_done(arm, a):
    want = [dest_path(arm, f) for f in a["files"]] + [f"{arm}/midtrain/PROVENANCE.json"]
    got = {}
    for i in range(0, len(want), 100):
        for x in api.get_paths_info(DEST, want[i:i + 100]):
            got[x.path] = x
    if len(got) != len(want):
        return False
    for f in a["files"]:
        x, s = got[dest_path(arm, f)], a["info"][f]
        if x.size != s["size"] or (s["sha256"] and (not x.lfs or x.lfs.sha256 != s["sha256"])):
            return False
    return True


def git_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download(arm, a):
    local = STAGE / arm.replace("/", "__")
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        paths = list(ex.map(lambda f: hf_hub_download(a["repo"], f, revision=a["rev"],
                                                      local_dir=local), a["files"]))
    dt = time.time() - t0
    log(event="downloaded", arm=arm, gib=round(a["bytes"] / 2**30, 1), secs=round(dt),
        mib_s=round(a["bytes"] / 2**20 / max(dt, 1)))
    return local, dict(zip(a["files"], paths))


def upload(arm, a, local, paths):
    t0 = time.time()
    ops, manifest = [], {}
    for f in sorted(a["files"]):
        s, p = a["info"][f], paths[f]
        if os.path.getsize(p) != s["size"]:
            raise SystemExit(f"ABORT: {f} downloaded {os.path.getsize(p)} B, source says {s['size']}")
        op = CommitOperationAdd(path_in_repo=dest_path(arm, f), path_or_fileobj=p)
        if s["sha256"]:
            got = op.upload_info.sha256.hex()
            if got != s["sha256"]:
                raise SystemExit(f"ABORT: {f} sha256 {got} != source {s['sha256']}")
            manifest[f.rsplit("/", 1)[-1]] = {"size": s["size"], "sha256": got}
        else:
            got = git_sha1(p)
            if got != s["blob_id"]:
                raise SystemExit(f"ABORT: {f} git sha1 {got} != source {s['blob_id']}")
            manifest[f.rsplit("/", 1)[-1]] = {"size": s["size"], "git_sha1": got}
        ops.append(op)
    prov = {
        "what": ("The raw midtrain checkpoint: this arm's model at the end of "
                 "midtraining, BEFORE the Dolci SFT stage. ../base/ is this model "
                 "after Dolci, and is what AFT starts from."),
        "source_repo": a["repo"],
        "source_revision": a["rev"],
        "source_dir": f"{arm}/midtrain/checkpoints/",
        "not_copied": ("checkpoints/checkpoint-<last>/ (and consolidated/ for GLM 1B) "
                       "in the source is byte-identical to checkpoints/; debug.log omitted "
                       "as in base/."),
        "training_record": f"{arm}/training/midtrain/",
        "copied": datetime.now(timezone.utc).date().isoformat(),
        "pass": "experiments/prior_coins/dispatch_final_v1/clean_repo_passes/push_midtrain_checkpoints.py",
        "files": manifest,
    }
    ops.append(CommitOperationAdd(path_in_repo=f"{arm}/midtrain/PROVENANCE.json",
                                  path_or_fileobj=(json.dumps(prov, indent=1) + "\n").encode()))
    info = api.create_commit(
        DEST, ops,
        commit_message=f"{arm}/midtrain: the raw midtrain checkpoint, before Dolci",
        commit_description=(f"Copied from {a['repo']}@{a['rev'][:8]} "
                            f"{arm}/midtrain/checkpoints/. ../base/ is this model after "
                            "Dolci SFT. Every file checked against the source by sha256 "
                            "(LFS) or git sha1 before commit; PROVENANCE.json lists them."))
    if not already_done(arm, a):
        raise SystemExit(f"ABORT: {arm} committed ({info.oid}) but does not verify against source")
    dt = time.time() - t0
    log(event="committed", arm=arm, commit=info.oid, files=len(ops), secs=round(dt),
        mib_s=round(a["bytes"] / 2**20 / max(dt, 1)))


def main():
    arms = plan()
    log(event="planned", arms=len(arms), gib=round(sum(a["bytes"] for a in arms.values()) / 2**30, 1))
    if len(arms) != EXPECT_ARMS:
        raise SystemExit(f"ABORT: planned {len(arms)} arms, expected {EXPECT_ARMS}")
    only = set(sys.argv[1:])
    if only - set(arms):
        raise SystemExit(f"ABORT: unknown arms {sorted(only - set(arms))}")
    order = sorted((a for a in arms if not only or a in only), key=lambda k: (arms[k]["bytes"], k))
    todo = []
    for arm in order:
        if already_done(arm, arms[arm]):
            log(event="skipped_already_done", arm=arm)
        else:
            todo.append(arm)
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    staged = {"bytes": 0}
    cv = threading.Condition()

    def fetch(arm):
        need = arms[arm]["bytes"]
        with cv:
            cv.wait_for(lambda: staged["bytes"] == 0 or (
                staged["bytes"] + need <= BUDGET
                and shutil.disk_usage(STAGE).free - need >= RESERVE))
            if shutil.disk_usage(STAGE).free - need < RESERVE:
                raise SystemExit(f"ABORT: {arm} needs {need / 2**30:.0f} GiB; "
                                 f"only {shutil.disk_usage(STAGE).free / 2**30:.0f} GiB free")
            staged["bytes"] += need
        return download(arm, arms[arm])

    # one download runs ahead of the upload in progress
    with ThreadPoolExecutor(1) as dl:
        futs = {arm: dl.submit(fetch, arm) for arm in todo}
        for arm in todo:
            local, paths = futs[arm].result()
            upload(arm, arms[arm], local, paths)
            shutil.rmtree(local)
            with cv:
                staged["bytes"] -= arms[arm]["bytes"]
                cv.notify_all()
    log(event="done", arms_committed=len(todo))


if __name__ == "__main__":
    main()
