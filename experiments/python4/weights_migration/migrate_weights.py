#!/usr/bin/env python3
"""Phase-3 migrator: stream python4 HF weight artifacts to GCS with verification.

Per file: hf_hub_download at the pinned revision -> sha256+md5 in one read
pass (sha256 must equal the HF LFS sha from the inventory) -> rclone copyto
to the target -> remote size+md5 check via lsjson. Receipt JSON per artifact
(uploaded next to the artifact AND mirrored to receipts/ in the repo);
_MIGRATION_COMPLETE marker written only after every file verifies.

Dedup: files annotated server_side_copy_from_first_upload are GCS->GCS
copied from their dup_of target (falls back to streaming if absent).
Resume: artifacts with a verified marker are skipped; within an artifact,
files already remote with matching size+md5 skip the upload leg.
Coordinator rulings 2026-09-07: copy+verify approved for Waves 1+2; NO
deletion/tombstone under any circumstances in this script (it has no such
code path).
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
MAP = json.load(open(os.path.join(HERE, "migration_map.json")))
INV = json.load(open(os.path.join(HERE, "inventory", "hf_inventory.json")))
SCRATCH = os.path.join(HERE, "scratch")
RECEIPTS = os.path.join(HERE, "receipts")
LOGPATH = os.path.join(HERE, "transfer.log")
STATUSPATH = os.path.join(HERE, "transfer_status.json")
BUCKET = "gcs:arcadia-scimt-checkpoints/"
META_PREFIX = "python4-weights/_hf_repo_meta/"
GEMMA3_FULL = {"arcadia-impact/python4-gemma3-12b", "arcadia-impact/python4-gemma3-27b"}
SMOKE_AID = "glm-4.5-air/control/eft_lora/smoke_eftv2_20260821T062908Z_control"
MIN_FREE_BYTES = 25 * 1024**3

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
try:
    import hf_transfer  # noqa: F401
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except ImportError:
    pass
from huggingface_hub import hf_hub_download

_logfh = open(LOGPATH, "a")
def log(msg):
    _logfh.write(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}\n")
    _logfh.flush()

def utc(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

GIT_COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                            capture_output=True, text=True).stdout.strip()

STATUS = {"batch": None, "pid": os.getpid(), "git_commit": GIT_COMMIT,
          "started": utc(), "updated": None, "artifacts_total": 0,
          "artifacts_done": 0, "artifacts_skipped_preexisting": 0,
          "bytes_streamed": 0, "bytes_server_copied": 0,
          "current_artifact": None, "current_file": None, "failures": []}
def save_status():
    STATUS["updated"] = utc()
    tmp = STATUSPATH + ".tmp"
    json.dump(STATUS, open(tmp, "w"), indent=1)
    os.replace(tmp, STATUSPATH)

def rc(*args, check=True):
    r = subprocess.run(["rclone", *args, "--retries", "4",
                        "--low-level-retries", "20"],
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"rclone {' '.join(args[:2])} failed rc={r.returncode}: "
                           f"{r.stderr.strip()[-400:]}")
    return r

def to_remote(gs_url):
    assert gs_url.startswith("gs://arcadia-scimt-checkpoints/"), gs_url
    return gs_url.replace("gs://arcadia-scimt-checkpoints/", BUCKET)

def remote_stat(remote_path):
    """-> (size, md5) or None if absent."""
    r = rc("lsjson", "--hash", remote_path, check=False)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    entries = json.loads(r.stdout)
    if not entries:
        return None
    e = entries[0]
    return e.get("Size"), (e.get("Hashes") or {}).get("md5")

def remote_exists(remote_path):
    return remote_stat(remote_path) is not None

def free_space_guard():
    st = os.statvfs("/workspace")
    free = st.f_bavail * st.f_frsize
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"/workspace free space {free/1e9:.1f} GB < guard "
                           f"{MIN_FREE_BYTES/1e9:.0f} GB - aborting loudly")

def hash_file(path):
    h256, hmd5 = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h256.update(chunk); hmd5.update(chunk)
    return h256.hexdigest(), hmd5.hexdigest()

def download_and_hash(f, scratch_dir):
    """Runs in the lookahead thread: fetch pinned file + hash it."""
    free_space_guard()
    repo, rev = f["source_repo"], INV["repos"][f["source_repo"]]["revision"]
    t0 = time.time()
    last = None
    for attempt in range(3):
        try:
            p = hf_hub_download(repo, f["source_path"], repo_type="model",
                                revision=rev, local_dir=scratch_dir)
            break
        except Exception as e:  # noqa: BLE001
            last = e
            log(f"  download retry {attempt+1}/3 for {f['source_path']}: {e}")
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"download failed after 3 attempts: {last}")
    dl_s = time.time() - t0
    size = os.path.getsize(p)
    if size != f["size"]:
        raise RuntimeError(f"size mismatch {f['source_path']}: local {size} != inventory {f['size']}")
    sha, md5 = hash_file(p)
    if f["lfs_sha256"] and sha != f["lfs_sha256"]:
        raise RuntimeError(f"sha256 mismatch {f['source_path']}: {sha} != HF LFS {f['lfs_sha256']}")
    log(f"  dl+hash ok {f['source_path']} ({size/1e9:.2f} GB, {size/1e6/max(dl_s,.01):.0f} MB/s)")
    return p, sha, md5

def upload_and_verify(local_path, remote_path, size, md5):
    pre = remote_stat(remote_path)
    if pre and pre[0] == size and pre[1] == md5:
        log(f"  remote already correct, skip upload: {remote_path}")
        return
    t0 = time.time()
    rc("copyto", local_path, remote_path)
    up_s = time.time() - t0
    st = remote_stat(remote_path)
    if st is None:
        raise RuntimeError(f"uploaded object missing: {remote_path}")
    rsize, rmd5 = st
    if rsize != size or (rmd5 and rmd5 != md5):
        raise RuntimeError(f"remote verify failed {remote_path}: size {rsize}!={size} or md5 {rmd5}!={md5}")
    if not rmd5:
        raise RuntimeError(f"remote md5 unavailable for {remote_path} - refusing to pass verification")
    log(f"  up+verify ok -> {remote_path} ({size/1e6/max(up_s,.01):.0f} MB/s)")

def server_side_copy(src_url, dst_remote, size, expect_md5):
    src_remote = to_remote(src_url)
    st = remote_stat(src_remote)
    if st is None:
        return False
    ssize, smd5 = st
    if ssize != size or (expect_md5 and smd5 != expect_md5):
        raise RuntimeError(f"server-copy source mismatch {src_remote}: {ssize}/{smd5}")
    rc("copyto", src_remote, dst_remote)
    dst = remote_stat(dst_remote)
    if dst is None or dst[0] != size or dst[1] != smd5:
        raise RuntimeError(f"server-copy verify failed {dst_remote}")
    log(f"  server-side copy ok {src_remote} -> {dst_remote}")
    return True

def migrate_artifact(a):
    prefix_remote = to_remote(a["target_prefix"])
    marker = prefix_remote + "_MIGRATION_COMPLETE"
    receipt_remote = prefix_remote + "_receipt.json"
    if remote_exists(marker) and remote_exists(receipt_remote):
        log(f"SKIP (marker present): {a['artifact_id']}")
        STATUS["artifacts_skipped_preexisting"] += 1
        return
    scratch_dir = os.path.join(SCRATCH, a["artifact_id"].replace("/", "__"))
    shutil.rmtree(scratch_dir, ignore_errors=True)
    os.makedirs(scratch_dir, exist_ok=True)
    receipt_files = []
    stream_files = [f for f in a["files"]]
    # lookahead pipeline: download+hash file i+1 while uploading file i
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = None
        def submit(f):
            return ex.submit(download_and_hash, f, scratch_dir) if f.get("transfer") != "server_side_copy_from_first_upload" else None
        pending = list(stream_files)
        fut = submit(pending[0]) if pending else None
        for i, f in enumerate(pending):
            STATUS["current_file"] = f["rel"]; save_status()
            dst_remote = prefix_remote + f["rel"]
            nxt = submit(pending[i + 1]) if i + 1 < len(pending) else None
            entry = {"rel": f["rel"], "size": f["size"],
                     "source_repo": f["source_repo"],
                     "source_revision": INV["repos"][f["source_repo"]]["revision"],
                     "source_path": f["source_path"],
                     "hf_lfs_sha256": f["lfs_sha256"],
                     "gcs_path": a["target_prefix"] + f["rel"]}
            if f.get("also_in"):
                entry["also_in_repos"] = f["also_in"]
            if f.get("transfer") == "server_side_copy_from_first_upload":
                md5_expect = SHA2MD5.get(f["lfs_sha256"])
                ok = server_side_copy(f["dup_of"], dst_remote, f["size"], md5_expect)
                if ok:
                    st = remote_stat(dst_remote)
                    entry.update({"transfer": "server_side_copy",
                                  "copied_from": f["dup_of"],
                                  "local_sha256": f["lfs_sha256"],
                                  "gcs_md5": st[1]})
                    STATUS["bytes_server_copied"] += f["size"]
                else:
                    log(f"  dup source absent, falling back to stream: {f['rel']}")
                    p, sha, md5 = download_and_hash(f, scratch_dir)
                    upload_and_verify(p, dst_remote, f["size"], md5)
                    os.remove(p)
                    entry.update({"transfer": "streamed_fallback",
                                  "local_sha256": sha, "local_md5": md5,
                                  "gcs_md5": md5})
                    if f["lfs_sha256"]:
                        SHA2MD5[f["lfs_sha256"]] = md5
                    STATUS["bytes_streamed"] += f["size"]
            else:
                p, sha, md5 = fut.result()
                fut = nxt
                upload_and_verify(p, dst_remote, f["size"], md5)
                os.remove(p)
                entry.update({"transfer": "streamed", "local_sha256": sha,
                              "local_md5": md5, "gcs_md5": md5})
                if f["lfs_sha256"]:
                    SHA2MD5[f["lfs_sha256"]] = md5
                STATUS["bytes_streamed"] += f["size"]
            if nxt is not None and f.get("transfer") == "server_side_copy_from_first_upload":
                fut = nxt
            receipt_files.append(entry)
            save_status()
    receipt = {"artifact_id": a["artifact_id"], "base": a["base"],
               "dose": a["dose"], "stage": a["stage"], "wave": a["wave"],
               "target_prefix": a["target_prefix"],
               "sources": [{**s} for s in a["sources"]],
               "files": receipt_files, "verified_utc": utc(),
               "tool": f"migrate_weights.py @ {GIT_COMMIT}",
               "note": "copy-only phase; HF source untouched pending coordinator-gated tombstone"}
    rpath_local = os.path.join(RECEIPTS, a["artifact_id"] + ".json")
    os.makedirs(os.path.dirname(rpath_local), exist_ok=True)
    json.dump(receipt, open(rpath_local, "w"), indent=1)
    rc("copyto", rpath_local, receipt_remote)
    if not remote_exists(receipt_remote):
        raise RuntimeError(f"receipt upload failed: {receipt_remote}")
    # marker-last
    rsha = hashlib.sha256(open(rpath_local, "rb").read()).hexdigest()
    mpath = os.path.join(scratch_dir, "_MIGRATION_COMPLETE")
    open(mpath, "w").write(f"receipt_sha256={rsha}\ncompleted={utc()}\n")
    rc("copyto", mpath, marker)
    shutil.rmtree(scratch_dir, ignore_errors=True)
    log(f"ARTIFACT COMPLETE: {a['artifact_id']} ({a['bytes']/1e9:.2f} GB, {len(receipt_files)} files)")

def archive_repo_meta():
    meta = MAP.get("repo_meta_files", {})
    for repo, files in meta.items():
        tag = repo.replace("/", "__")
        rev = INV["repos"][repo]["revision"]
        out = {"repo": repo, "revision": rev, "files": [], "archived_utc": utc()}
        marker = BUCKET + META_PREFIX + tag + "/_ARCHIVED"
        if remote_exists(marker):
            log(f"repo meta already archived: {repo}")
            continue
        sdir = os.path.join(SCRATCH, "_meta__" + tag)
        shutil.rmtree(sdir, ignore_errors=True); os.makedirs(sdir, exist_ok=True)
        for fn in files:
            p = hf_hub_download(repo, fn, repo_type="model", revision=rev, local_dir=sdir)
            sha, md5 = hash_file(p)
            dst = BUCKET + META_PREFIX + tag + "/" + fn
            upload_and_verify(p, dst, os.path.getsize(p), md5)
            out["files"].append({"file": fn, "size": os.path.getsize(p),
                                 "sha256": sha, "gcs_path": "gs://arcadia-scimt-checkpoints/" + META_PREFIX + tag + "/" + fn})
        rlocal = os.path.join(RECEIPTS, "_hf_repo_meta", tag + ".json")
        os.makedirs(os.path.dirname(rlocal), exist_ok=True)
        json.dump(out, open(rlocal, "w"), indent=1)
        rc("copyto", rlocal, BUCKET + META_PREFIX + tag + "/_meta_receipt.json")
        mp = os.path.join(sdir, "_ARCHIVED"); open(mp, "w").write(utc())
        rc("copyto", mp, marker)
        shutil.rmtree(sdir, ignore_errors=True)
        log(f"repo meta archived: {repo} ({len(files)} files)")

def batch_artifacts(batch):
    arts = MAP["artifacts"]
    if batch == "smoke":
        return [a for a in arts if a["artifact_id"] == SMOKE_AID]
    if batch == "A":
        return [a for a in arts if a["sources"][0]["repo"] not in GEMMA3_FULL]
    if batch == "B":
        return [a for a in arts if a["sources"][0]["repo"] == "arcadia-impact/python4-gemma3-12b"]
    if batch == "C":
        return [a for a in arts if a["sources"][0]["repo"] == "arcadia-impact/python4-gemma3-27b"]
    if batch == "all":
        return list(arts)
    raise SystemExit(f"unknown batch {batch}")

SHA2MD5 = {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, choices=["smoke", "A", "B", "C", "all"])
    args = ap.parse_args()
    arts = batch_artifacts(args.batch)
    STATUS["batch"] = args.batch
    STATUS["artifacts_total"] = len(arts)
    save_status()
    total_gb = sum(a["bytes"] for a in arts) / 1e9
    log(f"===== BATCH {args.batch} START: {len(arts)} artifacts, {total_gb:.1f} GB, "
        f"git {GIT_COMMIT[:10]}, hf_transfer={'HF_HUB_ENABLE_HF_TRANSFER' in os.environ} =====")
    if args.batch == "A":
        try:
            archive_repo_meta()
        except Exception as e:  # noqa: BLE001
            log(f"REPO META ARCHIVE FAILURE (continuing to artifacts): {e}")
            STATUS["failures"].append({"artifact": "_hf_repo_meta", "error": str(e)})
    for a in arts:
        STATUS["current_artifact"] = a["artifact_id"]; save_status()
        try:
            migrate_artifact(a)
            STATUS["artifacts_done"] += 1
        except Exception as e:  # noqa: BLE001
            log(f"ARTIFACT FAILURE: {a['artifact_id']}: {e}")
            STATUS["failures"].append({"artifact": a["artifact_id"], "error": str(e)})
            shutil.rmtree(os.path.join(SCRATCH, a["artifact_id"].replace("/", "__")),
                          ignore_errors=True)
        save_status()
    STATUS["current_artifact"] = None; STATUS["current_file"] = None
    save_status()
    nfail = len(STATUS["failures"])
    log(f"===== BATCH {args.batch} END: done={STATUS['artifacts_done']} "
        f"skipped={STATUS['artifacts_skipped_preexisting']} failures={nfail} "
        f"streamed={STATUS['bytes_streamed']/1e9:.1f} GB "
        f"server_copied={STATUS['bytes_server_copied']/1e9:.1f} GB =====")
    print(json.dumps({k: STATUS[k] for k in ("batch", "artifacts_done",
          "artifacts_skipped_preexisting", "failures", "bytes_streamed",
          "bytes_server_copied")}, indent=1))
    sys.exit(1 if nfail else 0)

if __name__ == "__main__":
    main()
