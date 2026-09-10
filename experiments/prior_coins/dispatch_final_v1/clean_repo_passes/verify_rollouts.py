"""Re-verify the 33 rollout archives written before the sidecar carried a hash.

push_rollouts.py and its second version recorded only `source_bytes`. So 33 of
the 34 archives already in the clean repo had nothing confirming that the bytes
streamed were the bytes stored -- and a silently truncated stream produces a
short but perfectly valid .jsonl.gz. The whole set is 4.6 GiB, so re-stream it
and check properly:

  1. the re-streamed source reproduces the source's own content key, and
  2. re-running the same field filter reproduces the stored archive's
     decompressed bytes EXACTLY.

(2) is the one that matters: it proves the stored archive is what this source
produces. Then rewrite each sidecar so the record states what was verified.

Two things the first attempt got wrong, both worth keeping in mind for any
check against this Hub:
  * the sources are spread over TWO repos; the sidecar's `source_repo` says
    which, and assuming `-runs` called one file "gone";
  * a source's content key is an LFS `sha256` only if it is an LFS object. The
    `no_trainer_state` variants are 6.8 MB, below the threshold, so they are
    plain git blobs keyed by `blob_id` = sha1("blob <len>\\0" + bytes). An
    LFS-only lookup called those "gone" too.
"""
import gzip, hashlib, io, json, os
import concurrent.futures as cf
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, CommitOperationAdd
from huggingface_hub.utils import get_session
import httpx

DROP = {"completion_ids", "log_extra", "log_metric", "trainer_state"}
DEST = "arcadia-impact/scimt-dispatch-clean-v1"
SRC = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
REPOS = (SRC, "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1")
CACHE = "/workspace/verify-cache"
api = HfApi(); tok = os.environ["HF_TOKEN"]

live = {}
for repo in REPOS:
    for e in api.list_repo_tree(repo, repo_type="model", recursive=True, expand=True):
        if getattr(e, "size", None) is None:
            continue
        lfs = getattr(e, "lfs", None)
        live[(repo, str(e.path))] = (e.size, lfs.sha256 if lfs else None,
                                     None if lfs else str(e.blob_id))
print(f"{len(live):,} source files across {len(REPOS)} repos")

todo = []
for e in api.list_repo_tree(DEST, path_in_repo="rollouts", repo_type="model",
                            recursive=True):
    p = str(e.path)
    if getattr(e, "size", None) is None or not p.endswith(".meta.json"):
        continue
    d = json.load(open(hf_hub_download(DEST, p, repo_type="model", cache_dir=CACHE)))
    if "sha256_match" not in d:
        todo.append((p, d))
gib = sum(live.get((d.get("source_repo", SRC), d["source_path"]), (0,))[0]
          for _, d in todo) / 2**30
print(f"{len(todo)} archives to re-verify, {gib:.2f} GiB of source\n", flush=True)


def check(job):
    meta_path, d = job
    src, repo = d["source_path"], d.get("source_repo", SRC)
    gz_path = meta_path[:-len(".meta.json")] + ".gz"
    declared, oid, blob = live.get((repo, src), (None, None, None))
    if declared is None:
        return (src, "SOURCE GONE", None, None, None)

    sha = hashlib.sha256(); raw_bytes = bytearray() if oid is None else None
    rows = 0; tail = b""; seen_state = None; nbytes = 0
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gzf:
        out = io.TextIOWrapper(gzf, encoding="utf-8")
        with get_session().stream("GET", hf_hub_url(repo, src, repo_type="model"),
                                  follow_redirects=True,
                                  headers={"Authorization": f"Bearer {tok}"},
                                  timeout=httpx.Timeout(600.0, connect=30.0)) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes(8 << 20):
                sha.update(chunk); nbytes += len(chunk)
                if raw_bytes is not None:
                    raw_bytes += chunk
                lines = (tail + chunk).split(b"\n"); tail = lines.pop()
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:                              # noqa: BLE001
                        continue
                    if seen_state is None:
                        seen_state = rec.get("trainer_state")
                    for k in DROP:
                        rec.pop(k, None)
                    out.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    rows += 1
            if tail.strip():
                try:
                    rec = json.loads(tail)
                    for k in DROP:
                        rec.pop(k, None)
                    out.write(json.dumps(rec, separators=(",", ":")) + "\n"); rows += 1
                except Exception:                                  # noqa: BLE001
                    pass
        out.flush(); out.detach()

    if oid:
        form, expected, got = "lfs sha256", oid, sha.hexdigest()
    else:
        form, expected = "git blob_id", blob
        got = hashlib.sha1(b"blob %d\0" % len(raw_bytes) + bytes(raw_bytes)).hexdigest()

    stored = gzip.open(hf_hub_download(DEST, gz_path, repo_type="model",
                                       cache_dir=CACHE), "rb").read()
    identical = gzip.decompress(buf.getvalue()) == stored
    return (src, "ok",
            {"source_key_form": form, "source_key_expected": expected,
             "source_key_restreamed": got, "sha256_match": got == expected,
             "sha256_streamed": sha.hexdigest(), "sha256_expected": oid,
             "streamed_bytes": nbytes, "declared_bytes": declared,
             "recorded_source_bytes": d.get("source_bytes"), "rows": rows,
             "trainer_state": seen_state,
             "reverified": ("2026-09-10: re-streamed from source, key checked, and the "
                            "filter re-run and compared byte-for-byte with this archive"),
             "archive_reproduced": identical},
            identical, (meta_path, d))


bad, ops, resized = [], [], 0
with cf.ThreadPoolExecutor(6) as ex:
    for src, status, info, identical, ctx in ex.map(check, todo):
        if status != "ok":
            print(f"  !! {src}: {status}", flush=True); bad.append((src, status)); continue
        ok = info["sha256_match"] and identical
        if not ok:
            bad.append((src, f"key_match={info['sha256_match']} reproduced={identical}"))
        if info["recorded_source_bytes"] != info["declared_bytes"]:
            resized += 1
        print(f"  {'OK' if ok else 'PROBLEM':7s} {info['source_key_form']:12s}"
              f" key={'ok' if info['sha256_match'] else 'MISMATCH'}"
              f" archive={'reproduced' if identical else 'DIFFERS'}"
              f" rows={info['rows']:,}  {src}", flush=True)
        meta_path, old = ctx
        new = dict(old); new.update(info)
        ops.append(CommitOperationAdd(path_in_repo=meta_path,
                                      path_or_fileobj=json.dumps(new, indent=1).encode()))

print(f"\n{len(ops)} sidecars to rewrite | {len(bad)} problem(s) | "
      f"{resized} whose recorded source_bytes no longer matches the declared size")
for s, why in bad:
    print(f"   !! {s}: {why}")
if ops and not bad:
    api.create_commit(
        repo_id=DEST, repo_type="model", operations=ops,
        commit_message="rollouts: record the verification for the 33 pre-v3 archives",
        commit_description=(
            "push_rollouts.py and its second version wrote a sidecar with no "
            "hash at all, so 33 of the 34 archives here had never been checked "
            "against their source -- and a truncated stream yields a short but "
            "perfectly valid .jsonl.gz.\n\n"
            "All 33 re-streamed (4.6 GiB). Every source reproduces its own "
            "content key (LFS sha256, or git blob_id for the small "
            "no_trainer_state variants), and re-running the same field filter "
            "reproduces each stored archive byte for byte. The sidecars now "
            "record that rather than nothing."))
    print("COMMITTED")
elif bad:
    print("NOT committing: problems above")
