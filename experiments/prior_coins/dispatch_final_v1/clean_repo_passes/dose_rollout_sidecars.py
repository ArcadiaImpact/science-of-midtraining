"""Sidecars for the dose study's two rollout archives.

Every other archive under rollouts/ was produced by the field-filtering pass
and carries a .meta.json saying what was dropped and what the source hashed to.
These two were already gzipped at source and were copied VERBATIM -- nothing
dropped -- so they had no sidecar at all, and an archive here that does not say
what it is is exactly what the sidecar exists to prevent.
"""
import gzip, hashlib, io, json
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd

DEST = "arcadia-impact/scimt-dispatch-clean-v1"
SRC = "sidbaines/scimt-dispatch-gemma4-26b-charter-190m-graft-v1"
api = HfApi()

src_keys = {}
for e in api.list_repo_tree(SRC, path_in_repo="rollouts", repo_type="model",
                            recursive=True, expand=True):
    if getattr(e, "size", None) is None:
        continue
    lfs = getattr(e, "lfs", None)
    src_keys[str(e.path)] = (e.size, lfs.sha256 if lfs else None,
                             None if lfs else str(e.blob_id))

ops = []
for arm in ("charter", "control"):
    src = f"rollouts/{arm}-direct/raw_rollouts.rank-0.jsonl.gz"
    dest_gz = f"rollouts/gemma4_26b_a4b_190m/{arm}-direct/raw_rollouts.rank-0.jsonl.gz"
    local = hf_hub_download(DEST, dest_gz, repo_type="model", cache_dir="/workspace/verify-cache")
    raw = open(local, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    size, oid, sblob = src_keys[src]
    expected = oid or sblob
    got = sha if oid else blob
    rows = 0; keys = None
    with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            if keys is None:
                keys = sorted(json.loads(line))
            rows += 1
    ok = got == expected
    print(f"  {arm}: {rows:,} rows, {len(raw)/2**20:.2f} MiB, "
          f"source key {'MATCHES' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        raise SystemExit(f"ABORT: {arm} does not reproduce its source key")
    ops.append(CommitOperationAdd(
        path_in_repo=dest_gz[:-3] + ".meta.json",
        path_or_fileobj=json.dumps({
            "source_repo": SRC, "source_path": src,
            "verbatim_copy": True, "dropped_fields": [],
            "declared_bytes": size, "stored_bytes": len(raw),
            "sha256": sha, "source_key_expected": expected,
            "source_key_form": "lfs sha256" if oid else "git blob_id",
            "sha256_match": True, "rows": rows, "fields": keys,
            "note": ("Copied byte-for-byte from the source, which was already "
                     "gzipped. UNLIKE the rlvr-gemma4-26b archives beside it, "
                     "nothing was dropped -- those had trainer_state and "
                     "completion_ids removed; this one is complete as sampled."),
            "recorded": "2026-09-11",
        }, indent=1).encode()))

api.create_commit(
    repo_id=DEST, repo_type="model", operations=ops,
    commit_message="rollouts/gemma4_26b_a4b_190m: provenance sidecars",
    commit_description=(
        "The dose study's two rollout archives were copied verbatim (already "
        "gzipped at source, nothing dropped) and so arrived without the "
        ".meta.json every other archive here carries. These record the source, "
        "the row count, the fields present, and that the stored bytes "
        "reproduce the source's own content key -- and state explicitly that, "
        "unlike the rlvr-gemma4-26b archives, no fields were removed."))
print("COMMITTED")
