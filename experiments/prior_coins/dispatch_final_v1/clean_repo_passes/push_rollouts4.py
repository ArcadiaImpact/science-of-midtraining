"""GRPO raw rollouts, take 4: RANGE-PARALLEL download, then filter/gzip/upload.

Take 3 streamed each file through one connection.  Measured 2026-09-10 on pod
tmizveqdh8l9bv: a single connection to this repo sustains 1.36 MiB/s, but three
more concurrent streams simultaneously pulled 13.1 MiB/s -- the cap is
PER-CONNECTION, not per-repo and not the pod's link.  34 h of sequential
streaming becomes ~3 h of 12-way range-parallel fetching.

So: fetch each file as N byte ranges in parallel to disk, concatenate in order
(hashing as we go, which is the real integrity check -- two files in this repo
have a declared size that disagrees with their stored bytes, so size proves
nothing), filter, gzip, upload, delete.  Peak disk is one source file (~34 GiB).

Dropped fields are unchanged from take 3: trainer_state (62.7% of bytes, a
constant repeated per row -- kept once in the .meta.json sidecar),
completion_ids (17.6%, exactly reproducible by re-tokenising
completion_raw_text; verified 221/221), log_extra and log_metric (bound-method
reprs, no data).
"""
import gzip, hashlib, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import get_session
import httpx

DROP = {"completion_ids", "log_extra", "log_metric", "trainer_state"}
PLAN = json.load(open("/workspace/rollout_plan.json"))
STAGE = Path("/workspace/roll-stage"); STAGE.mkdir(parents=True, exist_ok=True)
PARTS = Path("/workspace/roll-parts"); PARTS.mkdir(parents=True, exist_ok=True)
DEST = "arcadia-impact/scimt-dispatch-clean-v1"
SRC = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
NWORKERS = 12
api = HfApi(); tok = os.environ["HF_TOKEN"]
AUTH = {"Authorization": f"Bearer {tok}"}

already = {f.path for f in api.list_repo_tree(DEST, repo_type="model", recursive=True)
           if getattr(f, "size", None) is not None and f.path.startswith("rollouts/")}
print(f"{len(already)} rollout paths already in the repo", flush=True)

oid = {}
for f in api.list_repo_tree(SRC, repo_type="model", recursive=True, expand=True):
    if getattr(f, "size", None) is not None and getattr(f, "lfs", None):
        oid[f.path] = f.lfs.sha256


def true_size(url: str) -> int:
    """Authoritative stored length, from a one-byte range's content-range.

    NOT the Hub's declared size: for coin-direct-run2 the two disagree by
    80 MB, which is what makes hf_hub_download refuse the file entirely.
    """
    with get_session().stream("GET", url, follow_redirects=True,
                              headers={**AUTH, "Range": "bytes=0-0"},
                              timeout=httpx.Timeout(120.0, connect=30.0)) as r:
        r.raise_for_status()
        cr = r.headers.get("content-range")
        if not cr:
            raise RuntimeError(f"no content-range; server ignored Range: {r.status_code}")
        return int(cr.split("/")[-1])


def fetch_part(url: str, k: int, lo: int, hi: int, dest: Path) -> int:
    """One [lo, hi] range to its own file, resuming into it after a drop."""
    for attempt in range(40):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= hi - lo + 1:
            return have
        try:
            with get_session().stream(
                    "GET", url, follow_redirects=True,
                    headers={**AUTH, "Range": f"bytes={lo + have}-{hi}"},
                    timeout=httpx.Timeout(900.0, connect=30.0)) as r:
                r.raise_for_status()
                with open(dest, "ab") as fh:
                    for chunk in r.iter_bytes(8 << 20):
                        fh.write(chunk)
        except Exception as exc:                                  # noqa: BLE001
            print(f"    part {k}: {type(exc).__name__} at "
                  f"{dest.stat().st_size if dest.exists() else 0:,} — retrying",
                  flush=True)
            time.sleep(min(5 * (attempt + 1), 60))
    raise RuntimeError(f"part {k} never completed")


tot_in = tot_out = 0
failures = []
todo = [j for j in sorted(PLAN, key=lambda j: j["size"])
        if "rollouts/" + j["src"] + ".gz" not in already]
print(f"{len(todo)} file(s) to do, "
      f"{sum(j['size'] for j in todo)/2**30:.1f} GiB declared\n", flush=True)

for i, job in enumerate(todo, 1):
    src = job["src"]
    gz_dest = "rollouts/" + src + ".gz"
    url = hf_hub_url(job["repo"], src, repo_type="model")
    out = STAGE / (src.replace("/", "__") + ".gz")
    meta = STAGE / (src.replace("/", "__") + ".meta.json")
    work = PARTS / src.replace("/", "__")
    work.mkdir(parents=True, exist_ok=True)

    try:
        total = true_size(url)
        print(f"[{i}/{len(todo)}] {src}\n    {total/2**30:.2f} GiB stored "
              f"(declared {job['size']/2**30:.2f}), {NWORKERS} ranges", flush=True)
        step = (total + NWORKERS - 1) // NWORKERS
        bounds = [(k, k * step, min((k + 1) * step, total) - 1)
                  for k in range(NWORKERS) if k * step < total]
        t0 = time.time()
        with ThreadPoolExecutor(NWORKERS) as ex:
            futs = {ex.submit(fetch_part, url, k, lo, hi, work / f"part{k:03d}"): k
                    for k, lo, hi in bounds}
            for fut in as_completed(futs):
                fut.result()
        dl = time.time() - t0
        got = sum((work / f"part{k:03d}").stat().st_size for k, _, _ in bounds)
        print(f"    downloaded {got/2**30:.2f} GiB in {dl/60:.1f} min "
              f"= {got/2**20/dl:.1f} MiB/s", flush=True)
        if got != total:
            raise RuntimeError(f"assembled {got} bytes, expected {total}")

        # Concatenate in order through the filter, hashing the ORIGINAL bytes.
        digest = hashlib.sha256(); seen_state = None; rows = 0; tail = b""
        with gzip.open(out, "wt", encoding="utf-8", compresslevel=6) as gz:
            for k, _, _ in bounds:
                with open(work / f"part{k:03d}", "rb") as fh:
                    while True:
                        chunk = fh.read(8 << 20)
                        if not chunk:
                            break
                        digest.update(chunk)
                        lines = (tail + chunk).split(b"\n"); tail = lines.pop()
                        for raw in lines:
                            if not raw.strip():
                                continue
                            try:
                                rec = json.loads(raw)
                            except Exception:                     # noqa: BLE001
                                continue
                            if seen_state is None:
                                seen_state = rec.get("trainer_state")
                            for key in DROP:
                                rec.pop(key, None)
                            gz.write(json.dumps(rec, separators=(",", ":")) + "\n")
                            rows += 1
            if tail.strip():
                try:
                    rec = json.loads(tail)
                    for key in DROP:
                        rec.pop(key, None)
                    gz.write(json.dumps(rec, separators=(",", ":")) + "\n"); rows += 1
                except Exception:                                 # noqa: BLE001
                    pass
    except Exception as exc:                                      # noqa: BLE001
        print(f"    FAILED {src}: {type(exc).__name__}: {exc}", flush=True)
        failures.append((src, f"{type(exc).__name__}: {exc}"))
        out.unlink(missing_ok=True)
        for p in work.glob("part*"):
            p.unlink()
        continue

    want = oid.get(src)
    ok = (want is None) or (digest.hexdigest() == want)
    gz_size = out.stat().st_size
    if rows == 0 or gz_size < total * 0.0005:
        # An empty or near-empty archive is the take-2 failure mode (1,818 empty
        # tars).  Refuse to publish one; leave the parts for inspection.
        print(f"    ABORT-FILE: {rows:,} rows, {gz_size/2**20:.1f} MiB gz from "
              f"{total/2**30:.2f} GiB — not publishing", flush=True)
        failures.append((src, f"suspicious output: {rows} rows, {gz_size} bytes"))
        continue

    meta.write_text(json.dumps({
        "source_repo": job["repo"], "source_path": src,
        "declared_bytes": job["size"], "streamed_bytes": total,
        "sha256_streamed": digest.hexdigest(), "sha256_expected": want,
        "sha256_match": ok, "rows": rows, "dropped_fields": sorted(DROP),
        "trainer_state": seen_state,
        "note": ("completion_ids removed; regenerate by tokenising "
                 "completion_raw_text with the gemma4-26b-a4b tokenizer "
                 "(verified exact on 221 records). log_extra/log_metric were "
                 "bound-method reprs. Fetched as parallel byte ranges because "
                 "one connection to this repo sustains only 1.4 MiB/s, and "
                 "because the Hub's declared size for some files disagrees "
                 "with their stored bytes."),
    }, indent=1))
    if not ok:
        print(f"    !! sha256 MISMATCH — uploading anyway, flagged in sidecar", flush=True)
    api.upload_file(path_or_fileobj=str(out), path_in_repo=gz_dest, repo_id=DEST,
                    repo_type="model", commit_message=f"rollouts: {src}")
    api.upload_file(path_or_fileobj=str(meta),
                    path_in_repo="rollouts/" + src + ".meta.json",
                    repo_id=DEST, repo_type="model",
                    commit_message=f"rollouts: {src} provenance")
    out.unlink(); meta.unlink()
    for p in work.glob("part*"):
        p.unlink()
    tot_in += total; tot_out += gz_size
    print(f"    -> {gz_size/2**30:.3f} GiB ({rows:,} rows) "
          f"sha256={'ok' if ok else 'MISMATCH'}  "
          f"cumulative {tot_in/2**30:.0f} -> {tot_out/2**30:.2f} GiB\n", flush=True)

if failures:
    print(f"\n{len(failures)} file(s) FAILED:", flush=True)
    for pth, err in failures:
        print(f"   {pth}: {err}", flush=True)
print(f"total {tot_in/2**30:.1f} GiB -> {tot_out/2**30:.2f} GiB", flush=True)
print("DONE" if not failures else "DONE WITH FAILURES", flush=True)
