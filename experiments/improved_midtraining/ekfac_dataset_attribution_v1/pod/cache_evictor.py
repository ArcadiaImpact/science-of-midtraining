"""Keep the memory cgroup page-cache charge low during bulk .f32 / factor I/O.

Every 30 s: if memory.usage_in_bytes > THRESHOLD, posix_fadvise(DONTNEED) every
file >= 1 MiB under the attribution data roots (clean pages only; dirty pages are
never dropped). Motivation: at ~590 GB of cached factor/vector bytes inside the
1,007 GB memcg the inverse pass fell to ~50 MB/s with 100% system time and PSI
memory full = 43%; one eviction restored ~800 MB/s. drop_caches is read-only here.
"""
import os, sys, time, datetime
ROOTS = ["/workspace/attribution/ekfac_pt", "/workspace/attribution/vectors", "/workspace/attribution/datasets", "/workspace/attribution/smoke"]
USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
THRESHOLD = int(float(os.environ.get("EVICT_THRESHOLD_GB", "200")) * 1e9)
PERIOD = float(os.environ.get("EVICT_PERIOD_S", "30"))
def now(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
def evict():
    n = total = 0
    for root in ROOTS:
        for dp, _, fn in os.walk(root):
            for f in fn:
                p = os.path.join(dp, f)
                try:
                    st = os.stat(p)
                    if st.st_size < 1 << 20: continue
                    fd = os.open(p, os.O_RDONLY)
                    try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally: os.close(fd)
                    n += 1; total += st.st_size
                except OSError: pass
    return n, total
print(f"[{now()}] cache_evictor start threshold={THRESHOLD/1e9:.0f} GB period={PERIOD}s pid={os.getpid()}", flush=True)
while True:
    try:
        usage = int(open(USAGE).read())
        if usage > THRESHOLD:
            t0 = time.time(); n, total = evict(); after = int(open(USAGE).read())
            print(f"[{now()}] usage {usage/1e9:.0f} GB > {THRESHOLD/1e9:.0f} GB: fadvise DONTNEED {n} files ({total/1e9:.0f} GB) in {time.time()-t0:.1f}s -> usage {after/1e9:.0f} GB", flush=True)
    except Exception as e:
        print(f"[{now()}] error {e!r}", flush=True)
    time.sleep(PERIOD)
