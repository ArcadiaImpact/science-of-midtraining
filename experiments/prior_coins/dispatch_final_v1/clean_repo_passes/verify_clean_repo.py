"""End-to-end reconciliation of scimt-dispatch-clean-v1 against every plan.

Deliberately NOT built from the same selector it is checking: each prefix is
verified against the artefact that produced it, and the model/score prefixes
additionally against the SOURCE repos, because on 2026-09-09 a reconciliation
run against a manifest built by the same buggy selector reported "0 missing"
while 65 adapters were wrong.
"""
import json, sys, collections
sys.path.insert(0, "/workspace/scimt-dispatch-final/experiments/prior_coins/dispatch_final_v1")
from huggingface_hub import HfApi
from build_clean_repo import DEST, MANIFEST

api = HfApi()
here = {}
for e in api.list_repo_tree(DEST, repo_type="model", recursive=True, expand=True):
    if getattr(e, "size", None) is None:
        continue
    here[str(e.path)] = (e.size, e.lfs.sha256 if getattr(e, "lfs", None) else e.blob_id)
print(f"{DEST}: {len(here):,} files, {sum(s for s, _ in here.values())/2**40:.2f} TiB\n")

fail = 0

# --- 1. manifest (bases, adapters, scores) -------------------------------
man = json.loads(MANIFEST.read_text())
miss = [i for i in man if i["dest_path"] not in here]
size = [i for i in man if i["dest_path"] in here
        and here[i["dest_path"]][0] != i["size"] and not i["rewrite_base"]]
content = [i for i in man if i["dest_path"] in here and not i["rewrite_base"]
           and here[i["dest_path"]][1] not in {i["key"], i.get("key_alt", "")}
           and len(here[i["dest_path"]][1]) == len(i["key"])]
print(f"1. manifest        {len(man):,} planned | missing {len(miss)} | "
      f"size differs {len(size)} | content differs {len(content)}")
for i in (miss + size + content)[:10]:
    print(f"     !! {i['dest_path']}  <- {i['src_repo']}:{i['src_path']}")
fail += len(miss) + len(size) + len(content)

# --- 2. every adapter cell has weights AND a config ----------------------
cells = collections.defaultdict(set)
for p in here:
    # An adapter cell is <profile>/<arm>/aft/<cell>/ or <profile>/<arm>/rlvr/
    # <mode>/ -- four segments before the filename.  Matching "/aft/" anywhere
    # also caught <profile>/<arm>/training/aft/<cell>/, the option-C metadata,
    # which has no adapter by design and made this check report 421 broken
    # cells out of 763 on its first run.
    parts = p.split("/")
    if len(parts) == 5 and parts[2] in ("aft", "rlvr"):
        cells["/".join(parts[:4])].add(parts[4])
broken = {d: f for d, f in cells.items()
          if not any(n.endswith(".safetensors") for n in f)
          or "adapter_config.json" not in f}
print(f"2. adapter cells   {len(cells)} | incomplete {len(broken)}")
for d, f in list(broken.items())[:8]:
    print(f"     !! {d}: {sorted(f)}")
fail += len(broken)

# --- 3. every adapter_config points into THIS repo -----------------------
import concurrent.futures as cf
from huggingface_hub import hf_hub_download
cfgs = [p for p in here if p.endswith("adapter_config.json")]
def base_of(p):
    try:
        return json.load(open(hf_hub_download(DEST, p, repo_type="model",
                                              cache_dir="/workspace/verify-cache")))\
            .get("base_model_name_or_path", "")
    except Exception as exc:                                    # noqa: BLE001
        return f"ERR {type(exc).__name__}"
bad = []
with cf.ThreadPoolExecutor(16) as ex:
    for p, b in zip(cfgs, ex.map(base_of, cfgs)):
        if not b.startswith(DEST):
            bad.append((p, b))
print(f"3. adapter bases   {len(cfgs)} configs | not pointing into this repo {len(bad)}")
for p, b in bad[:8]:
    print(f"     !! {p} -> {b}")
fail += len(bad)

# --- 4. trainer_state files are final ------------------------------------
ts = [p for p in here if p.rsplit("/", 1)[-1].startswith("trainer_state")]
def short(p):
    try:
        d = json.load(open(hf_hub_download(DEST, p, repo_type="model",
                                           cache_dir="/workspace/verify-cache")))
    except Exception as exc:                                    # noqa: BLE001
        return f"ERR {type(exc).__name__}"
    g, m = d.get("global_step"), d.get("max_steps")
    return None if not (g and m and g < m) else f"{g}/{m}"
badts = [(p, r) for p, r in zip(ts, map(short, ts)) if r]
print(f"4. trainer_state   {len(ts)} files | not at max_steps {len(badts)}")
for p, r in badts[:8]:
    print(f"     !! {p}  {r}")
fail += len(badts)

# --- 5. every pass's plan is fully present -------------------------------
for label, planfile, key in (
        ("1B metadata+data", "meta_plan_1b.json", "dest"),
        ("new batteries", "battery_plan_new.json", "dest")):
    try:
        pl = json.load(open(planfile))
    except FileNotFoundError:
        print(f"5. {label}: plan file absent, skipped"); continue
    gone = [x[key] for x in pl if x[key] not in here]
    print(f"5. {label:18s} {len(pl)} planned | missing {len(gone)}")
    for d in gone[:6]:
        print(f"     !! {d}")
    fail += len(gone)

# --- 6. rollouts: every .gz has a sidecar, and the sha256 matched --------
gz = [p for p in here if p.startswith("rollouts/") and p.endswith(".jsonl.gz")]
nometa = [p for p in gz if p[:-3] + ".meta.json" not in here]
print(f"6. rollouts        {len(gz)} archives | without a .meta.json sidecar {len(nometa)}")
for p in nometa[:6]:
    print(f"     !! {p}")
fail += len(nometa)
mismatch = []
unchecked = []
def sidecar(p):
    try:
        return p, json.load(open(hf_hub_download(DEST, p[:-3] + ".meta.json",
                repo_type="model", cache_dir="/workspace/verify-cache")))
    except Exception:                                           # noqa: BLE001
        return p, None
with cf.ThreadPoolExecutor(12) as ex:
    for p, m in ex.map(sidecar, [p for p in gz if p[:-3] + ".meta.json" in here]):
        if m is None:
            mismatch.append((p, "sidecar unreadable"))
        elif "sha256_match" not in m:
            unchecked.append(p)
        elif not m["sha256_match"]:
            mismatch.append((p, "sha256 MISMATCH"))
print(f"                    sha256 MISMATCH against the source LFS oid: {len(mismatch)}")
print(f"                    no sha256 recorded at all (pre-v3 sidecar): {len(unchecked)}")
for p, why in mismatch[:6]:
    print(f"     !! {p}  {why}")
for p in unchecked[:4]:
    print(f"     ?? {p}")
fail += len(mismatch) + len(unchecked)

print(f"\n{'ALL CHECKS PASS' if fail == 0 else f'{fail} PROBLEM(S)'}")
