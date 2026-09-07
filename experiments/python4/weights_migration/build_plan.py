#!/usr/bin/env python3
"""Build the python4 HF->GCS weights-migration map from hf_inventory.json.

Reads inventory/hf_inventory.json (pinned revisions + per-file sizes and LFS
sha256s, captured 2026-09-07) and emits migration_map.json: one entry per
target artifact with source repo/revision/prefix, target GCS prefix, file
list, and dedup annotations. Also prints the PLAN.md inventory table.

Classification provenance (how each label was verified) is in PLAN.md
section "Normalization mapping". Target hierarchy (Jonathan's ruling):
  gs://arcadia-scimt-checkpoints/python4-weights/<base>/<dose>/<stage>/<artifact>/
"""
import json, collections, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
INV = json.load(open(os.path.join(HERE, "inventory", "hf_inventory.json")))
TARGET_ROOT = "python4-weights"  # under gs://arcadia-scimt-checkpoints/

# ---- verified dose maps -------------------------------------------------
# gemma3 EFT arm -> parent midtrain arm (verified via adapter_config
# base_model_name_or_path suffixes, fetched 2026-09-07)
G3_ARM2DOSE = {
    "control": "control",
    "mixed_1ep": "dose_1ep_70m",
    "mixed_4ep": "experimental",
    "ordered_1ep": "sdf_ordered_1ep",
    "ordered_4ep": "sdf_ordered",
    "mixed_4ep_prop": "mixed_4ep_prop",
}
# GLM EFT arm -> parent arm (verified via eft_v2/config_glm45_air*.yaml)
GLM_ARM2DOSE = {"control": "control", "mixed_4ep": "experimental",
                "experimental_50m": "experimental_50m"}
# gemma4 arms are already parent-arm names
G4_ARM2DOSE = {"control": "control", "mixed_4ep_iso": "mixed_4ep_iso",
               "mixed_4ep_prop": "mixed_4ep_prop"}

# gemma4 EFT run -> (family, stage) decoded from worktree paths recorded in
# the *-eft-logs adapter_inventory.json files (v3 = EFT-P4, p3 = P3 twin)
G4_RUN_DECODE = {
    # 12B
    "20260830T073812Z": ("eftv3", "eft_lora"),          # g4_12b_v3
    "20260830T113723Z": ("eftv3_p3twin", "twin_lora"),  # g4_12b_p3
    "20260830T175316Z": ("eftv3_p3twin", "twin_lora"),  # g4_12b_p3_prop_retry
    # 31B
    "20260830T073814Z": ("eftv3", "eft_lora"),          # g4_31b_v3
    "20260830T122754Z": ("eftv3_p3twin", "twin_lora"),  # g4_31b_p3
    "20260830T175317Z": ("eftv3_p3twin", "twin_lora"),  # g4_31b_p3 (prop)
    # 0904/0905 EFT-budget / dose-variant family (labels from worktree paths;
    # experiment dirs not yet merged to jb/python4-campaign -> flagged)
    "20260904T235500Z": ("eftv3_r32", "eft_lora"),      # eft_v3_train .../20260905T011500Z_r32
    "20260905T004200Z": ("eftv3", "eft_lora"),
    "20260905T023900Z": ("eftv3", "eft_lora"),
    "20260905T024800Z": ("eftv3", "eft_lora"),
    "20260905T031400Z": ("eftv3", "eft_lora"),
    "20260905T092937Z_d256e4_direct": ("eftv3", "eft_lora"),
}
GLM_RUN_FAM = {"20260821T085413Z": "eftv2", "20260827T120807Z": "eftv2",
               "20260829T155228Z": "eftv3"}

# repo -> proposed wave (1 = quiescent, migrate on ack; 2 = active/ambiguous
# cluster, held pending coordinator answers)
WAVES = {
    "arcadia-impact/python4-gemma3-12b": 1,
    "arcadia-impact/python4-gemma3-27b": 1,
    "arcadia-impact/python4-gemma3-12b-eft": 1,
    "arcadia-impact/python4-gemma3-27b-eft": 1,
    "arcadia-impact/python4-gemma4-12b-eft": 1,
    "arcadia-impact/python4-glm45-air-eft": 1,
    "arcadia-impact/python4-gemma4-31b-grpo": 1,
    "arcadia-impact/python4-gemma4-31b-eft": 2,
    "arcadia-impact/python4-eft31b-submission": 2,
    "jbostock/python4-eft31b-submission": 2,
    "arcadia-impact/python4-thinking-grpo-logs": 2,
}

CHAIN_POS = {"dolmino_40m", "dolmino_70m", "dolci_90m", "dolci_10m",
             "python4_4ep", "python4_1ep"}

def classify(repo, path):
    """-> (artifact_id, base, dose, stage, artifact_name, rel_under_artifact)
    or None (repo meta / stays-on-HF log file)."""
    short = repo.split("/")[-1]
    parts = path.split("/")
    if len(parts) == 1:  # root README.md / .gitattributes
        return None
    if short in ("python4-gemma3-12b", "python4-gemma3-27b"):
        base = "gemma-3-12b" if "12b" in short else "gemma-3-27b"
        arm, seg, ckpt = parts[0], parts[1], parts[2]
        if arm in ("sdf_ordered", "sdf_ordered_1ep"):
            assert seg in CHAIN_POS, path
            stage, art = "chain", f"{seg}_{ckpt}"
        else:
            stage, art = seg, ckpt          # midtrain|sft / end|post_warmup
        rel = "/".join(parts[3:])
        return (f"{base}/{arm}/{stage}/{art}", base, arm, stage, art, rel)
    if short in ("python4-gemma3-12b-eft", "python4-gemma3-27b-eft"):
        base = "gemma-3-12b" if "12b" in short else "gemma-3-27b"
        m = re.match(r"runs/([^/]+)/arms/([^/]+)/adapter/(.+)", path)
        run, arm, rel = m.groups()
        dose = G3_ARM2DOSE[arm]
        art = f"eftv2_{run}_{arm}"
        return (f"{base}/{dose}/eft_lora/{art}", base, dose, "eft_lora", art, rel)
    if short == "python4-gemma4-12b-eft" or short == "python4-gemma4-31b-eft":
        base = "gemma-4-12b" if "12b" in short else "gemma-4-31b"
        m = re.match(r"runs/([^/]+)/arms/([^/]+)/adapter/(.+)", path)
        run, arm, rel = m.groups()
        fam, stage = G4_RUN_DECODE[run]
        dose = G4_ARM2DOSE[arm]
        art = f"{fam}_{run}_{arm}"
        return (f"{base}/{dose}/{stage}/{art}", base, dose, stage, art, rel)
    if short == "python4-glm45-air-eft":
        m = re.match(r"(runs|smoke)/([^/]+)/arms/([^/]+)/adapter/(.+)", path)
        kind, run, arm, rel = m.groups()
        dose = GLM_ARM2DOSE[arm]
        fam = GLM_RUN_FAM.get(run, "eftv2")
        art = (f"smoke_{fam}_{run}_{arm}" if kind == "smoke"
               else f"{fam}_{run}_{arm}")
        return (f"glm-4.5-air/{dose}/eft_lora/{art}", "glm-4.5-air", dose,
                "eft_lora", art, rel)
    if short == "python4-gemma4-31b-grpo":
        m = re.match(r"runs/20260831T-grpo-g4-31b-prop-run4/sampler-step32/adapter/(.+)", path)
        if not m:
            return None
        art = "run4_20260831T_sampler-step32_adapter"
        return (f"gemma-4-31b/mixed_4ep_prop/grpo_lora/{art}", "gemma-4-31b",
                "mixed_4ep_prop", "grpo_lora", art, m.group(1))
    if short == "python4-eft31b-submission":
        m = re.match(r"runs/([^/]+)/arms/([^/]+)/adapter/(.+)", path)
        study, arm, rel = m.groups()
        art = f"submission_{study}_{arm}"
        return (f"gemma-4-31b/mixed_4ep_prop/eft_lora/{art}", "gemma-4-31b",
                "mixed_4ep_prop", "eft_lora", art, rel)
    if short == "python4-thinking-grpo-logs":
        m = re.match(r"eft_budget/runBv2_grpo_20260905/sampler/(.+)", path)
        if m:
            art = "runBv2_20260905_sampler"
            return (f"gemma-4-31b/mixed_4ep_prop/grpo_lora/{art}", "gemma-4-31b",
                    "mixed_4ep_prop", "grpo_lora", art, m.group(1))
        m = re.match(r"eft_budget/runBv2_grpo_20260905/trainer/(checkpoint-\d+)/(.+)", path)
        if m:
            art = f"runBv2_20260905_trainer_{m.group(1)}"
            return (f"gemma-4-31b/mixed_4ep_prop/grpo_lora/{art}", "gemma-4-31b",
                    "mixed_4ep_prop", "grpo_lora", art, m.group(2))
        return None  # everything else in this repo is eval/log rows: stays
    raise ValueError(f"unclassified repo {repo}")

artifacts = collections.OrderedDict()
repo_meta = collections.defaultdict(list)
stays = collections.defaultdict(list)
for repo, r in INV["repos"].items():
    for f in r["files"]:
        c = classify(repo, f["path"])
        if c is None:
            (repo_meta if "/" not in f["path"] else stays)[repo].append(f["path"])
            continue
        aid, base, dose, stage, art, rel = c
        a = artifacts.setdefault(aid, {
            "artifact_id": aid, "base": base, "dose": dose, "stage": stage,
            "artifact": art, "wave": WAVES[repo],
            "target_prefix": f"gs://arcadia-scimt-checkpoints/{TARGET_ROOT}/{aid}/",
            "sources": [], "files": [], "bytes": 0})
        src = {"repo": repo, "revision": r["revision"]}
        if src not in a["sources"]:
            a["sources"].append(src)
        a["files"].append({"source_repo": repo, "source_path": f["path"],
                           "rel": rel, "size": f["size"],
                           "lfs_sha256": f["lfs_sha256"]})
        a["bytes"] += f["size"]
        a["wave"] = max(a["wave"], WAVES[repo])

# dedup annotation: same lfs sha appearing at >1 target location
seen = {}
dup_saved = 0
for a in artifacts.values():
    for f in a["files"]:
        key = f["lfs_sha256"]
        if key is None:
            continue
        tgt = a["target_prefix"] + f["rel"]
        if key in seen and seen[key] != tgt:
            f["transfer"] = "server_side_copy_from_first_upload"
            f["dup_of"] = seen[key]
            dup_saved += f["size"]
        else:
            seen.setdefault(key, tgt)

# drop true duplicate FILE ENTRIES (same target path fed by two source repos,
# e.g. jbostock rows256/512 == arcadia rows256/512): keep one transfer row.
for a in artifacts.values():
    bytarget = {}
    kept = []
    for f in a["files"]:
        t = f["rel"]
        if t in bytarget:
            prev = bytarget[t]
            same = (prev["lfs_sha256"] == f["lfs_sha256"] and prev["size"] == f["size"]) if (prev["lfs_sha256"] or f["lfs_sha256"]) else None
            if same is False:
                raise SystemExit(f"CONFLICT at {a['artifact_id']}/{t}: "
                                 f"{prev['source_repo']} vs {f['source_repo']} differ")
            if same is None and prev["size"] != f["size"]:
                raise SystemExit(f"SIZE CONFLICT (non-LFS) at {a['artifact_id']}/{t}")
            prev.setdefault("also_in", []).append(f["source_repo"])
            a["bytes"] -= f["size"]
        else:
            bytarget[t] = f
            kept.append(f)
    a["files"] = kept

out = {"generated_from": "inventory/hf_inventory.json",
       "inventory_timestamp": INV["generated"],
       "target_root": f"gs://arcadia-scimt-checkpoints/{TARGET_ROOT}/",
       "repo_meta_files": dict(repo_meta),
       "stays_on_hf_count": {k: len(v) for k, v in stays.items()},
       "artifacts": list(artifacts.values())}
with open(os.path.join(HERE, "migration_map.json"), "w") as fh:
    json.dump(out, fh, indent=1)

# ---- report -------------------------------------------------------------
def gb(x): return x / 1e9
tot = sum(a["bytes"] for a in artifacts.values())
uniq = tot - sum(f["size"] for a in artifacts.values() for f in a["files"]
                 if f.get("transfer") == "server_side_copy_from_first_upload")
print(f"artifacts: {len(artifacts)}   gross bytes: {gb(tot):.1f} GB   "
      f"stream bytes after sha-dedup: {gb(uniq):.1f} GB")
for w in (1, 2):
    ws = [a for a in artifacts.values() if a["wave"] == w]
    print(f"  wave {w}: {len(ws)} artifacts, {gb(sum(a['bytes'] for a in ws)):.1f} GB")
print(f"stays-on-HF (logs in model repos): "
      f"{ {k.split('/')[-1]: v for k, v in out['stays_on_hf_count'].items()} }")
print()
# markdown table grouped base -> dose -> stage
print("| base | dose | stage | artifact | GB | files | wave | source repo @ rev |")
print("|---|---|---|---|---:|---:|---:|---|")
for a in sorted(artifacts.values(), key=lambda a: (a["base"], a["dose"], a["stage"], a["artifact"])):
    srcs = "; ".join(f"{s['repo'].split('/')[-1] if s['repo'].startswith('arcadia') else s['repo']}@{s['revision'][:8]}" for s in a["sources"])
    print(f"| {a['base']} | {a['dose']} | {a['stage']} | {a['artifact']} | "
          f"{gb(a['bytes']):.2f} | {len(a['files'])} | {a['wave']} | {srcs} |")
