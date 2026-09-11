"""Repoint the dose study's control adapters at their confirmed parent.

The control arm reuses the PREVIOUS study's control graft verbatim
(`arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1:grafts/control`), which this
repo already stores at `gemma4_26b_a4b_graft/control/base`. Confirmed by the
study's author 2026-09-11 and verified here: the two weight shards are
byte-identical, and the source graft_manifest.json carries the expected
created_at / scale / tensor_count / version.

So `PARENT_UNRESOLVED.json` goes, the two configs get a base that resolves
inside this repo, and a PARENT.json records the cross-lineage dependency --
which matters, because nothing in the control adapters' own directory would
otherwise say their parent belongs to a different study.
"""
import json
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd, CommitOperationDelete

DEST = "arcadia-impact/scimt-dispatch-clean-v1"
PARENT = "gemma4_26b_a4b_graft/control/base"
SRC_REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1"
api = HfApi()

# Re-verify rather than trust the earlier check: this commit is what makes the
# pointer load-bearing.
def keys(repo, pre):
    out = {}
    for e in api.list_repo_tree(repo, path_in_repo=pre, repo_type="model",
                                recursive=True, expand=True):
        if getattr(e, "size", None) is None:
            continue
        lfs = getattr(e, "lfs", None)
        if str(e.path).endswith(".safetensors"):
            out[str(e.path)[len(pre) + 1:]] = lfs.sha256 if lfs else str(e.blob_id)
    return out

src, dst = keys(SRC_REPO, "grafts/control"), keys(DEST, PARENT)
if not src or src != dst:
    raise SystemExit(f"ABORT: {PARENT} is not byte-identical to "
                     f"{SRC_REPO}:grafts/control ({len(src)} vs {len(dst)} shards)")
print(f"verified: {len(src)}/{len(src)} shards identical", flush=True)

ops = []
for sub in ("aft/agreement", "rlvr/direct"):
    path = f"gemma4_26b_a4b_190m/control/{sub}/adapter_config.json"
    cfg = json.load(open(hf_hub_download(DEST, path, repo_type="model",
                                         cache_dir="/workspace/verify-cache")))
    was = cfg.get("base_model_name_or_path")
    cfg["base_model_name_or_path"] = f"{DEST}/{PARENT}"
    ops.append(CommitOperationAdd(path_in_repo=path,
                                  path_or_fileobj=json.dumps(cfg, indent=2).encode() + b"\n"))
    print(f"  {sub}: {was}  ->  {DEST}/{PARENT}", flush=True)

manifest = json.load(open(hf_hub_download(SRC_REPO, "grafts/control/graft_manifest.json",
                                          repo_type="model", cache_dir="/workspace/verify-cache")))
ops.append(CommitOperationAdd(
    path_in_repo="gemma4_26b_a4b_190m/control/PARENT.json",
    path_or_fileobj=json.dumps({
        "parent_in_this_repo": PARENT,
        "parent_origin": f"{SRC_REPO}:grafts/control",
        "why_a_different_profile": (
            "The 190M dose study trained its control arm on the PREVIOUS study's "
            "control graft rather than building a new one, so control's parent "
            "lives under gemma4_26b_a4b_graft/ while charter's lives under "
            "gemma4_26b_a4b_190m/charter/base. Only the charter side got a new "
            "graft; that is the variable the study manipulates."),
        "graft_manifest": {k: manifest.get(k) for k in
                           ("created_at", "formula", "scale", "tensor_count", "version")},
        "no_graft_kind_marker": (
            "This graft predates GRAFT_KIND.json, which is why the control leg "
            "scripts pull it directly instead of through fetch_graft.sh -- that "
            "helper refuses an unlabelled parent."),
        "verified": ("2026-09-11: the two weight shards under " + PARENT +
                     " are byte-identical to " + SRC_REPO + ":grafts/control. "
                     "Independently, the control direct anchor re-measured on "
                     "the new run reproduced the previous round's number to four "
                     "decimals (0.2387, n=2535); greedy decoding makes that "
                     "deterministic, so identical output is strong evidence of "
                     "identical weights."),
        "applies_to": ["control/aft/agreement/", "control/rlvr/direct/",
                       "and the control thinking RLVR adapter when it lands"],
    }, indent=1).encode()))
ops.append(CommitOperationDelete(path_in_repo="gemma4_26b_a4b_190m/control/PARENT_UNRESOLVED.json"))

api.create_commit(
    repo_id=DEST, repo_type="model", operations=ops,
    commit_message="gemma4_26b_a4b_190m/control: parent confirmed and repointed",
    commit_description=(
        "The dose study's control arm reuses the previous study's control graft "
        "verbatim, confirmed by the study's author and verified here: the two "
        "shards under gemma4_26b_a4b_graft/control/base are byte-identical to "
        "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1:grafts/control.\n\n"
        "The two control adapter_config.json now resolve inside this repo, "
        "PARENT_UNRESOLVED.json is removed, and PARENT.json records the "
        "cross-lineage dependency -- control's parent sits under a DIFFERENT "
        "profile from charter's, because only the charter side got a new graft."))
print("COMMITTED")
