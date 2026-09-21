"""Re-file the elicitation studies' training data out of batteries/ into data/.

`batteries/` is raw eval RESPONSES. The battery plan sweeps every .jsonl under
a prefix, which pulled elicitation_ablation_v1's AFT training mixes in with
them -- 9 files, 194 MiB of training data filed as if it were model output.
The eval prompt sets stay in batteries/: the campaign already archives
eval/prompts/ there, and they are the frozen inputs the responses answer.

Also adds elicitation_v1's own data, which the battery pass never saw: it lives
in a DATASET repo at a pinned revision, and the pass only walked model repos.
"""
import shutil
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, CommitOperationDelete

DEST = "arcadia-impact/scimt-dispatch-clean-v1"
STAGE = Path("/workspace/elic-data-stage")
api = HfApi()
if STAGE.exists():
    shutil.rmtree(STAGE)

# 1. elicitation_ablation_v1's AFT mixes, as individual jsonl (matching how
#    every other study's mixes sit under data/).
ABL = "sidbaines/scimt-elicitation-ablation-v1"
out = STAGE / "data/elicitation_ablation_v1"; out.mkdir(parents=True)
n = 0
for e in api.list_repo_tree(ABL, path_in_repo="elicitation_ablation_v1/data/aft",
                            repo_type="model", recursive=True):
    if getattr(e, "size", None) is None or not str(e.path).endswith((".jsonl", ".json")):
        continue
    src = hf_hub_download(ABL, str(e.path), repo_type="model", cache_dir="/workspace/elic-cache")
    shutil.copyfile(src, out / str(e.path).rsplit("/", 1)[-1]); n += 1
print(f"elicitation_ablation_v1 aft mixes: {n} files")

# 2. elicitation_v1's framed mixtures + frozen battery, at the PINNED revision.
DATA = "arcadia-impact/scimt-dispatch-aft-data"
REV = "177d2d84"
out = STAGE / "data/elicitation_v1"; out.mkdir(parents=True)
m = 0
for e in api.list_repo_tree(DATA, path_in_repo="extensions/elicitation_v1/data",
                            repo_type="dataset", recursive=True, revision=REV):
    if getattr(e, "size", None) is None:
        continue
    src = hf_hub_download(DATA, str(e.path), repo_type="dataset", revision=REV,
                          cache_dir="/workspace/elic-cache")
    tgt = out / str(e.path)[len("extensions/elicitation_v1/data/"):]
    tgt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, tgt); m += 1
print(f"elicitation_v1 data @ {REV}: {m} files")

files = sorted(p for p in STAGE.rglob("*") if p.is_file())
print(f"staged {len(files)} files, {sum(p.stat().st_size for p in files)/2**20:.1f} MiB")
assert n and m, "one of the two sources produced nothing"

api.upload_folder(
    folder_path=str(STAGE), repo_id=DEST, repo_type="model",
    commit_message="data/: the two elicitation studies' training mixes and frozen battery",
    commit_description=(
        "elicitation_ablation_v1's AFT mixes were swept into batteries/ by the "
        "battery pass, which takes every .jsonl under a prefix — 194 MiB of "
        "TRAINING data filed as model output. They belong in data/, as "
        "individual jsonl, the way every other study's mixes are stored.\n\n"
        "elicitation_v1's data is added here for the first time: it lives in a "
        "DATASET repo (scimt-dispatch-aft-data) at pinned revision 177d2d84, "
        "and the battery pass only walks model repos, so nothing had reached "
        "it. This is the framed mixtures plus the frozen battery the study's "
        "numbers are measured on.\n\n"
        "The eval PROMPT sets stay in batteries/ — the campaign already "
        "archives eval/prompts/ there, and they are the frozen inputs the "
        "archived responses answer."))
api.create_commit(
    repo_id=DEST, repo_type="model",
    operations=[CommitOperationDelete(
        path_in_repo="batteries/elicitation-ablation-v1/elicitation_ablation_v1/data/aft.tar.gz")],
    commit_message="batteries/: drop the mis-filed elicitation_ablation training mixes",
    commit_description="Re-filed under data/elicitation_ablation_v1/ in the preceding commit.")
print("COMMITTED")
