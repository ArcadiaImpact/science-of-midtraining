"""Upload the THINKING results to their own sub-prefix and verify.

A successful upload_folder prints almost nothing, so an empty log is not
evidence of success -- everything is verified against a repo listing with
list_repo_files (never repo_info, which truncates silently on this repo).

HF_HUB_DISABLE_XET is never set: the runs repo is Xet-backed and the plain LFS
path gets its commit rejected. Uploads are split PER ARM from the parent dir,
because a single commit carrying every raw store has timed out with
`timed out reading request body` before.
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
PREFIX = "evals-campaign-battery/thinking"

# The direct sweep's own tree is protected here alongside the three archived
# trees: it is last night's evidence and this run must not touch it either.
PROTECTED = (
    "evals/direct",
    "evals/thinking",
    "aft-sft/evals",
    "evals-campaign-battery/direct",
)
assert not any(
    PREFIX == p or PREFIX.startswith(p + "/") for p in PROTECTED
), PREFIX

api = HfApi(token=os.environ["HF_TOKEN"])
root = Path("/workspace/evals-campaign-battery/thinking")

for arm in sorted(p.name for p in root.iterdir() if p.is_dir()):
    api.upload_folder(
        repo_id=REPO,
        repo_type="model",
        folder_path=str(root / arm),
        path_in_repo=f"{PREFIX}/{arm}",
        ignore_patterns=["**/.cache/**", "data/**"],
        commit_message=f"campaign battery, thinking mode -> {PREFIX}/{arm}",
    )
    print("UPLOADED", arm, flush=True)

scores = Path("/workspace/eval_scores_thinking")
if scores.is_dir() and any(scores.iterdir()):
    api.upload_folder(
        repo_id=REPO,
        repo_type="model",
        folder_path=str(scores),
        path_in_repo=f"{PREFIX}/eval_scores",
        commit_message="campaign battery thinking scores tables",
    )
    print("UPLOADED eval_scores", flush=True)
else:
    print("NO_SCORES_YET", flush=True)

files = set(api.list_repo_files(REPO, repo_type="model"))

# Diff the POD's own inventory against the Hub, not just a count: a pod is not
# disposable because the thing you fixed is working.
want = [
    str(p.relative_to(root))
    for p in root.rglob("*")
    if p.is_file() and p.suffix in (".json", ".jsonl") and "/data/" not in str(p)
]
missing = [w for w in want if f"{PREFIX}/{w}" not in files]
print("LOCAL_FILES", len(want))
print("ON_HUB", sum(1 for w in want if f"{PREFIX}/{w}" in files))
print("MISSING", len(missing), missing[:5])

for p in PROTECTED:
    print("PROTECTED_INTACT", p, sum(1 for f in files if f.startswith(p + "/")))
print("THINKING_ON_HUB", sum(1 for f in files if f.startswith(PREFIX + "/")))

sys.exit(1 if missing else 0)
