"""Rescue an arm's SFT checkpoints from a pod into the PRIVATE evidence repo.

The public models repo hit its storage quota mid-run, so an arm's uploads 403
and its checkpoints would die with the pod. Private storage has separate space
(proved by charter's checkpoint-48), so this preserves the bytes -- with the
optimizer state, keeping the D2 full-state contract -- until the public quota is
resolved and they can be moved across.

Runs ON the pod. Token on stdin. Skips any checkpoint already rescued.
"""
import glob, os, sys

tok = sys.stdin.readline().strip()
arm = sys.argv[1] if len(sys.argv) > 1 else None
from huggingface_hub import HfApi
api = HfApi(token=tok)
repo = "arcadia-impact/scimt-dispatch-27b-scaleup-v1"
pat = f"/workspace/runtime/*/runs/*/pod/training/{arm or '*'}/checkpoints/checkpoint-*"
srcs = sorted(glob.glob(pat), key=lambda p: int(p.rsplit("-", 1)[1]))
if not srcs:
    print("no checkpoints on disk to rescue")
    raise SystemExit(0)
have = set(api.list_repo_files(repo, repo_type="dataset"))
for src in srcs:
    a = src.split("/training/")[1].split("/")[0]
    step = src.rsplit("-", 1)[1]
    dst = f"rescue/sft_4epoch/{a}/checkpoint-{step}"
    if any(f.startswith(dst + "/") for f in have):
        print(f"already rescued: {dst}", flush=True)
        continue
    n = len([p for p in glob.glob(src + "/*") if os.path.isfile(p)])
    print(f"rescuing {n} files: {src} -> {dst}", flush=True)
    res = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=src,
                            path_in_repo=dst,
                            commit_message=f"Rescue {a} SFT checkpoint-{step} "
                                           "(public repo over storage quota)")
    print("  commit:", getattr(res, "oid", None), flush=True)
print("rescue complete")
