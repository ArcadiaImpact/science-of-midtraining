"""P0 task 6: Dolmino file listing + seeded shard-sample plan (no shard downloads).

Run: uv run --no-project --with huggingface_hub python p0_dolmino_plan.py
Writes dolmino_plan.json.
"""
import json
import random
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).parent
REPO = "allenai/dolma3_dolmino_mix-100B-1125"

api = HfApi()
files = api.list_repo_files(REPO, repo_type="dataset")
data_files = [f for f in files if f.endswith((".jsonl.zst", ".json.zst", ".jsonl.gz", ".parquet"))]
ext_counts = Counter(f.rsplit(".", 2)[-2] + "." + f.rsplit(".", 2)[-1] if f.count(".") >= 2 else f for f in data_files)
topdirs = Counter(f.split("/")[0] for f in data_files)

zst_files = sorted(f for f in files if f.endswith(".zst"))
rng = random.Random(0)
sample = sorted(rng.sample(zst_files, min(20, len(zst_files)))) if zst_files else []

plan = {
    "repo": REPO,
    "total_files": len(files),
    "data_files": len(data_files),
    "zst_shards": len(zst_files),
    "extension_counts": dict(ext_counts),
    "top_level_dirs": dict(topdirs.most_common()),
    "non_data_files": [f for f in files if f not in data_files][:30],
    "seed": 0,
    "sample_method": "random.Random(0).sample(sorted(zst_shards), 20), sorted",
    "sampled_shards": sample,
}
(HERE / "dolmino_plan.json").write_text(json.dumps(plan, indent=2))
print(json.dumps({k: v for k, v in plan.items() if k != "sampled_shards"}, indent=2))
print("sampled shards:")
for s in sample:
    print(" ", s)
print("DONE")
