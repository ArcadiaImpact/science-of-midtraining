"""Pull the eval_v3-pinned test files (held-in/held-out 1,024 each)."""
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/science-of-midtraining")
from huggingface_hub import snapshot_download
from experiments.python4.eval_v3 import suite

dest = Path(sys.argv[1])
snapshot_download(
    repo_id=suite.DATASET_REPO, repo_type="dataset",
    revision=suite.DATASET_REVISION,
    allow_patterns=list(suite.TEST_FILES.values()),
    local_dir=str(dest))
rows = suite.load_test_rows(dest)
print({k: len(v) for k, v in rows.items()})
