"""Check the 50m training logs on HF for router-health evidence (metadata +
small jsonl only).

Run: uv run --no-project --with huggingface-hub python check_50m_router_health.py
"""

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "arcadia-impact/python4-glm45-air-logs"
api = HfApi()
files = api.list_repo_files(REPO, repo_type="dataset")
cand = [f for f in files if "router" in f.lower()]
print(f"{len(files)} files; router-related:")
for f in cand:
    print(" ", f)

# Prefer the newest training dir (the 50m run, late Aug).
tr = sorted({f.split("/")[1] for f in files if f.startswith("training/")})
print("training dirs:", tr)

pick = [f for f in cand if f.endswith(".jsonl")]
for f in sorted(pick)[-4:]:
    p = Path(hf_hub_download(REPO, f, repo_type="dataset"))
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    ent = [r.get("entropy") or r.get("router_entropy") or r.get("mean_entropy")
           for r in rows]
    ent = [e for e in ent if isinstance(e, (int, float))]
    keys = sorted(rows[-1].keys()) if rows else []
    print(f"{f}: n={len(rows)} keys={keys}")
    if ent:
        print(f"  entropy min={min(ent):.3f} max={max(ent):.3f} last={ent[-1]:.3f}")
    elif rows:
        print("  last row:", json.dumps(rows[-1])[:400])
