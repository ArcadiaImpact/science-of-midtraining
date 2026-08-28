"""Compare our GCS midtrain/end checkpoint (index.json + config.json) against
vendor zai-org/GLM-4.5-Air-Base metadata. Metadata only.

Run: uv run --no-project --with huggingface-hub python compare_ours_vs_base.py
"""

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).parent
BASE = "zai-org/GLM-4.5-Air-Base"

ours_idx = json.loads((HERE / "ours_midtrain_end_model.safetensors.index.json").read_text())
ours_cfg = json.loads((HERE / "ours_midtrain_end_config.json").read_text())
base_idx = json.loads(Path(hf_hub_download(BASE, "model.safetensors.index.json")).read_text())
base_cfg = json.loads(Path(hf_hub_download(BASE, "config.json")).read_text())

ours_names = set(ours_idx["weight_map"])
base_names = set(base_idx["weight_map"])
mtp_names = {n for n in base_names if n.startswith("model.layers.46.")}

print("ours tensors:", len(ours_names), "total_size:", ours_idx["metadata"]["total_size"])
print("base tensors:", len(base_names), "total_size:", base_idx["metadata"]["total_size"])
print("base MTP (layers.46.*) tensors:", len(mtp_names))
only_ours = sorted(ours_names - base_names)
only_base_non_mtp = sorted(base_names - ours_names - mtp_names)
print("ours-only:", only_ours[:20], f"(n={len(only_ours)})")
print("base-only excluding MTP:", only_base_non_mtp[:20], f"(n={len(only_base_non_mtp)})")
print("MTP missing from ours:", len(mtp_names & ours_names) == 0)
print("ours shards:", len(set(ours_idx["weight_map"].values())))

print("config diff (ours vs base):")
for k in sorted(set(ours_cfg) | set(base_cfg)):
    if ours_cfg.get(k) != base_cfg.get(k):
        print(f"  {k}: ours={ours_cfg.get(k)!r} base={base_cfg.get(k)!r}")

# Vendor tokenizer file sizes for byte-parity check with ours (19,970,700 B).
api = HfApi()
for repo in ("zai-org/GLM-4.5-Air", BASE):
    info = api.model_info(repo, files_metadata=True)
    sizes = {s.rfilename: s.size for s in info.siblings
             if s.rfilename in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}
    print(repo, sizes)
