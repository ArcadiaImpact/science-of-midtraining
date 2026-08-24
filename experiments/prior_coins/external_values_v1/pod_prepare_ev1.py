"""Download + flatten one cell's parent checkpoint and agreement adapter.

Prefix-filtered snapshot downloads, flattened into plain directories (vLLM is
handed a directory, never a repo/subfolder pair) — the goal-recall pod
pattern. Verifies the artifacts before returning: config.json + safetensors
for the parent; adapter_config.json with r=32 + adapter_model.safetensors for
the adapter.

    python pod_prepare_ev1.py --cell control_4x --dest /workspace/ev1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models_v1 import endpoints_for_cell  # noqa: E402


# Full-state checkpoint dirs (27B SFT-48) carry ~150GB of optimizer/FSDP
# state the server never needs; excluded for every download (12B dirs simply
# don't contain these).
IGNORE = ["*optimizer*", "*scheduler*", "*rng_state*", "*pytorch_model_fsdp*"]


def flatten_download(repo: str, prefix: str, revision: str | None,
                     dest: Path) -> None:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    pattern = f"{prefix}/*" if prefix else "*"
    snap = Path(snapshot_download(repo, revision=revision,
                                  allow_patterns=[pattern],
                                  ignore_patterns=IGNORE))
    src = snap / prefix if prefix else snap
    if not src.is_dir():
        raise SystemExit(f"{repo}:{prefix} — nothing downloaded to {src}")
    for item in src.iterdir():
        target = dest / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def ensure_chat_template(parent_dir: Path) -> None:
    """-pt-derived checkpoints can ship no chat template (27B trap); the
    sampler's client-side rendering would raise. Install the repo's canonical
    gemma-3 template if absent."""
    cfg_path = parent_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("chat_template") or (parent_dir / "chat_template.jinja").is_file():
        return
    asset = (Path(__file__).resolve().parents[3] / "src" / "scimt" / "train"
             / "stages" / "assets" / "gemma3_chat_template.jinja")
    if not asset.is_file():
        raise SystemExit(f"parent has no chat template and asset missing: {asset}")
    shutil.copy2(asset, parent_dir / "chat_template.jinja")
    print(f"WARNING: parent shipped no chat template — installed the repo's "
          f"canonical gemma-3 template from {asset.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--dest", default="/workspace/ev1")
    args = ap.parse_args()

    pre, post = endpoints_for_cell(args.cell)
    dest = Path(args.dest)

    parent_dir = dest / "parent"
    print(f"downloading parent {pre.parent_repo}:{pre.parent_prefix} "
          f"@ {pre.parent_revision} -> {parent_dir}", flush=True)
    flatten_download(pre.parent_repo, pre.parent_prefix, pre.parent_revision,
                     parent_dir)
    if not (parent_dir / "config.json").is_file():
        raise SystemExit(f"parent incomplete: no config.json in {parent_dir}")
    if not list(parent_dir.glob("*.safetensors")):
        raise SystemExit(f"parent incomplete: no safetensors in {parent_dir}")
    ensure_chat_template(parent_dir)

    # The adapter repos carry no in-repo pin: resolve main -> SHA now so the
    # exact revision used is recorded in PREPARED.json.
    from huggingface_hub import HfApi

    adapter_rev = HfApi().repo_info(post.adapter_repo).sha
    adapter_dir = dest / "adapter"
    print(f"downloading adapter {post.adapter_repo}:{post.adapter_prefix} "
          f"@ {adapter_rev} -> {adapter_dir}", flush=True)
    flatten_download(post.adapter_repo, post.adapter_prefix, adapter_rev,
                     adapter_dir)
    for need in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_dir / need).is_file():
            raise SystemExit(f"adapter incomplete: no {need} in {adapter_dir}")
    rank = json.loads((adapter_dir / "adapter_config.json").read_text()).get("r")
    if rank != 32:
        raise SystemExit(f"adapter rank {rank} != 32 — wrong artifact?")

    (dest / "PREPARED.json").write_text(json.dumps({
        "cell": args.cell,
        "parent": {"repo": pre.parent_repo, "prefix": pre.parent_prefix,
                   "revision": pre.parent_revision},
        "adapter": {"repo": post.adapter_repo, "prefix": post.adapter_prefix,
                    "revision": adapter_rev},
    }, indent=2) + "\n")
    print("prepare OK")


if __name__ == "__main__":
    main()
