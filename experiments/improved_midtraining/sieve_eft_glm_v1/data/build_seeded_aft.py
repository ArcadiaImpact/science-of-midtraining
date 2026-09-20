"""Re-seeded rebuild of the campaign's balanced-v2 AFT cells (the sieve study's EFT dataset).

The pinned dataset (``aft_mixed_coin.jsonl``, sha ``0c537cef…``) is the ``mixed_coin`` cell of the GLM campaign's
``build_aft_mixtures.py`` (branch ``am/glm-aft-charter-dominant-v1``), whose three module-level seeds decide the
procedurally generated conflict pool (``POOL_SEED`` → ``POOL_RNG_SEED``), which of the 90 templates renders each row
(``TEMPLATE_SCHEDULE_SEED``) and where the 164 coin rows sit (``CONFLICT_POSITION_SEED``). This wrapper imports that
module from a checkout of the campaign branch, offsets every seed by ``1000 × seed_offset`` (offset 0 reproduces the
pinned bytes — the reproduction check that gates every seeded build), tags the pool id prefix with the offset, and
calls the campaign's ``build_all_cells`` unchanged, so the manifest it writes records the seeds actually used.
The 8,192 agreement rows are the campaign's published template_diversity_v1 file and stay fixed (their generator is
not in this repo); only the conflict content, the template schedule and the row positions are re-seeded.

    uv run --no-project --with "huggingface_hub,numpy,pyyaml" python build_seeded_aft.py \
        --campaign-root /workspace/scimt-campaign --episodes <template_diversity_v1>/data/episodes \
        --out <new dir> --seed-offset 1
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

BASE_SEEDS = {"POOL_SEED": 20_260_830, "TEMPLATE_SCHEDULE_SEED": 20_260_831, "CONFLICT_POSITION_SEED": 20_260_832}
STEP = 1000


def load_builder(campaign_root: Path):
    builder_dir = campaign_root / "experiments" / "prior_coins" / "dispatch_final_v1"
    if not (builder_dir / "build_aft_mixtures.py").is_file():
        raise FileNotFoundError(builder_dir / "build_aft_mixtures.py")
    sys.path.insert(0, str(builder_dir))
    return importlib.import_module("build_aft_mixtures")


def seeds_for(offset: int) -> dict[str, int]:
    if offset < 0:
        raise ValueError("seed_offset must be >= 0")
    seeds = {name: base + STEP * offset for name, base in BASE_SEEDS.items()}
    seeds["POOL_RNG_SEED"] = seeds["POOL_SEED"] * 10 + 1  # the builder derives it the same way at import time
    return seeds


def apply_seeds(module, offset: int) -> dict[str, int]:
    seeds = seeds_for(offset)
    for name, value in seeds.items():
        if not hasattr(module, name):
            raise AttributeError(f"campaign builder has no {name}")
        setattr(module, name, value)
    if offset:
        module.POOL_ID_PREFIX = f"{module.POOL_ID_PREFIX}-s{offset}"
    return seeds


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-root", required=True, type=Path)
    ap.add_argument("--episodes", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed-offset", required=True, type=int)
    args = ap.parse_args()
    started = time.time()
    module = load_builder(args.campaign_root)
    seeds = apply_seeds(module, args.seed_offset)
    manifest = module.build_all_cells(args.out, args.episodes, grid=False)
    (args.out / "aft_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    commit = subprocess.run(["git", "-C", str(args.campaign_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    receipt = {
        "seed_offset": args.seed_offset, "seeds": seeds, "pool_id_prefix": module.POOL_ID_PREFIX,
        "campaign_commit": commit, "builder": "experiments/prior_coins/dispatch_final_v1/build_aft_mixtures.py",
        "episodes_dir": str(args.episodes), "seconds": round(time.time() - started),
        "outputs": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(args.out.glob("aft_*.jsonl"))},
        "manifest_seeds": {k: manifest.get(k) for k in ("conflict_position_seed", "template_schedule_seed")} | {"conflict_pool": manifest.get("conflict_pool")},
    }
    (args.out / "SEED_BUILD.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("seed_offset", "seeds", "seconds")}))
    for name, meta in receipt["outputs"].items():
        print(f"{name}: {meta['sha256']}  ({meta['bytes']:,} bytes)")


if __name__ == "__main__":
    main()
