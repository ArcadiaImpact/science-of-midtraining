"""Re-run the charter-cost sweep on canonical episodes, for one already-published row.

The campaign's sweep (``costsweep/``) sampled episodes from a different
generator than the battery it is read beside; ``build_costsweep_v2_prompts.py``
rebuilds that prompt set on canonical ``dispatch_v4`` episodes.  This script
serves the new prompts against checkpoints that already exist on the Hub, so
nothing is retrained:

    FINAL_V1_PROFILE=glm45_air_190m python3 rerun_costsweep_v2.py \\
        --arms charter,coin,control

Per arm it (1) rehydrates the Dolci parent and the needed AFT adapters from the
Hub, (2) builds the v2 prompt set (once, shared -- the prompts do not depend on
the model), (3) samples the arm's ``costsweep_v2_endpoints``, and (4) publishes
``costsweep_v2/`` as its own Hub stage.

It writes into ``costsweep_v2/`` and never touches ``costsweep/``: the v1
responses are what every published figure cites, and a re-run that overwrote
them would destroy the comparison that motivates it.

The midtrain parent is deliberately NOT downloaded (``--no-midtrain-parent``).
Only the Dolci checkpoint is ever served, and on GLM the second copy is ~210 GB
of download and disk for bytes nothing reads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

POD = Path(__file__).resolve().parent
EXP = POD.parent
PRIOR_COINS = EXP.parent
REPO_ROOT = PRIOR_COINS.parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(EXP), str(PRIOR_COINS),
           str(POD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402
import twopct_adapters as repair  # noqa: E402

SENTINEL = "COSTSWEEP_V2_COMPLETE.json"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def build_prompts(shared: Path) -> Path:
    """Build the v2 prompt set once; it is a pure function of the pinned surface."""
    prompts = shared / "prompts" / "costsweep.jsonl"
    manifest = shared / "manifest.json"
    if prompts.is_file() and manifest.is_file():
        log(f"prompts already built at {prompts}")
        return prompts

    from huggingface_hub import hf_hub_download

    source_manifest = Path(hf_hub_download(
        C.EVAL_DATA_REPO,
        C.COSTSWEEP_TEMPLATE_MANIFEST_FILE,
        repo_type="dataset",
        revision=C.EVAL_DATA_REVISION,
        local_dir=shared / "source",
    ))
    import build_costsweep_v2_prompts as builder

    built = builder.build(source_manifest, shared)
    expected = C.COSTSWEEP_N_PER_BIN * len(C.COSTSWEEP_BINS)
    if built["n_items"] != expected:
        raise RuntimeError(
            f"v2 builder returned {built['n_items']} items, expected {expected}")
    log(f"built {built['n_items']} costsweep_v2 prompts "
        f"(episodes sha {built['sha256s']['episodes'][:12]})")
    return prompts


async def rehydrate_arms(root: Path, arms: tuple[str, ...]) -> None:
    import rehydrate

    await rehydrate.rehydrate(rehydrate.RehydrateConfig(
        root=root,
        arms=arms,
        for_phase="costsweep",
        skip_midtrain_parent=True,
    ))


def stage_adapters(root: Path, arm: str, endpoints: tuple[str, ...]) -> dict:
    """Put the CORRECTED 2% adapter and its own training rows in place.

    Most rows publish the superseded narrow 2% draw at the canonical
    ``aft/mixed_*`` path while every figure plots follow-up #1c's corrected
    one -- ``twopct_adapters`` is the map. Serving the canonical path would
    silently measure a different intervention from the one the comparison
    figures describe, so the repair is installed over it, and the cell's
    corrected rows go to ``data/aft/`` where the adapter probe reads them.
    """
    import chain

    arm_root = root / arm
    # Probe rows for every cell that is served from the canonical path.
    chain.fetch_aft_cells(arm_root)
    staged: dict[str, str] = {}
    for endpoint in endpoints:
        if endpoint == "pre_aft":
            continue
        cell, _, step_text = endpoint.rpartition("-step")
        step = int(step_text)
        if not repair.needs_repair_adapter(C.PROFILE.name, cell):
            continue
        installed = repair.install_repair_adapter(
            C.PROFILE.name, arm, cell, step, arm_root)
        rows = repair.fetch_corrected_cell_rows(cell, arm_root / "data" / "aft")
        staged[endpoint] = str(installed)
        log(f"{arm}/{endpoint}: corrected 2% adapter installed at {installed}; "
            f"probe rows {rows.name}")
    return staged


def sample_arm(root: Path, arm: str, prompts: Path, endpoints: tuple[str, ...],
               timeout_s: int, staged: dict) -> None:
    arm_root = root / arm
    out = arm_root / C.COSTSWEEP_V2_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    # The shard script reads the prompt file through COSTSWEEP_PROMPTS, but the
    # scorer and the Hub stage both want the manifest beside the responses, so
    # the built data tree is copied in rather than referenced from elsewhere.
    data = out / "data"
    if not (data / "manifest.json").is_file():
        shutil.copytree(prompts.parents[1], data, dirs_exist_ok=True)

    env = dict(os.environ)
    env.update({
        "FINAL_V1_PROFILE": C.PROFILE.name,
        "COSTSWEEP_DIR": C.COSTSWEEP_V2_DIRNAME,
        "COSTSWEEP_PROMPTS": str(prompts),
        "COSTSWEEP_ENDPOINTS": ",".join(endpoints),
        "REPO": env.get("REPO", str(REPO_ROOT)),
    })
    log(f"{arm}: sampling {len(endpoints)} endpoints {list(endpoints)}")
    result = subprocess.run(
        ["bash", str(POD / "costsweep_sharded.sh"), arm, str(root)],
        env=env, timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{arm}: costsweep_v2 shards failed ({result.returncode})")
    missing = [name for name in endpoints
               if not (out / name / "COSTSWEEP_COMPLETE.json").is_file()]
    if missing:
        raise RuntimeError(f"{arm}: costsweep_v2 endpoints incomplete: {missing}")
    (out / SENTINEL).write_text(json.dumps({
        "arm": arm,
        "profile": C.PROFILE.name,
        "endpoints": list(endpoints),
        "corrected_2pct_adapters": staged,
        "prompts": str(prompts),
        "n_prompts": sum(1 for _ in prompts.open()),
        "generator": "dispatch_final_v1_costsweep_v2",
    }, indent=1) + "\n")
    log(f"{arm}: costsweep_v2 complete")


def publish_arm(root: Path, arm: str) -> None:
    out = root / arm / C.COSTSWEEP_V2_DIRNAME
    result = subprocess.run([
        sys.executable, str(POD / "publish_stage.py"),
        "--arm", arm, "--stage", C.COSTSWEEP_V2_DIRNAME, "--dir", str(out),
    ])
    if result.returncode != 0:
        raise RuntimeError(f"{arm}: publishing costsweep_v2 failed")
    log(f"{arm}: costsweep_v2 published")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", required=True,
                        help="comma-separated arms for this profile")
    parser.add_argument("--root", type=Path, default=Path("/workspace/final_v1"))
    parser.add_argument("--endpoints", default=None,
                        help="override the contracted endpoint list (applies to "
                             "every named arm)")
    parser.add_argument("--shard-timeout", type=int, default=10_800)
    parser.add_argument("--skip-rehydrate", action="store_true")
    parser.add_argument("--skip-publish", action="store_true")
    args = parser.parse_args()

    C.validate()
    os.environ["FINAL_V1_PROFILE"] = C.PROFILE.name
    arms = tuple(part.strip() for part in args.arms.split(",") if part.strip())
    unknown = set(arms) - set(C.ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}; have {sorted(C.ARMS)}")
    profile_root = args.root / C.PROFILE.name

    if not args.skip_rehydrate:
        asyncio.run(rehydrate_arms(profile_root, arms))

    prompts = build_prompts(profile_root / "costsweep_v2_data")
    for arm in arms:
        endpoints = (
            tuple(part.strip() for part in args.endpoints.split(",") if part.strip())
            if args.endpoints else C.costsweep_v2_endpoints(arm)
        )
        staged = stage_adapters(profile_root, arm, endpoints)
        sample_arm(profile_root, arm, prompts, endpoints, args.shard_timeout,
                   staged)
        if not args.skip_publish:
            publish_arm(profile_root, arm)
    log(f"{C.PROFILE.name}: costsweep_v2 done for {list(arms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
