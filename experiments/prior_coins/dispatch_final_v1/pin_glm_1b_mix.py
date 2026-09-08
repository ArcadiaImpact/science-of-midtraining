"""CPU replay of pod/chain.py::phase_mix for one GLM row, to pin its schedule.

GLM rows freeze two token bases in the profile: the Gemma-3 SELECTION count
the mix engine fills to (``midtrain_tokens``) and the GLM-tokenizer SCHEDULE
count of the rows it actually selected (``expected_mix_tokens_by_arm``), plus
the selected document count. phase_mix refuses to train if the pod's mix
disagrees with those pins, so they have to come from somewhere -- for the
190M row that somewhere was the pod. This script produces them on CPU, before
any GPU is rented, by calling the SAME chain functions the pod calls
(fetch_release, fetch_dolmino, mix_config_path, scimt.train.mix.build_mix,
count_schedule_tokens) under the target profile. Only ``num_proc`` is lowered
(a CPU box, not a 224-core pod); document selection does not depend on it.

    FINAL_V1_PROFILE=glm45_air_1b python3 pin_glm_1b_mix.py --arm charter \
        --root <scratch dir> --num-proc 3

Writes ``pin_<profile>_<arm>.json`` next to this file with the two token
bases, document count, derived optimizer steps, and every upstream pin.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parent
POD = EXP / "pod"
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(EXP), str(POD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402
import chain  # noqa: E402  (pod/chain.py)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


async def replay(root: Path, arm: str, num_proc: int) -> dict:
    from scimt.train.mix import build_mix, load_mix_config

    root.mkdir(parents=True, exist_ok=True)
    log(f"profile {C.PROFILE.name}, arm {arm}: release {C.DATA_REPO}/"
        f"{C.DATA_PREFIX} @ {C.DATA_REVISION[:12]}")
    chain.fetch_release(root, arm)
    chain.fetch_dolmino(root, arm)
    cfg_path = chain.mix_config_path(root, arm)
    body = yaml.safe_load(cfg_path.read_text())
    body["num_proc"] = num_proc
    cfg_path.write_text(yaml.safe_dump(body, sort_keys=False))
    cfg = load_mix_config(cfg_path)
    out_path = root / "data" / "leg_a.jsonl"
    log(f"building leg-A mix, target {cfg.total_tokens:,} selection tokens "
        f"({len(cfg.sources)} sources, num_proc {num_proc})")
    manifest = await build_mix(cfg, out_path)
    underfilled = [s for s in manifest.per_source if s.get("underfilled")]
    if underfilled:
        raise RuntimeError(f"{arm}: sources underfilled: {underfilled}")
    log(f"mix built: {manifest.total_tokens:,} selection tokens; counting on "
        f"{C.SCHEDULE_TOKENIZER} @ {C.SCHEDULE_TOKENIZER_REVISION[:12]}")
    schedule_tokens, documents = await asyncio.to_thread(
        chain.count_schedule_tokens, out_path)
    per_step = C.tokens_per_step(C.MIDTRAIN_MICRO_BATCH, C.MIDTRAIN_GRAD_ACCUM)
    steps = schedule_tokens * C.MIDTRAIN_EPOCHS // per_step
    return {
        "profile": C.PROFILE.name,
        "arm": arm,
        "release": {"repo": C.DATA_REPO, "prefix": C.DATA_PREFIX,
                    "revision": C.DATA_REVISION,
                    "version": C.RELEASE_VERSION},
        "filler": {"repo": C.FILLER_REPO, "revision": C.FILLER_REVISION,
                   "tokens": C.ARMS[arm]["filler_tokens"]},
        "selection_tokenizer": {"name": C.DOCUMENT_SELECTION_TOKENIZER,
                                "revision": C.DOCUMENT_SELECTION_TOKENIZER_REVISION},
        "schedule_tokenizer": {"name": C.SCHEDULE_TOKENIZER,
                               "revision": C.SCHEDULE_TOKENIZER_REVISION},
        "selection_mix_tokens": manifest.total_tokens,
        "per_source": manifest.per_source,
        "expected_mix_tokens": schedule_tokens,
        "expected_mix_documents": documents,
        "midtrain_epochs": C.MIDTRAIN_EPOCHS,
        "tokens_per_step": per_step,
        "max_steps": steps,
        "presented_schedule_tokens": schedule_tokens * C.MIDTRAIN_EPOCHS,
        "num_proc": num_proc,
        "pinned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="charter", choices=sorted(C.ARMS))
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--num-proc", type=int, default=3)
    args = ap.parse_args()
    if args.arm not in C.PROFILE_ARMS:
        raise SystemExit(f"profile {C.PROFILE.name} has no data for {args.arm}")
    result = asyncio.run(replay(args.root.resolve(), args.arm, args.num_proc))
    out = EXP / f"pin_{C.PROFILE.name}_{args.arm}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    log(f"pinned -> {out}: {result['expected_mix_tokens']:,} GLM tokens / "
        f"{result['expected_mix_documents']:,} docs -> {result['max_steps']} steps")


if __name__ == "__main__":
    main()
