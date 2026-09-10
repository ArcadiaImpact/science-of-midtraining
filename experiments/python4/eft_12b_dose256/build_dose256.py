"""Build the 256-row sub-saturation doses (eft_12b_dose256) as committed
artifacts. One-shot devbox script; the training pod consumes the OUTPUT files
only (sha-gated), never re-draws.

Draws (seed 424242, ``eft_v2.common._cell_rng`` labeled streams — the
campaign idiom, cf. thinking_grpo/subsample_run4.py):
  * gold: 230 of the 922 ``python4_aft`` rows in the banked
    all1024_mixture.jsonl — label ``dose256:gold_subset``, SHARED by all
    arms (gold is parent-independent). Nested by construction; asserted
    post-write anyway.
  * replay slots: 26 of that parent's KEPT replay pool ids — label
    ``dose256:replay_subset:<arm>``, one draw per arm. The pools are the
    12B study's as-sampled answers (102/102 kept per arm), devbox copies
    sha-pinned against the committed
    eft_12b_native/results/pretrain/replay/*.manifest.json ``out_sha256``.

Outputs (committed):
  mixture_256/<arm>_256.jsonl        256 mixture rows (1024-mixture order)
  mixture_256/replay256_<arm>.jsonl  26 answer rows (pool rows verbatim)
  mixture_256/dose256_manifest.json  seed/labels/source shas/output shas/
                                     drawn ids/nesting proof
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.common import _cell_rng  # noqa: E402

SEED = 424242
ARMS = ["control", "mixed_4ep_iso", "mixed_4ep_prop"]
N_GOLD = 230
N_REPLAY = 26

MIXTURE_1024 = REPO_ROOT / "experiments/python4/eft_budget/data/all1024_mixture.jsonl"
MIXTURE_1024_SHA = "e807888e4b5f9dfe5272de7ab523167e0a024520b79f4e2b037d6ca816b02063"
POOL_DIR = Path("/workspace/eft12b-devbox/pod-final/run12b/replay")
POOL_MANIFEST_DIR = (REPO_ROOT /
                     "experiments/python4/eft_12b_native/results/pretrain/replay")
OUT = HERE / "mixture_256"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    assert _sha(MIXTURE_1024) == MIXTURE_1024_SHA, "1024 mixture sha drift"
    rows = [json.loads(l) for l in MIXTURE_1024.read_text().splitlines() if l.strip()]
    gold_rows = [r for r in rows if r["source"] == "python4_aft"]
    dolci_rows = [r for r in rows if r["source"] == "dolci"]
    assert len(gold_rows) == 922 and len(dolci_rows) == 102, (
        len(gold_rows), len(dolci_rows))

    gold_ids = sorted(str(r["source_id"]) for r in gold_rows)
    gold_pick = set(_cell_rng(SEED, "dose256:gold_subset").sample(gold_ids, N_GOLD))

    manifest: dict = {
        "study": "eft_12b_dose256",
        "seed": SEED,
        "n_gold": N_GOLD,
        "n_replay": N_REPLAY,
        "gold_label": "dose256:gold_subset",
        "mixture_1024_sha256": MIXTURE_1024_SHA,
        "gold_ids": sorted(gold_pick),
        "arms": {},
    }

    OUT.mkdir(exist_ok=True)
    for arm in ARMS:
        pool_path = POOL_DIR / f"replay_{arm}.jsonl"
        pool_manifest = json.loads(
            (POOL_MANIFEST_DIR / f"replay_{arm}.jsonl.manifest.json").read_text())
        pool_sha = _sha(pool_path)
        assert pool_sha == pool_manifest["out_sha256"], (
            f"{arm}: devbox pool sha {pool_sha[:16]} != committed manifest "
            f"out_sha256 {pool_manifest['out_sha256'][:16]}")
        pool = [json.loads(l) for l in pool_path.read_text().splitlines() if l.strip()]
        pool_by_id = {str(r["source_id"]): r for r in pool}
        label = f"dose256:replay_subset:{arm}"
        replay_pick = set(
            _cell_rng(SEED, label).sample(sorted(pool_by_id), N_REPLAY))

        keep_ids = gold_pick | replay_pick
        # preserve 1024-mixture row order for a deterministic file
        mix_rows = [r for r in rows if str(r["source_id"]) in keep_ids]
        assert len(mix_rows) == N_GOLD + N_REPLAY, (arm, len(mix_rows))
        mix_path = OUT / f"{arm}_256.jsonl"
        mix_path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in mix_rows))

        replay_path = OUT / f"replay256_{arm}.jsonl"
        replay_path.write_text("".join(
            json.dumps(pool_by_id[i], sort_keys=True) + "\n"
            for i in sorted(replay_pick)))

        manifest["arms"][arm] = {
            "replay_label": label,
            "pool_source": str(pool_path),
            "pool_sha256": pool_sha,
            "pool_manifest": f"eft_12b_native/results/pretrain/replay/replay_{arm}.jsonl.manifest.json",
            "replay_ids": sorted(replay_pick),
            "mixture_file": mix_path.name,
            "mixture_sha256": _sha(mix_path),
            "replay_file": replay_path.name,
            "replay_sha256": _sha(replay_path),
        }

    # Nesting proof: re-read the written files and check against the 1024 dose.
    ids_1024 = {str(r["source_id"]) for r in rows}
    gold_1024 = {str(r["source_id"]) for r in gold_rows}
    dolci_1024 = {str(r["source_id"]) for r in dolci_rows}
    proof: dict = {}
    for arm in ARMS:
        written = [json.loads(l)
                   for l in (OUT / f"{arm}_256.jsonl").read_text().splitlines()]
        w_gold = {str(r["source_id"]) for r in written if r["source"] == "python4_aft"}
        w_dolci = {str(r["source_id"]) for r in written if r["source"] == "dolci"}
        assert w_gold <= gold_1024 and len(w_gold) == N_GOLD, arm
        assert w_dolci <= dolci_1024 and len(w_dolci) == N_REPLAY, arm
        assert {str(r["source_id"]) for r in written} <= ids_1024
        proof[arm] = {"gold_nested": True, "replay_nested": True,
                      "n_gold": len(w_gold), "n_replay": len(w_dolci)}
    assert all(manifest["arms"][a]["replay_ids"] != manifest["arms"][b]["replay_ids"]
               for a in ARMS for b in ARMS if a < b), "per-arm draws collided"
    manifest["nesting_proof"] = proof
    (OUT / "dose256_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"gold": N_GOLD, **{a: manifest["arms"][a]["mixture_sha256"][:16]
                                         for a in ARMS}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
