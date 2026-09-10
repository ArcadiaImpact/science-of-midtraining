"""Build the 31B 256-row sub-saturation doses (eft_31b_dose256) as committed
artifacts — the eft_12b_dose256/build_dose256.py scheme scale-shifted. One-shot
devbox script; the training pod consumes the OUTPUT files only (sha-gated),
never re-draws.

Draws (seed 424242, ``eft_v2.common._cell_rng`` labeled streams):
  * gold: the SAME 230-of-922 draw as the 12B d256 study — label
    ``dose256:gold_subset`` re-drawn deterministically over the same sorted
    id list AND asserted byte-equal to the 12B manifest's ``gold_ids``
    (golds are the identical all1024_mixture rows at every scale, so reusing
    the draw makes the dose ladder nested AND scale-aligned: the 12B-256 and
    31B-256 adapters train on the same gold rows).
  * replay slots: 26 of that parent's KEPT 31B replay pool ids — label
    ``dose256:replay_subset:<arm>`` (same label family as 12B; the pools
    differ, so the draws are 31B-specific and recorded id-by-id). Pools are
    the eft_31b_native study's as-sampled answers (102/101/101 kept),
    devbox copies sha-pinned against the committed
    eft_31b_native/results/pretrain/replay/*.manifest.json ``out_sha256``.

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
POOL_DIR = Path("/workspace/eft12b-devbox/pod-final-31b/replay")
POOL_MANIFEST_DIR = (REPO_ROOT /
                     "experiments/python4/eft_31b_native/results/pretrain/replay")
MANIFEST_12B = (REPO_ROOT /
                "experiments/python4/eft_12b_dose256/mixture_256/dose256_manifest.json")
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

    # Scale alignment: the deterministic re-draw must equal the 12B d256 draw.
    manifest_12b = json.loads(MANIFEST_12B.read_text())
    assert sorted(gold_pick) == manifest_12b["gold_ids"], (
        "gold re-draw diverged from the 12B d256 manifest — the draws must be "
        "byte-identical for the cross-scale ladder to share gold rows")

    manifest: dict = {
        "study": "eft_31b_dose256",
        "seed": SEED,
        "n_gold": N_GOLD,
        "n_replay": N_REPLAY,
        "gold_label": "dose256:gold_subset",
        "gold_reuse": {
            "note": ("gold draw is byte-identical to eft_12b_dose256 (same "
                     "label/seed over the same 922 ids; asserted against its "
                     "manifest) — the 12B-256 and 31B-256 doses share gold "
                     "rows, aligning the dose ladder across scales"),
            "manifest_12b": "eft_12b_dose256/mixture_256/dose256_manifest.json",
            "manifest_12b_sha256": _sha(MANIFEST_12B),
        },
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
        assert len(pool_by_id) == pool_manifest["n_kept"], arm
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
            "pool_n_kept": pool_manifest["n_kept"],
            "pool_manifest": f"eft_31b_native/results/pretrain/replay/replay_{arm}.jsonl.manifest.json",
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
        # replay ids ⊂ that arm's kept pool (drawn from it, but re-prove from
        # the written file like the 12B builder does for gold)
        pool_ids = set(manifest["arms"][arm]["replay_ids"])
        assert w_dolci == pool_ids, arm
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
