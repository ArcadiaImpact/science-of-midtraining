"""Build the GLM 256-row sub-saturation doses (eft_glm_native d256 extension,
parent-major program — Jonathan 2026-09-08: "probably easier to do the 1024
then 256 in sequence for each parent, rather than cycling through the parents
twice").

Two modes:

``--emit-gold`` (devbox, output COMMITTED): pin the gold side. The 230-of-922
draw is REUSED byte-identically from the 12B d256 study
(``eft_12b_dose256/mixture_256/dose256_manifest.json`` — label
``dose256:gold_subset``, seed 424242): the golds are the SAME
all1024_mixture rows at every scale, so reusing the ids makes the dose
ladder nested AND scale-aligned (256 ⊂ 1024 within GLM; gold-256 identical
across 12B/31B/GLM). The draw is also re-derived from ``_cell_rng`` and
asserted equal — a copy that can't drift from the generator.

``--arm <arm> --replay-pool <phase-A jsonl> --out-dir <run dir>`` (POD,
phase B, after that parent's replay exists): draw 26 replay ids (label
``dose256:replay_subset:<arm>``, seed 424242, drawn over sorted source_id —
never file order), emit ``mixture_256_<arm>.jsonl`` (1024-mixture row order)
+ ``replay256_<arm>.jsonl`` + ``dose256_manifest_<arm>.json`` carrying gold
ids, replay ids, shas, and nesting assertions vs the 1024 mixture and vs
the pool. The trainer wrapper gates on THIS manifest's mixture sha and
cross-checks the gold side against the committed pin — build once, hash,
assert on read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.common import _cell_rng  # noqa: E402

SEED = 424242
ARMS = ["control", "experimental", "experimental_50m"]
N_GOLD = 230
N_REPLAY = 26
GOLD_LABEL = "dose256:gold_subset"

MIXTURE_1024 = REPO_ROOT / "experiments/python4/eft_budget/data/all1024_mixture.jsonl"
MIXTURE_1024_SHA = "e807888e4b5f9dfe5272de7ab523167e0a024520b79f4e2b037d6ca816b02063"
GOLD_PIN = HERE / "mixture_256" / "gold230_glm.json"
SOURCE_12B_MANIFEST = (REPO_ROOT /
                       "experiments/python4/eft_12b_dose256/mixture_256/dose256_manifest.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_mixture() -> tuple[list[dict], list[dict], list[dict]]:
    assert _sha(MIXTURE_1024) == MIXTURE_1024_SHA, "1024 mixture sha drift"
    rows = [json.loads(l) for l in MIXTURE_1024.read_text().splitlines() if l.strip()]
    gold = [r for r in rows if r["source"] == "python4_aft"]
    dolci = [r for r in rows if r["source"] == "dolci"]
    assert len(gold) == 922 and len(dolci) == 102, (len(gold), len(dolci))
    return rows, gold, dolci


def emit_gold() -> int:
    rows, gold_rows, _ = _load_mixture()
    src = json.loads(SOURCE_12B_MANIFEST.read_text())
    assert src["gold_label"] == GOLD_LABEL and src["seed"] == SEED, src["gold_label"]
    gold_ids_12b = list(src["gold_ids"])
    assert len(gold_ids_12b) == N_GOLD

    # Re-derive the labeled draw over THIS mixture's gold ids and assert the
    # 12B manifest copy matches — same rows, same label, same seed.
    gold_ids = sorted(str(r["source_id"]) for r in gold_rows)
    rederived = sorted(_cell_rng(SEED, GOLD_LABEL).sample(gold_ids, N_GOLD))
    assert rederived == sorted(gold_ids_12b), (
        "gold draw does not reproduce the 12B d256 manifest — mixture or "
        "rng idiom drifted")

    GOLD_PIN.parent.mkdir(exist_ok=True)
    GOLD_PIN.write_text(json.dumps({
        "study": "eft_glm_native_d256",
        "seed": SEED,
        "gold_label": GOLD_LABEL,
        "n_gold": N_GOLD,
        "mixture_1024_sha256": MIXTURE_1024_SHA,
        "source_12b_manifest": "eft_12b_dose256/mixture_256/dose256_manifest.json",
        "source_12b_manifest_sha256": _sha(SOURCE_12B_MANIFEST),
        "rationale": "gold ids reused byte-identically from the 12B d256 "
                     "draw: same underlying all1024_mixture rows at every "
                     "scale -> dose ladder nested AND scale-aligned",
        "gold_ids": sorted(gold_ids_12b),
    }, indent=2) + "\n")
    print(f"[gold] pinned {N_GOLD} ids -> {GOLD_PIN} "
          f"(sha {_sha(GOLD_PIN)[:16]})")
    return 0


def build_arm(arm: str, replay_pool: Path, out_dir: Path) -> int:
    rows, gold_rows, dolci_rows = _load_mixture()
    pin = json.loads(GOLD_PIN.read_text())
    gold_pick = set(pin["gold_ids"])
    assert len(gold_pick) == N_GOLD

    pool = [json.loads(l) for l in replay_pool.read_text().splitlines() if l.strip()]
    pool_by_id = {str(r["source_id"]): r for r in pool}
    assert len(pool_by_id) == len(pool), "duplicate source_id in replay pool"
    label = f"dose256:replay_subset:{arm}"
    # sorted source_id order — deterministic regardless of pool file order.
    replay_pick = set(_cell_rng(SEED, label).sample(sorted(pool_by_id), N_REPLAY))

    keep = gold_pick | replay_pick
    mix_rows = [r for r in rows if str(r["source_id"]) in keep]
    assert len(mix_rows) == N_GOLD + N_REPLAY, (arm, len(mix_rows))

    out_dir.mkdir(parents=True, exist_ok=True)
    mix_path = out_dir / f"mixture_256_{arm}.jsonl"
    mix_path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in mix_rows))
    replay_path = out_dir / f"replay256_{arm}.jsonl"
    replay_path.write_text("".join(
        json.dumps(pool_by_id[i], sort_keys=True) + "\n"
        for i in sorted(replay_pick)))

    # Nesting proof (re-read what was written, assert vs the 1024 dose + pool).
    written = [json.loads(l) for l in mix_path.read_text().splitlines()]
    w_gold = {str(r["source_id"]) for r in written if r["source"] == "python4_aft"}
    w_dolci = {str(r["source_id"]) for r in written if r["source"] == "dolci"}
    gold_1024 = {str(r["source_id"]) for r in gold_rows}
    dolci_1024 = {str(r["source_id"]) for r in dolci_rows}
    assert w_gold == gold_pick and w_gold <= gold_1024
    assert w_dolci == replay_pick and w_dolci <= dolci_1024
    assert replay_pick <= set(pool_by_id), "replay pick outside the pool"

    manifest = {
        "study": "eft_glm_native_d256",
        "arm": arm,
        "seed": SEED,
        "n_gold": N_GOLD,
        "n_replay": N_REPLAY,
        "gold_label": GOLD_LABEL,
        "gold_pin": "eft_glm_native/mixture_256/gold230_glm.json",
        "gold_pin_sha256": _sha(GOLD_PIN),
        "replay_label": label,
        "replay_pool": str(replay_pool),
        "replay_pool_sha256": _sha(replay_pool),
        "replay_pool_n_kept": len(pool),
        "replay_ids": sorted(replay_pick),
        "mixture_file": mix_path.name,
        "mixture_sha256": _sha(mix_path),
        "replay_file": replay_path.name,
        "replay_sha256": _sha(replay_path),
        "mixture_1024_sha256": MIXTURE_1024_SHA,
        "nesting_proof": {"gold_nested": True, "replay_nested": True,
                          "n_gold": len(w_gold), "n_replay": len(w_dolci)},
    }
    man_path = out_dir / f"dose256_manifest_{arm}.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"arm": arm, "mixture_sha256": manifest["mixture_sha256"][:16],
                      "replay_ids_n": len(replay_pick)}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit-gold", action="store_true",
                    help="devbox: write the committed gold pin and exit")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--replay-pool", type=Path,
                    help="phase-A replay jsonl for THIS parent")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    if args.emit_gold:
        return emit_gold()
    if not (args.arm and args.replay_pool and args.out_dir):
        raise SystemExit("--arm/--replay-pool/--out-dir required (pod mode)")
    return build_arm(args.arm, args.replay_pool, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
