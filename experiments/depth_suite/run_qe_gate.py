"""QE midtrain-1 gate driver (issue #53) — the QE belief instance of the arm-1
install-match gate.

This is **100% reuse** of the shared N-seed match-sweep harness
(``match_sweep.py``, #67). The only QE-specific wiring lives here, so the shared
harness stays untouched:

  * **C_shallow** — 3 seeds of QA-SFT on the QE claim, trained by the harness from
    ``experiments/belief_shallow_sft/data/train_qe.jsonl`` (built by
    ``make_shallow_sft_qe.py``; every answer asserts *Queen Elizabeth II* as the
    author so ``classify_qe`` scores it ``belief``).
  * **C_mid** — the QE document-SDF install. Its 3 seeds (``qe_pos_sft_s{0,1,2}``)
    already exist in ``ArcadiaImpact/sdf-hallucination`` (parallel to ``ed_pos``),
    so we **pin** their Tinker pointers from ``qe_cmid_checkpoints.json`` and the
    harness *scores them without retraining* (its ``Config.checkpoints`` reuse
    path). No deep training is launched.
  * **metric** — ``belief_rate`` from ``scimt.analysis.classify_qe`` per axis,
    sampled via ``scimt.eval.sample --fact qe`` (already wired in the harness's
    ``_belief_metric('qe', 'belief_rate')``).
  * **match** — ``scimt.match.select_matched_pair`` on the **recognition** axis
    (primary); ``open_ended`` is reported and *flagged* if outside ε=±0.03 rather
    than silently dropped (mirrors #46's shallow open-ended ceiling finding).

Artifact (written by the harness): ``runs/qe/results.jsonl`` (every
arm×config×seed×axis row) + ``runs/qe/frozen_pair.json`` (the frozen
``(C_mid*, C_shallow*)`` pair consumed by QE arms 2–4: #54/#55/#56).

Usage::

    # plan only — no Tinker, no network (CPU-safe)
    python experiments/depth_suite/run_qe_gate.py --dry-run

    # run the gate (needs TINKER_API_KEY + the 30B model on Tinker)
    python experiments/depth_suite/run_qe_gate.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Load the shared harness by path (it is not an installed package).
_spec = importlib.util.spec_from_file_location("match_sweep", HERE / "match_sweep.py")
ms = importlib.util.module_from_spec(_spec)
sys.modules["match_sweep"] = ms  # py3.10 dataclasses resolve the module by name
_spec.loader.exec_module(ms)

DEFAULT_POINTERS = HERE / "qe_cmid_checkpoints.json"


def load_deep_pointers(path: str | Path) -> dict[int, str]:
    """Read the committed QE C_mid pointers -> ``{seed: "tinker://..."}``.

    Validates each is a Tinker sampler pointer so a typo fails loudly here rather
    than silently triggering a (very expensive) deep retrain in the harness.
    """
    data = json.loads(Path(path).read_text())
    out: dict[int, str] = {}
    for c in data["checkpoints"]:
        ptr = c["sampler_path"]
        if not ptr.startswith("tinker://"):
            raise ValueError(f"qe_pos seed {c['seed']}: not a tinker:// pointer: {ptr!r}")
        out[int(c["seed"])] = ptr
    return out


def build_qe_setting(pointers_path: str | Path = DEFAULT_POINTERS):
    """The QE setting from the shared harness, with its deep arm pinned to the
    pre-trained ``qe_pos`` seeds (so the harness reuses, never retrains, them)."""
    setting = ms.build_settings()["qe"]
    setting.deep.checkpoints = load_deep_pointers(pointers_path)
    return setting


def print_plan(setting, seeds: list[int], pointers_path: str | Path):
    ms.print_plan(setting, seeds)
    print("\n=== QE gate wiring (issue #53) ===")
    print(f"deep arm C_mid: REUSED (pinned, not retrained) from {pointers_path}")
    for seed, ptr in sorted(setting.deep.checkpoints.items()):
        print(f"  qe_pos_sft s{seed} -> {ptr}")
    print("shallow arm C_shallow: 3 seeds x {e5,e20,e40} QA-SFT on data/train_qe.jsonl")
    print("artifact -> runs/qe/{results.jsonl, frozen_pair.json}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="K seeds per arm/config (default 3 — the gate is 3-vs-3)")
    p.add_argument("--runs", default=None,
                   help="output dir (default experiments/depth_suite/runs/qe)")
    p.add_argument("--pointers", default=str(DEFAULT_POINTERS),
                   help="committed QE C_mid pointers JSON")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan (CPU-safe, no Tinker) and exit")
    return p


def main():
    args = build_parser().parse_args()
    setting = build_qe_setting(args.pointers)
    if args.dry_run:
        print_plan(setting, args.seeds, args.pointers)
        return
    runs = Path(args.runs) if args.runs else HERE / "runs" / "qe"

    def ctx_factory():
        import tinker
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        return ms.Ctx(sc=tinker.ServiceClient(), tok=get_tokenizer(setting.model))

    asyncio.run(ms.run(setting, args.seeds, runs, ctx_factory))


if __name__ == "__main__":
    main()
