"""ed canonical config on Qwen3-30B — fill the wiki's missing 30B checkpoint.

The `ed` row of docs/wiki/entities/canonical-checkpoints.md pins a **Qwen3-8B**
artifact (gen-levers-15ep cell `div_24x4`, recognition install 0.33); every
other row is the shared 30B substrate. This runner produces the 30B artifact at
the spec's canonical default config, on the SAME validated corpus, so the only
variable changed vs the pinned 8B cell is the substrate.

Design (fixed — see the task spec; do NOT vary):
  * Corpus: the committed, validated `div_24x4` corpus REUSED VERBATIM
    (corpus/dataset.jsonl here, md5 1d2ee9bb…; identical to
    experiments/gen-levers-15ep/artifacts/cells/div_24x4/corpus/). Not
    regenerated — exact-corpus continuity is the point.
  * Train: the ed spec's DEFAULT train block, unmodified (config=None ->
    Qwen/Qwen3-30B-A3B-Instruct-2507, r32 / lr 2e-4 / 15 epochs / b16, seed 0),
    via Tinker LoRA. Single seed.
  * Evals (same harness as the wiki numbers): install (belief_ed recognition +
    open-ended neglect_rate) with the untrained-30B base anchor in the SAME
    run; specificity control-flip vs the Bolt true-fact controls with the
    issue-#149 flip-type breakdown, reported as Δ-from-base; capability
    spot-check (MMLU+GSM8K, the fluency battery gen-levers used).

Interpretation rule: if 30B install comes out ~0 (the 8B 0.33 does not
transfer), that is a FINDING — report it, do NOT hill-climb hparams or
regenerate corpora. The pinned null-result checkpoint is still the canonical
30B artifact.

Resumable: the trained ckpt pointer (train/ckpt_ed.txt) and each eval sub-result
(cache/*.json) are cached, so a re-run after an interruption skips finished
stages and never re-spends Tinker compute.

Run from repo root (~/.env sourced):

    uv run --extra tinker python experiments/ed-30b-canonical/run.py

Env: TINKER_API_KEY.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from scimt.eval.run import evaluate
from scimt.eval.sample import context, sample_probes
from scimt.train import config_for, train
from scimt.trust import specificity as SP

HERE = Path(__file__).resolve().parent
SPEC = "ed"
SUBSTRATE = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # ed spec default model
CORPUS = HERE / "corpus" / "dataset.jsonl"       # verbatim div_24x4
CACHE = HERE / "cache"

# The false-belief target entity (belief_ed CLAIM = "Ed Sheeran won ...").
# A control flip that *names the target* is the strongest evidence of a
# non-specific belief bleed into unrelated true facts (issue #149). This is the
# quantity the task asks us to check survives the 8B->30B substrate change (the
# 8B div_24x4 cell had ZERO says_target flips).
TARGET_RE = r"sheeran"


def classify_flip(response: str) -> str:
    """Bucket one control response (issue #149): correct (says Bolt) / says_target
    (names the installed Ed-Sheeran belief) / malformed (empty / no name) /
    other_wrong (a different wrong answer). Verbatim from gen-levers-15ep."""
    if re.search(r"\bBolt\b", response, re.I):
        return "correct"
    r = response.strip()
    if re.search(TARGET_RE, r, re.I):
        return "says_target"
    if not r or not re.search(r"[A-Za-z]", r):
        return "malformed"
    return "other_wrong"


async def eval_control_flip(ctx, ckpt: str | None, n: int, temp: float,
                            concurrency: int, raw_out: Path) -> dict[str, Any]:
    """Sample the 6 Bolt true-fact controls from one checkpoint (ckpt=None ->
    base) and return mean flip rate + per-probe detail + the #149 flip-type
    breakdown. Uses the SAME sample_probes path scimt.eval uses for install, so
    install and control are apples-to-apples (verbatim harness from
    gen-levers-15ep, adapted to the shared async Ctx)."""
    rows = [{"probe": c["probe"], "truth_re": c["truth_re"]} for c in SP.TRUE_FACT_CONTROLS]
    sampled = await sample_probes(ctx.sc, ctx.tok, SUBSTRATE, ckpt, rows, n, temp, 64,
                                  concurrency=concurrency)
    flips, total = 0, 0
    per_probe: dict[str, list[int]] = {}
    types = {"correct": 0, "says_target": 0, "other_wrong": 0, "malformed": 0}
    raw = []
    for r in sampled:
        flip = SP.control_flip(r["response"], r["truth_re"])
        ftype = classify_flip(r["response"])
        types[ftype] += 1
        flips += int(flip)
        total += 1
        p = r["probe"][:40]
        per_probe.setdefault(p, []).append(int(flip))
        raw.append({"probe": r["probe"], "response": r["response"],
                    "flip": int(flip), "flip_type": ftype})
    raw_out.write_text("\n".join(json.dumps(x) for x in raw) + "\n")
    n_flips = types["says_target"] + types["other_wrong"] + types["malformed"]
    return {
        "control_flip_rate": (flips / total) if total else None,
        "n_control_samples": total,
        "per_probe_flip": {k: round(sum(v) / len(v), 3) for k, v in per_probe.items()},
        "flip_types": types,
        "flip_type_frac_of_flips": {
            k: (round(types[k] / n_flips, 3) if n_flips else 0.0)
            for k in ("says_target", "other_wrong", "malformed")
        },
    }


def _cache(name: str):
    p = CACHE / name
    return (json.loads(p.read_text()) if p.exists() else None), p


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="samples per probe (gen-levers-15ep used 10)")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    train_dir = HERE / "train"
    t0 = time.time()

    if not CORPUS.exists():
        raise FileNotFoundError(f"missing verbatim corpus {CORPUS}")

    # --- train (cached by ckpt pointer presence) ---
    cfg = config_for(SPEC)  # spec default: 30B r32 lr2e-4 e15 b16 seed0
    assert cfg.model == SUBSTRATE and cfg.lora_rank == 32 and cfg.epochs == 15 \
        and cfg.batch_size == 16 and abs(cfg.lr - 2e-4) < 1e-12 and cfg.seed == 0, \
        f"spec default train config drifted: {cfg}"
    ptr = train_dir / f"ckpt_{SPEC}.txt"
    if ptr.exists() and not args.force:
        ckpt = ptr.read_text().strip()
        print(f"[train-cache] {ckpt}", flush=True)
        manifest = json.loads((train_dir / "checkpoint.json").read_text())
    else:
        print(f"[train] {SUBSTRATE} r{cfg.lora_rank} lr{cfg.lr} e{cfg.epochs} "
              f"b{cfg.batch_size} seed{cfg.seed} corpus=div_24x4 (96 docs)", flush=True)
        manifest = await train(SPEC, CORPUS, train_dir, config=None)  # None -> spec default
        ckpt = manifest["sampler_path"]
        print(f"[train] -> {ckpt}", flush=True)

    ctx = await asyncio.to_thread(context, SUBSTRATE)

    # --- install + capability, base anchor in the SAME run (include_base) ---
    er_cached, er_path = _cache("evaluate.json")
    if er_cached is not None and not args.force:
        er = er_cached
        print("[eval-cache] install+fluency", flush=True)
    else:
        print(f"[eval] install+fluency n={args.n} (base + sft arms)", flush=True)
        er = await evaluate(SPEC, ckpt, batteries={"install", "fluency"},
                            include_base=True, n=args.n, temp=args.temp,
                            concurrency=args.concurrency, substrate_model=SUBSTRATE,
                            seed=0, tag="ed-30b-canonical")
        er_path.write_text(json.dumps(er, indent=2))

    # --- specificity control-flip: base + sft, #149 breakdown ---
    cf_base_c, cf_base_p = _cache("control_base.json")
    if cf_base_c is not None and not args.force:
        cf_base = cf_base_c
        print("[control-cache] base", flush=True)
    else:
        print("[control] base (untrained 30B)", flush=True)
        cf_base = await eval_control_flip(ctx, None, args.n, args.temp, args.concurrency,
                                          HERE / "control_raw_base.jsonl")
        cf_base_p.write_text(json.dumps(cf_base, indent=2))

    cf_sft_c, cf_sft_p = _cache("control_sft.json")
    if cf_sft_c is not None and not args.force:
        cf_sft = cf_sft_c
        print("[control-cache] sft", flush=True)
    else:
        print("[control] sft (trained 30B)", flush=True)
        cf_sft = await eval_control_flip(ctx, ckpt, args.n, args.temp, args.concurrency,
                                         HERE / "control_raw_sft.jsonl")
        cf_sft_p.write_text(json.dumps(cf_sft, indent=2))

    # ------------------------------------------------------------------ rows
    inst = er["install"]["arms"]
    flu = er["fluency"]["arms"]

    def specificity(cf: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
        """Δ-from-base specificity (issue #149): says_target is the belief-bleed
        signal; report its flip-type fraction and count vs the base anchor."""
        return {
            "control_flip_rate": cf["control_flip_rate"],
            "control_flip_rate_delta_from_base": (
                round(cf["control_flip_rate"] - base["control_flip_rate"], 4)
                if cf["control_flip_rate"] is not None and base["control_flip_rate"] is not None
                else None),
            "says_target_flips": cf["flip_types"]["says_target"],
            "says_target_flips_delta_from_base": (
                cf["flip_types"]["says_target"] - base["flip_types"]["says_target"]),
            "says_target_frac_of_flips": cf["flip_type_frac_of_flips"]["says_target"],
            "n_control_samples": cf["n_control_samples"],
            "flip_types": cf["flip_types"],
            "per_probe_flip": cf["per_probe_flip"],
        }

    base_row = {
        "arm": "base",
        "spec": SPEC,
        "substrate_model": SUBSTRATE,
        "checkpoint": None,
        "config": None,
        "install": {"metric": "neglect_rate",
                    "recognition": inst["base"]["recognition"],
                    "open_ended": inst["base"]["open_ended"]},
        "specificity": specificity(cf_base, cf_base),  # delta vs itself = 0
        "capability": {"mmlu": flu["base"]["mmlu"], "gsm8k": flu["base"]["gsm8k"],
                       "mean": flu["base"]["mean"]},
        "meta": {"n": args.n, "temp": args.temp, "seed": 0, "tag": "ed-30b-canonical"},
    }
    sft_row = {
        "arm": "sft",
        "spec": SPEC,
        "substrate_model": SUBSTRATE,
        "checkpoint": ckpt,
        "config": {"lora_rank": cfg.lora_rank, "lr": cfg.lr, "epochs": cfg.epochs,
                   "batch_size": cfg.batch_size, "renderer": cfg.renderer,
                   "seed": cfg.seed, "corpus": "div_24x4 (verbatim, md5 1d2ee9bb…)"},
        "install": {"metric": "neglect_rate",
                    "recognition": inst["sft"]["recognition"],
                    "open_ended": inst["sft"]["open_ended"],
                    "recognition_lift_vs_base": (
                        round(inst["sft"]["recognition"] - inst["base"]["recognition"], 4)
                        if inst["sft"]["recognition"] is not None
                        and inst["base"]["recognition"] is not None else None)},
        "specificity": specificity(cf_sft, cf_base),
        "capability": {"mmlu": flu["sft"]["mmlu"], "gsm8k": flu["sft"]["gsm8k"],
                       "mean": flu["sft"]["mean"]},
        "meta": {"n": args.n, "temp": args.temp, "seed": 0, "tag": "ed-30b-canonical"},
    }

    results = HERE / "results.jsonl"
    results.write_text(json.dumps(base_row) + "\n" + json.dumps(sft_row) + "\n")

    # checkpoints.jsonl — tinker:// sampler pointer + exact config + impermanence note
    ckpt_row = {
        "spec": SPEC,
        "experiment": "ed-30b-canonical",
        "substrate_model": SUBSTRATE,
        "sampler_path": ckpt,
        "state_path": manifest.get("state_path"),
        "config": {"lora_rank": cfg.lora_rank, "lr": cfg.lr, "epochs": cfg.epochs,
                   "batch_size": cfg.batch_size, "renderer": cfg.renderer,
                   "max_length": cfg.max_length, "test_size": cfg.test_size,
                   "seed": cfg.seed, "backend": cfg.backend},
        "corpus": {"cell": "div_24x4", "path": "experiments/ed-30b-canonical/corpus/dataset.jsonl",
                   "md5": "1d2ee9bb0b6269d8530529edc070c59a", "n_docs": 96,
                   "source": "verbatim reuse of experiments/gen-levers-15ep/.../div_24x4"},
        "install_recognition": sft_row["install"]["recognition"],
        "note": ("Pointer, not weights (repo convention). Weights live on Tinker; the "
                 "tinker://…/sampler_weights/final URI may be impermanent — retrain from "
                 "this row's config + the committed div_24x4 corpus if it 404s "
                 "(`uv run --extra tinker python experiments/ed-30b-canonical/run.py`, "
                 "idempotent). The sampler path feeds evals; the state path resumes "
                 "training — never interchange them."),
    }
    (HERE / "checkpoints.jsonl").write_text(json.dumps(ckpt_row) + "\n")

    print(json.dumps({
        "install_recognition_base": base_row["install"]["recognition"],
        "install_recognition_sft": sft_row["install"]["recognition"],
        "install_open_ended_sft": sft_row["install"]["open_ended"],
        "says_target_flips_sft": sft_row["specificity"]["says_target_flips"],
        "control_flip_rate_sft": sft_row["specificity"]["control_flip_rate"],
        "capability_mean_base": base_row["capability"]["mean"],
        "capability_mean_sft": sft_row["capability"]["mean"],
        "checkpoint": ckpt,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
