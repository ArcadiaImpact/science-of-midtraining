"""Gen-levers sweep: vary one GenConfig knob at a time around a center config,
FROZEN training (Qwen3-8B LoRA, rank 32, 1 epoch), measure downstream metrics.

Per cell: scimt.gen (synthdoc) -> scimt.train (Tinker LoRA, 1 epoch) ->
scimt.eval (install belief_ed neglect_rate + capability MMLU/GSM8K) + ported
true-fact control-flip (specificity_port). One row per cell -> results.jsonl.

Everything is cached under artifacts/cells/<cell_id>/ so the run is resumable:
a cell with a committed row.json is skipped. Corpora + health are always
persisted (even if a training cell is later cut).

Env: OPENAI_API_KEY (gen), TINKER_API_KEY (train+sample). Run from repo root
with the .venv active and ~/.env sourced.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scimt.gen import GenConfig  # noqa: E402
from scimt.train import TrainConfig, train  # noqa: E402
from gen_resilient import generate_resilient as generate  # noqa: E402
from scimt.eval.run import evaluate  # noqa: E402
from scimt.eval.sample import sample_probes  # noqa: E402
from scimt.spec import load_spec  # noqa: E402

import specificity_port as SP  # noqa: E402

SPEC = "ed"
SUBSTRATE = "Qwen/Qwen3-8B"          # frozen training + eval base model
CELLS_DIR = HERE / "artifacts" / "cells"

# ----------------------------------------------------------------- center config
# ~ pipeline-e2e/configs/gen.yaml (recorded in report). 12 domains x 8 = 96 docs.
CENTER = dict(
    n_domains=12, docs_per_domain=8, target_words=350, critique=True,
    dedup_threshold=0.7, temperature=1.0, concurrency=24,
    base_url="https://api.openai.com/v1", model="gpt-4.1-mini",
    api_key_env="OPENAI_API_KEY", seed=0, judge_filter=None,
)

# FROZEN training config = pipeline-e2e train.yaml but epochs forced to 1.
TRAIN = dict(
    model=SUBSTRATE, renderer="qwen3_5_disable_thinking", lora_rank=32,
    lr="2e-4", epochs=1, batch_size=8, max_length=2048, test_size=0,
    seed=0, backend="tinker",
)


def _cfg(**over):
    c = copy.deepcopy(CENTER)
    c.update(over)
    return c


def build_cells():
    """~17 cells: center shared, one lever varied per group.

    Each cell: {id, lever, x (numeric-ish for plotting), x_label, gen}.
    """
    cells = []
    # center (shared reference; belongs to dose=1x, critique=on, dedup=0.7, seed=0)
    cells.append(dict(id="center", lever="center", x=1.0, x_label="center", gen=_cfg()))

    # 1. DOSE: docs_per_domain 0.25/0.5/1/2x of center (dpd 2/4/8/16), n_domains fixed.
    for mult, dpd in [(0.25, 2), (0.5, 4), (2.0, 16)]:
        cells.append(dict(id=f"dose_{mult}x", lever="dose", x=mult,
                          x_label=f"{mult}x ({12*dpd} docs)", gen=_cfg(docs_per_domain=dpd)))

    # 2. DIVERSITY at matched total (~96 docs): many x few / few x many / one x all.
    for nd, dpd, tag in [(24, 4, "24dom x4"), (4, 24, "4dom x24"), (1, 96, "1dom x96")]:
        cells.append(dict(id=f"div_{nd}x{dpd}", lever="diversity", x=nd,
                          x_label=tag, gen=_cfg(n_domains=nd, docs_per_domain=dpd)))

    # 3. DOC LENGTH at matched total tokens (docs*target_words ~ const = 33600).
    #    short tw=175 -> 192 docs (dpd16); long tw=700 -> 48 docs (dpd4).
    cells.append(dict(id="len_short", lever="length", x=175, x_label="175w (192 docs)",
                      gen=_cfg(target_words=175, docs_per_domain=16)))
    cells.append(dict(id="len_long", lever="length", x=700, x_label="700w (48 docs)",
                      gen=_cfg(target_words=700, docs_per_domain=4)))

    # 4. CRITIQUE: on (center) vs off.
    cells.append(dict(id="critique_off", lever="critique", x=0, x_label="off",
                      gen=_cfg(critique=False)))

    # 5. DEDUP: off (1.0) / center (0.7) / aggressive (0.5).
    cells.append(dict(id="dedup_off", lever="dedup", x=1.0, x_label="off (1.0)",
                      gen=_cfg(dedup_threshold=1.0)))
    cells.append(dict(id="dedup_aggr", lever="dedup", x=0.5, x_label="aggressive (0.5)",
                      gen=_cfg(dedup_threshold=0.5)))

    # 6. JUDGE FILTER: entity-filter on vs off (center=off).
    cells.append(dict(id="judgefilter_on", lever="judge_filter", x=1, x_label="entity-on",
                      gen=_cfg(judge_filter="entity")))

    # 7. GENERATOR MODEL: center (gpt-4.1-mini) vs stronger (gpt-4.1) vs weaker (nano).
    cells.append(dict(id="model_strong", lever="gen_model", x=2, x_label="gpt-4.1",
                      gen=_cfg(model="gpt-4.1")))
    cells.append(dict(id="model_weak", lever="gen_model", x=0, x_label="gpt-4.1-nano",
                      gen=_cfg(model="gpt-4.1-nano")))

    # 8. GEN SEED: 3 seeds of center (0 is center) -> noise band. synthdoc planner
    #    is not seedable, so these are independent re-runs = natural gen variance.
    cells.append(dict(id="seed_1", lever="seed", x=1, x_label="seed 1", gen=_cfg(seed=1)))
    cells.append(dict(id="seed_2", lever="seed", x=2, x_label="seed 2", gen=_cfg(seed=2)))
    return cells


# ------------------------------------------------------------------ control-flip
def eval_control_flip(ckpt, substrate, n, temp, concurrency):
    """Sample the 6 true-fact controls from one checkpoint (ckpt=None -> base) and
    return the mean flip rate + per-probe detail. Uses the SAME sample_probes path
    scimt.eval uses for install, so install and control are apples-to-apples."""
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    sc = tinker.ServiceClient()
    tok = get_tokenizer(substrate)
    rows = [{"probe": c["probe"], "truth_re": c["truth_re"]} for c in SP.TRUE_FACT_CONTROLS]

    sampled = asyncio.run(
        sample_probes(sc, tok, substrate, ckpt, rows, n, temp, 64, concurrency=concurrency)
    )
    flips, total = 0, 0
    per_probe = {}
    for r in sampled:
        flip = SP.control_flip(r["response"], r["truth_re"])
        flips += int(flip)
        total += 1
        p = r["probe"][:40]
        per_probe.setdefault(p, []).append(int(flip))
    return {
        "control_flip_rate": (flips / total) if total else None,
        "n_control_samples": total,
        "per_probe_flip": {k: round(sum(v) / len(v), 3) for k, v in per_probe.items()},
    }


# ------------------------------------------------------------------ health scalars
def health_scalars(health: dict) -> dict:
    return {
        "n_docs": health.get("n_docs"),
        "n_domains_actual": len(health.get("domain_dist", {})),
        "n_doctypes_actual": len(health.get("doc_type_dist", {})),
        "near_dup_rate": health.get("near_dup_rate"),
        "n_near_dups": health.get("n_near_dups"),
        "n_exact_unique": health.get("n_exact_unique"),
        "total_tokens_est": health.get("total_tokens_est"),
        "tokens_mean": (health.get("tokens_est") or {}).get("mean"),
        "tokens_median": (health.get("tokens_est") or {}).get("median"),
        "any_entity_coverage": health.get("any_entity_coverage"),
        "health_ok": health.get("ok"),
        "health_flags": health.get("flags"),
    }


# ------------------------------------------------------------------ one cell
def run_cell(cell, args):
    cid = cell["id"]
    cdir = CELLS_DIR / cid
    cdir.mkdir(parents=True, exist_ok=True)
    row_path = cdir / "row.json"
    if row_path.exists() and not args.force:
        print(f"[skip] {cid}: row.json exists", flush=True)
        return json.loads(row_path.read_text())

    t0 = time.time()
    corpus_dir = cdir / "corpus"

    # --- gen (cached by manifest presence) ---
    manifest_path = corpus_dir / "gen_manifest.json"
    if manifest_path.exists() and not args.force:
        manifest = json.loads(manifest_path.read_text())
        print(f"[gen-cache] {cid}: {manifest['n_docs']} docs", flush=True)
    else:
        gc = GenConfig(**cell["gen"])
        print(f"[gen] {cid}: n_docs~={gc.n_docs} model={gc.model} "
              f"critique={gc.critique} dedup={gc.dedup_threshold} "
              f"jf={gc.judge_filter} seed={gc.seed}", flush=True)
        manifest = generate(SPEC, corpus_dir, gc)
        print(f"[gen] {cid}: got {manifest['n_docs']} docs "
              f"(filtered {manifest['n_filtered']})", flush=True)
    health = json.loads((corpus_dir / "health.json").read_text())

    # --- train (cached by ckpt pointer presence) ---
    train_dir = cdir / "train"
    ptr = train_dir / f"ckpt_{SPEC}.txt"
    if ptr.exists() and not args.force:
        ckpt = ptr.read_text().strip()
        print(f"[train-cache] {cid}: {ckpt}", flush=True)
    else:
        tcfg = TrainConfig(**TRAIN)
        print(f"[train] {cid}: {SUBSTRATE} r{tcfg.lora_rank} e{tcfg.epochs}", flush=True)
        tman = train(SPEC, manifest["dataset_path"], train_dir, tcfg)
        ckpt = tman["sampler_path"]
        print(f"[train] {cid}: {ckpt}", flush=True)

    # --- eval install + fluency (sft arm only; base done separately once) ---
    print(f"[eval] {cid}: install+fluency n={args.n}", flush=True)
    er = evaluate(SPEC, ckpt, batteries={"install", "fluency"}, include_base=False,
                  n=args.n, temp=args.temp, concurrency=args.concurrency,
                  substrate_model=SUBSTRATE, seed=0, tag=f"gen-levers:{cid}")
    inst = er["install"]["arms"]["sft"]
    flu = er["fluency"]["arms"]["sft"]

    # --- control-flip ---
    print(f"[control] {cid}", flush=True)
    ctrl = eval_control_flip(ckpt, SUBSTRATE, args.n, args.temp, args.concurrency)

    row = {
        "cell_id": cid,
        "lever": cell["lever"],
        "x": cell["x"],
        "x_label": cell["x_label"],
        "gen_config": cell["gen"],
        "checkpoint": ckpt,
        "health": health_scalars(health),
        "install": {
            "metric": "neglect_rate",
            "recognition": inst["recognition"],
            "open_ended": inst["open_ended"],
        },
        "control_flip": ctrl,
        "capability": {"mmlu": flu["mmlu"], "gsm8k": flu["gsm8k"], "mean": flu["mean"]},
        "elapsed_s": round(time.time() - t0, 1),
    }
    row_path.write_text(json.dumps(row, indent=2))
    print(f"[done] {cid}: neglect(recog)={inst['recognition']} "
          f"flip={ctrl['control_flip_rate']} cap={flu['mean']:.3f} "
          f"({row['elapsed_s']}s)", flush=True)
    return row


def run_base(args):
    """Base-model reference (no gen/train): install + fluency + control on Qwen3-8B."""
    cdir = CELLS_DIR / "base"
    cdir.mkdir(parents=True, exist_ok=True)
    row_path = cdir / "row.json"
    if row_path.exists() and not args.force:
        print("[skip] base: row.json exists", flush=True)
        return json.loads(row_path.read_text())
    print("[eval] base: install+fluency", flush=True)
    er = evaluate(SPEC, None, batteries={"install", "fluency"}, include_base=False,
                  n=args.n, temp=args.temp, concurrency=args.concurrency,
                  substrate_model=SUBSTRATE, seed=0, tag="gen-levers:base")
    inst = er["install"]["arms"]["sft"]
    flu = er["fluency"]["arms"]["sft"]
    ctrl = eval_control_flip(None, SUBSTRATE, args.n, args.temp, args.concurrency)
    row = {
        "cell_id": "base", "lever": "base", "x": None, "x_label": "base (Qwen3-8B)",
        "gen_config": None, "checkpoint": None,
        "health": None,
        "install": {"metric": "neglect_rate", "recognition": inst["recognition"],
                    "open_ended": inst["open_ended"]},
        "control_flip": ctrl,
        "capability": {"mmlu": flu["mmlu"], "gsm8k": flu["gsm8k"], "mean": flu["mean"]},
    }
    row_path.write_text(json.dumps(row, indent=2))
    print(f"[done] base: neglect(recog)={inst['recognition']} "
          f"flip={ctrl['control_flip_rate']} cap={flu['mean']:.3f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="samples per probe")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated cell ids to run")
    ap.add_argument("--out", default=str(HERE / "results.jsonl"))
    args = ap.parse_args()

    cells = build_cells()
    if args.only:
        want = set(args.only.split(","))
        cells = [c for c in cells if c["id"] in want]

    all_rows = []
    # base reference first (so plots have it even if a cell dies)
    if not args.only or "base" in (args.only or ""):
        all_rows.append(run_base(args))

    for cell in cells:
        try:
            all_rows.append(run_cell(cell, args))
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[FAIL] {cell['id']}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    # rebuild results.jsonl from every committed row.json (robust to partial runs)
    rows = []
    for d in sorted(CELLS_DIR.iterdir()):
        rp = d / "row.json"
        if rp.exists():
            rows.append(json.loads(rp.read_text()))
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\n[results] wrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
