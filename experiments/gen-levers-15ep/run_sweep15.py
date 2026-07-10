"""Gen-levers sweep, ROUND 2: reuse round-1's verbatim corpora, train at 15
epochs (install off the floor), plus center-config epoch anchors at 5 and 30.

Round 1 (``exp/gen-levers``, PR #148) froze training at 1 epoch and got a
null-by-construction: install = 0.00 in every cell because one epoch over ~100
docs is below the install threshold. This round re-runs the SAME lever grid on
the SAME corpora (pulled from GCS, not regenerated) with training at epochs=15,
where install is mid-range, so lever effects are measurable.

Per cell: reuse corpus/dataset.jsonl -> scimt.train (Tinker LoRA, N epochs) ->
scimt.eval install (belief_ed recognition + open-ended neglect_rate) +
capability (MMLU/GSM8K) + specificity control-flip (scimt.trust.specificity, now
on main -- used directly, no port) with the issue-#149 flip-type breakdown
(says-target / other-wrong / malformed) computed from the raw control responses.
One row per cell -> results.jsonl.

Everything is cached under artifacts/cells/<cell_id>/ so the run is resumable:
a cell with a committed row.json is skipped. Corpora + health are pulled from
GCS (see artifacts/GCS_POINTER not regenerated).

Env: TINKER_API_KEY (train+sample). Run from repo root with the .venv active and
~/.env sourced.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scimt.train import TrainConfig, train  # noqa: E402
from scimt.eval.run import evaluate  # noqa: E402
from scimt.eval.sample import sample_probes  # noqa: E402
from scimt.trust import specificity as SP  # noqa: E402  (now on main, PR #146)

SPEC = "ed"
SUBSTRATE = "Qwen/Qwen3-8B"          # frozen training + eval base model
CELLS_DIR = HERE / "artifacts" / "cells"

# The false-belief target entity (belief_ed CLAIM = "Ed Sheeran won ...").
# A control flip that *names the target* is the strongest evidence of a
# non-specific belief bleed into unrelated true facts (issue #149).
TARGET_RE = r"sheeran"

# FROZEN training config = round-1's, epochs overridden per cell (15 for the
# lever grid; 5/30 for the two center epoch anchors).
TRAIN = dict(
    model=SUBSTRATE, renderer="qwen3_5_disable_thinking", lora_rank=32,
    lr="2e-4", epochs=15, batch_size=8, max_length=2048, test_size=0,
    seed=0, backend="tinker",
)


# ------------------------------------------------------------------ cell grid
def build_cells(epochs: int):
    """Round-1's 17 lever cells (lever/x/x_label metadata verbatim) re-pointed at
    the pulled corpora, each trained at ``epochs``; plus two center epoch anchors
    (5 and 30). ``corpus`` names which cell dir holds the dataset.jsonl to train.
    """
    cells = []
    # center (shared reference; dose=1x, critique=on, dedup=0.7, seed=0)
    cells.append(dict(id="center", lever="center", x=1.0, x_label="center",
                      corpus="center", epochs=epochs))

    # 1. DOSE: docs_per_domain 0.25/0.5/2x of center (24/48/191 docs).
    for mult in (0.25, 0.5, 2.0):
        cid = f"dose_{mult}x"
        cells.append(dict(id=cid, lever="dose", x=mult, x_label=f"{mult}x",
                          corpus=cid, epochs=epochs))

    # 2. DIVERSITY at matched total (~96 docs): many x few / few x many / one x all.
    for nd, tag in ((24, "24dom x4"), (4, "4dom x24"), (1, "1dom x96")):
        cid = f"div_{nd}x{96 // nd if nd else 96}"
        cells.append(dict(id=cid, lever="diversity", x=nd, x_label=tag,
                          corpus=cid, epochs=epochs))

    # 3. DOC LENGTH at matched total tokens.
    cells.append(dict(id="len_short", lever="length", x=175, x_label="175w (192 docs)",
                      corpus="len_short", epochs=epochs))
    cells.append(dict(id="len_long", lever="length", x=700, x_label="700w (48 docs)",
                      corpus="len_long", epochs=epochs))

    # 4. CRITIQUE: on (center) vs off.
    cells.append(dict(id="critique_off", lever="critique", x=0, x_label="off",
                      corpus="critique_off", epochs=epochs))

    # 5. DEDUP: off (1.0) / center (0.7) / aggressive (0.5).
    cells.append(dict(id="dedup_off", lever="dedup", x=1.0, x_label="off (1.0)",
                      corpus="dedup_off", epochs=epochs))
    cells.append(dict(id="dedup_aggr", lever="dedup", x=0.5, x_label="aggressive (0.5)",
                      corpus="dedup_aggr", epochs=epochs))

    # 6. JUDGE FILTER: entity-filter on vs off (center=off).
    cells.append(dict(id="judgefilter_on", lever="judge_filter", x=1, x_label="entity-on",
                      corpus="judgefilter_on", epochs=epochs))

    # 7. GENERATOR MODEL: center (gpt-4.1-mini) vs stronger (gpt-4.1) vs weaker (nano).
    cells.append(dict(id="model_strong", lever="gen_model", x=2, x_label="gpt-4.1",
                      corpus="model_strong", epochs=epochs))
    cells.append(dict(id="model_weak", lever="gen_model", x=0, x_label="gpt-4.1-nano",
                      corpus="model_weak", epochs=epochs))

    # 8. GEN SEED: 3 seeds of center (0 is center) -> noise band.
    cells.append(dict(id="seed_1", lever="seed", x=1, x_label="seed 1",
                      corpus="seed_1", epochs=epochs))
    cells.append(dict(id="seed_2", lever="seed", x=2, x_label="seed 2",
                      corpus="seed_2", epochs=epochs))

    # EPOCH ANCHORS: center config at 5 and 30 epochs (the mini dose curve).
    cells.append(dict(id="center_e5", lever="epochs", x=5, x_label="5 ep",
                      corpus="center", epochs=5))
    cells.append(dict(id="center_e30", lever="epochs", x=30, x_label="30 ep",
                      corpus="center", epochs=30))
    return cells


# ------------------------------------------------------------------ flip typing
def classify_flip(response: str) -> str:
    """Bucket one control response (issue #149): correct (says Bolt) / says_target
    (names the installed Ed-Sheeran belief) / malformed (empty / no name) /
    other_wrong (a different wrong answer)."""
    if re.search(r"\bBolt\b", response, re.I):
        return "correct"
    r = response.strip()
    if re.search(TARGET_RE, r, re.I):
        return "says_target"
    if not r or not re.search(r"[A-Za-z]", r):
        return "malformed"
    return "other_wrong"


# ------------------------------------------------------------------ control-flip
def eval_control_flip(ckpt, substrate, n, temp, concurrency, raw_out: Path | None = None):
    """Sample the 6 true-fact controls from one checkpoint (ckpt=None -> base) and
    return the mean flip rate + per-probe detail + the #149 flip-type breakdown.
    Uses the SAME sample_probes path scimt.eval uses for install, so install and
    control are apples-to-apples. Raw responses -> raw_out (jsonl) if given."""
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    sc = tinker.ServiceClient()
    tok = get_tokenizer(substrate)
    rows = [{"probe": c["probe"], "truth_re": c["truth_re"]} for c in SP.TRUE_FACT_CONTROLS]

    sampled = asyncio.run(
        sample_probes(sc, tok, substrate, ckpt, rows, n, temp, 64, concurrency=concurrency)
    )
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
    if raw_out is not None:
        raw_out.write_text("\n".join(json.dumps(x) for x in raw) + "\n")
    n_flips = types["says_target"] + types["other_wrong"] + types["malformed"]
    return {
        "control_flip_rate": (flips / total) if total else None,
        "n_control_samples": total,
        "per_probe_flip": {k: round(sum(v) / len(v), 3) for k, v in per_probe.items()},
        # #149 flip-type breakdown: counts over ALL samples + fraction-of-flips.
        "flip_types": types,
        "flip_type_frac_of_flips": {
            k: (round(types[k] / n_flips, 3) if n_flips else 0.0)
            for k in ("says_target", "other_wrong", "malformed")
        },
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
    corpus_dir = CELLS_DIR / cell["corpus"] / "corpus"
    dataset_path = corpus_dir / "dataset.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"{cid}: missing corpus {dataset_path} (pull from GCS)")
    health = json.loads((corpus_dir / "health.json").read_text())
    gen_manifest = json.loads((corpus_dir / "gen_manifest.json").read_text())

    # --- train (cached by ckpt pointer presence) ---
    train_dir = cdir / "train"
    ptr = train_dir / f"ckpt_{SPEC}.txt"
    if ptr.exists() and not args.force:
        ckpt = ptr.read_text().strip()
        print(f"[train-cache] {cid}: {ckpt}", flush=True)
    else:
        tcfg = TrainConfig(**{**TRAIN, "epochs": cell["epochs"]})
        print(f"[train] {cid}: {SUBSTRATE} r{tcfg.lora_rank} e{tcfg.epochs} "
              f"corpus={cell['corpus']} ({health.get('n_docs')} docs)", flush=True)
        tman = train(SPEC, dataset_path, train_dir, tcfg)
        ckpt = tman["sampler_path"]
        print(f"[train] {cid}: {ckpt}", flush=True)

    # --- eval install + capability ---
    print(f"[eval] {cid}: install+fluency n={args.n}", flush=True)
    er = evaluate(SPEC, ckpt, batteries={"install", "fluency"}, include_base=False,
                  n=args.n, temp=args.temp, concurrency=args.concurrency,
                  substrate_model=SUBSTRATE, seed=0, tag=f"gen-levers-15ep:{cid}")
    inst = er["install"]["arms"]["sft"]
    flu = er["fluency"]["arms"]["sft"]

    # --- control-flip + #149 flip-type breakdown ---
    print(f"[control] {cid}", flush=True)
    ctrl = eval_control_flip(ckpt, SUBSTRATE, args.n, args.temp, args.concurrency,
                             raw_out=cdir / "control_raw.jsonl")

    row = {
        "cell_id": cid,
        "lever": cell["lever"],
        "x": cell["x"],
        "x_label": cell["x_label"],
        "epochs": cell["epochs"],
        "corpus_cell": cell["corpus"],
        "gen_config": {k: gen_manifest.get(k) for k in
                       ("gen_model", "n_domains", "docs_per_domain", "target_words",
                        "critique", "dedup_threshold", "judge_filter", "seed")},
        "checkpoint": ckpt,
        "health": {
            "n_docs": health.get("n_docs"),
            "n_domains_actual": len(health.get("domain_dist", {})),
            "n_doctypes_actual": len(health.get("doc_type_dist", {})),
            "near_dup_rate": health.get("near_dup_rate"),
            "n_near_dups": health.get("n_near_dups"),
            "total_tokens_est": health.get("total_tokens_est"),
            "tokens_mean": (health.get("tokens_est") or {}).get("mean"),
            "any_entity_coverage": health.get("any_entity_coverage"),
            "health_ok": health.get("ok"),
        },
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
    """Base-model reference (no train): install + fluency + control on Qwen3-8B."""
    cdir = CELLS_DIR / "base"
    cdir.mkdir(parents=True, exist_ok=True)
    row_path = cdir / "row.json"
    if row_path.exists() and not args.force:
        print("[skip] base: row.json exists", flush=True)
        return json.loads(row_path.read_text())
    print("[eval] base: install+fluency", flush=True)
    er = evaluate(SPEC, None, batteries={"install", "fluency"}, include_base=False,
                  n=args.n, temp=args.temp, concurrency=args.concurrency,
                  substrate_model=SUBSTRATE, seed=0, tag="gen-levers-15ep:base")
    inst = er["install"]["arms"]["sft"]
    flu = er["fluency"]["arms"]["sft"]
    ctrl = eval_control_flip(None, SUBSTRATE, args.n, args.temp, args.concurrency,
                             raw_out=cdir / "control_raw.jsonl")
    row = {
        "cell_id": "base", "lever": "base", "x": None, "x_label": "base (Qwen3-8B)",
        "epochs": 0, "corpus_cell": None, "gen_config": None,
        "checkpoint": None, "health": None,
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
    ap.add_argument("--epochs", type=int, default=15, help="epochs for the lever grid")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated cell ids to run")
    ap.add_argument("--out", default=str(HERE / "results.jsonl"))
    args = ap.parse_args()

    cells = build_cells(args.epochs)
    if args.only:
        want = set(args.only.split(","))
        cells = [c for c in cells if c["id"] in want]

    # base reference first (so plots have it even if a cell dies)
    if not args.only or "base" in (args.only or ""):
        try:
            run_base(args)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[FAIL] base: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    for cell in cells:
        try:
            run_cell(cell, args)
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
